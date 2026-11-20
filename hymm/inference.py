import os
import time
import random
import functools
import numpy as np
import math
import re
from PIL import Image
from typing import List, Optional, Tuple, Union

from pathlib import Path
from loguru import logger

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hymm.constants import (
    PROMPT_TEMPLATE,
    NEGATIVE_PROMPT,
)
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.models.text_encoder import TextEncoder
from hymm.diffusion.schedulers import FlowMatchDiscreteScheduler
from hymm.diffusion.pipelines import HunyuanVideoPipeline
from hymm.utils.file_utils import get_next_available_save_id
from hymm.dataset.data_kits.csv_dataset import CSVDataset
from hymm.dataset.caps.caption_process_v4 import CaptionAug
from hymm.utils.file_utils import safe_file
from hymm.utils.eval_utils import batch_data_repr
from hymm.utils.load import load_transformer, load_text_encoder, load_glyph_byT5_v2
from hymm.models.vae import load_vae

from safetensors.torch import load_file as safetensors_load_file

from hymm.utils.parallel_states import (
    initialize_sequence_parallel_state,
    nccl_info,
)
from hymm.utils.load import convert_hunyuan_dict_for_tensor_parallel
from hymm.models.vision_encoder import VisionEncoder
from hymm.utils.fsdp_util import (
    get_dit_fsdp_kwargs_v2,
    apply_fsdp2
)
from torch.distributed.tensor import DTensor

def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


def _get_event_desc_pattern(prefix_zh=None, prefix_en=None):
    if prefix_zh is None:
        prefix_zh = ""
    if prefix_en is None:
        prefix_en = ""
    prefixes = []
    for prefix in (prefix_zh, prefix_en):
        if prefix is None:
            continue
        prefix = str(prefix)
        if prefix != "":
            prefixes.append(re.escape(prefix))
    if prefixes:
        prefix_group = "|".join(prefixes)
        prefix_part = rf"(?:{prefix_group})?\s*"
    else:
        prefix_part = r""
    # Support both "Xs-Ys" / "X秒-Y秒" and unified "[X-Y]" bracket format
    pattern_seconds = (
        rf"{prefix_part}([0-9]+(?:\.[0-9]+)?)\s*"
        rf"(?:秒|s(?:ec(?:ond)?s?)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*"
        rf"(?:秒|s(?:ec(?:ond)?s?)?)"
    )
    pattern_bracket = (
        rf"{prefix_part}\[([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\]"
    )
    combined = rf"(?:{pattern_seconds})|(?:{pattern_bracket})"
    return re.compile(combined, re.IGNORECASE)


def _format_time_value(value):
    if value is None:
        return None
    try:
        time_value = float(value)
    except (TypeError, ValueError):
        return None
    if abs(time_value - round(time_value)) < 1e-6:
        return str(int(round(time_value)))
    return f"{time_value:.6f}".rstrip("0").rstrip(".")


def _format_time_range(start, end, mode, fps):
    """Convert parsed time values to the representation expected by segment_layout.

    The input start/end are always parsed from prompt text in seconds
    (e.g. "2.5" from "2.5秒-5秒"). The mode controls what coordinate
    system the output should use:
      - "second": keep as seconds (float)
      - "latent": convert seconds → latent frame index via
                  (sec * fps - 1) / 4 + 1
      - other:    convert seconds → pixel frame index (sec * fps)
    """
    if start is None or end is None:
        return None
    mode = str(mode or "frame").strip().lower()
    if mode in ("second", "seconds"):
        start_value = float(start)
        end_value = float(end)
    elif mode == "latent":
        # Prompt text is always in seconds; convert to latent frame index.
        # 3D causal VAE: pixel_frame = round(sec * fps),
        # latent_idx = (pixel_frame - 1) // temporal_stride + 1 (0 maps to 0).
        # temporal_stride = 4 for 16164 VAE.
        temporal_stride = 4
        start_sec = float(start)
        end_sec = float(end)
        start_pf = int(round(start_sec * fps))
        end_pf = int(round(end_sec * fps))
        start_value = 0 if start_pf <= 0 else (start_pf - 1) // temporal_stride + 1
        end_value = 0 if end_pf <= 0 else (end_pf - 1) // temporal_stride + 1
    else:
        start_value = int(float(start) * fps)
        end_value = int(float(end) * fps)
    start_text = _format_time_value(start_value)
    end_text = _format_time_value(end_value)
    if start_text is None or end_text is None:
        return None
    return f"{start_text}-{end_text}"


def _parse_event_segments_from_prompt(
    prompt, time_range_mode, fps, prefix_zh=None, prefix_en=None
):
    pattern = _get_event_desc_pattern(prefix_zh, prefix_en)
    matches = list(pattern.finditer(prompt))
    if not matches:
        return None

    segment_infos = []
    leading = prompt[: matches[0].start()].strip()
    if leading:
        segment_infos.append({"segment_id": 0, "text": leading, "time_range": None})

    for idx, match in enumerate(matches):
        segment_start = match.start()
        segment_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(prompt)
        segment_text = prompt[segment_start:segment_end].strip()
        groups = match.groups()
        # Combined regex: groups 0,1 = seconds pattern ("Xs-Ys"),
        #                 groups 2,3 = bracket pattern ("[X-Y]").
        # Bracket values are already in the target coordinate system
        # (latent indices if mode=latent, seconds if mode=second).
        # Seconds values need conversion via _format_time_range.
        if groups[0] is not None and groups[1] is not None:
            # Seconds pattern matched — convert via _format_time_range
            start, end = groups[0], groups[1]
            time_range = _format_time_range(start, end, time_range_mode, fps)
        elif groups[2] is not None and groups[3] is not None:
            # Bracket pattern matched — values are already in target coords.
            # Pass through as "start-end" string without conversion.
            start_text = _format_time_value(float(groups[2]))
            end_text = _format_time_value(float(groups[3]))
            time_range = f"{start_text}-{end_text}" if start_text and end_text else None
        else:
            continue
        segment_infos.append(
            {
                "segment_id": idx + 1,
                "text": segment_text,
                "time_range": time_range,
            }
        )
    return segment_infos


def _replace_event_prefix(text, src_prefix_zh, src_prefix_en, dst_prefix_zh, dst_prefix_en):
    raw = str(text or "").strip()
    if not raw:
        return raw
    for src_prefix, dst_prefix in (
        (src_prefix_zh, dst_prefix_zh),
        (src_prefix_en, dst_prefix_en),
    ):
        if src_prefix is None:
            continue
        src_prefix = str(src_prefix)
        if src_prefix == "":
            continue
        if raw.startswith(src_prefix):
            body = raw[len(src_prefix):].lstrip()
            if dst_prefix is None:
                dst_prefix = src_prefix
            dst_prefix = str(dst_prefix)
            if dst_prefix == "":
                return body
            return f"{dst_prefix}{body}"
    return raw


def _build_prompt_from_segments(
    segment_infos,
    src_prefix_zh=None,
    src_prefix_en=None,
    dst_prefix_zh=None,
    dst_prefix_en=None,
):
    parts = []
    for segment in segment_infos:
        segment_text = str(segment.get("text", "")).strip()
        if not segment_text:
            continue
        if src_prefix_zh is not None or src_prefix_en is not None:
            segment_text = _replace_event_prefix(
                segment_text,
                src_prefix_zh,
                src_prefix_en,
                dst_prefix_zh,
                dst_prefix_en,
            )
        parts.append(segment_text)
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else "".join(parts)


def _apply_prompt_fps_prefix(prompt, fps_prefix, data_type="video"):
    if data_type != "video" or fps_prefix is None:
        return prompt
    fps_prefix = str(fps_prefix).strip()
    if not fps_prefix:
        return prompt
    fps_prefix = fps_prefix.rstrip(",").strip() + ", "

    def _apply_one(text):
        text = str(text).strip()
        if not text:
            return text
        if re.match(r"^FPS:\s*\d+(?:\.\d+)?,\s*", text):
            return text
        return f"{fps_prefix}{text}"

    if isinstance(prompt, str):
        return _apply_one(prompt)
    if isinstance(prompt, tuple):
        return tuple(_apply_one(item) for item in prompt)
    if isinstance(prompt, list):
        return [_apply_one(item) for item in prompt]
    return prompt


def _prepare_physrope_prompt_inputs(
    prompt,
    args,
    text_encoder,
    device,
    data_type="video",
):
    if text_encoder is None:
        return prompt, None, None, None, None

    is_scalar_prompt = isinstance(prompt, str)
    if is_scalar_prompt:
        prompt_list = [prompt.strip()]
    elif isinstance(prompt, (list, tuple)):
        prompt_list = [str(item).strip() for item in prompt]
    else:
        return prompt, None, None, None, None

    time_range_mode = getattr(args, "physical_rope_time_range_mode", "second")
    rope_fps = getattr(args, "video_fps", 24)
    try:
        rope_fps = float(rope_fps)
    except (TypeError, ValueError):
        rope_fps = 24.0
    if rope_fps <= 0:
        rope_fps = 24.0

    fps_prefix = getattr(args, "prompt_fps_prefix", None)
    if fps_prefix is not None:
        fps_prefix = str(fps_prefix).strip()
        if fps_prefix:
            fps_prefix = fps_prefix.rstrip(",").strip() + ", "
        else:
            fps_prefix = None

    caption_sample_ratio = getattr(args, "video_caption_sample_ratio", None)
    if caption_sample_ratio is None:
        caption_sample_ratio = {"long_caption": 1.0}

    caption_aug = CaptionAug(
        caption_sample_ratio=caption_sample_ratio,
        time_range_mode=time_range_mode,
        logger=logger,
    )

    rebuilt_prompts = []
    segment_ids_list = []
    segment_time_ranges_list = []
    text_ids_list = []
    text_mask_list = []

    for raw_prompt in prompt_list:
        segment_infos = _parse_event_segments_from_prompt(
            raw_prompt,
            time_range_mode,
            rope_fps,
            "事件描述：",
            "Event description:",
        )
        if not segment_infos:
            return prompt, None, None, None, None
        if fps_prefix:
            for segment_info in segment_infos:
                seg_text = str(segment_info.get("text", ""))
                if seg_text:
                    segment_info["text"] = f"{fps_prefix}{seg_text}"
                    break
        rebuilt_prompts.append(_build_prompt_from_segments(segment_infos))
        text_ids, text_mask, segment_ids, segment_time_ranges = caption_aug.tokenize_segments(
            text_encoder,
            args.text_len,
            segment_infos,
            data_type=data_type,
        )
        text_ids_list.append(text_ids)
        text_mask_list.append(text_mask)
        segment_ids_list.append(segment_ids)
        segment_time_ranges_list.append(segment_time_ranges)

    text_ids_batch = torch.stack(text_ids_list, dim=0).to(device)
    text_mask_batch = torch.stack(text_mask_list, dim=0).to(device)
    text_outputs = text_encoder.encode(
        {"input_ids": text_ids_batch, "attention_mask": text_mask_batch},
        data_type=data_type,
        device=device,
    )
    prompt_embeds = text_outputs.hidden_state
    attention_mask = text_outputs.attention_mask
    text_segment_ids = torch.tensor(segment_ids_list, dtype=torch.long, device=device)
    prompt_out = rebuilt_prompts[0] if is_scalar_prompt else rebuilt_prompts
    return (
        prompt_out,
        prompt_embeds,
        attention_mask,
        text_segment_ids,
        segment_time_ranges_list,
    )


def _prepare_physrope_prompt_metadata(
    prompt,
    args,
    target_video_length,
    data_type="video",
):
    if data_type != "video" or not getattr(args, "use_physical_rope", False):
        return None

    is_scalar_prompt = isinstance(prompt, str)
    if is_scalar_prompt:
        prompt_list = [prompt.strip()]
    elif isinstance(prompt, (list, tuple)):
        prompt_list = [str(item).strip() for item in prompt]
    else:
        return None

    time_range_mode = getattr(args, "physical_rope_time_range_mode", "second")
    rope_fps = getattr(args, "video_fps", 24)
    try:
        rope_fps = float(rope_fps)
    except (TypeError, ValueError):
        rope_fps = 24.0
    if rope_fps <= 0:
        rope_fps = 24.0

    use_second_time = str(time_range_mode).strip().lower() in ("second", "seconds")
    use_latent_time = str(time_range_mode).strip().lower() in ("latent",)
    if use_second_time:
        total_range_end = max(int(target_video_length) - 1, 0) / rope_fps
    elif use_latent_time:
        # 3D causal VAE: pixel_frames → latent_frames = (pixel_frames - 1) // 4 + 1
        total_range_end = float((max(int(target_video_length) - 1, 0)) // 4 + 1)
    else:
        total_range_end = float(max(int(target_video_length) - 1, 0))

    all_rows = []
    for raw_prompt in prompt_list:
        segment_infos = _parse_event_segments_from_prompt(
            raw_prompt,
            time_range_mode,
            rope_fps,
            "事件描述：",
            "Event description:",
        )
        if not segment_infos:
            return None
        rows = []
        for seg in segment_infos:
            try:
                seg_id = int(seg.get("segment_id", -1))
            except (TypeError, ValueError):
                continue
            time_range = seg.get("time_range")
            if seg_id == 0:
                time_range = (0.0, float(total_range_end))
            if time_range is None:
                continue
            rows.append({"segment_id": seg_id, "time_range": time_range})
        if not rows:
            return None
        all_rows.append(rows)

    return all_rows[0] if is_scalar_prompt else all_rows

class Inference(object):
    def __init__(
        self,
        args,
        vae,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=None,
        logger=None,
        parallel_args=None,
    ):
        self.vae = vae

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload

        self.args = args
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = logger
        self.parallel_args = parallel_args

    @staticmethod
    def save_batch_image(images, save_names):
        if len(images) != len(save_names):
            raise ValueError(
                f"Length of images ({len(images)}) should be equal to length of save_names ({len(save_names)})."
            )

        if isinstance(images, torch.Tensor):
            # Tensor -> numpy.ndarray
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        if isinstance(images, np.ndarray):
            # numpy.ndarray -> PIL.Image
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            if images.shape[-1] == 1:
                # special case for grayscale (single channel) images
                images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
            else:
                images = [Image.fromarray(image) for image in images]

        for i, (image, save_name) in enumerate(zip(images, save_names)):
            image.save(safe_file(save_name))

    def batch_sample(self, csv, save_template, image_size, batch_size=1,
                     seed_type="auto", seed=123456, skip_exist=False):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_exp_dir_and_ckpt_id(self):
        if self.args.ckpt is None:
            raise ValueError("The checkpoint path is not provided.")

        ckpt = Path(self.args.ckpt)
        if ckpt.parents[1].name == "checkpoints":
            # It should be a standard checkpoint path. We use the parent directory as the default save directory.
            exp_dir = ckpt.parents[2]
        else:
            raise ValueError(f"We cannot infer the experiment directory from the checkpoint path: {ckpt}. "
                             f"It seems that the checkpoint path is not standard. Please explicitly provide the "
                             f"save path by --save-path.")
        return exp_dir, ckpt.parent.name

    def get_sample_save_dir(self, testset, image_size, logger=None):
        args = self.args
        if testset is None:
            testset = '{}'

        if args.save_path is None:
            exp_dir, ckpt_id = self.get_exp_dir_and_ckpt_id()
            fixed_suffix = self.get_sample_dir_suffix(testset, image_size)
            save_base = exp_dir / "samples" / f"{ckpt_id}_{fixed_suffix}{args.save_path_suffix}"
        else:
            save_base = Path(args.save_path)
            save_base = save_base.parent / f"{save_base.name}{args.save_path_suffix}"

        if logger is not None:
            logger.info(f"Set --save-path to {save_base} based on the checkpoint path.")
        return save_base
    
    def get_default_sample_save_paths(self, num_images, prompt):
        args = self.args

        save_path = Path("vis/images/output" if args.save_path is None else args.save_path)
        if save_path.suffix == '':
            # We treat it as a directory
            start_id = get_next_available_save_id(save_path)
            save_paths = []
            for i in range(num_images):
                save_stem = f"{i + start_id}_{prompt}"
                save_paths.append(save_path / (save_stem[:240] + ".png"))
        else:
            if num_images == 1:
                save_paths = [save_path]
            else:
                save_paths = [save_path.parent / f"{save_path.stem}({i + 1}){save_path.suffix}"
                              for i in range(num_images)]

        return save_paths

    def get_metric_save_path(self, metric_type, logger=None):
        args = self.args

        if args.val_save_path is None:
            exp_dir, ckpt_id = self.get_exp_dir_and_ckpt_id()
            val_save_dir = exp_dir / "evaluation"
            val_save_file = f"{ckpt_id}_{metric_type}{args.val_name_suffix}.json"
            val_save_path = str(val_save_dir / val_save_file)
            if logger is not None:
                logger.info(f"Set --val-save-path to {val_save_path} based on the checkpoint path.")
        else:
            val_save_path = args.val_save_path
        if not val_save_path.endswith(".json"):
            raise ValueError(f"--val-save-path should be a json file, but got {val_save_path}")

        return val_save_path

    @classmethod
    def from_pretrained(cls, dit_model_name_or_path, pretrained_model_path, args, device=None,
                        parallel_dims=None, world_mesh=None, model=None, **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v, text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (int): The device for inference. Default is 0.
        """
        # ========================================================================
        logger.info(f"Got text-to-video model root path: {pretrained_model_path}")

        # ==================== Initialize Distributed Environment ================
        # old
        # if nccl_info.sp_size > 1:
        #     device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        # if device is None:
        #     device = "cuda" if torch.cuda.is_available() else "cpu"



        # ======================== Get the args path =============================

        # Disable gradient
        torch.set_grad_enabled(False)
        if model is None:
            # =========================== Build main model ===========================
            logger.info("Building model...")
            # factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
            factor_kwargs = {'device': device,
                            'dtype': torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16}
            model = load_transformer(args.model_type, dit_model_name_or_path="null", factor_kwargs=factor_kwargs,
                                    args=args, convert_state_dict_16168_to_16164=args.convert_state_dict_16168_to_16164)
            # model = model.to(device)
            # model = torch.compile(model)
            model.eval()

            # Full safetensors contain ordinary Tensors. Load them before
            # RefrainEngine replaces parameters with sharded DTensors; ordinary
            # load_state_dict after FSDP wrapping fails on Tensor/DTensor copy.
            _dit_path = Path(dit_model_name_or_path)
            _safetensors_path = None
            if _dit_path.is_file() and _dit_path.suffix == ".safetensors":
                _safetensors_path = _dit_path
            elif _dit_path.is_dir():
                _st_files = sorted(_dit_path.glob("*.safetensors"))
                if _st_files and not list(_dit_path.glob("*.distcp")):
                    if len(_st_files) != 1:
                        raise ValueError(
                            f"Expected one full safetensors checkpoint in {_dit_path}, "
                            f"found {len(_st_files)}; sharded safetensors need an indexed loader."
                        )
                    _safetensors_path = _st_files[0]
            if _safetensors_path is not None:
                logger.info(f"Loading safetensors before FSDP wrapping: {_safetensors_path}")
                from safetensors import safe_open
                with safe_open(str(_safetensors_path), framework="pt", device="cpu") as checkpoint:
                    expected = model.state_dict()
                    actual_keys = set(checkpoint.keys())
                    missing = sorted(set(expected) - actual_keys)
                    unexpected = sorted(actual_keys - set(expected))
                    mismatched = [
                        (key, tuple(value.shape), tuple(checkpoint.get_slice(key).get_shape()))
                        for key, value in expected.items()
                        if key in actual_keys and tuple(value.shape) != tuple(checkpoint.get_slice(key).get_shape())
                    ]
                if missing or unexpected or mismatched:
                    raise ValueError(f"Checkpoint architecture mismatch: missing={missing}, "
                                     f"unexpected={unexpected}, shapes={mismatched}")
                _sd = safetensors_load_file(str(_safetensors_path))
                model.load_state_dict(_sd, strict=True)
                del _sd
                # Some inherited dense MLP constructors use PyTorch's default
                # device. FSDP requires every parameter on the local GPU.
                model.to(device=device)
                logger.info("Safetensors weights loaded successfully.")

            # ==================================================================
            from hymm.parallelism.engines.refrain_parallel import RefrainEngine

            model_engine = RefrainEngine(
                model,
                parallel_dims=parallel_dims,
                enable_autocast=True,

                # pp 相关参数
                auto_benchmark=False,
                m_microbatch=1,  # 需要所有rank的batchsize都能整除这个数
                # pipeline_parallel_schedule='1F1B', # 1F1B 需要 m_microbatch >= pp_size
            )
            model_engine.eval()
            model = model_engine

            logger.info(f"[from_pretrained] dit_model_name_or_path={dit_model_name_or_path}, "
                        f"is_file={_dit_path.is_file()}, suffix={_dit_path.suffix}")
            if _safetensors_path is None:
                _, _ = model_engine.load_checkpoint(dit_model_name_or_path)
            # ==================================================================
        else:
            model.eval()

        # ============================= Build extra models ========================
        # VAE
        vae, vae_path, s_ratio, t_ratio = load_vae(args.vae, args.vae_precision, logger=logger, device=device)

        # Text encoder
        text_encoder, text_encoder_2 = load_text_encoder(pretrained_model_path, device, logger, args)

        return cls(
            args=args,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model,
            device=device,
            logger=logger,
        )

    @staticmethod
    def load_state_dict(args, model, pretrained_model_path):
        load_key = args.load_key
        dit_weight = Path(args.dit_weight)

        if dit_weight is None:
            model_dir = pretrained_model_path / f"t2v_{args.model_resolution}"
            files = list(model_dir.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {model_dir}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                bare_model = True
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [f for f in files if str(f).endswith("_model_states.pt")]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(
                        f"Multiple model weights found in {dit_weight}, using {model_path}"
                    )
                bare_model = False
            else:
                raise ValueError(
                    f"Invalid model path: {dit_weight} with unrecognized weight format: "
                    f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                    f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                    f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                    f"specific weight file, please provide the full path to the file."
                )
        else:
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                if str(files[0]).startswith("pytorch_model_"):
                    model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    bare_model = True
                elif any(str(f).endswith("_model_states.pt") for f in files):
                    files = [f for f in files if str(f).endswith("_model_states.pt")]
                    model_path = files[0]
                    if len(files) > 1:
                        logger.warning(
                            f"Multiple model weights found in {dit_weight}, using {model_path}"
                        )
                    bare_model = False
                else:
                    raise ValueError(
                        f"Invalid model path: {dit_weight} with unrecognized weight format: "
                        f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                        f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                        f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                        f"specific weight file, please provide the full path to the file."
                    )
            elif dit_weight.is_file():
                model_path = dit_weight
                bare_model = "unknown"
            else:
                raise ValueError(f"Invalid model path: {dit_weight}")

        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        logger.info(f"Loading torch model {model_path}...")
        if model_path.suffix == ".safetensors":
            # Use safetensors library for .safetensors files
            state_dict = safetensors_load_file(model_path)
        elif model_path.suffix == ".pt":
            # Use torch for .pt files
            state_dict = torch.load(
                model_path, map_location=lambda storage, loc: storage
            )
        else:
            raise ValueError(f"Unsupported file format: {model_path}")

        if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(
                    f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                    f"are: {list(state_dict.keys())}."
                )
        new_dict = convert_hunyuan_dict_for_tensor_parallel(state_dict)
        model.load_state_dict(new_dict, strict=True)
        return model

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size


class HunyuanVideoSampler(Inference):
    def __init__(
        self,
        args,
        vae,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=0,
        logger=None,
    ):
        super().__init__(
            args,
            vae,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            use_cpu_offload=use_cpu_offload,
            device=device,
            logger=logger,
        )

        byt5_kwarg = None
        if args.glyph_byT5_v2:
            logger.info(f"Loading byT5 model and multilingual prompt formatter")
            byt5_kwarg = load_glyph_byT5_v2(args, device)

        self.pipeline = self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
            byt5_kwarg=byt5_kwarg,
            logger=logger
        )

        self.default_negative_prompt = NEGATIVE_PROMPT

    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
        data_type="video",
        byt5_kwarg=None,
        logger=None
    ):
        """Load the denoising scheduler for inference."""
        if scheduler is None:
            scheduler = FlowMatchDiscreteScheduler(
                shift=args.infer_flow_shift_video if data_type == "video" else args.infer_flow_shift,
                reverse=True,
                solver="euler",
            )

        extra_model = {}
        if args.multitask_semantic_mode == "siglip":
            vision_encoder = VisionEncoder(
                vision_encoder_type=args.vision_encoder,
                vision_encoder_precision=args.vision_encoder_precision,
                vision_encoder_path=args.vision_encoder_path,
                processor_type=args.processor_type,
                processor_path=args.processor_path,
                output_key=args.vision_output_key,
                logger=logger,
                device=device
            )
            extra_model["vision_encoder"] = vision_encoder

        pipeline = HunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
            byt5_kwarg=byt5_kwarg,
            device=device,
            is_eval=True,
            extra_model=extra_model,
        )
        # if self.use_cpu_offload:
        #     pipeline.enable_sequential_cpu_offload()
        # else:
        #     pipeline = pipeline.to(device)

        return pipeline

    @torch.no_grad()
    def predict(
        self,
        prompt,
        height=None,
        width=None,
        video_length=None,
        seed=None,
        negative_prompt=None,
        infer_steps=None,
        guidance_scale=6,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        output_type=None,
        generator=None,
        vae_ver=None,
        enable_tiling=None,
        is_progress_bar=None,
        # I2V related parameters
        multitask_mask_training_type=None,
        multitask_type=None,
        cond_latents=None,
        last_frame_cond_latents=None,
        multitask_mask=None,
        reference_image=None,
        vision_num_semantic_tokens=None,
        vision_states_dim=None,
        bucket_hw_base_size=None,
        bucket_hw_bucket_stride=None,
        **kwargs,
    ):
        """
        Predict the image/video from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                height (int): The height of the output video. Default is 192.
                width (int): The width of the output video. Default is 336.
                video_length (int): The frame number of the output video. Default is 129.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_images_per_prompt (int): The number of images per prompt. Default is 1.
                infer_steps (int): The number of inference steps. Default is 100.
                generator (torch.Generator or List[torch.Generator]): Random number generator(s). If provided, takes precedence over seed.
                vae_ver (str): VAE model version. If None, uses args.vae.
                enable_tiling (bool): Whether to enable VAE tiling. Default is True.
                is_progress_bar (bool): Whether to show progress bar during inference. Default is True.
                
                # I2V (Image-to-Video) related parameters:
                multitask_mask_training_type (str): Type of multitask mask training. "concat" for I2V.
                multitask_type (str): Task type, "i2v" for image-to-video, "t2v" for text-to-video.
                cond_latents (torch.Tensor): Conditional latents from reference image VAE encoding.
                multitask_mask (torch.Tensor): Mask tensor indicating which frames to condition on.
                reference_image (PIL.Image): Reference image for I2V generation.
                vision_num_semantic_tokens (int): Number of semantic tokens for vision encoder.
                vision_states_dim (int): Dimension of vision encoder states.
                bucket_hw_base_size (int): Base size for bucket-based image resizing.
                bucket_hw_bucket_stride (int): Stride for bucket-based image resizing.
        """

        out_dict = dict()

        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if generator is None:
            # Only create generator if not provided
            if isinstance(seed, torch.Tensor):
                seed = seed.tolist()
            if seed is None:
                seeds = [
                    random.randint(0, 1_000_000)
                    for _ in range(batch_size * num_videos_per_prompt)
                ]
            elif isinstance(seed, int):
                seeds = [
                    seed + i
                    for _ in range(batch_size)
                    for i in range(num_videos_per_prompt)
                ]
            elif isinstance(seed, (list, tuple)):
                if len(seed) == batch_size:
                    seeds = [
                        int(seed[i]) + j
                        for i in range(batch_size)
                        for j in range(num_videos_per_prompt)
                    ]
                elif len(seed) == batch_size * num_videos_per_prompt:
                    seeds = [int(s) for s in seed]
                else:
                    raise ValueError(
                        f"Length of seed must be equal to number of prompt(batch_size) or "
                        f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got [size={len(seed)}]{seed}."
                    )
            else:
                raise ValueError(
                    f"Seed must be an integer, a list of integers, or None, got {seed}."
                )
            # Peiyuan: using GPU seed will cause A100 and H100 to generate different results...
            generator = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
            out_dict["seeds"] = seeds
        else:
            # Use provided generator
            if not isinstance(generator, list):
                generator = [generator]
            seeds = [None] * len(generator)  # Seeds not needed when generator is provided
            out_dict["seeds"] = seeds

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        #if not isinstance(prompt, str):
        #    raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        #prompt = [prompt.strip()]

        # negative prompt
        #if negative_prompt is None or negative_prompt == "":
        #    negative_prompt = self.default_negative_prompt
        #if not isinstance(negative_prompt, str):
        #    raise TypeError(
        #        f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
        #    )
        #negative_prompt = [negative_prompt.strip()]
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt.strip()]
        elif isinstance(prompt, (list, tuple)):
            batch_size = len(prompt)
        else:
            raise ValueError(f"Prompt must be a string or a list of strings, got {prompt}.")

        negative_prompt = kwargs.get("negative_prompt", None)
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt.strip()] * batch_size

        text_segment_time_ranges = kwargs.get("text_segment_time_ranges")
        if text_segment_time_ranges is None:
            text_segment_time_ranges = _prepare_physrope_prompt_metadata(
                prompt,
                self.args,
                target_video_length,
                data_type="video" if target_video_length > 1 else "image",
            )

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=True,
            solver="euler",
        )
        self.pipeline.scheduler = scheduler

        latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        reorg_token = '888' in self.args.model_type or '16168' in self.args.model_type
        if reorg_token and latents_size[0] > 1:
            latents_size = [(latents_size[0] + 1) // 2, latents_size[1], latents_size[2]]

        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # ========================================================================
        # Print infer args
        # ========================================================================
        debug_str = f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                          seed: {seed if generator is None else 'provided via generator'}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
                   output_type: {output_type}
                     attn_mode: {self.args.attn_mode}
       embedded_guidance_scale: {embedded_guidance_scale}
     multitask_mask_training_type: {multitask_mask_training_type}
                multitask_type: {multitask_type}
                  cond_latents: {cond_latents.shape if cond_latents is not None else None}
        last_frame_cond_latents: {last_frame_cond_latents.shape if last_frame_cond_latents is not None else None}
                multitask_mask: {multitask_mask.shape if multitask_mask is not None else None}
               reference_image: {type(reference_image).__name__ if reference_image is not None else None}"""
        logger.debug(debug_str)

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        # Handle default values for parameters
        if vae_ver is None:
            vae_ver = self.args.vae
        if enable_tiling is None:
            enable_tiling = False
        if is_progress_bar is None:
            is_progress_bar = True

        prompt = _apply_prompt_fps_prefix(
            prompt,
            getattr(self.args, "prompt_fps_prefix", None),
            data_type="video" if target_video_length > 1 else "image",
        )
        if text_segment_time_ranges is not None:
            kwargs["text_segment_time_ranges"] = text_segment_time_ranges
        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type=output_type,
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=is_progress_bar,
            vae_ver=vae_ver,
            enable_tiling=enable_tiling,
            reorg_token=reorg_token,
            # I2V related parameters
            multitask_mask_training_type=multitask_mask_training_type,
            multitask_type=multitask_type,
            cond_latents=cond_latents,
            last_frame_cond_latents=last_frame_cond_latents,
            multitask_mask=multitask_mask,
            reference_image=reference_image,
            vision_num_semantic_tokens=vision_num_semantic_tokens,
            vision_states_dim=vision_states_dim,
            bucket_hw_base_size=bucket_hw_base_size,
            bucket_hw_bucket_stride=bucket_hw_bucket_stride,
            **kwargs  # Pass all i2v related parameters through kwargs
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict

    @torch.no_grad()
    def predict_video(
            self,
            prompt,
            height=None,
            width=None,
            video_length=None,
            seed=None,
            negative_prompt=None,
            infer_steps=None,
            guidance_scale=6,
            flow_shift=5.0,
            embedded_guidance_scale=None,
            batch_size=1,
            num_videos_per_prompt=1,
            output_type=None,
            generator=None,
            vae_ver=None,
            enable_tiling=None,
            is_progress_bar=None,
            # I2V related parameters
            multitask_mask_training_type=None,
            multitask_type=None,
            cond_latents=None,
            last_frame_cond_latents=None,
            multitask_mask=None,
            semantic_images=None,
            reference_image=None,
            vision_num_semantic_tokens=None,
            vision_states_dim=None,
            bucket_hw_base_size=None,
            bucket_hw_bucket_stride=None,
            **kwargs,
    ):
        """
        Predict the image/video from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                height (int): The height of the output video. Default is 192.
                width (int): The width of the output video. Default is 336.
                video_length (int): The frame number of the output video. Default is 129.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_images_per_prompt (int): The number of images per prompt. Default is 1.
                infer_steps (int): The number of inference steps. Default is 100.
                generator (torch.Generator or List[torch.Generator]): Random number generator(s). If provided, takes precedence over seed.
                vae_ver (str): VAE model version. If None, uses args.vae.
                enable_tiling (bool): Whether to enable VAE tiling. Default is True.
                is_progress_bar (bool): Whether to show progress bar during inference. Default is True.

                # I2V (Image-to-Video) related parameters:
                multitask_mask_training_type (str): Type of multitask mask training. "concat" for I2V.
                multitask_type (str): Task type, "i2v" for image-to-video, "t2v" for text-to-video.
                cond_latents (torch.Tensor): Conditional latents from reference image VAE encoding.
                multitask_mask (torch.Tensor): Mask tensor indicating which frames to condition on.
                reference_image (PIL.Image): Reference image for I2V generation.
                vision_num_semantic_tokens (int): Number of semantic tokens for vision encoder.
                vision_states_dim (int): Dimension of vision encoder states.
                bucket_hw_base_size (int): Base size for bucket-based image resizing.
                bucket_hw_bucket_stride (int): Stride for bucket-based image resizing.
        """

        out_dict = dict()

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        # if not isinstance(prompt, str):
        #    raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        # prompt = [prompt.strip()]

        # negative prompt
        # if negative_prompt is None or negative_prompt == "":
        #    negative_prompt = self.default_negative_prompt
        # if not isinstance(negative_prompt, str):
        #    raise TypeError(
        #        f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
        #    )
        # negative_prompt = [negative_prompt.strip()]
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt.strip()]
        elif isinstance(prompt, (list, tuple)):
            batch_size = len(prompt)
        else:
            raise ValueError(f"Prompt must be a string or a list of strings, got {prompt}.")

        negative_prompt = kwargs.get("negative_prompt", None)
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt.strip()] * batch_size

        text_segment_time_ranges = kwargs.get("text_segment_time_ranges")
        if text_segment_time_ranges is None:
            text_segment_time_ranges = _prepare_physrope_prompt_metadata(
                prompt,
                self.args,
                target_video_length,
                data_type="video" if target_video_length > 1 else "image",
            )

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=True,
            solver="euler",
        )
        self.pipeline.scheduler = scheduler

        latents_size = [(video_length - 1) // 4 + 1, height // 16, width // 16]
        reorg_token = '888' in self.args.model_type or '16168' in self.args.model_type
        if reorg_token and latents_size[0] > 1:
            latents_size = [(latents_size[0] + 1) // 2, latents_size[1], latents_size[2]]

        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # ========================================================================
        # Print infer args
        # ========================================================================
        debug_str = f"""
                           height: {target_height}
                            width: {target_width}
                     video_length: {target_video_length}
                           prompt: {prompt}
                       neg_prompt: {negative_prompt}
                             seed: {seed if generator is None else 'provided via generator'}
                      infer_steps: {infer_steps}
            num_videos_per_prompt: {num_videos_per_prompt}
                   guidance_scale: {guidance_scale}
                         n_tokens: {n_tokens}
                       flow_shift: {flow_shift}
                      output_type: {output_type}
                        attn_mode: {self.args.attn_mode}
          embedded_guidance_scale: {embedded_guidance_scale}
        multitask_mask_training_type: {multitask_mask_training_type}
                   multitask_type: {multitask_type}
                     cond_latents: {cond_latents.shape if cond_latents is not None else None}
                   last_frame_cond_latents: {last_frame_cond_latents.shape if last_frame_cond_latents is not None else None}
                   multitask_mask: {multitask_mask.shape if multitask_mask is not None else None}
                  reference_image: {type(reference_image).__name__ if reference_image is not None else None}"""
        logger.debug(debug_str)

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        # Handle default values for parameters
        if vae_ver is None:
            vae_ver = self.args.vae
        if enable_tiling is None:
            enable_tiling = False
        if is_progress_bar is None:
            is_progress_bar = True

        prompt = _apply_prompt_fps_prefix(
            prompt,
            getattr(self.args, "prompt_fps_prefix", None),
            data_type="video" if target_video_length > 1 else "image",
        )
        if text_segment_time_ranges is not None:
            kwargs["text_segment_time_ranges"] = text_segment_time_ranges
        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type=output_type,
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=is_progress_bar,
            vae_ver=vae_ver,
            enable_tiling=enable_tiling,
            reorg_token=reorg_token,
            # I2V related parameters
            multitask_mask_training_type=multitask_mask_training_type,
            multitask_type=multitask_type,
            cond_latents=cond_latents,
            last_frame_cond_latents=last_frame_cond_latents,
            multitask_mask=multitask_mask,
            semantic_images=semantic_images,
            reference_image=reference_image,
            vision_num_semantic_tokens=vision_num_semantic_tokens,
            vision_states_dim=vision_states_dim,
            bucket_hw_base_size=bucket_hw_base_size,
            bucket_hw_bucket_stride=bucket_hw_bucket_stride,
            **kwargs  # Pass all i2v related parameters through kwargs
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict
