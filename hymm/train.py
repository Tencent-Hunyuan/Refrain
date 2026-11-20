import logging
import math
import os
import re
import sys
import gc
import shutil
from hymm.parallelism.utils import sync_object_for_parallel_training, sync_random_states

from torch.distributed.tensor import DTensor

gc.set_threshold(7000, 100, 100)
import time
import datetime
import json
import random
from einops import rearrange
from accelerate.utils import set_seed
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
import torchvision
from torch.profiler import profile, record_function, ProfilerActivity

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version



from hymm.models.repa_encoder import RepaEncoder
from hymm.utils.load import load_transformer, load_text_encoder, load_glyph_byT5_v2
from hymm.models.vae import load_vae
from hymm.config import parse_args
from hymm.eval.eval_loss import LossEvaluator
from hymm.eval.eval_score_image import ScoreEvaluator
from hymm.constants import (
    C_SCALE,
    LOSS_METRICS,
    SCORE_METRICS,
    SAMPLE_METRICS,
    PROMPT_TEMPLATE,
    apply_model_region_override,
)
from hymm.dataset import get_image_dataloader
from hymm.dataset import get_video_dataloader
from hymm.dataset.caps.caption_process_v4 import CaptionAug
from hymm.dataset.video_image_iterator import VideoImageBatchIterator
from hymm.utils.parallel_states import (
    ParallelDims,
    initialize_sequence_parallel_state_v2,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from hymm.utils.dataset_utils import get_infinite_iterator
from hymm.utils.physrope_prompt_rewrite import rewrite_bracket_prompt_for_reference_conditions
from hymm.utils.prompt_cleaning import strip_identity_tags
from hymm.utils.video_layout import normalize_video_bcfhw
from hymm.utils.reference_temporal_pack import (
    REFERENCE_TIMELINE,
    normalize_temporal_pack_mode,
    scatter_reference_timeline_latent,
)
from hymm.utils.fsdp_util import (
    get_dit_fsdp_kwargs, 
    get_dit_fsdp_kwargs_v2,
    apply_fsdp_checkpointing,
    apply_fsdp2
)
from hymm.utils.logging_ import setup_logger
from hymm.utils.file_utils import save_videos_grid, validate_video_csv_files
from hymm.utils.checkpoint import (
    save_checkpoint_without_optim,
    resume_training,
)
from hymm.utils.helpers import ScalarStates, CycleStates, prepare_model_inputs, as_tuple, as_list_of_2tuple
from hymm.utils.curriculum_scheduler import CurriculumScheduler
from hymm.diffusion import load_denoiser
from hymm.diffusion.pipelines import HunyuanVideoPipeline
from hymm.diffusion.schedulers import FlowMatchDiscreteScheduler

from hymm.utils.torch_utils import all_gather_sum
from hymm.utils.multitask_utils import numpy_to_pil
import csv
from hymm.utils.multitask_utils import get_multitask_mask, get_cond_latents, get_multitask_mask_with_ranges
from hymm.utils.data_utils import get_closest_ratio, generate_crop_size_list
from hymm.utils.temporal_resample import resample_latent_temporal
from hymm.utils.native_reference import as_bool
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from hymm.models.vision_encoder import VisionEncoder



def _get_event_desc_pattern(prefix_zh=None, prefix_en=None):
    prefixes = []
    for prefix in (prefix_zh, prefix_en):
        if prefix is None:
            continue
        prefix = str(prefix)
        if prefix != "":
            prefixes.append(re.escape(prefix))
    prefix_part = rf"(?:{'|'.join(prefixes)})?\s*" if prefixes else r""
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
    if start is None or end is None:
        return None
    mode = str(mode or "second").strip().lower()
    if mode in ("second", "seconds"):
        start_value = float(start)
        end_value = float(end)
    elif mode == "latent":
        # Convert seconds → latent frame index.
        # 3D causal VAE: pixel_frame = round(sec * fps),
        # latent_idx = (pixel_frame - 1) // temporal_stride + 1 (0 maps to 0).
        # Matches hymm/inference.py _format_time_range and video_loader.py
        temporal_stride = 4
        start_sec = float(start)
        end_sec = float(end)
        start_pf = int(round(start_sec * fps))
        end_pf = int(round(end_sec * fps))
        start_value = 0 if start_pf <= 0 else (start_pf - 1) // temporal_stride + 1
        end_value = 0 if end_pf <= 0 else (end_pf - 1) // temporal_stride + 1
    else:
        # Frame mode: seconds → pixel frame index
        start_value = int(float(start) * fps)
        end_value = int(float(end) * fps)
    start_text = _format_time_value(start_value)
    end_text = _format_time_value(end_value)
    if start_text is None or end_text is None:
        return None
    return f"{start_text}-{end_text}"


def _parse_event_segments_from_prompt(prompt, time_range_mode, fps, prefix_zh=None, prefix_en=None):
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
        groups = match.groups()
        # Combined regex: groups 0,1 = seconds pattern ("Xs-Ys"),
        #                 groups 2,3 = bracket pattern ("[X-Y]").
        # Bracket values are already in target coords; seconds need conversion.
        if groups[0] is not None and groups[1] is not None:
            # Seconds pattern — convert via _format_time_range
            time_range = _format_time_range(groups[0], groups[1], time_range_mode, fps)
        elif len(groups) >= 4 and groups[2] is not None and groups[3] is not None:
            # Bracket pattern — values already in target coords, pass through
            s_text = _format_time_value(float(groups[2]))
            e_text = _format_time_value(float(groups[3]))
            time_range = f"{s_text}-{e_text}" if s_text and e_text else None
        else:
            continue
        segment_infos.append(
            {
                "segment_id": idx + 1,
                "text": prompt[segment_start:segment_end].strip(),
                "time_range": time_range,
            }
        )
    return segment_infos


def _build_prompt_from_segments(segment_infos):
    parts = []
    for segment in segment_infos:
        segment_text = str(segment.get("text", "")).strip()
        if segment_text:
            parts.append(segment_text)
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else "".join(parts)


def _prepare_physrope_prompt_inputs(prompt, args, text_encoder, device, data_type="video"):
    prompt = strip_identity_tags(prompt)
    if text_encoder is None:
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
        logger=logging.getLogger(__name__),
    )
    segment_infos = _parse_event_segments_from_prompt(
        prompt,
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
    prompt = _build_prompt_from_segments(segment_infos)
    text_ids, text_mask, segment_ids, segment_time_ranges = caption_aug.tokenize_segments(
        text_encoder,
        args.text_len,
        segment_infos,
        data_type=data_type,
    )
    if not torch.is_tensor(text_ids):
        text_ids = torch.tensor(text_ids)
    if not torch.is_tensor(text_mask):
        text_mask = torch.tensor(text_mask)
    text_ids = text_ids.unsqueeze(0).to(device)
    text_mask = text_mask.unsqueeze(0).to(device)
    text_outputs = text_encoder.encode(
        {"input_ids": text_ids, "attention_mask": text_mask},
        data_type=data_type,
        device=device,
    )
    return (
        prompt,
        text_outputs.hidden_state,
        text_outputs.attention_mask,
        segment_ids,
        segment_time_ranges,
    )


def _prepare_physrope_prompt_metadata(prompt, args, target_video_length, data_type="video"):
    """Parse time ranges from prompt and optionally rewrite prompt to latent format.

    Returns:
        (segment_time_ranges, rewritten_prompt) where rewritten_prompt has
        time prefixes in the target time_range_mode format (e.g. [0-15] for
        latent mode). Returns (None, prompt) if no segments found.
    """
    prompt = strip_identity_tags(prompt)
    if data_type != "video":
        return None, prompt
    if not getattr(args, "use_physical_rope", False) and not getattr(args, "use_visual_events", False):
        return None, prompt
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
        total_range_end = float((max(int(target_video_length) - 1, 0)) // 4 + 1)
    else:
        total_range_end = float(max(int(target_video_length) - 1, 0))

    segment_infos = _parse_event_segments_from_prompt(
        prompt,
        time_range_mode,
        rope_fps,
        "事件描述：",
        "Event description:",
    )
    if not segment_infos:
        return None, prompt

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

    # Rewrite prompt: replace seconds-format time prefixes with latent bracket format.
    # e.g. "0秒-2.5秒 desc" → "[0-15] desc"  (latent mode, fps=24)
    # Bracket format "[0-15] desc" is already in latent coords — keep as-is.
    rewritten_prompt = prompt
    if use_latent_time:
        temporal_stride = 4
        def _sec_to_latent(sec):
            pf = int(round(sec * rope_fps))
            if pf <= 0:
                return 0
            return (pf - 1) // temporal_stride + 1

        pattern = _get_event_desc_pattern("事件描述：", "Event description:")
        matches = list(pattern.finditer(prompt))
        if matches:
            parts = []
            prev_end = 0
            for match in matches:
                # Text before this match
                parts.append(prompt[prev_end:match.start()])
                groups = match.groups()
                if groups[0] is not None and groups[1] is not None:
                    # Seconds pattern (e.g. "0秒-2.5秒") — convert to latent
                    s_sec, e_sec = float(groups[0]), float(groups[1])
                    s_lat, e_lat = _sec_to_latent(s_sec), _sec_to_latent(e_sec)
                    parts.append(f"[{s_lat}-{e_lat}]")
                elif len(groups) >= 4 and groups[2] is not None and groups[3] is not None:
                    # Bracket pattern (e.g. "[0-15]") — already in latent coords, keep as-is
                    parts.append(match.group())
                else:
                    parts.append(match.group())
                prev_end = match.end()
            parts.append(prompt[prev_end:])
            rewritten_prompt = "".join(parts)

    if not getattr(_prepare_physrope_prompt_metadata, '_printed', False) and str(os.environ.get("RANK", "0")) == "0":
        _prepare_physrope_prompt_metadata._printed = True
        print(f"[PhysRoPE INFER-META] _prepare_physrope_prompt_metadata: "
              f"n_segments={len(segment_infos)} | n_rows={len(rows)} | "
              f"time_range_mode={time_range_mode} | rope_fps={rope_fps} | "
              f"total_range_end={total_range_end} | "
              f"rows={rows[:3]}{'...' if len(rows) > 3 else ''}")
        if rewritten_prompt != prompt:
            print(f"[PhysRoPE INFER-META] prompt rewritten:\n"
                  f"  ORIG: {prompt[:120]}\n"
                  f"  NEW:  {rewritten_prompt[:120]}")

    return rows or None, rewritten_prompt


def _apply_prompt_fps_prefix(prompt, fps_prefix, data_type="video"):
    prompt = strip_identity_tags(prompt)
    if data_type != "video" or fps_prefix is None:
        return prompt
    fps_prefix = str(fps_prefix).strip()
    if not fps_prefix:
        return prompt
    fps_prefix = fps_prefix.rstrip(",").strip() + ", "
    text = str(prompt).strip()
    if not text:
        return text
    if re.match(r"^FPS:\s*\d+(?:\.\d+)?,\s*", text):
        return text
    return f"{fps_prefix}{text}"


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


def _to_jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Common non-JSON-native objects
    try:
        from pathlib import Path

        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    try:
        import enum

        if isinstance(obj, enum.Enum):
            return _to_jsonable(obj.value)
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(obj, (np.generic,)):
            return obj.item()
    except Exception:
        pass

    try:
        if isinstance(obj, (torch.dtype, torch.device)):
            return str(obj)
    except Exception:
        pass

    # Last resort: stringify
    return str(obj)


def save_args_to_json(args, output_dir: str, filename: str = "config.json"):
    """Save all args into a JSON file under output_dir."""

    if output_dir is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, filename)

    # argparse.Namespace and alike
    if hasattr(args, "__dict__"):
        payload = {k: _to_jsonable(v) for k, v in vars(args).items() if not k.startswith("_")}
    else:
        payload = _to_jsonable(args)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)

def sync_cuda_time(sync=True, barrier=True): # For accurate time measurement
    if barrier:
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    if sync:
        torch.cuda.synchronize()
        t = time.time()
    else:
        t = time.time()
    return t
    
@torch.no_grad()
def evaluate_validation(args, device, vae, text_encoder, text_encoder_2, model,
                        global_step, logger, tb_writer, scalar_states,
                        loss_evaluator=None, score_evaluator=None, byt5_kwarg=None,
                        extra_model=None):
    try:
        logger.info(f"---------------------------- Evaluation Stage --------------------------")
        model.eval()
        #vae.enable_slicing()
        # gc before validation to avoid gpu memory error when validation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        logger.info(f"Start evaluation after step={scalar_states.update_steps} " +
                    (f"(update_step={scalar_states.update_steps:07d}) " if args.gradient_accumulation_steps > 1 else ""))

        # Performing validation on multiple image sizes.
        lst_of_image_sizes = as_list_of_2tuple(args.image_size)
        val_batch_sizes = as_tuple(args.val_batch_size)
        val_summary_events = []
        for vi, (image_size, batch_size) in enumerate(zip(lst_of_image_sizes, val_batch_sizes)):
            size = f'{image_size[0]}x{image_size[1]}'
            logger.info(
                f"================================= "
                f"[task-{vi + 1}] image size: {size}, batch size: {batch_size} "
                f"================================="
            )
            if loss_evaluator is not None:
                logger.info(f"---------------------------- Loss Evaluator ----------------------------")
                suffix = '' if vi == 0 else f"_{size}"  # For compatibility with single size.
                save_template = f'{args.output_dir}/evaluation/{scalar_states.update_steps:07d}_val_loss{suffix}.json'
                val_index_files = as_tuple(args.val_index_file)
                results = loss_evaluator.eval(index_file=val_index_files[vi],
                                            multireso=args.val_multireso,
                                            image_size=image_size,
                                            save_path=save_template,
                                            batch_size=batch_size,
                                            timesteps=args.val_timesteps,
                                            byt5_kwarg=byt5_kwarg,
                                            extra_model=extra_model,
                                            index_kwargs=dict(
                                                image_caption_rate=args.val_image_caption_rate,
                                                image_text_arrow_suffix=args.val_image_text_arrow_suffix,
                                                image_text_col=args.val_image_text_col,
                                                image_caption_arrow_suffix=args.val_image_caption_arrow_suffix,
                                                image_caption_col=args.val_image_caption_col,
                                                index_strategy=args.val_index_strategy,
                                                index_probability=args.val_index_probability,
                                                caption_sample_ratio=args.val_caption_sample_ratio
                                            ))
                val_loss = results[-1]['loss']
                # Only log metrics on rank 0
                if tb_writer is not None:
                    tb_writer.add_scalar(f"Val/Steps/val_loss{suffix}", val_loss, scalar_states.update_steps)
            if score_evaluator is not None:
                logger.info(f"---------------------------- Score Evaluator ----------------------------")
                suffix = '' if vi == 0 else f"_{size}"  # For compatibility with single size.
                image_save_base = f"{args.output_dir}/samples_val/{str(global_step).zfill(7)}/images" if args.score_save_image else None
                logger.info(f"image_save_base: {image_save_base}")
                save_template = f'{args.output_dir}/evaluation/{scalar_states.update_steps:07d}_score{suffix}.json'
                score_evaluator.load_score_models(image_size[0])
                results = score_evaluator.eval(image_size=image_size,
                                                batch_size=batch_size,
                                                save_path=save_template,
                                                image_save_base=image_save_base,
                                                )
                # Release score models to save GPU memory.
                score_evaluator.release_score_models(image_size[0])
                # Only log metrics on rank 0
                if tb_writer is not None:
                    for item in results:
                        tb_writer.add_scalar(f"Val/Steps/{item['metric']}{suffix}", item['value'], scalar_states.update_steps)

        vae.disable_slicing()
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")

    # gc after validation to free memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    model.train()


def _run_inline_evaluation(args, global_step, logger):
    """Run multishot evaluation metrics inline after fixed_prompt validation.

    Only called on dp_rank==0. Entire function is wrapped in try/except by caller,
    so any failure is non-fatal and training continues.
    """
    import subprocess
    import sys as _sys

    # Determine video directory from the saved samples
    output_dir = getattr(args, 'output_samples_dir', None) or getattr(args, 'output_dir', None)
    if not output_dir:
        logger.info("[InlineEval] No output_dir, skipping.")
        return

    # Find the video directories for each csv
    video_csv_list = getattr(args, 'video_csv', None) or []
    if isinstance(video_csv_list, str):
        video_csv_list = [video_csv_list]
    if not video_csv_list:
        logger.info("[InlineEval] No video_csv, skipping.")
        return

    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "run_all_metrics_one_dir.py")
    if not os.path.exists(eval_script):
        logger.info(f"[InlineEval] eval script not found: {eval_script}, skipping.")
        return

    # Also need to set up PYTHONPATH for evaluation imports
    eval_dir = os.path.dirname(os.path.abspath(eval_script))
    vbench_root = "assets/local/resource_c80d68a799"

    step_str = f"{global_step:07d}"

    for csv_path in video_csv_list:
        csv_path = str(csv_path).strip()
        if not csv_path or not os.path.exists(csv_path):
            continue

        # Reconstruct the video save directory: output_dir/samples/{step}/{folder_name}/{file_name}/
        parent_dir = os.path.basename(os.path.dirname(csv_path))
        file_name = os.path.basename(csv_path).replace('.csv', '')
        video_dir = os.path.join(output_dir, "samples", step_str, parent_dir, file_name)

        if not os.path.isdir(video_dir):
            logger.info(f"[InlineEval] video_dir not found: {video_dir}, skipping.")
            continue

        # Count mp4 files
        mp4_count = len([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
        if mp4_count == 0:
            logger.info(f"[InlineEval] No mp4 in {video_dir}, skipping.")
            continue

        # Result name
        exp_name = os.path.basename(output_dir) if output_dir else "unknown"
        result_name = f"{exp_name}_step{global_step}"
        results_root = os.path.join(eval_dir, "results_inline")

        # Check if already done
        summary_path = os.path.join(results_root, result_name, "summary.csv")
        if os.path.exists(summary_path):
            logger.info(f"[InlineEval] Already done: {summary_path}")
            continue

        logger.info(f"[InlineEval] Running evaluation: {mp4_count} videos in {video_dir}")
        logger.info(f"[InlineEval] CSV: {csv_path}, result_name: {result_name}")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{eval_dir}:{vbench_root}:" + env.get("PYTHONPATH", "")
        env["VBENCH_CACHE_DIR"] = os.path.join(eval_dir, "assets", "vbench_cache")
        # Force evaluation to use only 1 GPU (current device)
        env["CUDA_VISIBLE_DEVICES"] = str(os.environ.get("LOCAL_RANK", "0"))

        cmd = [
            _sys.executable, eval_script,
            "--video-dir", video_dir,
            "--csv", csv_path,
            "--result-name", result_name,
            "--results-root", results_root,
            "--device", "cuda",
            "--transition-device", "cuda",
            "--txt-first",
        ]

        logger.info(f"[InlineEval] cmd: {' '.join(cmd)}")

        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=1800,  # 30min timeout
        )

        if proc.returncode == 0:
            logger.info(f"[InlineEval] Success: {summary_path}")
            # Print summary
            if os.path.exists(summary_path):
                with open(summary_path) as _sf:
                    logger.info(f"[InlineEval] Summary: {_sf.read().strip()}")
        else:
            logger.warning(f"[InlineEval] Failed (rc={proc.returncode})")
            if proc.stderr:
                # Only log last 500 chars of stderr to avoid flooding
                logger.warning(f"[InlineEval] stderr (tail): ...{proc.stderr[-500:]}")


@torch.no_grad()
def sample_validation(args, device, vae, text_encoder, text_encoder_2, model,
                        batch, global_step, logger, byt5_kwarg, extra_model,
                        dp_rank, multitask_mask, mask_type, use_i2v_instruction, parallel_dims):
    logger.info(f"---------------------------- Sample Stage --------------------------")
    model.eval()
    # gc before validation to avoid gpu memory error when validation
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    seeds = [args.global_seed]
    #rank = torch.distributed.get_rank()
    generator = torch.Generator(device=device).manual_seed(args.global_seed)

    # unpack batch_args, same with broadcast_data in pretrain.py
    media, latents, text_ids, text_mask, kwargs = batch
    latents = latents.to(device)
  
    n_sample_each_node = 1
    base_images = media[:n_sample_each_node]

    prompts = kwargs['text'][:n_sample_each_node] if 'text' in kwargs else ['unkown_prompts']*n_sample_each_node
    if args.video_caption_i2v_instruction_enable:
        prompts_i2v_instruction = kwargs['text_i2v_instruction'][:n_sample_each_node] if 'text_i2v_instruction' in kwargs else ['unkown_prompts'] * n_sample_each_node
        valid_i2v_instruction = kwargs['valid_i2v_instruction'][:n_sample_each_node] if 'valid_i2v_instruction' in kwargs else ['unkown_prompts'] * n_sample_each_node

    if 'videoid' in kwargs: videoid = kwargs['videoid'][:n_sample_each_node]
    else: videoid = ['unknown']*n_sample_each_node
    last_frame_tensor = kwargs["last_frame_tensor"][:n_sample_each_node] if 'last_frame_tensor' in kwargs else None
    data_type = kwargs['type'][0]
    infer_flow_shift = args.infer_flow_shift_video if data_type == "video" else args.infer_flow_shift
    force_startup_fixed_prompt = bool(
        getattr(args, "try_first_iter", False)
        and (
            global_step == 0
            or getattr(args, "resume_fixed_prompt_step0", False)
        )
    )
    is_fixed_prompt_validation_step = (
        (global_step > 0 and global_step % args.fixed_prompt_validation_every == 0)
        or force_startup_fixed_prompt
    )

    if is_fixed_prompt_validation_step:
        infer_flow_shift = args.infer_flow_shift_video

    scheduler = FlowMatchDiscreteScheduler(
                shift=infer_flow_shift,
                reverse=True,
                solver="euler",
            )
    pipeline = HunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            args=args,
            byt5_kwarg=byt5_kwarg,
            extra_model=extra_model
        )

    # decode vae to pixel_values
    is_image = False
    base_images_need_denorm = False
    if len(base_images.shape) == 1:
        latents = latents[:n_sample_each_node].to(device)
        latents = latents * vae.config.scaling_factor
        base_images = pipeline.decode_latents(latents, enable_tiling=args.vae_tiling)  # b c t h w
        ndim = len(latents.shape) - 2

    elif len(base_images.shape) == 4:
        is_image = True
        base_images = base_images.unsqueeze(2)
        base_images_need_denorm = True
        ndim = 3
    elif len(base_images.shape) == 5:
        # Online-VAE datasets carry raw pixels as [B,F,C,H,W]. Normalize to
        # [B,C,F,H,W] before deriving target length or concatenating generated
        # and GT videos for sample logging.
        base_images = normalize_video_bcfhw(base_images)
        base_images_need_denorm = True
        ndim = 3
    else:
        raise ValueError(
            f"sample_validation expected media with 1, 4, or 5 dims, "
            f"got shape={tuple(base_images.shape)}"
        )

    target_length, target_height, target_width = base_images.shape[-3:]
    if dist.get_rank() == 0:
        logger.info(
            "[SampleValidation INPUT] "
            f"base_images_bcfhw={tuple(base_images.shape)} "
            f"target_frames={int(target_length)} "
            f"need_denorm={base_images_need_denorm}"
        )

    use_fixed_prompt = False

    if args.output_samples_dir is None:
        output_dir = args.output_dir
    else:
        output_dir = args.output_samples_dir

    if is_fixed_prompt_validation_step:
        args.try_first_iter = False
        fixed_prompt_path = args.video_csv
        if fixed_prompt_path is not None:
            prompts = []
            ref_image_path = []
            seeds = []
            indexs = []
            csv_file_names = []
            eval_cases = []  # New: full case dicts for JSON testset
            save_sample_flags = []
            base_images = None
            use_fixed_prompt = True

            for pmt_path in fixed_prompt_path:
                if pmt_path.endswith('.json'):
                    # ─── New JSON testset format ───
                    testset = json.load(open(pmt_path, 'r', encoding='utf-8'))
                    _cases = testset.get('cases', testset) if isinstance(testset, dict) else testset
                    if isinstance(_cases, dict):
                        _cases = _cases.get('cases', [])
                    parent_dir = os.path.basename(os.path.dirname(pmt_path))
                    file_name = os.path.basename(pmt_path).replace('.json', '')
                    for case in _cases:
                        prompts.append(strip_identity_tags(case['prompt']))
                        ref_image_path.append(None)
                        seeds.append(int(case.get('seed', 42)))
                        indexs.append(str(case.get('index', len(indexs)+1)))
                        csv_file_names.append((parent_dir, file_name))
                        eval_cases.append(case)
                        save_sample_flags.append(True)
                else:
                    # ─── Legacy CSV format (backward compatible) ───
                    reader = csv.DictReader(open(pmt_path, 'r', encoding="utf-8-sig"))
                    parent_dir = os.path.basename(os.path.dirname(pmt_path))
                    file_name = os.path.basename(pmt_path).replace('.csv', '')
                    for i, line in enumerate(reader):
                        indexs.append(line['index'])
                        prompts.append(strip_identity_tags(line['prompt']))
                        ref_image_path.append(line['ref_image_path'] if 'ref_image_path' in line and os.path.exists(line.get('ref_image_path', '')) else None)
                        seeds.append(int(line['seed']))
                        csv_file_names.append((parent_dir, file_name))
                        eval_cases.append(line)  # Store raw dict for potential extra columns
                        save_sample_flags.append(True)

            logger.info(f"Total inference samples: {len(prompts)}")
            total_samples = len(prompts)
            # Distribute all prompts across dp ranks evenly
            dp_size = getattr(parallel_dims, 'dp_mesh', None)
            dp_size = dp_size.size() if dp_size is not None else 1
            if use_fixed_prompt and total_samples > 0 and dp_size > 1:
                padded_total = int(math.ceil(total_samples / dp_size) * dp_size)
                pad_num = padded_total - total_samples
                for pad_i in range(pad_num):
                    src_i = pad_i % total_samples
                    prompts.append(prompts[src_i])
                    ref_image_path.append(ref_image_path[src_i])
                    seeds.append(seeds[src_i])
                    indexs.append(f"{indexs[src_i]}_pad{pad_i}")
                    csv_file_names.append(csv_file_names[src_i])
                    eval_cases.append(eval_cases[src_i])
                    save_sample_flags.append(False)
                if pad_num > 0:
                    logger.info(
                        f"Fixed-prompt samples padded from {total_samples} to {padded_total} "
                        f"for dp_size={dp_size}; padded samples run forward but are not saved."
                    )
                total_samples = padded_total
            # Assign prompts to this rank: rank 0 gets [0, dp_size, 2*dp_size, ...],
            # rank 1 gets [1, dp_size+1, ...], etc.
            my_indices = list(range(dp_rank, total_samples, dp_size))
            indexs = [indexs[i] for i in my_indices]
            prompts = [prompts[i] for i in my_indices]
            ref_image_path = [ref_image_path[i] for i in my_indices]
            seeds = [seeds[i] for i in my_indices]
            csv_file_names = [csv_file_names[i] for i in my_indices]
            eval_cases = [eval_cases[i] for i in my_indices]
            save_sample_flags = [save_sample_flags[i] for i in my_indices]
            logger.info(
                f"Rank {dp_rank}/{dp_size}: assigned {len(my_indices)} samples, "
                f"save={sum(1 for flag in save_sample_flags if flag)}"
            )
    infer_steps = args.infer_steps if data_type == "image" else args.infer_steps_video
    # print(f"Rank {dp_rank}, sample_validation Data type: {data_type} \t target_length: {target_length}\t target_height: {target_height}\t target_width: {target_width} \t flow_shift: {infer_flow_shift} \t infer_steps: {infer_steps} \t prompts: {prompts}")
    sample_videos = []
    gt_videos = []
    reorg_token = '888' in args.model_type or '16168' in args.model_type
    native_t_concat = as_bool(getattr(args, "range_condition_native_t_concat", False))
    temporal_pack_mode = normalize_temporal_pack_mode(
        getattr(args, "range_condition_temporal_pack_mode", "legacy_trilinear")
    )
    reference_timeline = temporal_pack_mode == REFERENCE_TIMELINE
    for sidx, prompt in enumerate(prompts):
        cur_condition_frame_ranges = []  # default: no range_condition
        cur_condition_physical_dst_frame_ranges = []
        cur_condition_src_frame_ranges = []
        cur_condition_kinds = []
        cur_condition_prompt_seg_ids = []
        cur_condition_prompt_ranges = []
        cur_native_reference_packed_ranges = []
        # Default for training-batch validation. Fixed-prompt validation below
        # overrides this from requested pixel-frame length. Online-VAE training
        # carries a scalar placeholder in batch["latents"], so do not assume a
        # cached [B,C,T,H,W] tensor exists before the fixed-prompt case is read.
        if isinstance(latents, torch.Tensor) and latents.ndim >= 3:
            latent_target_length = int(latents.shape[2])
        else:
            latent_target_length = (int(args.sample_n_frames) - 1) // 4 + 1
        if True:
            if args.multitask_mask_training_type == "concat" or args.multitask_mask_training_type == "token_replace":
                if use_fixed_prompt:
                    data_type = "video"
                    case = eval_cases[sidx] if eval_cases else {}

                    # ─── Task type routing ───
                    task_type = case.get('task_type', 'i2v' if ref_image_path[sidx] else 't2v')
                    conditions = case.get('conditions', None)

                    def _fixed_prompt_required_latent_length(case_conditions):
                        required_t = 0
                        if not isinstance(case_conditions, dict):
                            return required_t
                        for seg in case_conditions.get('segments', []):
                            try:
                                _, seg_de = [int(x) for x in seg['dst_range']]
                            except Exception:
                                continue
                            required_t = max(required_t, seg_de)
                        for anchor in case_conditions.get('anchors', []):
                            try:
                                required_t = max(required_t, int(anchor['frame_idx']) + 1)
                            except (KeyError, TypeError, ValueError):
                                continue
                        return required_t

                    # ─── Dynamic target length from JSON testset ───
                    if case.get('latent_target_length'):
                        latent_target_length = int(case['latent_target_length'])
                    else:
                        sample_n_frames = int(round(5 * float(args.video_fps))) + 1
                        latent_target_length = (
                            int((sample_n_frames - 1) // 8 + 1)
                            if reorg_token
                            else int((sample_n_frames - 1) // 4 + 1)
                        )
                    required_latent_target_length = _fixed_prompt_required_latent_length(conditions)
                    if required_latent_target_length > latent_target_length:
                        print(f"[FixedPrompt L4 WARN] expand_validation_latent_T "
                              f"from={latent_target_length} to={required_latent_target_length} "
                              f"case={indexs[sidx]} reason=condition_range_requires_target_length",
                              flush=True)
                        latent_target_length = required_latent_target_length
                    sample_n_frames = (
                        (latent_target_length - 1) * 8 + 1
                        if reorg_token
                        else (latent_target_length - 1) * 4 + 1
                    )
                    target_length = int(sample_n_frames)

                    # ─── Dynamic target size from JSON testset ───
                    if case.get('target_height') and case.get('target_width'):
                        target_height = int(case['target_height'])
                        target_width = int(case['target_width'])
                    else:
                        target_size = {
                            256: (192, 336),
                            480: (352, 624),
                            640: (480, 864),
                            720: (544, 960),
                            960: (720, 1280),
                            1440: (1080, 1920),
                        }
                        target_height, target_width = target_size[args.video_bucket_hw_base_size]

                    generator = torch.Generator(device=device).manual_seed(seeds[sidx])

                    if task_type == 't2v' and ref_image_path[sidx] is None:
                        multitask_mask, mask_type = get_multitask_mask(latent_target_length, {"i2v":{"prob":0.0}, "t2v":{"prob":1.0}}, seed=args.global_seed)
                    elif task_type == 't2v' and ref_image_path[sidx] is not None:
                        # Legacy i2v from CSV ref_image_path
                        multitask_mask, mask_type = get_multitask_mask(latent_target_length, {"i2v":{"prob":1.0}, "t2v":{"prob":0.0}}, seed=args.global_seed)
                    elif task_type in ('v2v', 'ti2v', 'tiv2v') and conditions is not None:
                        # ─── V2V / TI2V / TIV2V: load conditions ───
                        mask_type = "range_condition"
                        lat_h = target_height // 16
                        lat_w = target_width // 16
                        cond_latents = None if native_t_concat else torch.zeros(
                            1, 48, latent_target_length, lat_h, lat_w,
                            device=device, dtype=torch.float32,
                        )
                        native_condition_chunks = []
                        native_reference_packed_ranges = []
                        condition_frame_ranges = []
                        condition_physical_dst_frame_ranges = []
                        condition_src_frame_ranges = []
                        condition_kinds = []
                        condition_prompt_seg_ids = []
                        condition_prompt_ranges = []
                        sparse_condition_mask = torch.zeros(
                            latent_target_length, device=device, dtype=torch.float32
                        )
                        sparse_mapping_debug = []

                        def _store_fixed_prompt_condition(
                            condition_tensor,
                            ds,
                            physical_de,
                            visible_de=None,
                            kind=1,
                        ):
                            nonlocal cond_latents
                            nonlocal sparse_condition_mask
                            condition_tensor = condition_tensor.to(device=device, dtype=torch.float32)
                            if condition_tensor.ndim != 4 or condition_tensor.shape[0] != 48:
                                raise ValueError(
                                    f"fixed-prompt condition must be [48,T,H,W], got {tuple(condition_tensor.shape)}"
                                )
                            if native_t_concat:
                                padded = torch.zeros(
                                    48, condition_tensor.shape[1], lat_h, lat_w,
                                    device=device, dtype=torch.float32,
                                )
                                h_copy = min(int(condition_tensor.shape[2]), lat_h)
                                w_copy = min(int(condition_tensor.shape[3]), lat_w)
                                padded[:, :, :h_copy, :w_copy] = condition_tensor[:, :, :h_copy, :w_copy]
                                start = sum(int(chunk.shape[1]) for chunk in native_condition_chunks)
                                native_condition_chunks.append(padded)
                                native_reference_packed_ranges.append(
                                    (start, start + int(padded.shape[1]))
                                )
                                return None
                            elif reference_timeline and int(kind) == 1:
                                visible_de = int(physical_de if visible_de is None else visible_de)
                                padded = torch.zeros(
                                    48, condition_tensor.shape[1], lat_h, lat_w,
                                    device=device, dtype=torch.float32,
                                )
                                h_copy = min(int(condition_tensor.shape[2]), lat_h)
                                w_copy = min(int(condition_tensor.shape[3]), lat_w)
                                padded[:, :, :h_copy, :w_copy] = condition_tensor[:, :, :h_copy, :w_copy]
                                canvas, sparse_mask, mapping = scatter_reference_timeline_latent(
                                    padded,
                                    target_t=latent_target_length,
                                    physical_dst_range=(int(ds), int(physical_de)),
                                    visible_dst_range=(int(ds), visible_de),
                                )
                                cond_latents[0] = cond_latents[0] + canvas
                                sparse_condition_mask = torch.maximum(
                                    sparse_condition_mask, sparse_mask
                                )
                                sparse_mapping_debug.append(mapping)
                                return mapping
                            else:
                                visible_de = int(physical_de if visible_de is None else visible_de)
                                visible_t = min(visible_de - int(ds), int(condition_tensor.shape[1]))
                                h_copy = min(int(condition_tensor.shape[2]), lat_h)
                                w_copy = min(int(condition_tensor.shape[3]), lat_w)
                                cond_latents[0, :, ds:ds + visible_t, :h_copy, :w_copy] = (
                                    condition_tensor[:, :visible_t, :h_copy, :w_copy]
                                )
                                sparse_condition_mask[ds:ds + visible_t] = 1.0
                                return None

                        def _fixed_prompt_range_overlaps(existing_ranges, start, end):
                            return any(max(int(start), int(rs)) < min(int(end), int(re))
                                       for rs, re in existing_ranges)

                        def _parse_fixed_prompt_src_range(value):
                            if isinstance(value, str):
                                text = value.strip()
                                if (text.startswith("(") and text.endswith(")")) or (text.startswith("[") and text.endswith("]")):
                                    text = text[1:-1]
                                sep = "," if "," in text else "-"
                                value = [item.strip() for item in text.split(sep) if item.strip()]
                            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                                return None
                            try:
                                ss = int(round(float(value[0])))
                                se = int(round(float(value[1])))
                            except (TypeError, ValueError):
                                return None
                            if se <= ss:
                                return None
                            return ss, se

                        def _encode_fixed_prompt_video_path(video_path, latent_frames):
                            if not os.path.exists(video_path):
                                raise FileNotFoundError(f"video_path missing: {video_path}")
                            latent_frames = int(latent_frames)
                            if latent_frames <= 0:
                                raise ValueError(f"latent_frames must be positive, got {latent_frames}")
                            pixel_frame_count = 4 * (latent_frames - 1) + 1
                            import cv2

                            cap = cv2.VideoCapture(video_path)
                            try:
                                if not cap.isOpened():
                                    raise RuntimeError(f"cv2 failed to open video_path={video_path}")
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                if frame_count <= 0:
                                    raise RuntimeError(f"empty video_path={video_path}")
                                indices = np.linspace(0, frame_count - 1, pixel_frame_count).round().astype("int64")
                                frames = []
                                for frame_idx in indices:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                                    ok, frame_bgr = cap.read()
                                    if not ok or frame_bgr is None:
                                        raise RuntimeError(
                                            f"failed to read frame={int(frame_idx)} from video_path={video_path}"
                                        )
                                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                    src_h, src_w = frame_rgb.shape[:2]
                                    scale = max(target_width / src_w, target_height / src_h)
                                    resize_w = int(round(src_w * scale))
                                    resize_h = int(round(src_h * scale))
                                    frame_rgb = cv2.resize(
                                        frame_rgb,
                                        (resize_w, resize_h),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                    x0 = max((resize_w - target_width) // 2, 0)
                                    y0 = max((resize_h - target_height) // 2, 0)
                                    frames.append(frame_rgb[y0:y0 + target_height, x0:x0 + target_width])
                            finally:
                                cap.release()

                            frames_np = np.stack(frames, axis=0)
                            vae_dtype = next(vae.parameters()).dtype
                            pixel = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0)
                            pixel = pixel.to(device=device, dtype=vae_dtype)
                            pixel = pixel / 127.5 - 1.0
                            with torch.no_grad(), torch.autocast(
                                device_type=torch.device(device).type,
                                dtype=vae_dtype,
                                enabled=(torch.device(device).type == "cuda"),
                            ):
                                encoded = vae.encode(pixel).latent_dist.mode()
                            expected_shape = (1, 48, latent_frames, lat_h, lat_w)
                            if tuple(encoded.shape) != expected_shape:
                                raise RuntimeError(
                                    f"online video latent shape mismatch: got={tuple(encoded.shape)} "
                                    f"expected={expected_shape} path={video_path}"
                                )
                            return encoded[0].to(device=device, dtype=torch.float32)

                        def _encode_fixed_prompt_image_path(image_path):
                            if not os.path.exists(image_path):
                                raise FileNotFoundError(f"image anchor missing: {image_path}")
                            image = Image.open(image_path).convert("RGB")
                            origin_w, origin_h = image.size
                            scale = max(target_width / origin_w, target_height / origin_h)
                            resize_w = int(round(origin_w * scale))
                            resize_h = int(round(origin_h * scale))
                            image_transform = transforms.Compose([
                                transforms.Resize((resize_h, resize_w), interpolation=transforms.InterpolationMode.LANCZOS),
                                transforms.CenterCrop((target_height, target_width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5]),
                            ])
                            vae_dtype = next(vae.parameters()).dtype
                            pixel = image_transform(image).unsqueeze(0).unsqueeze(2).to(device=device, dtype=vae_dtype)
                            with torch.no_grad(), torch.autocast(
                                device_type=torch.device(device).type,
                                dtype=vae_dtype,
                                enabled=(torch.device(device).type == "cuda"),
                            ):
                                encoded = vae.encode(pixel).latent_dist.mode()
                            expected_shape = (1, 48, 1, lat_h, lat_w)
                            if tuple(encoded.shape) != expected_shape:
                                raise RuntimeError(
                                    f"image anchor latent shape mismatch: got={tuple(encoded.shape)} "
                                    f"expected={expected_shape} path={image_path}"
                                )
                            return encoded[0].to(device=device, dtype=torch.float32)

                        # Load video segment conditions
                        for seg in conditions.get('segments', []):
                            try:
                                ds, de = [int(x) for x in seg['dst_range']]
                            except Exception as exc:
                                print(f"[FixedPrompt L4 SKIP] invalid dst_range={seg.get('dst_range', None)} "
                                      f"case={indexs[sidx]} exc={repr(exc)}", flush=True)
                                continue
                            if ds < 0 or de <= ds or ds >= latent_target_length:
                                print(f"[FixedPrompt L4 SKIP] dst_range_out_of_bounds dst=({ds},{de}) "
                                      f"latent_T={latent_target_length} case={indexs[sidx]}", flush=True)
                                continue
                            lp = seg.get('latent_path', '')
                            video_path = seg.get('video_path', '') or seg.get('path', '')
                            if lp and os.path.exists(lp):
                                try:
                                    seg_lat = np.load(lp)
                                except Exception as exc:
                                    print(f"[FixedPrompt L4 SKIP] latent_load_failed path={lp} "
                                          f"case={indexs[sidx]} exc={repr(exc)}", flush=True)
                                    continue
                                if seg_lat.ndim == 5:
                                    if seg_lat.shape[0] != 1:
                                        print(f"[FixedPrompt L4 SKIP] segment latent batch dim must be 1, "
                                              f"got shape={tuple(seg_lat.shape)} path={lp} case={indexs[sidx]}", flush=True)
                                        continue
                                    seg_lat = seg_lat[0]
                                if seg_lat.ndim != 4:
                                    print(f"[FixedPrompt L4 SKIP] segment latent ndim must be 4 after squeeze, "
                                          f"got shape={tuple(seg_lat.shape)} path={lp} case={indexs[sidx]}", flush=True)
                                    continue
                                # seg_lat shape: (C, T_seg, H, W) after removing batch dim
                                try:
                                    seg_tensor = torch.from_numpy(seg_lat).to(device=device, dtype=torch.float32)
                                except (TypeError, ValueError, RuntimeError) as exc:
                                    print(f"[FixedPrompt L4 SKIP] latent_tensor_convert_failed "
                                          f"shape={tuple(seg_lat.shape)} path={lp} case={indexs[sidx]} "
                                          f"exc={repr(exc)}", flush=True)
                                    continue
                                t_src = int(seg_tensor.shape[1])
                                cond_path = lp
                            elif video_path:
                                explicit_src_for_online = seg.get('src_range', seg.get('src_latent_range', None))
                                parsed_src_for_online = _parse_fixed_prompt_src_range(explicit_src_for_online)
                                if explicit_src_for_online is not None and parsed_src_for_online is None:
                                    print(f"[FixedPrompt L4 SKIP] src_range_parse_failed src_range={explicit_src_for_online} "
                                          f"path={video_path} case={indexs[sidx]}", flush=True)
                                    continue
                                if parsed_src_for_online is not None:
                                    t_src_for_online = parsed_src_for_online[1] - parsed_src_for_online[0]
                                else:
                                    t_src_for_online = int(seg.get('src_latent_length', de - ds))
                                try:
                                    seg_tensor = _encode_fixed_prompt_video_path(video_path, t_src_for_online)
                                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                                    print(f"[FixedPrompt L4 SKIP] video_path_online_encode_failed "
                                          f"path={video_path} src_T={t_src_for_online} case={indexs[sidx]} "
                                          f"exc={repr(exc)}", flush=True)
                                    continue
                                t_src = int(seg_tensor.shape[1])
                                cond_path = video_path
                            elif lp:
                                print(f"[FixedPrompt L4 SKIP] latent_path missing path={lp} case={indexs[sidx]}", flush=True)
                                continue
                            else:
                                print(f"[FixedPrompt L4 SKIP] missing latent_path/video_path case={indexs[sidx]}", flush=True)
                                continue

                            if True:
                                dst_end = min(de, latent_target_length)
                                if dst_end != de:
                                    print(f"[FixedPrompt L4 WARN] dst_range_visible_overlap physical=({ds},{de}) "
                                          f"visible=({ds},{dst_end}) latent_T={latent_target_length} "
                                          f"case={indexs[sidx]}", flush=True)
                                seg_t = dst_end - ds
                                if _fixed_prompt_range_overlaps(condition_frame_ranges, ds, dst_end):
                                    print(f"[FixedPrompt L4 SKIP] condition_overlap dst=({ds},{dst_end}) "
                                          f"existing={condition_frame_ranges} path={cond_path} case={indexs[sidx]}",
                                          flush=True)
                                    continue
                                # seg_tensor dims: [C=48, T_file, H_file, W_file]
                                # cond_latents target: [48, seg_t, lat_h, lat_w]
                                # L4: T_file may differ from seg_t; resample temporally
                                # instead of cropping so variable-length references are
                                # actually used.
                                explicit_src = seg.get('src_range', seg.get('src_latent_range', None))
                                if explicit_src is None and ('src_start' in seg or 'src_end' in seg):
                                    explicit_src = [seg.get('src_start', ds), seg.get('src_end', ds + t_src)]
                                parsed_src = _parse_fixed_prompt_src_range(explicit_src)
                                if explicit_src is not None and parsed_src is None:
                                    print(f"[FixedPrompt L4 SKIP] src_range_parse_failed src_range={explicit_src} "
                                          f"path={cond_path} case={indexs[sidx]}", flush=True)
                                    continue
                                if parsed_src is not None:
                                    ss, se = parsed_src
                                    if se - ss != t_src:
                                        original_se = se
                                        se = ss + t_src
                                        print(f"[FixedPrompt L4 ADJUST] src_range_length_mismatch "
                                              f"src=({ss},{original_se}) len={original_se - ss} "
                                              f"raw_t_src={t_src} adjusted_src=({ss},{se}) "
                                              f"path={cond_path} case={indexs[sidx]}", flush=True)
                                else:
                                    ss, se = ds, ds + t_src
                                h_avail = min(seg_tensor.shape[2], lat_h)
                                w_avail = min(seg_tensor.shape[3], lat_w)
                                if seg_tensor.shape[0] != 48 or t_src <= 0 or seg_t <= 0 or h_avail <= 0 or w_avail <= 0:
                                    print(f"[FixedPrompt L4 SKIP] bad segment latent shape={tuple(seg_tensor.shape)} "
                                          f"dst=({ds},{de}) case={indexs[sidx]}", flush=True)
                                    continue
                                seg_tensor = seg_tensor[:, :, :h_avail, :w_avail]
                                dst_len = de - ds
                                if (not native_t_concat) and (not reference_timeline) and t_src != dst_len:
                                    try:
                                        seg_tensor = resample_latent_temporal(
                                            seg_tensor,
                                            src_frames=t_src,
                                            dst_frames=dst_len,
                                            align_corners=False,
                                            mode="trilinear",
                                        )
                                    except (ValueError, RuntimeError) as exc:
                                        print(f"[FixedPrompt L4 SKIP] temporal_resample_failed "
                                              f"src_T={t_src} dst_T={dst_len} path={cond_path} case={indexs[sidx]} "
                                              f"exc={repr(exc)}", flush=True)
                                        continue
                                if (not native_t_concat) and (not reference_timeline) and seg_tensor.shape[1] != dst_len:
                                    print(f"[FixedPrompt L4 SKIP] resampled_temporal_mismatch "
                                          f"T={seg_tensor.shape[1]} dst_len={dst_len} "
                                          f"path={cond_path} case={indexs[sidx]}", flush=True)
                                    continue
                                if (not native_t_concat) and (not reference_timeline):
                                    seg_tensor = seg_tensor[:, :seg_t]
                                _store_fixed_prompt_condition(
                                    seg_tensor,
                                    ds,
                                    de,
                                    visible_de=dst_end,
                                    kind=1,
                                )
                                condition_frame_ranges.append(
                                    (ds, dst_end) if native_t_concat else (ds, de)
                                )
                                condition_physical_dst_frame_ranges.append((ds, de))
                                condition_src_frame_ranges.append((ss, se))
                                condition_kinds.append(1)
                                try:
                                    seg_id = int(seg.get('seg_id', -1))
                                except (TypeError, ValueError):
                                    seg_id = -1
                                if seg_id > 0:
                                    condition_prompt_seg_ids.append(seg_id)
                                else:
                                    condition_prompt_ranges.append((ds, de))

                        # Load GT latent for anchor frames
                        gt_latent_for_anchor = None
                        if conditions.get('anchors') and case.get('gt_latent_cos_path'):
                            gt_cos_path = case['gt_latent_cos_path']
                            cache_folder = os.environ.get("REFRAIN_LATENT_ROOT", "train/latents")
                            gt_full_path = os.path.join(cache_folder, gt_cos_path)
                            if os.path.exists(gt_full_path):
                                try:
                                    gt_latent_for_anchor = torch.from_numpy(np.load(gt_full_path)).to(
                                        device=device, dtype=torch.float32)
                                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                                    print(f"[FixedPrompt L4 SKIP] gt_anchor_latent_load_failed "
                                          f"path={gt_full_path} case={indexs[sidx]} exc={repr(exc)}",
                                          flush=True)
                                    gt_latent_for_anchor = None
                                if gt_latent_for_anchor is not None:
                                    if gt_latent_for_anchor.ndim == 5:
                                        if gt_latent_for_anchor.shape[0] != 1:
                                            print(f"[FixedPrompt L4 SKIP] gt_anchor_latent_batch_dim "
                                                  f"shape={tuple(gt_latent_for_anchor.shape)} "
                                                  f"path={gt_full_path} case={indexs[sidx]}",
                                                  flush=True)
                                            gt_latent_for_anchor = None
                                        else:
                                            gt_latent_for_anchor = gt_latent_for_anchor[0]
                                    if gt_latent_for_anchor is not None and gt_latent_for_anchor.ndim != 4:
                                        print(f"[FixedPrompt L4 SKIP] gt_anchor_latent_ndim "
                                              f"shape={tuple(gt_latent_for_anchor.shape)} "
                                              f"path={gt_full_path} case={indexs[sidx]}",
                                              flush=True)
                                        gt_latent_for_anchor = None
                                    if gt_latent_for_anchor is not None:
                                        expected_tail = (48, lat_h, lat_w)
                                        got_tail = (
                                            gt_latent_for_anchor.shape[0],
                                            gt_latent_for_anchor.shape[2],
                                            gt_latent_for_anchor.shape[3],
                                        )
                                        if got_tail != expected_tail or gt_latent_for_anchor.shape[1] <= 0:
                                            print(f"[FixedPrompt L4 SKIP] gt_anchor_latent_shape_mismatch "
                                                  f"got={tuple(gt_latent_for_anchor.shape)} "
                                                  f"expected_C/H/W={expected_tail} "
                                                  f"path={gt_full_path} case={indexs[sidx]}",
                                                  flush=True)
                                            gt_latent_for_anchor = None
                            else:
                                print(f"[FixedPrompt L4 SKIP] gt_anchor_latent_missing "
                                      f"path={gt_full_path} case={indexs[sidx]}",
                                      flush=True)

                        for anchor in conditions.get('anchors', []):
                            try:
                                t_anchor = int(anchor['frame_idx'])
                            except (KeyError, TypeError, ValueError) as exc:
                                print(f"[FixedPrompt L4 SKIP] invalid anchor frame_idx={anchor.get('frame_idx', None)} "
                                      f"case={indexs[sidx]} exc={repr(exc)}", flush=True)
                                continue
                            if t_anchor < 0 or t_anchor >= latent_target_length:
                                print(f"[FixedPrompt L4 SKIP] anchor_out_of_bounds frame_idx={t_anchor} "
                                      f"latent_T={latent_target_length} case={indexs[sidx]}", flush=True)
                                continue
                            if _fixed_prompt_range_overlaps(condition_frame_ranges, t_anchor, t_anchor + 1):
                                print(f"[FixedPrompt L4 SKIP] anchor_overlap frame_idx={t_anchor} "
                                      f"existing={condition_frame_ranges} case={indexs[sidx]}", flush=True)
                                continue
                            anchor_path = anchor.get('path', '') or anchor.get('image_path', '')
                            anchor_loaded = False
                            if gt_latent_for_anchor is not None and t_anchor < gt_latent_for_anchor.shape[1]:
                                _store_fixed_prompt_condition(
                                    gt_latent_for_anchor[:, t_anchor:t_anchor+1],
                                    t_anchor,
                                    t_anchor + 1,
                                    kind=2,
                                )
                                condition_frame_ranges.append((t_anchor, t_anchor + 1))
                                condition_physical_dst_frame_ranges.append((t_anchor, t_anchor + 1))
                                condition_src_frame_ranges.append((t_anchor, t_anchor + 1))
                                condition_kinds.append(2)
                                anchor_loaded = True
                            elif gt_latent_for_anchor is not None:
                                print(f"[FixedPrompt L4 SKIP] anchor_gt_too_short frame_idx={t_anchor} "
                                      f"gt_T={gt_latent_for_anchor.shape[1]} case={indexs[sidx]}",
                                      flush=True)
                            if (not anchor_loaded) and anchor_path:
                                try:
                                    anchor_tensor = _encode_fixed_prompt_image_path(anchor_path)
                                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                                    print(f"[FixedPrompt L4 SKIP] anchor_image_encode_failed "
                                          f"path={anchor_path} frame_idx={t_anchor} case={indexs[sidx]} "
                                          f"exc={repr(exc)}", flush=True)
                                    continue
                                _store_fixed_prompt_condition(
                                    anchor_tensor, t_anchor, t_anchor + 1, kind=2
                                )
                                condition_frame_ranges.append((t_anchor, t_anchor + 1))
                                condition_physical_dst_frame_ranges.append((t_anchor, t_anchor + 1))
                                condition_src_frame_ranges.append((t_anchor, t_anchor + 1))
                                condition_kinds.append(2)

                        if condition_frame_ranges:
                            candidate_mask = (
                                sparse_condition_mask
                                if reference_timeline
                                else get_multitask_mask_with_ranges(
                                    latent_target_length, condition_frame_ranges, device=device
                                )
                            )
                            cond_frame_count = int(candidate_mask.sum().item())
                            if (
                                cond_frame_count >= latent_target_length
                                and not native_t_concat
                                and not reference_timeline
                            ):
                                print(f"[FixedPrompt L4 SKIP] condition_union_full_frame "
                                      f"cond_frames={cond_frame_count} latent_T={latent_target_length} "
                                      f"ranges={condition_frame_ranges} case={indexs[sidx]} "
                                      f"falling_back_to_t2v", flush=True)
                                multitask_mask, mask_type = get_multitask_mask(
                                    latent_target_length, {"t2v": {"prob": 1.0}}, seed=args.global_seed)
                                cond_latents = None
                                cur_condition_prompt_seg_ids = []
                                cur_condition_prompt_ranges = []
                            else:
                                multitask_mask = candidate_mask
                                if native_t_concat:
                                    if not native_condition_chunks:
                                        raise RuntimeError(
                                            "fixed-prompt native T concat has ranges but no reference chunks"
                                        )
                                    cond_latents = torch.cat(native_condition_chunks, dim=1).unsqueeze(0)
                                    cur_native_reference_packed_ranges = list(
                                        native_reference_packed_ranges
                                    )
                                cond_latents = cond_latents * vae.config.scaling_factor
                                cur_condition_frame_ranges = condition_frame_ranges
                                cur_condition_physical_dst_frame_ranges = (
                                    condition_physical_dst_frame_ranges
                                )
                                cur_condition_src_frame_ranges = condition_src_frame_ranges
                                cur_condition_kinds = condition_kinds
                                cur_condition_prompt_seg_ids = condition_prompt_seg_ids
                                cur_condition_prompt_ranges = condition_prompt_ranges
                        else:
                            # No valid conditions loaded → fallback to t2v
                            multitask_mask, mask_type = get_multitask_mask(
                                latent_target_length, {"t2v":{"prob":1.0}}, seed=args.global_seed)
                            cond_latents = None
                            cur_condition_prompt_seg_ids = []
                            cur_condition_prompt_ranges = []
                    else:
                        # Fallback
                        multitask_mask, mask_type = get_multitask_mask(latent_target_length, {"t2v":{"prob":1.0}}, seed=args.global_seed)
                        cur_condition_prompt_seg_ids = []
                        cur_condition_prompt_ranges = []

                    if mask_type == "i2v":
                        bucket_hw_base_size = args.video_bucket_hw_base_size
                        bucket_hw_bucket_stride = args.video_bucket_hw_bucket_stride
                        ref_images = [Image.open(image_path).convert('RGB') for image_path in [ref_image_path[sidx]] if image_path is not None]
                        assert len(ref_images) == 1, "Only support one reference image for i2v task."
                        origin_size = ref_images[0].size
                        assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960], \
                            f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960], but got {bucket_hw_base_size}"

                        crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
                        aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
                        closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

                        # 修正resize策略，避免黑边
                        target_height, target_width = closest_size
                        original_width, original_height = origin_size

                        # 计算保持宽高比的resize尺寸，确保能完全覆盖目标尺寸
                        scale_factor = max(target_width / original_width, target_height / original_height)
                        resize_width = int(round(original_width * scale_factor))
                        resize_height = int(round(original_height * scale_factor))

                        ref_image_transform = transforms.Compose([
                            transforms.Resize((resize_height, resize_width), interpolation=transforms.InterpolationMode.LANCZOS),
                            transforms.CenterCrop((target_height, target_width)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])

                        ref_images_pixel_values = [ref_image_transform(ref_image) for ref_image in ref_images]
                        ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(device)
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                            cond_latents = pipeline.vae.encode(ref_images_pixel_values).latent_dist.mode() # B, C, F, H, W
                            cond_latents.mul_(pipeline.vae.config.scaling_factor)

                        latents = cond_latents.repeat(1, 1, latent_target_length, 1, 1)

                        target_height = closest_size[0]
                        target_width = closest_size[1]
                # else:
                #     multitask_mask, mask_type = get_multitask_mask(latents.shape[2] if data_type == "video" else 1, args.multitask_mask_training_args, seed=args.local_seed)
                if mask_type == "i2v":
                    if args.video_caption_i2v_instruction_enable and not use_fixed_prompt:
                        valid_i2v_instruction_i = valid_i2v_instruction[sidx]
                        if use_i2v_instruction:
                            prompt = prompts_i2v_instruction[sidx]

                    cond_latents, _, semantic_images_np, semantic_images_pil = get_cond_latents(args, latents, vae,
                                                                                             is_uncond=False, is_mode=True)
                    last_frame_cond_latents = None
                elif mask_type == "range_condition":
                    # For fixed-prompt JSON testset: cond_latents was already built
                    # from file-based conditions at the correct target dimensions
                    # (lines 852-898 above). Skip the training-batch GT fallback.
                    if use_fixed_prompt and conditions is not None and cond_latents is not None:
                        # cond_latents already loaded from files with correct shape
                        pass
                    else:
                        # Training-batch validation fallback: use GT latent from batch
                        # NOTE: `latents` is already scaled by vae.config.scaling_factor
                        # at the top of sample_validation (line ~699). Do NOT scale again.
                        gt_latents = latents[:n_sample_each_node].to(device)
                        rc_latent_target_length = (
                            int(multitask_mask.numel()) if multitask_mask is not None
                            else int(latent_target_length)
                        )
                        # Truncate/pad to current validation latent length.
                        if gt_latents.shape[2] >= rc_latent_target_length:
                            gt_latents = gt_latents[:, :, :rc_latent_target_length, :, :]
                        else:
                            # Pad by repeating last frame
                            pad_len = rc_latent_target_length - gt_latents.shape[2]
                            gt_latents = torch.cat([gt_latents, gt_latents[:, :, -1:].repeat(1, 1, pad_len, 1, 1)], dim=2)
                        if multitask_mask is None:
                            raise ValueError("[Unified PhysRoPE] sample_validation range_condition requires multitask_mask")
                        cond_mask_bool = (multitask_mask == 1)
                        if native_t_concat:
                            if not cur_condition_frame_ranges:
                                active_indices = torch.nonzero(
                                    cond_mask_bool, as_tuple=False
                                ).flatten().detach().cpu().tolist()
                                derived_ranges = []
                                for frame_idx in active_indices:
                                    if not derived_ranges or frame_idx != derived_ranges[-1][1]:
                                        derived_ranges.append([frame_idx, frame_idx + 1])
                                    else:
                                        derived_ranges[-1][1] = frame_idx + 1
                                cur_condition_frame_ranges = [tuple(item) for item in derived_ranges]
                                cur_condition_src_frame_ranges = list(cur_condition_frame_ranges)
                            native_chunks = [
                                gt_latents[:, :, int(start):int(end)].clone()
                                for start, end in cur_condition_frame_ranges
                            ]
                            if not native_chunks:
                                native_chunks = [gt_latents[:, :, cond_mask_bool].clone()]
                            cond_latents = torch.cat(native_chunks, dim=2)
                            cur_native_reference_packed_ranges = []
                            native_cursor = 0
                            for chunk in native_chunks:
                                cur_native_reference_packed_ranges.append(
                                    (native_cursor, native_cursor + int(chunk.shape[2]))
                                )
                                native_cursor += int(chunk.shape[2])
                        else:
                            # Build full target-canvas conditions for legacy concat.
                            cond_latents = torch.zeros_like(gt_latents)
                            cond_latents[:, :, cond_mask_bool] = gt_latents[:, :, cond_mask_bool]
                    semantic_images_np = None
                    semantic_images_pil = None
                    last_frame_cond_latents = None
                elif mask_type == "t2v":
                    cond_latents = None
                    semantic_images_np = None
                    semantic_images_pil = None
                    last_frame_cond_latents = None

                elif mask_type == "fl2v":
                    (cond_latents,
                    last_frame_cond_latents,  
                    semantic_images_np, 
                    semantic_images_pil) = get_cond_latents(args, latents, vae,
                                                           last_frame_tensor = last_frame_tensor,
                                                            is_uncond=False, 
                                                            is_mode=True)
                else:
                    raise ValueError(f"mask_type {mask_type} not supported")
            else:
                if use_fixed_prompt:
                    data_type = "image"
                    target_length = 1
                    target_height, target_width = args.val_image_size[0], args.val_image_size[1]
                    generator = torch.Generator(device=device).manual_seed(seeds[sidx])
                mask_type = "t2v"
                cond_latents = None
                multitask_mask = None
                semantic_images_np = None
                semantic_images_pil = None
                last_frame_cond_latents = None

            text_segment_time_ranges = None
            if use_fixed_prompt and isinstance(prompt, str):
                # ─── Rewrite prompt: explicit reference prompts for conditioned segments ───
                _has_prompt_anchors = (
                    isinstance(conditions, dict)
                    and bool(conditions.get("anchors"))
                )
                if (
                    cur_condition_prompt_seg_ids
                    or cur_condition_prompt_ranges
                    or _has_prompt_anchors
                ) and mask_type == "range_condition":
                    _global_prompt_prefix = ""
                    if use_fixed_prompt and isinstance(case, dict) and "global_prompt_prefix" in case:
                        _global_prompt_prefix = strip_identity_tags(
                            str(case["global_prompt_prefix"]).strip()
                        )
                    prompt, _ = rewrite_bracket_prompt_for_reference_conditions(
                        prompt,
                        condition_prompt_seg_ids=cur_condition_prompt_seg_ids,
                        condition_prompt_ranges=cur_condition_prompt_ranges,
                        conditions=conditions,
                        global_prompt_prefix=_global_prompt_prefix,
                    )
                    prompt = strip_identity_tags(prompt)

                text_segment_time_ranges, prompt = _prepare_physrope_prompt_metadata(
                    prompt,
                    args,
                    target_length,
                    data_type=data_type,
                )

            if not getattr(sample_validation, '_physrope_val_printed', False) and str(os.environ.get("RANK", "0")) == "0":
                sample_validation._physrope_val_printed = True
                print(f"[PhysRoPE VALIDATION] (train.py:L694) sample_validation: "
                      f"use_physical_rope={getattr(args, 'use_physical_rope', False)} | "
                      f"has_time_ranges={text_segment_time_ranges is not None} | "
                      f"n_ranges={len(text_segment_time_ranges) if text_segment_time_ranges else 0} | "
                      f"data_type={data_type} | target_length={target_length} | "
                      f"prompt[:80]={str(prompt)[:80]}")

            # Shape validation for cond_latents before pipeline call
            if cond_latents is not None and mask_type == "range_condition":
                expected_lat_h = target_height // 16
                expected_lat_w = target_width // 16
                shape_ok = (cond_latents.shape[-2] == expected_lat_h and
                            cond_latents.shape[-1] == expected_lat_w and
                            ((native_t_concat and cond_latents.shape[2] > 0) or
                             ((not native_t_concat) and cond_latents.shape[2] == latent_target_length)))
                if not shape_ok:
                    print(f"[WARNING] Rank {dp_rank} cond_latents shape mismatch: "
                          f"got {tuple(cond_latents.shape)}, native_t_concat={native_t_concat}, "
                          f"expected target T={latent_target_length} "
                          f"H={expected_lat_h} W={expected_lat_w}. "
                          f"Falling back to t2v for case {indexs[sidx]}")
                    cond_latents = None
                    mask_type = "t2v"
                    multitask_mask, _ = get_multitask_mask(
                        latent_target_length, {"t2v": {"prob": 1.0}}, seed=args.global_seed)
                    cur_condition_frame_ranges = []
                    cur_condition_physical_dst_frame_ranges = []
                    cur_condition_src_frame_ranges = []
                    cur_condition_kinds = []
                    cur_condition_prompt_seg_ids = []
                    cur_condition_prompt_ranges = []
                    cur_native_reference_packed_ranges = []

            print(
                f"Rank {dp_rank}, sample_validation Data type: {data_type} \t mask_type: {mask_type} "
                f"\t target_length: {target_length}\t target_height: {target_height}\t target_width: {target_width} "
                f"\t native_t_concat: {native_t_concat and mask_type == 'range_condition'} "
                f"\t reference_latent_T: {None if cond_latents is None else cond_latents.shape[2]} "
                f"\t target_latent_T: {latent_target_length} "
                f"\t native_ranges: {cur_native_reference_packed_ranges} "
                f"\t temporal_pack_mode: {temporal_pack_mode} "
                f"\t dst_ranges: {cur_condition_frame_ranges} "
                f"\t src_ranges: {cur_condition_src_frame_ranges} "
                f"\t physical_dst_ranges: {cur_condition_physical_dst_frame_ranges} "
                f"\t sparse_fixed_target_clock: {temporal_pack_mode == REFERENCE_TIMELINE} "
                f"\t flow_shift: {infer_flow_shift} "
                f"\t infer_steps: {infer_steps} \t seed: {seeds} \t prompts: {prompts}"
            )
            prompt = _apply_prompt_fps_prefix(
                prompt,
                getattr(args, "prompt_fps_prefix", None),
                data_type=data_type,
            )
            samples = pipeline(prompt=prompt,
                               height=target_height,
                               width=target_width,
                               video_length=target_length,
                               num_inference_steps=infer_steps,
                               guidance_scale=args.cfg_scale,
                               negative_prompt=args.neg_prompt,
                               num_images_per_prompt=1,
                               generator=generator,
                               vae_ver=args.vae,
                               enable_tiling=args.vae_tiling,
                               data_type=data_type,
                               output_type = 'pt',
                               reorg_token='888' in args.model_type or '16168' in args.model_type,
                               multitask_mask_training_type=args.multitask_mask_training_type,
                               multitask_type=mask_type,
                               cond_latents=cond_latents,
                               last_frame_cond_latents=last_frame_cond_latents,
                               multitask_mask=multitask_mask,
                               semantic_images=semantic_images_pil,
                               reference_image=semantic_images_np,
                               vision_num_semantic_tokens=args.vision_num_semantic_tokens,
                               vision_states_dim=args.vision_states_dim,
                               bucket_hw_base_size=args.video_bucket_hw_base_size,
                               bucket_hw_bucket_stride=args.video_bucket_hw_bucket_stride,
                               text_segment_time_ranges=text_segment_time_ranges,
                               condition_frame_ranges=cur_condition_frame_ranges,
                               condition_physical_dst_frame_ranges=cur_condition_physical_dst_frame_ranges,
                               condition_src_frame_ranges=cur_condition_src_frame_ranges,
                               condition_kinds=cur_condition_kinds,
                               condition_seg_ids=cur_condition_prompt_seg_ids,
                               range_condition_temporal_pack_mode=temporal_pack_mode,
                               native_reference_packed_ranges=cur_native_reference_packed_ranges,
                               )[0]

            is_save_rank = (not get_sequence_parallel_state()) or int(nccl_info.rank_within_spgroup) == 0
            should_save_sample = is_save_rank and (
                (not use_fixed_prompt) or bool(save_sample_flags[sidx])
            )
            if not should_save_sample:
                del samples
                torch.cuda.empty_cache()
                continue

            # only save once per sequence-parallel group
            if dp_rank < (total_samples if use_fixed_prompt else 128):
                sample = samples[0].unsqueeze(0)
                sub_samples = []
                if base_images is not None:
                    gt_pixel_value = base_images[sidx][None, ...]
                    if base_images_need_denorm:
                        gt_pixel_value = (gt_pixel_value / 2 + 0.5).clamp(0, 1)
                    sample = sample.cpu()
                    gt_pixel_value = gt_pixel_value.cpu()
                    sub_samples.append(sample)
                    sub_samples.append(gt_pixel_value)
                    sample_videos.append(sample)  # b c t h w, [0, 1]
                    gt_videos.append(gt_pixel_value)
                    sample_num = len(sub_samples)
                    sub_samples = torch.concat(sub_samples)
                else:
                    sub_samples.append(sample)
                    sample_num = len(sub_samples)
                    sub_samples = torch.concat(sub_samples)

                if use_fixed_prompt:
                    # Save to folder named after csv file, video named by index
                    folder_name, file_name = csv_file_names[sidx]
                    save_dir = os.path.join(output_dir, "samples", f"{global_step:07d}", folder_name, file_name)
                    os.makedirs(save_dir, exist_ok=True)
                    if mask_type == "range_condition" and cur_condition_frame_ranges:
                        cond_str = "_".join(f"cond{s}-{e}" for s, e in cur_condition_frame_ranges)
                        save_path = os.path.join(save_dir, f"{indexs[sidx]}_{cond_str}.mp4")
                    else:
                        save_path = os.path.join(save_dir, f"{indexs[sidx]}.mp4")
                else:
                    if args.video_caption_i2v_instruction_enable:
                        if mask_type == "i2v" and valid_i2v_instruction_i == "true" and use_i2v_instruction:
                            save_base_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}[inst]_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                        elif mask_type == "i2v" and valid_i2v_instruction_i == "false" and use_i2v_instruction:
                            save_base_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}[orig]_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                        elif mask_type == "i2v" and valid_i2v_instruction_i == "true" and not use_i2v_instruction:
                            save_base_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}[orig]_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                        elif mask_type == "i2v" and valid_i2v_instruction_i == "false" and not use_i2v_instruction:
                            save_base_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}[orig]_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                        elif mask_type == "i2v" and valid_i2v_instruction_i == "unkown_prompts":
                            save_base_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}[unkown]_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                        elif mask_type == "t2v":
                            save_base_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                        elif mask_type == "fl2v":
                            save_base_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                        else:
                            raise ValueError(f"mask_type {mask_type} not supported")
                        save_prompt_name = prompt[:100].replace('/', ' ')
                        save_path = f"{save_base_path}-{save_prompt_name}.mp4"
                    else:
                        save_base_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                        save_prompt_name = prompt[:100].replace('/', ' ')
                        save_path = f"{save_base_path}-{save_prompt_name}.mp4"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)

                if target_length > 1:
                    try:
                        save_videos_grid(sub_samples, save_path, n_rows=sample_num, fps=args.video_fps)
                    except OSError:
                        if use_fixed_prompt:
                            raise
                        save_prompt_name = prompt[:10].replace('/', ' ')
                        save_path = f"{save_base_path}-{save_prompt_name}.mp4"
                        save_videos_grid(sub_samples, save_path, n_rows=sample_num, fps=args.video_fps)
                else:
                    sample = rearrange(sub_samples, "b c f h w -> c h (b f w)")
                    try:
                        torchvision.utils.save_image(sample, save_path.replace('mp4', 'png'))
                    except OSError:
                        if use_fixed_prompt:
                            raise
                        save_path = f"{output_dir}/samples/{global_step:07d}/rank{dp_rank}_{mask_type}_{target_length}x{target_height}x{target_width}-{videoid[sidx]}-{prompt[:10].replace('/', ' ')}.mp4"
                        torchvision.utils.save_image(sample, save_path.replace('mp4', 'png'))
                with open(save_path.replace('.mp4', '.txt'), 'w') as f:
                    f.write(f"{mask_type}: {prompt}")
                print(f'Rank {dp_rank} saved sample to {save_path}')

                # ─── Fixed-prompt: GT comparison + Reference saving ───
                if use_fixed_prompt and case and target_length > 1:
                    _gen_video = sample.cpu()  # [1, C, T, H, W] in [0,1]
                    del sample  # free GPU memory before GT decode
                    torch.cuda.empty_cache()

                    # 1) Load GT latent → decode → save gen|gt side-by-side
                    gt_cos_path = case.get('gt_latent_cos_path', '')
                    if gt_cos_path:
                        cache_folder = os.environ.get("REFRAIN_LATENT_ROOT", "train/latents")
                        gt_full_path = os.path.join(cache_folder, gt_cos_path)
                        if os.path.exists(gt_full_path):
                            try:
                                _gt_lat = torch.from_numpy(np.load(gt_full_path)).to(device=device, dtype=torch.float32)
                                if _gt_lat.ndim == 5:
                                    _gt_lat = _gt_lat[0:1]  # [1, C, T, H, W]
                                elif _gt_lat.ndim == 4:
                                    _gt_lat = _gt_lat.unsqueeze(0)
                                # Truncate/pad GT latent to match gen video LATENT length
                                # (NOT pixel T! latent T → pixel T via VAE temporal stride)
                                if _gt_lat.shape[2] > latent_target_length:
                                    _gt_lat = _gt_lat[:, :, :latent_target_length]
                                elif _gt_lat.shape[2] < latent_target_length:
                                    _pad = _gt_lat[:, :, -1:].repeat(1, 1, latent_target_length - _gt_lat.shape[2], 1, 1)
                                    _gt_lat = torch.cat([_gt_lat, _pad], dim=2)
                                # Decode GT (stored latent is unscaled — raw mode output)
                                # No need to divide by scaling_factor; VAE decode expects raw latent
                                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                    if args.vae_tiling:
                                        vae.enable_tiling()
                                    _gt_vid = vae.decode(_gt_lat.to(device), return_dict=False)[0]
                                    if args.vae_tiling:
                                        vae.disable_tiling()
                                _gt_vid = (_gt_vid / 2 + 0.5).clamp(0, 1).cpu().float()  # [1, C, T, H, W]
                                # Spatial resize GT to match gen if needed
                                if _gt_vid.shape[-2:] != _gen_video.shape[-2:]:
                                    B, C, T, H, W = _gt_vid.shape
                                    _gt_vid = _gt_vid.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
                                    _gt_vid = torch.nn.functional.interpolate(
                                        _gt_vid, size=_gen_video.shape[-2:], mode='bilinear', align_corners=False
                                    )
                                    _gt_vid = _gt_vid.reshape(B, T, C, *_gen_video.shape[-2:]).permute(0, 2, 1, 3, 4)
                                # Concat: gen (left) | gt (right) along width
                                _compare = torch.cat([_gen_video, _gt_vid], dim=-1)  # [1, C, T, H, W*2]
                                compare_path = save_path.replace('.mp4', '_compare.mp4')
                                save_videos_grid(_compare, compare_path, n_rows=1, fps=args.video_fps)
                                print(f'Rank {dp_rank} saved compare to {compare_path}')
                                del _gt_lat, _gt_vid, _compare
                                torch.cuda.empty_cache()
                            except Exception as e:
                                print(f'[WARNING] Rank {dp_rank} GT decode failed for case {indexs[sidx]}: {e}')
                                torch.cuda.empty_cache()

                    # 2) Save reference: condition segment videos + anchor images
                    _conditions = case.get('conditions', None)
                    if _conditions:
                        # Reference segments (V2V, TIV2V)
                        for seg in _conditions.get('segments', []):
                            seg_lp = seg.get('latent_path', '')
                            seg_video_path = seg.get('video_path', '') or seg.get('path', '')
                            if seg_lp and os.path.exists(seg_lp):
                                try:
                                    _seg_lat = torch.from_numpy(np.load(seg_lp)).to(device=device, dtype=torch.float32)
                                    if _seg_lat.ndim == 5:
                                        _seg_lat = _seg_lat[0:1]
                                    elif _seg_lat.ndim == 4:
                                        _seg_lat = _seg_lat.unsqueeze(0)
                                    # Stored latent is unscaled (raw mode output), decode directly
                                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                        if args.vae_tiling:
                                            vae.enable_tiling()
                                        _seg_vid = vae.decode(_seg_lat.to(device), return_dict=False)[0]
                                        if args.vae_tiling:
                                            vae.disable_tiling()
                                    _seg_vid = (_seg_vid / 2 + 0.5).clamp(0, 1).cpu().float()
                                    ds, de = seg['dst_range']
                                    ref_path = save_path.replace('.mp4', f'_ref_seg{seg.get("seg_id", 0)}_{ds}-{de}.mp4')
                                    save_videos_grid(_seg_vid, ref_path, n_rows=1, fps=args.video_fps)
                                    print(f'Rank {dp_rank} saved ref segment to {ref_path}')
                                    del _seg_lat, _seg_vid
                                    torch.cuda.empty_cache()
                                except Exception as e:
                                    print(f'[WARNING] Rank {dp_rank} ref segment decode failed: {e}')
                                    torch.cuda.empty_cache()
                            elif seg_video_path and os.path.exists(seg_video_path):
                                try:
                                    ds, de = seg['dst_range']
                                    ref_path = save_path.replace('.mp4', f'_ref_seg{seg.get("seg_id", 0)}_{ds}-{de}.mp4')
                                    shutil.copyfile(seg_video_path, ref_path)
                                    print(f'Rank {dp_rank} saved ref segment to {ref_path}')
                                except Exception as e:
                                    print(f'[WARNING] Rank {dp_rank} ref segment copy failed: {e}')

                        # Reference anchor frames (TI2V, TIV2V)
                        for anchor in _conditions.get('anchors', []):
                            try:
                                t_anc = int(anchor['frame_idx'])
                            except (KeyError, TypeError, ValueError) as exc:
                                print(f"[WARNING] Rank {dp_rank} invalid anchor frame for ref save "
                                      f"case {indexs[sidx]} anchor={anchor} exc={repr(exc)}")
                                continue
                            anchor_path = anchor.get('path', '') or anchor.get('image_path', '')
                            if gt_cos_path:
                                gt_full_path_anc = os.path.join(os.environ.get("REFRAIN_LATENT_ROOT", "train/latents"), gt_cos_path)
                                if os.path.exists(gt_full_path_anc):
                                    try:
                                        _anc_lat = torch.from_numpy(np.load(gt_full_path_anc)).to(device=device, dtype=torch.float32)
                                        if _anc_lat.ndim == 5:
                                            _anc_lat = _anc_lat[0:1]
                                        elif _anc_lat.ndim == 4:
                                            _anc_lat = _anc_lat.unsqueeze(0)
                                        if t_anc < _anc_lat.shape[2]:
                                            _anc_frame = _anc_lat[:, :, t_anc:t_anc+1]  # [1, C, 1, H, W]
                                            # Stored latent is unscaled, decode directly
                                            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                                _anc_img = vae.decode(_anc_frame.to(device), return_dict=False)[0]
                                            _anc_img = (_anc_img / 2 + 0.5).clamp(0, 1).cpu().float()
                                            # Save as image: [C, H, W]
                                            anc_path = save_path.replace('.mp4', f'_ref_anchor_t{t_anc}.png')
                                            torchvision.utils.save_image(_anc_img[0, :, 0], anc_path)
                                            print(f'Rank {dp_rank} saved anchor frame to {anc_path}')
                                            del _anc_lat, _anc_frame, _anc_img
                                            torch.cuda.empty_cache()
                                    except Exception as e:
                                        print(f'[WARNING] Rank {dp_rank} anchor decode failed: {e}')
                                        torch.cuda.empty_cache()
                            elif anchor_path and os.path.exists(anchor_path):
                                try:
                                    anc_path = save_path.replace('.mp4', f'_ref_anchor_t{t_anc}.png')
                                    Image.open(anchor_path).convert('RGB').save(anc_path)
                                    print(f'Rank {dp_rank} saved anchor frame to {anc_path}')
                                except Exception as e:
                                    print(f'[WARNING] Rank {dp_rank} anchor image copy failed: {e}')


    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    model.train()

def print_training_configuration(args, logger, model, world_size, local_rank, rank, 
                                dp_degree, dp_rank, image_total_batch_size, video_total_batch_size, image_num, video_num,
                                image_loader=None, video_loader=None, 
                                image_dataset=None, video_dataset=None, scalar_states=None):
    """
    Print comprehensive training configuration information in a well-organized format.
    
    Args:
        args: Training arguments
        logger: Logger instance
        model: The model instance
        world_size: Total number of processes
        local_rank: Local rank within node
        rank: Global rank
        dp_degree: Data parallel degree
        dp_rank: Data parallel rank
        total_batch_size: Total batch size across all GPUs
        image_num: Number of image samples
        video_num: Number of video samples
        image_loader: Image data loader (optional)
        video_loader: Video data loader (optional)
        image_dataset: Image dataset (optional)
        video_dataset: Video dataset (optional)
    """
    params_count = model.params_count()
    
    logger.info("****************************** Running training ******************************")
    
    # ===============================================================================
    # System & Hardware Configuration
    # ===============================================================================
    logger.info("=" * 80)
    logger.info("SYSTEM & HARDWARE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Number of GPUs                                   : {world_size}")
    logger.info(f"Local rank                                       : {local_rank}")
    logger.info(f"Global rank                                      : {rank}")
    logger.info(f"Device                                           : cuda:{local_rank}")
    logger.info(f"Master weight dtype                              : {model.parameters().__next__().dtype}")
    
    # ===============================================================================
    # Model Parameters & Architecture
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("MODEL PARAMETERS & ARCHITECTURE")
    logger.info("-" * 80)
    for k, v in params_count.items():
        logger.info(f"Number of {k:<25}                     : {v:,}")
    total_params_b = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    logger.info(f"Total trainable parameters per FSDP shard       : {total_params_b:.3f}B")
    logger.info(f"Model type                                       : {args.model_type}")
    logger.info(f"DIT model path                                   : {args.dit_model_name_or_path}")
    logger.info(f"Pretrained model path                            : {args.pretrained_model_name_or_path}")
    logger.info(f"Gradient checkpointing                           : {args.gradient_checkpointing}")
    logger.info(f"Text encoder type                                 : {args.text_encoder}")
    logger.info(f"Text length                                      : {args.text_len}")
    if hasattr(args, 'text_encoder_2'):
        logger.info(f"Text encoder 2 type                              : {args.text_encoder_2}")
        logger.info(f"Text length 2                                    : {args.text_len_2}")
    if hasattr(args, 'selective_checkpointing'):
        logger.info(f"Selective checkpointing                          : {args.selective_checkpointing}")
    if hasattr(args, 'attn_mode'):
        logger.info(f"Attention mode                                   : {args.attn_mode}")
        if hasattr(args, 'win_type'):
            logger.info(f"Window type                                      : {args.win_type}")
        if hasattr(args, 'win_size'):
            logger.info(f"Window size                                      : {args.win_size}")
    if hasattr(args, 'use_dynamic_ring_attention'):
        logger.info(f"Dynamic ring attention                           : {args.use_dynamic_ring_attention}")
    

    if hasattr(args, 'use_cond_3d_rope'):
        logger.info(f"Use cond 3d rope                                 : {args.use_cond_3d_rope}")


    # ===============================================================================
    # Distributed Training Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("DISTRIBUTED TRAINING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"World size                                       : {world_size}")
    logger.info(f"DP degree                                        : {dp_degree}")
    logger.info(f"DP rank                                          : {dp_rank}")
    logger.info(f"SP size                                          : {args.sp_size}")
    logger.info(f"FSDP sharding strategy                           : {args.fsdp_sharding_strategy}")
    logger.info(f"CPU offload                                      : {args.use_cpu_offload}")
    
    # ===============================================================================
    # Batch Size & Data Flow Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("BATCH SIZE & DATA FLOW CONFIGURATION")
    logger.info("-" * 80)
    per_gpu_batch = args.image_micro_batch_size[0] if isinstance(args.image_micro_batch_size, list) else args.image_micro_batch_size
    dp_batch = per_gpu_batch * dp_degree
    effective_batch_image = image_total_batch_size * args.gradient_accumulation_steps
    effective_batch_video = video_total_batch_size * args.gradient_accumulation_steps
    
    logger.info(f"Image micro batch size                                 : {args.image_micro_batch_size}")
    logger.info(f"Per GPU batch size                               : {per_gpu_batch}")
    logger.info(f"DP degree batch size                             : {dp_batch}")
    logger.info(f"Total batch size (all GPUs)                      : {image_total_batch_size} (image) + {video_total_batch_size} (video)")
    logger.info(f"Gradient accumulation steps                      : {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size (with grad accum)           : {effective_batch_image} (image) + {effective_batch_video} (video)")
    logger.info(f"Samples per optimizer update                     : {effective_batch_image} (image) + {effective_batch_video} (video)")
    
    # ===============================================================================
    # Dataset & DataLoader Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("DATASET & DATALOADER CONFIGURATION")
    logger.info("-" * 80)
    total_samples = image_num + video_num
    logger.info(f"Data type                                        : {args.data_type}")
    logger.info(f"Total samples (image + video)                    : {total_samples:,}")
    logger.info(f"Image examples                                   : {image_num:,}")
    logger.info(f"Video examples                                   : {video_num:,}")
    
    if hasattr(args, 'image_data_path') and 'image' in args.data_type:
        logger.info(f"Image data path                                  : {args.image_data_path}")
    if hasattr(args, 'video_data_path') and 'video' in args.data_type:
        logger.info(f"Video data path                                  : {args.video_data_path}")
    if hasattr(args, 'image_size'):
        logger.info(f"Image size                                       : {args.image_size}")
    if hasattr(args, 'video_size'):
        logger.info(f"Video size                                       : {args.video_size}")
    
    logger.info(f"Video sampling probability                       : {args.video_sampling_prob}")
    logger.info(f"Video multireso                                  : {args.video_multireso}")
    if hasattr(args, 'video_bucket_hw_bucket_stride'):
        logger.info(f"Video bucket HW stride                           : {args.video_bucket_hw_bucket_stride}")
    if hasattr(args, 'vae_sample_stride'):
        logger.info(f"VAE sample stride                                : {args.vae_sample_stride}")
    
    # DataLoader details
    if image_loader is not None:
        logger.info(f"Image dataloader batch size                      : {image_loader.batch_size}")
        logger.info(f"Image dataloader drop last                       : {image_loader.drop_last}")
        if image_dataset is not None:
            img_len = len(image_dataset) if hasattr(image_dataset, '__len__') else 'Unknown'
            logger.info(f"Image dataset length                             : {img_len}")
            if hasattr(image_dataset, 'total_length'):
                logger.info(f"Image dataset total length                       : {image_dataset.total_length}")
    
    if video_loader is not None:
        logger.info(f"Video dataloader batch size                      : {video_loader.batch_size}")
        logger.info(f"Video dataloader drop last                       : {video_loader.drop_last}")
        if video_dataset is not None:
            vid_len = len(video_dataset) if hasattr(video_dataset, '__len__') else 'Unknown'
            logger.info(f"Video dataset length                             : {vid_len}")
            if hasattr(video_dataset, 'total_length'):
                logger.info(f"Video dataset total length                       : {video_dataset.total_length}")
    
    if hasattr(args, 'use_ll_cap_dataset'):
        logger.info(f"Use LL-Cap dataset                               : {args.use_ll_cap_dataset}")
        logger.info(f"LL-Cap sample ratio                              : {args.ll_cap_sample_ratio}")
        logger.info(f"Text length for LL-Cap dataset                   : {args.text_len_ll_cap}")
        logger.info(f"LL-Cap caption sample ratio                       : {args.ll_cap_caption_sample_ratio}")
    if hasattr(args, 'use_video_ll_cap_dataset'):
        logger.info(f"Use Video LL-Cap dataset                         : {args.use_video_ll_cap_dataset}")
        logger.info(f"Video LL-Cap sample ratio                        : {args.video_ll_cap_sample_ratio}")
        logger.info(f"Video text length for LL-Cap dataset             : {args.video_text_len_ll_cap}")
        logger.info(f"Video LL-Cap caption sample ratio                : {args.video_ll_cap_caption_sample_ratio}")
    else:
        logger.info(f"Text length                                      : {args.text_len}")

    # logger.info(f"Consumed image samples per DP                    : {args.consumed_image_samples_per_dp}")
    # logger.info(f"Consumed video samples per DP                    : {args.consumed_video_samples_per_dp}")
    # logger.info(f"Consumed LL-Cap image samples per DP              : {args.consumed_ll_cap_image_samples_per_dp}")
    # logger.info(f"Consumed image samples total                      : {args.consumed_image_samples_total}")
    # logger.info(f"Consumed video samples total                      : {args.consumed_video_samples_total}")
    # logger.info(f"Consumed LL-Cap image samples total               : {args.consumed_ll_cap_image_samples_total}")
    
    # ===============================================================================
    # Training Schedule & Steps Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("TRAINING SCHEDULE & STEPS CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Number of epochs                                 : {args.num_train_epochs}")
    logger.info(f"Max training iterations                           : {args.train_iters}")
    
    total_update_steps = args.train_iters // args.gradient_accumulation_steps
    logger.info(f"Total forward/backward steps                     : {args.train_iters}")
    logger.info(f"Total optimizer update steps                     : {total_update_steps}")
    logger.info(f"Forward steps per optimizer update               : {args.gradient_accumulation_steps}")
    
    # ===============================================================================
    # Optimization Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("OPTIMIZATION CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Optimizer                                        : {args.optimizer}")
    logger.info(f"Learning rate                                    : {args.lr}")
    logger.info(f"Weight decay                                     : {args.weight_decay}")
    logger.info(f"Max gradient norm                                : {args.max_grad_norm}")
    logger.info(f"LR scheduler                                     : {args.lr_scheduler}")
    logger.info(f"LR warmup steps                                  : {args.lr_warmup_steps}")
    logger.info(f"LR num cycles                                    : {args.lr_num_cycles}")
    logger.info(f"LR power                                         : {args.lr_power}")
    
    # ===============================================================================
    # Logging & Checkpointing Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("LOGGING & CHECKPOINTING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Output directory                                 : {args.output_dir}")
    logger.info(f"Checkpointing steps                              : {args.checkpointing_steps}")
    logger.info(f"Sample interval                                  : {args.sample_interval}")
    logger.info(f"Log interval                                     : {args.log_interval}")
    logger.info(f"Resume training                                  : {args.resume}")
    logger.info(f"Profiler enabled                                 : {args.is_profiler}")
    logger.info(f"Dry run enabled                                  : {getattr(args, 'dry_run', False)}")
    
    # ===============================================================================
    # Inference & Validation Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("INFERENCE & VALIDATION CONFIGURATION")
    logger.info("-" * 80)
    if hasattr(args, 'infer_steps'):
        logger.info(f"Inference steps                                  : {args.infer_steps}")
    if hasattr(args, 'cfg_scale'):
        logger.info(f"CFG scale                                        : {args.cfg_scale}")
    if hasattr(args, 'infer_flow_shift'):
        logger.info(f"Inference flow shift                             : {args.infer_flow_shift}")
    if hasattr(args, 'neg_prompt'):
        logger.info(f"Negative prompt                                  : {args.neg_prompt}")
    if hasattr(args, 'vae_tiling'):
        logger.info(f"VAE tiling                                       : {args.vae_tiling}")
    if hasattr(args, 'vae'):
        logger.info(f"VAE version                                      : {args.vae}")
    
    # ===============================================================================
    # Seeds & Reproducibility Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("SEEDS & REPRODUCIBILITY CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Global seed                                      : {args.global_seed}")
    logger.info(f"Local seed                                       : {args.local_seed}")
    
    # ===============================================================================
    # Additional Configuration
    # ===============================================================================
    # 分拆逻辑：分别判断每个配置项，单独打印
    if hasattr(args, 'glyph_byT5_v2'):
        logger.info("-" * 80)
        logger.info("ADDITIONAL CONFIGURATION: GLYPH BYT5 V2")
        logger.info("-" * 80)
        logger.info(f"Glyph byT5 v2                                    : {args.glyph_byT5_v2}")
        logger.info(f"ByT5 max length                                   : {args.byt5_max_length}")

    if hasattr(args, 'moe_loss_weight') and args.moe_loss_weight > 0:
        logger.info("-" * 80)
        logger.info("ADDITIONAL CONFIGURATION: MOE LOSS")
        logger.info("-" * 80)
        logger.info(f"MoE loss weight                                  : {args.moe_loss_weight}")
    
    if hasattr(args, 'use_repa'):
        logger.info("-" * 80)
        logger.info("ADDITIONAL CONFIGURATION: REPA")
        logger.info("-" * 80)
        logger.info(f"Use REPA                                        : {args.use_repa}")
        logger.info(f"Repa loss weight                                 : {args.repa_loss_weight}")
        logger.info(f"Repa encoder                                      : {args.repa_encoder}")
        logger.info(f"Repa encoder precision                           : {args.repa_encoder_precision}")

    if hasattr(args, 'convert_state_dict_in_out'):
        logger.info("-" * 80)
        logger.info("ADDITIONAL CONFIGURATION: CONVERT STATE DICT IN/OUT")
        logger.info("-" * 80)
        logger.info(f"Convert state dict in/out                        : {args.convert_state_dict_in_out}")
    
    # ===============================================================================
    # Multitask I2V Configuration
    # ===============================================================================
    if (hasattr(args, 'multitask_mask_training_type') and args.multitask_mask_training_type is not None) or \
       (hasattr(args, 'multitask_semantic_mode') and args.multitask_semantic_mode is not None) or \
       (hasattr(args, 'convert_state_dict_img_in_for_i2v') and args.convert_state_dict_img_in_for_i2v):
        logger.info("-" * 80)
        logger.info("MULTITASK I2V CONFIGURATION")
        logger.info("-" * 80)
        if hasattr(args, 'multitask_mask_training_type'):
            logger.info(f"Multitask mask training type                     : {args.multitask_mask_training_type}")
        if hasattr(args, 'multitask_mask_training_args'):
            logger.info(f"Multitask mask training args                     : {args.multitask_mask_training_args}")
        if hasattr(args, 'multitask_semantic_mode'):
            logger.info(f"Multitask semantic mode                          : {args.multitask_semantic_mode}")
        if hasattr(args, 'vision_projection'):
            logger.info(f"Vision projection                                : {args.vision_projection}")
        if hasattr(args, 'vision_states_dim'):
            logger.info(f"Vision states dimension                          : {args.vision_states_dim}")
        if hasattr(args, 'use_multitask_co_drop_semantic_and_text'):
            logger.info(f"Co-drop semantic and text                        : {args.use_multitask_co_drop_semantic_and_text}")
        if hasattr(args, 'multitask_semantic_drop_independent_p'):
            logger.info(f"Semantic drop independent probability            : {args.multitask_semantic_drop_independent_p}")

    # ===============================================================================
    # Performance Estimates
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("PERFORMANCE ESTIMATES")
    logger.info("-" * 80)
    logger.info(f"Target throughput per update                     : {effective_batch_image} (image) + {effective_batch_video} (video)")
    
    # ===============================================================================
    # Mask Type Statistics
    # ===============================================================================
    if scalar_states and scalar_states.consumed_samples_by_mask_type_total:
        logger.info("-" * 80)
        logger.info("MASK TYPE STATISTICS (RESUMED TRAINING)")
        logger.info("-" * 80)
        mask_type_stats = dict(scalar_states.consumed_samples_by_mask_type_total)
        total_mask_samples = sum(mask_type_stats.values())
        for mt, count in mask_type_stats.items():
            percentage = (count / total_mask_samples * 100) if total_mask_samples > 0 else 0
            logger.info(f"Mask type '{mt}' samples                            : {count:,} ({percentage:.1f}%)")
        logger.info(f"Total mask type samples                          : {total_mask_samples:,}")
        
        # Data type statistics for t2v
        if scalar_states.consumed_samples_by_data_type_total:
            logger.info("-" * 40)
            logger.info("DATA TYPE STATISTICS FOR T2V (RESUMED TRAINING)")
            logger.info("-" * 40)
            data_type_stats = dict(scalar_states.consumed_samples_by_data_type_total)
            total_data_type_samples = sum(data_type_stats.values())
            for dt, count in data_type_stats.items():
                percentage = (count / total_data_type_samples * 100) if total_data_type_samples > 0 else 0
                logger.info(f"T2V Data type '{dt}' samples                        : {count:,} ({percentage:.1f}%)")
            logger.info(f"Total T2V data type samples                      : {total_data_type_samples:,}")
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)



def get_kwargs(batch, args):
    media, latents, text_ids,text_mask, kwargs = batch
    return kwargs

def dump_spike_gradients_fsdp(model_engine, output_dir: str, *, step: int, rank: int, tag: str) -> None:
    """Dump local (sharded) gradients for FSDP debugging.

    Note: In FSDP, gradients are sharded per-rank. This function saves the local
    shard for each parameter; reconstructing full gradients requires all ranks.
    """
    if output_dir is None:
        return

    grad_dir = os.path.join(output_dir, "grads", str(tag))
    os.makedirs(grad_dir, exist_ok=True)

    grads = {}
    grad_stats = {}

    for name, param in model_engine.named_parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if not torch.is_tensor(grad):
            continue

        if isinstance(grad, DTensor):
            local_grad = grad.to_local()
            shard_info = "dtensor_local"
        else:
            local_grad = grad
            shard_info = "local"

        g = local_grad.detach()
        g_fp32 = g.float()
        grads[name] = g_fp32.cpu().contiguous()
        grad_stats[name] = {
            "shape": tuple(g.shape),
            "dtype": str(g.dtype),
            "device": str(g.device),
            "l2_norm": float(g_fp32.norm().item()) if g.numel() > 0 else 0.0,
            "max_abs": float(g_fp32.abs().max().item()) if g.numel() > 0 else 0.0,
            "shard": shard_info,
        }

    payload = {
        "step": int(step),
        "rank": int(rank),
        "tag": str(tag),
        "fsdp_sharded": True,
        "grad_stats": grad_stats,
        "grads_fp32_cpu": grads,
    }

    torch.save(payload, os.path.join(grad_dir, f"grads_rank{rank}.pt"))

def main(args):

    # ============================= Setup ==============================
    temporal_pack_mode = normalize_temporal_pack_mode(
        getattr(args, "range_condition_temporal_pack_mode", "legacy_trilinear")
    )
    if temporal_pack_mode == REFERENCE_TIMELINE:
        if as_bool(getattr(args, "range_condition_native_t_concat", False)):
            raise ValueError("reference_timeline requires range_condition_native_t_concat=false")
        if args.multitask_mask_training_type != "concat":
            raise ValueError("reference_timeline requires multitask_mask_training_type=concat")
        if args.range_condition_mode != "soft_reference":
            raise ValueError("reference_timeline requires range_condition_mode=soft_reference")
        if float(args.unified_physrope_temporal_resample_prob) != 0.0:
            raise ValueError(
                "reference_timeline forbids unified_physrope_temporal_resample_prob != 0"
            )
        if float(args.unified_physrope_cond_loss_weight) != 1.0:
            raise ValueError(
                "reference_timeline soft references require unified_physrope_cond_loss_weight=1"
            )
        if getattr(args, "physical_rope_cut", False):
            raise ValueError("reference_timeline requires physical_rope_cut=false")
        if str(getattr(args, "physical_rope_offset_mode", "fixed")) != "fixed":
            raise ValueError("reference_timeline cumulative PhysRoPE requires offset_mode=fixed")

    if as_bool(getattr(args, "range_condition_native_t_concat", False)):
        if "888" in args.model_type or "16168" in args.model_type:
            raise ValueError(
                "range_condition_native_t_concat requires the existing 97-channel "
                "48+48+1 input projection; reorg-token model types are unsupported"
            )
        if args.multitask_mask_training_type != "concat":
            raise ValueError(
                "range_condition_native_t_concat requires multitask_mask_training_type=concat"
            )
        if args.range_condition_mode != "soft_reference":
            raise ValueError(
                "range_condition_native_t_concat requires range_condition_mode=soft_reference"
            )
        if float(args.unified_physrope_temporal_resample_prob) != 0.0:
            raise ValueError(
                "range_condition_native_t_concat requires unified_physrope_temporal_resample_prob=0"
            )
        if float(args.unified_physrope_cond_loss_weight) != 1.0:
            raise ValueError(
                "range_condition_native_t_concat predicts every target frame and requires "
                "unified_physrope_cond_loss_weight=1.0"
            )
        if any(int(batch_size) != 1 for batch_size in (args.video_micro_batch_size or [])):
            raise ValueError(
                "range_condition_native_t_concat requires every video micro-batch size to be 1"
            )
        if getattr(args, "use_repa", False):
            raise ValueError(
                "range_condition_native_t_concat is not compatible with REPA feature loss"
            )
        if getattr(args, "physical_rope_cut", False):
            raise ValueError(
                "range_condition_native_t_concat requires physical_rope_cut=false"
            )
        if (
            getattr(args, "physical_rope_speed_aware", False)
            and str(getattr(args, "physical_rope_speed_mode", "uniform")) != "uniform"
        ):
            raise ValueError(
                "range_condition_native_t_concat does not support non-uniform physical RoPE speed warping"
            )

    # Setup distributed training environment
    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.fsdp_sharding_strategy == 'full':
        dp_replicate = 1
    else:
        assert args.fsdp_sharding_strategy == 'hybrid'
        dp_replicate = max(1, world_size // 8)
    
    if world_size <= 8 and args.fsdp_sharding_strategy == 'hybrid':
        args.fsdp_sharding_strategy = "full"
        print("node=1 fall back hybrid to full-shard")

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=args.distributed_timeout_sec)
    )
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    from hymm.parallelism.parallel_states import init_parallel_state

    parallel_dims = init_parallel_state(
        dp_replicate=dp_replicate,
        dp_shard=-1,
        sp=args.sp_size,
        tp=1,
        pp=args.pp_size,
        ep=args.ep_size,
        world_size=world_size,
    )
    if parallel_dims.dp_enabled:
        dp_mesh = parallel_dims.dp_mesh
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    initialize_sequence_parallel_state_v2(parallel_dims, parallel_dims.device_mesh_for_pp_dp_sp_tp, args.use_dynamic_ring_attention)

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # sync output_dir over all ranks
    if args.output_dir is not None:
        obj_list = [args.output_dir]
        dist.broadcast_object_list(obj_list, src=0)
        args.output_dir = obj_list[0]

    # Save full config once on rank 0
    if rank <= 0 and args.output_dir is not None:
        try:
            save_args_to_json(args, args.output_dir, filename="config.json")
        except Exception as e:
            print(f"[WARNING] Failed to save args to JSON: {type(e).__name__}: {e}")

    # setup logger
    logger, _ = setup_logger(args.output_dir)
    if rank <= 0:
        tb_dir = os.path.join(args.output_dir, "tb")
        tb_writer = SummaryWriter(log_dir=tb_dir)
    else:
        tb_writer = None
    logger.info(f"--> args {args}")
    logger.info(f"dp_degree, dp_rank: {dp_degree, dp_rank}")
    


    # Validate video_csv files before starting training
    validate_video_csv_files(args, logger)

    # profiler
    if args.is_profiler:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sort_by_keyword = "self_" + str(device) + "_time_total"

    # =========================== Build main model ===========================
    # Create model:
    logger.info(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    factor_kwargs = {'device': device, 
                     'dtype': torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16}

    model = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        factor_kwargs,
        args.convert_state_dict_in_out,
        args.convert_state_dict_img_in_for_i2v,
        args.convert_state_dict_16168_to_16164,
        args,
        logger
    )
    reorg_token = '888' in args.model_type or '16168' in args.model_type

    # Set model as trainable.
    model.train()
    # t within the same seq parallel group should be the same. Noise should be different.
    
    args.local_seed = args.global_seed + rank//args.sp_size # sp内的seed应该相同
    set_seed(args.local_seed)
    random.seed(args.local_seed)
    print(f"dp_rank: {dp_rank}, rank: {rank},  local_seed: {args.local_seed}")

    # ============================= Build extra models =========================
    # load text encoder and vae
    vae, vae_path, s_ratio, t_ratio = load_vae(args.vae, args.vae_precision, logger=logger, device=device)
    args.vae_spatial_ratio = s_ratio
    args.vae_temporal_ratio = t_ratio
    logger.info(f"--> loading text encoder from {args.pretrained_model_name_or_path}")
    text_encoder, text_encoder_2 = load_text_encoder(args.pretrained_model_name_or_path, device, logger, args)
    denoiser = load_denoiser(args)

    byt5_kwarg = None
    if args.glyph_byT5_v2:
        byt5_kwarg = load_glyph_byT5_v2(args, device)

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

    if getattr(args, "use_repa", False):
        if "repa_encoder" not in extra_model:
            repa_encoder = RepaEncoder(
                repa_encoder_type=args.repa_encoder,
                repa_encoder_precision=args.repa_encoder_precision,
                logger=logger,
                device=device
            )
            extra_model["repa_encoder"] = repa_encoder
        repa_encoder_config = extra_model["repa_encoder"].model.config
        model.configure_repa_projector(
            repa_encoder_config=repa_encoder_config,
            projector_hidden_dim=args.repa_projector_hidden_dim,
            projector_out_dim=args.repa_projector_out_dim,
            repa_type=args.repa_encoder,
        )

    logger.info(
        f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M"
    )

    from hymm.parallelism.engines.refrain_parallel import RefrainEngine
    from hymm.parallelism.engines.parallel_engine import get_optimizer_config
    if args.optimizer == 'adamw':
        optimizer_config = get_optimizer_config(
            torch.optim.AdamW,
            optimizer_kwargs=dict(
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    def get_lr_scheduler_func(optimizer_container):
        init_steps = 0
        return get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_container,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.train_iters,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
            last_epoch=init_steps - 1,
        )

    model_engine = RefrainEngine(
        model,
        optimizer_config=optimizer_config,
        get_lr_scheduler_func=get_lr_scheduler_func,
        parallel_dims=parallel_dims,
        enable_autocast=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        enable_gradient_checkpointing=True,

        # pp 相关参数
        auto_benchmark=False,
        m_microbatch=1, # 需要所有rank的batchsize都能整除这个数
        # pipeline_parallel_schedule='1F1B', # 1F1B 需要 m_microbatch >= pp_size
    )
    model = model_engine
    # model_engine.load_full_sd_by_cvt_to_dcp('assets/local/resource_e32a1d22a0.safetensors')
    
    # ============================== Initialize Curriculum Learning ==============================
    curriculum = CurriculumScheduler(
        start_phase=getattr(args, 'curriculum_start_phase', 0),
        enable_curriculum=getattr(args, 'enable_curriculum', True),
        strict_mode=getattr(args, 'curriculum_strict_mode', False),
    )
    
    if rank == 0:
        logger.info(f"--> Curriculum Learning initialized")
        logger.info(curriculum.get_summary())



    # ============================== resume ==============================
    init_steps = 0
    skip_initial_fixed_prompt_validation = False
    skip_initial_fixed_prompt_validation_step = None
    last_sample_validation_update_step = None
    resume_fixed_prompt_step0_pending = False
    if args.resume and args.resume != "None":
        if not os.path.exists(args.resume):
            raise ValueError("Invalid resume path: %s" % args.resume)

        _, scalar_states = model_engine.load_checkpoint(args.resume, load_optimizer_states=False)
        scalar_states = scalar_states['client_state']
        
        # Also try to load curriculum checkpoint
        curriculum_ckpt_path = os.path.join(os.path.dirname(args.resume), "curriculum_final.json")
        if os.path.exists(curriculum_ckpt_path):
            try:
                curriculum.load_checkpoint(curriculum_ckpt_path)
                if rank == 0:
                    logger.info(f"Loaded curriculum checkpoint from {curriculum_ckpt_path}")
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Failed to load curriculum checkpoint: {e}. Starting fresh.")

        # Old implementation
        # (model, _scalar_states_list) = resume_training(
        #     model, 
        #     args.resume,
        #     args.convert_state_dict_16168_to_16164
        # )

        # Resume ScalarStates
        # scalar_states = ScalarStates.from_pretrained(
        #     _scalar_states_list,
        #     rank=rank,
        #     world_size=world_size,
        #     default_rank0_ss=False,
        # )
        print(f'--> resume training scalar states {scalar_states}')
        init_steps = scalar_states.train_steps # 续训学习率参数
        resume_train_steps = int(getattr(scalar_states, "train_steps", 0) or 0)
        resume_update_steps = int(getattr(scalar_states, "update_steps", 0) or 0)
        if resume_train_steps > 0 or resume_update_steps > 0:
            if getattr(args, "resume_fixed_prompt_step0", False):
                args.try_first_iter = True
                resume_fixed_prompt_step0_pending = True
                if rank == 0:
                    logger.info(
                        "Resume starts from non-zero step; run startup fixed prompt validation "
                        f"for current run (train_steps={resume_train_steps}, "
                        f"update_steps={resume_update_steps})."
                    )
            else:
                args.try_first_iter = False
                skip_initial_fixed_prompt_validation = True
                skip_initial_fixed_prompt_validation_step = resume_update_steps
                if rank == 0:
                    logger.info(
                        "Resume starts from non-zero step; skip initial fixed prompt validation "
                        f"(train_steps={resume_train_steps}, update_steps={resume_update_steps})."
                    )
    else:
        init_scalar_state = {
            'lr': args.lr,
        }
        scalar_states = ScalarStates(**init_scalar_state)


    # ============================== Load dataset ==============================
    if not args.use_video_ll_cap_dataset: args.video_ll_cap_sample_ratio = 0
    if not args.use_ll_cap_dataset: args.ll_cap_sample_ratio = 0

    if args.data_type == 'image':
        args.video_sampling_prob = 0
    elif args.data_type == 'video':
        args.video_sampling_prob = 1
            
    def get_sample_probs(args):
        video_sampling_prob = args.video_sampling_prob
        ll_cap_sample_ratio = args.ll_cap_sample_ratio
        video_ll_cap_sample_ratio = args.video_ll_cap_sample_ratio
        
        # 当video_sampling_prob=1时，只分配视频相关的数据类型
        if video_sampling_prob == 1.0:
            video_prob = 1.0 - video_ll_cap_sample_ratio
            video_ll_cap_prob = video_ll_cap_sample_ratio
            image_prob = 0.0
            ll_cap_image_prob = 0.0
        # 当video_sampling_prob=0时，只分配image相关的数据类型
        elif video_sampling_prob == 0.0:
            video_prob = 0.0
            video_ll_cap_prob = 0.0
            image_prob = 1.0 - ll_cap_sample_ratio
            ll_cap_image_prob = ll_cap_sample_ratio
        else:
            # 正常的分配逻辑
            if video_ll_cap_sample_ratio == 0:
                video_prob = video_sampling_prob
                video_ll_cap_prob = 0.0
            else:
                video_prob = video_sampling_prob * (1.0 - video_ll_cap_sample_ratio)
                video_ll_cap_prob = video_sampling_prob * video_ll_cap_sample_ratio
            
            if ll_cap_sample_ratio == 0:
                image_prob = 1.0 - video_sampling_prob
                ll_cap_image_prob = 0.0
            else:
                image_prob = (1.0 - video_sampling_prob) * (1.0 - ll_cap_sample_ratio)
                ll_cap_image_prob = 1.0 - video_prob - video_ll_cap_prob - image_prob
        
        sampling_probs = {
            'video': video_prob,
            'll_cap_video': video_ll_cap_prob,
            'image': image_prob,
            'll_cap_image': ll_cap_image_prob
        }
        return sampling_probs
    
    sampling_probs = get_sample_probs(args)
    
    from hymm.utils.sampler_utils import prepare_distributed_sampling, MultimodalTasksState
    import easydict
    
    args.sampling_probs = json.dumps(sampling_probs)
    mm_state = MultimodalTasksState.from_args(args)
    p_state = easydict.EasyDict({
        'dp_size': dp_degree,
        'dp_rank': dp_rank
    })
    mm_state = prepare_distributed_sampling(args, mm_state, p_state)
    print(f"[Rank {dp_rank}] This rank will use dataloader type: {mm_state.distributed_sampling_state.cur_key}")

    video_dataset = video_sampler = video_loader = None
    image_dataset = image_sampler = image_loader = None
    ll_cap_image_dataset = ll_cap_image_sampler = ll_cap_image_loader = None
    ll_cap_video_dataset = ll_cap_video_sampler = ll_cap_video_loader = None
    
    cur_key = mm_state.distributed_sampling_state.cur_key
    common_args = (args, text_encoder, text_encoder_2, dp_degree, dp_rank, logger, byt5_kwarg)
    dist_state = mm_state.distributed_sampling_state
    
    if cur_key == 'video':
        video_dataset, video_sampler, video_loader = get_video_dataloader(*common_args, local_seed=args.local_seed, dist_sampler_state=dist_state)
    elif cur_key == 'image':
        image_dataset, image_sampler, image_loader = get_image_dataloader(*common_args, is_ll_cap=False, dist_sampler_state=dist_state)
    elif cur_key == 'll_cap_image':
        ll_cap_image_dataset, ll_cap_image_sampler, ll_cap_image_loader = get_image_dataloader(*common_args, is_ll_cap=True, dist_sampler_state=dist_state)
    elif cur_key == 'll_cap_video':
        ll_cap_video_dataset, ll_cap_video_sampler, ll_cap_video_loader = get_video_dataloader(*common_args, is_ll_cap=True, local_seed=args.local_seed, dist_sampler_state=dist_state)
    else:
        raise ValueError(f"Invalid dataloader type: {cur_key}")


    loader = VideoImageBatchIterator(
        dataloaders={
            'video': video_loader,
            'image': image_loader,
            'll_cap_image': ll_cap_image_loader,
            'll_cap_video': ll_cap_video_loader
        },
        samplers = {
            'video': video_sampler,
            'image': image_sampler,
            'll_cap_image': ll_cap_image_sampler,
            'll_cap_video': ll_cap_video_sampler
        },
        scalar_states = scalar_states,
        video_sampling_prob=args.video_sampling_prob,
        ll_cap_sample_ratio=args.ll_cap_sample_ratio,
        video_ll_cap_sample_ratio=args.video_ll_cap_sample_ratio,
        video_multireso=args.video_multireso,
        rank_assign_mode=args.rank_assign_mode,
        dp_rank=dp_rank,
        logger=logger,
        args=args
    )

    # ========================== Prepare evaluator ==========================
    loss_evaluator, loss_save_template = None, None
    score_evaluator, score_save_template = None, None
    sample_evaluator, sample_save_template = None, None
    if args.validation_every > 0 and args.validation_metrics:
        logger.info("Enable validation during training.")
        vae_kwargs = {'s_ratio': s_ratio, 't_ratio': t_ratio}
        loss_metrics = set(args.validation_metrics) & LOSS_METRICS
        score_metrics = set(args.validation_metrics) & SCORE_METRICS

        eval_kwargs = dict(vae=vae,
                           text_encoder=text_encoder,
                           model=model,
                           text_encoder_2=text_encoder_2,
                           rank=rank,
                           world_size=world_size,
                           device=device,
                           logger=logger,
                           )

        if loss_metrics:
            # Calculate validation loss. Loss metrics doesn't need to sample and decode images.
            loss_evaluator = LossEvaluator(args, **eval_kwargs, parallel_dims=parallel_dims, dp_degree=dp_degree, dp_rank=dp_rank)
            if args.val_index_file is None:
                raise ValueError("Please specify the evaluation index file.")

        if score_metrics:
            # Calculate score metrics. Score metrics need to sample and decode images.
            score_evaluator = ScoreEvaluator(args, **eval_kwargs)
            # Here we must sort the score_metrics to assert all the processes have the same order, in order to
            # call all_gather_object to correctly gather results.
            score_evaluator.initialize_scores(sorted(list(score_metrics)))

        if args.val_save_path is not None:
            logger.warning(f"In training mode, validation loss will be saved to the `evaluation` directory. "
                               f"The provided `--val-save-path` will be ignored.")
        if args.save_path is not None:
            logger.warning(f"In training mode, samples will be saved to the `samples` directory. "
                               f"The provided `--save-path` will be ignored.")
            
    # ============================== Print Key Info. ==============================
    image_total_batch_size = args.image_micro_batch_size[-1] * (world_size // dp_degree) if image_dataset is not None else 0
    video_total_batch_size = args.video_micro_batch_size[-1] * (world_size // dp_degree) if video_dataset is not None else 0
    if args.train_iters:
        args.num_train_epochs_image = len(image_dataset) // image_total_batch_size if image_dataset is not None else 0
        args.num_train_epochs_video = len(video_dataset) // video_total_batch_size if video_dataset is not None else 0

    video_num = 0
    image_num = 0
    if video_dataset is not None:
        video_num = video_dataset.total_length
    if image_dataset is not None:
        image_num = image_dataset.total_length

    params_count = model.params_count()
    print_training_configuration(args, logger, model, world_size, local_rank, rank, 
                                dp_degree, dp_rank, image_total_batch_size, video_total_batch_size, image_num, video_num,
                                image_loader, video_loader, image_dataset, video_dataset, scalar_states)
    
    # ============================= Start training =============================
    # # HACK: save pretrained weights as step-0 checkpoint before any training. Remove after use.
    # logger.info(f"--> saving pretrained checkpoint at step {scalar_states.train_steps}")
    # model_engine.save_checkpoint(args.output_dir, client_state=scalar_states, tag=f"global_step-{scalar_states.train_steps}")
    # dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

    global_step = 0 # for profiler
    # tmptmp
    # constant_batch = None
    for epoch in range(args.num_train_epochs):
        cycle_states = CycleStates()
        data_iter = iter(loader)
        model_engine.zero_grad()
        global_start_time = sync_cuda_time()

        if getattr(args, 'dry_run', False) and rank == 0:
            logger.info("=" * 80)
            logger.info("DRY RUN MODE ENABLED")
            logger.info("Will only iterate through data without training.")
            logger.info("=" * 80)

        while True:
            start_time = sync_cuda_time()
            try:
                batch = next(data_iter)
                batch = sync_object_for_parallel_training(batch, parallel_dims=parallel_dims) # syncing is cheap: <0.01s
            except StopIteration:
                break

            # ====================================================================================================
            # # tmptmp
            # if constant_batch is None:
            #     constant_batch = batch
            # else:
            #     batch = constant_batch
            # ====================================================================================================

            if getattr(args, 'dry_run', False):
                media, latents, text_ids, text_mask, kwargs = batch
                data_type = kwargs.get('type', ['unknown'])[0]
                cur_batch_size = latents.shape[0] / (world_size/dp_degree)

                scalar_states.add(train_steps=1)
                is_update_step = scalar_states.train_steps % args.gradient_accumulation_steps == 0
                cycle_states.add(log_steps=1)

                if is_update_step:
                    scalar_states.add(update_steps=1)

                scalar_states = sync_object_for_parallel_training(scalar_states, parallel_dims=parallel_dims, force_object=True) # syncing is cheap: <0.01s

                if is_update_step and scalar_states.update_steps % args.log_interval == 0:
                    end_time = sync_cuda_time()
                    sec_per_step = (end_time - global_start_time) / cycle_states.log_steps
                    if rank == 0:
                        logger.info(
                            f"[Dry Run] Progress: {scalar_states.update_steps}/{args.train_iters} | "
                            f"Step time: {sec_per_step:.2f}s | "
                            f"Data type: {data_type}"
                        )
                    
                    cycle_states.reset_epoch_based_states()
                    global_start_time = sync_cuda_time()

                if scalar_states.update_steps >= args.train_iters:
                    logger.info("Dry run finished.")
                    if get_sequence_parallel_state():
                        destroy_sequence_parallel_group()
                    return
                
                continue

            dataloader_time_ = sync_cuda_time()     
            # prepare input data
            latents, model_kwargs, data_type, n_tokens, multitask_mask, cond_latents, last_frame_cond_latents, mask_type, use_i2v_instruction, videoid, repa_feats, index, text = \
                prepare_model_inputs(batch, device, model, vae, text_encoder, text_encoder_2, args, byt5_kwarg, extra_model, seed=args.local_seed)
            
            # Perform sample validation
            scalar_states = sync_object_for_parallel_training(scalar_states, parallel_dims=parallel_dims, force_object=True) # syncing is cheap: < 0.01s
            current_update_step = int(getattr(scalar_states, "update_steps", 0) or 0)
            should_resume_fixed_prompt_step0 = (
                resume_fixed_prompt_step0_pending
                and not args.is_profiler
                and current_update_step > 0
                and last_sample_validation_update_step != current_update_step
            )
            should_sample_this_update = should_resume_fixed_prompt_step0 or (
                not args.is_profiler
                and args.sample_interval > 0
                and current_update_step > 0
                and current_update_step % args.sample_interval == 0
                and last_sample_validation_update_step != current_update_step
            )
            if should_sample_this_update:
                last_sample_validation_update_step = current_update_step
                _is_fixed_prompt_step = (
                    should_resume_fixed_prompt_step0
                    or (
                        current_update_step > 0
                        and getattr(args, 'fixed_prompt_validation_every', 0) > 0
                        and current_update_step % args.fixed_prompt_validation_every == 0
                    )
                )
                _skip_this_fixed_prompt_validation = (
                    skip_initial_fixed_prompt_validation
                    and _is_fixed_prompt_step
                    and skip_initial_fixed_prompt_validation_step == current_update_step
                )
                if _skip_this_fixed_prompt_validation:
                    if rank == 0:
                        logger.info(
                            "Skipping initial fixed prompt validation immediately after resume "
                            f"at update_step={current_update_step}."
                        )
                    skip_initial_fixed_prompt_validation = False
                    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                else:
                    skip_initial_fixed_prompt_validation = False
                    sample_validation(args, device, vae, text_encoder, text_encoder_2, model,
                                batch, current_update_step, logger, byt5_kwarg, extra_model, dp_rank,
                                multitask_mask, mask_type, use_i2v_instruction,
                                parallel_dims=parallel_dims)
                    resume_fixed_prompt_step0_pending = False
                    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

                    # ── Auto-evaluation on fixed_prompt videos (rank 0 only) ──
                    # Only run when this step is a fixed_prompt_validation step (has test videos).
                    if _is_fixed_prompt_step and rank == 0 and getattr(args, 'video_csv', None):
                        try:
                            _run_inline_evaluation(args, current_update_step, logger)
                        except Exception as _eval_exc:
                            logger.warning(f"[InlineEval] Failed (non-fatal): {_eval_exc}")
                    if _is_fixed_prompt_step:
                        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            cur_batch_size = latents.shape[0] / (world_size/ dp_degree)
            data_time_ = sync_cuda_time()

            def _compact_train_debug(value):
                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        return []
                    if value.numel() == 1:
                        return value.detach().cpu().item()
                    return value.detach().cpu().tolist()
                if isinstance(value, (list, tuple)):
                    return [_compact_train_debug(item) for item in value]
                return value

            def _print_training_batch_debug(stage):
                debug_limit = max(
                    int(getattr(args, "reference_timeline_debug_limit", 0)),
                    0,
                )
                debug_count = int(
                    getattr(main, "_reference_timeline_training_debug_count", 0)
                )
                if (
                    temporal_pack_mode != REFERENCE_TIMELINE
                    or rank % 8 != 0
                    or debug_count >= debug_limit
                ):
                    return
                main._reference_timeline_training_debug_count = debug_count + 1
                extra_kwargs_dbg = model_kwargs.get("extra_kwargs", {})
                condition_ranges_dbg = extra_kwargs_dbg.get("condition_frame_ranges", [])
                condition_src_ranges_dbg = extra_kwargs_dbg.get("condition_src_frame_ranges", [])
                physical_ranges_dbg = extra_kwargs_dbg.get(
                    "condition_physical_dst_frame_ranges", []
                )
                debug_task_mode = extra_kwargs_dbg.get("debug_task_mode", mask_type)
                debug_source = extra_kwargs_dbg.get("debug_condition_source", "unknown")
                debug_dropout = extra_kwargs_dbg.get("debug_cond_dropout_applied", False)
                debug_dropped_slots = extra_kwargs_dbg.get("debug_cond_dropped_slots", [])
                debug_levels = extra_kwargs_dbg.get("debug_cond_slot_levels", [])
                debug_l4_active = extra_kwargs_dbg.get("debug_l4_active", False)
                source_lengths_dbg = [
                    int(end) - int(start)
                    for start, end in condition_src_ranges_dbg
                ]
                target_lengths_dbg = [
                    int(end) - int(start)
                    for start, end in physical_ranges_dbg
                ]
                print(
                    f"[TRAIN REFERENCE-TIMELINE FIXED TARGET DEBUG "
                    f"{debug_count + 1}/{debug_limit}] rank={rank} stage={stage} "
                    f"train_step={scalar_states.train_steps} update_step={scalar_states.update_steps} "
                    f"task={debug_task_mode} source={debug_source} data_type={data_type} "
                    f"mask_type={mask_type} range_condition={'ON' if mask_type == 'range_condition' else 'OFF'} "
                    f"range_mode={args.range_condition_mode} "
                    f"temporal_pack_mode={getattr(args, 'range_condition_temporal_pack_mode', 'legacy_trilinear')} "
                    f"cond_latents={'present' if cond_latents is not None else 'none'} "
                    f"cond_dropout_applied={debug_dropout} cond_dropped_slots={debug_dropped_slots} "
                    f"levels={debug_levels} l4_active={debug_l4_active} "
                    f"condition_ranges={condition_ranges_dbg} condition_src_ranges={condition_src_ranges_dbg} "
                    f"condition_physical_dst_ranges={physical_ranges_dbg} "
                    f"source_control_lengths={source_lengths_dbg} "
                    f"target_condition_lengths={target_lengths_dbg} "
                    f"multitask_mask_sum={None if multitask_mask is None else int(multitask_mask.sum().item())} "
                    f"latents_shape={tuple(latents.shape)} "
                    f"target_latent_T={int(latents.shape[2])} "
                    f"generated_latent_T={int(latents.shape[2])} "
                    f"cond_shape={None if cond_latents is None else tuple(cond_latents.shape)} "
                    f"videoid={_compact_train_debug(videoid)} index={_compact_train_debug(index)}",
                    flush=True,
                )

            # broadcast a zero size tensor to indicate starting of step for profiling
            start_flag_tensor = torch.cuda.FloatTensor([])
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(start_flag_tensor, 0, async_op=True)

            with torch.autocast("cuda", torch.bfloat16):
                if args.is_profiler and global_step >= 20:
                    with profile(activities=activities) as prof:
                        _print_training_batch_debug("pre_denoiser_profiler")
                        _, loss_dict = denoiser.training_losses(model, latents, args=args, model_kwargs=model_kwargs, data_type=data_type,
                                                                multitask_mask=multitask_mask, cond_latents=cond_latents, last_frame_cond_latents=last_frame_cond_latents,
                                                                reorg_token=reorg_token,
                                                                multitask_mask_training_type=args.multitask_mask_training_type, mask_type=mask_type,
                                                                drop_first_frame_loss=args.drop_first_frame_loss,
                                                                videoid=videoid,
                                                                parallel_dims=parallel_dims, 
                                                                repa_feats=repa_feats)
                    prof.export_chrome_trace(f"{args.output_dir}/trace-{rank}-step{global_step}.json")
                    if global_step > 22:
                        sys.exit(0)
                else:
                    sync_random_states(parallel_dims)
                    (
                        latents,
                        model_kwargs,
                        cond_latents,
                        last_frame_cond_latents,
                        reorg_token,
                        multitask_mask,
                        args.multitask_mask_training_type,
                        mask_type,
                        args.drop_first_frame_loss,
                        repa_feats,
                    ) = sync_object_for_parallel_training(
                        [
                            latents,
                            model_kwargs,
                            cond_latents,
                            last_frame_cond_latents,
                            reorg_token,
                            multitask_mask,
                            args.multitask_mask_training_type,
                            mask_type,
                            args.drop_first_frame_loss,
                            repa_feats,
                        ],
                        parallel_dims=parallel_dims,
                    )
                    _print_training_batch_debug("pre_denoiser_synced")
                    _, loss_dict = denoiser.training_losses(model, latents, args=args, model_kwargs=model_kwargs, data_type=data_type, repa_feats=repa_feats,
                                                            multitask_mask=multitask_mask, cond_latents=cond_latents, last_frame_cond_latents=last_frame_cond_latents,
                                                            reorg_token=reorg_token,
                                                            multitask_mask_training_type=args.multitask_mask_training_type, mask_type=mask_type,
                                                            drop_first_frame_loss=args.drop_first_frame_loss,
                                                            parallel_dims=parallel_dims,
                                                            videoid=videoid,
                                                            )
            loss = 0
            if torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any():
                kwargs = get_kwargs(batch, args)
                # print(f"train_step: {scalar_states.train_steps},  Rank {rank}: Loss is NaN or Inf. "
                #       f"md5: {kwargs.get('image_id', '')}, image_path: {kwargs.get('image_path', '')}, "
                #       f"Latent shape: {latents.shape}. Mean: {latents.mean()}. Max: {latents.max()}. Min: {latents.min()}")

            loss = loss_dict['total_loss']
            diffusion_loss = loss_dict['diffusion_loss']

            if loss.numel() > 1:
                loss = loss.mean()
            if diffusion_loss.numel() > 1:
                diffusion_loss = diffusion_loss.mean()

            # deffensive check
            sync_object_for_parallel_training(loss, parallel_dims=parallel_dims, debug_with_check=True, trace='loss') # syncing is cheap: < 0.01s
            #sync_object_for_parallel_training_debug(loss, parallel_dims=parallel_dims, debug_with_check=True)

            # Simple loss monitoring - just print when loss is high
            current_loss = loss.item()
            loss_threshold = getattr(args, 'loss_spike_threshold', 0.5)

            # # Check for NaN/Inf in loss
            # if torch.isnan(loss) or torch.isinf(loss):
            #     kwargs = get_kwargs(batch, args)
            #     print(f"Rank {rank}: NaN/Inf loss detected at step {scalar_states.train_steps}. "
            #           f"Loss: {current_loss}, Data type: {data_type}, Batch size: {cur_batch_size}, "
            #           f"MD5: {kwargs.get('image_id', '')}, Mask type: {mask_type}")

            # # Print when loss exceeds threshold
            # elif current_loss > loss_threshold:
            #     print(f"Rank {rank}: High loss detected at step {scalar_states.train_steps}. "
            #           f"Loss: {current_loss:.6f} > {loss_threshold}, Data type: {data_type}, "
            #           f"Batch size: {cur_batch_size} "
            #           f"Mask type: {mask_type}, Diffusion loss: {diffusion_loss.item():.6f}")
            #     if 'moe_loss' in loss_dict:
            #         print(f"  MoE loss: {loss_dict['moe_loss'].item():.6f}")
            #     if 'repa_loss' in loss_dict:
            #         print(f"  Repa loss: {loss_dict['repa_loss'].item():.6f}")
            #     if 'diffusion_loss' in loss_dict:
            #         print(f"  Diffusion loss: {loss_dict['diffusion_loss'].item():.6f}")
            #     print(f"  Latent stats - Mean: {latents.mean():.6f}, Std: {latents.std():.6f}, "
            #           f"Max: {latents.max():.6f}, Min: {latents.min():.6f}")

            forward_time_ = sync_cuda_time()

            # update scalar state
            scalar_states.add(train_steps=1)


            # Update comsumed samples per dp
            kwargs = get_kwargs(batch, args)
            data_dtype = kwargs['dataset_type'][0]
            scalar_states.consumed_samples_per_dp[data_dtype] = loader.scalar_states.consumed_samples_per_dp[data_dtype]
            scalar_states.epoch[data_dtype] = loader.scalar_states.epoch[data_dtype] ## update epoch
            is_update_step = scalar_states.train_steps % args.gradient_accumulation_steps == 0

            grad_norm = 0.0
            model_engine.backward(loss / args.gradient_accumulation_steps)
            if is_update_step:
                grad_norm_ = model_engine.clip_grad_norm_(model_engine.parameters(), max_norm=args.max_grad_norm)
                grad_norm = grad_norm_.item()
                if torch.isnan(grad_norm_) or torch.isinf(grad_norm_):
                    bad_grad_names = []
                    for name, param in model_engine.named_parameters():
                        grad = getattr(param, "grad", None)
                        if grad is None:
                            continue
                        try:
                            if not torch.isfinite(grad).all():
                                bad_grad_names.append(name)
                                if len(bad_grad_names) >= 8:
                                    break
                        except Exception:
                            continue
                    extra_kwargs = model_kwargs.get("extra_kwargs", {}) if isinstance(model_kwargs, dict) else {}
                    nan_videoid = videoid[0] if isinstance(videoid, (list, tuple)) and len(videoid) > 0 else videoid
                    nan_text = text[0] if isinstance(text, (list, tuple)) and len(text) > 0 else text
                    nan_segment_time_ranges = extra_kwargs.get("text_segment_time_ranges")
                    if isinstance(nan_segment_time_ranges, list) and len(nan_segment_time_ranges) > 0:
                        nan_segment_time_ranges = nan_segment_time_ranges[0]
                    nan_has_segment_ids = "text_segment_ids" in extra_kwargs
                    # print(f"Rank {rank}: NaN/Inf gradient norm detected at step {scalar_states.update_steps}. "
                    #         f"Grad norm: {grad_norm}, Loss: {current_loss:.6f}, Data type: {data_type}"
                    #         f"Latent shape: {latents.shape}. Mean: {latents.mean()}. Max: {latents.max()}. Min: {latents.min()}"
                    #         f" Bad grad params: {bad_grad_names}"
                    #         f" VideoID: {nan_videoid}"
                    #         f" Text preview: {str(nan_text)[:1200]}"
                    #         f" Has segment ids: {nan_has_segment_ids}"
                    #         f" Segment time ranges: {str(nan_segment_time_ranges)[:1200]}")
                    model_engine.zero_grad()
                    if hasattr(model_engine, "micro_steps"):
                        model_engine.micro_steps += 1
                else:
                    model_engine.step()
                    scalar_states.add(update_steps=1, current_run_update_steps=1)
                    scalar_states.lr = model_engine.get_last_lr()[0]
            else:
                model_engine.step()
            # model_engine.zero_grad()
            backward_time_ = sync_cuda_time()
            # Log training information in current cycle:
            cycle_states.add(log_steps=1, running_loss=loss.item())
            cycle_states.running_samples[data_dtype] += cur_batch_size
            cycle_states.running_tokens[data_dtype] += cur_batch_size * n_tokens
            cycle_states.add_mask_type_samples(mask_type, cur_batch_size)
            cycle_states.add_mask_type_loss(mask_type, loss.item())

            # Add data_type loss tracking for t2v mask type
            if (args.multitask_mask_training_type == 'concat' or args.multitask_mask_training_type == 'token_replace') and mask_type == 't2v':
                cycle_states.add_data_type_loss(data_type, loss.item(), cur_batch_size)

            # Update mask type statistics in scalar_states
            scalar_states.consumed_samples_by_mask_type_per_dp[mask_type] += cur_batch_size

            # Update data type statistics in scalar_states for t2v
            if (args.multitask_mask_training_type == 'concat' or args.multitask_mask_training_type == 'token_replace') and mask_type == 't2v':
                scalar_states.consumed_samples_by_data_type_per_dp[data_type] += cur_batch_size

            if is_update_step and scalar_states.update_steps % args.log_interval == 0:
                # Collect all unique mask types and data types across all processes
                # 为了确保分布式同步，需要在所有进程间同步mask_type和data_type集合
                local_mask_types = list(cycle_states.running_loss_by_mask_type.keys())
                local_data_types = list(cycle_states.running_loss_by_data_type.keys())

                # 收集所有进程的mask_types和data_types
                all_processes_mask_types = [None for _ in range(world_size)]
                all_processes_data_types = [None for _ in range(world_size)]

                dist.all_gather_object(all_processes_mask_types, local_mask_types)
                dist.all_gather_object(all_processes_data_types, local_data_types)

                # 合并并排序，确保所有进程的顺序一致
                all_mask_types = sorted(set().union(*all_processes_mask_types))
                all_data_types = sorted(set().union(*all_processes_data_types))

                # Reduce loss history over all processes:
                global_loss = all_gather_sum(cycle_states.running_loss / cycle_states.log_steps, device) / world_size

                cum_samples, cum_tokens = {}, {}
                for data_key in loader.all_keys_with_none:
                    cum_samples[data_key] = all_gather_sum(cycle_states.running_samples[data_key], device)
                    cum_tokens[data_key] = all_gather_sum(cycle_states.running_tokens[data_key], device)

                # Gather and accumulate mask type statistics from all processes
                for mt in all_mask_types:
                    count = cycle_states.running_samples_by_mask_type.get(mt, 0)
                    mask_type_samples_total = all_gather_sum(count, device)
                    scalar_states.consumed_samples_by_mask_type_total[mt] += mask_type_samples_total

                # Gather and accumulate data type statistics from all processes for t2v
                for dt in all_data_types:
                    count = cycle_states.running_samples_by_data_type.get(dt, 0)
                    data_type_samples_total = all_gather_sum(count, device)
                    scalar_states.consumed_samples_by_data_type_total[dt] += data_type_samples_total

                # Gather current cycle samples for mask types and data types
                # 使用统一的mask_type列表确保所有进程执行相同的all_gather_sum操作
                global_cycle_samples_by_mask_type = {}
                for mt in all_mask_types:
                    local_samples = cycle_states.running_samples_by_mask_type.get(mt, 0)
                    global_samples = all_gather_sum(local_samples, device)
                    global_cycle_samples_by_mask_type[mt] = global_samples

                global_cycle_samples_by_data_type = {}
                for dt in all_data_types:
                    local_samples = cycle_states.running_samples_by_data_type.get(dt, 0)
                    global_samples = all_gather_sum(local_samples, device)
                    global_cycle_samples_by_data_type[dt] = global_samples

                # Gather mask type and data type loss values across all processes for accurate averaging
                # We need to gather both the total loss and count to compute correct global averages

                # Gather mask type losses
                global_avg_loss_by_mask_type = {}
                for mt in all_mask_types:
                    local_loss = cycle_states.running_loss_by_mask_type.get(mt, 0.0)
                    local_count = cycle_states.running_loss_count_by_mask_type.get(mt, 0)
                    total_loss = all_gather_sum(local_loss, device)
                    total_count = all_gather_sum(local_count, device)
                    if total_count > 0:
                        global_avg_loss_by_mask_type[mt] = total_loss / total_count
                    else:
                        global_avg_loss_by_mask_type[mt] = 0.0

                # Gather data type losses
                global_avg_loss_by_data_type = {}
                for dt in all_data_types:
                    local_loss = cycle_states.running_loss_by_data_type.get(dt, 0.0)
                    local_count = cycle_states.running_loss_count_by_data_type.get(dt, 0)
                    total_loss = all_gather_sum(local_loss, device)
                    total_count = all_gather_sum(local_count, device)
                    if total_count > 0:
                        global_avg_loss_by_data_type[dt] = total_loss / total_count
                    else:
                        global_avg_loss_by_data_type[dt] = 0.0

                cum_samples_total = sum(cum_samples.values())
                cum_tokens_total = sum(cum_tokens.values())

                # Measure training speed:
                end_time = sync_cuda_time()
                samples_per_sec = cum_samples_total / (end_time - global_start_time)
                sec_per_step = (end_time - global_start_time) / cycle_states.log_steps
                steps_per_sec = cycle_states.log_steps / (end_time - global_start_time)

                #update scalar_states
                for data_key in loader.all_keys_with_none:
                    scalar_states.consumed_samples_total[data_key] += cum_samples[data_key]
                    scalar_states.epoch_consumed_samples[data_key] += cum_samples[data_key]
                    scalar_states.consumed_tokens_total[data_key] += cum_tokens[data_key]
                scalar_states.add(
                        consumed_computations_attn=6 * params_count['attn+mlp'] * cum_tokens_total / C_SCALE,
                        consumed_computations_total=6 * params_count['total'] * cum_tokens_total / C_SCALE,
                    )

                cycle_states.reset_epoch_based_states()
                global_start_time = sync_cuda_time()

            if rank == 0 and (is_update_step and scalar_states.update_steps % args.log_interval == 0):
                # Initialize empty dictionaries if the variables don't exist (when first condition wasn't met)
                if 'global_cycle_samples_by_mask_type' not in locals():
                    global_cycle_samples_by_mask_type = {}
                if 'global_avg_loss_by_mask_type' not in locals():
                    global_avg_loss_by_mask_type = {}
                if 'global_avg_loss_by_data_type' not in locals():
                    global_avg_loss_by_data_type = {}
                if 'global_cycle_samples_by_data_type' not in locals():
                    global_cycle_samples_by_data_type = {}

                progress_info = {
                    "avg_loss": f"{global_loss:.8f}",
                    "step_loss": f"{loss.item():.8f}",
                    "grad_norm": f"{grad_norm:.10f}",
                    "step_time": f"{sec_per_step:.2f}s",
                    "steps_per_sec": f"{steps_per_sec:.2f}",
                    "samples_per_sec": f"{samples_per_sec:.2f}",
                    "dataloader_time": f"{dataloader_time_ - start_time:.2f}s",
                    "data_time": f"{data_time_ - dataloader_time_:.2f}s",
                    "forward_time": f"{forward_time_ - data_time_:.2f}s",
                    "backward_time": f"{backward_time_ - forward_time_:.2f}s",
                    "learning_rate": f"{model_engine.get_last_lr()[0]:.8f}",
                    "current_batch_n_tokens": f"{n_tokens}",
                    "consumed_computations_attn": f"{scalar_states.consumed_computations_attn}",
                    "consumed_computations_total": f"{scalar_states.consumed_computations_total}",
                    "current_mask_type": mask_type
                }
                for data_key in loader.all_keys_with_none:
                    progress_info[f"cum_samples_{data_key}"] = f"{cum_samples[data_key]}"
                    progress_info[f"consumed_tokens_{data_key}_total"] = f"{scalar_states.consumed_tokens_total[data_key]}"
                    progress_info[f"consumed_samples_{data_key}_total"] = f"{scalar_states.consumed_samples_total[data_key]}"
                    progress_info[f"consumed_samples_{data_key}_epoch"] = f"{scalar_states.epoch_consumed_samples[data_key]}"
                    progress_info[f"consumed_samples_{data_key}_per_dp"] = f"{scalar_states.consumed_samples_per_dp[data_key]}"

                # Add mask type statistics to progress info
                mask_type_stats = dict(scalar_states.consumed_samples_by_mask_type_total)
                for mt, count in mask_type_stats.items():
                    progress_info[f"consumed_samples_{mt}_total"] = f"{count}"

                # Add data type statistics to progress info for t2v
                data_type_stats_total = dict(scalar_states.consumed_samples_by_data_type_total)
                for dt, count in data_type_stats_total.items():
                    progress_info[f"consumed_samples_t2v_{dt}_total"] = f"{count}"

                # Add current cycle mask type statistics (using global samples)
                for mt, count in global_cycle_samples_by_mask_type.items():
                    progress_info[f"cycle_samples_{mt}"] = f"{count}"

                # Add current cycle mask type loss statistics (using global averages)
                for mt, avg_loss in global_avg_loss_by_mask_type.items():
                    progress_info[f"cycle_avg_loss_{mt}"] = f"{avg_loss:.8f}"

                # Add current cycle data type loss statistics for t2v (using global averages)
                for dt, avg_loss in global_avg_loss_by_data_type.items():
                    progress_info[f"cycle_avg_loss_t2v_{dt}"] = f"{avg_loss:.8f}"

                # Add current cycle data type sample statistics for t2v (using global samples)
                for dt, count in global_cycle_samples_by_data_type.items():
                    progress_info[f"cycle_samples_t2v_{dt}"] = f"{count}"

                if "moe_loss" in loss_dict:
                    progress_info["moe_loss"] = f"{loss_dict['moe_loss'].item():.8f}"
                    progress_info["diffusion_loss"] = f"{diffusion_loss.item():.8f}"
                if "repa_loss" in loss_dict:
                    progress_info["repa_loss"] = f"{loss_dict['repa_loss'].item():.8f}"

                logger.info(f"Progress: {scalar_states.update_steps}/{args.train_iters} | Details: {progress_info}")



                # Log training metrics
                tb_writer.add_scalar("Train/Steps/train_loss", global_loss, scalar_states.update_steps)
                if 'diffusion_loss' in loss_dict:
                    tb_writer.add_scalar("Train/Steps/diffusion_loss", loss_dict['diffusion_loss'].item(), scalar_states.update_steps)
                if 'moe_loss' in loss_dict:
                    tb_writer.add_scalar("Train/Steps/moe_loss", loss_dict['moe_loss'].item(), scalar_states.update_steps)
                if 'repa_loss' in loss_dict:
                    tb_writer.add_scalar("Train/Steps/repa_loss", loss_dict['repa_loss'].item(), scalar_states.update_steps)
                tb_writer.add_scalar("Train/Steps/step_loss", loss.cpu().item(), scalar_states.update_steps)
                tb_writer.add_scalar("Train/Steps/grad_norm", grad_norm, scalar_states.update_steps)
                for data_key in loader.all_keys_with_none:
                    tb_writer.add_scalar(f"Train/Tokens/{data_key}_train_loss", global_loss, scalar_states.consumed_tokens_total[data_key])
                tb_writer.add_scalar("Train/ComputationsAttn/train_loss", global_loss, scalar_states.consumed_computations_attn)
                tb_writer.add_scalar("Train/ComputationsTotal/train_loss", global_loss, scalar_states.consumed_computations_total)

                # Log learning rate
                tb_writer.add_scalar("LR/learning_rate", model_engine.get_last_lr()[0], scalar_states.update_steps)

                # Log timing metrics
                tb_writer.add_scalar("Time/step_time", sec_per_step, scalar_states.update_steps)
                tb_writer.add_scalar("Time/steps_per_sec", steps_per_sec, scalar_states.update_steps)
                tb_writer.add_scalar("Time/samples_per_sec", int(samples_per_sec), scalar_states.update_steps)
                tb_writer.add_scalar("Time/dataloader_time", dataloader_time_ - start_time, scalar_states.update_steps)
                tb_writer.add_scalar("Time/data_time", data_time_ - dataloader_time_, scalar_states.update_steps)
                tb_writer.add_scalar("Time/forward_time", forward_time_ - data_time_, scalar_states.update_steps)
                tb_writer.add_scalar("Time/backward_time", backward_time_ - forward_time_, scalar_states.update_steps)

                # Consumed samples
                for data_key in loader.all_keys_with_none:
                    tb_writer.add_scalar(f"Data/{data_key}_samples_total", scalar_states.consumed_samples_total[data_key], scalar_states.update_steps)
                    tb_writer.add_scalar(f"Data/{data_key}_samples_epoch", scalar_states.epoch_consumed_samples[data_key], scalar_states.update_steps)

                for data_key in loader.all_keys_with_none:
                    tb_writer.add_scalar(f"Data/{data_key}_tokens_total", scalar_states.consumed_tokens_total[data_key], scalar_states.update_steps)

                # Log mask type statistics to tensorboard
                for mt, count in mask_type_stats.items():
                    tb_writer.add_scalar(f"Data/MaskType/{mt}_samples_total", count, scalar_states.update_steps)

                # Log global data type statistics to tensorboard for t2v
                for dt, count in data_type_stats_total.items():
                    tb_writer.add_scalar(f"Data/DataType/t2v_{dt}_samples_total", count, scalar_states.update_steps)

                # Log mask type loss statistics to tensorboard (using global averages)
                for mt, avg_loss in global_avg_loss_by_mask_type.items():
                    tb_writer.add_scalar(f"Train/MaskType/{mt}_avg_loss", avg_loss, scalar_states.update_steps)

                # Log mask type current cycle sample statistics to tensorboard (using global samples)
                for mt, count in global_cycle_samples_by_mask_type.items():
                    tb_writer.add_scalar(f"Data/MaskType/{mt}_samples", count, scalar_states.update_steps)

                # Log data type loss statistics to tensorboard for t2v (using global averages)
                for dt, avg_loss in global_avg_loss_by_data_type.items():
                    tb_writer.add_scalar(f"Train/DataType/t2v_{dt}_avg_loss", avg_loss, scalar_states.update_steps)

                # Log data type sample statistics to tensorboard for t2v (using global samples)
                for dt, count in global_cycle_samples_by_data_type.items():
                    tb_writer.add_scalar(f"Data/DataType/t2v_{dt}_samples", count, scalar_states.update_steps)
            # reset cycle states
            #cycle_states.reset_epoch_based_states()

            # Save checkpoint
            if is_update_step and scalar_states.update_steps % args.checkpointing_steps == 0:
                logger.info(f"--> save checkpoint at update step {scalar_states.update_steps}, {args.output_dir}")
                model_engine.save_checkpoint(args.output_dir, client_state=scalar_states, tag=f"global_step-{scalar_states.update_steps}")
                # save_checkpoint_without_optim(model, rank, args.output_dir, scalar_states)
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                
                # Also save curriculum checkpoint
                curriculum_ckpt_path = os.path.join(args.output_dir, f"curriculum_step_{scalar_states.update_steps}.json")
                curriculum.save_checkpoint(curriculum_ckpt_path)
                
                # Always keep a "final" version for resume
                curriculum_final_path = os.path.join(args.output_dir, "curriculum_final.json")
                curriculum.save_checkpoint(curriculum_final_path)
                
            # elif grad_norm > 0.75 and scalar_states.train_steps > 10000:
            #     tag = f"global_step-{scalar_states.train_steps}"
            #     dump_spike_gradients_fsdp(model_engine, args.output_dir, step=scalar_states.train_steps, rank=rank, tag=tag)
            #     logger.info(f"--> loss is spike, loss {current_loss}, grad_norm: {grad_norm}, save checkpoint at step {scalar_states.train_steps}, {args.output_dir}")
            #     model_engine.save_checkpoint(args.output_dir, client_state=scalar_states, tag=tag)
            #     dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            # Perform evaluation
            if args.validation_every > 0 and (
                    (is_update_step and scalar_states.update_steps % args.validation_every == 0)
                or (is_update_step and scalar_states.update_steps in args.validation_at_steps)
            ):
                evaluate_validation(args, device, vae, text_encoder, text_encoder_2, model,
                                  scalar_states.update_steps, logger, tb_writer, scalar_states,
                                  loss_evaluator, score_evaluator, byt5_kwarg, extra_model)
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            global_step += 1

    model_engine.save_checkpoint(
        args.output_dir, client_state=scalar_states
    )
    
    # Save final curriculum state
    final_curriculum_path = os.path.join(args.output_dir, "curriculum_final.json")
    curriculum.save_checkpoint(final_curriculum_path)
    if rank == 0:
        logger.info(f"Saved final curriculum checkpoint to {final_curriculum_path}")
        logger.info(curriculum.get_summary())

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


    
