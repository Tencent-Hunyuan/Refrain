# Standard library imports
import collections.abc
import inspect
import os
import json
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from itertools import repeat
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any

# Third-party imports
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Local imports
import random
from hymm.utils.multitask_utils import get_multitask_mask, get_cond_latents, get_cond_images, get_multitask_mask_with_ranges
from hymm.utils.native_reference import as_bool
from hymm.utils.reference_temporal_pack import REFERENCE_TIMELINE, normalize_temporal_pack_mode
from hymm.models.vision_encoder import VisionEncoder


def _parse_float_pair(value, name):
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if (text.startswith("(") and text.endswith(")")) or (text.startswith("[") and text.endswith("]")):
            text = text[1:-1]
        items = [item.strip() for item in text.split(",") if item.strip()]
    else:
        raise ValueError(f"{name} must be a length-2 list/tuple/string, got {type(value)}")

    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly 2 values, got {value}")
    low = float(items[0])
    high = float(items[1])
    assert math.isfinite(low) and math.isfinite(high), f"{name} values must be finite, got {value}"
    assert 0.0 < low <= high, f"{name} must satisfy 0 < low <= high, got {value}"
    return low, high


@dataclass
class BaseStates:
    epoch: int = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)

    def reset_epoch_based_states(self):
        pass

    def inc_epoch(self):
        """ Increase (image) epoch by 1. """
        self.epoch += 1
        self.reset_epoch_based_states()
        return self.epoch

    def __repr__(self):
        max_key_length = max([len(k) for k in asdict(self).keys()])
        return "\n".join(
            [f"{k:>{max_key_length}}: {v}" for k, v in asdict(self).items()]
        )

    def to_dict(self):
        states = deepcopy(vars(self))
        # Convert defaultdict to dict
        for key, value in states.items():
            if isinstance(value, defaultdict):
                states[key] = dict(value)

        return states

    @classmethod
    def from_pretrained(cls, scalar_state, rank=None, world_size=None, default_rank0_ss=None, default=None):
        _scalar_state = default or {}
        if isinstance(scalar_state, dict):
            _scalar_state.update(scalar_state)
        elif isinstance(scalar_state, list):
            assert rank is not None and world_size is not None and default_rank0_ss is not None, (
                f"rank({rank}), world_size({world_size}), and default_rank0_ss({default_rank0_ss}) should be provided "
                f"when scalar_state is a list."
            )
            if len(scalar_state) != world_size:
                # Check if scalar_state all the same
                if all([scalar_state[0] == x for x in scalar_state]):
                    default_rank0_ss = True

                if default_rank0_ss:
                    from loguru import logger
                    logger.info(f" len(scalar_state)={len(scalar_state)} != world_size={world_size}, Set default rank0 scalar state.")
                    _scalar_state.update(scalar_state[0])
                else:
                    _scalar_state.update(scalar_state[0])
                    for i in range(len(scalar_state)):
                        if isinstance(scalar_state[i], dict):
                            for k, v in scalar_state[i].items():
                                if isinstance(v, dict):
                                    _scalar_state[k].update(v)
                    from loguru import logger
                    logger.info(f"default_rank0_ss is alse, Update scalar_state={_scalar_state}")
            else:
                _scalar_state.update(scalar_state[rank])
        else:
            raise ValueError(f"Unknown scalar state type: {type(scalar_state)}")
        
        # Convert dict fields back to defaultdict for mask type statistics
        if 'consumed_samples_by_mask_type_total' in _scalar_state:
            _scalar_state['consumed_samples_by_mask_type_total'] = defaultdict(int, _scalar_state['consumed_samples_by_mask_type_total'])
        if 'consumed_samples_by_mask_type_per_dp' in _scalar_state:
            _scalar_state['consumed_samples_by_mask_type_per_dp'] = defaultdict(int, _scalar_state['consumed_samples_by_mask_type_per_dp'])
        
        # Convert dict fields back to defaultdict for data type statistics
        if 'consumed_samples_by_data_type_total' in _scalar_state:
            _scalar_state['consumed_samples_by_data_type_total'] = defaultdict(int, _scalar_state['consumed_samples_by_data_type_total'])
        if 'consumed_samples_by_data_type_per_dp' in _scalar_state:
            _scalar_state['consumed_samples_by_data_type_per_dp'] = defaultdict(int, _scalar_state['consumed_samples_by_data_type_per_dp'])
        
        return cls(**_scalar_state)


@dataclass
class ScalarStates(BaseStates):
    """
    Training states for the whole training lifecycle. Should be saved/loaded along with the model checkpoint.
    """
    # - Global level
    epoch: dict = field(default_factory=lambda: defaultdict(int))
    train_steps: int = 0  # Accumulated training steps
    update_steps: int = 0  # Accumulated update steps
    current_run_update_steps: int = 0  # Update steps in current run (Reset for every resume)
    consumed_samples_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples
    epoch_consumed_samples: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples

    consumed_samples_per_dp: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples per data-parallel group
    consumed_tokens_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed tokens
    consumed_computations_attn: float = 0         # Accumulated consumed computations of attention + mlp
    consumed_computations_total: float = 0        # Accumulated consumed computations of total
    
    # - Mask type statistics
    consumed_samples_by_mask_type_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples by mask type
    consumed_samples_by_mask_type_per_dp: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples by mask type per data-parallel group
    
    # - Data type statistics for t2v mask type
    consumed_samples_by_data_type_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples by data type for t2v
    consumed_samples_by_data_type_per_dp: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples by data type per data-parallel group for t2v
    
    lr: float = 0.0  # Current learning rate


@dataclass
class CycleStates(BaseStates):
    log_steps: int = 0
    running_loss: float = 0
    running_tokens: dict = field(default_factory=lambda: defaultdict(int))
    running_samples: dict = field(default_factory=lambda: defaultdict(int))
    running_samples_by_mask_type: dict = field(default_factory=lambda: defaultdict(int))
    running_loss_by_mask_type: dict = field(default_factory=lambda: defaultdict(float))
    running_loss_count_by_mask_type: dict = field(default_factory=lambda: defaultdict(int))
    # Add data_type tracking for t2v mask type
    running_loss_by_data_type: dict = field(default_factory=lambda: defaultdict(float))
    running_loss_count_by_data_type: dict = field(default_factory=lambda: defaultdict(int))
    running_samples_by_data_type: dict = field(default_factory=lambda: defaultdict(int))
    
    def reset_epoch_based_states(self):
        self.log_steps = 0
        self.running_loss = 0
        self.running_tokens = defaultdict(int)
        self.running_samples = defaultdict(int)
        self.running_samples_by_mask_type = defaultdict(int)
        self.running_loss_by_mask_type = defaultdict(float)
        self.running_loss_count_by_mask_type = defaultdict(int)
        self.running_loss_by_data_type = defaultdict(float)
        self.running_loss_count_by_data_type = defaultdict(int)
        self.running_samples_by_data_type = defaultdict(int)
    
    def add_mask_type_samples(self, mask_type, samples):
        self.running_samples_by_mask_type[mask_type] += samples
    
    def add_mask_type_loss(self, mask_type, loss_value):
        self.running_loss_by_mask_type[mask_type] += loss_value
        self.running_loss_count_by_mask_type[mask_type] += 1
    
    def add_data_type_loss(self, data_type, loss_value, samples):
        """Add loss and samples for specific data_type"""
        self.running_loss_by_data_type[data_type] += loss_value
        self.running_loss_count_by_data_type[data_type] += 1
        self.running_samples_by_data_type[data_type] += samples
    
    def get_avg_loss_by_mask_type(self):
        """获取各mask_type的平均loss"""
        avg_losses = {}
        for mask_type in self.running_loss_by_mask_type.keys():
            count = self.running_loss_count_by_mask_type[mask_type]
            if count > 0:
                avg_losses[mask_type] = self.running_loss_by_mask_type[mask_type] / count
            else:
                avg_losses[mask_type] = 0.0
        return avg_losses
    
    def get_avg_loss_by_data_type(self):
        """获取各data_type的平均loss"""
        avg_losses = {}
        for data_type in self.running_loss_by_data_type.keys():
            count = self.running_loss_count_by_data_type[data_type]
            if count > 0:
                avg_losses[data_type] = self.running_loss_by_data_type[data_type] / count
            else:
                avg_losses[data_type] = 0.0
        return avg_losses

def add_special_token(
    tokenizer,
    text_encoder,
    add_color,
    add_font,
    color_ann_path,
    font_ann_path,
    multilingual=False,
):
    """Add special tokens for color and font to tokenizer and text encoder."""
    with open(font_ann_path, 'r') as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, 'r') as f:
        idx_color_dict = json.load(f)

    if multilingual:
        font_token = []
        for font_code in idx_font_dict:
            prefix = font_code[:2]
            font_token.append(f'<{prefix}-font-{idx_font_dict[font_code]}>')
    else:
        font_token = [f'<font-{i}>' for i in range(len(idx_font_dict))]
    
    color_token = [f'<color-{i}>' for i in range(len(idx_color_dict))]
    additional_special_tokens = []
    
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token

    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_byt5_and_byt5_tokenizer(
    byt5_name='google/byt5-small',
    special_token=False,
    color_special_token=False,
    font_special_token=False,
    color_ann_path='assets/color_idx.json',
    font_ann_path='assets/font_idx_512.json',
    huggingface_cache_dir=None,
    multilingual=False,
):
    """Load ByT5 tokenizer and text encoder with optional special tokens."""
    byt5_tokenizer = AutoTokenizer.from_pretrained(
        byt5_name, cache_dir=huggingface_cache_dir,
    )
    byt5_text_encoder = T5ForConditionalGeneration.from_pretrained(
        byt5_name, cache_dir=huggingface_cache_dir,
    ).get_encoder()

    if special_token:
        add_special_token(
            byt5_tokenizer,
            byt5_text_encoder,
            add_color=color_special_token,
            add_font=font_special_token,
            color_ann_path=color_ann_path,
            font_ann_path=font_ann_path,
            multilingual=multilingual,
        )

    return byt5_text_encoder, byt5_tokenizer

def unpack_batch_args(batch, device):
    media, latents, text_ids, text_mask, kwargs = batch
    text_ids_2 = kwargs.get("text_ids_2", None)
    text_mask_2 = kwargs.get("text_mask_2", None)
    byt5_text_ids = kwargs.get("byt5_text_ids", None)
    byt5_text_mask = kwargs.get("byt5_text_mask", None)
    byt5_text_valid = kwargs.get("byt5_text_valid", None)
    text_ids_i2v_instruction = kwargs.get("text_ids_i2v_instruction", None)
    text_mask_i2v_instruction = kwargs.get("text_mask_i2v_instruction", None)
    
    if text_ids_2 is not None:
        text_ids_2 = text_ids_2.to(device)
    if text_mask_2 is not None:
        text_mask_2 = text_mask_2.to(device)
    if byt5_text_ids is not None:
        byt5_text_ids = byt5_text_ids.to(device)
    if byt5_text_mask is not None:
        byt5_text_mask = byt5_text_mask.to(device)
    if text_ids_i2v_instruction is not None:
        text_ids_i2v_instruction = text_ids_i2v_instruction.to(device)
    if text_mask_i2v_instruction is not None:
        text_mask_i2v_instruction = text_mask_i2v_instruction.to(device)
    
    
    # Move batch to device
    media = media.to(device)
    latents = latents.to(device)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)


    # Move kargs to device if tensor 
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}

    
    return (
            media, 
            latents,
            text_ids, 
            text_mask,
           text_ids_2,
           text_mask_2,
           byt5_text_ids,
           byt5_text_mask,
           byt5_text_valid,
           text_ids_i2v_instruction,
           text_mask_i2v_instruction,
           kwargs)

def prepare_model_inputs(batch, device, model, vae, text_encoder, text_encoder_2=None, args=None, byt5_kwarg=None,
                         extra_model=None, seed=None):
    """Prepare inputs for model training/inference."""
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    # 根据 video_loader.py 的构建逻辑解包 batch_args
    # 固定结构：[media, latents, text_ids, text_mask, kwargs]
    # 条件变量都放在 kwargs 中，通过 key 来索引
    (media, 
    latents,
    text_ids, 
    text_mask,
    text_ids_2,
    text_mask_2,
    byt5_text_ids,
    byt5_text_mask,
    byt5_text_valid,
    text_ids_i2v_instruction,
    text_mask_i2v_instruction,
    kwargs) = unpack_batch_args(batch, device)
    
    videoid = kwargs.get('videoid', None)
    part_version = kwargs.get('part_version', None)

    def _debug_compact(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return []
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return [_debug_compact(item) for item in value]
        return value

    train_task_debug_printed = False
    # ======================================== Encode media ======================================
    # Used for 3D VAE with 2D inputs(image).
    # Prepare media shape for 2D/3D VAE
    if len(latents.shape) == 1:
        if len(media.shape) == 4:
            # media is a batch of image with shape [b, c, h, w]
            media = media.unsqueeze(2)  # [b, c, 1, h, w]
        elif len(media.shape) == 5:
            # `load_vae()` may return any of the supported 3D VAE
            # implementations (for example WanVAE_ for 16164-48c-v3), so a
            # check against the legacy AutoencoderKLCausal3D class is both
            # undefined here and too restrictive. The online-video path only
            # requires a VAE exposing the standard encode() interface.
            if not callable(getattr(vae, "encode", None)):
                raise TypeError(
                    f"Online video VAE extraction requires a 3D VAE with encode(), "
                    f"got {type(vae).__module__}.{type(vae).__name__}"
                )
            # VideoArrowDataset emits each raw video as [F, C, H, W], so the
            # dataloader batch is [B, F, C, H, W]. All supported video VAEs
            # consume [B, C, F, H, W]. Cached-latent training never enters this
            # branch, so normalize the raw-video layout before online encoding.
            channel_sizes = (1, 3, 4)
            if media.shape[1] not in channel_sizes and media.shape[2] in channel_sizes:
                media = media.permute(0, 2, 1, 3, 4).contiguous()
            elif media.shape[1] not in channel_sizes:
                raise ValueError(
                    f"Cannot infer online-video channel dimension from media shape {tuple(media.shape)}; "
                    "expected [B,F,C,H,W] or [B,C,F,H,W] with C in {1,3,4}"
                )
        else:
            raise ValueError(f"Only support media with shape (b, c, h, w) or (b, c, f, h, w), but got {media.shape}.")

        expected_online_latent_t = (int(media.shape[2]) - 1) // 4 + 1
        if rank == 0 and not getattr(prepare_model_inputs, "_online_vae_input_printed", False):
            prepare_model_inputs._online_vae_input_printed = True
            print(
                "[ONLINE VAE INPUT] "
                f"media_shape_bcfhw={tuple(media.shape)} "
                f"target_pixel_frames={int(media.shape[2])} "
                f"expected_latent_T={expected_online_latent_t}"
            )

        vae_dtype = torch.float16
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
            if args.vae_encode_tile_size > 0:
                all_latents = []
                for media_chunk in torch.split(media, args.vae_encode_tile_size):
                    latents_chunk = vae.encode(media_chunk).latent_dist.mode()
                    all_latents.append(latents_chunk)
                latents = torch.cat(all_latents)
            else:
                latents = vae.encode(media).latent_dist.mode()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor:
                latents.sub_(vae.config.shift_factor).mul_(vae.config.scaling_factor)
            else:
                latents.mul_(vae.config.scaling_factor)
        if int(latents.shape[2]) != expected_online_latent_t:
            raise ValueError(
                f"Online VAE latent T={int(latents.shape[2])} does not match "
                f"target pixel frames={int(media.shape[2])} "
                f"(expected latent T={expected_online_latent_t})"
            )
        if rank == 0 and not getattr(prepare_model_inputs, "_online_vae_output_printed", False):
            prepare_model_inputs._online_vae_output_printed = True
            print(
                "[ONLINE VAE OUTPUT] "
                f"latent_shape_bcthw={tuple(latents.shape)} "
                f"target_pixel_frames={int(media.shape[2])} "
                f"generated_latent_T={int(latents.shape[2])}"
            )
    elif len(latents.shape) == 5 or len(latents.shape) == 4:  # Using video/image cache
        latents = latents * vae.config.scaling_factor  # vae cache is not multiplied by scaling_factor
    else:
        raise ValueError(
            f"Only support media/latent with shape (b, c, h, w) or (b, c, f, h, w), but got {media.shape} {latents.shape}.")
        
    # to remove for next version
    # reorg_token = '888' in args.model_type or '16168' in args.model_type
    reorg_token = False
    if reorg_token:
        latents = torch.cat((latents[:, :, :1], latents), dim=2)
        latents = rearrange(latents, 'b c f h w -> b f c h w')
        latents = rearrange(latents, 'b (f n) c h w -> b f (n c) h w', n=2)
        latents = rearrange(latents, 'b f c h w -> b c f h w').contiguous()

    # ============================== Get multitask mask and mask type ===========================
    multitask_mask = None
    cond_latents = None
    semantic_images_np = None
    semantic_images_pil = None
    last_frame_cond_latents = None
    is_uncond = False
    mask_type = "t2v"
    vision_states = None
    use_i2v_instruction = False
    if args.multitask_mask_training_type == "concat" or args.multitask_mask_training_type == "token_replace":
        assert args.multitask_mask_training_args is not None, f"concat must provide args.multitask_mask_training_args, " \
                                                              f"but found args.multitask_mask_training_args: " \
                                                              f"{args.multitask_mask_training_args}"

        # --- SP-safe multitask sampling via local RNG ---
        # get_multitask_mask uses random.random() internally, which under SP
        # would desync across ranks. We broadcast a seed from SP rank 0,
        # construct a local random.Random(seed), and monkey-patch the global
        # random module during the call so get_multitask_mask draws from it.
        # The global random state is never modified.
        from hymm.utils.parallel_states import nccl_info
        _sp_size = nccl_info.sp_size if nccl_info.sp_size is not None else 1
        if _sp_size > 1 and nccl_info.sp_group is not None:
            _sp_local_rank = nccl_info.rank_within_spgroup if hasattr(nccl_info, 'rank_within_spgroup') else 0
            # Use torch RNG (not Python random) to generate seed — avoids
            # polluting global random state on any rank.
            # All SP ranks consume one torch RNG draw so their torch RNG states
            # do not diverge; rank 0's value is then used as the shared seed.
            _mt_seed_t = torch.randint(0, 2**31, (1,), dtype=torch.int64, device=device)
            dist.broadcast(_mt_seed_t, src=dist.get_global_rank(nccl_info.sp_group, 0),
                           group=nccl_info.sp_group)
            # Local RNG: all SP ranks deterministically produce the same sequence
            _sp_rng = random.Random(_mt_seed_t.item())
            # Monkey-patch random.random during get_multitask_mask call
            _orig_random = random.random
            try:
                random.random = _sp_rng.random
                multitask_mask, mask_type = get_multitask_mask(latents.shape[2], args.multitask_mask_training_args)
            finally:
                random.random = _orig_random
            # Broadcast the final mask as well. Most mask types are deterministic
            # after mask_type sampling, but mask_type="random" uses torch.randint
            # inside get_multitask_mask and must be synchronized explicitly.
            multitask_mask = multitask_mask.to(device)
            dist.broadcast(multitask_mask, src=dist.get_global_rank(nccl_info.sp_group, 0),
                           group=nccl_info.sp_group)
        else:
            _sp_rng = None  # no SP — use global random normally
            multitask_mask, mask_type = get_multitask_mask(latents.shape[2], args.multitask_mask_training_args)

        if mask_type == "i2v":
            if args.video_caption_i2v_instruction_enable:
                if random.random() < args.video_caption_i2v_instruction_prob:
                    text_ids = text_ids_i2v_instruction
                    text_mask = text_mask_i2v_instruction
                    use_i2v_instruction = True

            if args.multitask_semantic_mode is not None and args.multitask_semantic_mode != "none":
                if args.use_multitask_co_drop_semantic_and_text:
                    if 'text' in kwargs:
                        prompt = kwargs["text"][0]
                        if prompt == "":
                            is_uncond = True
                        else:
                            is_uncond = True if random.random() < args.multitask_semantic_drop_independent_p else False
                else:
                    is_uncond = True if random.random() < args.multitask_semantic_drop_independent_p else False

            cond_latents, _, semantic_images_np, semantic_images_pil = get_cond_latents(args, latents, vae, reorg_token=reorg_token, is_uncond=is_uncond)
            if args.multitask_mask_training_type == "token_replace" and mask_type == "i2v" and reorg_token: # for reorg_token, we need to repeat cond_latents
                cond_latents = cond_latents.repeat(1, 2, 1, 1, 1)

        elif mask_type == "fl2v":
            cond_latents, last_frame_cond_latents, semantic_images_np, semantic_images_pil = get_cond_latents(args, latents, vae,
                                                                                         last_frame_tensor=kwargs['last_frame_tensor'],
                                                                                         reorg_token=reorg_token, is_uncond=is_uncond)

        vision_states = torch.zeros(latents.shape[0], args.vision_num_semantic_tokens, args.vision_states_dim).to(device)
        if args.multitask_semantic_mode == "siglip" and semantic_images_np is not None:
            vision_encoder = extra_model["vision_encoder"]
            image_encoder_output = vision_encoder.encode_images(semantic_images_np)
            vision_states = image_encoder_output.last_hidden_state
            assert vision_states.shape[1] == args.vision_num_semantic_tokens, f"vision_states.shape[1] should be {args.vision_num_semantic_tokens}, but got {vision_states.shape[1]}"
    else:
        assert args.multitask_mask_training_type in [None, "none"] , f"current only support None and concat type, " \
                                                                     f"but found args.multitask_mask_training_type: " \
                                                                     f"{args.multitask_mask_training_type}"
        assert args.multitask_semantic_mode in [None, "none"], f"when args.multitask_mask_training_type is None " \
                                                                     f"args.multitask_semantic_mode must be None, but get " \
                                                                     f"{args.multitask_semantic_mode}"

    # ======================================== Encode text ======================================
    # Autocast is handled by text_encoder itself.
    # Whether to apply text_mask is determined by args.use_attention_mask.
    text_outputs = text_encoder.encode({"input_ids": text_ids, "attention_mask": text_mask}, data_type=kwargs['type'][0],
                                       semantic_images=semantic_images_pil, is_uncond=is_uncond)
    text_states = text_outputs.hidden_state
    text_mask = text_outputs.attention_mask
    text_states_2 = (
        text_encoder_2.encode({"input_ids": text_ids_2, "attention_mask": text_mask_2}).hidden_state
        if text_encoder_2 is not None else None
    )

    # ======================================== Build RoPE ======================================
    # Import here to avoid circular import
    from hymm.models.modules.posemb_layers import get_nd_rotary_pos_embed
    
    target_ndim = 3     # n-d RoPE
    ndim = len(latents.shape) - 2
    latents_size = list(latents.shape[-ndim:])
    assert all(s % model.patch_size[i] == 0 for i, s in enumerate(latents_size)), \
        f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), " \
        f"but got {latents_size}."
    rope_sizes = [s // model.patch_size[i] for i, s in enumerate(latents_size)]
    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = model.hidden_size // model.heads_num
    rope_dim_list = model.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list=rope_dim_list,
        start=rope_sizes,
        theta=args.rope_theta,
        use_real=True,
        theta_rescale_factor=1.0,
        use_scale_rope=args.use_scale_rope
    )
    model_kwargs = dict(
        text_states=text_states,        # [b, 256, 4096]
        text_states_2=text_states_2,    # [b, 768]
        encoder_attention_mask=text_mask,            # [b, 256]
        vision_states=vision_states,                # [b, 729, 1152]
        mask_type=mask_type,
        return_dict=False
    )

    # ========================================== byT5 ===========================================
    model_kwargs["extra_kwargs"] = {}
    if byt5_text_ids is not None:
        byt5_model = byt5_kwarg["byt5_model"]
        byt5_prompt_embeds = byt5_model(byt5_text_ids, attention_mask=byt5_text_mask.float())
        byt5 = byt5_prompt_embeds[0]

        model_kwargs["extra_kwargs"]["byt5_text_states"] = byt5
        model_kwargs["extra_kwargs"]["byt5_text_mask"] = byt5_text_mask

    # ========================================== Recreation Mask ==========================================
    # Extract recreation mask from kwargs and add to model_kwargs
    recreation_mask = kwargs.get('recreation_mask', None)
    model_kwargs["extra_kwargs"]["enable_recreation_mask"] = False

    # skip if recreation_mask is None(i.e. T2I task)
    if hasattr(args, 'enable_recreation_mask') and args.enable_recreation_mask and recreation_mask is not None:
        if not isinstance(recreation_mask, torch.Tensor):
            recreation_mask = torch.tensor(recreation_mask)
        recreation_mask = recreation_mask.to(device)
        model_kwargs["extra_kwargs"]["enable_recreation_mask"] = True
        model_kwargs["extra_kwargs"]["recreation_mask"] = recreation_mask
    model_kwargs["extra_kwargs"]["part_version"] = part_version

    # ========================================== PhysRoPE ==========================================
    if getattr(args, 'use_physical_rope', False):
        model_kwargs["extra_kwargs"]["use_physical_rope"] = True
        model_kwargs["extra_kwargs"]["physical_rope_theta"] = getattr(args, 'physical_rope_theta', 256.0)
        model_kwargs["extra_kwargs"]["physical_rope_scale"] = getattr(args, 'physical_rope_scale', 0.1)
        model_kwargs["extra_kwargs"]["physical_rope_dim"] = getattr(args, 'physical_rope_dim', 32)
        model_kwargs["extra_kwargs"]["disable_text_physical_rope"] = getattr(args, 'disable_text_physical_rope', False)
        model_kwargs["extra_kwargs"]["physical_rope_time_range_mode"] = getattr(args, 'physical_rope_time_range_mode', 'second')
        model_kwargs["extra_kwargs"]["video_fps"] = getattr(args, 'video_fps', 24.0)
        # PhysRoPE v2 parameters (alpha != None activates v2)
        _alpha = getattr(args, 'physical_rope_alpha', None)
        if _alpha is not None:
            model_kwargs["extra_kwargs"]["physical_rope_alpha"] = float(_alpha)
            model_kwargs["extra_kwargs"]["physical_rope_offset_mode"] = getattr(args, 'physical_rope_offset_mode', 'physical')
            model_kwargs["extra_kwargs"]["physical_rope_cut"] = getattr(args, 'physical_rope_cut', False)
            model_kwargs["extra_kwargs"]["physical_rope_speed_aware"] = getattr(args, 'physical_rope_speed_aware', False)
            model_kwargs["extra_kwargs"]["physical_rope_speed_mode"] = getattr(args, 'physical_rope_speed_mode', 'uniform')
            model_kwargs["extra_kwargs"]["physical_rope_v2_dim"] = getattr(args, 'physical_rope_v2_dim', 0)
            model_kwargs["extra_kwargs"]["physrope_route_debug_limit"] = getattr(
                args, "physrope_route_debug_limit", 2
            )
            model_kwargs["extra_kwargs"]["reference_timeline_debug_limit"] = getattr(
                args, "reference_timeline_debug_limit", 0
            )
            model_kwargs["extra_kwargs"]["range_condition_temporal_pack_mode"] = getattr(
                args, "range_condition_temporal_pack_mode", "legacy_trilinear"
            )
        if not getattr(prepare_model_inputs, '_physrope_printed', False) and str(os.environ.get("RANK", "0")) == "0":
            prepare_model_inputs._physrope_printed = True
            if _alpha is not None:
                print(f"[PhysRoPE TRAIN] v2 active: alpha={_alpha} | "
                      f"offset_mode={model_kwargs['extra_kwargs'].get('physical_rope_offset_mode', 'physical')} | "
                      f"cut={model_kwargs['extra_kwargs'].get('physical_rope_cut', False)} | "
                      f"v2_dim={model_kwargs['extra_kwargs'].get('physical_rope_v2_dim', 0)} | "
                      f"route_debug_limit={model_kwargs['extra_kwargs'].get('physrope_route_debug_limit', 2)} | "
                      f"reference_timeline_debug_limit={model_kwargs['extra_kwargs'].get('reference_timeline_debug_limit', 0)} | "
                      f"mode={model_kwargs['extra_kwargs']['physical_rope_time_range_mode']} | "
                      f"fps={model_kwargs['extra_kwargs']['video_fps']}")
            else:
                print(f"[PhysRoPE TRAIN] v1 active: "
                      f"theta={model_kwargs['extra_kwargs']['physical_rope_theta']} | "
                      f"scale={model_kwargs['extra_kwargs']['physical_rope_scale']} | "
                      f"dim={model_kwargs['extra_kwargs']['physical_rope_dim']} | "
                      f"disable_text={model_kwargs['extra_kwargs']['disable_text_physical_rope']} | "
                      f"mode={model_kwargs['extra_kwargs']['physical_rope_time_range_mode']} | "
                      f"fps={model_kwargs['extra_kwargs']['video_fps']}")
        # Pass segment data from dataloader kwargs
        text_segment_ids = kwargs.get('text_segment_ids', None)
        text_segment_time_ranges = kwargs.get('text_segment_time_ranges', None)
        if text_segment_ids is not None:
            if isinstance(text_segment_ids, torch.Tensor):
                if not torch.all(text_segment_ids < 0):
                    model_kwargs["extra_kwargs"]["text_segment_ids"] = text_segment_ids
            else:
                model_kwargs["extra_kwargs"]["text_segment_ids"] = text_segment_ids
        if text_segment_time_ranges is not None:
            if isinstance(text_segment_time_ranges, str):
                try:
                    text_segment_time_ranges = json.loads(text_segment_time_ranges)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"text_segment_time_ranges JSON parse failed: {text_segment_time_ranges[:200]}"
                    ) from exc
            elif isinstance(text_segment_time_ranges, (list, tuple)) and text_segment_time_ranges:
                if isinstance(text_segment_time_ranges[0], str):
                    parsed_rows = []
                    for item in text_segment_time_ranges:
                        try:
                            parsed_rows.append(json.loads(item))
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"text_segment_time_ranges row JSON parse failed: {str(item)[:200]}"
                            ) from exc
                    text_segment_time_ranges = parsed_rows
            if isinstance(text_segment_time_ranges, (list, tuple)):
                has_any_segment_time = any(bool(row) for row in text_segment_time_ranges)
            else:
                has_any_segment_time = text_segment_time_ranges is not None
            if has_any_segment_time:
                model_kwargs["extra_kwargs"]["text_segment_time_ranges"] = text_segment_time_ranges
                if not getattr(prepare_model_inputs, '_physrope_seg_printed', False) and str(os.environ.get("RANK", "0")) == "0":
                    prepare_model_inputs._physrope_seg_printed = True
                    _seg_ids_present = "text_segment_ids" in model_kwargs["extra_kwargs"]
                    _n_ranges = len(text_segment_time_ranges) if isinstance(text_segment_time_ranges, (list, tuple)) else "?"
                    print(f"[PhysRoPE TRAIN] (helpers.py:L546) segment data: "
                          f"has_segment_ids={_seg_ids_present} | n_time_ranges={_n_ranges} | "
                          f"text_ids.shape={text_ids.shape}")

    # ========================================== REPA ===========================================
    repa_feats = None
    if getattr(args, "use_repa", False):
        repa_encoder = None if extra_model is None else extra_model.get("repa_encoder", None)
        if repa_encoder is not None:
            if args.repa_online_decode_video and latents.shape[2] > 1:
                decode_image = decode_latents_to_pixels(
                    latents,
                    vae,
                    enable_tiling=True,
                    tile_size=args.vae_encode_tile_size,
                )
                clean_pixels = decode_image
            else:
                clean_pixels = media
                if clean_pixels.dim() == 5:
                    clean_pixels = clean_pixels[:, :, 0]
                elif clean_pixels.dim() == 4:
                    clean_pixels = clean_pixels
                else:
                    raise ValueError(f"Unsupported media shape for REPA encoder: {clean_pixels.shape}")
            # map back to [0, 1] for vision encoders
            clean_pixels = (clean_pixels / 2 + 0.5).clamp(0, 1)
            # current_step = getattr(args, "global_step", 0) 
            # if current_step % 5 == 0: # 每100步保存一次
            #     save_debug_sample(clean_pixels, step=current_step)
            with torch.no_grad():
                repa_out = repa_encoder.encode_images(clean_pixels, texts=kwargs.get("text"), data_type=args.data_type)
                repa_feats = (
                    repa_out.last_hidden_state
                    if hasattr(repa_out, "last_hidden_state")
                    else repa_out
                )

    if latents.shape[2] == 1:
        data_type = 'image'
    else:
        data_type = 'video'

    if 'index' in kwargs:
        index = kwargs['index']
    else:
        index = None
    if 'text' in kwargs:
        text = kwargs['text']
    else:
        text = None

    # =============================== Unified PhysRoPE Training ================================
    # Two activation paths:
    #   (A) mask_type == "range_condition" from multitask sampler (NEW: direct probability)
    #   (B) Legacy: unified_physrope_enable flips t2v→range_condition (backward compat)
    # Both paths construct multi-segment condition frame ranges via SP-safe broadcast.
    unified_physrope_enable = getattr(args, "unified_physrope_enable", False)
    if isinstance(unified_physrope_enable, str):
        unified_physrope_enable = unified_physrope_enable.strip().lower() in ("true", "on", "1", "yes")
    unified_physrope_condition_prob = float(getattr(args, "unified_physrope_condition_prob", 0.0))
    range_condition_mode_cfg = args.range_condition_mode
    assert range_condition_mode_cfg in ("soft_reference", "hard_legacy"), (
        f"range_condition_mode must be soft_reference or hard_legacy, got {range_condition_mode_cfg}"
    )
    native_t_concat = as_bool(getattr(args, "range_condition_native_t_concat", False))
    temporal_pack_mode = normalize_temporal_pack_mode(
        getattr(args, "range_condition_temporal_pack_mode", "legacy_trilinear")
    )
    reference_timeline = temporal_pack_mode == REFERENCE_TIMELINE
    if reference_timeline:
        assert not native_t_concat, (
            "reference_timeline uses the fixed target-time channel-concat representation "
            "and cannot be combined with native temporal concat"
        )
    if native_t_concat:
        assert range_condition_mode_cfg == "soft_reference", (
            "range_condition_native_t_concat is a soft-reference representation and "
            "cannot be combined with hard_legacy pinning"
        )
    range_condition_debug = False  # Enable only for short bring-up runs.
    if range_condition_debug and not getattr(prepare_model_inputs, '_range_condition_config_printed', False) and rank == 0:
        prepare_model_inputs._range_condition_config_printed = True
        print(
            f"[RangeCondition CONFIG] enabled={unified_physrope_enable} "
            f"mode={range_condition_mode_cfg} "
            f"cond_loss_weight={args.unified_physrope_cond_loss_weight} "
            f"multitask_type={args.multitask_mask_training_type} "
            f"multitask_args={args.multitask_mask_training_args} "
            f"per_segment_enable={args.per_segment_condition_enable} "
            f"per_segment_prob={args.per_segment_condition_prob} "
            f"condition_dropout_prob={args.condition_dropout_prob} "
            f"l4_self_resample_prob={args.unified_physrope_temporal_resample_prob} "
            f"l4_self_resample_range={args.unified_physrope_temporal_resample_range}",
            flush=True,
        )

    # Path (C): per-segment data-driven condition (TIV2V) — takes precedence
    # over both the random-span path (A) and the legacy t2v→range_condition
    # flip (B). When the dataset emits at least one active segment, route
    # straight into range_condition with the loaded edited latents and
    # disjoint dst_range slots; bypass the random-span generator entirely.
    cond_seg_count_b = kwargs.get('cond_seg_count', None)
    cond_seg_dst_ranges_b = kwargs.get('cond_seg_dst_ranges', None)
    cond_seg_physical_dst_ranges_b = kwargs.get('cond_seg_physical_dst_ranges', None)
    cond_seg_src_ranges_b = kwargs.get('cond_seg_src_ranges', None)
    cond_seg_latents_packed_b = kwargs.get('cond_seg_latents_packed', None)
    cond_seg_native_latents_b = kwargs.get('cond_seg_native_latents', None)
    cond_seg_native_ranges_b = kwargs.get('cond_seg_native_ranges', None)
    cond_seg_mask_b = kwargs.get('cond_seg_mask', None)
    cond_seg_kind_b = kwargs.get('cond_seg_kind', None)
    cond_seg_prompt_ids_b = kwargs.get('cond_seg_prompt_ids', None)
    rank_dbg = dist.get_rank() if dist.is_initialized() else 0
    _per_segment_active = (
        cond_seg_count_b is not None
        and isinstance(cond_seg_count_b, torch.Tensor)
        and cond_seg_count_b.numel() > 0
        and bool((cond_seg_count_b > 0).any().item())
    )
    if _per_segment_active and latents.shape[2] > 4:
        from hymm.utils.parallel_states import nccl_info

        assert isinstance(cond_seg_latents_packed_b, torch.Tensor), (
            "cond_seg_count indicates active per-segment conditions, but "
            "cond_seg_latents_packed is missing or not a tensor"
        )
        assert isinstance(cond_seg_dst_ranges_b, torch.Tensor), (
            "cond_seg_count indicates active per-segment conditions, but "
            "cond_seg_dst_ranges is missing or not a tensor"
        )
        if reference_timeline:
            assert isinstance(cond_seg_physical_dst_ranges_b, torch.Tensor), (
                "reference_timeline requires cond_seg_physical_dst_ranges from the dataset"
            )
            assert cond_seg_physical_dst_ranges_b.shape == cond_seg_dst_ranges_b.shape, (
                f"physical dst ranges {tuple(cond_seg_physical_dst_ranges_b.shape)} "
                f"must match visible dst ranges {tuple(cond_seg_dst_ranges_b.shape)}"
            )
        assert isinstance(cond_seg_mask_b, torch.Tensor), (
            "cond_seg_count indicates active per-segment conditions, but "
            "cond_seg_mask is missing or not a tensor"
        )
        assert isinstance(cond_seg_kind_b, torch.Tensor), (
            "cond_seg_count indicates active per-segment conditions, but "
            "cond_seg_kind is missing or not a tensor"
        )
        assert isinstance(cond_seg_prompt_ids_b, torch.Tensor), (
            "per-segment conditions require cond_seg_prompt_ids from the dataset"
        )
        # Per-rank batch is 1 in current configs; multitask_mask in transport.py
        # is shape [T] (batch-shared), so we use sample-0's mask. Assert that
        # the rest of the batch has identical layout if batch > 1.
        b = cond_seg_latents_packed_b.shape[0]
        assert cond_seg_latents_packed_b.ndim == 6, (
            f"cond_seg_latents_packed must be [B,S,C,T,H,W], got {tuple(cond_seg_latents_packed_b.shape)}"
        )
        assert cond_seg_dst_ranges_b.ndim == 3 and cond_seg_dst_ranges_b.shape[:2] == cond_seg_latents_packed_b.shape[:2] and cond_seg_dst_ranges_b.shape[2] == 2, (
            f"cond_seg_dst_ranges must be [B,S,2] matching packed latents, "
            f"got ranges={tuple(cond_seg_dst_ranges_b.shape)} latents={tuple(cond_seg_latents_packed_b.shape)}"
        )
        if cond_seg_src_ranges_b is not None:
            assert isinstance(cond_seg_src_ranges_b, torch.Tensor), (
                f"cond_seg_src_ranges must be a tensor when provided, got {type(cond_seg_src_ranges_b)}"
            )
            assert cond_seg_src_ranges_b.shape == cond_seg_dst_ranges_b.shape, (
                f"cond_seg_src_ranges shape {tuple(cond_seg_src_ranges_b.shape)} "
                f"must match cond_seg_dst_ranges {tuple(cond_seg_dst_ranges_b.shape)}"
            )
        if native_t_concat:
            assert b == 1, (
                "range_condition_native_t_concat currently requires video micro-batch size 1 "
                f"because native reference T varies per sample, got batch={b}"
            )
            assert isinstance(cond_seg_native_latents_b, torch.Tensor), (
                "native T concat requires cond_seg_native_latents from the dataset"
            )
            assert isinstance(cond_seg_native_ranges_b, torch.Tensor), (
                "native T concat requires cond_seg_native_ranges from the dataset"
            )
            assert cond_seg_native_latents_b.ndim == 5, (
                "cond_seg_native_latents must be [B,C,T_ref,H,W], got "
                f"{tuple(cond_seg_native_latents_b.shape)}"
            )
            assert cond_seg_native_latents_b.shape[0] == b, (
                f"native reference batch {cond_seg_native_latents_b.shape[0]} must match {b}"
            )
            assert cond_seg_native_latents_b.shape[1] == latents.shape[1] and cond_seg_native_latents_b.shape[-2:] == latents.shape[-2:], (
                "native reference C/H/W must match target latents: "
                f"reference={tuple(cond_seg_native_latents_b.shape)} target={tuple(latents.shape)}"
            )
            assert cond_seg_native_ranges_b.shape == cond_seg_dst_ranges_b.shape, (
                f"cond_seg_native_ranges shape {tuple(cond_seg_native_ranges_b.shape)} "
                f"must match dst ranges {tuple(cond_seg_dst_ranges_b.shape)}"
            )
        else:
            assert cond_seg_latents_packed_b.shape[2:] == latents.shape[1:], (
                f"cond_seg_latents_packed tail {tuple(cond_seg_latents_packed_b.shape[2:])} "
                f"must match latents tail {tuple(latents.shape[1:])}"
            )
        assert cond_seg_mask_b.ndim == 3 and cond_seg_mask_b.shape[0] == b and cond_seg_mask_b.shape[-1] == latents.shape[2], (
            f"cond_seg_mask must be [B,S,T={latents.shape[2]}], got {tuple(cond_seg_mask_b.shape)}"
        )
        assert cond_seg_mask_b.shape[1] == cond_seg_latents_packed_b.shape[1], (
            f"cond_seg_mask S={cond_seg_mask_b.shape[1]} must match packed S={cond_seg_latents_packed_b.shape[1]}"
        )
        assert cond_seg_kind_b.ndim == 2 and cond_seg_kind_b.shape == cond_seg_latents_packed_b.shape[:2], (
            f"cond_seg_kind must be [B,S] matching packed latents, "
            f"got kind={tuple(cond_seg_kind_b.shape)} latents={tuple(cond_seg_latents_packed_b.shape)}"
        )
        assert (
            cond_seg_prompt_ids_b.ndim == 2
            and cond_seg_prompt_ids_b.shape == cond_seg_latents_packed_b.shape[:2]
        ), (
            f"cond_seg_prompt_ids must be [B,S] matching packed latents, "
            f"got ids={tuple(cond_seg_prompt_ids_b.shape)} "
            f"latents={tuple(cond_seg_latents_packed_b.shape)}"
        )
        if b > 1:
            ref_dst_ranges = cond_seg_dst_ranges_b[0:1].expand_as(cond_seg_dst_ranges_b)
            assert torch.equal(cond_seg_dst_ranges_b, ref_dst_ranges), (
                "per-segment condition expects all samples in a batch to share "
                "the same active dst_range layout (transport.py uses one [T] mask)."
            )
            if cond_seg_src_ranges_b is not None:
                ref_src_ranges = cond_seg_src_ranges_b[0:1].expand_as(cond_seg_src_ranges_b)
                assert torch.equal(cond_seg_src_ranges_b, ref_src_ranges), (
                    "per-segment condition expects all samples in a batch to share "
                    "the same src_range layout for a single condition_src_frame_ranges list."
                )
            ref_mask_union_b = cond_seg_mask_b.sum(dim=1).clamp(max=1.0)  # [B, T]
            ref_first = ref_mask_union_b[0:1].expand_as(ref_mask_union_b)
            assert torch.equal(ref_mask_union_b, ref_first), (
                "per-segment condition expects all samples in a batch to share "
                "the same active reference_mask layout."
            )
            ref_kind = cond_seg_kind_b[0:1].expand_as(cond_seg_kind_b)
            assert torch.equal(cond_seg_kind_b, ref_kind), (
                "per-segment condition expects all samples in a batch to share "
                "the same condition kind layout for debug/task routing."
            )
            ref_prompt_ids = cond_seg_prompt_ids_b[0:1].expand_as(
                cond_seg_prompt_ids_b
            )
            assert torch.equal(cond_seg_prompt_ids_b, ref_prompt_ids), (
                "per-segment condition expects all samples in a batch to share "
                "the same prompt-shot identity layout."
            )

        def _collect_active_slots(slot_ids=None):
            dst_list = cond_seg_dst_ranges_b[0].tolist()
            physical_dst_list = (
                cond_seg_physical_dst_ranges_b[0].tolist()
                if cond_seg_physical_dst_ranges_b is not None
                else dst_list
            )
            src_list = (cond_seg_src_ranges_b[0].tolist()
                        if cond_seg_src_ranges_b is not None
                        else [(-1, -1)] * len(dst_list))
            kind_list = cond_seg_kind_b[0].tolist()
            prompt_seg_id_list = cond_seg_prompt_ids_b[0].tolist()
            native_list = (cond_seg_native_ranges_b[0].tolist()
                           if native_t_concat else [(-1, -1)] * len(dst_list))
            iter_slot_ids = range(len(dst_list)) if slot_ids is None else slot_ids
            slots = []
            for slot_idx in iter_slot_ids:
                ds, de = dst_list[int(slot_idx)]
                pds, pde = physical_dst_list[int(slot_idx)]
                ss, se = src_list[int(slot_idx)]
                kind = int(kind_list[int(slot_idx)])
                native_start, native_end = native_list[int(slot_idx)]
                if ds >= 0 and de > ds:
                    slots.append({
                        "slot": int(slot_idx),
                        "dst": (int(ds), int(de)),
                        "physical_dst": (
                            (int(pds), int(pde))
                            if pds >= 0 and pde > pds
                            else (int(ds), int(de))
                        ),
                        "src": (int(ss), int(se)) if ss >= 0 and se > ss else (int(ds), int(de)),
                        "kind": kind,
                        "prompt_seg_id": int(prompt_seg_id_list[int(slot_idx)]),
                        "length": int(de - ds),
                        "native": (int(native_start), int(native_end)),
                    })
            slots.sort(key=lambda item: item["dst"][0])
            return slots

        active_slots = _collect_active_slots()

        if active_slots:
            keep_slot_ids = [item["slot"] for item in active_slots]
            dropout_before_slot_ids = list(keep_slot_ids)
            dropped_slot_ids = []
            dropout_applied = False

            # ─── Condition Dropout (future_plan §1.4c) ───
            # After determining which slots are active, randomly drop some
            # to train the model's robustness to partial condition specification.
            # This is distinct from per_segment_condition_prob (which controls
            # whether a segment is loaded at all): dropout happens AFTER loading.
            cond_dropout_prob = float(getattr(args, "condition_dropout_prob", 0.0))
            if cond_dropout_prob > 0.0 and len(keep_slot_ids) > 1:
                # SP-safe: broadcast dropout decision within SP group (NOT global!)
                # Using global broadcast (src=0, default group) would deadlock because
                # not all ranks enter this branch (some have no conditions).
                # SP group ranks always process the same data, so SP-local broadcast is safe.
                _dropout_seed_t = torch.randint(0, 2**31, (1,), dtype=torch.int64, device=device)
                if dist.is_initialized() and nccl_info.sp_group is not None:
                    dist.broadcast(_dropout_seed_t,
                                   src=dist.get_global_rank(nccl_info.sp_group, 0),
                                   group=nccl_info.sp_group)
                _dropout_rng = random.Random(_dropout_seed_t.item())
                survived_slots = []
                for slot_id in keep_slot_ids:
                    if _dropout_rng.random() >= cond_dropout_prob:
                        survived_slots.append(slot_id)
                # Always keep at least one condition (never drop all)
                if not survived_slots:
                    survived_slots = [keep_slot_ids[_dropout_rng.randint(0, len(keep_slot_ids) - 1)]]
                if range_condition_debug and len(survived_slots) < len(keep_slot_ids) and rank_dbg == 0:
                    print(f"[Condition Dropout] Dropped {len(keep_slot_ids) - len(survived_slots)}"
                          f"/{len(keep_slot_ids)} slots (prob={cond_dropout_prob})", flush=True)
                dropped_slot_ids = [slot_id for slot_id in keep_slot_ids if slot_id not in set(survived_slots)]
                dropout_applied = True
                keep_slot_ids = survived_slots
                active_slots = [s for s in active_slots if s["slot"] in set(keep_slot_ids)]

            # L4: Apply temporal resample to selected per-segment conditions
            # Only apply if enabled and we have multiple segments or explicit resample config
            temporal_resample_prob = float(getattr(args, "unified_physrope_temporal_resample_prob", 0.0))
            if reference_timeline and temporal_resample_prob != 0.0:
                raise ValueError(
                    "range_condition_temporal_pack_mode=reference_timeline forbids both "
                    "loader interpolation and unified temporal self-resampling"
                )
            if native_t_concat and temporal_resample_prob > 0.0:
                raise ValueError(
                    "unified_physrope_temporal_resample_prob must be 0 when "
                    "range_condition_native_t_concat is enabled; native references are never resampled"
                )
            if temporal_resample_prob > 0 and len(active_slots) > 0:
                from hymm.utils.multitask_utils import temporal_resample_condition_segments

                # Generate a seed for this batch using torch RNG (SP-safe).
                _l4_seed_t = torch.randint(0, 2**31, (1,), dtype=torch.int64, device=device)
                if dist.is_initialized() and nccl_info.sp_group is not None:
                    # Broadcast seed within SP group (NOT global!) to avoid deadlock
                    # when not all DP ranks enter this branch.
                    dist.broadcast(_l4_seed_t,
                                   src=dist.get_global_rank(nccl_info.sp_group, 0),
                                   group=nccl_info.sp_group)
                _l4_seed = int(_l4_seed_t.item())

                temporal_resample_range = _parse_float_pair(
                    getattr(args, "unified_physrope_temporal_resample_range", (0.8, 1.25)),
                    "unified_physrope_temporal_resample_range",
                )
                cond_seg_latents_packed_b, cond_seg_dst_ranges_b = temporal_resample_condition_segments(
                    cond_seg_latents_packed_b,
                    cond_seg_dst_ranges_b,
                    cond_seg_src_ranges_b if cond_seg_src_ranges_b is not None else cond_seg_dst_ranges_b,
                    cond_seg_mask_b,
                    resample_prob=temporal_resample_prob,
                    resample_range=temporal_resample_range,
                    seed=_l4_seed,
                )
                # Ranges and masks may have changed; rebuild active slots from
                # tensors so condition_frame_ranges matches cond_latents exactly.
                active_slots = _collect_active_slots(keep_slot_ids)
                assert active_slots, "L4 temporal resample removed all active slots"
            if native_t_concat or reference_timeline:
                union_mask = cond_seg_mask_b[0, keep_slot_ids].sum(dim=0).clamp(max=1.0)
                assert int(union_mask.sum().item()) > 0, (
                    "range conditions must contain at least one mapped reference frame"
                )
            else:
                from hymm.utils.multitask_utils import ensure_condition_segments_have_generation_frame
                keep_slot_ids = ensure_condition_segments_have_generation_frame(
                    cond_seg_latents_packed_b,
                    cond_seg_dst_ranges_b,
                    cond_seg_src_ranges_b,
                    cond_seg_mask_b,
                    active_slot_ids=keep_slot_ids,
                    cond_seg_kind=cond_seg_kind_b,
                )
            active_slots = _collect_active_slots(keep_slot_ids)
            assert active_slots, "range_condition adjustment removed all active slots"
            native_reference_packed_ranges = []
            if native_t_concat:
                native_chunks = []
                native_cursor = 0
                for item in active_slots:
                    native_start, native_end = item["native"]
                    assert 0 <= native_start < native_end <= cond_seg_native_latents_b.shape[2], (
                        f"invalid native range {(native_start, native_end)} for "
                        f"T_ref={cond_seg_native_latents_b.shape[2]} slot={item['slot']}"
                    )
                    chunk = cond_seg_native_latents_b[:, :, native_start:native_end]
                    native_chunks.append(chunk)
                    native_reference_packed_ranges.append(
                        (native_cursor, native_cursor + int(chunk.shape[2]))
                    )
                    native_cursor += int(chunk.shape[2])
                cond_latents = torch.cat(native_chunks, dim=2).to(
                    device=device, dtype=latents.dtype
                )
            else:
                # Sum over the kept seg slots → merged [B, C, T, H, W].
                cond_latents = cond_seg_latents_packed_b[:, keep_slot_ids].sum(dim=1).to(device=device, dtype=latents.dtype)
            # Re-apply VAE scaling because the merge happens after the
            # `latents *= scaling_factor` step above. Per-segment latents are
            # stored unscaled (latent_scale="unscaled_vae_mode").
            cond_latents = cond_latents * vae.config.scaling_factor
            multitask_mask = cond_seg_mask_b[0, keep_slot_ids].sum(dim=0).clamp(max=1.0).to(device=device, dtype=torch.float32)
            cond_frame_count = int(multitask_mask.sum().item())
            assert 0 < cond_frame_count and (
                native_t_concat or reference_timeline or cond_frame_count < latents.shape[2]
            ), (
                f"invalid per-segment range_condition reference mask: "
                f"cond_frames={cond_frame_count}, T={latents.shape[2]}, slots={keep_slot_ids}"
            )
            condition_frame_ranges = [item["dst"] for item in active_slots]
            condition_physical_dst_frame_ranges = [
                item["physical_dst"] for item in active_slots
            ]
            condition_src_frame_ranges = [item["src"] for item in active_slots]
            condition_kinds = [int(item["kind"]) for item in active_slots]
            condition_seg_ids = [
                int(item["prompt_seg_id"]) for item in active_slots
            ]
            has_video_condition = any(int(item["kind"]) == 1 for item in active_slots)
            has_anchor_condition = any(int(item["kind"]) == 2 for item in active_slots)
            if has_video_condition and has_anchor_condition:
                debug_task_mode = "tiv2v"
            elif has_video_condition:
                debug_task_mode = "v2v"
            elif has_anchor_condition:
                debug_task_mode = "ti2v"
            else:
                debug_task_mode = "range_condition"
            mask_type = "range_condition"
            model_kwargs["mask_type"] = mask_type
            model_kwargs["extra_kwargs"]["condition_frame_ranges"] = condition_frame_ranges
            # Preserve source-range provenance for ReferenceTimeline selection and
            # logging. Source length never changes the target PhysRoPE clock.
            model_kwargs["extra_kwargs"]["condition_src_frame_ranges"] = condition_src_frame_ranges
            model_kwargs["extra_kwargs"]["condition_physical_dst_frame_ranges"] = (
                condition_physical_dst_frame_ranges
            )
            model_kwargs["extra_kwargs"]["condition_kinds"] = condition_kinds
            model_kwargs["extra_kwargs"]["condition_seg_ids"] = condition_seg_ids
            model_kwargs["extra_kwargs"]["range_condition_temporal_pack_mode"] = (
                temporal_pack_mode
            )
            if native_t_concat:
                model_kwargs["extra_kwargs"]["range_condition_native_t_concat"] = True
                model_kwargs["extra_kwargs"]["native_reference_packed_ranges"] = native_reference_packed_ranges
            l4_active = any(
                d != s for d, s in zip(condition_frame_ranges, condition_src_frame_ranges)
            )
            slot_levels = []
            for item in active_slots:
                dst = item["dst"]
                src = item["src"]
                dst_len = dst[1] - dst[0]
                src_len = src[1] - src[0]
                if int(item["kind"]) == 2:
                    slot_levels.append("L1/L2-anchor")
                elif src_len == dst_len:
                    slot_levels.append("L3-video")
                else:
                    slot_levels.append(f"L4-video({src_len}->{dst_len})")
            model_kwargs["extra_kwargs"]["debug_task_mode"] = debug_task_mode
            model_kwargs["extra_kwargs"]["debug_condition_source"] = "per_segment"
            model_kwargs["extra_kwargs"]["debug_cond_dropout_applied"] = dropout_applied
            model_kwargs["extra_kwargs"]["debug_cond_dropped_slots"] = dropped_slot_ids
            model_kwargs["extra_kwargs"]["debug_cond_before_slots"] = dropout_before_slot_ids
            model_kwargs["extra_kwargs"]["debug_cond_after_slots"] = keep_slot_ids
            model_kwargs["extra_kwargs"]["debug_cond_slot_levels"] = slot_levels
            model_kwargs["extra_kwargs"]["debug_l4_active"] = l4_active
            model_kwargs["extra_kwargs"]["debug_cond_raw_candidates"] = _debug_compact(kwargs.get('cond_debug_raw_candidates', None))
            model_kwargs["extra_kwargs"]["debug_cond_prob_dropped"] = _debug_compact(kwargs.get('cond_debug_prob_dropped', None))
            model_kwargs["extra_kwargs"]["debug_cond_bad_skipped"] = _debug_compact(kwargs.get('cond_debug_bad_skipped', None))
            if range_condition_debug:
                print(
                    f"[TrainTask DEBUG] rank={rank_dbg} source=per_segment task={debug_task_mode} "
                    f"mask_type={mask_type} range_condition=ON range_mode={range_condition_mode_cfg} "
                    f"cond_latents=present cond_dropout_applied={dropout_applied} "
                    f"cond_dropped_slots={dropped_slot_ids} before_slots={dropout_before_slot_ids} "
                    f"after_slots={keep_slot_ids} levels={slot_levels} "
                    f"dst_ranges={condition_frame_ranges} src_ranges={condition_src_frame_ranges} "
                    f"physical_dst_ranges={condition_physical_dst_frame_ranges} "
                    f"temporal_pack_mode={temporal_pack_mode} "
                    f"l4_active={l4_active} mask_sum={int(multitask_mask.sum().item())} "
                    f"native_t_concat={native_t_concat} native_ranges={native_reference_packed_ranges} "
                    f"cond_shape={tuple(cond_latents.shape)} target_T={latents.shape[2]} videoid={_debug_compact(videoid)} "
                    f"index={_debug_compact(index)} raw_candidates={_debug_compact(kwargs.get('cond_debug_raw_candidates', None))} "
                    f"prob_dropped={_debug_compact(kwargs.get('cond_debug_prob_dropped', None))} "
                    f"bad_skipped={_debug_compact(kwargs.get('cond_debug_bad_skipped', None))}",
                    flush=True,
                )
            train_task_debug_printed = True
            return latents, model_kwargs, data_type, freqs_cos.shape[0], multitask_mask, cond_latents, last_frame_cond_latents, mask_type, use_i2v_instruction, videoid, repa_feats, index, text

        if range_condition_debug:
            print(
                f"[TrainTask DEBUG] rank={rank_dbg} source=per_segment_empty task=t2v "
                f"mask_type={mask_type} range_condition=OFF range_mode={range_condition_mode_cfg} "
                f"cond_latents=none cond_dropout_applied=False cond_dropped_slots=[] "
                f"videoid={_debug_compact(videoid)} index={_debug_compact(index)} "
                f"raw_candidates={_debug_compact(kwargs.get('cond_debug_raw_candidates', None))} "
                f"prob_dropped={_debug_compact(kwargs.get('cond_debug_prob_dropped', None))} "
                f"bad_skipped={_debug_compact(kwargs.get('cond_debug_bad_skipped', None))}",
                flush=True,
            )
        train_task_debug_printed = True

    # Reference-timeline production references must come from dataset-owned prompt
    # segments. Never synthesize a random self-reference range when the current
    # sample has no valid per-segment condition; fall back to plain t2v instead.
    if reference_timeline and mask_type == "range_condition":
        mask_type = "t2v"
        model_kwargs["mask_type"] = "t2v"
        if range_condition_debug:
            print(
                f"[TrainTask DEBUG] rank={rank_dbg} "
                "source=reference_timeline_no_valid_dataset_reference "
                "task=t2v mask_type=t2v range_condition=OFF",
                flush=True,
            )

    # Path (A): range_condition from multitask sampler — legacy/random path
    # Path (B): legacy flip from t2v — use unified_physrope_condition_prob
    _activate_range_condition = False
    if mask_type == "range_condition" and latents.shape[2] > 4:
        _activate_range_condition = True
    elif unified_physrope_enable and mask_type == "t2v" and latents.shape[2] > 4:
        _activate_range_condition = True  # will check prob below via SP broadcast

    if _activate_range_condition:
        range_source = "multitask_range_condition" if mask_type == "range_condition" else "legacy_t2v_flip"
        # --- SP-safe random via broadcast tensor ---
        # All random decisions are generated on rank 0 (or single-GPU) using
        # torch RNG (not Python random) to avoid polluting global random state
        # on any rank. The packed tensor is broadcast to all SP ranks.
        from hymm.utils.parallel_states import nccl_info
        n_frames = latents.shape[2]
        _sp_size = nccl_info.sp_size if nccl_info.sp_size is not None else 1

        # Pack random decisions into a broadcast tensor:
        # [0]: activate flag (always 1.0 for path A, prob-check for path B)
        # [1]: n_segments (1, 2, or 3)
        # [2..2+2*MAX_SEGS-1]: (frac_i, start_frac_i) pairs for each segment
        _MAX_SEGS = 3
        _RC_TENSOR_LEN = 2 + 2 * _MAX_SEGS  # = 8
        _rc_rand = torch.zeros(_RC_TENSOR_LEN, dtype=torch.float32, device=device)
        _sp_local_rank = nccl_info.rank_within_spgroup if hasattr(nccl_info, 'rank_within_spgroup') else 0

        # All SP ranks consume the same torch RNG draw count, then rank 0's
        # decisions are broadcast. This avoids both Python RNG and torch RNG
        # state divergence across SP ranks.
        _rvals = torch.rand(2 + 2 * _MAX_SEGS, device=device)

        if _sp_size <= 1 or _sp_local_rank == 0:
            if mask_type == "range_condition":
                # Path A: already sampled as range_condition, always activate
                _rc_rand[0] = 1.0
            else:
                # Path B: legacy t2v flip
                _rc_rand[0] = 1.0 if _rvals[0].item() < unified_physrope_condition_prob else 0.0

            # Decide number of segments: 50% → 1, 30% → 2, 20% → 3
            _seg_roll = _rvals[1].item()
            if _seg_roll < 0.5:
                n_segs = 1
            elif _seg_roll < 0.8:
                n_segs = 2
            else:
                n_segs = 3
            _rc_rand[1] = float(n_segs)

            # Generate per-segment random params: (frac, start_frac)
            # Single segment: 10%-40% of frames (original behavior)
            # Multi-segment: each segment 5%-20% of frames
            for si in range(n_segs):
                if n_segs == 1:
                    _rc_rand[2 + si * 2] = 0.1 + _rvals[2 + si * 2].item() * 0.3       # uniform(0.1, 0.4)
                else:
                    _rc_rand[2 + si * 2] = 0.05 + _rvals[2 + si * 2].item() * 0.15      # uniform(0.05, 0.20)
                _rc_rand[2 + si * 2 + 1] = _rvals[2 + si * 2 + 1].item()                # start position

        if _sp_size > 1 and nccl_info.sp_group is not None:
            dist.broadcast(_rc_rand, src=dist.get_global_rank(nccl_info.sp_group, 0),
                           group=nccl_info.sp_group)

        if _rc_rand[0].item() > 0.5:
            n_segs = int(_rc_rand[1].item())
            # Build non-overlapping segments sorted by start position
            raw_segments = []
            for si in range(n_segs):
                frac = _rc_rand[2 + si * 2].item()
                start_frac = _rc_rand[2 + si * 2 + 1].item()
                seg_len = max(1, int(n_frames * frac))
                max_start = max(n_frames - seg_len, 0)
                seg_start = int(start_frac * max_start)
                seg_start = min(seg_start, max_start)
                raw_segments.append((seg_start, seg_start + seg_len))

            # Sort by start and resolve overlaps: shift later segments after earlier ones
            raw_segments.sort(key=lambda x: x[0])
            condition_frame_ranges = []
            occupied_end = 0
            for seg_start, seg_end in raw_segments:
                seg_len = seg_end - seg_start
                actual_start = max(seg_start, occupied_end)
                if actual_start + seg_len > n_frames:
                    # Not enough room for this segment — skip
                    continue
                condition_frame_ranges.append((actual_start, actual_start + seg_len))
                occupied_end = actual_start + seg_len

            # Safety: must have at least 1 segment and at least 1 gen frame
            total_cond = sum(e - s for s, e in condition_frame_ranges)
            if len(condition_frame_ranges) == 0 or total_cond >= n_frames:
                # Fallback: single segment, 25% of frames from start
                cond_len = max(1, n_frames // 4)
                condition_frame_ranges = [(0, cond_len)]

            multitask_mask = get_multitask_mask_with_ranges(n_frames, condition_frame_ranges, device=device)
            mask_type = "range_condition"

            if native_t_concat:
                native_chunks = [latents[:, :, start:end].clone() for start, end in condition_frame_ranges]
                cond_latents = torch.cat(native_chunks, dim=2)
                native_reference_packed_ranges = []
                native_cursor = 0
                for chunk in native_chunks:
                    native_reference_packed_ranges.append(
                        (native_cursor, native_cursor + int(chunk.shape[2]))
                    )
                    native_cursor += int(chunk.shape[2])
                model_kwargs["extra_kwargs"]["range_condition_native_t_concat"] = True
                model_kwargs["extra_kwargs"]["native_reference_packed_ranges"] = native_reference_packed_ranges
            else:
                cond_latents = latents.clone()
                gen_mask = (multitask_mask == 0)
                cond_latents[:, :, gen_mask] = 0.0

            model_kwargs["extra_kwargs"]["condition_frame_ranges"] = condition_frame_ranges
            model_kwargs["extra_kwargs"]["condition_src_frame_ranges"] = list(condition_frame_ranges)
            model_kwargs["extra_kwargs"]["condition_physical_dst_frame_ranges"] = list(
                condition_frame_ranges
            )
            model_kwargs["extra_kwargs"]["condition_kinds"] = [1] * len(
                condition_frame_ranges
            )
            model_kwargs["extra_kwargs"]["range_condition_temporal_pack_mode"] = (
                temporal_pack_mode
            )
            model_kwargs["mask_type"] = mask_type

            total_cond = sum(e - s for s, e in condition_frame_ranges)
            model_kwargs["extra_kwargs"]["debug_task_mode"] = "v2v"
            model_kwargs["extra_kwargs"]["debug_condition_source"] = range_source
            model_kwargs["extra_kwargs"]["debug_cond_dropout_applied"] = False
            model_kwargs["extra_kwargs"]["debug_cond_dropped_slots"] = []
            model_kwargs["extra_kwargs"]["debug_cond_before_slots"] = []
            model_kwargs["extra_kwargs"]["debug_cond_after_slots"] = []
            model_kwargs["extra_kwargs"]["debug_cond_slot_levels"] = ["random-range"] * len(condition_frame_ranges)
            model_kwargs["extra_kwargs"]["debug_l4_active"] = False
            if range_condition_debug:
                print(
                    f"[TrainTask DEBUG] rank={rank} source={range_source} task=v2v "
                    f"mask_type={mask_type} range_condition=ON range_mode={range_condition_mode_cfg} "
                    f"cond_latents=present cond_dropout_applied=False cond_dropped_slots=[] "
                    f"ranges={condition_frame_ranges} src_ranges={condition_frame_ranges} "
                    f"total_cond={total_cond} gen_frames={n_frames - total_cond} "
                    f"mask_sum={int(multitask_mask.sum().item())} cond_shape={tuple(cond_latents.shape)} "
                    f"sp_size={_sp_size} videoid={_debug_compact(videoid)} index={_debug_compact(index)}",
                    flush=True,
                )
            train_task_debug_printed = True
        else:
            if range_condition_debug:
                print(
                    f"[TrainTask DEBUG] rank={rank} source={range_source} task=t2v "
                    f"mask_type={mask_type} range_condition=OFF range_mode={range_condition_mode_cfg} "
                    f"cond_latents=none cond_dropout_applied=False cond_dropped_slots=[] "
                    f"legacy_flip_prob={unified_physrope_condition_prob} "
                    f"videoid={_debug_compact(videoid)} index={_debug_compact(index)}",
                    flush=True,
                )
            model_kwargs["extra_kwargs"]["debug_task_mode"] = "t2v"
            model_kwargs["extra_kwargs"]["debug_condition_source"] = range_source
            model_kwargs["extra_kwargs"]["debug_cond_dropout_applied"] = False
            model_kwargs["extra_kwargs"]["debug_cond_dropped_slots"] = []
            train_task_debug_printed = True

    if not train_task_debug_printed:
        fallback_task = "t2v"
        if mask_type == "i2v":
            fallback_task = "ti2v"
        elif mask_type == "fl2v":
            fallback_task = "fl2v"
        if range_condition_debug:
            print(
                f"[TrainTask DEBUG] rank={rank} source=standard_multitask task={fallback_task} "
                f"mask_type={mask_type} range_condition=OFF range_mode={range_condition_mode_cfg} "
                f"cond_latents={'present' if cond_latents is not None else 'none'} "
                f"cond_dropout_applied=False cond_dropped_slots=[] "
                f"multitask_mask_sum={None if multitask_mask is None else int(multitask_mask.sum().item())} "
                f"videoid={_debug_compact(videoid)} index={_debug_compact(index)}",
                flush=True,
            )
        model_kwargs["extra_kwargs"]["debug_task_mode"] = fallback_task
        model_kwargs["extra_kwargs"]["debug_condition_source"] = "standard_multitask"
        model_kwargs["extra_kwargs"]["debug_cond_dropout_applied"] = False
        model_kwargs["extra_kwargs"]["debug_cond_dropped_slots"] = []

    return latents, model_kwargs, data_type, freqs_cos.shape[0], multitask_mask, cond_latents, last_frame_cond_latents, mask_type, use_i2v_instruction, videoid, repa_feats, index, text
def decode_latents_to_pixels(
    latents: torch.Tensor,
    vae,
    enable_tiling: bool = False,
    tile_size: int = -1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Decode VAE latents back to normalized pixel space.

    Args:
        latents: Tensor shaped as (b, c, h, w) or (b, c, f, h, w).
        vae: Loaded VAE model used for decoding.
        enable_tiling: Whether to enable the VAE's spatial tiling mode.
        tile_size: Number of samples per decoding chunk to reduce memory usage. <=0 decodes in a single chunk.
        generator: Optional random generator forwarded to the VAE decode call.

    Returns:
        Tensor with values in [0, 1] and shape matching the input rank (image/video).
    """
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"latents must be a torch.Tensor, got {type(latents)}")

    needs_frame_squeeze = False
    if latents.dim() == 4:
        latents = latents.unsqueeze(2)
        needs_frame_squeeze = True
    elif latents.dim() != 5:
        raise ValueError(f"latents must be 4D or 5D tensor, got shape {tuple(latents.shape)}")

    try:
        vae_param = next(vae.parameters())
    except StopIteration:
        raise ValueError("VAE has no parameters to infer device/dtype.")
    vae_device = vae_param.device
    vae_dtype = vae_param.dtype

    scaling_factor = getattr(vae.config, "scaling_factor", None)
    if scaling_factor is None or scaling_factor == 0:
        raise ValueError("VAE config must provide a non-zero scaling_factor for decoding.")
    shift_factor = getattr(vae.config, "shift_factor", None)

    latents = latents.to(device=vae_device, dtype=vae_dtype)
    latents = latents / scaling_factor
    if shift_factor is not None:
        if isinstance(shift_factor, torch.Tensor):
            shift = shift_factor.to(device=vae_device, dtype=vae_dtype)
            has_shift = bool(torch.count_nonzero(shift))
        else:
            shift = torch.as_tensor(shift_factor, device=vae_device, dtype=vae_dtype)
            has_shift = float(shift_factor) != 0
        if has_shift:
            latents = latents + shift

    def _decode(chunk: torch.Tensor) -> torch.Tensor:
        decoded = vae.decode(chunk, return_dict=False, generator=generator)[0]
        return decoded

    if enable_tiling:
        vae.enable_tiling()

    try:
        if tile_size is not None and tile_size > 0 and latents.size(0) > tile_size:
            decoded_chunks = []
            for chunk in torch.split(latents, tile_size, dim=0):
                decoded_chunks.append(_decode(chunk))
            image = torch.cat(decoded_chunks, dim=0)
        else:
            image = _decode(latents)
    finally:
        if enable_tiling:
            vae.disable_tiling()

    if needs_frame_squeeze:
        image = image.squeeze(2)
    return image.to(dtype=torch.float32)

def _ntuple(n):
    """Create a function that converts input to n-tuple."""
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse


# Convenience functions for common tuple sizes
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def as_tuple(x):
    """Convert input to tuple."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def as_list_of_2tuple(x):
    """Convert input to list of 2-tuples."""
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst


def dynamic_values_wrapper(anchor_sizes, values):
    """
    Dynamically mapping the size (w, h) to one of the list batch sizes.

    Args:
        anchor_sizes (tuple): list of anchor sizes.
        values (tuple): list of values for different image scale.
    """
    if not isinstance(anchor_sizes, (list, tuple)):
        raise ValueError(f"anchor_sizes should be a list or tuple, but got {type(anchor_sizes)}")
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"batch_sizes should be a list or tuple, but got {type(values)}")
    if len(anchor_sizes) != len(values):
        raise ValueError(f"anchor_sizes and values should have the same length, but got {len(anchor_sizes)} and "
                         f"{len(values)}.")
    anchor_sizes = np.array(anchor_sizes)

    def get_value(size):
        sqrt_area = (size[0] * size[1]) ** 0.5
        val = values[np.argmin(np.abs(anchor_sizes - sqrt_area))]
        return val

    return get_value


def normalize_dit_input(model_type, latents):
    """Normalize DiT input based on model type."""
    if model_type == "hunyuan":
        return latents * 0.476986
    else:
        raise NotImplementedError(f"model_type {model_type} not supported")


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    """Create a linear-quadratic schedule for noise."""
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps ** 2
    )
    const = quadratic_coef * (linear_steps ** 2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i ** 2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. 
    Handles custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def encode_audio_vae_feat(audio_waveform_batch, vae_model, device, args):
    """
    Takes a batch of preprocessed and padded audio waveforms and extracts VAE features on the GPU.
    """
    if isinstance(audio_waveform_batch, list):
        audio_waveform_batch = torch.stack(audio_waveform_batch, dim=0) # [B, 1, 1, T]
    batch_tensor = audio_waveform_batch.to(device)

    # Ensure correct shape for model
    if batch_tensor.ndim == 4 and batch_tensor.shape[1] == 1:
        batch_tensor = batch_tensor.squeeze(1)
    
    # VAE model inference
    x = vae_model.preprocess(batch_tensor, vae_model.sample_rate)
    dtype = x.dtype
    x = x.to(dtype=torch.float32)
    posterior, _, _, _, _ = vae_model.encode(x)
    # posterior = posterior.to(dtype=dtype)

    # VAE Sampling
    audio_tokens = posterior.sample()

    # use fixed length dataset
    if not args.get('use_var_dataloader', False):
        # Align latent feature length
        if audio_tokens.shape[-1] >= args.audio_vae_latent_max_length:
            audio_embedding = audio_tokens[..., :args.audio_vae_latent_max_length]
        else:
            pad_len = args.audio_vae_latent_max_length - audio_tokens.shape[-1]
            audio_embedding = F.pad(audio_tokens, (0, pad_len), 'replicate')
    else:
        audio_embedding = audio_tokens
        
    return audio_embedding

def move_only_tensor_to_device(batch, maybe_tensor_name, device):
    if maybe_tensor_name not in batch:
        return None
    maybe_tensor = batch[maybe_tensor_name]
    if isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor.to(device)
    return maybe_tensor

def prepare_audio_model_inputs(batch, device, vae, text_encoder, args):
    # =================================== Online VAE extraction ================================
    if args.online_vae_extraction and "audio_waveform" in batch:
        audio_waveforms = batch["audio_waveform"]
        batch["audio_token"] = encode_audio_vae_feat(audio_waveforms, vae, device, args)
    
    # ==================================== Move tensor to device ===============================
    audio_embedding = batch["audio_token"].to(device)  # [B x N x T] 
    padded_caption_ids = move_only_tensor_to_device(batch, "padded_caption_ids", device)
    padded_caption_mask = move_only_tensor_to_device(batch, "padded_caption_mask", device)
    # audiossl_feats = move_only_tensor_to_device(batch, "audiossl_feat", device)
    # eat_feats = move_only_tensor_to_device(batch, "eat_feat", device)
    # ssl_feats = []

    # if audiossl_feats is not None:
    #     ssl_feats.append(audiossl_feats)
    # if eat_feats is not None:
    #     ssl_feats.append(eat_feats)

    # ======================================== Encode text ======================================
    text_outputs = text_encoder.encode({'input_ids': padded_caption_ids, 'attention_mask': padded_caption_mask})
    # text_embedding = text_outputs # text_outputs -> {'hidden_state': embeddings, 'attention_mask': mask}
    text_embedding = text_outputs.get('hidden_state', None)
    text_mask = text_outputs.get('attention_mask', None)

    if text_embedding.shape[1] > args.max_text_length:
        text_embedding = text_embedding[:, : args.max_text_length]
        if text_mask is not None:
            text_mask = text_mask[:, : args.max_text_length]

    batch_size = audio_embedding.shape[0]
    if audio_embedding.ndim >= 2 and audio_embedding.ndim <= 3:
        n_tokens = np.prod(audio_embedding.shape[1:])
    else:
        raise ValueError(f"audio_embedding should have 2 or 3 dimensions but got {audio_embedding.ndim}")

    # ===================================== Pack model kwargs ==================================
    model_kwargs = dict(
        text_states=text_embedding,
        encoder_attention_mask=text_mask
    )

    return audio_embedding, model_kwargs, batch_size, n_tokens

from hymm.models.waveflowvae.bigvgan_flow_vae_export import BigVGANFlowVAE

def prepare_t2va_model_inputs(batch, device, text_encoder, audio_vae, args):
    output_dict = {}

    ### Audio feature preparation ###
    with torch.no_grad():
        audio_text_encoder_inputs = text_encoder.text2tokens(batch["audio_caption"], data_type='audio', max_length=args.text_len)
        audio_text_encoder_outputs = text_encoder.encode(
            {"input_ids": audio_text_encoder_inputs["input_ids"], "attention_mask": audio_text_encoder_inputs["attention_mask"]},
            data_type='audio',
            device=device,
            semantic_images=None,
        )
        audio_text_states = audio_text_encoder_outputs.hidden_state
        audio_text_mask = audio_text_encoder_outputs.attention_mask

    output_dict["audio_text_cond"] = audio_text_states.to(device=device)
    output_dict["audio_text_mask"] = audio_text_mask.to(device=device)

    # - audio vae for online feature extraction
    if not isinstance(audio_vae._orig_mod if hasattr(audio_vae, "_orig_mod") else audio_vae, BigVGANFlowVAE):
        with torch.no_grad():
            x = audio_vae.preprocess(batch['waveform'].to(device=device), audio_vae.sample_rate)
            x = x.to(dtype=torch.float32)
            posterior, _, _, _, _ = audio_vae.encode(x)
            audio_tokens = posterior.sample()
    else:
        audio_tokens = audio_vae.encode(batch['waveform'].to(device=device))
    output_dict["audio_latents"] = audio_tokens.to(device=device)

    # Encoders for video branch
    with torch.no_grad():
        video_text_encoder_inputs = text_encoder.text2tokens(batch["video_caption"], data_type='video', max_length=args.text_len)
        video_text_encoder_outputs = text_encoder.encode(
            {"input_ids": video_text_encoder_inputs["input_ids"], "attention_mask": video_text_encoder_inputs["attention_mask"]},
            data_type='video',
            device=device,
            semantic_images=None,
        )
        video_text_states = video_text_encoder_outputs.hidden_state
        video_text_mask = video_text_encoder_outputs.attention_mask

    output_dict["video_text_cond"] = video_text_states.to(device=device)
    output_dict["video_text_mask"] = video_text_mask.to(device=device)

    output_dict['video_latents'] = batch['video_latents'].to(device=device)

    #FIXME: align with Refrain-T2V
    if args.video_vae_scaling_factor is not None:
        output_dict['video_latents'] = output_dict['video_latents'] * args.video_vae_scaling_factor

    video_latents = output_dict.pop('video_latents', None)
    audio_latents = output_dict.pop('audio_latents', None).unsqueeze(-1).unsqueeze(-1)

    n_tokens = math.prod(video_latents.shape[:1] + video_latents.shape[-3:])

    return video_latents, audio_latents, output_dict, n_tokens
