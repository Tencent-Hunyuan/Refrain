# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
import copy
import inspect
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from packaging import version

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
# from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.utils.prompt_cleaning import strip_identity_tags_nested
from hymm.models.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hymm.models.text_encoder import TextEncoder
from hymm.models.modules import RefrainTransformer

from einops import rearrange
from hymm.utils.parallel_states import get_sequence_parallel_state, nccl_info
# from hymm.utils.communications import all_gather, all_to_all_4D
import torch.nn.functional as F
import re
from hymm.dataset.format_prompt import MultilingualPromptFormat
from hymm.utils.multitask_utils import black_image, merge_tensor_by_mask, get_multitask_mask_with_ranges
from hymm.utils.native_reference import (
    as_bool,
    pack_native_reference_input,
    slice_native_target_output,
)
from PIL import Image
from hymm.utils.data_utils import generate_crop_size_list, get_closest_ratio, resize_and_center_crop

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

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


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


# ═══════════════════════════════════════════════════════════════════════════════
# L6: Timeline Warping utilities (future_plan §1.2 L6)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_warp_fn_from_params(params: dict):
    """Build a time warp function from declarative parameters.

    Supported warp types:
      - "slowmo":  stretch time in [range[0], range[1]] by factor (>1 = slower)
      - "speedup": compress time in [range[0], range[1]] by factor (>1 = faster)
      - "loop":    repeat time range N times
      - "reverse": reverse time within [range[0], range[1]]
      - "linear":  t_out = scale * t + offset (global affine)
      - "custom_keypoints": piecewise-linear from (t_in, t_out) keypoint pairs
    """
    warp_type = params.get("type", "linear")
    range_start = float(params.get("range", [0, 1e9])[0])
    range_end = float(params.get("range", [0, 1e9])[1])

    if warp_type == "slowmo":
        factor = float(params.get("factor", 2.0))  # 2.0 = 2x slower
        def warp_fn(t):
            if range_start <= t <= range_end:
                # Stretch: more time per unit of content
                local_t = (t - range_start) / factor + range_start
                return local_t
            elif t > range_end:
                # Shift subsequent frames by the expansion
                expansion = (range_end - range_start) * (1.0 - 1.0 / factor)
                return t - expansion
            return t
        return warp_fn

    elif warp_type == "speedup":
        factor = float(params.get("factor", 2.0))  # 2.0 = 2x faster
        def warp_fn(t):
            if range_start <= t <= range_end:
                local_t = (t - range_start) * factor + range_start
                return local_t
            elif t > range_end:
                compression = (range_end - range_start) * (1.0 - factor)
                return t + compression
            return t
        return warp_fn

    elif warp_type == "loop":
        repeats = int(params.get("repeats", 2))
        def warp_fn(t):
            if range_start <= t <= range_end:
                duration = range_end - range_start
                local = (t - range_start) % (duration / repeats)
                return range_start + local
            return t
        return warp_fn

    elif warp_type == "reverse":
        def warp_fn(t):
            if range_start <= t <= range_end:
                return range_end - (t - range_start)
            return t
        return warp_fn

    elif warp_type == "linear":
        scale = float(params.get("scale", 1.0))
        offset = float(params.get("offset", 0.0))
        def warp_fn(t):
            return scale * t + offset
        return warp_fn

    elif warp_type == "custom_keypoints":
        # Piecewise-linear from user-provided (t_in, t_out) pairs
        keypoints = params.get("keypoints", [[0, 0], [1, 1]])
        keypoints = sorted(keypoints, key=lambda p: p[0])

        def warp_fn(t):
            if t <= keypoints[0][0]:
                return keypoints[0][1]
            if t >= keypoints[-1][0]:
                return keypoints[-1][1]
            for i in range(len(keypoints) - 1):
                t_in_a, t_out_a = keypoints[i]
                t_in_b, t_out_b = keypoints[i + 1]
                if t_in_a <= t <= t_in_b:
                    alpha = (t - t_in_a) / max(t_in_b - t_in_a, 1e-8)
                    return t_out_a + alpha * (t_out_b - t_out_a)
            return t
        return warp_fn

    else:
        raise ValueError(f"Unknown timeline_warp_params type: {warp_type}. "
                         f"Supported: slowmo, speedup, loop, reverse, linear, custom_keypoints")


def _apply_timeline_warp(time_range_rows: list, warp_fn) -> list:
    """Apply a user-defined warp function to all time ranges.

    time_range_rows: list of rows, each row is a list of segment dicts
                     [{"segment_id": X, "time_range": [start, end], ...}, ...]
    warp_fn: callable t_out = warp_fn(t_in)

    Returns: modified time_range_rows with warped time_range values.
    """
    import copy
    warped_rows = []
    for row in time_range_rows:
        warped_row = []
        for seg in row:
            new_seg = copy.deepcopy(seg)
            tr = new_seg.get("time_range")
            if tr is not None and isinstance(tr, (list, tuple)) and len(tr) == 2:
                try:
                    new_start = warp_fn(float(tr[0]))
                    new_end = warp_fn(float(tr[1]))
                    # Ensure valid range (start < end)
                    if new_end <= new_start:
                        new_end = new_start + 0.1
                    new_seg["time_range"] = [new_start, new_end]
                except (TypeError, ValueError):
                    pass  # Keep original if warp fails
            warped_row.append(new_seg)
        warped_rows.append(warped_row)
    return warped_rows


# ═══════════════════════════════════════════════════════════════════════════════


class HunyuanVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`TextEncoder`]):
            Frozen text-encoder.
        text_encoder_2 ([`TextEncoder`]):
            Frozen text-encoder_2.
        transformer ([`HYVideoDiffusionTransformer`]):
            A `HYVideoDiffusionTransformer` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: RefrainTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        text_len: int = 1000,
        args=None,
        byt5_kwarg=None,
        device=None,
        is_eval=False,
        extra_model=None
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.args = args
        # ==========================================================================================

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if self.args.glyph_byT5_v2:
            self.byt5_max_length = byt5_kwarg["byt5_max_length"]
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder_2=text_encoder_2,
                byt5_model=byt5_kwarg["byt5_model"],
                byt5_tokenizer=byt5_kwarg["byt5_tokenizer"],
            )
            self.prompt_format = MultilingualPromptFormat(font_path=self.args.multilingual_prompt_format_font_path,
                                                          color_path=self.args.multilingual_prompt_format_color_path)
        else:
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                text_encoder_2=text_encoder_2,
            )
            self.byt5_model = None
            self.byt5_tokenizer = None

        if hasattr(self.vae.config, 'block_out_channels'):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = self.vae.config.spatial_compression_ratio

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.text_len = text_len
        if extra_model:
            self.vision_encoder = extra_model.get("vision_encoder", None)
            self.register_modules(vision_encoder=self.vision_encoder)
        else:
            self.vision_encoder = None
            self.image_encoder = None

        if is_eval:
            if self.vae is not None:
                self.vae = self.vae.to(device=device)
            if self.text_encoder is not None:
                self.text_encoder = self.text_encoder.to(device=device)
            if self.transformer is not None:
                self.transformer = self.transformer.to(device=device)
            if self.text_encoder_2 is not None:
                self.text_encoder_2 = self.text_encoder_2.to(device=device)
            if self.byt5_model is not None:
                self.byt5_model = self.byt5_model.to(device=device)
            if self.vision_encoder is not None:
                self.vision_encoder = self.vision_encoder.to(device=device)

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        semantic_images: Optional[torch.Tensor] = None
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        prompt = strip_identity_tags_nested(prompt)

        if text_encoder is None:
            text_encoder = self.text_encoder

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        # if lora_scale is not None and isinstance(self, LoraLoaderMixin):
        #     self._lora_scale = lora_scale
        #
        #     # dynamically adjust the LoRA scale
        #     if not USE_PEFT_BACKEND:
        #         adjust_lora_scale_text_encoder(text_encoder.model, lora_scale)
        #     else:
        #         scale_lora_layers(text_encoder.model, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # # textual inversion: process multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type, max_length=self.text_len)
            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device, semantic_images=semantic_images
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                    semantic_images=semantic_images
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # # textual inversion: process multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, text_encoder.tokenizer)

            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type, max_length=self.text_len)

            # if hasattr(text_encoder.model.config, "use_attention_mask") and text_encoder.model.config.use_attention_mask:
            #     attention_mask = uncond_input.attention_mask.to(device)
            # else:
            #     attention_mask = None

            if semantic_images is not None:
                uncond_image = [black_image(img.size[0], img.size[1]) for img in semantic_images]
            else:
                uncond_image = None
            negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type, semantic_images=uncond_image)
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def decode_latents(self, latents, enable_tiling=True):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        use_vae_tiling = (
            enable_tiling
            and hasattr(self.vae, "enable_tiling")
            and hasattr(self.vae, "disable_tiling")
        )
        if use_vae_tiling:
            self.vae.enable_tiling()
        try:
            image = self.vae.decode(latents, return_dict=False)[0]
        finally:
            if use_vae_tiling:
                self.vae.disable_tiling()
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if image.ndim == 4:
            image = image.cpu().permute(0, 2, 3, 1).float()
        else:
            image = image.cpu().float()
        return image

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        video_length,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        vae_ver="88-4c-sd",
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if video_length is not None:
            if "884" in vae_ver:
                if video_length != 1 and (video_length - 1) % 4 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 4 but is {video_length}."
                    )
            elif "888" in vae_ver or "16168" in vae_ver:
                if video_length != 1 and (video_length - 1) % 8 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 8 but is {video_length}."
                    )

        if callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        """
        Tokenize text prompt for byT5 model.
        
        Args:
            byt5_tokenizer: The byT5 tokenizer
            byt5_max_length: Maximum sequence length
            text_prompt: Text prompt to tokenize
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs.input_ids, byt5_text_inputs.attention_mask

    def _extract_glyph_texts(self, prompt):
        """
        Extract glyph texts from prompt using regex pattern.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            List of extracted glyph texts
        """
        pattern = r'\"(.*?)\"|“(.*?)”'
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        result = list(dict.fromkeys(result)) if len(result) > 1 else result
        return result

    def _process_single_byt5_prompt(self, prompt_text, device):
        """
        Process a single prompt for byT5 encoding.
        
        Args:
            prompt_text: The prompt text to process
            device: Target device for tensors
            
        Returns:
            Tuple of (byt5_embeddings, byt5_mask)
        """
        # Initialize with zero tensors
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=device, dtype=torch.int64)
        
        # Extract glyph texts from prompt
        glyph_texts = self._extract_glyph_texts(prompt_text)
        
        if len(glyph_texts) > 0:
            # Create style list for formatting
            text_styles = [{'color': None, 'font-family': None} for _ in range(len(glyph_texts))]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            
            # Tokenize and encode
            text_ids, text_mask = self.get_byt5_text_tokens(
                self.byt5_tokenizer, self.byt5_max_length, formatted_text
            )
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)
            
            # Get embeddings from byT5 model
            byt5_outputs = self.byt5_model(text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask
            
        return byt5_embeddings, byt5_mask

    def _prepare_byt5_embeddings(self, prompts, device):
        """
        Prepare byT5 embeddings for both positive and negative prompts.
        
        Args:
            prompts: List of prompt strings or single prompt string
            device: Target device for tensors
            
        Returns:
            Dict containing byT5 embeddings and masks
        """
        if not self.args.glyph_byT5_v2:
            return {}
            
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        # Process each prompt
        for prompt in prompt_list:
            # Process positive prompt
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            # Process negative prompt (empty string for negative)
            if self.do_classifier_free_guidance:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        # Concatenate all embeddings and masks
        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)
        
        if self.do_classifier_free_guidance:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)
            
            # Combine negative and positive for CFG
            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return {
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks
        }

    def extract_image_features(self, reference_image):
        """
        Extract features from a reference image using VisionEncoder

        Args:
            reference_image: PIL Image or tensor of shape (3, H, W)

        Returns:
            image_features: Tensor of shape (1, feature_dim)
        """
        assert isinstance(reference_image, np.ndarray)
        assert reference_image.ndim == 3 and reference_image.shape[2] == 3
        assert reference_image.dtype == np.uint8

        image_encoder_output = self.vision_encoder.encode_images(reference_image)

        return image_encoder_output

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        multitask_mask_training_type: Optional[str] = None,
        multitask_type: Optional[str] = None,
        cond_latents: Optional[torch.Tensor] = None,
        last_frame_cond_latents: Optional[torch.Tensor] = None,
        uncond_latents: Optional[torch.Tensor] = None,
        multitask_mask: Optional[torch.Tensor] = None,
        semantic_images: Optional[torch.Tensor] = None,
        reference_image=None,
        vision_num_semantic_tokens: Optional[int] = None,
        vision_states_dim: Optional[int] = None,
        bucket_hw_base_size=None,
        bucket_hw_bucket_stride=None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.

            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        target_dtype = PRECISION_TO_TYPE[self.args.precision]

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = (
            # torch.device(f"cuda:{dist.get_rank()}")
            # if dist.is_initialized()
            # else self._execution_device
        # )
        device = self.transformer.device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            data_type=data_type,
            semantic_images=semantic_images
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        extra_kwargs = {}
        if self.args.glyph_byT5_v2:
            extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if getattr(self.args, "use_physical_rope", False):
            text_segment_ids = kwargs.get("text_segment_ids")
            text_segment_time_ranges = kwargs.get("text_segment_time_ranges")

            def _normalize_time_range_rows(value):
                if value is None:
                    return None
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except Exception:
                        return None
                if not isinstance(value, (list, tuple)):
                    return None
                if len(value) == 0:
                    return []
                if isinstance(value[0], dict):
                    return [copy.deepcopy(list(value))]
                if isinstance(value[0], str):
                    rows = []
                    for item in value:
                        try:
                            rows.append(json.loads(item))
                        except Exception:
                            return None
                    return rows
                if isinstance(value[0], (list, tuple)):
                    return copy.deepcopy(list(value))
                return None

            def _expand_rows(rows, target_rows):
                if rows is None:
                    return None
                if len(rows) == target_rows:
                    return rows
                if len(rows) == 1:
                    return [copy.deepcopy(rows[0]) for _ in range(target_rows)]
                return rows

            cond_batch = prompt_embeds.shape[0]
            time_range_rows = _normalize_time_range_rows(text_segment_time_ranges)
            time_range_rows = _expand_rows(time_range_rows, cond_batch)
            if time_range_rows is not None:
                if self.do_classifier_free_guidance:
                    time_range_rows = time_range_rows + [copy.deepcopy(row) for row in time_range_rows]
                extra_kwargs["text_segment_time_ranges"] = time_range_rows

            if text_segment_ids is not None:
                extra_kwargs["text_segment_ids"] = text_segment_ids

            extra_kwargs["use_physical_rope"] = True
            extra_kwargs["physical_rope_theta"] = getattr(self.args, "physical_rope_theta", 256.0)
            extra_kwargs["physical_rope_scale"] = getattr(self.args, "physical_rope_scale", 0.1)
            extra_kwargs["physical_rope_dim"] = getattr(self.args, "physical_rope_dim", 32)
            extra_kwargs["disable_text_physical_rope"] = getattr(self.args, "disable_text_physical_rope", False)
            extra_kwargs["physical_rope_time_range_mode"] = getattr(self.args, "physical_rope_time_range_mode", "second")
            extra_kwargs["video_fps"] = getattr(self.args, "video_fps", 24.0)
            # PhysRoPE v2 parameters (alpha != None activates v2)
            _alpha = getattr(self.args, "physical_rope_alpha", None)
            if _alpha is not None:
                extra_kwargs["physical_rope_alpha"] = float(_alpha)
                extra_kwargs["physical_rope_offset_mode"] = getattr(self.args, "physical_rope_offset_mode", "physical")
                extra_kwargs["physical_rope_cut"] = getattr(self.args, "physical_rope_cut", False)
                extra_kwargs["physical_rope_speed_aware"] = getattr(self.args, "physical_rope_speed_aware", False)
                extra_kwargs["physical_rope_speed_mode"] = getattr(self.args, "physical_rope_speed_mode", "uniform")
                extra_kwargs["physical_rope_v2_dim"] = getattr(self.args, "physical_rope_v2_dim", 0)
                extra_kwargs["physrope_route_debug_limit"] = getattr(
                    self.args, "physrope_route_debug_limit", 2
                )
                extra_kwargs["reference_timeline_debug_limit"] = getattr(
                    self.args, "reference_timeline_debug_limit", 0
                )
            extra_kwargs["range_condition_temporal_pack_mode"] = kwargs.get(
                "range_condition_temporal_pack_mode",
                getattr(self.args, "range_condition_temporal_pack_mode", "legacy_trilinear"),
            )

            # ─── L6: User-defined Timeline Warping (future_plan §1.2 L6) ───
            # At inference time, users can pass a custom time warp function φ via kwargs.
            # This remaps ALL temporal coordinates in text_segment_time_ranges before
            # they reach the PhysRoPE encoder, enabling:
            #   - Slow-motion segments (stretch time ranges)
            #   - Fast-forward segments (compress time ranges)
            #   - Time loops / reversals (arbitrary remapping)
            #   - Non-uniform speed curves (ease-in/out)
            #
            # Usage:
            #   # Linear slow-motion (2x slower from t=2 to t=5):
            #   pipeline(..., timeline_warp_fn=lambda t: t*0.5 + 1.0 if 2<=t<=5 else t)
            #
            #   # Or pass warp parameters for built-in warps:
            #   pipeline(..., timeline_warp_params={"type": "slowmo", "range": [2.0, 5.0], "factor": 0.5})
            #   pipeline(..., timeline_warp_params={"type": "speedup", "range": [0.0, 3.0], "factor": 2.0})
            #   pipeline(..., timeline_warp_params={"type": "loop", "range": [1.0, 4.0], "repeats": 2})
            timeline_warp_fn = kwargs.get("timeline_warp_fn", None)
            timeline_warp_params = kwargs.get("timeline_warp_params", None)

            if timeline_warp_fn is None and timeline_warp_params is not None:
                timeline_warp_fn = _build_warp_fn_from_params(timeline_warp_params)

            if timeline_warp_fn is not None and time_range_rows is not None:
                time_range_rows = _apply_timeline_warp(time_range_rows, timeline_warp_fn)
                extra_kwargs["text_segment_time_ranges"] = time_range_rows
                if str(os.environ.get("RANK", "0")) == "0":
                    print(f"[L6 Timeline Warp] Applied user warp function to "
                          f"{sum(len(row) for row in time_range_rows)} segment entries", flush=True)

            if not getattr(self, '_physrope_infer_printed', False) and str(os.environ.get("RANK", "0")) == "0":
                self._physrope_infer_printed = True
                print(f"[PhysRoPE INFERENCE] (pipeline_hunyuan_video.py:L1165) Pipeline PhysRoPE kwargs: "
                      f"use_physical_rope=True | "
                      f"theta={extra_kwargs['physical_rope_theta']} | "
                      f"scale={extra_kwargs['physical_rope_scale']} | "
                      f"dim={extra_kwargs['physical_rope_dim']} | "
                      f"disable_text={extra_kwargs['disable_text_physical_rope']} | "
                      f"time_range_mode={extra_kwargs['physical_rope_time_range_mode']} | "
                      f"fps={extra_kwargs['video_fps']} | "
                      f"has_segment_ids={text_segment_ids is not None} | "
                      f"n_time_range_rows={len(time_range_rows) if time_range_rows else 0} | "
                      f"prompt_embeds.shape={prompt_embeds.shape}")

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )
        if "884" in vae_ver or "16164" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
            if kwargs.get('reorg_token', False):
                video_length = (video_length + 1) // 2            
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1
        else:
            video_length = video_length

        # 5. Prepare latent variables
        native_t_concat = (
            as_bool(getattr(self.args, "range_condition_native_t_concat", False))
            and multitask_mask_training_type == "concat"
            and multitask_type == "range_condition"
        )
        native_reference_t = 0
        num_channels_latents = self.transformer.config.in_channels
        video_length = video_length+1 if multitask_type == "fl2v" else video_length
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            video_length,
            # prompt_embeds.dtype,
            torch.float32,  # hoardcode fp32, align training_losses
            device,
            generator,
            latents,
        )

        vision_states = None
        if multitask_mask_training_type == "concat":
            if multitask_type == 'i2v':
                assert cond_latents is not None
                gt_latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
                gt_latents_concat[:, :, 1:, :, :] = 0.0
            elif multitask_type == 'fl2v':
                assert cond_latents is not None and last_frame_cond_latents is not None

                #加一个dummy的尾帧，用于尾帧注入
                gt_latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
                gt_latents_concat[:, :, 1:-1, :, :] = 0
                gt_latents_concat[:, :, -1:, :, :] = last_frame_cond_latents

            elif multitask_type == 'range_condition':
                range_condition_mode = self.args.range_condition_mode
                assert range_condition_mode in ("soft_reference", "hard_legacy"), (
                    f"range_condition_mode must be soft_reference or hard_legacy, got {range_condition_mode}"
                )
                # Unified PhysRoPE concat mode: condition frames get clean latent
                # in the reference channel. The main latent remains generated in
                # soft_reference mode.
                condition_frame_ranges = kwargs.get("condition_frame_ranges", [])
                # Parallel source-range metadata is retained for ReferenceTimeline
                # provenance/debugging. It never warps target PhysRoPE.
                condition_src_frame_ranges = kwargs.get("condition_src_frame_ranges", None)
                if not condition_src_frame_ranges:
                    condition_src_frame_ranges = list(condition_frame_ranges)
                condition_physical_dst_frame_ranges = kwargs.get(
                    "condition_physical_dst_frame_ranges", None
                )
                if not condition_physical_dst_frame_ranges:
                    condition_physical_dst_frame_ranges = list(condition_frame_ranges)
                condition_kinds = kwargs.get("condition_kinds", None)
                if not condition_kinds:
                    condition_kinds = [1] * len(condition_frame_ranges)
                condition_seg_ids = kwargs.get("condition_seg_ids", None)
                if not condition_seg_ids:
                    condition_seg_ids = [-1] * len(condition_frame_ranges)
                assert cond_latents is not None, (
                    "[Unified PhysRoPE] range_condition requires cond_latents. "
                    "Pass cond_latents with clean condition frame latents, "
                    "not None. All-zero condition is never useful."
                )
                assert multitask_mask is not None or condition_frame_ranges, (
                    "[Unified PhysRoPE] range_condition requires either "
                    "multitask_mask or condition_frame_ranges to be provided."
                )
                if multitask_mask is None and condition_frame_ranges:
                    multitask_mask = get_multitask_mask_with_ranges(
                        latents.shape[2], condition_frame_ranges, device=latents.device
                    )
                assert multitask_mask.ndim == 1 and multitask_mask.shape[0] == latents.shape[2], (
                    f"[Unified PhysRoPE] reference_mask shape {tuple(multitask_mask.shape)} "
                    f"must be [T={latents.shape[2]}]"
                )
                reference_mask_bool = (multitask_mask == 1)
                assert bool(reference_mask_bool.any().item()), (
                    "[Unified PhysRoPE] reference_mask has no condition frames (all zeros). "
                    f"condition_frame_ranges={condition_frame_ranges}"
                )
                cond_latents = cond_latents.to(device=latents.device, dtype=latents.dtype)
                if native_t_concat:
                    assert range_condition_mode == "soft_reference", (
                        "native T concat cannot be combined with hard_legacy pinning"
                    )
                    assert not kwargs.get('reorg_token', True), (
                        "native T concat currently requires equal target/reference channel counts"
                    )
                    native_ranges = kwargs.get("native_reference_packed_ranges", [])
                    # Validate once here. The actual noisy target is packed at each
                    # denoising step so the clean reference prefix remains unchanged.
                    _, native_reference_t = pack_native_reference_input(
                        latents,
                        cond_latents,
                        native_ranges,
                        condition_frame_ranges,
                    )
                    extra_kwargs["range_condition_native_t_concat"] = True
                    extra_kwargs["native_reference_prefix_t"] = native_reference_t
                    extra_kwargs["native_reference_packed_ranges"] = native_ranges
                    if "_native_t_concat_pipeline_printed" not in self.__dict__:
                        self._native_t_concat_pipeline_printed = True
                        print(
                            "[RangeCondition NATIVE-T PIPELINE] "
                            f"reference_T={native_reference_t} target_T={latents.shape[2]} "
                            f"packed_T={native_reference_t + latents.shape[2]} "
                            f"native_ranges={native_ranges} dst_ranges={condition_frame_ranges}",
                            flush=True,
                        )
                elif kwargs.get('reorg_token', True):
                    gt_latents_concat = torch.zeros(
                        latents.shape[0], latents.shape[1]//2, latents.shape[2],
                        latents.shape[3], latents.shape[4],
                        device=latents.device, dtype=latents.dtype,
                    )
                else:
                    gt_latents_concat = torch.zeros(
                        latents.shape[0], latents.shape[1], latents.shape[2],
                        latents.shape[3], latents.shape[4],
                        device=latents.device, dtype=latents.dtype,
                    )
                if not native_t_concat:
                    assert cond_latents.shape[0] == gt_latents_concat.shape[0], (
                        f"cond_latents batch {cond_latents.shape[0]} must match concat batch {gt_latents_concat.shape[0]}"
                    )
                    assert cond_latents.shape[1] == gt_latents_concat.shape[1], (
                        f"cond_latents channels {cond_latents.shape[1]} must match concat channels {gt_latents_concat.shape[1]}"
                    )
                    assert cond_latents.shape[2:] == gt_latents_concat.shape[2:], (
                        f"cond_latents tail {tuple(cond_latents.shape[2:])} "
                        f"must match concat tail {tuple(gt_latents_concat.shape[2:])}"
                    )
                    gt_latents_concat[:, :, reference_mask_bool] = cond_latents[:, :, reference_mask_bool]
                extra_kwargs["condition_frame_ranges"] = condition_frame_ranges
                extra_kwargs["condition_src_frame_ranges"] = condition_src_frame_ranges
                extra_kwargs["condition_physical_dst_frame_ranges"] = (
                    condition_physical_dst_frame_ranges
                )
                extra_kwargs["condition_kinds"] = condition_kinds
                extra_kwargs["condition_seg_ids"] = condition_seg_ids
                extra_kwargs["range_condition_temporal_pack_mode"] = kwargs.get(
                    "range_condition_temporal_pack_mode",
                    getattr(self.args, "range_condition_temporal_pack_mode", "legacy_trilinear"),
                )
                extra_kwargs["range_condition_mode"] = range_condition_mode
                if (
                    extra_kwargs["range_condition_temporal_pack_mode"]
                    == "reference_timeline"
                ):
                    debug_limit = max(
                        int(getattr(self.args, "reference_timeline_debug_limit", 0)),
                        0,
                    )
                    debug_count = int(
                        getattr(self, "_reference_timeline_pipeline_debug_count", 0)
                    )
                    if (
                        str(os.environ.get("RANK", "0")) == "0"
                        and debug_count < debug_limit
                    ):
                        self._reference_timeline_pipeline_debug_count = (
                            debug_count + 1
                        )
                        source_lengths = [
                            int(end) - int(start)
                            for start, end in condition_src_frame_ranges
                        ]
                        target_lengths = [
                            int(end) - int(start)
                            for start, end in condition_physical_dst_frame_ranges
                        ]
                        print(
                            "[PIPELINE REFERENCE-TIMELINE FIXED TARGET DEBUG] "
                            f"debug={debug_count + 1}/{debug_limit} "
                            f"source_ranges={condition_src_frame_ranges} "
                            f"physical_dst_ranges="
                            f"{condition_physical_dst_frame_ranges} "
                            f"source_control_lengths={source_lengths} "
                            f"target_condition_lengths={target_lengths} "
                            f"target_latent_T={int(latents.shape[2])} "
                            f"mask_sum={int(reference_mask_bool.sum().item())} "
                            "condition_changes_rope=False",
                            flush=True,
                        )

            elif multitask_type == 't2v':
                if kwargs.get('reorg_token', True):
                    gt_latents_concat = torch.zeros(latents.shape[0], latents.shape[1]//2, latents.shape[2], latents.shape[3], latents.shape[4]).to(latents.device)
                else:
                    gt_latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3], latents.shape[4]).to(latents.device)
            else:
                raise ValueError(f"multitask_type {multitask_type} not supported")

            if not native_t_concat:
                mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
                mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
                mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=latents.device)

        elif multitask_mask_training_type == "token_replace" and multitask_type == "i2v":
            if kwargs.get('reorg_token', True):
                cond_latents = cond_latents.repeat(1, 2, 1, 1, 1)
            else:
                cond_latents = cond_latents
        elif multitask_mask_training_type == "token_replace" and multitask_type == "range_condition":
            range_condition_mode = self.args.range_condition_mode
            assert range_condition_mode in ("soft_reference", "hard_legacy"), (
                f"range_condition_mode must be soft_reference or hard_legacy, got {range_condition_mode}"
            )
            if range_condition_mode != "hard_legacy":
                raise ValueError(
                    "token_replace + range_condition is hard_legacy-only. "
                    "Use concat for soft_reference range_condition."
                )
            # Unified PhysRoPE: arbitrary frame range conditioning
            condition_frame_ranges = kwargs.get("condition_frame_ranges", [])
            condition_src_frame_ranges = kwargs.get("condition_src_frame_ranges", None)
            if not condition_src_frame_ranges:
                condition_src_frame_ranges = list(condition_frame_ranges)
            assert cond_latents is not None, (
                "[Unified PhysRoPE] token_replace range_condition requires cond_latents."
            )
            assert multitask_mask is not None or condition_frame_ranges, (
                "[Unified PhysRoPE] token_replace range_condition requires either "
                "multitask_mask or condition_frame_ranges."
            )
            if multitask_mask is None and condition_frame_ranges:
                multitask_mask = get_multitask_mask_with_ranges(
                    latents.shape[2], condition_frame_ranges, device=latents.device
                )
            assert multitask_mask is not None and (multitask_mask == 1).any(), (
                "[Unified PhysRoPE] token_replace range_condition: multitask_mask has "
                f"no condition frames (all zeros). condition_frame_ranges={condition_frame_ranges}"
            )
            if kwargs.get('reorg_token', True) and cond_latents is not None:
                cond_latents = cond_latents.repeat(1, 2, 1, 1, 1) if cond_latents.shape[1] != latents.shape[1] else cond_latents
            cond_latents = cond_latents.to(device=latents.device, dtype=latents.dtype)
            extra_kwargs["condition_frame_ranges"] = condition_frame_ranges
            extra_kwargs["condition_src_frame_ranges"] = condition_src_frame_ranges
            extra_kwargs["range_condition_mode"] = range_condition_mode
            if "_rc_pipeline_debug_printed" not in self.__dict__:
                self._rc_pipeline_debug_printed = True
                print(f"[Unified PhysRoPE PIPELINE] range_condition init: "
                      f"mode={range_condition_mode}, "
                      f"condition_frame_ranges={condition_frame_ranges}, "
                      f"latents={tuple(latents.shape)}, "
                      f"cond_latents={'None' if cond_latents is None else tuple(cond_latents.shape)}, "
                      f"multitask_mask_sum={multitask_mask.sum().item() if multitask_mask is not None else 'None'}")

        if multitask_mask_training_type == "concat" or multitask_mask_training_type == "token_replace":
            # Extract reference image features if provided
            vision_states = None
            if reference_image is not None:
                reference_image = np.array(reference_image) if isinstance(reference_image, Image.Image) else reference_image
                if len(reference_image.shape) == 4:
                    reference_image = reference_image[0]
                H, W, C = reference_image.shape

                crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
                aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
                closest_size, closest_ratio = get_closest_ratio(H, W, aspect_ratios, crop_size_list)
                height, width = closest_size[0], closest_size[1]

                if self.vision_encoder is not None:
                    input_image_np = resize_and_center_crop(reference_image, target_width=width, target_height=height)
                    vision_states = self.vision_encoder.encode_images(input_image_np)
                    vision_states = vision_states.last_hidden_state.to(device=device, dtype=target_dtype)
                else:
                    vision_states = None

            # Repeat image features for batch size if needed
            if self.do_classifier_free_guidance and vision_states is not None:
                vision_states = vision_states.repeat(2, 1, 1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.val_disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.val_disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if multitask_mask_training_type == "token_replace" and multitask_type == "i2v":
            extra_kwargs["is_token_replace"] = True
        else:
            # range_condition does NOT use token_replace modulation:
            # first_frame_mask assumes cond tokens are at the front of the sequence,
            # but range_condition has cond tokens in the middle. hard_legacy
            # pinning is handled explicitly below; soft_reference never pins.
            extra_kwargs["is_token_replace"] = False

        # Unified PhysRoPE: pass condition_frame_ranges to model
        condition_frame_ranges = kwargs.get("condition_frame_ranges", [])
        if condition_frame_ranges:
            extra_kwargs["condition_frame_ranges"] = condition_frame_ranges
        # L4 src-range plumbing — Refrain._build_segment_layout uses this to override
        # cond segment time-pair from dst to src when src != dst.
        condition_src_frame_ranges = kwargs.get("condition_src_frame_ranges", None)
        if condition_src_frame_ranges:
            extra_kwargs["condition_src_frame_ranges"] = condition_src_frame_ranges
        condition_physical_dst_frame_ranges = kwargs.get(
            "condition_physical_dst_frame_ranges", None
        )
        if condition_physical_dst_frame_ranges:
            extra_kwargs["condition_physical_dst_frame_ranges"] = (
                condition_physical_dst_frame_ranges
            )
        condition_kinds = kwargs.get("condition_kinds", None)
        if condition_kinds:
            extra_kwargs["condition_kinds"] = condition_kinds
        condition_seg_ids = kwargs.get("condition_seg_ids", None)
        if condition_seg_ids:
            extra_kwargs["condition_seg_ids"] = condition_seg_ids
        extra_kwargs["range_condition_temporal_pack_mode"] = kwargs.get(
            "range_condition_temporal_pack_mode",
            getattr(self.args, "range_condition_temporal_pack_mode", "legacy_trilinear"),
        )

        # if is_progress_bar:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            import tqdm
            for i, t in tqdm.tqdm(enumerate(timesteps)):
                if self.interrupt:
                    continue

                if multitask_mask_training_type == "concat":
                    if multitask_type == "range_condition":
                        range_condition_mode = self.args.range_condition_mode
                        assert range_condition_mode in ("soft_reference", "hard_legacy"), (
                            f"range_condition_mode must be soft_reference or hard_legacy, got {range_condition_mode}"
                        )
                    # HARD LEGACY: pre-forward pin preserve_mask frames in the
                    # main latent for exact insert reproduction. soft_reference
                    # keeps conditions only in gt_latents_concat.
                    if multitask_type == "range_condition" and range_condition_mode == "hard_legacy":
                        assert cond_latents is not None, "hard_legacy range_condition requires cond_latents"
                        assert multitask_mask is not None, "hard_legacy range_condition requires preserve_mask"
                        assert multitask_mask.ndim == 1 and multitask_mask.shape[0] == latents.shape[2], (
                            f"hard_legacy preserve_mask shape {tuple(multitask_mask.shape)} "
                            f"must be [T={latents.shape[2]}]"
                        )
                        preserve_mask_bool = (multitask_mask == 1)
                        cond_latents_main = cond_latents
                        if kwargs.get('reorg_token', True) and cond_latents_main.shape[1] != latents.shape[1]:
                            assert cond_latents_main.shape[1] * 2 == latents.shape[1], (
                                f"Cannot reorg hard_legacy cond channels {cond_latents_main.shape[1]} "
                                f"to latents channels {latents.shape[1]}"
                            )
                            cond_latents_main = cond_latents_main.repeat(1, 2, 1, 1, 1)
                        assert cond_latents_main.shape == latents.shape, (
                            f"hard_legacy cond_latents shape {tuple(cond_latents_main.shape)} "
                            f"must match latents shape {tuple(latents.shape)}"
                        )
                        latents[:, :, preserve_mask_bool] = cond_latents_main[:, :, preserve_mask_bool]
                    if native_t_concat:
                        latents_concat, packed_reference_t = pack_native_reference_input(
                            latents,
                            cond_latents,
                            extra_kwargs["native_reference_packed_ranges"],
                            condition_frame_ranges,
                        )
                        assert packed_reference_t == native_reference_t
                    else:
                        assert gt_latents_concat.shape[0] == latents.shape[0], (
                            f"gt_latents_concat batch {gt_latents_concat.shape[0]} must match latents batch {latents.shape[0]}"
                        )
                        assert gt_latents_concat.shape[2:] == latents.shape[2:], (
                            f"gt_latents_concat tail {tuple(gt_latents_concat.shape[2:])} "
                            f"must match latents tail {tuple(latents.shape[2:])}"
                        )
                        assert mask_concat.shape[0] == latents.shape[0] and mask_concat.shape[1] == 1, (
                            f"mask_concat shape {tuple(mask_concat.shape)} must have batch {latents.shape[0]} and channel 1"
                        )
                        assert mask_concat.shape[2:] == latents.shape[2:], (
                            f"mask_concat tail {tuple(mask_concat.shape[2:])} "
                            f"must match latents tail {tuple(latents.shape[2:])}"
                        )
                        latents_concat = torch.concat([latents, gt_latents_concat, mask_concat], dim=1)
                    latent_model_input = torch.cat([latents_concat] * 2) if self.do_classifier_free_guidance else latents_concat
                        
                elif multitask_mask_training_type == "token_replace" and multitask_type == "i2v":
                    latents = torch.concat([cond_latents, latents[:, :, 1:, :, :]], dim=2)
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                elif multitask_mask_training_type == "token_replace" and multitask_type == "range_condition":
                    assert self.args.range_condition_mode == "hard_legacy", (
                        "token_replace + range_condition denoising is hard_legacy-only"
                    )
                    # In-place pin: no grad in inference, safe to write directly
                    cond_mask = (multitask_mask == 1)
                    latents[:, :, cond_mask] = cond_latents[:, :, cond_mask]
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                else:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )
                # predict the noise residual
                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    output = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latent_model_input,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        prompt_embeds,
                        prompt_embeds_2,
                        prompt_mask,  # [2, 256]fpdb
                        vision_states=vision_states,
                        mask_type=multitask_type,
                        guidance=guidance_expand,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                    )
                    noise_pred = output[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if native_t_concat:
                    target_t = int(latents.shape[2])
                    noise_pred = slice_native_target_output(
                        noise_pred, native_reference_t, target_t
                    )
                    if self.do_classifier_free_guidance:
                        noise_pred_text = slice_native_target_output(
                            noise_pred_text, native_reference_t, target_t
                        )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if multitask_mask_training_type == "token_replace" and multitask_type == "i2v":
                    latents = self.scheduler.step(noise_pred[:, :, 1:, :, :], t, latents[:, :, 1:, :, :], **extra_step_kwargs, return_dict=False)[0]
                    latents = torch.concat([cond_latents, latents], dim=2)
                elif multitask_mask_training_type == "token_replace" and multitask_type == "range_condition":
                    assert self.args.range_condition_mode == "hard_legacy", (
                        "token_replace + range_condition scheduler step is hard_legacy-only"
                    )
                    gen_mask = (multitask_mask == 0)
                    cond_mask = (multitask_mask == 1)
                    noise_pred_gen = noise_pred[:, :, gen_mask, ...]
                    latents_gen = latents[:, :, gen_mask, ...]
                    latents_gen = self.scheduler.step(
                        noise_pred_gen, t, latents_gen, **extra_step_kwargs, return_dict=False
                    )[0]
                    # In-place write back: no grad in inference, avoids per-step clone
                    latents[:, :, gen_mask] = latents_gen
                    latents[:, :, cond_mask] = cond_latents[:, :, cond_mask]
                else:
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # HARD LEGACY: post-step re-pin preserve_mask frames. Default
                # soft_reference does not write condition latents into the main latent.
                if (
                    multitask_mask_training_type == "concat"
                    and multitask_type == "range_condition"
                    and self.args.range_condition_mode == "hard_legacy"
                ):
                    preserve_mask_bool = (multitask_mask == 1)
                    assert cond_latents is not None, "hard_legacy post-step pin requires cond_latents"
                    cond_latents_main = cond_latents
                    if kwargs.get('reorg_token', True) and cond_latents_main.shape[1] != latents.shape[1]:
                        assert cond_latents_main.shape[1] * 2 == latents.shape[1], (
                            f"Cannot reorg hard_legacy cond channels {cond_latents_main.shape[1]} "
                            f"to latents channels {latents.shape[1]}"
                        )
                        cond_latents_main = cond_latents_main.repeat(1, 2, 1, 1, 1)
                    assert cond_latents_main.shape == latents.shape, (
                        f"hard_legacy post-step cond_latents shape {tuple(cond_latents_main.shape)} "
                        f"must match latents shape {tuple(latents.shape)}"
                    )
                    latents[:, :, preserve_mask_bool] = cond_latents_main[:, :, preserve_mask_bool]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # if multitask_type is fl2v, remove the last index dummy latent
        if multitask_type == "fl2v":
            latents = latents[:, :, :-1, :, :]

        self.last_generated_latent_shape = list(latents.shape)
        if int(os.environ.get('RANK', '0')) == 0:
            logger.info(f"[generation LATENTS] shape={self.last_generated_latent_shape}")

        if not output_type == "latent":
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                if kwargs['reorg_token']:
                    latents = rearrange(latents, 'b c f h w -> b f c h w')
                    latents = rearrange(latents, 'b f (n c) h w -> b (f n) c h w', n=2)
                    latents = rearrange(latents, 'b f c h w -> b c f h w')
                    latents = latents[:, :, 1:]
                
                use_vae_tiling = (
                    enable_tiling
                    and hasattr(self.vae, "enable_tiling")
                    and hasattr(self.vae, "disable_tiling")
                )
                if use_vae_tiling:
                    self.vae.enable_tiling()
                try:
                    image = self.vae.decode(
                        latents, return_dict=False
                    )[0]
                finally:
                    if use_vae_tiling:
                        self.vae.disable_tiling()
        else:
            image = latents

        if output_type == "pil": # 生成图片，转PIL输出
            image = image.squeeze(2)
            image = self.image_processor.postprocess(image, output_type='pil')
        else:
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().float()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)
