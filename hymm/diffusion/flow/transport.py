import enum
import math
from typing import Callable

import numpy as np
import torch as th
import torch

from . import path
from .integrators import ode, sde
from .utils import mean_flat
from hymm.utils.multitask_utils import merge_tensor_by_mask
from hymm.utils.native_reference import (
    as_bool,
    pack_native_reference_input,
    slice_native_target_output,
)
from hymm.utils.reference_temporal_pack import REFERENCE_TIMELINE, normalize_temporal_pack_mode

from hymm.parallelism.engines.parallel_engine import BaseParallelEngine

from hymm.parallelism.utils import sync_object_for_parallel_training as putorch_sync_object_for_parallel_training
from hymm.parallelism.utils import get_parallel_state
from torch.nn import functional as F
__all__ = ["ModelType", "PathType", "WeightType", "Transport", "Sampler", "SNRType"]


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class SNRType(enum.Enum):
    UNIFORM = enum.auto()
    LOGNORM = enum.auto()
    MIX = enum.auto()


# Copied from Flux.1
def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


# Copied from Flux.1
def time_shift(mu: float, sigma: float, t: th.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class Transport:
    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps, snr_type,
                 training_timesteps=1000, reverse_time_schedule=False, image_shift=1.0, video_shift=None, audio_shift=1.0, t2va_shift=1.0, reverse=False):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type](reverse=reverse)
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        self.snr_type = snr_type
        self.t2va_shift = t2va_shift
        # timestep shift: http://arxiv.org/abs/2403.03206
        self.image_shift = image_shift # set image shift to be the same as flow shift
        if video_shift is None: video_shift = image_shift # if video shift is not given, set it to be the same as flow shift
        self.video_shift = video_shift
        self.audio_shift = audio_shift
        self.reverse = reverse

        self.training_timesteps = training_timesteps
        self.reverse_time_schedule = reverse_time_schedule

    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """
        if isinstance(x1, (list, tuple)):
            x0 = [th.randn_like(img_start) for img_start in x1]
        else:
            x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if self.snr_type == SNRType.UNIFORM:
            t = th.rand((len(x1),)) * (t1 - t0) + t0
        elif self.snr_type == SNRType.LOGNORM:
            u = th.normal(mean=0.0, std=1.0, size=(len(x1),))
            t = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0

        elif self.snr_type == SNRType.MIX:
            u = th.normal(mean=0.0, std=1.0, size=(len(x1),))
            t_lognorm = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
            t0_ = t0 + 0.2 # 从 t0 + 0.2 ~ t1 均衡采样(固定0.2, 降低低噪采样概率)
            t_clip_uniform = th.rand((len(x1),)) * (t1 - t0_) + t0_
            mask = (th.rand((len(x1),)) > 0.5).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform

        else:
            raise ValueError(f"Unknown snr type: {self.snr_type}")

        if self.shift != 1.:
            if self.reverse:
                # xt = (1 - t) * x1 + t * x0
                t_ = (self.shift * t) / (1 + (self.shift - 1) * t)
            else:
                # xt = t * x1 + (1 - t) * x0
                t_ = t / (self.shift - (self.shift - 1) * t)

            t = t_

        t = t.to(x1.device).float() # 保证t的精度
        return t, x0, x1
    
    def sample_t2va(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """
        if isinstance(x1, (list, tuple)):
            x0 = [th.randn_like(img_start) for img_start in x1]
        else:
            x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if self.snr_type == SNRType.UNIFORM:
            t = th.rand((len(x1),)) * (t1 - t0) + t0
        elif self.snr_type == SNRType.LOGNORM:
            u = th.normal(mean=0.0, std=1.0, size=(len(x1),))
            t = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
        elif self.snr_type == SNRType.MIX:
            u = th.normal(mean=0.0, std=1.0, size=(len(x1),))
            t_lognorm = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
            t0_ = t0 + 0.2 # 从 t0 + 0.2 ~ t1 均衡采样(固定0.2, 降低低噪采样概率)
            t_clip_uniform = th.rand((len(x1),)) * (t1 - t0_) + t0_
            mask = (th.rand((len(x1),)) > 0.5).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform
        else:
            raise ValueError(f"Unknown snr type: {self.snr_type}")

        if self.shift != 1.:
            if self.reverse:
                # xt = (1 - t) * x1 + t * x0
                t_ = (self.shift * t) / (1 + (self.shift - 1) * t)
            else:
                # xt = t * x1 + (1 - t) * x0
                t_ = t / (self.shift - (self.shift - 1) * t)

            t = t_

        t = t.to(x1[0])
        return t, x0, x1

    def get_model_t(self, t):
        if self.reverse_time_schedule:
            return (1 - t) * self.training_timesteps
        else:
            return t * self.training_timesteps

    def training_losses_t2va(self, 
                             model, 
                             x1, # video
                             y1, # audio
                             args=None,
                             model_kwargs=None, 
                             timestep=None, 
                             data_type="t2va",
                             multitask_mask=None, 
                             cond_latents=None, 
                             reorg_token=False, 
                             multitask_mask_training_type=None, 
                             mask_type=None,
                            drop_first_frame_loss=False,
                            parallel_dims=None,
                            video_loss_weight=1.0,
                            audio_loss_weight=1.0,):
        """Loss for training the score model
        Args:
            model: backbone model; could be score, noise, or velocity
            x1: datapoint
            model_kwargs: additional arguments for the model
            timestep: the timestep at which to evaluate loss.
        """

        if not isinstance(model, BaseParallelEngine):
            raise TypeError("Refrain training expects a BaseParallelEngine-wrapped model.")
        parallel_state = None
        
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if data_type == "image":
                self.shift = self.image_shift
            elif data_type == "video":
                self.shift = self.video_shift
            elif data_type == 'audio':
                self.shift = self.audio_shift
            elif data_type == 't2va':
                self.shift = self.t2va_shift
            else:
                raise NotImplementedError
        
            if model_kwargs == None:
                model_kwargs = {}

            t, x0, x1 = self.sample_t2va(x1)
            _, y0, y1 = self.sample_t2va(y1)

            if timestep is not None:
                t = th.ones_like(t) * timestep

            t, xt, ut = self.path_sampler.plan(t, x0, x1)
            t, yt, vt = self.path_sampler.plan(t, y0, y1)

            # sp group rank 内同步随机输入 t
            if parallel_dims is not None:
                t = sync_object_for_parallel_training(t, parallel_dims=parallel_dims)

            input_t = self.get_model_t(t)

            if multitask_mask is not None:
                if multitask_mask_training_type == "token_replace":
                    if "extra_kwargs" not in model_kwargs:
                        model_kwargs["extra_kwargs"] = {}
                    if x1.shape[2] > 1 and mask_type == "i2v":
                        xt = th.concat([cond_latents, xt[:, :, 1:, :, :]], dim=2)
                        model_kwargs["extra_kwargs"]["is_token_replace"] = True
                    else:
                        model_kwargs["extra_kwargs"]["is_token_replace"] = False
                elif multitask_mask_training_type == "concat":
                    if cond_latents is not None:
                        x1_concat = cond_latents.repeat(1, 1, x1.shape[2], 1, 1)
                        x1_concat[:,:,1:,:,:] = 0.0
                    else:
                        if reorg_token:
                            x1_concat = th.zeros(x1.shape[0], x1.shape[1]//2, x1.shape[2], x1.shape[3], x1.shape[4]).to(x1.device)
                        else:
                            x1_concat = th.zeros_like(x1).to(x1.device)
                    
                    mask_zeros = th.zeros(x1.shape[0], 1, x1.shape[2], x1.shape[3], x1.shape[4])
                    mask_ones = th.ones(x1.shape[0], 1, x1.shape[2], x1.shape[3], x1.shape[4])
                    mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=x1.device)
                    xt = th.concat([xt, x1_concat, mask_concat], dim=1)

                    if "extra_kwargs" not in model_kwargs:
                        model_kwargs["extra_kwargs"] = {}   
                    model_kwargs["extra_kwargs"]["is_token_replace"] = False
        
        else:
            t, x0, x1 = None, None, None
            y0, y1 = None, None
            model_kwargs = {'video_text_cond': None, 'audio_text_cond': None}
            xt = None
            yt = None
            ut = None
            vt = None
            input_t = None
            
            if "extra_kwargs" not in model_kwargs:
                model_kwargs["extra_kwargs"] = {}   
            model_kwargs["extra_kwargs"]["is_token_replace"] = False
        
        model_kwargs['ut'] = ut
        model_kwargs['vt'] = vt
        model_outputs = model(video_latents=xt, audio_latents=yt, timestep=input_t, **model_kwargs)

        if not parallel_state.is_pipeline_last_stage():
            terms = None
            return model_outputs, terms

        # ut, vt for parallel training
        output_x, output_y, ut, vt = model_outputs
        terms = {}
        if self.model_type == ModelType.VELOCITY:
            terms["loss_x"] = mean_flat(((output_x - ut) ** 2))
            terms["loss_y"] = mean_flat(((output_y - vt) ** 2))
            terms['loss'] = terms['loss_x'] * video_loss_weight + terms['loss_y'] * audio_loss_weight # TODO: loss ratio
        else:
            _, drift_var_x = self.path_sampler.compute_drift(xt, t)
            _, drift_var_y = self.path_sampler.compute_drift(yt, t)

            sigma_t_x, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            sigma_t_y, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, yt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight_x = (drift_var_x / sigma_t_x) ** 2
                weight_y = (drift_var_y / sigma_t_y) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight_x = drift_var_x / (sigma_t_x ** 2)
                weight_y = drift_var_y / (sigma_t_y ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight_x = 1
                weight_y = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms['loss_x'] = mean_flat(weight_x * ((output_x - x0) ** 2))
                terms['loss_y'] = mean_flat(weight_y * ((output_y - y0) ** 2))
            else:
                terms['loss_x'] = mean_flat(weight_x * ((output_x * sigma_t_x + x0) ** 2))
                terms['loss_y'] = mean_flat(weight_y * ((output_y * sigma_t_y + y0) ** 2))
            terms['loss'] = terms['loss_x'] * video_loss_weight + terms['loss_y'] * audio_loss_weight # TODO: loss ratio

        terms['t'] = t

        # for video_micro_batch_size > 1, need to average the loss
        loss = 0
        diffusion_loss = terms['loss'].mean()
        loss += diffusion_loss
        terms['diffusion_loss'] = diffusion_loss
        video_loss = terms['loss_x'].mean()
        audio_loss = terms['loss_y'].mean()
        terms['video_loss'] = video_loss
        terms['audio_loss'] = audio_loss
        if 'moe_loss' in terms:
            loss += terms['moe_loss']* args.moe_loss_weight

        terms['avg_loss'] = loss    
        
        return model_outputs, terms

    def training_losses(self, model, x1, args, model_kwargs=None, timestep=None, data_type="image",
                        multitask_mask=None, cond_latents=None, last_frame_cond_latents=None,
                        reorg_token=False, multitask_mask_training_type=None, mask_type=None,
                        drop_first_frame_loss=False,
                        parallel_dims=None, videoid=None, repa_feats=None):
        """Loss for training the score model
        Args:
            model: backbone model; could be score, noise, or velocity
            x1: datapoint
            model_kwargs: additional arguments for the model
            timestep: the timestep at which to evaluate loss.
        """
        train_args = args
        native_t_concat = as_bool(
            getattr(train_args, "range_condition_native_t_concat", False)
        )
        temporal_pack_mode = normalize_temporal_pack_mode(
            (model_kwargs or {}).get("extra_kwargs", {}).get(
                "range_condition_temporal_pack_mode",
                getattr(
                    train_args,
                    "range_condition_temporal_pack_mode",
                    "legacy_trilinear",
                ),
            )
        )
        native_reference_t = 0

        # t, xt, ut use fp32
        x1 = x1.float()

        if not isinstance(model, BaseParallelEngine):
            raise TypeError("Refrain training expects a BaseParallelEngine-wrapped model.")
        parallel_state = None
        
        # add dummy x1 latent
        if multitask_mask_training_type == "concat" and mask_type == "fl2v":
            x1 = th.cat([x1, last_frame_cond_latents], dim=2) # 时序维度concat尾帧X1, Note: 新增尾部index会参与loss计算

        if isinstance(model, BaseParallelEngine) or parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if data_type == "image":
                self.shift = self.image_shift
            elif data_type == "video":
                self.shift = self.video_shift
            if model_kwargs == None:
                model_kwargs = {}

            # t, xt, ut use fp32
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t, x0, x1 = self.sample(x1)

                if timestep is not None:
                    t = th.ones_like(t) * timestep
                t, xt, ut = self.path_sampler.plan(t, x0, x1)

            # sp group rank 内同步随机输入 t
            if parallel_dims is not None:
                # putorch sync inputs
                t = putorch_sync_object_for_parallel_training(t, parallel_dims=parallel_dims)        

            # t, xt, ut use fp32
            with torch.amp.autocast('cuda', dtype=torch.float32):
                input_t = self.get_model_t(t)

            if multitask_mask is not None:
                if multitask_mask_training_type == "token_replace":
                    if "extra_kwargs" not in model_kwargs:
                        model_kwargs["extra_kwargs"] = {}
                    if x1.shape[2] > 1 and mask_type == "i2v":
                        xt = th.concat([cond_latents, xt[:, :, 1:, :, :]], dim=2)
                        model_kwargs["extra_kwargs"]["is_token_replace"] = True
                    elif x1.shape[2] > 1 and mask_type == "range_condition":
                        range_condition_mode = train_args.range_condition_mode
                        assert range_condition_mode in ("soft_reference", "hard_legacy"), (
                            f"range_condition_mode must be soft_reference or hard_legacy, got {range_condition_mode}"
                        )
                        if range_condition_mode != "hard_legacy":
                            raise ValueError(
                                "token_replace + range_condition is hard_legacy-only. "
                                "Use concat for soft_reference range_condition."
                            )
                        # HARD LEGACY: pin preserve_mask frames in xt for exact insert.
                        preserve_mask_bool = (multitask_mask == 1)
                        if cond_latents is not None:
                            cond_src = cond_latents.to(device=xt.device, dtype=xt.dtype)
                            assert cond_src.shape == xt.shape, (
                                f"token_replace hard_legacy cond_latents shape {tuple(cond_src.shape)} "
                                f"must match xt shape {tuple(xt.shape)}"
                            )
                            xt[:, :, preserve_mask_bool] = cond_src[:, :, preserve_mask_bool]
                        else:
                            xt[:, :, preserve_mask_bool] = x1[:, :, preserve_mask_bool]
                        # NOT using token_replace modulation: cond frames are in
                        # the middle, first_frame_mask requires them at the front.
                        # HARD LEGACY latent pinning is sufficient.
                        model_kwargs["extra_kwargs"]["is_token_replace"] = False
                    else:
                        model_kwargs["extra_kwargs"]["is_token_replace"] = False
                elif multitask_mask_training_type == "concat":
                    native_input_packed = False
                    if mask_type == "i2v":
                        assert cond_latents is not None
                        x1_concat = cond_latents.repeat(1, 1, x1.shape[2], 1, 1)
                        x1_concat[:,:,1:,:,:] = 0.0
                    elif mask_type == "fl2v":
                        assert cond_latents is not None
                        x1_concat = cond_latents.repeat(1, 1, x1.shape[2], 1, 1)
                        x1_concat[:,:,1:-1,:,:] = 0.0
                        x1_concat[:,:,-1:,:,:] = last_frame_cond_latents
                    elif mask_type == "range_condition":
                        range_condition_mode = train_args.range_condition_mode
                        assert range_condition_mode in ("soft_reference", "hard_legacy"), (
                            f"range_condition_mode must be soft_reference or hard_legacy, got {range_condition_mode}"
                        )
                        assert multitask_mask.ndim == 1 and multitask_mask.shape[0] == x1.shape[2], (
                            f"range_condition reference_mask shape {tuple(multitask_mask.shape)} "
                            f"must be [T={x1.shape[2]}]"
                        )
                        # reference_mask marks concat condition frames only.
                        # HARD LEGACY preserve_mask is opt-in and pins main xt below.
                        reference_mask_bool = (multitask_mask == 1)
                        assert bool(reference_mask_bool.any().item()), "range_condition reference_mask is empty"
                        if temporal_pack_mode == REFERENCE_TIMELINE:
                            assert not native_t_concat, (
                                "reference_timeline must remain on the fixed-T channel-concat path"
                            )
                            assert cond_latents is not None and cond_latents.shape == x1.shape, (
                                "reference_timeline condition canvas must exactly match target latent shape: "
                                f"condition={None if cond_latents is None else tuple(cond_latents.shape)} "
                                f"target={tuple(x1.shape)}"
                            )
                        if native_t_concat:
                            assert range_condition_mode == "soft_reference", (
                                "native T concat cannot be combined with hard_legacy pinning"
                            )
                            assert not reorg_token, (
                                "native T concat currently requires equal target/reference channel counts"
                            )
                            assert cond_latents is not None, (
                                "native T concat range_condition requires native cond_latents"
                            )
                            native_ranges = model_kwargs.get("extra_kwargs", {}).get(
                                "native_reference_packed_ranges", []
                            )
                            condition_ranges = model_kwargs.get("extra_kwargs", {}).get(
                                "condition_frame_ranges", []
                            )
                            xt, native_reference_t = pack_native_reference_input(
                                xt,
                                cond_latents,
                                native_ranges,
                                condition_ranges,
                            )
                            native_input_packed = True
                            model_kwargs["extra_kwargs"]["range_condition_native_t_concat"] = True
                            model_kwargs["extra_kwargs"]["native_reference_prefix_t"] = native_reference_t
                            if not getattr(self, "_native_t_concat_train_printed", False):
                                self._native_t_concat_train_printed = True
                                print(
                                    "[RangeCondition NATIVE-T TRAIN] "
                                    f"videoid={videoid} reference_T={native_reference_t} "
                                    f"target_T={x1.shape[2]} packed_T={xt.shape[2]} "
                                    f"native_ranges={native_ranges} dst_ranges={condition_ranges}",
                                    flush=True,
                                )
                        else:
                            if reorg_token:
                                x1_concat = th.zeros(
                                    x1.shape[0], x1.shape[1]//2, x1.shape[2],
                                    x1.shape[3], x1.shape[4],
                                    device=x1.device, dtype=x1.dtype,
                                )
                            else:
                                x1_concat = th.zeros_like(x1)
                            assert cond_latents is not None, "range_condition requires cond_latents for concat reference"
                            cond_src = cond_latents.to(device=x1_concat.device, dtype=x1_concat.dtype)
                            assert cond_src.shape[0] == x1_concat.shape[0], (
                                f"cond_latents batch {cond_src.shape[0]} must match x1_concat batch {x1_concat.shape[0]}"
                            )
                            assert cond_src.shape[1] == x1_concat.shape[1], (
                                f"cond_latents channels {cond_src.shape[1]} must match concat reference channels {x1_concat.shape[1]}"
                            )
                            assert cond_src.shape[2:] == x1_concat.shape[2:], (
                                f"cond_latents shape tail {tuple(cond_src.shape[2:])} "
                                f"must match concat reference tail {tuple(x1_concat.shape[2:])}"
                            )
                            x1_concat[:, :, reference_mask_bool] = cond_src[:, :, reference_mask_bool]
                            if range_condition_mode == "hard_legacy":
                                # HARD LEGACY: preserve_mask is the reference mask reused
                                # to pin main xt. Soft reference never writes into xt.
                                preserve_mask_bool = reference_mask_bool
                                cond_src_xt = cond_latents.to(device=xt.device, dtype=xt.dtype)
                                if reorg_token and cond_src_xt.shape[1] != xt.shape[1]:
                                    assert cond_src_xt.shape[1] * 2 == xt.shape[1], (
                                        f"Cannot reorg hard_legacy cond channels {cond_src_xt.shape[1]} "
                                        f"to xt channels {xt.shape[1]}"
                                    )
                                    cond_src_xt = cond_src_xt.repeat(1, 2, 1, 1, 1)
                                assert cond_src_xt.shape == xt.shape, (
                                    f"hard_legacy cond_latents shape {tuple(cond_src_xt.shape)} "
                                    f"must match xt shape {tuple(xt.shape)}"
                                )
                                xt[:, :, preserve_mask_bool] = cond_src_xt[:, :, preserve_mask_bool]
                        range_condition_transport_debug = False
                        if range_condition_transport_debug:
                            debug_cond_shape = (
                                tuple(cond_latents.shape)
                                if native_t_concat else tuple(cond_src.shape)
                            )
                            debug_reference_shape = (
                                None if native_t_concat else tuple(x1_concat.shape)
                            )
                            print(
                                f"[RangeCondition TRANSPORT DEBUG] videoid={videoid} "
                                f"mode={range_condition_mode} mask_type={mask_type} "
                                f"reference_frames={int(reference_mask_bool.sum().item())}/{x1.shape[2]} "
                                f"cond_latents_shape={debug_cond_shape} "
                                f"x1_concat_shape={debug_reference_shape} xt_shape={tuple(xt.shape)} "
                                f"hard_pin={'ON' if range_condition_mode == 'hard_legacy' else 'OFF'} "
                                f"soft_reference={'ON' if range_condition_mode == 'soft_reference' else 'OFF'}",
                                flush=True,
                            )
                    elif mask_type == "t2v":
                        if reorg_token:
                            x1_concat = th.zeros(x1.shape[0], x1.shape[1]//2, x1.shape[2], x1.shape[3], x1.shape[4]).to(x1.device)
                        else:
                            x1_concat = th.zeros_like(x1).to(x1.device)
                    else:
                        raise ValueError(f"not supported mask_type={mask_type}")
                    if not native_input_packed:
                        mask_zeros = th.zeros(x1.shape[0], 1, x1.shape[2], x1.shape[3], x1.shape[4])
                        mask_ones = th.ones(x1.shape[0], 1, x1.shape[2], x1.shape[3], x1.shape[4])
                        mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=x1.device)
                        xt = th.concat([xt, x1_concat, mask_concat], dim=1)

                    if "extra_kwargs" not in model_kwargs:
                        model_kwargs["extra_kwargs"] = {}
                    model_kwargs["extra_kwargs"]["is_token_replace"] = False
        else:
            t, x0, x1 = None, None, None
            model_kwargs = {'text_states': None, 'text_states_2': None, 'encoder_attention_mask': None}
            xt = None
            ut = None
            input_t = None

            if "extra_kwargs" not in model_kwargs:
                model_kwargs["extra_kwargs"] = {}
            model_kwargs["extra_kwargs"]["is_token_replace"] = False



        def loss_closure(model_forward_out, model_input):

            features_list = None
            if isinstance(model_forward_out, tuple):
                model_output = model_forward_out[0]
                if len(model_forward_out) > 1:
                    features_list = model_forward_out[1]
            else:
                model_output = model_forward_out
            ut = model_input['kwargs']['ut']

            if native_t_concat and multitask_mask_training_type == "concat" and mask_type == "range_condition":
                model_output = slice_native_target_output(
                    model_output, native_reference_t, x1.shape[2]
                )
                assert model_output.shape == ut.shape, (
                    f"native target output {tuple(model_output.shape)} must match velocity target {tuple(ut.shape)}"
                )

            if multitask_mask_training_type == "token_replace" and x1.shape[2] > 1 and mask_type == "i2v" and drop_first_frame_loss:
                assert self.model_type == ModelType.VELOCITY, f"self.model_type: {self.model_type} must be ModelType.VELOCITY"
                model_output = model_output[:, :, 1:, :, :]
                ut = ut[:, :, 1:, :, :]

            if multitask_mask_training_type in ("token_replace", "concat") and x1.shape[2] > 1 and mask_type == "range_condition":
                # Unified PhysRoPE loss semantics:
                # reference_mask tells the model where concat conditions exist.
                # loss_mask is separate; soft_reference must train condition
                # frames too, while hard_legacy may reproduce old gen-only loss.
                assert self.model_type == ModelType.VELOCITY
                range_condition_mode = train_args.range_condition_mode
                assert range_condition_mode in ("soft_reference", "hard_legacy"), (
                    f"range_condition_mode must be soft_reference or hard_legacy, got {range_condition_mode}"
                )
                cond_loss_weight = float(train_args.unified_physrope_cond_loss_weight)
                assert cond_loss_weight >= 0.0, (
                    f"unified_physrope_cond_loss_weight must be >= 0.0, got {cond_loss_weight}"
                )
                if range_condition_mode == "soft_reference":
                    assert cond_loss_weight > 0.0, (
                        "soft_reference range_condition requires condition frames to participate in loss; "
                        "set unified_physrope_cond_loss_weight > 0.0, preferably 1.0"
                    )
                assert "extra_kwargs" in model_kwargs, "model_kwargs must contain extra_kwargs for range_condition"
                if "_rc_loss_frame_weights" in model_kwargs["extra_kwargs"]:
                    del model_kwargs["extra_kwargs"]["_rc_loss_frame_weights"]

                assert multitask_mask.ndim == 1 and multitask_mask.shape[0] == x1.shape[2], (
                    f"range_condition reference_mask shape {tuple(multitask_mask.shape)} "
                    f"must be [T={x1.shape[2]}]"
                )
                reference_mask_bool = (multitask_mask == 1)
                assert bool(reference_mask_bool.any().item()), "range_condition reference_mask is empty"
                loss_mask_bool = (multitask_mask == 0)
                n_gen = loss_mask_bool.sum().item()
                n_total = x1.shape[2]
                assert native_t_concat or temporal_pack_mode == REFERENCE_TIMELINE or n_gen > 0, (
                    f"[Unified PhysRoPE] All frames are condition frames — "
                    f"no generation frames for loss. mask={multitask_mask.tolist()}"
                )
                range_condition_loss_debug = False
                if range_condition_loss_debug:
                    print(f"[Unified PhysRoPE LOSS] range_condition: "
                          f"mode={range_condition_mode}, "
                          f"total_frames={n_total}, "
                          f"gen_frames={n_gen}, "
                          f"cond_frames={n_total - n_gen}, "
                          f"gen_ratio={n_gen / n_total:.2f}, "
                          f"cond_loss_weight={cond_loss_weight}, "
                          f"loss_mode={'hard_gen_only' if cond_loss_weight == 0.0 else ('full_frame' if cond_loss_weight >= 1.0 else f'weighted_{cond_loss_weight}')}",
                          flush=True)

                if native_t_concat:
                    # Reference tokens are an external prefix and are removed above;
                    # every remaining target frame is generated and receives loss.
                    pass
                elif cond_loss_weight == 0.0:
                    assert range_condition_mode == "hard_legacy", (
                        "cond_loss_weight=0.0 is valid only for hard_legacy reproduction"
                    )
                    # HARD LEGACY: loss_mask excludes preserved frames.
                    model_output = model_output[:, :, loss_mask_bool, ...]
                    ut = ut[:, :, loss_mask_bool, ...]
                elif cond_loss_weight < 1.0:
                    # Weighted soft reference: keep all frames but apply per-frame loss weight.
                    # Build weight tensor: gen frames = 1.0, cond frames = cond_loss_weight.
                    # Normalize weights so their temporal mean = 1.0. This ensures
                    # mean_flat() divides by the effective element count
                    # (n_gen + w * n_cond) instead of n_total, making the loss
                    # scale continuous from w=0 (gen-only) to w=1 (full-frame).
                    n_cond = n_total - n_gen
                    _loss_frame_weights = torch.ones(1, 1, n_total, 1, 1, device=model_output.device, dtype=model_output.dtype)
                    _loss_frame_weights[:, :, reference_mask_bool] = cond_loss_weight
                    # Normalize: effective_count = n_gen + w * n_cond;
                    # scale = n_total / effective_count so mean over T is 1.0
                    _effective = float(n_gen) + cond_loss_weight * float(n_cond)
                    assert _effective > 0.0, "range_condition loss effective element count must be positive"
                    _loss_frame_weights = _loss_frame_weights * (float(n_total) / _effective)
                    # Store for use in loss_tensor multiplication below
                    model_kwargs["extra_kwargs"]["_rc_loss_frame_weights"] = _loss_frame_weights
                # cond_loss_weight >= 1.0: full-frame loss (OmniWeaving style).
                # No loss_mask is applied.

            # fl2v remove the last dummy token
            #if multitask_mask_training_type == "concat" and mask_type == "fl2v":
            #    model_output = model_output[:, :, :-1, :, :]

            if multitask_mask is None:
                if not get_parallel_state().pp_enabled:
                    pass
                    # assert model_output.size() == xt.size(), f"Output shape from model does not match input shape: " \
                    #                                     f"{model_output.size()} != {xt.size()}"

            terms = {}
            if self.model_type == ModelType.VELOCITY:
                # loss_tensor dtype is fp32
                loss_tensor = (model_output.float() - ut) ** 2

                # Apply Unified PhysRoPE per-frame loss weights (for weighted cond frame loss)
                _rc_loss_frame_weights = None
                if "extra_kwargs" in model_kwargs and "_rc_loss_frame_weights" in model_kwargs["extra_kwargs"]:
                    _rc_loss_frame_weights = model_kwargs["extra_kwargs"]["_rc_loss_frame_weights"]
                if _rc_loss_frame_weights is not None:
                    loss_tensor = loss_tensor * _rc_loss_frame_weights.float()

                # Apply recreation mask if available
                recreation_mask = model_kwargs.get("extra_kwargs", {}).get("recreation_mask", None)
                enable_recreation_mask = model_kwargs.get("extra_kwargs", {}).get("enable_recreation_mask", False)
                
                # 开启二创loss的数据条件：
                # 1. part_version为part1, t2v/i2v/fl2v都开启
                # 2. part_version不为part1, 且mask_type为i2v
                # 3. part_version为None, 不区分part版本时，例如refrain1.5 训练
                part_version = model_kwargs.get("extra_kwargs", {}).get("part_version", [])
                if part_version is None:
                    part_enable_mask = []
                else:
                    part_enable_mask = [part == "part1" or (part != "part1" and mask_type == "i2v") or (part is None) for part in part_version]

                #print(f"part_version: {part_version}, part_enable_mask: {part_enable_mask}, mask_type: {mask_type}, data_type: {data_type}")
                
                if enable_recreation_mask and recreation_mask is not None and data_type == "video":
                    assert len(recreation_mask.shape) == 4, f"recreation_mask shape: {recreation_mask.shape} must be [B, T, H, W]"
                    if mask_type == "fl2v":
                        recreation_mask = th.cat([recreation_mask, recreation_mask[:,0:1, :, :]], dim=1) # add dummy latent for fl2v
                    recreation_mask_expanded = recreation_mask.unsqueeze(1).expand(-1, loss_tensor.shape[1], -1, -1, -1)
                    
                    # Create inverse mask (1 for unmasked regions, 0 for masked)
                    inverse_mask = (~recreation_mask_expanded).float() 
                    if len(part_enable_mask) == 0:
                        masked_loss_tensor = loss_tensor * inverse_mask
                    elif len(part_enable_mask) == loss_tensor.shape[0]:
                        part_enable_mask_tensor = th.tensor(part_enable_mask, device=loss_tensor.device, dtype=th.bool).view(-1, 1, 1, 1, 1)
                        mask_indices = part_enable_mask_tensor.expand_as(loss_tensor)
                        masked_loss_tensor = th.where(
                            mask_indices,
                            loss_tensor * inverse_mask,  # True
                            loss_tensor                  # False
                        )
                    else:
                        raise ValueError(f"part_enable_mask length: {len(part_enable_mask)} must be equal to batch size: {loss_tensor.shape[0]}")

                    #mask_to_print = inverse_mask[0, 0, 0]  # [H, W]
                    #mask_str = ""
                    #for row in mask_to_print:
                    #    mask_str += "".join(["█" if v > 0.5 else " " for v in row]) + "\n"
                    #print(f"Rank: {th.distributed.get_rank()}, videoid: {videoid}, Inverse Recreation mask (batch 0, frame 0):\n" + mask_str)
                    
                    # Average only over non-masked regions PER SAMPLE, then mean across batch
                    reduce_dims = tuple(range(1, masked_loss_tensor.ndim))  # sum over C, T, H, W
                    loss_sum_per_sample = masked_loss_tensor.sum(dim=reduce_dims)  # [B]
                    valid_per_sample = inverse_mask.sum(dim=reduce_dims)           # [B]
                    per_sample_loss = loss_sum_per_sample / th.clamp(valid_per_sample, min=1.0)
                    # If a sample has no valid elements, set its loss to 0
                    per_sample_loss = th.where(valid_per_sample > 0, per_sample_loss, th.zeros_like(per_sample_loss))
                    terms["loss"] = per_sample_loss
                else:
                    terms["loss"] = mean_flat(loss_tensor)
            else:
                if get_parallel_state().pp_enabled:
                    raise NotImplementedError
                _, drift_var = self.path_sampler.compute_drift(xt, t)
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
                if self.loss_type in [WeightType.VELOCITY]:
                    weight = (drift_var / sigma_t) ** 2
                elif self.loss_type in [WeightType.LIKELIHOOD]:
                    weight = drift_var / (sigma_t ** 2)
                elif self.loss_type in [WeightType.NONE]:
                    weight = 1
                else:
                    raise NotImplementedError()

                if self.model_type == ModelType.NOISE:
                    terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
                else:
                    terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
            
            if hasattr(model, 'loss_moe') and model.loss_moe:
                terms['moe_loss_tuple'] = model.loss_moe
                # terms['count_top'] = model.count_top

            if args.use_repa and model_input is not None:
                if repa_feats is not None and features_list is not None and len(features_list) > 0:
                    mid_hidden = features_list[len(features_list) // 2]
                    repa_vec = repa_feats.to(dtype=mid_hidden.dtype).detach()
                    repa_vec = F.normalize(repa_vec, dim=-1)
                    mid_hidden = F.normalize(mid_hidden, dim=-1)
                    repa_loss = mean_flat(-(repa_vec * mid_hidden).sum(dim=-1))
                    repa_loss = repa_loss.mean()
                    terms["repa_loss"] = repa_loss * args.repa_loss_weight

            loss = 0
            diffusion_loss = terms['loss'].mean()
            terms['diffusion_loss'] = diffusion_loss
            loss += diffusion_loss
            if 'moe_loss_tuple' in terms and args.use_moe:
                # TODO: Check loss implementation
                moe_loss = terms['moe_loss_tuple'][0] + terms['moe_loss_tuple'][1]
                terms['moe_loss'] = moe_loss
                loss += moe_loss * args.moe_loss_weight
            if args.use_repa and 'repa_loss' in terms:
                loss += terms['repa_loss']
            terms['total_loss'] = loss
            return loss, terms

        if isinstance(model, BaseParallelEngine):
            model.register_loss_closure(loss_closure)
            model_output = model(xt, input_t, ut=ut, **model_kwargs, repa_feats=repa_feats, output_features=args.use_repa)
            loss = model.get_cached_result('loss')
            terms = model.get_cached_result('loss_dict')
        else:
            model_output = model(xt, input_t, ut=ut, **model_kwargs, repa_feats=repa_feats, output_features=args.use_repa)
            loss, terms = loss_closure(model_output, model_input={'kwargs': {'ut': ut, 'repa_feats': repa_feats}})

        return model_output, terms

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = (
                lambda x, t, model, **kwargs: model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                model(x, t, **kwargs), x, t
            )
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):
        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        sde_drift = lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(
            x, t, model, **kwargs
        )

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][
                0
            ] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        time_shifting_factor=None,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_shifting_factor=time_shifting_factor,
        )
        self.ode = _ode
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
