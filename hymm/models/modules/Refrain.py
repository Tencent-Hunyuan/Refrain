from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import loguru
import math
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection, VisionProjection
from .attention import parallel_attention, preprocess_dynamic_ring_attention, postprocess_dynamic_ring_attention
from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer, LinearWarpforSingle, DenseWithMoEMLP
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner
from .posemb_layers import get_nd_rotary_pos_embed

from hymm.utils.parallel_states import get_sequence_parallel_state, nccl_info
# from hymm.utils.communications import all_gather
from hymm.parallelism.communications import all_gather, all_to_all_4D
from hymm.models.modules.byt5 import ByT5Mapper
from hymm.utils.native_reference import as_bool, build_native_temporal_positions
from hymm.utils.reference_temporal_pack import (
    REFERENCE_TIMELINE,
    build_reference_timeline_mapping,
    normalize_temporal_pack_mode,
)

from .moe.hunyuan_moe import HunYuanMoE

from .moe.hunyuan_moe import MoeConfig as MOEConfig

class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        moe_config: Optional[MOEConfig] = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        is_dense_with_moe_mlp=False,
        dense_with_moe_mlp_intermediate_size=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode
        self.moe_config = moe_config
        self.is_dense_with_moe_mlp = is_dense_with_moe_mlp
        
        if self.moe_config:
            assert self.moe_config.use_fused_moe
            # determine which MoE to use
            if self.moe_config.use_padded_moe:
                raise ValueError("The release uses the dense Refrain model; MoE kernels are unsupported.")
            moe_class = HunYuanMoE
        
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = get_norm_layer("layer_f32")(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_q = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.img_attn_k = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.img_attn_v = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = get_norm_layer("layer_f32")(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        if moe_config is None:
            if is_dense_with_moe_mlp:
                self.img_mlp = DenseWithMoEMLP(
                    hidden_size,
                    dense_with_moe_mlp_intermediate_size,
                )
            else:
                self.img_mlp = MLP(
                    hidden_size,
                    mlp_hidden_dim,
                    act_layer=get_activation_layer(mlp_act_type),
                    bias=True,
                    **factory_kwargs,
                )
        else:
            self.img_mlp = moe_class(moe_config)

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = get_norm_layer("layer_f32")(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_q = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.txt_attn_k = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.txt_attn_v = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = get_norm_layer("layer_f32")(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        import copy
        if moe_config is not None:
            txt_moe_config = copy.deepcopy(moe_config)
            if txt_moe_config.moe_pad_seqlen != 0:
                txt_moe_config.moe_pad_seqlen = 2000*5
        else:
            txt_moe_config = None

        if moe_config is None:
            if is_dense_with_moe_mlp:
                self.txt_mlp = DenseWithMoEMLP(
                    hidden_size,
                    dense_with_moe_mlp_intermediate_size,
                )
            else:
                self.txt_mlp = MLP(
                    hidden_size,
                    mlp_hidden_dim,
                    act_layer=get_activation_layer(mlp_act_type),
                    bias=True,
                    **factory_kwargs,
                )
        else:
            self.txt_mlp = moe_class(txt_moe_config)
        self.hybrid_seq_parallel_attn = None
        
        if moe_config is not None:
            # for test
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self.loss_moe = [torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)]
            self.count_top = [[torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)], [torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)]]

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False
        
    def get_moe_loss(self):
        return self.loss_moe

    def get_count_top(self):
        return self.count_top

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        freqs_cis_cond: tuple = None,
        img_physical_freqs_cis: tuple = None,
        txt_physical_freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        is_token_replace=False,
        token_replace_vec=None,
        first_frame_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_token_replace:
            img_mod1, token_replace_img_mod1 = self.img_mod(vec,
                                                            is_token_replace=is_token_replace,
                                                            token_replace_vec=token_replace_vec)
            (img_mod1_shift,
             img_mod1_scale,
             img_mod1_gate,
             img_mod2_shift,
             img_mod2_scale,
             img_mod2_gate) = img_mod1.chunk(6, dim=-1)
            (tr_img_mod1_shift,
             tr_img_mod1_scale,
             tr_img_mod1_gate,
             tr_img_mod2_shift,
             tr_img_mod2_scale,
             tr_img_mod2_gate) = token_replace_img_mod1.chunk(6, dim=-1)
        else:
            # vec.shape = torch.Size([1, 3072])
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)

        if is_token_replace:
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale, is_token_replace=is_token_replace,
                tr_shift=tr_img_mod1_shift, tr_scale=tr_img_mod1_scale,
                first_frame_mask=first_frame_mask
            )
        else:
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
            )

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)

        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:

            # def shrink_head(encoder_state, dim):
                # local_heads = encoder_state.shape[dim] // nccl_info.sp_size
                # return encoder_state.narrow(
                    # dim, nccl_info.rank_within_group * local_heads, local_heads
                # )

            # freqs_cis = (
                # shrink_head(freqs_cis[0], dim=0),
                # shrink_head(freqs_cis[1], dim=0),
            # )

            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Apply PhysRoPE (second rotation) on image tokens
        if img_physical_freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, img_physical_freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"phys img_qq: {img_qq.shape}, img_q: {img_q.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )

        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)

        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)


        if freqs_cis_cond is not None:
            txt_qq, txt_kk = apply_rotary_emb(txt_q, txt_k, freqs_cis_cond, head_first=False)
            assert (
                txt_qq.shape == txt_q.shape and txt_kk.shape == txt_k.shape
            ), f"txt_kk: {txt_qq.shape}, txt_q: {txt_q.shape}, txt_kk: {txt_kk.shape}, txt_k: {txt_k.shape}"
            txt_q, txt_k = txt_qq, txt_kk

        # Apply PhysRoPE (second rotation) on text tokens
        if txt_physical_freqs_cis is not None:
            txt_qq, txt_kk = apply_rotary_emb(txt_q, txt_k, txt_physical_freqs_cis, head_first=False)
            assert (
                txt_qq.shape == txt_q.shape and txt_kk.shape == txt_k.shape
            ), f"phys txt_qq: {txt_qq.shape}, txt_q: {txt_q.shape}"
            txt_q, txt_k = txt_qq, txt_kk

        if is_flash:
            attn_mode = 'flash'
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )

        # attention computation end
        # img shard(1), so use img_q shard(-1) instead
        img_attn, txt_attn = attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous()

        def apply_text_mlp(x):
            if self.moe_config is not None:
                return self.txt_mlp(x, used_token=text_mask)
            else:
                return self.txt_mlp(x)
        
        def apply_img_mlp(x):
            if self.moe_config is not None:
                return self.img_mlp(x)
            else:
                return self.img_mlp(x)


        if is_token_replace:
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate, is_token_replace=is_token_replace,
                                   tr_gate=tr_img_mod1_gate, first_frame_mask=first_frame_mask)
            img = img + apply_gate(
                apply_img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale, is_token_replace=is_token_replace,
                        tr_shift=tr_img_mod2_shift, tr_scale=tr_img_mod2_scale, first_frame_mask=first_frame_mask
                    )
                ),
                gate=img_mod2_gate, is_token_replace=is_token_replace,
                tr_gate=tr_img_mod2_gate, first_frame_mask=first_frame_mask
            )
        else:
            # Calculate the img bloks.
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
            img = img + apply_gate(
                apply_img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                    )
                ),
                gate=img_mod2_gate,
            )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            apply_text_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        
        if self.moe_config is not None:
            self.loss_moe[0] = self.img_mlp.get_moe_loss()[0] + self.txt_mlp.get_moe_loss()[0]
            self.loss_moe[1] = self.img_mlp.get_moe_loss()[1] + self.txt_mlp.get_moe_loss()[1]

            # self.count_top[0][0] = self.img_mlp.get_count_top()[0]
            # self.count_top[0][1] = self.img_mlp.get_count_top()[1]
            # self.count_top[1][0] = self.txt_mlp.get_count_top()[0]
            # self.count_top[1][1] = self.txt_mlp.get_count_top()[1]

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        moe_config: Optional[MOEConfig] = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        is_dense_with_moe_mlp=False,
        dense_with_moe_mlp_intermediate_size=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode
        self.moe_config = moe_config
        self.is_dense_with_moe_mlp = is_dense_with_moe_mlp
        
        if self.moe_config:
            if self.moe_config.use_padded_moe:
                raise ValueError("The release uses the dense Refrain model; MoE kernels are unsupported.")
            moe_class = HunYuanMoE

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1_q = nn.Linear(
            hidden_size, hidden_size, **factory_kwargs
        )
        self.linear1_k = nn.Linear(
            hidden_size, hidden_size, **factory_kwargs
        )
        self.linear1_v = nn.Linear(
            hidden_size, hidden_size, **factory_kwargs
        )
        if moe_config is None:
            if is_dense_with_moe_mlp:
                self.mlp = DenseWithMoEMLP(
                    hidden_size,
                    dense_with_moe_mlp_intermediate_size,
                )
            else:
                # self.linear1_mlp = nn.Linear(
                # hidden_size, mlp_hidden_dim, **factory_kwargs
                # )
                # self.linear2 = LinearWarpforSingle(
                # hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs
                # )
                # self.mlp_act = get_activation_layer(mlp_act_type)()
                self.mlp = MLP(
                    hidden_size,
                    mlp_hidden_dim,
                    act_layer=get_activation_layer(mlp_act_type),
                    bias=True,
                    **factory_kwargs,
                )
        else:
            self.mlp = moe_class(moe_config)
            self.loss_moe = [0., 0.]

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def get_moe_loss(self):
        return self.loss_moe

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_cond: Tuple[torch.Tensor, torch.Tensor] = None,
        img_physical_freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        txt_physical_freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        is_token_replace=False,
        token_replace_vec=None,
        first_frame_mask=None
    ) -> torch.Tensor:
        if is_token_replace:
            mod, tr_mod = self.modulation(vec,
                                          is_token_replace=is_token_replace,
                                          token_replace_vec=token_replace_vec)
            (mod_shift,
             mod_scale,
             mod_gate) = mod.chunk(3, dim=-1)
            (tr_mod_shift,
             tr_mod_scale,
             tr_mod_gate) = tr_mod.chunk(3, dim=-1)

            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale, is_token_replace=is_token_replace,
                             tr_shift=tr_mod_shift, tr_scale=tr_mod_scale, first_frame_mask=first_frame_mask)
        else:
            mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        
        assert vec.dtype == torch.float32, f"vec.dtype: {vec.dtype}"
        assert mod_shift.dtype == torch.float32, f"mod_shift.dtype: {x_mod.dtype}"
        assert x_mod.dtype == torch.float32, f"x_mod.dtype: {x_mod.dtype}"
        
        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)
        
        # if self.moe_config is None and not self.is_dense_with_moe_mlp:
        #     mlp = self.linear1_mlp(x_mod)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # def shrink_head(encoder_state, dim):
            # local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            # return encoder_state.narrow(
                # dim, nccl_info.rank_within_group * local_heads, local_heads
            # )

        # freqs_cis = (shrink_head(freqs_cis[0], dim=0), shrink_head(freqs_cis[1], dim=0))

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk

        # Apply PhysRoPE (second rotation) on image tokens
        if img_physical_freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, img_physical_freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"phys img_qq: {img_qq.shape}, img_q: {img_q.shape}"
            img_q, img_k = img_qq, img_kk

        if freqs_cis_cond is not None:
            txt_qq, txt_kk = apply_rotary_emb(txt_q, txt_k, freqs_cis_cond, head_first=False)
            assert (
                txt_qq.shape == txt_q.shape and txt_kk.shape == txt_k.shape
            ), f"txt_kk: {txt_qq.shape}, txt_q: {txt_q.shape}, txt_kk: {txt_kk.shape}, txt_k: {txt_k.shape}"
            txt_q, txt_k = txt_qq, txt_kk

        # Apply PhysRoPE (second rotation) on text tokens
        if txt_physical_freqs_cis is not None:
            txt_qq, txt_kk = apply_rotary_emb(txt_q, txt_k, txt_physical_freqs_cis, head_first=False)
            assert (
                txt_qq.shape == txt_q.shape and txt_kk.shape == txt_k.shape
            ), f"phys txt_qq: {txt_qq.shape}, txt_q: {txt_q.shape}"
            txt_q, txt_k = txt_qq, txt_kk

        if is_flash:
            attn_mode = 'flash'
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )
        if self.moe_config is None:
            if self.is_dense_with_moe_mlp:
                output = self.mlp(attn)
            else:
                # output = self.linear2(attn, self.mlp_act(mlp))
                output = self.mlp(attn)
        else:
            used_token = F.pad(text_mask, (img_q.shape[1], 0), value=1)
            # loguru.logger.info(f"used_token: {used_token}")
            # output = self.mlp(x_mod, attn)
            output = self.mlp(attn, used_token=used_token)
            self.loss_moe[0] = self.mlp.get_moe_loss()[0]
            self.loss_moe[1] = self.mlp.get_moe_loss()[1]
        
        if is_token_replace:
            output = x + apply_gate(output, gate=mod_gate, is_token_replace=is_token_replace,
                                    tr_gate=tr_mod_gate, first_frame_mask=first_frame_mask)
            return output
        else:
            return x + apply_gate(output, gate=mod_gate)

class RefrainTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        win_size: List[int] = [3, 3, 3],
        tile_size: List[int] = [8, 4, 4],
        win_type: str = "fix",
        win_ratio: int = 9,
        ba_attn_chunk_size: int=512,
        ba_attn_topk: int=3,
        ba_attn_causal: bool = False,
        ba_attn_slice_method: str= "3D",
        ba_attn_full_attn_block_num: int=0,
        ba_attn_full_attn_every_n_blocks: int=0,
        ssta_topk: int=0,
        ssta_threshold: float=0.0,
        ssta_enforce_share_topk_block: int=0,
        sparse_attn_sparse_type: str = "none",
        sparse_attn_pad_type: str = "repeat",
        sparse_attn_use_text_mask: int = 1,
        sparse_attn_mask_share_within_head: int = 1,
        moe_config: MOEConfig = None,
        glyph_byT5_v2=False,
        vision_projection: str="none",
        vision_states_dim: int=1280,
        multitask_mask_training_type: Optional[str] = None,
        is_reshape_temporal_channels=False,
        use_cond_type_embedding=False,
        use_scale_rope=False,
        use_cond_3d_rope=False,
        physical_rope_dim: int = 0,
        is_dense_with_moe_mlp=False,
        dense_with_moe_mlp_intermediate_size=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # tmptmp
        # PP related
        from hymm.parallelism.parallel_states import get_parallel_state
        from torch import distributed as dist
        self.pp_enabled = get_parallel_state().pp_enabled
        if self.pp_enabled:
            self.pp_preprocess = dist.get_rank() == get_parallel_state().pp_mesh.mesh[0]
            self.pp_postprocess = dist.get_rank() == get_parallel_state().pp_mesh.mesh[-1]


        else:
            self.pp_preprocess = True
            self.pp_postprocess = True
        # self.pp_preprocess = True
        # self.pp_postprocess = True
        # self.pp_enabled = False


        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.moe_config = moe_config
        self.attn_mode = attn_mode
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.multitask_mask_training_type = multitask_mask_training_type
        self.vision_states_dim = vision_states_dim

        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.pp_preprocess:
            if self.glyph_byT5_v2:
                self.byt5_in = ByT5Mapper(
                    in_dim=1472,
                    out_dim=2048,
                    hidden_dim=2048,
                    out_dim1=hidden_size,
                    use_residual=False)

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        if self.pp_preprocess:
            self.img_in = PatchEmbed(
                self.patch_size, self.in_channels, self.hidden_size, self.multitask_mask_training_type,
                is_reshape_temporal_channels=is_reshape_temporal_channels, **factory_kwargs
            )

            # vision projection for i2v
            if vision_projection == "linear":
                self.vision_in = VisionProjection(in_dim=self.vision_states_dim, out_dim=self.hidden_size, flf_pos_emb=False)
            else:
                self.vision_in = None

            # text projection
            if self.text_projection == "linear":
                self.txt_in = TextProjection(
                    text_states_dim,
                    self.hidden_size,
                    get_activation_layer("silu"),
                    **factory_kwargs,
                )
            elif self.text_projection == "single_refiner":
                self.txt_in = SingleTokenRefiner(
                    text_states_dim,
                    hidden_size,
                    heads_num,
                    depth=2,
                    **factory_kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported text_projection: {self.text_projection}"
                )

            # time modulation
            self.time_in = TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            
            # text modulation
            self.vector_in = MLPEmbedder(
                self.config.text_states_dim_2, self.hidden_size, **factory_kwargs
            ) if self.text_pool_type is not None else None

            # guidance modulation
            self.guidance_in = (
                TimestepEmbedder(
                    self.hidden_size, get_activation_layer("silu"), **factory_kwargs
                )
                if guidance_embed
                else None
            )

        n_layers = mm_double_blocks_depth + mm_single_blocks_depth

        pp_splits = None
        # tmptmp
        if not self.pp_enabled:
            pp_splits = str(n_layers)
        elif pp_splits is None:
            assert n_layers % get_parallel_state().pp_mesh.size() == 0, f"n_layers ({n_layers}) must be divisible by pp_mesh.size() ({get_parallel_state().pp_mesh.size()})"
            n_each_pp_rank = n_layers // get_parallel_state().pp_mesh.size()
            pp_splits = ','.join([str(n_each_pp_rank) for _ in range(get_parallel_state().pp_mesh.size())])

        def is_pp_split(index, n_layers, pp_splits, bias=0):
            if not self.pp_enabled:
                return True
            splits = pp_splits.split(',')
            splits = [int(x) for x in splits]
            pp_rank = get_parallel_state().pp_mesh.get_local_rank()

            assert n_layers == sum(splits), f"n_layers ({n_layers}) and sum(pp_splits) ({sum(splits)}) mismatch"

            start = sum(splits[:pp_rank])
            end = start + splits[pp_rank]

            if index + bias < start or index + bias >= end:
                return False
            return True


        # tmptmp
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    moe_config=moe_config,
                    is_dense_with_moe_mlp=is_dense_with_moe_mlp,
                    dense_with_moe_mlp_intermediate_size=dense_with_moe_mlp_intermediate_size,
                    **factory_kwargs,
                )
                if is_pp_split(d_block_index, n_layers, pp_splits) else None
                for d_block_index in range(mm_double_blocks_depth)
            ]
        )
        # self.double_blocks = nn.ModuleList(
        #     [
        #         MMDoubleStreamBlock(
        #             self.hidden_size,
        #             self.heads_num,
        #             mlp_width_ratio=mlp_width_ratio,
        #             mlp_act_type=mlp_act_type,
        #             attn_mode=attn_mode,
        #             qk_norm=qk_norm,
        #             qk_norm_type=qk_norm_type,
        #             qkv_bias=qkv_bias,
        #             moe_config=moe_config,
        #             is_dense_with_moe_mlp=is_dense_with_moe_mlp,
        #             dense_with_moe_mlp_intermediate_size=dense_with_moe_mlp_intermediate_size,
        #             **factory_kwargs,
        #         )
        #         for d_block_index in range(mm_double_blocks_depth)
        #     ]
        # )
        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    moe_config=moe_config,
                    is_dense_with_moe_mlp=is_dense_with_moe_mlp,
                    dense_with_moe_mlp_intermediate_size=dense_with_moe_mlp_intermediate_size,
                    **factory_kwargs,
                )
                if is_pp_split(s_block_index, n_layers, pp_splits, bias=mm_double_blocks_depth) else None
                for s_block_index in range(mm_single_blocks_depth)
            ]
        )

        if self.pp_enabled:
            self.mix_stage = len(self.double_blocks) > 0 and len(self.single_blocks) > 0
        else:
            self.mix_stage = True

        self.only_double = len(self.double_blocks) > 0 and len(self.single_blocks) == 0
        self.only_single = len(self.double_blocks) == 0 and len(self.single_blocks) > 0

        if self.pp_postprocess:
            self.final_layer = FinalLayer(
                self.hidden_size,
                self.patch_size,
                self.out_channels,
                get_activation_layer("silu"),
                **factory_kwargs,
            )

        # STA
        self.attn_param = { # STA
                            'flex_mask_cache': {},
                            'win_size': [win_size],
                            'win_type': win_type,
                            'win_ratio': win_ratio,
                            'tile_size': tile_size,
                            # SSTA
                            'ssta_topk': ssta_topk,
                            'ssta_threshold': ssta_threshold,
                            'ssta_enforce_share_topk_block': ssta_enforce_share_topk_block,
                            #BlockAttn
                            'ba_attn_chunk_size': ba_attn_chunk_size,
                            'ba_attn_topk': ba_attn_topk,
                            'ba_attn_causal': ba_attn_causal,
                            'ba_attn_slice_method': ba_attn_slice_method,
                            'ba_attn_full_attn_block_num': ba_attn_full_attn_block_num,
                            'ba_attn_full_attn_every_n_blocks': ba_attn_full_attn_every_n_blocks,
                            # flex-sparse-attn:
                            "sparse_attn_sparse_type": sparse_attn_sparse_type,
                            "sparse_attn_pad_type": sparse_attn_pad_type,
                            "sparse_attn_use_text_mask": sparse_attn_use_text_mask,
                            "sparse_attn_mask_share_within_head": sparse_attn_mask_share_within_head,
        }
        if self.pp_preprocess:
            if use_cond_type_embedding:
                self.cond_type_embedding = nn.Embedding(3, self.hidden_size) 
                self.cond_type_embedding.weight.data.fill_(0)
                assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
                assert vision_projection is not None, "text type embedding is only used when vision_projection is not None"
                # 0: qwen feature
                # 1: byt5 feature
                # 2: siglip feature
            else:
                self.cond_type_embedding = None 
        self.use_cond_3d_rope = use_cond_3d_rope
        self.use_scale_rope = use_scale_rope
        self.physical_rope_dim = int(physical_rope_dim)
        self._physrope_config_printed = False
        self._physrope_forward_printed = False
        self._physrope_txt_printed = False
        self._physrope_l4_printed = False
        self._physrope_v2_route_debug_count = 0
        self._physrope_v2_double_block_printed = False
        self._physrope_v2_variable_route_printed = False
        self._reference_timeline_detailed_printed = False
        self._last_reference_timeline_variable_delta_max = 0.0
        self._last_reference_timeline_mapping_debug = []
        self._last_reference_timeline_boundary_debug = []


    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes, custom_temporal_positions=None, v2_dim=0):
        """Compute 3D RoPE. If custom_temporal_positions is given, replace
        the uniform temporal grid [0,1,...,tt-1] with custom values.

        Args:
            rope_sizes: (tt, th, tw)
            custom_temporal_positions: optional [tt] tensor with non-uniform time coords
            v2_dim: number of RoPE dims to replace with temporal RoPE.
                    0 = use rope_dim_list[0] only (default 16).
                    32/64 = replace more dims with temporal offset signal.
        """
        target_ndim = 3
        ndim = 5 - 2
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"

        if custom_temporal_positions is None:
            # Standard path: uniform temporal grid
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=self.rope_theta,
                use_real=True,
                theta_rescale_factor=1,
                use_scale_rope=self.use_scale_rope
            )
        else:
            # PhysRoPE v2 path: get standard RoPE first, then replace temporal dims
            # Step 1: get standard cos/sin (with correct spatial grid)
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=self.rope_theta,
                use_real=True,
                theta_rescale_factor=1,
                use_scale_rope=self.use_scale_rope
            )
            # Step 2: recompute temporal RoPE with custom positions
            # freqs_cos/sin shape: [tt*th*tw, head_dim]
            # temporal dims occupy [:rope_dim_list[0]] by default
            tt, th, tw = rope_sizes
            t_dim = rope_dim_list[0]  # 16 for HunyuanVideo
            effective_dim = max(v2_dim, t_dim) if v2_dim > 0 else t_dim
            effective_dim = min(effective_dim, head_dim)  # clamp
            # Expand custom_temporal_positions [tt] -> [tt*th*tw]
            ctp = custom_temporal_positions.float().cpu()  # get_1d works on cpu
            t_grid = ctp.unsqueeze(-1).unsqueeze(-1).expand(tt, th, tw).reshape(-1)
            from .posemb_layers import get_1d_rotary_pos_embed as _get_1d

            # IMPORTANT: When effective_dim > t_dim, we must NOT use a single
            # get_1d call with effective_dim, because that generates a DIFFERENT
            # frequency basis (1/theta^(2i/effective_dim)) than what the pretrained
            # model expects. Instead, we preserve the original frequency basis for
            # each axis and only replace the position coordinates with temporal ones.
            #
            # For dims [0 : t_dim]:        use temporal freqs (dim=t_dim) + custom positions
            # For dims [t_dim : eff_dim]:  use the ORIGINAL spatial freqs for those dims,
            #                              but apply custom temporal positions instead of
            #                              spatial positions.
            #
            # This way the frequency basis stays identical to pretrained, and only
            # the position input changes.

            # Part 1: native temporal dims [0 : t_dim]
            t_cos, t_sin = _get_1d(
                t_dim, t_grid, self.rope_theta,
                use_real=True, theta_rescale_factor=1,
            )
            replaced_cos = [t_cos]
            replaced_sin = [t_sin]

            if effective_dim > t_dim:
                # Part 2: extra dims borrowed from spatial axes
                # Walk through rope_dim_list to figure out which spatial axis
                # each extra dim belongs to, and use that axis's frequency basis.
                extra_start = t_dim
                extra_end = effective_dim
                dim_cursor = rope_dim_list[0]  # skip temporal
                for axis_idx in range(1, len(rope_dim_list)):
                    axis_dim = rope_dim_list[axis_idx]
                    axis_start = dim_cursor
                    axis_end = dim_cursor + axis_dim
                    # How many dims from this axis do we need to replace?
                    overlap_start = max(extra_start, axis_start)
                    overlap_end = min(extra_end, axis_end)
                    if overlap_start >= overlap_end:
                        dim_cursor = axis_end
                        continue
                    overlap_size = overlap_end - overlap_start
                    # offset within this axis's dim range
                    offset_in_axis = overlap_start - axis_start
                    # Compute freqs for this axis's full dim, then slice
                    # freqs_axis = 1/(theta^(arange(0, axis_dim, 2)[:axis_dim//2] / axis_dim))
                    freqs_axis = 1.0 / (
                        self.rope_theta ** (torch.arange(0, axis_dim, 2)[: (axis_dim // 2)].float() / axis_dim)
                    )
                    # Select only the frequency components we need
                    # overlap_size covers dims, each pair of dims uses one freq
                    # In use_real mode: cos/sin are repeat_interleaved, so
                    # overlap_size dims = overlap_size//2 freq components starting at offset_in_axis//2
                    freq_start = offset_in_axis // 2
                    freq_end = freq_start + overlap_size // 2
                    freqs_slice = freqs_axis[freq_start:freq_end]  # [overlap_size//2]
                    # outer product with temporal positions
                    angles = torch.outer(t_grid, freqs_slice)  # [S, overlap_size//2]
                    extra_cos = angles.cos().repeat_interleave(2, dim=1)  # [S, overlap_size]
                    extra_sin = angles.sin().repeat_interleave(2, dim=1)  # [S, overlap_size]
                    replaced_cos.append(extra_cos)
                    replaced_sin.append(extra_sin)
                    dim_cursor = axis_end

            new_cos = torch.cat(replaced_cos, dim=1)  # [S, effective_dim]
            new_sin = torch.cat(replaced_sin, dim=1)  # [S, effective_dim]
            # Replace first effective_dim dims with temporal RoPE
            freqs_cos = torch.cat([new_cos, freqs_cos[:, effective_dim:]], dim=1)
            freqs_sin = torch.cat([new_sin, freqs_sin[:, effective_dim:]], dim=1)

        return freqs_cos, freqs_sin
        # x: torch.Tensor,
        # t: torch.Tensor,  # Should be in range(0, 1000).
        # text_states: torch.Tensor = None,
        # text_mask: torch.Tensor = None,  # Now we don't use it.
        # text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        # guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        # return_dict: bool = True,

    def get_cond_rotary_pos_embed(self, vision_shape, cond_length):
        target_ndim = 3
        ndim = 5 - 2
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            vision_shape,
            cond_length,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
            mode="text",
            use_scale_rope=self.use_scale_rope
        )
        return freqs_cos, freqs_sin

    # ===================== PhysRoPE v2: Time-Coordinate Reparameterization =====================

    @staticmethod
    def _route_physrope_v2_freqs_to_blocks(
        custom_freqs_cos: torch.Tensor,
        custom_freqs_sin: torch.Tensor,
        standard_full_freqs_cis,
        standard_local_freqs_cis,
        custom_time_pos: torch.Tensor,
        temporal_dim: int,
        sp_size: int = 1,
        sp_rank: int = 0,
        verify_change: bool = False,
    ):
        """Route full-sequence PhysRoPE v2 frequencies to the active block shard.

        PhysRoPE v2 is constructed after the standard RoPE has already been
        sequence-parallel sharded. Therefore the custom tensors must be sharded
        again and must replace ``freqs_cis`` for every architecture, not only
        native-T concat.
        """

        if custom_freqs_cos.shape != custom_freqs_sin.shape:
            raise RuntimeError(
                "PhysRoPE v2 cos/sin shapes differ before block routing: "
                f"cos={tuple(custom_freqs_cos.shape)} "
                f"sin={tuple(custom_freqs_sin.shape)}"
            )
        if sp_size < 1 or sp_rank < 0 or sp_rank >= sp_size:
            raise RuntimeError(
                f"invalid PhysRoPE v2 SP route: sp_size={sp_size} sp_rank={sp_rank}"
            )

        full_shape = tuple(custom_freqs_cos.shape)
        if sp_size > 1:
            cos_chunks = torch.chunk(custom_freqs_cos, sp_size, dim=0)
            sin_chunks = torch.chunk(custom_freqs_sin, sp_size, dim=0)
            if len(cos_chunks) != sp_size or len(sin_chunks) != sp_size:
                raise RuntimeError(
                    "PhysRoPE v2 could not produce one frequency shard per SP rank: "
                    f"full_shape={full_shape} sp_size={sp_size} "
                    f"cos_chunks={len(cos_chunks)} sin_chunks={len(sin_chunks)}"
                )
            routed_cos = cos_chunks[sp_rank]
            routed_sin = sin_chunks[sp_rank]
        else:
            routed_cos = custom_freqs_cos
            routed_sin = custom_freqs_sin

        if standard_local_freqs_cis is not None:
            standard_local_cos, standard_local_sin = standard_local_freqs_cis
            if (
                routed_cos.shape != standard_local_cos.shape
                or routed_sin.shape != standard_local_sin.shape
            ):
                raise RuntimeError(
                    "PhysRoPE v2 routed frequency shape does not match the block's "
                    "standard RoPE shard: "
                    f"custom={tuple(routed_cos.shape)}/{tuple(routed_sin.shape)} "
                    f"standard={tuple(standard_local_cos.shape)}/"
                    f"{tuple(standard_local_sin.shape)} "
                    f"full={full_shape} sp_size={sp_size} sp_rank={sp_rank}"
                )

        metrics = {
            "full_shape": full_shape,
            "routed_shape": tuple(routed_cos.shape),
            "time_delta_max": 0.0,
            "freq_cos_delta_max": 0.0,
            "freq_sin_delta_max": 0.0,
            "expects_frequency_change": False,
        }
        if verify_change:
            uniform_time = torch.arange(
                custom_time_pos.numel(),
                dtype=custom_time_pos.dtype,
                device=custom_time_pos.device,
            )
            time_delta_max = float(
                (custom_time_pos - uniform_time).abs().max().detach().cpu().item()
            )
            expects_frequency_change = time_delta_max > 1e-6
            metrics["time_delta_max"] = time_delta_max
            metrics["expects_frequency_change"] = expects_frequency_change

            if standard_full_freqs_cis is None:
                raise RuntimeError(
                    "PhysRoPE v2 verification requires the full standard RoPE tensors"
                )
            standard_full_cos, standard_full_sin = standard_full_freqs_cis
            if (
                custom_freqs_cos.shape != standard_full_cos.shape
                or custom_freqs_sin.shape != standard_full_sin.shape
            ):
                raise RuntimeError(
                    "PhysRoPE v2 full frequency shape does not match standard RoPE: "
                    f"custom={tuple(custom_freqs_cos.shape)} "
                    f"standard={tuple(standard_full_cos.shape)}"
                )

            checked_dim = min(
                max(int(temporal_dim), 1),
                custom_freqs_cos.shape[-1],
            )
            cos_delta_max = float(
                (
                    custom_freqs_cos[:, :checked_dim]
                    - standard_full_cos[:, :checked_dim]
                )
                .abs()
                .max()
                .detach()
                .cpu()
                .item()
            )
            sin_delta_max = float(
                (
                    custom_freqs_sin[:, :checked_dim]
                    - standard_full_sin[:, :checked_dim]
                )
                .abs()
                .max()
                .detach()
                .cpu()
                .item()
            )
            metrics["freq_cos_delta_max"] = cos_delta_max
            metrics["freq_sin_delta_max"] = sin_delta_max
            if expects_frequency_change and max(cos_delta_max, sin_delta_max) <= 1e-7:
                raise RuntimeError(
                    "PhysRoPE v2 custom time coordinates differ from the uniform "
                    "timeline, but generated frequencies are identical to standard RoPE"
                )

        return (routed_cos, routed_sin), metrics

    def _build_reference_timeline_physical_time_grid(
        self,
        tt: int,
        th: int,
        tw: int,
        segment_layout,
        extra_kwargs: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the normal target-only PhysRoPE clock for sparse references.

        ReferenceTimeline changes only which unmodified reference latents are copied
        into the fixed target canvas. Source length, selected source indices,
        and sparse rounding must never warp the target PhysRoPE timeline.
        """

        alpha = float(extra_kwargs.get("physical_rope_alpha", 0.5))
        if str(extra_kwargs.get("physical_rope_offset_mode", "fixed")) != "fixed":
            raise ValueError("reference_timeline PhysRoPE requires offset_mode=fixed")
        if extra_kwargs.get("physical_rope_cut", False):
            raise ValueError("reference_timeline PhysRoPE does not support physical_rope_cut")
        if (
            extra_kwargs.get("physical_rope_speed_aware", False)
            and str(extra_kwargs.get("physical_rope_speed_mode", "uniform")) != "uniform"
        ):
            raise ValueError(
                "reference_timeline PhysRoPE does not support a second non-uniform speed warp"
            )

        tokens_per_frame = max(int(th * tw), 1)
        img_starts = segment_layout["img_start_rows"]
        img_ends = segment_layout["img_end_rows"]
        img_seg_ids = segment_layout.get("img_seg_ids")
        if img_starts.ndim == 2:
            img_starts = img_starts[0]
            img_ends = img_ends[0]
        frame_starts = img_starts[::tokens_per_frame][:tt].to(
            device=device, dtype=torch.float32
        )
        frame_ends = img_ends[::tokens_per_frame][:tt].to(
            device=device, dtype=torch.float32
        )
        if img_seg_ids is None:
            frame_seg_ids = torch.zeros(tt, dtype=torch.long, device=device)
        else:
            if img_seg_ids.ndim == 2:
                img_seg_ids = img_seg_ids[0]
            frame_seg_ids = img_seg_ids[::tokens_per_frame][:tt].to(device)

        visible_ranges = extra_kwargs.get("condition_frame_ranges", []) or []
        physical_ranges = (
            extra_kwargs.get("condition_physical_dst_frame_ranges", None)
            or visible_ranges
        )
        source_ranges = extra_kwargs.get("condition_src_frame_ranges", []) or []
        condition_kinds = extra_kwargs.get("condition_kinds", None) or [
            1
        ] * len(physical_ranges)
        raw_condition_seg_ids = (
            extra_kwargs.get("condition_seg_ids", None) or []
        )
        if len(raw_condition_seg_ids) == len(physical_ranges):
            condition_seg_ids = list(raw_condition_seg_ids)
        elif len(raw_condition_seg_ids) == sum(
            int(kind) == 1 for kind in condition_kinds
        ):
            condition_seg_ids = []
            video_id_cursor = 0
            for kind in condition_kinds:
                if int(kind) == 1:
                    condition_seg_ids.append(
                        int(raw_condition_seg_ids[video_id_cursor])
                    )
                    video_id_cursor += 1
                else:
                    condition_seg_ids.append(-1)
        elif not raw_condition_seg_ids:
            condition_seg_ids = [-1] * len(physical_ranges)
        else:
            raise ValueError(
                "condition_seg_ids must be parallel to all conditions or contain "
                "one id per video condition: "
                f"ranges={physical_ranges} kinds={condition_kinds} "
                f"seg_ids={raw_condition_seg_ids}"
            )
        if not (
            len(physical_ranges)
            == len(source_ranges)
            == len(condition_kinds)
            == len(condition_seg_ids)
        ):
            raise ValueError(
                "reference_timeline condition metadata must be parallel: "
                f"physical_dst={physical_ranges} src={source_ranges} "
                f"kinds={condition_kinds} seg_ids={condition_seg_ids}"
            )

        video_conditions = []
        if visible_ranges and len(visible_ranges) != len(physical_ranges):
            raise ValueError(
                "reference_timeline visible/physical condition ranges must be parallel: "
                f"visible_dst={visible_ranges} physical_dst={physical_ranges}"
            )
        effective_visible_ranges = (
            list(visible_ranges)
            if visible_ranges
            else list(physical_ranges)
        )
        for visible_dst, physical_dst, src, kind, prompt_seg_id in zip(
            effective_visible_ranges,
            physical_ranges,
            source_ranges,
            condition_kinds,
            condition_seg_ids,
        ):
            if int(kind) != 1:
                continue
            vds, vde = int(visible_dst[0]), int(visible_dst[1])
            pds, pde = int(physical_dst[0]), int(physical_dst[1])
            ss, se = int(src[0]), int(src[1])
            if (
                vds < pds
                or vde > pde
                or vde <= vds
                or pds < 0
                or pde <= pds
                or ss < 0
                or se <= ss
            ):
                raise ValueError(
                    "invalid reference_timeline video condition: "
                    f"visible_dst={(vds, vde)} physical_dst={(pds, pde)} "
                    f"src={(ss, se)}"
                )
            source_len = se - ss
            mapping = build_reference_timeline_mapping(
                source_len,
                (pds, pde),
                (vds, vde),
            )
            video_conditions.append(
                {
                    "visible_range": (vds, vde),
                    "target_range": (pds, pde),
                    "target_T": pde - pds,
                    "source_range": (ss, se),
                    "source_T": source_len,
                    "prompt_seg_id": int(prompt_seg_id),
                    "relation": mapping["relation"],
                    "source_indices": mapping["source_indices"],
                    "target_indices": mapping["target_indices"],
                    "visible_source_indices": mapping[
                        "visible_source_indices"
                    ],
                    "visible_target_indices": mapping[
                        "visible_target_indices"
                    ],
                }
            )
        video_conditions.sort(key=lambda item: item["target_range"][0])
        for left, right in zip(video_conditions, video_conditions[1:]):
            if left["target_range"][1] > right["target_range"][0]:
                raise ValueError(
                    "reference_timeline video-reference physical ranges must not overlap: "
                    f"{video_conditions}"
                )

        runs = []
        run_start = 0
        for frame_idx in range(1, tt + 1):
            if frame_idx == tt or int(frame_seg_ids[frame_idx].item()) != int(
                frame_seg_ids[frame_idx - 1].item()
            ):
                full_start = float(frame_starts[run_start].item())
                full_end = float(frame_ends[run_start].item())
                if full_end <= full_start:
                    full_start = float(run_start)
                    full_end = float(frame_idx)
                runs.append(
                    {
                        "seg_id": int(frame_seg_ids[run_start].item()),
                        "visible_start": run_start,
                        "visible_end": frame_idx,
                        "full_start": full_start,
                        "full_end": full_end,
                    }
                )
                run_start = frame_idx

        mapping_debug = []
        for condition in video_conditions:
            pds, pde = condition["target_range"]
            vds, vde = condition["visible_range"]
            visible_start = max(0, vds)
            visible_end = min(tt, vde)
            if visible_start < visible_end:
                visible_seg_ids = sorted(
                    {
                        int(value)
                        for value in frame_seg_ids[
                            visible_start:visible_end
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                    }
                )
                expected_prompt_seg_id = int(
                    condition.get("prompt_seg_id", -1)
                )
                if len(visible_seg_ids) != 1:
                    raise RuntimeError(
                        "a video reference must belong to exactly one original "
                        "prompt shot: "
                        f"condition={condition} visible={(visible_start, visible_end)} "
                        f"segment_ids={visible_seg_ids}"
                    )
                if (
                    expected_prompt_seg_id > 0
                    and visible_seg_ids[0] != expected_prompt_seg_id
                ):
                    raise RuntimeError(
                        "video reference prompt-shot ownership mismatch after "
                        "segment layout: "
                        f"expected={expected_prompt_seg_id} actual={visible_seg_ids[0]} "
                        f"condition={condition}"
                    )
            mapping_debug.append(
                {
                    **condition,
                    "visible": (visible_start, visible_end),
                    "clock_policy": "fixed_target_timeline_condition_independent",
                    "condition_changes_rope": False,
                }
            )

        # ReferenceTimeline uses exactly the same target timeline as a sample without
        # references. The only non-unit offset is the configured fixed shot gap.
        uniform = torch.arange(tt, dtype=torch.float32, device=device)
        standard_fixed_segment_offsets = torch.empty(
            tt, dtype=torch.float32, device=device
        )
        for segment_rank, run in enumerate(runs):
            standard_fixed_segment_offsets[
                run["visible_start"] : run["visible_end"]
            ] = float(segment_rank) * alpha

        segment_offsets = standard_fixed_segment_offsets.clone()
        custom_pos = uniform + segment_offsets
        boundary_debug = []
        for frame_idx in range(1, tt):
            if int(frame_seg_ids[frame_idx].item()) != int(
                frame_seg_ids[frame_idx - 1].item()
            ):
                raw_step = float(
                    (uniform[frame_idx] - uniform[frame_idx - 1]).item()
                )
                boundary_debug.append(
                    {
                        "frame": frame_idx,
                        "from_seg_id": int(frame_seg_ids[frame_idx - 1].item()),
                        "to_seg_id": int(frame_seg_ids[frame_idx].item()),
                        "raw_physical_step": raw_step,
                        "phase_shift_increment": float(alpha),
                        "previous_shot_phase": float(
                            segment_offsets[frame_idx - 1].item()
                        ),
                        "current_shot_phase": float(
                            segment_offsets[frame_idx].item()
                        ),
                        "final_token_step": float(raw_step + alpha),
                    }
                )
        self._last_reference_timeline_boundary_debug = boundary_debug

        expected_pos = uniform + standard_fixed_segment_offsets
        condition_clock_delta_max = float(
            (custom_pos - expected_pos).abs().max().detach().cpu().item()
            if tt > 0
            else 0.0
        )
        if condition_clock_delta_max > 1e-7:
            raise RuntimeError(
                "reference_timeline condition unexpectedly changed target PhysRoPE: "
                f"delta_max={condition_clock_delta_max} mappings={mapping_debug}"
            )
        if tt > 1 and not bool(torch.all(custom_pos[1:] > custom_pos[:-1])):
            raise RuntimeError(
                "fixed-target reference_timeline PhysRoPE must be strictly monotonic: "
                f"positions={custom_pos.detach().cpu().tolist()}"
            )
        for boundary in boundary_debug:
            expected_boundary_step = 1.0 + alpha
            if abs(boundary["final_token_step"] - expected_boundary_step) > 1e-6:
                raise RuntimeError(
                    "reference_timeline shot boundary must match normal PhysRoPE: "
                    f"boundary={boundary} expected={expected_boundary_step}"
                )

        self._last_reference_timeline_variable_delta_max = 0.0
        self._last_reference_timeline_mapping_debug = mapping_debug
        global_rank = int(os.environ.get("RANK", "0"))
        debug_limit = max(
            int(extra_kwargs.get("reference_timeline_debug_limit", 0)),
            0,
        )
        debug_count = int(
            getattr(self, "_reference_timeline_fixed_clock_debug_count", 0)
        )
        if global_rank % 8 == 0 and debug_count < debug_limit:
            self._reference_timeline_fixed_clock_debug_count = debug_count + 1
            global_step_deltas = (
                custom_pos[1:] - custom_pos[:-1]
                if tt > 1
                else torch.empty(0, device=device)
            )
            print(
                "\n[PhysRoPE REFERENCE-TIMELINE FIXED TARGET BEGIN]\n"
                f"debug={debug_count + 1}/{debug_limit} global_rank={global_rank} "
                f"target_T={tt} alpha={alpha}\n"
                f"policy=fixed_target_timeline_condition_independent\n"
                f"shot_runs={runs}\n"
                f"shot_boundaries={boundary_debug}\n"
                f"condition_mappings={mapping_debug}\n"
                f"target_positions={custom_pos.detach().cpu().tolist()}\n"
                f"target_step_deltas={global_step_deltas.detach().cpu().tolist()}\n"
                f"condition_clock_delta_max={condition_clock_delta_max:.9f}\n"
                "[PhysRoPE REFERENCE-TIMELINE FIXED TARGET END]\n",
                flush=True,
            )
        return custom_pos

    def _build_physical_time_grid(
        self,
        tt: int,
        th: int,
        tw: int,
        segment_layout,
        extra_kwargs: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """Build non-uniform temporal positions based on physical time.

        Instead of uniform [0, 1, ..., tt-1], produces positions with
        segment-aware offsets.

        offset_mode controls how the offset is computed:
          "physical": offset = segment_start_time_in_seconds * alpha
                      → gap between segments proportional to physical duration
          "fixed":    offset = segment_index * alpha  (like MSM's shot_phase_shift)
                      → every segment boundary gets the same fixed gap

        Args:
            tt, th, tw: latent grid sizes
            segment_layout: dict from _build_segment_layout (img_start_rows, img_end_rows)
            extra_kwargs: contains physical_rope_alpha, physical_rope_offset_mode, etc.
            device: target device

        Returns:
            custom_time_pos: [tt] non-uniform temporal coordinates
        """
        alpha = float(extra_kwargs.get("physical_rope_alpha", 1.0))
        use_rope_cut = extra_kwargs.get("physical_rope_cut", False)
        speed_aware = extra_kwargs.get("physical_rope_speed_aware", False)
        offset_mode = str(extra_kwargs.get("physical_rope_offset_mode", "physical"))
        native_t_concat = as_bool(extra_kwargs.get("range_condition_native_t_concat", False))
        temporal_pack_mode = normalize_temporal_pack_mode(
            extra_kwargs.get("range_condition_temporal_pack_mode", "legacy_trilinear")
        )
        if temporal_pack_mode == REFERENCE_TIMELINE:
            if native_t_concat:
                raise ValueError("reference_timeline PhysRoPE cannot use native temporal concat")
            return self._build_reference_timeline_physical_time_grid(
                tt, th, tw, segment_layout, extra_kwargs, device
            )
        native_positions = None
        native_prefix_t = 0
        if native_t_concat:
            native_prefix_t = int(extra_kwargs.get("native_reference_prefix_t", 0))
            target_t = tt - native_prefix_t
            native_positions = build_native_temporal_positions(
                native_prefix_t,
                target_t,
                extra_kwargs.get("native_reference_packed_ranges", []),
                extra_kwargs.get("condition_frame_ranges", []),
                device=device,
            )
            if use_rope_cut:
                raise ValueError(
                    "range_condition_native_t_concat does not support physical_rope_cut; "
                    "the active native-reference config must keep physical_rope_cut disabled"
                )
            if speed_aware and str(extra_kwargs.get("physical_rope_speed_mode", "uniform")) != "uniform":
                raise ValueError(
                    "range_condition_native_t_concat does not support non-uniform physical speed warping"
                )

        tokens_per_frame = max(int(th * tw), 1)

        # Extract per-frame (start_time, end_time) from segment_layout
        # img_start_rows: [num_img_tokens] or [B, num_img_tokens]
        img_starts = segment_layout["img_start_rows"]
        img_ends = segment_layout["img_end_rows"]
        if img_starts.ndim == 2:
            img_starts = img_starts[0]  # use first batch element for grid
            img_ends = img_ends[0]

        # Downsample from per-token to per-frame: take first token of each frame
        frame_starts = img_starts[::tokens_per_frame][:tt].to(device, dtype=torch.float32)
        frame_ends = img_ends[::tokens_per_frame][:tt].to(device, dtype=torch.float32)

        # Use integer segment IDs for robust boundary detection (no float fragility).
        # img_seg_ids: [num_img_tokens] or [B, num_img_tokens], dtype=long
        img_seg_ids = segment_layout.get("img_seg_ids")
        if img_seg_ids is not None:
            if img_seg_ids.ndim == 2:
                img_seg_ids = img_seg_ids[0]
            frame_seg_ids = img_seg_ids[::tokens_per_frame][:tt].to(device)
        else:
            # Fallback for legacy layouts without img_seg_ids
            frame_seg_ids = None

        # Build segment index per frame (for "fixed" offset mode)
        seg_index = torch.zeros(tt, dtype=torch.float32, device=device)
        if native_t_concat and frame_seg_ids is not None:
            target_seg_ids = frame_seg_ids[native_prefix_t:]
            segment_rank = {}
            for seg_id in target_seg_ids.detach().cpu().tolist():
                if int(seg_id) not in segment_rank:
                    segment_rank[int(seg_id)] = len(segment_rank)
            for t in range(tt):
                seg_index[t] = float(segment_rank.get(int(frame_seg_ids[t].item()), 0))
        else:
            cur_seg = 0
            for t in range(1, tt):
                if frame_seg_ids is not None:
                    is_boundary = (frame_seg_ids[t].item() != frame_seg_ids[t - 1].item())
                else:
                    # Fallback for legacy layouts without integer segment IDs.
                    is_boundary = abs(frame_starts[t].item() - frame_starts[t - 1].item()) > 1e-4
                if is_boundary:
                    cur_seg += 1
                seg_index[t] = float(cur_seg)

        custom_pos = torch.zeros(tt, dtype=torch.float32, device=device)

        for t in range(tt):
            # Compute offset based on mode
            if offset_mode == "fixed":
                # Like MSM: offset = segment_index * alpha (e.g., alpha=0.5)
                time_offset = seg_index[t].item() * alpha
            else:
                # "physical": offset = segment_start_time * alpha
                time_offset = frame_starts[t].item() * alpha

            if use_rope_cut:
                # RoPE Cut: each segment restarts local index from 0
                if frame_seg_ids is not None:
                    is_seg_start = (t == 0 or frame_seg_ids[t].item() != frame_seg_ids[t - 1].item())
                else:
                    is_seg_start = (t == 0 or abs(frame_starts[t].item() - frame_starts[t - 1].item()) > 1e-4)
                if is_seg_start:
                    _local_base = t  # remember where this segment starts
                local_idx = float(t - _local_base)
                custom_pos[t] = local_idx + time_offset
            else:
                # Continuous: base = frame_index + offset
                base_position = native_positions[t].item() if native_positions is not None else float(t)
                custom_pos[t] = base_position + time_offset

        # Optional: intra-segment speed warping (for future use)
        # Currently a no-op unless physical_rope_speed_aware is enabled
        # and speed info is present in extra_kwargs
        if speed_aware:
            speed_mode = str(extra_kwargs.get("physical_rope_speed_mode", "uniform"))
            if speed_mode != "uniform":
                custom_pos = self._apply_speed_warp(custom_pos, frame_starts, speed_mode,
                                                    frame_seg_ids=frame_seg_ids)

        # L4 variable-length condition mapping. condition_frame_ranges are target
        # dst slots; condition_src_frame_ranges preserve the source temporal span.
        # For src_len != dst_len, assign each target frame a linearly rescaled
        # source coordinate while keeping the existing segment offset.
        cond_dst_ranges = extra_kwargs.get("condition_frame_ranges", []) if extra_kwargs is not None else []
        cond_src_ranges = extra_kwargs.get("condition_src_frame_ranges", None) if extra_kwargs is not None else None
        l4_applied = []
        if cond_src_ranges and not native_t_concat:
            assert cond_dst_ranges and len(cond_dst_ranges) == len(cond_src_ranges), (
                f"condition_src_frame_ranges length must match condition_frame_ranges: "
                f"dst={cond_dst_ranges}, src={cond_src_ranges}"
            )
            for (ds, de), (ss, se) in zip(cond_dst_ranges, cond_src_ranges):
                ds, de = int(ds), int(de)
                ss, se = int(ss), int(se)
                assert 0 <= ds < de <= tt, (
                    f"L4 dst range {(ds, de)} must be inside temporal length T={tt}"
                )
                assert ss >= 0 and se > ss, f"L4 src range {(ss, se)} must be valid"
                dst_len = de - ds
                src_len = se - ss
                if src_len == dst_len and ss == ds:
                    continue
                scale = float(src_len) / float(dst_len)
                for t_idx in range(ds, de):
                    base_offset = custom_pos[t_idx] - float(t_idx)
                    mapped_src_pos = float(ss) + float(t_idx - ds) * scale
                    custom_pos[t_idx] = mapped_src_pos + base_offset
                l4_applied.append(((ds, de), (ss, se), scale))

        if (
            l4_applied
            and str(os.environ.get("RANK", "0")) == "0"
            and not self._physrope_l4_printed
        ):
            self._physrope_l4_printed = True
            print(
                f"[PhysRoPE L4] applied linear temporal mapping: {l4_applied[:8]}",
                flush=True,
            )

        return custom_pos

    def _apply_speed_warp(
        self,
        positions: torch.Tensor,
        frame_starts: torch.Tensor,
        speed_mode: str,
        frame_seg_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply intra-segment speed warping to temporal positions.

        Args:
            positions: [tt] current positions
            frame_starts: [tt] segment start times per frame
            speed_mode: 'accelerating', 'decelerating', 'slow', 'fast'
            frame_seg_ids: [tt] integer segment IDs per frame (robust boundary detection)

        Returns:
            warped positions [tt]
        """
        tt = positions.shape[0]
        result = positions.clone()

        # Find segments using integer seg_ids when available (robust),
        # falling back to float comparison (legacy)
        seg_boundaries = [0]
        for t in range(1, tt):
            if frame_seg_ids is not None:
                is_boundary = (frame_seg_ids[t].item() != frame_seg_ids[t - 1].item())
            else:
                is_boundary = abs(frame_starts[t].item() - frame_starts[t - 1].item()) > 1e-4
            if is_boundary:
                seg_boundaries.append(t)
        seg_boundaries.append(tt)

        for i in range(len(seg_boundaries) - 1):
            s, e = seg_boundaries[i], seg_boundaries[i + 1]
            n = e - s
            if n <= 1:
                continue
            t_norm = torch.linspace(0, 1, n, device=positions.device)
            if speed_mode == "accelerating":
                t_warped = t_norm ** 2
            elif speed_mode == "decelerating":
                t_warped = t_norm ** 0.5
            elif speed_mode == "slow":
                t_warped = t_norm * 0.5
            elif speed_mode == "fast":
                t_warped = t_norm * 2.0
            else:
                t_warped = t_norm
            # Map warped [0,1] back to local position range
            local_base = result[s].item()
            local_span = float(n - 1)
            for j in range(n):
                result[s + j] = local_base + t_warped[j].item() * local_span

        return result

    # ===================== PhysRoPE v1 Methods (Legacy) =====================

    def physrope_get_1d_rotary_pos_embed(
        self,
        dim: int,
        pos: torch.Tensor,
        theta: float = 256.0,
        theta_rescale_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 1D RoPE frequencies for physical time positions."""
        assert dim % 2 == 0, "dim must be even for RoPE"
        freqs = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
        freqs = 1.0 / (theta ** (freqs + theta_rescale_factor))
        angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)  # [N, dim//2]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [N, dim]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # [N, dim]
        return cos, sin

    def get_img_physical_time_rotary_pos_embed(
        self,
        window_start: torch.Tensor,
        window_end: torch.Tensor,
        dim_list: List[int],
        theta: float = 256.0,
        scale: float = 1.0,
        target_dim: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PhysRoPE for image tokens: dual-endpoint encoding (start, end).

        Args:
            window_start: [N] start times per token.
            window_end: [N] end times per token.
            dim_list: [dim_start, dim_end] dimensions for start/end encoding.
            theta: frequency theta.
            scale: time scale factor.
            target_dim: pad/truncate to this dim.

        Returns:
            (cos, sin) each [N, target_dim]
        """
        assert math.isfinite(scale) and scale >= 0.0, f"scale must be finite and >= 0, got {scale}"
        if not torch.is_tensor(window_start):
            window_start = torch.tensor(window_start, dtype=torch.float32)
        if not torch.is_tensor(window_end):
            window_end = torch.tensor(window_end, dtype=torch.float32)
        window_start = window_start * scale
        window_end = window_end * scale

        cos_start, sin_start = self.physrope_get_1d_rotary_pos_embed(
            dim_list[0], window_start, theta=theta, theta_rescale_factor=0.0,
        )
        cos_end, sin_end = self.physrope_get_1d_rotary_pos_embed(
            dim_list[1], window_end, theta=theta, theta_rescale_factor=0.0,
        )
        cos = torch.cat([cos_start, cos_end], dim=-1)
        sin = torch.cat([sin_start, sin_end], dim=-1)

        if target_dim is not None:
            cur_dim = cos.shape[-1]
            if cur_dim > target_dim:
                cos = cos[..., :target_dim]
                sin = sin[..., :target_dim]
            elif cur_dim < target_dim:
                pad_dim = target_dim - cur_dim
                pad_cos = torch.ones((*cos.shape[:-1], pad_dim), dtype=cos.dtype, device=cos.device)
                pad_sin = torch.zeros((*sin.shape[:-1], pad_dim), dtype=sin.dtype, device=sin.device)
                cos = torch.cat([cos, pad_cos], dim=-1)
                sin = torch.cat([sin, pad_sin], dim=-1)
        return cos, sin

    def get_txt_physical_time_rotary_pos_embed(
        self,
        window_start: torch.Tensor,
        window_end: torch.Tensor,
        dim_list: List[int],
        theta: float = 256.0,
        scale: float = 1.0,
        target_dim: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PhysRoPE for text tokens: batched [B, S] → [B, S, D].

        Args:
            window_start: [B, S] start times per text token per batch.
            window_end: [B, S] end times per text token per batch.
            dim_list: [dim_start, dim_end].
            theta, scale, target_dim: same as img version.

        Returns:
            (cos, sin) each [B, S, target_dim]
        """
        assert math.isfinite(scale) and scale >= 0.0
        if not torch.is_tensor(window_start):
            window_start = torch.tensor(window_start, dtype=torch.float32)
        if not torch.is_tensor(window_end):
            window_end = torch.tensor(window_end, dtype=torch.float32)
        if window_start.ndim == 1:
            window_start = window_start.unsqueeze(0)
        if window_end.ndim == 1:
            window_end = window_end.unsqueeze(0)
        window_start = window_start * scale
        window_end = window_end * scale

        batch_size = window_start.shape[0]
        cos_list, sin_list = [], []
        for b in range(batch_size):
            cos_s, sin_s = self.physrope_get_1d_rotary_pos_embed(
                dim_list[0], window_start[b], theta=theta, theta_rescale_factor=0.0,
            )
            cos_e, sin_e = self.physrope_get_1d_rotary_pos_embed(
                dim_list[1], window_end[b], theta=theta, theta_rescale_factor=0.0,
            )
            b_cos = torch.cat([cos_s, cos_e], dim=-1)
            b_sin = torch.cat([sin_s, sin_e], dim=-1)
            cos_list.append(b_cos.unsqueeze(0))
            sin_list.append(b_sin.unsqueeze(0))
        cos = torch.cat(cos_list, dim=0)  # [B, S, D]
        sin = torch.cat(sin_list, dim=0)

        if target_dim is not None:
            cur_dim = cos.shape[-1]
            if cur_dim > target_dim:
                cos = cos[..., :target_dim]
                sin = sin[..., :target_dim]
            elif cur_dim < target_dim:
                pad_dim = target_dim - cur_dim
                pad_cos = torch.ones((*cos.shape[:-1], pad_dim), dtype=cos.dtype, device=cos.device)
                pad_sin = torch.zeros((*sin.shape[:-1], pad_dim), dtype=sin.dtype, device=sin.device)
                cos = torch.cat([cos, pad_cos], dim=-1)
                sin = torch.cat([sin, pad_sin], dim=-1)
        return cos, sin

    def _build_img_physical_freqs_from_rows(
        self,
        img_start_rows: torch.Tensor,
        img_end_rows: torch.Tensor,
        dim_list: List[int],
        theta: float,
        scale: float,
        target_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build image PhysRoPE frequencies from per-token time rows."""
        if img_start_rows.ndim == 1:
            return self.get_img_physical_time_rotary_pos_embed(
                img_start_rows, img_end_rows, dim_list,
                theta=theta, scale=scale, target_dim=target_dim,
            )
        cos_rows, sin_rows = [], []
        for b in range(img_start_rows.shape[0]):
            b_cos, b_sin = self.get_img_physical_time_rotary_pos_embed(
                img_start_rows[b], img_end_rows[b], dim_list,
                theta=theta, scale=scale, target_dim=target_dim,
            )
            cos_rows.append(b_cos.unsqueeze(0))
            sin_rows.append(b_sin.unsqueeze(0))
        return torch.cat(cos_rows, dim=0), torch.cat(sin_rows, dim=0)

    def _build_segment_layout(
        self,
        text_segment_ids,
        text_segment_time_ranges,
        text_mask,
        txt,
        vision_states,
        extra_kwargs,
        tt: int, th: int, tw: int,
        time_range_mode: str,
        rope_fps: float,
        device: torch.device,
    ):
        """Build per-token time mapping from segment_ids and time_ranges.

        Returns dict with keys: img_start_rows, img_end_rows, txt_start_rows, txt_end_rows
        or None if no segment info available.
        """
        if text_segment_ids is None and text_segment_time_ranges is None:
            return None

        txt_batch = txt.shape[0]
        txt_len = txt.shape[1]
        vision_prefix_len = vision_states.shape[1] if vision_states is not None else 0
        text_only_len = max(txt_len - vision_prefix_len, 0)
        tokens_per_frame = max(int(th * tw), 1)
        use_second_time = str(time_range_mode).strip().lower() in ("second", "seconds")
        use_latent_time = str(time_range_mode).strip().lower() in ("latent",)
        native_t_concat = as_bool(
            extra_kwargs.get("range_condition_native_t_concat", False)
            if extra_kwargs is not None else False
        )
        temporal_pack_mode = normalize_temporal_pack_mode(
            extra_kwargs.get("range_condition_temporal_pack_mode", "legacy_trilinear")
            if extra_kwargs is not None else "legacy_trilinear"
        )
        native_positions = None
        if native_t_concat:
            native_prefix_t = int(extra_kwargs.get("native_reference_prefix_t", 0))
            native_positions = build_native_temporal_positions(
                native_prefix_t,
                tt - native_prefix_t,
                extra_kwargs.get("native_reference_packed_ranges", []),
                extra_kwargs.get("condition_frame_ranges", []),
                device=device,
            )

        def _align_tail(row, target_len, pad_val):
            out = list(row)
            if len(out) > target_len:
                out = out[-target_len:]
            elif len(out) < target_len:
                out = out + [pad_val] * (target_len - len(out))
            return out

        def _expand_rows(rows, target_rows):
            if len(rows) == target_rows:
                return rows
            if len(rows) == 1:
                return [list(rows[0]) for _ in range(target_rows)]
            raise RuntimeError(
                f"physical_rope rows mismatch: got rows={len(rows)} target={target_rows}"
            )

        def _parse_time_number(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                match = re.search(r"-?\d+(?:\.\d+)?", str(value))
                return float(match.group()) if match else None

        def _parse_time_range(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)) and len(value) == 2:
                start = _parse_time_number(value[0])
                end = _parse_time_number(value[1])
            elif isinstance(value, (int, float)):
                start = float(value)
                end = float(value)
            else:
                text_value = str(value).strip()
                if not text_value:
                    return None
                if "-" in text_value:
                    start_text, end_text = text_value.split("-", 1)
                    start = _parse_time_number(start_text)
                    end = _parse_time_number(end_text)
                else:
                    parsed = _parse_time_number(text_value)
                    start = parsed
                    end = parsed
            if start is None or end is None:
                return None
            if end < start:
                start, end = end, start
            return (float(start), float(end))

        # Normalize text_segment_ids to [B, txt_len]
        if text_segment_ids is None:
            seg_ids_rows = [[-1] * text_only_len]
        elif isinstance(text_segment_ids, torch.Tensor):
            if text_segment_ids.ndim == 1:
                seg_ids_rows = [text_segment_ids.detach().cpu().tolist()]
            else:
                seg_ids_rows = [text_segment_ids[b].detach().cpu().tolist() for b in range(text_segment_ids.shape[0])]
        elif isinstance(text_segment_ids, (list, tuple)):
            if len(text_segment_ids) > 0 and isinstance(text_segment_ids[0], (list, tuple, torch.Tensor)):
                seg_ids_rows = [
                    row.detach().cpu().tolist() if isinstance(row, torch.Tensor) else list(row)
                    for row in text_segment_ids
                ]
            else:
                seg_ids_rows = [list(text_segment_ids)]
        else:
            return None
        seg_ids_rows = _expand_rows(seg_ids_rows, txt_batch)
        seg_ids_rows = [_align_tail(row, text_only_len, -1) for row in seg_ids_rows]
        seg_ids_rows = [([-1] * vision_prefix_len) + row for row in seg_ids_rows]
        seg_ids_rows = [_align_tail(row, txt_len, -1) for row in seg_ids_rows]

        if isinstance(text_mask, torch.Tensor):
            if text_mask.ndim == 1:
                mask_rows = [text_mask.detach().cpu().tolist()]
            else:
                mask_rows = [text_mask[b].detach().cpu().tolist() for b in range(text_mask.shape[0])]
            mask_rows = _expand_rows(mask_rows, txt_batch)
            mask_rows = [_align_tail(row, txt_len, 0) for row in mask_rows]
            for batch_idx in range(txt_batch):
                for token_idx, mask_val in enumerate(mask_rows[batch_idx]):
                    if not bool(mask_val):
                        seg_ids_rows[batch_idx][token_idx] = -1

        # Normalize text_segment_time_ranges
        import json as _json
        if isinstance(text_segment_time_ranges, str):
            text_segment_time_ranges = _json.loads(text_segment_time_ranges)
        if isinstance(text_segment_time_ranges, (list, tuple)):
            if len(text_segment_time_ranges) > 0 and isinstance(text_segment_time_ranges[0], str):
                time_range_rows = [_json.loads(item) for item in text_segment_time_ranges]
            elif len(text_segment_time_ranges) > 0 and isinstance(text_segment_time_ranges[0], (list, tuple)):
                time_range_rows = text_segment_time_ranges
            elif len(text_segment_time_ranges) > 0 and isinstance(text_segment_time_ranges[0], dict):
                time_range_rows = [text_segment_time_ranges]
            else:
                time_range_rows = [[]]
        else:
            time_range_rows = [[]]
        time_range_rows = _expand_rows(time_range_rows, txt_batch)

        invalid_time = (0.0, 0.0)
        txt_start_rows, txt_end_rows = [], []
        img_start_rows, img_end_rows = [], []
        img_seg_id_rows = []

        for batch_idx in range(txt_batch):
            # Build segment_id → (start, end) time map
            segment_time_map = {}
            for item in time_range_rows[batch_idx]:
                if not isinstance(item, dict):
                    continue
                seg_id = item.get("segment_id")
                tr = item.get("time_range")
                if seg_id is None or tr is None:
                    continue
                seg_id = int(seg_id)
                parsed_time = _parse_time_range(tr)
                if parsed_time is not None:
                    segment_time_map[seg_id] = parsed_time

            # Build txt time rows
            row_start, row_end = [], []
            for seg_id in seg_ids_rows[batch_idx]:
                seg_id = int(seg_id)
                if seg_id < 0:
                    s, e = invalid_time
                else:
                    tr = segment_time_map.get(seg_id)
                    if tr is None:
                        tr = segment_time_map.get(0, invalid_time)
                    s, e = tr
                row_start.append(s)
                row_end.append(e)
            txt_start_rows.append(row_start)
            txt_end_rows.append(row_end)

            # Build img time rows: assign each frame to its segment based on time
            # Segments with time_range → latent frame assignment
            #
            # L4 plumbing (TIV2V variable-length rescale):
            #   condition_frame_ranges:     dst-range pairs in latent frames (where
            #                               cond tokens occupy the tensor)
            #   condition_src_frame_ranges: parallel list, source-range in latent frames
            # When src != dst (variable-length resampled cond), we *override* the
            # corresponding segment's PhysRoPE pair to (src_start, src_end) so the
            # encoded "physical span" represents the source's content duration.
            # The intra-segment 3D RoPE positions (token row indices) stay tied to
            # tensor coords, so cond and gen tokens still spatially align.
            cond_dst_ranges = extra_kwargs.get("condition_frame_ranges") if extra_kwargs is not None else None
            cond_physical_dst_ranges = (
                extra_kwargs.get("condition_physical_dst_frame_ranges")
                if extra_kwargs is not None
                else None
            )
            cond_src_ranges = extra_kwargs.get("condition_src_frame_ranges") if extra_kwargs is not None else None
            cond_kinds = (
                extra_kwargs.get("condition_kinds")
                if extra_kwargs is not None
                else None
            )
            raw_cond_seg_ids = (
                extra_kwargs.get("condition_seg_ids")
                if extra_kwargs is not None
                else None
            ) or []
            l4_dst2src = {}
            if (
                cond_src_ranges
                and not native_t_concat
                and temporal_pack_mode != REFERENCE_TIMELINE
            ):
                assert cond_dst_ranges and len(cond_dst_ranges) == len(cond_src_ranges), (
                    f"condition_src_frame_ranges length must match condition_frame_ranges: "
                    f"dst={cond_dst_ranges}, src={cond_src_ranges}"
                )
                for (ds, de), (ss, se) in zip(cond_dst_ranges, cond_src_ranges):
                    if int(ds) != int(ss) or int(de) != int(se):
                        l4_dst2src[(int(ds), int(de))] = (int(ss), int(se))

            segment_intervals = []
            for seg_id, (start_v, end_v) in segment_time_map.items():
                if seg_id > 0:  # skip global segment for frame assignment
                    # Convert to latent frame position
                    if use_latent_time:
                        # Latent mode: values are already latent indices, use directly
                        lat_start = max(0.0, float(start_v))
                        lat_end = max(0.0, float(end_v))
                    elif use_second_time:
                        lat_start = max(0.0, (start_v * rope_fps - 1.0) / 4.0 + 1.0) if start_v > 0 else 0.0
                        lat_end = max(0.0, (end_v * rope_fps - 1.0) / 4.0 + 1.0) if end_v > 0 else 0.0
                    else:
                        lat_start, lat_end = start_v, end_v
                    # L4 override: assignment stays on the target dst interval,
                    # while the encoded time-pair may use a different source span.
                    # This prevents a [dst_start, dst_end) condition from leaking
                    # frame ownership to source-length positions.
                    assign_lat_start, assign_lat_end = lat_start, lat_end
                    encoded_start_v, encoded_end_v = start_v, end_v
                    key = (int(round(lat_start)), int(round(lat_end)))
                    if key in l4_dst2src:
                        ss, se = l4_dst2src[key]
                        encoded_start_v, encoded_end_v = float(ss), float(se)
                    segment_intervals.append((
                        seg_id,
                        assign_lat_start,
                        assign_lat_end,
                        encoded_start_v,
                        encoded_end_v,
                    ))

            condition_owner_overrides = []
            owner_ranges = cond_physical_dst_ranges or cond_dst_ranges or []
            owner_kinds = cond_kinds or [1] * len(owner_ranges)
            if len(raw_cond_seg_ids) == len(owner_ranges):
                owner_seg_ids = list(raw_cond_seg_ids)
            elif len(raw_cond_seg_ids) == sum(
                int(kind) == 1 for kind in owner_kinds
            ):
                owner_seg_ids = []
                video_id_cursor = 0
                for kind in owner_kinds:
                    if int(kind) == 1:
                        owner_seg_ids.append(
                            int(raw_cond_seg_ids[video_id_cursor])
                        )
                        video_id_cursor += 1
                    else:
                        owner_seg_ids.append(-1)
            elif not raw_cond_seg_ids:
                owner_seg_ids = [-1] * len(owner_ranges)
            else:
                raise ValueError(
                    "condition_seg_ids must be parallel to all conditions or "
                    "contain one id per video condition: "
                    f"ranges={owner_ranges} kinds={owner_kinds} "
                    f"seg_ids={raw_cond_seg_ids}"
                )
            if not (
                len(owner_ranges) == len(owner_kinds) == len(owner_seg_ids)
            ):
                raise ValueError(
                    "condition prompt-shot metadata must be parallel: "
                    f"ranges={owner_ranges} kinds={owner_kinds} "
                    f"seg_ids={owner_seg_ids}"
                )
            interval_by_seg_id = {
                int(item[0]): item for item in segment_intervals
            }
            for dst_range, kind, owner_seg_id in zip(
                owner_ranges, owner_kinds, owner_seg_ids
            ):
                if int(kind) != 1 or int(owner_seg_id) <= 0:
                    continue
                ds, de = int(dst_range[0]), int(dst_range[1])
                owner_seg_id = int(owner_seg_id)
                if owner_seg_id not in interval_by_seg_id:
                    raise RuntimeError(
                        "video reference resolved to a prompt shot that is absent "
                        f"from segment layout: seg_id={owner_seg_id} "
                        f"dst={(ds, de)} available={sorted(interval_by_seg_id)}"
                    )
                condition_owner_overrides.append(
                    (ds, de, owner_seg_id)
                )

            if not segment_intervals:
                # No events with time ranges → all frames get global segment time
                global_tr = segment_time_map.get(0, invalid_time)
                img_s = [global_tr[0]] * (tt * tokens_per_frame)
                img_e = [global_tr[1]] * (tt * tokens_per_frame)
                img_seg = [0] * (tt * tokens_per_frame)
            else:
                img_s, img_e, img_seg = [], [], []
                for latent_idx in range(tt):
                    owner_matches = [
                        item
                        for item in condition_owner_overrides
                        if item[0] <= latent_idx < item[1]
                    ]
                    if len(owner_matches) > 1:
                        raise RuntimeError(
                            "overlapping video references claim different prompt "
                            f"shots at latent={latent_idx}: {owner_matches}"
                        )
                    if owner_matches:
                        owner_seg_id = owner_matches[0][2]
                        owner_interval = interval_by_seg_id[owner_seg_id]
                        assigned = (owner_interval[3], owner_interval[4])
                        assigned_seg_id = owner_seg_id
                        img_s.extend([assigned[0]] * tokens_per_frame)
                        img_e.extend([assigned[1]] * tokens_per_frame)
                        img_seg.extend([assigned_seg_id] * tokens_per_frame)
                        continue
                    # Find which segment this frame belongs to
                    assigned = None
                    assigned_seg_id = segment_intervals[0][0]
                    nearest_seg = segment_intervals[0]
                    nearest_dist = float("inf")
                    lat_pos = (
                        float(native_positions[latent_idx].item())
                        if native_positions is not None
                        else float(latent_idx)
                    )
                    for seg_id, ls, le, sv, ev in segment_intervals:
                        # Segment ranges are half-open latent intervals [start, end).
                        # Treating the end as inclusive shifts every shot boundary by
                        # one latent and breaks exact sparse source/destination alignment.
                        if ls <= lat_pos < le:
                            assigned = (sv, ev)
                            assigned_seg_id = seg_id
                            break
                        dist = min(abs(lat_pos - ls), abs(lat_pos - le))
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_seg = (seg_id, ls, le, sv, ev)
                            assigned_seg_id = seg_id
                    if assigned is None:
                        assigned = (nearest_seg[3], nearest_seg[4])
                        assigned_seg_id = nearest_seg[0]
                    img_s.extend([assigned[0]] * tokens_per_frame)
                    img_e.extend([assigned[1]] * tokens_per_frame)
                    img_seg.extend([assigned_seg_id] * tokens_per_frame)

            img_start_rows.append(img_s)
            img_end_rows.append(img_e)
            img_seg_id_rows.append(img_seg)

        return {
            "txt_start_rows": torch.tensor(txt_start_rows, dtype=torch.float32, device=device),
            "txt_end_rows": torch.tensor(txt_end_rows, dtype=torch.float32, device=device),
            "img_start_rows": torch.tensor(img_start_rows, dtype=torch.float32, device=device),
            "img_end_rows": torch.tensor(img_end_rows, dtype=torch.float32, device=device),
            "img_seg_ids": torch.tensor(img_seg_id_rows, dtype=torch.long, device=device),
            "has_text_segment_ids": text_segment_ids is not None,
        }


    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):

        if is_reorder:
            reorder_txt = []
            reorder_mask = []

            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat: #当使用block mask时，由于有近似计算，通过将pad设置为0,降低误差
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat([byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i],
                                           pad_byt5, pad_text], dim=0)
                else:
                    reorder_txt_i = torch.cat([byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i],
                                               byt5_txt_i[~byt5_text_mask_i], txt_i[~text_mask_i]], dim=0)
                reorder_mask_i = torch.cat([byt5_text_mask_i[byt5_text_mask_i], text_mask_i[text_mask_i],
                                            byt5_text_mask_i[~byt5_text_mask_i], text_mask_i[~text_mask_i]], dim=0)

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)

        return reorder_txt, reorder_mask

    def configure_repa_projector(
        self,
        repa_encoder_config,
        projector_hidden_dim: Optional[int] = None,
        projector_out_dim: Optional[int] = None,
        repa_type: str = None,
    ):
        """
        Initialize the REPA projector to map mid-layer DiT features to teacher feature space.
        """
        self.repa_type = repa_type
        projector_hidden = projector_hidden_dim or self.hidden_size
        projector_out = projector_out_dim
        if projector_out in (None, 0):
            projector_out = getattr(repa_encoder_config, "hidden_size", None)
            if projector_out is None and hasattr(repa_encoder_config, "vision_config"):
                projector_out = getattr(repa_encoder_config.vision_config, "hidden_size", None)
        if projector_out is None:
            raise ValueError("Cannot infer REPA projector input dim; please set --repa-projector-out-dim")
        factory_kwargs = {
            "device": next(self.parameters()).device,
            "dtype": next(self.parameters()).dtype,
        }
        self.repa_projector = nn.Sequential(
            nn.Linear(self.hidden_size, projector_hidden, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(projector_hidden, projector_hidden, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(projector_hidden, projector_out, **factory_kwargs),
        )

    def forward(
        self,
        ###################################################################################
        # LOOK AT ME!!!!
        # LOOK AT ME!!!!
        # LOOK AT ME!!!!
        # 
        # NOTE(kevinkhwu): DO NOT change the signature of the forward function (including order), it is used by the parallel engine.
        #       If you need to add new arguments, please let me know.
        #
        #
        ###################################################################################
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states: torch.Tensor=None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        ut=None,
        repa_feats=None,
        *, 
        pp_x=None,  # for single block
        pp_img=None, pp_txt=None,  # for double block
        pp_vec=None, pp_text_mask=None,
        pp_loss_moe_0=None, pp_loss_moe_1=None,
        pp_token_replace_vec=None,  # optional, put in the end
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if guidance == None:
            guidance = torch.tensor(
                [6016.0], device=hidden_states.device, dtype=torch.bfloat16
            )
        
        out = {}
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        # txt = encoder_hidden_states[:, 1:]
        # text_states_2 = encoder_hidden_states[:, 0, : self.config.text_states_dim_2]
        txt = text_states
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param['thw'] = [tt, th, tw]
        native_t_concat = as_bool(
            extra_kwargs.get("range_condition_native_t_concat", False)
            if extra_kwargs is not None else False
        )
        if native_t_concat:
            native_prefix_t = int(extra_kwargs.get("native_reference_prefix_t", 0))
            native_positions = build_native_temporal_positions(
                native_prefix_t,
                tt - native_prefix_t,
                extra_kwargs.get("native_reference_packed_ranges", []),
                extra_kwargs.get("condition_frame_ranges", []),
                device=x.device,
            )
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                (tt, th, tw), custom_temporal_positions=native_positions
            )
            if str(os.environ.get("RANK", "0")) == "0" and not getattr(self, "_native_t_concat_rope_printed", False):
                self._native_t_concat_rope_printed = True
                print(
                    "[RangeCondition NATIVE-T RoPE] "
                    f"reference_T={native_prefix_t} target_T={tt - native_prefix_t} "
                    f"packed_T={tt} positions={native_positions.detach().cpu().tolist()[:16]}",
                    flush=True,
                )
        elif freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
            freqs_cos = freqs_cos.to(x.device)
            freqs_sin = freqs_sin.to(x.device)
        if native_t_concat and freqs_cos.device != x.device:
            freqs_cos = freqs_cos.to(x.device)
            freqs_sin = freqs_sin.to(x.device)
            
        is_token_replace = extra_kwargs.get("is_token_replace", False)

        if is_token_replace:
            condition_frame_ranges = extra_kwargs.get("condition_frame_ranges")
            if condition_frame_ranges:
                # Unified PhysRoPE: mark arbitrary condition frame ranges
                first_frame_mask = torch.zeros((bs, ot, oh // self.patch_size[1], ow // self.patch_size[2]))
                for cond_start, cond_end in condition_frame_ranges:
                    s = max(0, int(cond_start))
                    e = min(ot, int(cond_end))
                    if s < e:
                        first_frame_mask[:, s:e, :, :] = 1.0
                first_frame_mask = first_frame_mask.flatten(1)
            else:
                # Legacy i2v: only first frame
                first_frame_mask = torch.zeros((bs, ot, oh // self.patch_size[1], ow // self.patch_size[2]))
                first_frame_mask[:, 0:1, :, :] = 1.0
                first_frame_mask = first_frame_mask.flatten(1)
        else:
            first_frame_mask = None

        ##############################################################
        if self.pp_enabled:
            assert not nccl_info.use_dynamic_ring_attention


        # Keep both forms: the full standard tensors are used to verify that a
        # non-uniform physical clock really changes RoPE, while ``freqs_cis`` is
        # the already-sharded standard tuple expected by the double blocks.
        standard_full_freqs_cis = (freqs_cos, freqs_sin)
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]
            if is_token_replace:
                first_frame_mask_sp = torch.chunk(first_frame_mask, sp_size, dim=1)[sp_rank]
                first_frame_mask = torch.sum(first_frame_mask_sp, dim=1)
        else:
            if is_token_replace:
                first_frame_mask = torch.sum(first_frame_mask, dim=1)


        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        ##############################################################
        
        if self.pp_preprocess:
            _, _, t_, h_, w_ = img.shape
            image_shape=(t_, h_//2, w_//2) # patchfiy 2x
            img = self.img_in(img)
            if get_sequence_parallel_state():
                sp_size = nccl_info.sp_size
                sp_rank = nccl_info.rank_within_spgroup
                if nccl_info.use_dynamic_ring_attention:
                    img, freqs_cos, freqs_sin = preprocess_dynamic_ring_attention(img, text_mask, freqs_cos, freqs_sin)
                else:
                    # assert img.shape[1] % sp_size == 0, f"Cannot split video sequence into ulysses_degree  ({sp_size}) parts evenly"
                    if img.shape[1] % sp_size != 0:
                        n_token = img.shape[1]
                        assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f'Token is too short for SP {sp_size}'

                    img = torch.chunk(img, sp_size, dim=1)[sp_rank]


            # Prepare modulation vectors.
            vec = self.time_in(t)

            if is_token_replace:
                token_replace_t = torch.zeros_like(t)
                token_replace_vec = self.time_in(token_replace_t)
            else:
                token_replace_vec = None

            # text modulation
            if text_states_2 is not None:
                vec_2 = self.vector_in(text_states_2)
                vec = vec + vec_2
                if is_token_replace:
                    token_replace_vec = token_replace_vec + vec_2
    
            # guidance modulation
            if self.guidance_embed:
                if guidance is None:
                    raise ValueError(
                        "Didn't get guidance strength for guidance distilled model."
                    )

                # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
                vec = vec + self.guidance_in(guidance)

            # Embed image and text.
                
            if self.text_projection == "linear":
                txt = self.txt_in(txt)
            elif self.text_projection == "single_refiner":
                txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
            else:
                raise NotImplementedError(
                    f"Unsupported text_projection: {self.text_projection}"
                )
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:,:,0], device=text_mask.device, dtype=torch.long))
                txt = txt + cond_emb

            if self.glyph_byT5_v2:
                byt5_text_states = extra_kwargs["byt5_text_states"]
                byt5_text_mask = extra_kwargs["byt5_text_mask"]
                byt5_txt = self.byt5_in(byt5_text_states)
                
                if self.cond_type_embedding is not None:
                    cond_emb = self.cond_type_embedding(torch.ones_like(byt5_txt[:,:,0], device=byt5_txt.device, dtype=torch.long))
                    byt5_txt = byt5_txt + cond_emb
                
                txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)
                
            if self.vision_in is not None and vision_states is not None:
                extra_encoder_hidden_states = self.vision_in(vision_states)

                if mask_type == "t2v" and torch.all(vision_states == 0): # if t2v, set extra_attention_mask to 0 to avoid attention to semantic tokens
                    extra_attention_mask = torch.zeros((bs, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device)
                    extra_encoder_hidden_states = extra_encoder_hidden_states * 0. # 避免ssta block mask近似误差
                else:
                    extra_attention_mask = torch.ones((bs, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device)

                #txt = torch.cat([extra_encoder_hidden_states, txt], dim=1)
                #text_mask = torch.cat([extra_attention_mask, text_mask], dim=1)
                # 保证前面是有效token, 后面是padding token
                if self.cond_type_embedding is not None:
                    cond_emb =  self.cond_type_embedding(2*torch.ones_like(extra_encoder_hidden_states[:,:,0], 
                                                                                        dtype=torch.long,
                                                                                        device=extra_encoder_hidden_states.device))
                    extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb
                                                

                txt, text_mask = self.reorder_txt_token(extra_encoder_hidden_states, txt,
                                                        extra_attention_mask, text_mask)
            self.loss_moe = [0, 0]
        else:
            img, txt, vec, text_mask = pp_img, pp_txt, pp_vec, pp_text_mask.to(torch.int64)
            pp_text_mask.grad = torch.zeros_like(pp_text_mask)

            if is_token_replace:
                token_replace_vec = pp_token_replace_vec
            else:
                token_replace_vec = None
            self.loss_moe = [pp_loss_moe_0, pp_loss_moe_1]


        # self.count_top = [[[], []], [[], []]] # not implemented yet

        if self.use_cond_3d_rope:
            freqs_cos_cond, freqs_sin_cond = self.get_cond_rotary_pos_embed(vision_shape=(tt, th, tw), 
                                                                        cond_length=txt.shape[1])
            freqs_cos_cond = freqs_cos_cond.to(x.device)
            freqs_sin_cond = freqs_sin_cond.to(x.device)
            freqs_cis_cond = (freqs_cos_cond, freqs_sin_cond)
        else:
            freqs_cis_cond = None

        # ===================== PhysRoPE v2: Replace standard RoPE grid =====================
        # v2 is activated when physical_rope_alpha is set (any value including 0).
        # v1 (legacy second-layer rotation) is activated when physical_rope_dim > 0
        # and physical_rope_alpha is absent.
        _physrope_v2_applied = False
        if extra_kwargs is not None and extra_kwargs.get("use_physical_rope", False):
            _has_alpha = "physical_rope_alpha" in extra_kwargs
            if _has_alpha:
                text_segment_time_ranges = extra_kwargs.get("text_segment_time_ranges")
                time_range_mode = str(extra_kwargs.get("physical_rope_time_range_mode", "second"))
                rope_fps = float(extra_kwargs.get("video_fps", 24.0))
                if rope_fps <= 0:
                    rope_fps = 24.0

                segment_layout = self._build_segment_layout(
                    text_segment_ids=extra_kwargs.get("text_segment_ids"),
                    text_segment_time_ranges=text_segment_time_ranges,
                    text_mask=text_mask,
                    txt=txt,
                    vision_states=vision_states,
                    extra_kwargs=extra_kwargs,
                    tt=tt, th=th, tw=tw,
                    time_range_mode=time_range_mode,
                    rope_fps=rope_fps,
                    device=txt.device if txt is not None else x.device,
                )

                if segment_layout is not None:
                    custom_time_pos = self._build_physical_time_grid(
                        tt, th, tw, segment_layout, extra_kwargs,
                        device=txt.device if txt is not None else x.device,
                    )
                    _v2_dim = int(extra_kwargs.get("physical_rope_v2_dim", 0))
                    freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                        (tt, th, tw), custom_temporal_positions=custom_time_pos,
                        v2_dim=_v2_dim,
                    )
                    freqs_cos = freqs_cos.to(x.device)
                    freqs_sin = freqs_sin.to(x.device)
                    _physrope_v2_applied = True

                    rank = os.environ.get("RANK", "0")
                    if str(rank) == "0" and not self._physrope_config_printed:
                        self._physrope_config_printed = True
                        alpha = float(extra_kwargs.get("physical_rope_alpha", 1.0))
                        use_cut = extra_kwargs.get("physical_rope_cut", False)
                        offset_mode = str(extra_kwargs.get("physical_rope_offset_mode", "physical"))
                        print("\n" + "=" * 80)
                        print(f"[PhysRoPE v2] alpha={alpha} | cut={use_cut} | "
                              f"offset_mode={offset_mode} | v2_dim={_v2_dim} | "
                              f"time_range_mode={time_range_mode} | fps={rope_fps} | "
                              f"img_shape=({tt},{th},{tw})")
                        print(f"[PhysRoPE v2] time_pos range=[{custom_time_pos.min():.2f}, "
                              f"{custom_time_pos.max():.2f}] | "
                              f"std={custom_time_pos.std():.3f}")
                        # Print first few and segment boundary positions
                        _tp = custom_time_pos.cpu().tolist()
                        _n = min(len(_tp), 8)
                        print(f"[PhysRoPE v2] first {_n} pos: {[f'{v:.2f}' for v in _tp[:_n]]}")
                        # Find segment boundaries and print around them
                        _fs = segment_layout["img_start_rows"]
                        if _fs.ndim == 2:
                            _fs = _fs[0]
                        _fs_frame = _fs[::max(int(th * tw), 1)][:tt].cpu().tolist()
                        _boundaries = [i for i in range(1, len(_fs_frame)) if abs(_fs_frame[i] - _fs_frame[i-1]) > 1e-4]
                        if _boundaries:
                            for bi in _boundaries[:3]:
                                lo = max(bi - 2, 0)
                                hi = min(bi + 3, len(_tp))
                                print(f"[PhysRoPE v2] boundary@frame{bi}: pos[{lo}:{hi}]="
                                      f"{[f'{v:.2f}' for v in _tp[lo:hi]]} | "
                                      f"seg_start={[f'{v:.2f}' for v in _fs_frame[lo:hi]]}")
                        print("=" * 80 + "\n")

        physrope_v2_block_freqs_cis = None
        if _physrope_v2_applied:
            if get_sequence_parallel_state():
                route_sp_size = nccl_info.sp_size
                route_sp_rank = nccl_info.rank_within_spgroup
            else:
                route_sp_size = 1
                route_sp_rank = 0

            debug_limit = max(
                int(extra_kwargs.get("physrope_route_debug_limit", 2)),
                0,
            )
            sparse_variable_delta_max = float(
                getattr(self, "_last_reference_timeline_variable_delta_max", 0.0)
            )
            force_variable_route_debug = (
                sparse_variable_delta_max > 1e-6
                and not self._physrope_v2_variable_route_printed
            )
            # Disabled noisy all-rank PhysRoPE route verification/debug output.
            # This only controls diagnostic checks/prints; routed frequencies are
            # still built and installed into every transformer block below.
            verify_route = False
            temporal_dim = (
                int(self.rope_dim_list[0])
                if self.rope_dim_list is not None
                else int(freqs_cos.shape[-1] // 3)
            )
            custom_full_freqs_cis = (freqs_cos, freqs_sin)
            physrope_v2_block_freqs_cis, route_metrics = (
                self._route_physrope_v2_freqs_to_blocks(
                    custom_freqs_cos=freqs_cos,
                    custom_freqs_sin=freqs_sin,
                    standard_full_freqs_cis=standard_full_freqs_cis,
                    standard_local_freqs_cis=freqs_cis,
                    custom_time_pos=custom_time_pos,
                    temporal_dim=temporal_dim,
                    sp_size=route_sp_size,
                    sp_rank=route_sp_rank,
                    verify_change=verify_route,
                )
            )
            freqs_cos, freqs_sin = physrope_v2_block_freqs_cis
            # This assignment is the actual PhysRoPE v2 injection consumed by
            # every double block below. It must not depend on native-T concat.
            freqs_cis = physrope_v2_block_freqs_cis

            if freqs_cis is not physrope_v2_block_freqs_cis:
                raise RuntimeError(
                    "PhysRoPE v2 block frequency tuple was not installed into freqs_cis"
                )
            if verify_route:
                global_rank = int(os.environ.get("RANK", "0"))
                print(
                    "[PhysRoPE v2 ROUTE] "
                    f"global_rank={global_rank} "
                    f"sp_rank={route_sp_rank}/{route_sp_size} "
                    f"native_t_concat={native_t_concat} "
                    f"double_blocks={len(self.double_blocks)} "
                    f"single_blocks={len(self.single_blocks)} "
                    f"full_shape={route_metrics['full_shape']} "
                    f"routed_shape={route_metrics['routed_shape']} "
                    f"time_delta_max={route_metrics['time_delta_max']:.6f} "
                    f"expects_change={route_metrics['expects_frequency_change']} "
                    f"cos_delta_max={route_metrics['freq_cos_delta_max']:.6e} "
                    f"sin_delta_max={route_metrics['freq_sin_delta_max']:.6e} "
                    f"sparse_variable_delta_max={sparse_variable_delta_max:.6f} "
                    f"sparse_variable_mapping={getattr(self, '_last_reference_timeline_mapping_debug', [])} "
                    f"shot_phase_shifts={getattr(self, '_last_reference_timeline_boundary_debug', [])} "
                    f"standard_cos_ptr={freqs_cis[0].data_ptr() if standard_full_freqs_cis is None else standard_full_freqs_cis[0].data_ptr()} "
                    f"routed_cos_ptr={freqs_cis[0].data_ptr()} "
                    "installed_into_freqs_cis=True",
                    flush=True,
                )
                self._physrope_v2_route_debug_count += 1
                if force_variable_route_debug and global_rank % 8 == 0:
                    mapping_debug = getattr(
                        self, "_last_reference_timeline_mapping_debug", []
                    )
                    debug_frames = {0, max(tt - 1, 0)}
                    for mapping in mapping_debug:
                        pds, pde = mapping["target_range"]
                        debug_frames.add(max(0, min(tt - 1, int(pds))))
                        debug_frames.add(max(0, min(tt - 1, int(pde) - 1)))
                        debug_frames.update(
                            int(value)
                            for value in mapping["target_indices"]
                            if 0 <= int(value) < tt
                        )
                    phase_rows = []
                    spatial_tokens_per_frame = max(int(th * tw), 1)
                    checked_dim = min(8, temporal_dim)
                    for frame_idx in sorted(debug_frames):
                        token_row = int(frame_idx) * spatial_tokens_per_frame
                        phase_rows.append(
                            {
                                "frame": int(frame_idx),
                                "standard_time": float(frame_idx),
                                "physical_time": float(
                                    custom_time_pos[int(frame_idx)].item()
                                ),
                                "time_delta": float(
                                    custom_time_pos[int(frame_idx)].item()
                                    - float(frame_idx)
                                ),
                                "standard_cos_first8": standard_full_freqs_cis[
                                    0
                                ][token_row, :checked_dim]
                                .detach()
                                .cpu()
                                .tolist(),
                                "custom_cos_first8": custom_full_freqs_cis[0][
                                    token_row, :checked_dim
                                ]
                                .detach()
                                .cpu()
                                .tolist(),
                                "standard_sin_first8": standard_full_freqs_cis[
                                    1
                                ][token_row, :checked_dim]
                                .detach()
                                .cpu()
                                .tolist(),
                                "custom_sin_first8": custom_full_freqs_cis[1][
                                    token_row, :checked_dim
                                ]
                                .detach()
                                .cpu()
                                .tolist(),
                            }
                        )
                    print(
                        "\n[PhysRoPE v2 PHASE DETAIL BEGIN]\n"
                        f"global_rank={global_rank} sp_rank={route_sp_rank}/"
                        f"{route_sp_size} tt={tt} th={th} tw={tw} "
                        f"temporal_dim={temporal_dim}\n"
                        f"custom_time_positions="
                        f"{custom_time_pos.detach().cpu().tolist()}\n"
                        f"sparse_variable_mappings={mapping_debug}\n"
                        "shot_phase_shifts="
                        f"{getattr(self, '_last_reference_timeline_boundary_debug', [])}\n"
                        f"phase_rows={phase_rows}\n"
                        f"full_cos_delta_max="
                        f"{route_metrics['freq_cos_delta_max']:.9e} "
                        f"full_sin_delta_max="
                        f"{route_metrics['freq_sin_delta_max']:.9e}\n"
                        "[PhysRoPE v2 PHASE DETAIL END]\n",
                        flush=True,
                    )
                if force_variable_route_debug:
                    self._physrope_v2_variable_route_printed = True

        # ===================== PhysRoPE v1 Construction (legacy) =====================
        img_physical_freqs_cis = None
        txt_physical_freqs_cis = None

        if not _physrope_v2_applied and extra_kwargs is not None and self.physical_rope_dim > 0:
            text_segment_ids = extra_kwargs.get("text_segment_ids")
            text_segment_time_ranges = extra_kwargs.get("text_segment_time_ranges")
            use_physical_rope = extra_kwargs.get("use_physical_rope", False)
            disable_text_physical_rope = extra_kwargs.get("disable_text_physical_rope", False)

            if use_physical_rope and text_segment_time_ranges is not None:
                total_dim = sum(self.rope_dim_list)
                physical_rope_theta = float(extra_kwargs.get("physical_rope_theta", 256.0))
                physical_rope_scale = float(extra_kwargs.get("physical_rope_scale", 0.1))
                physical_rope_dim = int(extra_kwargs.get("physical_rope_dim", self.physical_rope_dim))
                time_range_mode = str(extra_kwargs.get("physical_rope_time_range_mode", "second"))
                rope_fps = float(extra_kwargs.get("video_fps", 24.0))
                if rope_fps <= 0:
                    rope_fps = 24.0

                phys_dim = min(max(physical_rope_dim, 2), total_dim)
                if phys_dim % 2 != 0:
                    phys_dim = phys_dim - 1
                img_physical_dim_list = [phys_dim // 2, phys_dim - phys_dim // 2]

                rank = os.environ.get("RANK", "0")
                if str(rank) == "0" and not self._physrope_config_printed:
                    self._physrope_config_printed = True
                    print("\n" + "=" * 80)
                    print(f"[PhysRoPE CONFIG] (Refrain.py:L1770) theta={physical_rope_theta} | dim={physical_rope_dim} | "
                          f"dim_list={img_physical_dim_list} | scale={physical_rope_scale}")
                    print(f"[PhysRoPE CONFIG] (Refrain.py:L1774) total_dim={total_dim} | mode={time_range_mode} | "
                          f"disable_text={disable_text_physical_rope} | fps={rope_fps}")
                    print(f"[PhysRoPE CONFIG] (Refrain.py:L1776) has_segment_ids={text_segment_ids is not None} | "
                          f"n_time_ranges={len(text_segment_time_ranges) if isinstance(text_segment_time_ranges, list) else 'batch'} | "
                          f"img_shape=({tt},{th},{tw}) | txt_len={txt.shape[1] if txt is not None else None}")
                    print("=" * 80 + "\n")

                segment_layout = self._build_segment_layout(
                    text_segment_ids=text_segment_ids,
                    text_segment_time_ranges=text_segment_time_ranges,
                    text_mask=text_mask,
                    txt=txt,
                    vision_states=vision_states,
                    extra_kwargs=extra_kwargs,
                    tt=tt, th=th, tw=tw,
                    time_range_mode=time_range_mode,
                    rope_fps=rope_fps,
                    device=txt.device,
                )

                if segment_layout is not None:
                    physical_freqs_cos, physical_freqs_sin = self._build_img_physical_freqs_from_rows(
                        img_start_rows=segment_layout["img_start_rows"],
                        img_end_rows=segment_layout["img_end_rows"],
                        dim_list=img_physical_dim_list,
                        theta=physical_rope_theta,
                        scale=physical_rope_scale,
                        target_dim=total_dim,
                    )
                    img_physical_freqs_cis = (
                        physical_freqs_cos.to(x.device),
                        physical_freqs_sin.to(x.device),
                    )

                    if str(os.environ.get("RANK", "0")) == "0" and not self._physrope_forward_printed:
                        self._physrope_forward_printed = True
                        print(f"[PhysRoPE FORWARD] (Refrain.py:L1792) segment_layout BUILT: "
                              f"img_start_rows.shape={segment_layout['img_start_rows'].shape} | "
                              f"img_end_rows.shape={segment_layout['img_end_rows'].shape} | "
                              f"img_physical_freqs_cos.shape={physical_freqs_cos.shape} | "
                              f"has_text_segment_ids={segment_layout.get('has_text_segment_ids', False)}")

                    if (
                        not disable_text_physical_rope
                        and segment_layout.get("has_text_segment_ids", text_segment_ids is not None)
                    ):
                        txt_phys_cos, txt_phys_sin = self.get_txt_physical_time_rotary_pos_embed(
                            window_start=segment_layout["txt_start_rows"],
                            window_end=segment_layout["txt_end_rows"],
                            dim_list=img_physical_dim_list,
                            theta=physical_rope_theta,
                            scale=physical_rope_scale,
                            target_dim=total_dim,
                        )
                        txt_physical_freqs_cis = (
                            txt_phys_cos.to(x.device),
                            txt_phys_sin.to(x.device),
                        )

                        if str(os.environ.get("RANK", "0")) == "0" and not self._physrope_txt_printed:
                            self._physrope_txt_printed = True
                            print(f"[PhysRoPE FORWARD] (Refrain.py:L1829) txt PhysRoPE ENABLED: "
                                  f"txt_phys_cos.shape={txt_phys_cos.shape} | "
                                  f"txt_start_rows.shape={segment_layout['txt_start_rows'].shape} | "
                                  f"disable_text_physical_rope={disable_text_physical_rope}")

        # SP chunk for PhysRoPE
        if get_sequence_parallel_state() and img_physical_freqs_cis is not None:
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            pfc = img_physical_freqs_cis[0]
            split_dim = 1 if pfc.ndim == 3 else 0
            img_physical_freqs_cis = (
                torch.chunk(pfc, sp_size, dim=split_dim)[sp_rank],
                torch.chunk(img_physical_freqs_cis[1], sp_size, dim=split_dim)[sp_rank],
            )

        if len(self.double_blocks) > 0:
            if _physrope_v2_applied:
                if (
                    physrope_v2_block_freqs_cis is None
                    or freqs_cis is not physrope_v2_block_freqs_cis
                ):
                    raise RuntimeError(
                        "PhysRoPE v2 is active but double blocks are not receiving "
                        "the routed custom frequency tuple"
                    )
                if (
                    str(os.environ.get("RANK", "0")) == "0"
                    and not self._physrope_v2_double_block_printed
                ):
                    self._physrope_v2_double_block_printed = True
                    print(
                        "[PhysRoPE v2 DOUBLE-BLOCK INPUT] "
                        f"double_blocks={len(self.double_blocks)} "
                        f"single_blocks={len(self.single_blocks)} "
                        f"freqs_shape={tuple(freqs_cis[0].shape)} "
                        f"cos_ptr={freqs_cis[0].data_ptr()} "
                        f"sin_ptr={freqs_cis[1].data_ptr()} "
                        "source=custom_physrope_v2 verified=True",
                        flush=True,
                    )
            if output_features:
                features_list = []
            # --------------------- Pass through DiT blocks ------------------------
            for index, block in enumerate(self.double_blocks):
                if block is None:
                    continue
                force_full_attn = False
                sta_force_full_attn = self.attn_mode in ['sta_flex', 'ssta_flex', "flex_sparse_attn"] and \
                                    self.attn_param['win_type'] == 'hybrid' and \
                                    self.attn_param['win_ratio'] > 0 and \
                                    ((index+1) % self.attn_param['win_ratio'] == 0 or (index+1) == len(self.double_blocks))
                block_force_full_attn = self.attn_mode=="block_attn" and \
                                    self.attn_param['ba_attn_full_attn_every_n_blocks'] > 0 and \
                                    ((index+1) % self.attn_param['ba_attn_full_attn_every_n_blocks'] == 0 or (index+1) == len(self.double_blocks))
                force_full_attn = sta_force_full_attn or block_force_full_attn
                self.attn_param['layer-name'] = f"double_block_{index+1}"
                img, txt = block(img=img, txt=txt, vec=vec, freqs_cis=freqs_cis, text_mask=text_mask,
                            attn_param=self.attn_param,
                            is_flash=force_full_attn,
                            is_token_replace=is_token_replace,
                            token_replace_vec=token_replace_vec,
                            freqs_cis_cond=freqs_cis_cond,
                            img_physical_freqs_cis=img_physical_freqs_cis,
                            txt_physical_freqs_cis=txt_physical_freqs_cis,
                            first_frame_mask=first_frame_mask)

                if output_features and index == len(self.double_blocks) // 2:
                    img_feat = img
                    B, L, D = img_feat.shape
                    ori_h = th * 16
                    ori_w = tw * 16
                    ori_t = tt
                    
                    if "dinov3" in self.repa_type:
                        x_out = img_feat
                        target_token_length = th * tw * tt
                    else:
                        assert "qwen-3vl" in self.repa_type
                        patch_base = 32
                        target_h_grid = int(round(ori_h / patch_base))
                        target_w_grid = int(round(ori_w / patch_base))
                        target_token_length = target_h_grid * target_w_grid * ori_t
                        
                        x_2d = rearrange(
                            img_feat, 
                            "B (T H W) D -> (B T) D H W", 
                            T=tt, H=th, W=tw
                        )
                        
                        x_interpolated = F.interpolate(
                            x_2d, 
                            size=(target_h_grid, target_w_grid), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        x_out = rearrange(
                            x_interpolated, 
                            "(B T) D H W -> B (T H W) D", 
                            B=B, T=tt
                        )
                    out_feature = self.repa_projector(x_out.reshape(-1,D)).reshape(B, target_token_length, -1)
                    features_list.append(out_feature)
                
                if self.moe_config is not None:
                    self.loss_moe[0] = block.get_moe_loss()[0] + self.loss_moe[0]
                    self.loss_moe[1] = block.get_moe_loss()[1] + self.loss_moe[1]

                    # self.count_top[0][0].append(block.get_count_top()[0][0])
                    # self.count_top[0][1].append(block.get_count_top()[0][1])
                    # self.count_top[1][0].append(block.get_count_top()[1][0])
                    # self.count_top[1][1].append(block.get_count_top()[1][1])
        
        if self.pp_enabled and self.only_double and pp_x is not None:
            pp_x.grad = torch.zeros_like(pp_x)

        if not self.pp_postprocess and self.only_double:
            assert self.pp_enabled
            ret_tensors = [self.dummy_tensor, img.contiguous(), txt, vec, text_mask.float(),]
            if self.moe_config is not None:
                ret_tensors.append(self.loss_moe[0])
                ret_tensors.append(self.loss_moe[1])
            if is_token_replace:
                if self.moe_config is None:
                    ret_tensors.append(torch.tensor([-3]))
                    ret_tensors.append(torch.tensor([-3]))
                ret_tensors.append(token_replace_vec)
            return tuple(ret_tensors)
        

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        if (self.pp_preprocess and self.only_single) or self.mix_stage or (self.pp_postprocess and self.only_double):
            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
        else:
            assert self.pp_enabled
            x = pp_x

        if self.mix_stage or self.only_single:
            if len(self.single_blocks) > 0:
                for index, block in enumerate(self.single_blocks):
                    force_full_attn = False
                    sta_force_full_attn = self.attn_mode in ['sta_flex', 'ssta_flex', "flex_sparse_attn"] and \
                                        self.attn_param['win_type'] == 'hybrid' and \
                                        self.attn_param['win_ratio'] > 0 and \
                                        ((index+1) % self.attn_param['win_ratio'] == 0 or (index+1) == len(self.single_blocks))
                    block_force_full_attn = self.attn_mode=="block_attn" and \
                                        self.attn_param['ba_attn_full_attn_every_n_blocks'] > 0 and \
                                        ((index+1) % self.attn_param['ba_attn_full_attn_every_n_blocks'] == 0 or (index+1) == len(self.single_blocks))
                    force_full_attn = sta_force_full_attn or block_force_full_attn
                    self.attn_param['layer-name'] = f"single_block_{index+1}"
                    if block is None:
                        continue
                    x = block(
                            x=x, vec=vec, txt_len=txt_seq_len,
                            freqs_cis=(freqs_cos, freqs_sin),
                            text_mask=text_mask,
                            attn_param=self.attn_param,
                            is_flash=force_full_attn,
                            is_token_replace=is_token_replace,
                            token_replace_vec=token_replace_vec,
                            first_frame_mask=first_frame_mask,
                            freqs_cis_cond=freqs_cis_cond,
                            img_physical_freqs_cis=img_physical_freqs_cis,
                            txt_physical_freqs_cis=txt_physical_freqs_cis,
                    )       
                    if self.moe_config is not None:
                        self.loss_moe[0] = self.loss_moe[0] + block.get_moe_loss()[0]
                        self.loss_moe[1] = self.loss_moe[1] + block.get_moe_loss()[1]

        if not self.pp_postprocess and self.only_single:
            assert self.pp_enabled
            ret_tensors = [x.contiguous(), img.contiguous(), txt, vec, text_mask.float(),]
            if self.moe_config is not None:
                ret_tensors.append(self.loss_moe[0])
                ret_tensors.append(self.loss_moe[1])
            if is_token_replace:
                if self.moe_config is None:
                    ret_tensors.append(torch.tensor([-3]))
                    ret_tensors.append(torch.tensor([-3]))
                ret_tensors.append(token_replace_vec)
            return tuple(ret_tensors)

        if self.pp_postprocess:
            img = x[:, :img_seq_len, ...]

            # ---------------------------- Final layer ------------------------------
            img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
            if get_sequence_parallel_state():
                if nccl_info.use_dynamic_ring_attention:
                    img = postprocess_dynamic_ring_attention(img)
                else:
                    img = all_gather(img, dim=1, group=nccl_info.sp_group)
            img = self.unpatchify(img, tt, th, tw)
            shape = (tt, th, tw)
            assert return_dict == False, "return_dict is not supported."
            if output_features:
                features_list = torch.stack(features_list, dim=0)
                if get_sequence_parallel_state():
                    features_list = all_gather(features_list, dim=2, group=nccl_info.group)
            else:
                features_list = None
            return (img, features_list)
    
    def get_moe_loss(self):
        return self.loss_moe

    def get_count_top(self):
        raise NotImplementedError("get_count_top is not implemented for Refrain")
        return self.count_top

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_q.parameters())
                    + sum(p.numel() for p in block.img_attn_k.parameters())
                    + sum(p.numel() for p in block.img_attn_v.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_q.parameters())
                    + sum(p.numel() for p in block.txt_attn_k.parameters())
                    + sum(p.numel() for p in block.txt_attn_v.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks if block is not None
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1_q.parameters())
                    + sum(p.numel() for p in block.linear1_k.parameters())
                    + sum(p.numel() for p in block.linear1_v.parameters())
                    # + sum(p.numel() for p in block.linear1_mlp.parameters())
                    # + sum(p.numel() for p in block.linear2.parameters())
                    + sum(p.numel() for p in block.mlp.parameters())
                    for block in self.single_blocks if block is not None
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts
