import os
import loguru
from .Refrain import RefrainTransformer

MODEL_CONFIG = {
    'Refrain': {
        "in_channels": 48,
        "out_channels": 48,
        'mm_double_blocks_depth': 24,
        'mm_single_blocks_depth': 0,
        'rope_dim_list': [32, 48, 48],
        'hidden_size': 3072,
        'heads_num': 24,
        'mlp_width_ratio': 4,
        'patch_size': [1, 1, 1],
        'is_dense_with_moe_mlp': True,
        'dense_with_moe_mlp_intermediate_size': 8192
    },
}


def load_model(model_type, args, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    factor_kwargs['attn_mode'] = args.attn_mode

    # condtion Type Embedding
    factor_kwargs['use_cond_type_embedding'] = args.use_cond_type_embedding

    # use scale rope
    factor_kwargs['use_scale_rope'] = args.use_scale_rope


    # condition 3D-ROPE
    factor_kwargs['use_cond_3d_rope'] = args.use_cond_3d_rope
    factor_kwargs['rope_theta'] = args.rope_theta

    # PhysRoPE
    if getattr(args, 'use_physical_rope', False):
        factor_kwargs['physical_rope_dim'] = getattr(args, 'physical_rope_dim', 32)
        if str(os.environ.get("RANK", "0")) == "0":
            print(f"[PhysRoPE INIT] use_physical_rope=True, "
                  f"physical_rope_dim={factor_kwargs['physical_rope_dim']}")
    else:
        factor_kwargs['physical_rope_dim'] = 0

    if model_type == "Refrain" or "16164" in model_type:
        factor_kwargs['is_reshape_temporal_channels'] = False
    else:
        factor_kwargs['is_reshape_temporal_channels'] = True
    if args.attn_mode != "flash" and args.attn_mode != "flash3":
        #block attn
        factor_kwargs["ba_attn_chunk_size"]=args.ba_attn_chunk_size
        factor_kwargs["ba_attn_topk"]=args.ba_attn_topk
        factor_kwargs["ba_attn_causal"]=args.ba_attn_causal
        factor_kwargs["ba_attn_slice_method"]=args.ba_attn_slice_method
        factor_kwargs["ba_attn_full_attn_block_num"]=args.ba_attn_full_attn_block_num
        factor_kwargs["ba_attn_full_attn_every_n_blocks"]=args.ba_attn_full_attn_every_n_blocks

        # ssta & sta
        factor_kwargs['win_size'] = [int(s) for s in args.win_size.split('_')]
        factor_kwargs['win_type'] = args.win_type
        factor_kwargs['win_ratio'] = args.win_ratio
        factor_kwargs['tile_size'] = [int(s) for s in args.tile_size.split('_')]
        factor_kwargs['ssta_topk'] = args.ssta_topk
        factor_kwargs['ssta_threshold'] = args.ssta_threshold
        factor_kwargs['ssta_enforce_share_topk_block'] = args.ssta_enforce_share_topk_block
        factor_kwargs['sparse_attn_sparse_type'] = args.sparse_attn_sparse_type
        factor_kwargs['sparse_attn_pad_type'] = args.sparse_attn_pad_type
        factor_kwargs['sparse_attn_use_text_mask'] = args.sparse_attn_use_text_mask
        factor_kwargs['sparse_attn_mask_share_within_head'] = args.sparse_attn_mask_share_within_head

    if 'moe_config' in MODEL_CONFIG[model_type]:
        raise ValueError("The public training engine supports the dense Refrain configuration.")
    args.use_moe = False

    if 'Refrain' in model_type:
        factor_kwargs['text_pool_type'] = args.text_pool_type
        factor_kwargs['text_states_dim_2'] = args.text_states_dim_2
        factor_kwargs['glyph_byT5_v2'] = args.glyph_byT5_v2
        factor_kwargs['text_states_dim'] = args.text_states_dim

    if args.multitask_mask_training_type!=None:
        factor_kwargs['multitask_mask_training_type'] = args.multitask_mask_training_type
        factor_kwargs['vision_projection'] = args.vision_projection
        factor_kwargs['vision_states_dim'] = args.vision_states_dim

    if model_type in MODEL_CONFIG.keys():
        if 'Refrain' in model_type:
            model = RefrainTransformer(
                **MODEL_CONFIG[model_type],
                **factor_kwargs,
            )
        return model
    else:
        print(f"model_type={model_type} not in {list(MODEL_CONFIG.keys())}")
        raise NotImplementedError()
