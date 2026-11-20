from sympy import use
import torch
import torch.nn as nn
import os
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from peft.utils.other import fsdp_auto_wrap_policy

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

from .load import get_no_split_modules

from functools import partial

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
import functools


non_reentrant_wrapper = partial(
    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    # checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT,
)


def apply_fsdp_checkpointing(model, no_split_modules, p=1):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing...")
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        # Gradient communication precision.
        reduce_dtype=weight_type,
        # Buffer precision.
        buffer_dtype=weight_type,
        cast_forward_inputs=True,
    )
    return mixed_precision


def get_dit_fsdp_kwargs(
    transformer,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
):
    no_split_modules = get_no_split_modules(transformer)
    if use_lora:
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=no_split_modules,
        )

    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    device_id = torch.cuda.current_device()
    cpu_offload = (
        torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    )
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
        "use_orig_params": True, 
    }

    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update(
            {
                "use_orig_params": False,  # Required for LoRA memory savings
                "sync_module_states": True,
            }
        )

    return fsdp_kwargs, no_split_modules


def get_dit_fsdp_kwargs_v2(
    transformer,
    parallel_dims,
    world_mesh,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
):
    # Safety check for None parallel_dims
    if parallel_dims is None:
        raise ValueError("parallel_dims cannot be None. Please ensure parallel_dims and world_mesh are properly initialized.")
    
    no_split_modules = get_no_split_modules(transformer)
    if use_lora:
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=no_split_modules,
        )

    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)


    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
        assert not parallel_dims.dp_replicate_enabled
    elif sharding_strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        assert parallel_dims.dp_replicate_enabled
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if parallel_dims.dp_shard_enabled or parallel_dims.sp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_sp")
            sharding_strategy = ShardingStrategy.HYBRID_SHARD       # zero3 within each node 
            #sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2 # zero2 within each node 
        else:
            dp_mesh_dim_names = ("dp_shard_sp",)
            sharding_strategy = ShardingStrategy.FULL_SHARD
    else:
        dp_mesh_dim_names = ("dp_replicate", )
        sharding_strategy = ShardingStrategy.NO_SHARD #DDP
        

    device_mesh = world_mesh[tuple(dp_mesh_dim_names)]
    cpu_offload = (
        torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    )
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        #"device_id": torch.cuda.current_device(),
        "device_mesh": device_mesh,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
        "use_orig_params": True, 
    }

    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update(
            {
                "use_orig_params": False,  # Required for LoRA memory savings
                "sync_module_states": True,
            }
        )

    return fsdp_kwargs, no_split_modules


def get_discriminator_fsdp_kwargs(
    parallel_dims,
    world_mesh,
    sharding_strategy,
    cpu_offload=False,
    master_weight_type="fp32",
):
    auto_wrap_policy = None

    # Use existing mixed precision settings

    mixed_precision = get_mixed_precision(master_weight_type)
    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if parallel_dims.dp_shard_enabled or parallel_dims.sp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_sp")
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            dp_mesh_dim_names = ("dp_shard_sp",)
            sharding_strategy = ShardingStrategy.FULL_SHARD

    # device_id = torch.cuda.current_device()
    device_mesh = world_mesh[tuple(dp_mesh_dim_names)]
    cpu_offload = (
        torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    )
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        # "device_id": device_id,
        "device_mesh": device_mesh,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
    }

    return fsdp_kwargs


def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


def verify_sp_rank_consistency(
    tensors_dict,
    sp_group,
    sp_rank,
    dp_rank,
    logger=None,
    tolerance=1e-5,
    enable_check=True,
):
    """Verify that all ranks in one SP group received identical rollout tensors.

    GRPO computes one rollout per data-parallel sample, while sequence-parallel
    ranks should see the same rollout tensors before the sequence split is
    applied.  The check is intentionally one tensor at a time: gather cheap
    shape metadata first, then broadcast the sp_rank=0 tensor as reference.
    """
    if not enable_check or sp_group is None:
        return True
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    if not tensors_dict:
        if logger is not None:
            logger.warning("[SP Consistency] no tensors provided; skip check")
        return True

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size(sp_group)
    if world_size <= 1:
        return True

    try:
        group_ranks = torch.distributed.get_process_group_ranks(sp_group)
        src_rank = int(group_ranks[0])
    except Exception:
        src_rank = int(rank) - int(sp_rank)

    def _log_info(message):
        if logger is not None and int(sp_rank) == 0:
            logger.info(message)

    def _log_error(message):
        if logger is not None:
            logger.error(message)
        else:
            print(message)

    max_ndim = 15
    for name, tensor in tensors_dict.items():
        if tensor is None or not isinstance(tensor, torch.Tensor):
            continue

        tensor = tensor.detach()
        if tensor.dim() > max_ndim:
            raise ValueError(
                f"[SP Consistency] tensor '{name}' has ndim={tensor.dim()}, "
                f"larger than supported max_ndim={max_ndim}"
            )

        meta_device = tensor.device if tensor.is_cuda else torch.device("cuda", torch.cuda.current_device())
        shape_meta = torch.full((max_ndim + 1,), -1, dtype=torch.long, device=meta_device)
        shape_meta[0] = tensor.dim()
        if tensor.dim() > 0:
            shape_meta[1:1 + tensor.dim()] = torch.tensor(tensor.shape, dtype=torch.long, device=meta_device)
        gathered_shapes = [torch.empty_like(shape_meta) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_shapes, shape_meta, group=sp_group)

        shape_list = [tuple(int(x) for x in item[1:1 + int(item[0].item())].tolist()) for item in gathered_shapes]
        if any(shape != shape_list[0] for shape in shape_list):
            message = (
                f"[SP Consistency] shape mismatch for '{name}' at dp_rank={dp_rank}: "
                f"sp_group_shapes={shape_list}"
            )
            _log_error(message)
            raise ValueError(message)

        ref = tensor.contiguous().clone()
        torch.distributed.broadcast(ref, src=src_rank, group=sp_group)
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            local_ok = torch.allclose(tensor, ref, atol=float(tolerance), rtol=0.0, equal_nan=True)
            max_diff = float((tensor.to(torch.float32) - ref.to(torch.float32)).abs().max().item()) if tensor.numel() > 0 else 0.0
        else:
            local_ok = torch.equal(tensor, ref)
            max_diff = 0.0

        ok_tensor = torch.tensor(1 if local_ok else 0, dtype=torch.int32, device=meta_device)
        torch.distributed.all_reduce(ok_tensor, op=torch.distributed.ReduceOp.MIN, group=sp_group)
        if int(ok_tensor.item()) != 1:
            message = (
                f"[SP Consistency] value mismatch for '{name}' at dp_rank={dp_rank}, "
                f"sp_rank={sp_rank}, max_diff={max_diff}, tolerance={tolerance}"
            )
            _log_error(message)
            raise ValueError(message)

    _log_info(f"[SP Consistency] verified {len(tensors_dict)} tensors at dp_rank={dp_rank}")
    return True


#TODO no working now, I don't figure it out 
def apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for layer_id, transformer_block in enumerate(model.double_blocks + model.single_blocks):
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(layer_id) < len(model.double_blocks+model.single_blocks) - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)


def get_coordinate(rank, dp_replicate, dp_shard, ts, sp):
    total_ts_sp = ts * sp
    total_dp_shard_ts_sp = dp_shard * total_ts_sp

    id_dp_repl = rank // total_dp_shard_ts_sp
    remaining = rank % total_dp_shard_ts_sp

    id_dp_shard = remaining // total_ts_sp
    remaining %= total_ts_sp

    id_ts = remaining // sp
    id_sp = remaining % sp

    return (id_dp_repl, id_dp_shard, id_ts, id_sp)


def apply_compile(model):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model after applying DP.
    """
    for layer_id, transformer_block in model.double_blocks.named_children():
        transformer_block = torch.compile(transformer_block)
        model.double_blocks.register_module(layer_id, transformer_block)

    for layer_id, transformer_block in model.single_blocks.named_children():
        transformer_block = torch.compile(transformer_block)
        model.single_blocks.register_module(layer_id, transformer_block)
