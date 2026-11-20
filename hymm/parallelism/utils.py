"""Synchronization of sample inputs and RNG states within an SP group."""
import random
import numpy as np
import torch
import torch.distributed as dist
from .parallel_states import get_parallel_state


def sync_object_for_parallel_training(value, parallel_dims=None, force_object=False,
                                      debug_with_check=False, trace=None):
    state = parallel_dims or get_parallel_state()
    if not dist.is_initialized() or not state.sp_enabled:
        return value
    group = state.sp_mesh.get_group()
    src = dist.get_global_rank(group, 0)
    if force_object:
        payload = [value]
        dist.broadcast_object_list(payload, src=src, group=group)
        return payload[0]
    if debug_with_check:
        if not isinstance(value, torch.Tensor):
            raise TypeError("debug_with_check expects a tensor")
        original = value.detach().clone()
        reference = sync_object_for_parallel_training(value.detach().clone(), state)
        matches = original.shape == reference.shape and (
            torch.allclose(original, reference, equal_nan=True) if original.is_floating_point()
            else torch.equal(original, reference))
        device = torch.cuda.current_device() if dist.get_backend(group) == "nccl" else "cpu"
        flag = torch.tensor(int(matches), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        if not flag.item():
            raise RuntimeError(f"SP values disagree: {trace or 'tensor'}")
        return value

    def describe(obj):
        if isinstance(obj, torch.Tensor):
            return ("tensor", tuple(obj.shape), obj.dtype, obj.device.type)
        if isinstance(obj, dict): return ("dict", [(k, describe(v)) for k, v in obj.items()])
        if isinstance(obj, list): return ("list", [describe(v) for v in obj])
        if isinstance(obj, tuple): return ("tuple", [describe(v) for v in obj])
        return ("object", obj)

    # Reference counts, shapes and optional fields can vary between workers.
    # Share the source structure before issuing any tensor collectives.
    source_rank = dist.get_rank() == src
    metadata = [describe(value) if source_rank else None]
    dist.broadcast_object_list(metadata, src=src, group=group)
    nccl = dist.get_backend(group) == "nccl"

    def transfer(spec, current):
        kind = spec[0]
        if kind == "tensor":
            _, shape, dtype, device_type = spec
            target_device = torch.device("cuda", torch.cuda.current_device()) if device_type == "cuda" else torch.device(device_type)
            communication_device = torch.device("cuda", torch.cuda.current_device()) if nccl else target_device
            if source_rank:
                buffer = current.detach().to(communication_device).contiguous()
            else:
                buffer = torch.empty(shape, dtype=dtype, device=communication_device)
            dist.broadcast(buffer, src=src, group=group)
            if isinstance(current, torch.Tensor) and current.dtype == dtype and current.device == target_device:
                with torch.no_grad():
                    if tuple(current.shape) != shape:
                        if current.requires_grad:
                            return buffer.to(target_device)
                        current.resize_(shape)
                    current.copy_(buffer)
                return current
            return buffer.to(target_device)
        if kind == "dict":
            result = current if isinstance(current, dict) else {}
            source_keys = {k for k, _ in spec[1]}
            for k in list(result):
                if k not in source_keys: del result[k]
            for k, child in spec[1]: result[k] = transfer(child, result.get(k))
            return result
        if kind in ("list", "tuple"):
            old = current if isinstance(current, (list, tuple)) else []
            items = [transfer(child, old[i] if i < len(old) else None) for i, child in enumerate(spec[1])]
            if kind == "tuple": return tuple(items)
            if isinstance(current, list):
                current[:] = items
                return current
            return items
        return spec[1]

    return transfer(metadata[0], value)


def sync_random_states(parallel_dims=None):
    state = parallel_dims or get_parallel_state()
    if not dist.is_initialized() or not state.sp_enabled:
        return
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None)
    rng = sync_object_for_parallel_training(rng, state, force_object=True)
    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2].cpu())
    if rng[3] is not None: torch.cuda.set_rng_state(rng[3].cpu())
