"""Utilities for native-length temporal reference concatenation."""

from typing import Iterable, List, Sequence, Tuple

import torch


Range = Tuple[int, int]


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "on", "1", "yes")
    return bool(value)


def normalize_ranges(ranges: Iterable[Sequence[int]], name: str) -> List[Range]:
    normalized = []
    for item in ranges or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name} entries must be [start, end] pairs, got {item!r}")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end <= start:
            raise ValueError(f"{name} contains invalid range {(start, end)}")
        normalized.append((start, end))
    return normalized


def validate_native_reference_layout(
    reference_t: int,
    target_t: int,
    packed_ranges: Iterable[Sequence[int]],
    dst_ranges: Iterable[Sequence[int]],
) -> Tuple[List[Range], List[Range]]:
    reference_t = int(reference_t)
    target_t = int(target_t)
    if reference_t <= 0 or target_t <= 0:
        raise ValueError(
            f"native reference and target lengths must be positive, got ref={reference_t} target={target_t}"
        )
    packed = normalize_ranges(packed_ranges, "native_reference_packed_ranges")
    dst = normalize_ranges(dst_ranges, "condition_frame_ranges")
    if not packed or len(packed) != len(dst):
        raise ValueError(
            "native reference packed/destination ranges must be non-empty and parallel: "
            f"packed={packed} dst={dst}"
        )
    cursor = 0
    for packed_range, dst_range in zip(packed, dst):
        start, end = packed_range
        if start != cursor:
            raise ValueError(
                "native reference streams must cover the prefix contiguously in slot order: "
                f"expected_start={cursor} got={packed_range} all={packed}"
            )
        if end > reference_t:
            raise ValueError(
                f"native reference range {packed_range} exceeds reference_t={reference_t}"
            )
        ds, de = dst_range
        if de > target_t:
            raise ValueError(
                f"condition destination range {dst_range} exceeds target_t={target_t}"
            )
        cursor = end
    if cursor != reference_t:
        raise ValueError(
            f"native reference ranges end at {cursor}, but reference tensor has T={reference_t}"
        )
    return packed, dst


def build_native_temporal_positions(
    reference_t: int,
    target_t: int,
    packed_ranges: Iterable[Sequence[int]],
    dst_ranges: Iterable[Sequence[int]],
    *,
    device=None,
) -> torch.Tensor:
    """Map native reference frames into target coordinates without resampling features."""
    packed, dst = validate_native_reference_layout(
        reference_t, target_t, packed_ranges, dst_ranges
    )
    positions = torch.empty(
        int(reference_t) + int(target_t), dtype=torch.float32, device=device
    )
    for (start, end), (ds, de) in zip(packed, dst):
        src_len = end - start
        dst_len = de - ds
        if src_len == 1 or dst_len == 1:
            positions[start:end] = float(ds)
        else:
            positions[start:end] = float(ds) + (
                torch.arange(src_len, dtype=torch.float32, device=device)
                * (float(dst_len - 1) / float(src_len - 1))
            )
    positions[int(reference_t):] = torch.arange(
        int(target_t), dtype=torch.float32, device=device
    )
    return positions


def pack_native_reference_input(
    target_latents: torch.Tensor,
    reference_latents: torch.Tensor,
    packed_ranges: Iterable[Sequence[int]],
    dst_ranges: Iterable[Sequence[int]],
) -> Tuple[torch.Tensor, int]:
    """Build [noisy, clean-reference, mask] streams and concatenate them in T."""
    if target_latents.ndim != 5 or reference_latents.ndim != 5:
        raise ValueError(
            "target_latents and reference_latents must be [B,C,T,H,W], got "
            f"target={tuple(target_latents.shape)} reference={tuple(reference_latents.shape)}"
        )
    if target_latents.shape[:2] != reference_latents.shape[:2]:
        raise ValueError(
            "native T concat requires matching batch/channel dimensions, got "
            f"target={tuple(target_latents.shape[:2])} reference={tuple(reference_latents.shape[:2])}"
        )
    if target_latents.shape[-2:] != reference_latents.shape[-2:]:
        raise ValueError(
            "native T concat requires matching spatial dimensions, got "
            f"target={tuple(target_latents.shape[-2:])} reference={tuple(reference_latents.shape[-2:])}"
        )
    reference_latents = reference_latents.to(
        device=target_latents.device, dtype=target_latents.dtype
    )
    reference_t = int(reference_latents.shape[2])
    target_t = int(target_latents.shape[2])
    validate_native_reference_layout(reference_t, target_t, packed_ranges, dst_ranges)

    reference_noisy = torch.zeros_like(reference_latents)
    reference_mask = torch.ones(
        reference_latents.shape[0], 1, reference_t,
        reference_latents.shape[3], reference_latents.shape[4],
        device=reference_latents.device, dtype=reference_latents.dtype,
    )
    reference_block = torch.cat(
        [reference_noisy, reference_latents, reference_mask], dim=1
    )

    target_reference = torch.zeros_like(target_latents)
    target_mask = torch.zeros(
        target_latents.shape[0], 1, target_t,
        target_latents.shape[3], target_latents.shape[4],
        device=target_latents.device, dtype=target_latents.dtype,
    )
    target_block = torch.cat([target_latents, target_reference, target_mask], dim=1)
    return torch.cat([reference_block, target_block], dim=2), reference_t


def slice_native_target_output(model_output: torch.Tensor, reference_t: int, target_t: int) -> torch.Tensor:
    if model_output.ndim != 5:
        raise ValueError(f"model_output must be [B,C,T,H,W], got {tuple(model_output.shape)}")
    reference_t = int(reference_t)
    target_t = int(target_t)
    expected_t = reference_t + target_t
    if model_output.shape[2] != expected_t:
        raise ValueError(
            f"native packed model output T={model_output.shape[2]} must equal ref+target={expected_t}"
        )
    return model_output[:, :, reference_t:expected_t]
