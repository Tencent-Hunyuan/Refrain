"""Fixed-canvas temporal packing for variable-length video references.

The reference-timeline path never interpolates, repeats, averages, or synthesizes
reference latents. It copies exact source latent frames into a fixed target
timeline and leaves all other condition positions empty.
"""

import math
from typing import Dict, List, Sequence, Tuple

import torch


LEGACY_TRILINEAR = "legacy_trilinear"
REFERENCE_TIMELINE = "reference_timeline"
SUPPORTED_TEMPORAL_PACK_MODES = (LEGACY_TRILINEAR, REFERENCE_TIMELINE)


def normalize_temporal_pack_mode(value) -> str:
    mode = str(value or LEGACY_TRILINEAR).strip().lower()
    if mode not in SUPPORTED_TEMPORAL_PACK_MODES:
        raise ValueError(
            f"unsupported range-condition temporal pack mode {value!r}; "
            f"expected one of {SUPPORTED_TEMPORAL_PACK_MODES}"
        )
    return mode


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _as_range(value: Sequence[int], name: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a [start, end] pair, got {value!r}")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start:
        raise ValueError(f"{name} must be a positive half-open range, got {(start, end)}")
    return start, end


def build_reference_timeline_mapping(
    source_t: int,
    physical_dst_range: Sequence[int],
    visible_dst_range: Sequence[int] = None,
) -> Dict[str, object]:
    """Build an endpoint-preserving exact-frame mapping.

    For source_t <= destination_t every source frame is retained and scattered
    sparsely over the destination interval. For source_t > destination_t one
    exact source frame is selected for every destination position. Selection
    never changes the first reference frame/start point.
    """

    source_t = int(source_t)
    if source_t <= 0:
        raise ValueError(f"source_t must be positive, got {source_t}")
    ds, de = _as_range(physical_dst_range, "physical_dst_range")
    if visible_dst_range is None:
        visible_ds, visible_de = ds, de
    else:
        visible_ds, visible_de = _as_range(visible_dst_range, "visible_dst_range")
        if visible_ds < ds or visible_de > de:
            raise ValueError(
                f"visible range {(visible_ds, visible_de)} must be inside physical range {(ds, de)}"
            )

    dst_t = de - ds
    if source_t == 1 or dst_t == 1:
        source_indices = [0]
        target_indices = [ds]
    elif source_t <= dst_t:
        source_indices = list(range(source_t))
        target_indices = [
            ds + _round_half_up(i * float(dst_t - 1) / float(source_t - 1))
            for i in source_indices
        ]
    else:
        target_indices = list(range(ds, de))
        source_indices = [
            _round_half_up(j * float(source_t - 1) / float(dst_t - 1))
            for j in range(dst_t)
        ]

    if source_indices != sorted(set(source_indices)):
        raise RuntimeError(f"reference-timeline source indices are not strictly increasing: {source_indices}")
    if target_indices != sorted(set(target_indices)):
        raise RuntimeError(f"reference-timeline target indices are not strictly increasing: {target_indices}")
    if source_indices[0] != 0 or target_indices[0] != ds:
        raise RuntimeError(
            "reference-timeline mapping moved the selected reference start: "
            f"source={source_indices} target={target_indices} physical_dst={(ds, de)}"
        )
    if source_t > 1 and dst_t > 1:
        if source_indices[-1] != source_t - 1 or target_indices[-1] != de - 1:
            raise RuntimeError(
                "reference-timeline mapping must preserve both endpoints: "
                f"source={source_indices} target={target_indices}"
            )

    visible_pairs = [
        (source_idx, target_idx)
        for source_idx, target_idx in zip(source_indices, target_indices)
        if visible_ds <= target_idx < visible_de
    ]
    return {
        "source_t": source_t,
        "physical_dst_range": (ds, de),
        "visible_dst_range": (visible_ds, visible_de),
        "physical_dst_t": dst_t,
        "relation": "short" if source_t < dst_t else "long" if source_t > dst_t else "equal",
        "source_indices": source_indices,
        "target_indices": target_indices,
        "visible_source_indices": [pair[0] for pair in visible_pairs],
        "visible_target_indices": [pair[1] for pair in visible_pairs],
    }


def build_reference_timeline_physical_clock(
    source_t: int,
    physical_dst_range: Sequence[int],
) -> Dict[str, object]:
    """Return the condition-independent target clock used by ReferenceTimeline.

    The mapping metadata still records exact source/target evidence pairs, but
    source length and sparse rounding never alter target PhysRoPE positions.
    """

    mapping = build_reference_timeline_mapping(source_t, physical_dst_range)
    dst_t = int(mapping["physical_dst_t"])
    target_local_indices = [
        int(target_idx) - int(mapping["physical_dst_range"][0])
        for target_idx in mapping["target_indices"]
    ]
    source_indices = [int(value) for value in mapping["source_indices"]]
    target_standard_positions = [float(idx) for idx in range(dst_t)]
    physical_local_positions = list(target_standard_positions)
    step_deltas = [1.0] * max(dst_t - 1, 0)
    return {
        **mapping,
        "target_local_indices": target_local_indices,
        "source_local_positions": [float(value) for value in source_indices],
        "physical_local_positions": physical_local_positions,
        "source_endpoint_linear_positions": [],
        "target_standard_positions": target_standard_positions,
        "target_clock_errors": [0.0] * dst_t,
        "anchor_errors": [],
        "step_deltas": step_deltas,
        "endpoint_slope": 1.0,
        "source_to_target_scale": None,
        "source_interval_endpoint": float(mapping["source_t"]),
        "target_interval_endpoint": float(dst_t),
        "scaled_source_interval_endpoint": None,
        "effective_duration_t": dst_t,
        "duration_delta": 0,
        "singleton_no_warp": True,
        "has_physical_warp": False,
        "clock_policy": "fixed_target_timeline_condition_independent",
    }


def scatter_reference_timeline_latent(
    source_latent: torch.Tensor,
    target_t: int,
    physical_dst_range: Sequence[int],
    visible_dst_range: Sequence[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Copy exact [C,S,H,W] source frames into a fixed [C,T,H,W] canvas."""

    if not torch.is_tensor(source_latent) or source_latent.ndim != 4:
        raise ValueError(
            f"source_latent must be a [C,S,H,W] tensor, got "
            f"{type(source_latent)} shape={getattr(source_latent, 'shape', None)}"
        )
    target_t = int(target_t)
    if target_t <= 0:
        raise ValueError(f"target_t must be positive, got {target_t}")
    mapping = build_reference_timeline_mapping(
        int(source_latent.shape[1]), physical_dst_range, visible_dst_range
    )
    _, physical_de = mapping["physical_dst_range"]
    if int(physical_de) > target_t and visible_dst_range is None:
        raise ValueError(
            f"physical destination end {physical_de} exceeds target_t={target_t}"
        )

    canvas = torch.zeros(
        source_latent.shape[0],
        target_t,
        source_latent.shape[2],
        source_latent.shape[3],
        dtype=source_latent.dtype,
        device=source_latent.device,
    )
    mask = torch.zeros(target_t, dtype=torch.float32, device=source_latent.device)
    source_indices: List[int] = mapping["visible_source_indices"]
    target_indices: List[int] = mapping["visible_target_indices"]
    if source_indices:
        source_index_tensor = torch.tensor(
            source_indices, dtype=torch.long, device=source_latent.device
        )
        target_index_tensor = torch.tensor(
            target_indices, dtype=torch.long, device=source_latent.device
        )
        canvas[:, target_index_tensor] = source_latent[:, source_index_tensor]
        mask[target_index_tensor] = 1.0
    return canvas, mask, mapping
