#!/usr/bin/env python3
"""Device-agnostic resumable extraction of CSD style/content pair distances."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "outputs/sdc_inventory.json"
DEFAULT_CSD = ROOT / "weights/evaluation/csd/csd-vit-l.pth"
CSD_REPRESENTATION = "CSD_ViT_L14_OpenAI_QuickGELU_style_and_content_heads"
CSD_PREPROCESSING = "resize_shorter_side_224_bicubic_then_center_crop_224"
PAIR_GEOMETRY_CONTRACT = "exact_decoded_rgb_frame_cartesian_product_v1"


def read_video(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(np.ascontiguousarray(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise RuntimeError(f"no decoded RGB frames: {path}")
    return np.stack(frames)


def read_exact_sampled_frames(
    path: Path,
    spans_rgb: list[list[int]],
    frames_per_shot: int,
    guard_frames: int,
) -> tuple[list[np.ndarray], int, list[list[int]]]:
    """Decode the complete stream but materialize only exact sampled RGB frames.

    Container frame metadata is used only to know the indices before decoding.
    We still decode through end-of-stream and verify the exact decoded count. If
    metadata is absent or disagrees, the function visibly takes the conservative
    full-RGB fallback and recomputes indices from the actual count.  Thus this is
    an implementation optimization, never temporal remapping or frame skipping.
    """
    with av.open(str(path)) as container:
        metadata_count = int(container.streams.video[0].frames or 0)
    if metadata_count <= 0:
        frames = read_video(path)
        grouped = sample_rgb_indices(
            spans_rgb, len(frames), frames_per_shot, guard_frames
        )
        flat = [index for group in grouped for index in group]
        return [frames[index] for index in flat], len(frames), grouped

    try:
        grouped = sample_rgb_indices(
            spans_rgb, metadata_count, frames_per_shot, guard_frames
        )
    except RuntimeError:
        # Metadata is a performance hint, never an authority on temporal
        # semantics. Re-decoding makes the original validation authoritative.
        frames = read_video(path)
        grouped = sample_rgb_indices(
            spans_rgb, len(frames), frames_per_shot, guard_frames
        )
        flat = [index for group in grouped for index in group]
        return [frames[index] for index in flat], len(frames), grouped
    flat = [index for group in grouped for index in group]
    requested = set(flat)
    selected: dict[int, np.ndarray] = {}
    decoded_count = 0
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            decoded_count = index + 1
            if index in requested:
                selected[index] = np.ascontiguousarray(
                    frame.to_ndarray(format="rgb24")
                )
    if decoded_count != metadata_count:
        frames = read_video(path)
        grouped = sample_rgb_indices(
            spans_rgb, len(frames), frames_per_shot, guard_frames
        )
        flat = [index for group in grouped for index in group]
        return [frames[index] for index in flat], len(frames), grouped
    missing = [index for index in flat if index not in selected]
    if missing:
        raise RuntimeError(f"decoded stream lacks sampled RGB indices {missing}: {path}")
    return [selected[index] for index in flat], decoded_count, grouped


def sample_rgb_indices(
    spans_rgb: list[list[int]],
    frame_count: int,
    frames_per_shot: int,
    guard_frames: int,
) -> list[list[int]]:
    """Take up to ``frames_per_shot`` samples from guarded observed interiors.

    The shot interval is first intersected with the decoded stream and only
    then guarded on both sides.  This ordering is essential at the causal
    decoder tail: guarding the nominal endpoint before clipping would leave the
    final shot with no right guard.  Very short but valid interiors use every
    available frame down to a frozen minimum of two; shot-pair means keep their
    aggregation mass independent of that count.
    """
    if len(spans_rgb) < 2:
        raise RuntimeError("style consistency requires at least two shot intervals")
    normalized_spans = [[int(start), int(end)] for start, end in spans_rgb]
    for index, (start, end) in enumerate(normalized_spans):
        if start < 0 or end <= start:
            raise RuntimeError(f"invalid RGB shot interval [{start}-{end})")
        if index and start < normalized_spans[index - 1][1]:
            raise RuntimeError(
                "RGB shot intervals must be ordered and non-overlapping: "
                f"{normalized_spans[index - 1]} then {[start, end]}"
            )
    grouped = []
    for start, end in normalized_spans:
        observed_start = max(0, int(start))
        observed_end = min(frame_count, int(end))
        lo = observed_start + guard_frames
        hi = observed_end - 1 - guard_frames
        available = hi - lo + 1
        sample_count = min(frames_per_shot, available)
        if hi < lo or sample_count < 2:
            raise RuntimeError(
                f"RGB shot [{start}-{end}) has insufficient guarded interior "
                f"[{lo}-{hi}] for the minimum two samples"
            )
        fractions = (np.arange(sample_count, dtype=np.float64) + 1.0) / (
            sample_count + 1.0
        )
        indices = np.rint(lo + fractions * (hi - lo)).astype(int).tolist()
        if len(set(indices)) != sample_count:
            # Quantile rounding can collide only in a very short interior.
            # Evenly spanning the already-guarded support is deterministic and
            # preserves unique exact decoded indices.
            indices = np.rint(np.linspace(lo, hi, sample_count)).astype(int).tolist()
        if len(set(indices)) != sample_count:
            raise RuntimeError(f"duplicate guarded samples for RGB shot [{start}-{end})")
        grouped.append(indices)
    return grouped


def sample_indices(
    spans: list[list[int]],
    frame_count: int,
    frames_per_shot: int,
    guard_frames: int,
) -> list[list[int]]:
    """Compatibility wrapper for this benchmark's compression-4 latent spans."""
    rgb_spans = [[4 * int(start), 4 * int(end)] for start, end in spans]
    rgb_spans[-1][1] -= 3
    return sample_rgb_indices(
        rgb_spans,
        frame_count,
        frames_per_shot,
        guard_frames,
    )


def task_rgb_spans(task: dict) -> list[list[int]]:
    if "shot_spans_rgb" in task:
        return [[int(start), int(end)] for start, end in task["shot_spans_rgb"]]
    if "shot_spans" in task:
        scale = int(task.get("shot_span_scale_to_rgb", 4))
        spans = [
            [scale * int(start), scale * int(end)]
            for start, end in task["shot_spans"]
        ]
        spans[-1][1] -= int(
            task.get("shot_partition_causal_tail_rgb_frames", scale - 1)
        )
        return spans
    raise RuntimeError("inventory row lacks shot_spans_rgb or legacy shot_spans")


def preprocess_csd_image(image: np.ndarray) -> torch.Tensor:
    """Apply the released CSD/CLIP deterministic inference geometry."""
    resized = vision_functional.resize(
        Image.fromarray(image),
        224,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    cropped = vision_functional.center_crop(resized, [224, 224])
    return torch.from_numpy(np.asarray(cropped).copy()).permute(2, 0, 1)


class CsdDualHeadExtractor:
    """One frozen CSD backbone with its published content and style heads."""

    def __init__(self, weight_path: Path, batch_size: int, device: torch.device):
        import open_clip

        # CSD is initialized from OpenAI CLIP ViT-L/14, whose MLP uses
        # QuickGELU. The plain open_clip "ViT-L-14" config uses nn.GELU and is
        # therefore not architecture-equivalent even though its state keys fit.
        model = open_clip.create_model("ViT-L-14-quickgelu", pretrained=None)
        self.visual = model.visual
        self.visual.proj = None
        state = torch.load(weight_path, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
        prefix = "module.backbone."
        visual_state = self.visual.state_dict()
        backbone = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix) and key[len(prefix) :] in visual_state
        }
        missing, _ = self.visual.load_state_dict(backbone, strict=False)
        bad_missing = [key for key in missing if "proj" not in key]
        if not backbone or bad_missing:
            raise RuntimeError(f"CSD backbone mismatch missing={bad_missing[:8]}")
        for key in ("module.last_layer_style", "module.last_layer_content"):
            if key not in state:
                raise RuntimeError(f"CSD checkpoint lacks {key}")
        self.device = device
        self.style_projection = state["module.last_layer_style"].float().to(device)
        self.content_projection = state["module.last_layer_content"].float().to(device)
        self.visual = self.visual.to(device).eval()
        self.batch_size = batch_size
        del state, model

    @torch.inference_mode()
    def __call__(self, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        style_outputs, content_outputs = [], []
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=self.device)[None, :, None, None]
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=self.device)[None, :, None, None]
        for begin in range(0, len(images), self.batch_size):
            tensors = []
            for image in images[begin : begin + self.batch_size]:
                tensors.append(preprocess_csd_image(image))
            batch = torch.stack(tensors).to(self.device, non_blocking=True).float() / 255.0
            batch = (batch - mean) / std
            pooled = self.visual(batch).float()
            style_outputs.append(
                F.normalize(pooled @ self.style_projection, dim=1).cpu().numpy()
            )
            content_outputs.append(
                F.normalize(pooled @ self.content_projection, dim=1).cpu().numpy()
            )
        style = np.concatenate(style_outputs)
        content = np.concatenate(content_outputs)
        if (
            style.ndim != 2
            or content.ndim != 2
            or len(style) != len(images)
            or len(content) != len(images)
            or not np.isfinite(style).all()
            or not np.isfinite(content).all()
        ):
            raise RuntimeError(
                "CSD dual-head output must contain one finite descriptor per input RGB frame"
            )
        return style, content


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(max(1.0 - float(left @ right), 0.0))


def pair_distances(
    style: np.ndarray,
    content: np.ndarray,
    rows: list[list[int]],
    sample_indices: list[list[int]],
) -> tuple[list[dict], list[dict]]:
    flattened = [position for group in rows for position in group]
    if (
        style.ndim != 2
        or content.ndim != 2
        or style.shape[0] != content.shape[0]
        or flattened != list(range(style.shape[0]))
    ):
        raise RuntimeError(
            "feature/shot-row geometry must cover every descriptor exactly once in order"
        )
    decoded = sample_indices
    if (
        len(decoded) != len(rows)
        or any(len(frames) != len(positions) for frames, positions in zip(decoded, rows))
        or any(
            type(frame) is not int or frame < 0
            for frames in decoded
            for frame in frames
        )
        or any(len(frames) != len(set(frames)) or frames != sorted(frames) for frames in decoded)
    ):
        raise RuntimeError("decoded sample coordinates do not match descriptor rows")
    flattened_decoded = [frame for frames in decoded for frame in frames]
    if len(flattened_decoded) != len(set(flattened_decoded)):
        raise RuntimeError("decoded RGB sample coordinates overlap across shots")
    within, cross = [], []
    for shot, positions in enumerate(rows):
        for left_offset, left in enumerate(positions):
            for right_offset in range(left_offset + 1, len(positions)):
                right = positions[right_offset]
                within.append(
                    {
                        "shot": shot,
                        "left_frame": decoded[shot][left_offset],
                        "right_frame": decoded[shot][right_offset],
                        "content_distance": cosine_distance(content[left], content[right]),
                        "style_distance": cosine_distance(style[left], style[right]),
                    }
                )
    for left_shot in range(len(rows)):
        for right_shot in range(left_shot + 1, len(rows)):
            for left_offset, left in enumerate(rows[left_shot]):
                for right_offset, right in enumerate(rows[right_shot]):
                    cross.append(
                        {
                            "left_shot": left_shot,
                            "right_shot": right_shot,
                            "left_frame": decoded[left_shot][left_offset],
                            "right_frame": decoded[right_shot][right_offset],
                            "content_distance": cosine_distance(content[left], content[right]),
                            "style_distance": cosine_distance(style[left], style[right]),
                        }
                    )
    return within, cross


def completed_keys(path: Path) -> set[tuple[int, str, str]]:
    """Keys already extracted in `path`.

    TORN TAIL (2026-09-09): with --completed-from, a sibling shard is read while
    a live worker on another node is APPENDING to it. A feature row is 5.5-10.4 KB
    and is written with one write() on cephfs, which is not atomic at that size,
    so a reader can observe a record truncated mid-string. On 2026-09-09 the
    identical bug in eval_v4_paper_metrics.py killed 136 of 256 ranks at startup;
    the data was never corrupt, only momentarily incomplete.

    So an unterminated FINAL line is tolerated (a partial append its writer will
    complete) and skipped. A malformed line anywhere else, a duplicate, or a
    contract mismatch still raises: those cannot come from a partial write.
    """
    keys = set()
    if not path.is_file():
        return keys
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line_number == len(lines) and not line.endswith("\n"):
            print(
                f"[v4-features] skipping torn final line {path}:{line_number} "
                f"({len(line)} B, no newline) -- concurrent append in progress",
                flush=True,
            )
            continue
        try:
            row = json.loads(line)
            if row.get("pair_geometry_contract") != PAIR_GEOMETRY_CONTRACT:
                raise RuntimeError(
                    "resume row uses a different frame-pair geometry contract"
                )
            keys.add((int(row["case"]), row["method"], row["video_path"]))
        except Exception as error:
            raise RuntimeError(f"invalid resume JSONL {path}:{line_number}: {error}")
    return keys


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def repair_own_jsonl(path: Path) -> None:
    """Truncate a partial trailing record from THIS shard's own output file.

    See the torn-tail note in `completed_keys`. Nobody else will ever finish a
    fragment WE wrote, and appending after it would splice the next row onto the
    fragment, producing one permanently unparseable line. A killed run (node
    death, container reboot) leaves exactly this. Truncate to the last newline so
    the file is on a clean record boundary before we append.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1  # 0 when there is no newline at all
    print(
        f"[v4-features] repairing own output {path}: dropping "
        f"{len(data) - cut} B partial trailing record",
        flush=True,
    )
    with path.open("r+b") as handle:
        handle.truncate(cut)




