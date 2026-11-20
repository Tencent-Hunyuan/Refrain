#!/usr/bin/env python3
"""Extract guarded VGG-AdaIN style distances for SDC v5.

This is a v5-local GPU-capable copy of the earlier representation experiment.
It never writes into the v4 tree and preserves the supplied partition and exact
frame sampling contract.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.style_features.csd import (
    DEFAULT_INVENTORY,
    parse_csv,
    read_exact_sampled_frames,
    task_rgb_spans,
)


DEFAULT_VGG = Path("weights/evaluation/vgg16-397923af.pth")
REPRESENTATION = (
    "torchvision_vgg16_layers_3_8_15_22_channel_mean_std_l2_normalized_1920d"
)


class VggAdaINGPU:
    def __init__(self, weight_path: Path, batch_size: int, device: torch.device):
        from torchvision.models import vgg16

        model = vgg16(weights=None)
        self.features = model.features[:23].eval()
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        filtered = {
            key[len("features.") :]: value
            for key, value in state.items()
            if key.startswith("features.")
            and int(key[len("features.") :].split(".", 1)[0]) <= 22
        }
        missing, unexpected = self.features.load_state_dict(filtered, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"VGG mismatch missing={missing} unexpected={unexpected}")
        self.device = device
        self.features = self.features.to(device)
        self.batch_size = batch_size
        self.capture = {3, 8, 15, 22}
        self.mean = torch.tensor(
            (0.485, 0.456, 0.406), device=device
        )[None, :, None, None]
        self.std = torch.tensor(
            (0.229, 0.224, 0.225), device=device
        )[None, :, None, None]

    @torch.inference_mode()
    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        chunks = []
        for begin in range(0, len(images), self.batch_size):
            tensors = []
            for image in images[begin : begin + self.batch_size]:
                resized = Image.fromarray(image).resize(
                    (224, 224), Image.Resampling.BICUBIC
                )
                tensors.append(
                    torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)
                )
            x = (
                torch.stack(tensors).to(self.device, non_blocking=True).float() / 255.0
                - self.mean
            ) / self.std
            moments = []
            for index, layer in enumerate(self.features):
                x = layer(x)
                if index in self.capture:
                    stats = torch.cat(
                        (x.mean((2, 3)), x.std((2, 3), unbiased=False)), dim=1
                    )
                    moments.append(F.normalize(stats, dim=1))
            chunks.append(
                F.normalize(torch.cat(moments, dim=1), dim=1).cpu().numpy()
            )
        return np.concatenate(chunks)


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(max(1.0 - float(left @ right), 0.0))


def pair_distances(features: np.ndarray, groups: list[list[int]]) -> tuple[list, list]:
    within, cross = [], []
    for shot, positions in enumerate(groups):
        for offset, left in enumerate(positions):
            for right in positions[offset + 1 :]:
                within.append(
                    {
                        "shot": shot,
                        "style_distance": distance(features[left], features[right]),
                    }
                )
    for left_shot in range(len(groups)):
        for right_shot in range(left_shot + 1, len(groups)):
            for left in groups[left_shot]:
                for right in groups[right_shot]:
                    cross.append(
                        {
                            "left_shot": left_shot,
                            "right_shot": right_shot,
                            "style_distance": distance(features[left], features[right]),
                        }
                    )
    return within, cross


def resolve_video_path(value: str) -> Path:
    """Resolve an inventory video path; serialized provenance keeps it unchanged."""
    original = Path(value)
    if original.exists():
        return original
    raise FileNotFoundError(value)


def completed_keys_for(path: Path, representation: str) -> set[tuple[int, str, str]]:
    """Resume keys from a bank written by THIS extractor, validated by schema.

    Not csd.completed_keys: that one asserts
    `pair_geometry_contract == PAIR_GEOMETRY_CONTRACT`, a field only the CSD
    dual-head extractor writes. DINO and VGG rows have no such field, so calling it
    on a non-empty DINO/VGG bank raised

        invalid resume JSONL .../v5_dino_20260903/shard_0.jsonl:1:
        resume row uses a different frame-pair geometry contract

    The check itself is right in spirit -- a resume must refuse rows produced under
    a different feature contract, or the bank silently mixes two representations --
    so this keeps it and validates the field these rows actually carry:
    `representation`, the model/pooling identity string. A row from a different
    representation is a hard error, exactly as a mismatched geometry contract is for
    CSD.
    """
    keys: set[tuple[int, str, str]] = set()
    if not path.is_file():
        return keys
    # TORN TAIL (2026-09-09): with --completed-from, a sibling shard is read
    # while a live worker on another node is APPENDING to it. A feature row is
    # 5.5-10.4 KB and is written with one write() on cephfs, which is not atomic
    # at that size, so a reader can observe a record truncated mid-string. The
    # identical bug in eval_v4_paper_metrics.py killed 136 of 256 ranks at
    # startup on 2026-09-09; the data was never corrupt, only momentarily
    # incomplete. An unterminated FINAL line is therefore tolerated and skipped;
    # a malformed line anywhere else, or a representation mismatch, still raises.
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line_number == len(lines) and not line.endswith("\n"):
            print(
                f"[vgg-features] skipping torn final line {path}:{line_number} "
                f"({len(line)} B, no newline) -- concurrent append in progress",
                flush=True,
            )
            continue
        try:
            row = json.loads(line)
            if row.get("representation") != representation:
                raise RuntimeError(
                    "resume row uses a different feature representation: "
                    f"{row.get('representation')!r} != {representation!r}"
                )
            keys.add((int(row["case"]), row["method"], row["video_path"]))
        except Exception as error:
            raise RuntimeError(f"invalid resume JSONL {path}:{line_number}: {error}")
    return keys


def repair_own_jsonl(path: Path) -> None:
    """Truncate a partial trailing record from THIS shard's own output file.

    See the torn-tail note in completed_keys_for. Nobody else will ever finish a
    fragment WE wrote, and appending after it would splice the next row onto the
    fragment, producing one permanently unparseable line. A killed run (node
    death, container reboot) leaves exactly this.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    print(
        f"[vgg-features] repairing own output {path}: dropping "
        f"{len(data) - cut} B partial trailing record",
        flush=True,
    )
    with path.open("r+b") as handle:
        handle.truncate(cut)




