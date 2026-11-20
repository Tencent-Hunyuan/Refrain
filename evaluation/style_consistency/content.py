#!/usr/bin/env python3
"""Extract guarded DINOv2 content distances for SDC v5.

The implementation is GPU-capable and keeps exact sample indices identical to
the existing CSD/VGG feature rows. It writes only into the v5 tree.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.style_features.csd import (
    DEFAULT_INVENTORY,
    read_exact_sampled_frames,
    task_rgb_spans,
)
from evaluation.style_consistency.style import resolve_video_path


DEFAULT_DINO = Path("weights/evaluation/dinov2-small")
REPRESENTATION = "facebook_dinov2_small_cls_l2_normalized_local_snapshot"


class DinoGPU:
    def __init__(self, path: Path, batch_size: int, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(path, use_fast=False)
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.batch_size = batch_size
        self.device = device

    @torch.inference_mode()
    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        outputs = []
        for begin in range(0, len(images), self.batch_size):
            pixels = self.processor(
                images=images[begin : begin + self.batch_size],
                return_tensors="pt",
            )["pixel_values"].to(self.device, non_blocking=True)
            hidden = self.model(pixel_values=pixels).last_hidden_state.float()
            outputs.append(F.normalize(hidden[:, 0], dim=-1).cpu().numpy())
        return np.concatenate(outputs)


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
                        "content_distance": distance(features[left], features[right]),
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
                            "content_distance": distance(features[left], features[right]),
                        }
                    )
    return within, cross


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
                f"[dino-features] skipping torn final line {path}:{line_number} "
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
        f"[dino-features] repairing own output {path}: dropping "
        f"{len(data) - cut} B partial trailing record",
        flush=True,
    )
    with path.open("r+b") as handle:
        handle.truncate(cut)




