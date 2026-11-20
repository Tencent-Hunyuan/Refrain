#!/usr/bin/env python3
"""Generated-only TransNetV2 transition audit for any v4 video inventory."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image


def read_frames(
    video_path: str, target_width: int, target_height: int
) -> tuple[np.ndarray, float]:
    """Decode exact RGB frames without importing the optional OpenCV stack."""
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            resized = Image.fromarray(rgb).resize(
                (target_width, target_height), Image.Resampling.BILINEAR
            )
            frames.append(np.asarray(resized, dtype=np.uint8))
    if not frames:
        return np.empty((0,), dtype=np.uint8), fps
    return np.stack(frames), fps


def get_batches(frames: np.ndarray) -> torch.Tensor:
    reminder = (50 - len(frames) % 50) % 50
    padded = np.concatenate(
        [[frames[0]] * 25, frames, [frames[-1]] * (reminder + 25)], axis=0
    )
    return torch.cat(
        [
            torch.from_numpy(padded[index : index + 100]).unsqueeze(0)
            for index in range(0, len(padded) - 50, 50)
        ],
        dim=0,
    )


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    scenes = []
    previous = 0
    start = 0
    last_index = 0
    for index, value in enumerate(predictions):
        last_index = index
        if previous == 1 and value == 0:
            start = index
        if previous == 0 and value == 1 and index != 0:
            scenes.append([start, index])
        previous = int(value)
    if previous == 0:
        scenes.append([start, last_index])
    if not scenes:
        scenes = [[0, len(predictions) - 1]]
    return np.asarray(scenes, dtype=np.int32)


def infer(model, frames: np.ndarray, threshold: float) -> tuple[list[int], list[float]]:
    data = get_batches(frames).to(next(model.parameters()).device)
    predictions = []
    with torch.inference_mode():
        for chunk in torch.split(data, 120, dim=0):
            frame_pred, _ = model(chunk)
            predictions.append(frame_pred)
    predictions = torch.cat(predictions, 0)
    probabilities = []
    for batch_index in range(len(data)):
        values = torch.sigmoid(predictions[batch_index]).float().cpu().numpy()
        probabilities.append(values[25:75])
    # TransNet returns [batch, 50, 1].  Flatten the singleton logit channel so
    # the serialized maximum is a scalar and the scene parser receives [T].
    probabilities = np.concatenate(probabilities, 0).reshape(-1)[: len(frames)]
    scenes = predictions_to_scenes((probabilities >= threshold).astype(np.uint8))
    cuts = [int(scenes[index][1]) for index in range(len(scenes) - 1)]
    return cuts, probabilities.astype(float).tolist()


def cut_frames(probabilities: list[float], threshold: float) -> list[int]:
    values = np.asarray(probabilities, dtype=np.float32)
    scenes = predictions_to_scenes((values >= threshold).astype(np.uint8))
    return [int(scenes[index][1]) for index in range(len(scenes) - 1)]


def completed(path: Path) -> set[tuple[int, str, str]]:
    """Rows already scored in this shard file, keyed by the exact video.

    The key includes video_path: a method name can be repointed at a new sample
    tree, and then a row scored against the old tree must NOT count as having
    scored the new one. See the note in build_v5_transition_delta.py.
    """
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            keys.add((int(row["case"]), row["method"], row["video_path"]))
    return keys




