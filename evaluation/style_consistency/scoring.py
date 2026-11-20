#!/usr/bin/env python3
"""Shot-label distribution contrast core for corrected SDC v5.

For each shot pair, cosine-distance energy statistics remove the variability
already present inside each shot.  A content-energy matched within-shot split
null then asks whether the observed style-energy separation is exceptional.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from evaluation.style_features.calibration import (
    ConditionalResidualStyleBank,
)


CSD_LOG_WEIGHT = 2.0 / 3.0
FORMULA = (
    "mean_{i<j}(R_csd(E_style|E_content)^(2/3)"
    "*R_vgg(E_style|E_content)^(1/3))"
)


def symmetric_matrix(size: int, values: list[float]) -> np.ndarray:
    expected = size * (size - 1) // 2
    if len(values) != expected:
        raise RuntimeError(f"within matrix needs {expected} values, got {len(values)}")
    matrix = np.zeros((size, size), dtype=np.float64)
    cursor = 0
    for left in range(size):
        for right in range(left + 1, size):
            matrix[left, right] = matrix[right, left] = float(values[cursor])
            cursor += 1
    return matrix


def shot_matrices(row: dict, field: str) -> list[np.ndarray]:
    counts = [len(group) for group in row["sample_indices"]]
    grouped: dict[int, list[float]] = defaultdict(list)
    for pair in row["within_pairs"]:
        grouped[int(pair["shot"])].append(float(pair[field]))
    return [
        symmetric_matrix(count, grouped[index])
        for index, count in enumerate(counts)
    ]


def cross_matrices(row: dict, field: str) -> dict[tuple[int, int], np.ndarray]:
    counts = [len(group) for group in row["sample_indices"]]
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for pair in row["cross_pairs"]:
        grouped[int(pair["left_shot"]), int(pair["right_shot"])].append(
            float(pair[field])
        )
    expected = {
        (left, right)
        for left in range(len(counts))
        for right in range(left + 1, len(counts))
    }
    if set(grouped) != expected:
        raise RuntimeError("cross-shot topology mismatch")
    result = {}
    for left, right in expected:
        values = grouped[left, right]
        if len(values) != counts[left] * counts[right]:
            raise RuntimeError(f"cross-pair count mismatch for {(left, right)}")
        result[left, right] = np.asarray(values, dtype=np.float64).reshape(
            counts[left], counts[right]
        )
    return result


def energy(cross: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    value = 2.0 * float(np.mean(cross)) - float(np.mean(left)) - float(np.mean(right))
    if value < -1e-7:
        raise RuntimeError(f"unexpected negative energy: {value}")
    return max(value, 0.0)


def balanced_splits(size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if size < 2:
        raise RuntimeError("a shot requires at least two samples")
    left_size = size // 2
    result = []
    for selected in itertools.combinations(range(size), left_size):
        if size % 2 == 0 and 0 not in selected:
            continue
        left = np.asarray(selected, dtype=np.int64)
        right = np.asarray(
            [index for index in range(size) if index not in selected],
            dtype=np.int64,
        )
        result.append((left, right))
    return result


def null_examples(csd_row: dict, style_row: dict) -> list[dict]:
    if csd_row["sample_indices"] != style_row["sample_indices"]:
        raise RuntimeError("content/style exact samples differ")
    content = shot_matrices(csd_row, "content_distance")
    style = shot_matrices(style_row, "style_distance")
    result = []
    for shot, (content_matrix, style_matrix) in enumerate(zip(content, style)):
        for left, right in balanced_splits(len(content_matrix)):
            result.append(
                {
                    "shot": shot,
                    "content_distance": energy(
                        content_matrix[np.ix_(left, right)],
                        content_matrix[np.ix_(left, left)],
                        content_matrix[np.ix_(right, right)],
                    ),
                    "style_distance": energy(
                        style_matrix[np.ix_(left, right)],
                        style_matrix[np.ix_(left, left)],
                        style_matrix[np.ix_(right, right)],
                    ),
                }
            )
    return result


def query_examples(csd_row: dict, style_row: dict) -> list[dict]:
    if csd_row["sample_indices"] != style_row["sample_indices"]:
        raise RuntimeError("content/style exact samples differ")
    content_within = shot_matrices(csd_row, "content_distance")
    style_within = shot_matrices(style_row, "style_distance")
    content_cross = cross_matrices(csd_row, "content_distance")
    style_cross = cross_matrices(style_row, "style_distance")
    return [
        {
            "left_shot": left,
            "right_shot": right,
            "content_energy": energy(
                content_cross[left, right],
                content_within[left],
                content_within[right],
            ),
            "style_energy": energy(
                style_cross[left, right],
                style_within[left],
                style_within[right],
            ),
        }
        for left, right in sorted(content_cross)
    ]


def balanced_environment_examples(rows: list[tuple[dict, dict]]) -> list[dict]:
    result = []
    video_mass = 1.0 / len(rows)
    for csd_row, style_row in rows:
        examples = null_examples(csd_row, style_row)
        by_shot: dict[int, list[dict]] = defaultdict(list)
        for example in examples:
            by_shot[int(example["shot"])].append(example)
        shot_mass = video_mass / len(by_shot)
        for selected in by_shot.values():
            weight = shot_mass / len(selected)
            result.extend({**example, "weight": weight} for example in selected)
    return result


def build_environment_banks(
    csd_rows: list[dict],
    style_rows: list[dict],
    methods: list[str],
    fold_count: int = 5,
) -> dict[int, dict[str, ConditionalResidualStyleBank]]:
    style_map = {
        (int(row["case"]), row["method"], row["video_path"]): row
        for row in style_rows
    }
    result = {}
    for fold in range(fold_count):
        environments = {}
        for method in methods:
            selected = [
                (
                    row,
                    style_map[int(row["case"]), row["method"], row["video_path"]],
                )
                for row in csd_rows
                if row["method"] == method
                and int(row["case_order"]) % fold_count != fold
            ]
            if not selected:
                raise RuntimeError(f"empty calibration environment: {method}")
            environments[method] = ConditionalResidualStyleBank(
                balanced_environment_examples(selected)
            )
        result[fold] = environments
    return result


def environment_rank(
    content: float,
    style: float,
    banks: dict[str, ConditionalResidualStyleBank],
    methods: list[str],
) -> tuple[float, float]:
    ranks = np.asarray(
        [banks[method].survival(content, style)[0] for method in methods],
        dtype=np.float64,
    )
    # Every calibration branch is an equal environment. This prevents the
    # content marginal of one branch from controlling a query and exactly
    # matches the intended three-environment cap.
    return float(np.mean(ranks)), float(len(ranks))


def score_record(
    content_row: dict,
    csd_style_row: dict,
    vgg_row: dict,
    csd_banks: dict[str, ConditionalResidualStyleBank],
    vgg_banks: dict[str, ConditionalResidualStyleBank],
    methods: list[str],
) -> dict:
    csd_queries = query_examples(content_row, csd_style_row)
    vgg_queries = query_examples(content_row, vgg_row)
    if [
        (row["left_shot"], row["right_shot"]) for row in csd_queries
    ] != [
        (row["left_shot"], row["right_shot"]) for row in vgg_queries
    ]:
        raise RuntimeError("CSD/VGG query topology differs")
    pair_rows = []
    for csd_query, vgg_query in zip(csd_queries, vgg_queries):
        content = float(csd_query["content_energy"])
        csd_rank, csd_effective = environment_rank(
            content, float(csd_query["style_energy"]), csd_banks, methods
        )
        vgg_rank, vgg_effective = environment_rank(
            content, float(vgg_query["style_energy"]), vgg_banks, methods
        )
        fused = csd_rank ** CSD_LOG_WEIGHT * vgg_rank ** (1.0 - CSD_LOG_WEIGHT)
        pair_rows.append(
            {
                "left_shot": csd_query["left_shot"],
                "right_shot": csd_query["right_shot"],
                "content_energy": content,
                "csd_style_energy": float(csd_query["style_energy"]),
                "vgg_style_energy": float(vgg_query["style_energy"]),
                "csd_conditional_rank": csd_rank,
                "vgg_conditional_rank": vgg_rank,
                "csd_effective_environments": csd_effective,
                "vgg_effective_environments": vgg_effective,
                "distribution_contrast_witness": fused,
            }
        )
    witnesses = [row["distribution_contrast_witness"] for row in pair_rows]
    raw_content_by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    for pair in content_row["cross_pairs"]:
        raw_content_by_pair[
            int(pair["left_shot"]), int(pair["right_shot"])
        ].append(float(pair["content_distance"]))
    return {
        "sdc_v5": float(np.mean(witnesses)),
        "score_status": "multi_shot_scored",
        "minimum_pair_score": min(witnesses),
        "median_pair_score": float(np.median(witnesses)),
        "unordered_shot_pairs": len(pair_rows),
        "content_energy_diagnostic": float(
            np.mean([row["content_energy"] for row in pair_rows])
        ),
        "content_change_diagnostic": float(
            np.mean(
                [
                    np.quantile(values, 0.75)
                    for _, values in sorted(raw_content_by_pair.items())
                ]
            )
        ),
        "shot_pair_scores": pair_rows,
    }
