#!/usr/bin/env python3
"""Cross-fitted content-conditional residual style rank for SDC v4."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Weighted empirical inverse CDF selecting observed order statistics."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    targets = np.asarray(quantiles, dtype=np.float64) * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    indices = np.clip(indices, 0, len(sorted_values) - 1)
    return sorted_values[indices]


def balanced_calibration_pairs(rows: list[dict]) -> list[dict]:
    """Give every method, video, shot, and within-shot frame pair equal mass.

    Videos contain different shot counts and short shots can expose fewer
    guarded samples.  Weighting every raw pair equally would therefore let
    long/high-shot-count videos define the null distribution.  The hierarchy
    below makes the calibration population explicit and matches the score's
    equal-shot-pair aggregation.
    """
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    if not by_method:
        raise RuntimeError("no calibration methods")
    pairs = []
    method_mass = 1.0 / len(by_method)
    for method_rows in by_method.values():
        video_mass = method_mass / len(method_rows)
        for row in method_rows:
            by_shot: dict[int, list[dict]] = defaultdict(list)
            for pair in row["within_pairs"]:
                by_shot[int(pair["shot"])].append(pair)
            if not by_shot:
                raise RuntimeError(
                    f"calibration video has no within pairs: {row['method']} case={row['case']}"
                )
            shot_mass = video_mass / len(by_shot)
            for shot_pairs in by_shot.values():
                pair_weight = shot_mass / len(shot_pairs)
                pairs.extend(
                    {**pair, "weight": pair_weight} for pair in shot_pairs
                )
    return pairs


def load_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as error:
                    raise RuntimeError(f"invalid JSONL {path}:{line_number}: {error}")
    keys = [(row["case"], row["method"], row["video_path"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate feature rows")
    return rows


class ResidualStyleBank:
    """Empirical survival of robust content-residualized style distance.

    The nuisance slope is a bounded-influence Huber M-estimate through the
    origin, initialized by the weighted median of ``d_style / d_content``.
    Identical normalized descriptors have zero distance in both heads.  The
    standard 1.345 Huber transition retains efficiency while preventing a small
    number of corrupted within-shot pairs from controlling the slope.
    """

    def __init__(self, pairs: list[dict]):
        if len(pairs) < 32:
            raise RuntimeError(f"residual bank is too small: {len(pairs)}")
        content = np.asarray([pair["content_distance"] for pair in pairs], dtype=np.float64)
        style = np.asarray([pair["style_distance"] for pair in pairs], dtype=np.float64)
        weights = np.asarray(
            [float(pair.get("weight", 1.0)) for pair in pairs], dtype=np.float64
        )
        if np.any(weights <= 0.0) or not np.isfinite(weights).all():
            raise RuntimeError("residual bank weights must be finite and positive")
        self.effective_sample_size = float(
            np.sum(weights) ** 2 / np.sum(np.square(weights))
        )
        positive = content > np.finfo(np.float32).eps
        if int(np.sum(positive)) < 16 or float(np.sum(weights[positive])) <= 0.0:
            raise RuntimeError("too few nonzero-content pairs for robust residual slope")
        kappa = float(
            weighted_quantile(
                style[positive] / content[positive],
                weights[positive],
                np.asarray([0.5], dtype=np.float64),
            )[0]
        )
        self.huber_delta = 1.345
        self.huber_iterations = 0
        self.huber_scale = 0.0
        self.huber_converged = False
        for iteration in range(1, 101):
            residual = style - kappa * content
            center = float(
                weighted_quantile(residual, weights, np.asarray([0.5]))[0]
            )
            scale = 1.4826 * float(
                weighted_quantile(
                    np.abs(residual - center), weights, np.asarray([0.5])
                )[0]
            )
            self.huber_iterations = iteration
            self.huber_scale = scale
            if scale <= np.finfo(np.float64).eps:
                self.huber_converged = True
                break
            influence = np.minimum(
                1.0,
                self.huber_delta
                * scale
                / np.maximum(np.abs(residual), np.finfo(np.float64).eps),
            )
            denominator = float(
                np.sum(weights * influence * np.square(content))
            )
            if denominator <= 0.0:
                raise RuntimeError("Huber residual bank has zero content energy")
            updated = float(
                np.sum(weights * influence * content * style) / denominator
            )
            if abs(updated - kappa) <= 1e-12 * (1.0 + abs(kappa)):
                kappa = updated
                self.huber_converged = True
                break
            kappa = updated
        if not self.huber_converged:
            raise RuntimeError("Huber residual slope did not converge in 100 iterations")
        self.kappa = kappa
        if not np.isfinite(self.kappa) or self.kappa < 0.0:
            raise RuntimeError(f"invalid robust content slope kappa={self.kappa}")
        residual = style - self.kappa * content
        final_center = float(
            weighted_quantile(residual, weights, np.asarray([0.5]))[0]
        )
        self.huber_scale = 1.4826 * float(
            weighted_quantile(
                np.abs(residual - final_center), weights, np.asarray([0.5])
            )[0]
        )
        if self.huber_scale <= np.finfo(np.float64).eps:
            final_influence = np.ones_like(residual)
        else:
            final_influence = np.minimum(
                1.0,
                self.huber_delta
                * self.huber_scale
                / np.maximum(np.abs(residual), np.finfo(np.float64).eps),
            )
        fixed_denominator = float(
            np.sum(weights * final_influence * np.square(content))
        )
        if fixed_denominator <= 0.0:
            raise RuntimeError("final Huber fixed-point audit has zero content energy")
        fixed_update = float(
            np.sum(weights * final_influence * content * style) / fixed_denominator
        )
        self.huber_fixed_point_relative_error = abs(fixed_update - self.kappa) / (
            1.0 + abs(self.kappa)
        )
        # The final diagnostic is evaluated after the converged update rather
        # than inherited from the previous IRLS iterate.  It must satisfy the
        # same frozen convergence tolerance as the optimizer.
        if self.huber_fixed_point_relative_error > 1e-12 + 1e-15:
            raise RuntimeError(
                "Huber fixed-point audit exceeds convergence tolerance: "
                f"{self.huber_fixed_point_relative_error}"
            )
        total_weight = float(np.sum(weights))
        self.huber_clipped_weight_fraction = float(
            np.sum(weights[final_influence < 1.0]) / total_weight
        )
        self.huber_mean_influence = float(
            np.sum(weights * final_influence) / total_weight
        )
        residual_order = np.argsort(residual, kind="mergesort")
        self.sorted_residual = residual[residual_order]
        self.sorted_residual_weights = weights[residual_order]
        content_order = np.argsort(content, kind="mergesort")
        self.sorted_content = content[content_order]
        self.sorted_content_weights = weights[content_order]
        self.minimum = float(content.min())
        self.maximum = float(content.max())
        self._refresh_cumulative_weights()

    def _refresh_cumulative_weights(self) -> None:
        """Cache empirical mass prefixes for O(log N) rank queries."""
        self.residual_weight_prefix = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self.sorted_residual_weights))
        )
        self.content_weight_prefix = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self.sorted_content_weights))
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "ResidualStyleBank":
        bank = cls.__new__(cls)
        bank.minimum = float(payload["content_distance_min"])
        bank.maximum = float(payload["content_distance_max"])
        bank.kappa = float(payload["robust_content_slope_kappa"])
        bank.huber_delta = float(payload["huber_delta"])
        bank.huber_iterations = int(payload["huber_iterations"])
        bank.huber_scale = float(payload["huber_final_mad_scale"])
        bank.huber_converged = bool(payload["huber_converged"])
        bank.huber_fixed_point_relative_error = float(
            payload["huber_fixed_point_relative_error"]
        )
        bank.huber_clipped_weight_fraction = float(
            payload["huber_clipped_weight_fraction"]
        )
        bank.huber_mean_influence = float(payload["huber_mean_influence"])
        bank.effective_sample_size = float(
            payload.get("effective_sample_size", payload["within_pairs"])
        )
        bank.sorted_residual = np.asarray(
            payload["sorted_style_residuals"], dtype=np.float64
        )
        bank.sorted_residual_weights = np.asarray(
            payload["sorted_style_residual_weights"], dtype=np.float64
        )
        bank.sorted_content = np.asarray(
            payload["sorted_content_distances"], dtype=np.float64
        )
        bank.sorted_content_weights = np.asarray(
            payload["sorted_content_weights"], dtype=np.float64
        )
        reported_pairs = int(payload["within_pairs"])
        reported_total_weight = float(
            payload.get("total_weight", np.sum(bank.sorted_residual_weights))
        )
        residual_total_weight = float(np.sum(bank.sorted_residual_weights))
        content_total_weight = float(np.sum(bank.sorted_content_weights))
        if (
            len(bank.sorted_residual) == 0
            or len(bank.sorted_residual) != len(bank.sorted_residual_weights)
            or len(bank.sorted_residual) != reported_pairs
            or not np.isfinite(bank.sorted_residual).all()
            or np.any(bank.sorted_residual[:-1] > bank.sorted_residual[1:])
            or np.any(bank.sorted_residual_weights <= 0.0)
            or not np.isfinite(bank.sorted_residual_weights).all()
            or not np.isfinite(residual_total_weight)
            or residual_total_weight <= 0.0
            or not np.isclose(
                residual_total_weight, reported_total_weight, rtol=0.0, atol=1e-12
            )
            or not np.isfinite(bank.kappa)
            or bank.kappa < 0.0
            or bank.huber_delta != 1.345
            or not 1 <= bank.huber_iterations <= 100
            or not np.isfinite(bank.huber_scale)
            or bank.huber_scale < 0.0
            or not bank.huber_converged
            or not 0.0 <= bank.huber_fixed_point_relative_error <= 1e-12 + 1e-15
            or not 0.0 <= bank.huber_clipped_weight_fraction <= 1.0
            or not 0.0 <= bank.huber_mean_influence <= 1.0
        ):
            raise RuntimeError("frozen residual distribution is invalid")
        if (
            len(bank.sorted_content) != len(bank.sorted_content_weights)
            or len(bank.sorted_content) == 0
            or len(bank.sorted_content) != reported_pairs
            or not np.isfinite(bank.sorted_content).all()
            or np.any(bank.sorted_content[:-1] > bank.sorted_content[1:])
            or np.any(bank.sorted_content_weights <= 0.0)
            or not np.isfinite(bank.sorted_content_weights).all()
            or not np.isfinite(content_total_weight)
            or content_total_weight <= 0.0
            or not np.isclose(
                content_total_weight, residual_total_weight, rtol=0.0, atol=1e-12
            )
            or not np.isfinite(bank.minimum)
            or not np.isfinite(bank.maximum)
            or bank.minimum > bank.maximum
            or not np.isclose(
                bank.minimum, float(bank.sorted_content[0]), rtol=0.0, atol=1e-12
            )
            or not np.isclose(
                bank.maximum, float(bank.sorted_content[-1]), rtol=0.0, atol=1e-12
            )
            or not np.isfinite(bank.effective_sample_size)
            or not 0.0 < bank.effective_sample_size <= reported_pairs + 1e-12
        ):
            raise RuntimeError("frozen bank content support is invalid")
        bank._refresh_cumulative_weights()
        return bank

    def to_dict(self) -> dict:
        return {
            "content_distance_min": self.minimum,
            "content_distance_max": self.maximum,
            "robust_content_slope_kappa": self.kappa,
            "slope_estimator": "through_origin_huber_irls_delta_1.345_initialized_weighted_median_ratio",
            "huber_delta": self.huber_delta,
            "huber_iterations": self.huber_iterations,
            "huber_final_mad_scale": self.huber_scale,
            "huber_converged": self.huber_converged,
            "huber_fixed_point_relative_error": self.huber_fixed_point_relative_error,
            "huber_clipped_weight_fraction": self.huber_clipped_weight_fraction,
            "huber_mean_influence": self.huber_mean_influence,
            "sorted_style_residuals": self.sorted_residual.tolist(),
            "sorted_style_residual_weights": self.sorted_residual_weights.tolist(),
            "sorted_content_distances": self.sorted_content.tolist(),
            "sorted_content_weights": self.sorted_content_weights.tolist(),
            "within_pairs": len(self.sorted_residual),
            "effective_sample_size": self.effective_sample_size,
            "total_weight": float(np.sum(self.sorted_residual_weights)),
        }

    def content_percentile(self, content_distance: float) -> float:
        """Weighted empirical mid-rank of a query content distance."""
        value = float(content_distance)
        left = int(np.searchsorted(self.sorted_content, value, side="left"))
        right = int(np.searchsorted(self.sorted_content, value, side="right"))
        below = float(self.content_weight_prefix[left])
        equal = float(self.content_weight_prefix[right] - self.content_weight_prefix[left])
        return (below + 0.5 * equal) / float(self.content_weight_prefix[-1])

    def survival(self, content_distance: float, style_distance: float) -> tuple[float, bool]:
        residual = float(style_distance) - self.kappa * float(content_distance)
        values = self.sorted_residual
        left = int(np.searchsorted(values, residual, side="left"))
        right = int(np.searchsorted(values, residual, side="right"))
        total_weight = float(self.residual_weight_prefix[-1])
        greater_weight = float(total_weight - self.residual_weight_prefix[right])
        equal_weight = float(
            self.residual_weight_prefix[right] - self.residual_weight_prefix[left]
        )
        # Weighted empirical mid-rank survival. No pseudocount, temperature, or
        # per-video floor changes the rank.
        survival = (greater_weight + 0.5 * equal_weight) / total_weight
        supported = self.minimum <= content_distance <= self.maximum
        return float(survival), supported


class ConditionalResidualStyleBank:
    """Cross-dataset conditional conformal rank for cross-shot style drift.

    A robust global Huber fit first removes the first-order content nuisance,
    ``R = D_style - kappa * D_content``.  A single global residual CDF is not
    enough when the residual variance or shape changes with semantic distance.
    We therefore partition the *calibration* content distribution into
    equal-mass neighborhoods, form a weighted residual survival CDF in every
    neighborhood, and linearly interpolate adjacent CDFs at query time.

    The resolution is deterministic and scale-adaptive: ``round(sqrt(N_eff))``
    neighborhoods, clamped to [4, 256].  Thus both the number of neighborhoods
    and the effective observations per neighborhood grow with calibration
    coverage.  Query videos never fit or widen their own tolerance.
    """

    MINIMUM_BINS = 4
    MAXIMUM_BINS = 256
    RESOLUTION_RULE = "round_sqrt_hierarchical_effective_sample_size_clamped_4_256"
    INTERPOLATION_RULE = "linear_between_weighted_content_percentile_bin_centers"

    def __init__(self, pairs: list[dict]):
        global_bank = ResidualStyleBank(pairs)
        for name in (
            "kappa",
            "huber_delta",
            "huber_iterations",
            "huber_scale",
            "huber_converged",
            "huber_fixed_point_relative_error",
            "huber_clipped_weight_fraction",
            "huber_mean_influence",
            "effective_sample_size",
            "minimum",
            "maximum",
        ):
            setattr(self, name, getattr(global_bank, name))

        content = np.asarray(
            [pair["content_distance"] for pair in pairs], dtype=np.float64
        )
        style = np.asarray(
            [pair["style_distance"] for pair in pairs], dtype=np.float64
        )
        weights = np.asarray(
            [float(pair.get("weight", 1.0)) for pair in pairs], dtype=np.float64
        )
        residual = style - self.kappa * content
        self.requested_content_bins = max(
            self.MINIMUM_BINS,
            min(self.MAXIMUM_BINS, int(round(np.sqrt(self.effective_sample_size)))),
        )
        edges = np.unique(
            weighted_quantile(
                content,
                weights,
                np.linspace(0.0, 1.0, self.requested_content_bins + 1),
            )
        )
        if len(edges) < 3:
            raise RuntimeError("conditional residual bank has insufficient content support")
        assignment = np.searchsorted(edges[1:-1], content, side="right")
        self.content_edges = edges
        self.sorted_conditional_residuals: list[np.ndarray] = []
        self.sorted_conditional_residual_weights: list[np.ndarray] = []

        content_order = np.argsort(content, kind="mergesort")
        self.sorted_content = content[content_order]
        self.sorted_content_weights = weights[content_order]
        self._refresh_content_prefix()
        centers = []
        for index in range(len(edges) - 1):
            selected = assignment == index
            if not np.any(selected):
                raise RuntimeError("conditional residual bank contains an empty bin")
            order = np.argsort(residual[selected], kind="mergesort")
            self.sorted_conditional_residuals.append(residual[selected][order])
            self.sorted_conditional_residual_weights.append(weights[selected][order])
            # A weighted-median content location mapped through the global
            # empirical CDF remains valid even when quantile edges collapse on
            # tied descriptors.
            center_content = float(
                weighted_quantile(
                    content[selected], weights[selected], np.asarray([0.5])
                )[0]
            )
            centers.append(self.content_percentile(center_content))
        self.content_bin_percentile_centers = np.asarray(centers, dtype=np.float64)
        if np.any(np.diff(self.content_bin_percentile_centers) <= 0.0):
            raise RuntimeError("conditional residual content-bin centers are not increasing")
        self._refresh_conditional_prefixes()

    def _refresh_content_prefix(self) -> None:
        self.content_weight_prefix = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self.sorted_content_weights))
        )

    def _refresh_conditional_prefixes(self) -> None:
        self.conditional_weight_prefixes = [
            np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(weights)))
            for weights in self.sorted_conditional_residual_weights
        ]

    @property
    def content_bins(self) -> int:
        return len(self.sorted_conditional_residuals)

    def content_percentile(self, content_distance: float) -> float:
        value = float(content_distance)
        left = int(np.searchsorted(self.sorted_content, value, side="left"))
        right = int(np.searchsorted(self.sorted_content, value, side="right"))
        below = float(self.content_weight_prefix[left])
        equal = float(self.content_weight_prefix[right] - self.content_weight_prefix[left])
        return (below + 0.5 * equal) / float(self.content_weight_prefix[-1])

    def _bin_survival(self, index: int, residual: float) -> float:
        values = self.sorted_conditional_residuals[index]
        prefix = self.conditional_weight_prefixes[index]
        left = int(np.searchsorted(values, residual, side="left"))
        right = int(np.searchsorted(values, residual, side="right"))
        total = float(prefix[-1])
        greater = float(total - prefix[right])
        equal = float(prefix[right] - prefix[left])
        return (greater + 0.5 * equal) / total

    def conditional_center_supported(self, content_distance: float) -> bool:
        percentile = self.content_percentile(content_distance)
        return bool(
            self.content_bin_percentile_centers[0]
            <= percentile
            <= self.content_bin_percentile_centers[-1]
        )

    def survival(self, content_distance: float, style_distance: float) -> tuple[float, bool]:
        residual = float(style_distance) - self.kappa * float(content_distance)
        percentile = self.content_percentile(content_distance)
        centers = self.content_bin_percentile_centers
        right = int(np.searchsorted(centers, percentile, side="right"))
        if right == 0:
            value = self._bin_survival(0, residual)
        elif right >= len(centers):
            value = self._bin_survival(len(centers) - 1, residual)
        else:
            left = right - 1
            denominator = float(centers[right] - centers[left])
            fraction = (percentile - float(centers[left])) / denominator
            value = (1.0 - fraction) * self._bin_survival(
                left, residual
            ) + fraction * self._bin_survival(right, residual)
        return float(value), self.minimum <= content_distance <= self.maximum

    def to_dict(self) -> dict:
        return {
            "content_distance_min": self.minimum,
            "content_distance_max": self.maximum,
            "robust_content_slope_kappa": self.kappa,
            "slope_estimator": "through_origin_huber_irls_delta_1.345_initialized_weighted_median_ratio",
            "huber_delta": self.huber_delta,
            "huber_iterations": self.huber_iterations,
            "huber_final_mad_scale": self.huber_scale,
            "huber_converged": self.huber_converged,
            "huber_fixed_point_relative_error": self.huber_fixed_point_relative_error,
            "huber_clipped_weight_fraction": self.huber_clipped_weight_fraction,
            "huber_mean_influence": self.huber_mean_influence,
            "conditional_resolution_rule": self.RESOLUTION_RULE,
            "conditional_interpolation_rule": self.INTERPOLATION_RULE,
            "requested_content_bins": self.requested_content_bins,
            "content_bins": self.content_bins,
            "content_bin_edges": self.content_edges.tolist(),
            "content_bin_percentile_centers": self.content_bin_percentile_centers.tolist(),
            "sorted_conditional_style_residuals": [
                values.tolist() for values in self.sorted_conditional_residuals
            ],
            "sorted_conditional_style_residual_weights": [
                values.tolist() for values in self.sorted_conditional_residual_weights
            ],
            "sorted_content_distances": self.sorted_content.tolist(),
            "sorted_content_weights": self.sorted_content_weights.tolist(),
            "within_pairs": int(sum(len(values) for values in self.sorted_conditional_residuals)),
            "effective_sample_size": self.effective_sample_size,
            "total_weight": float(np.sum(self.sorted_content_weights)),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ConditionalResidualStyleBank":
        bank = cls.__new__(cls)
        bank.minimum = float(payload["content_distance_min"])
        bank.maximum = float(payload["content_distance_max"])
        bank.kappa = float(payload["robust_content_slope_kappa"])
        bank.huber_delta = float(payload["huber_delta"])
        bank.huber_iterations = int(payload["huber_iterations"])
        bank.huber_scale = float(payload["huber_final_mad_scale"])
        bank.huber_converged = bool(payload["huber_converged"])
        bank.huber_fixed_point_relative_error = float(
            payload["huber_fixed_point_relative_error"]
        )
        bank.huber_clipped_weight_fraction = float(
            payload["huber_clipped_weight_fraction"]
        )
        bank.huber_mean_influence = float(payload["huber_mean_influence"])
        bank.effective_sample_size = float(payload["effective_sample_size"])
        bank.requested_content_bins = int(payload["requested_content_bins"])
        bank.content_edges = np.asarray(payload["content_bin_edges"], dtype=np.float64)
        bank.content_bin_percentile_centers = np.asarray(
            payload["content_bin_percentile_centers"], dtype=np.float64
        )
        bank.sorted_conditional_residuals = [
            np.asarray(values, dtype=np.float64)
            for values in payload["sorted_conditional_style_residuals"]
        ]
        bank.sorted_conditional_residual_weights = [
            np.asarray(values, dtype=np.float64)
            for values in payload["sorted_conditional_style_residual_weights"]
        ]
        bank.sorted_content = np.asarray(
            payload["sorted_content_distances"], dtype=np.float64
        )
        bank.sorted_content_weights = np.asarray(
            payload["sorted_content_weights"], dtype=np.float64
        )
        reported_pairs = int(payload["within_pairs"])
        actual_pairs = sum(len(values) for values in bank.sorted_conditional_residuals)
        total_weight = float(np.sum(bank.sorted_content_weights))
        conditional_weight = float(
            sum(np.sum(values) for values in bank.sorted_conditional_residual_weights)
        )
        expected_requested = max(
            cls.MINIMUM_BINS,
            min(cls.MAXIMUM_BINS, int(round(np.sqrt(bank.effective_sample_size)))),
        )
        valid_arrays = (
            len(bank.sorted_content) == reported_pairs == actual_pairs
            and len(bank.sorted_content_weights) == reported_pairs
            and len(bank.content_edges) == len(bank.sorted_conditional_residuals) + 1
            and len(bank.content_bin_percentile_centers)
            == len(bank.sorted_conditional_residuals)
            and len(bank.sorted_conditional_residuals)
            == len(bank.sorted_conditional_residual_weights)
            and all(len(values) > 0 for values in bank.sorted_conditional_residuals)
            and all(
                len(values) == len(weights)
                for values, weights in zip(
                    bank.sorted_conditional_residuals,
                    bank.sorted_conditional_residual_weights,
                )
            )
            and np.isfinite(bank.sorted_content).all()
            and np.isfinite(bank.sorted_content_weights).all()
            and np.all(bank.sorted_content_weights > 0.0)
            and np.all(bank.sorted_content[:-1] <= bank.sorted_content[1:])
            and np.isfinite(bank.content_edges).all()
            and np.all(bank.content_edges[:-1] < bank.content_edges[1:])
            and np.isfinite(bank.content_bin_percentile_centers).all()
            and np.all(
                bank.content_bin_percentile_centers[:-1]
                < bank.content_bin_percentile_centers[1:]
            )
            and all(np.isfinite(values).all() for values in bank.sorted_conditional_residuals)
            and all(
                np.all(values[:-1] <= values[1:])
                for values in bank.sorted_conditional_residuals
            )
            and all(
                np.isfinite(weights).all() and np.all(weights > 0.0)
                for weights in bank.sorted_conditional_residual_weights
            )
        )
        valid_metadata = (
            payload.get("conditional_resolution_rule") == cls.RESOLUTION_RULE
            and payload.get("conditional_interpolation_rule")
            == cls.INTERPOLATION_RULE
            and bank.requested_content_bins == expected_requested
            and int(payload["content_bins"]) == len(bank.sorted_conditional_residuals)
            and np.isfinite(bank.kappa)
            and bank.kappa >= 0.0
            and bank.huber_delta == 1.345
            and 1 <= bank.huber_iterations <= 100
            and np.isfinite(bank.huber_scale)
            and bank.huber_scale >= 0.0
            and bank.huber_converged
            and 0.0 <= bank.huber_fixed_point_relative_error <= 1e-12 + 1e-15
            and 0.0 <= bank.huber_clipped_weight_fraction <= 1.0
            and 0.0 <= bank.huber_mean_influence <= 1.0
            and 0.0 < bank.effective_sample_size <= reported_pairs + 1e-12
            and np.isfinite(total_weight)
            and np.isclose(total_weight, float(payload["total_weight"]), atol=1e-12)
            and np.isclose(total_weight, conditional_weight, atol=1e-12)
            and bank.minimum == float(bank.sorted_content[0])
            and bank.maximum == float(bank.sorted_content[-1])
        )
        if not valid_arrays or not valid_metadata:
            raise RuntimeError("frozen conditional residual distribution is invalid")
        bank._refresh_content_prefix()
        bank._refresh_conditional_prefixes()
        return bank


def partition_coverage_diagnostic(row: dict) -> dict:
    """Report, but never score, unavailable or unassigned partition frames."""
    decoded = int(row.get("decoded_frames", -1))
    spans = row.get("shot_spans_rgb")
    if decoded <= 0 or not spans:
        return {}
    normalized = [(int(start), int(end)) for start, end in spans]
    declared = sum(end - start for start, end in normalized)
    if declared <= 0:
        raise RuntimeError("RGB shot partition has no positive declared support")
    observed = sum(
        max(0, min(end, decoded) - min(max(start, 0), decoded))
        for start, end in normalized
    )
    if observed > decoded:
        raise RuntimeError("overlapping RGB shot intervals exceed decoded support")
    return {
        "declared_partition_frames": declared,
        "observed_partition_frames": observed,
        "partition_coverage_fraction": observed / declared,
        "declared_partition_frames_unavailable": declared - observed,
        "decoded_frames_outside_partition": decoded - observed,
        "partition_fully_observed": observed == declared,
    }


def score_row(row: dict, bank, fold_count: int = 5) -> dict:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    grouped_support: dict[tuple[int, int], list[bool]] = defaultdict(list)
    grouped_center_support: dict[tuple[int, int], list[bool]] = defaultdict(list)
    supported = 0
    center_supported = 0
    content_percentiles = []
    for pair in row["cross_pairs"]:
        value, in_support = bank.survival(
            float(pair["content_distance"]), float(pair["style_distance"])
        )
        key = (int(pair["left_shot"]), int(pair["right_shot"]))
        grouped[key].append(value)
        grouped_support[key].append(in_support)
        in_center_support = (
            bank.conditional_center_supported(float(pair["content_distance"]))
            if hasattr(bank, "conditional_center_supported")
            else in_support
        )
        grouped_center_support[key].append(in_center_support)
        content_percentiles.append(
            bank.content_percentile(float(pair["content_distance"]))
        )
        supported += int(in_support)
        center_supported += int(in_center_support)
    shot_pair_scores = [
        {
            "left_shot": key[0],
            "right_shot": key[1],
            "conditional_style_rank": float(np.mean(values)),
            "frame_rank_std": float(np.std(values)),
            "calibration_support_fraction": float(np.mean(grouped_support[key])),
            "conditional_center_support_fraction": float(
                np.mean(grouped_center_support[key])
            ),
            "frame_pairs": len(values),
        }
        for key, values in sorted(grouped.items())
    ]
    shot_pair_values = [item["conditional_style_rank"] for item in shot_pair_scores]
    score = float(np.mean(shot_pair_values))
    direct = float(
        np.mean(
            [
                np.mean(
                    [
                        1.0 - pair["style_distance"]
                        for pair in row["cross_pairs"]
                        if int(pair["left_shot"]) == key[0]
                        and int(pair["right_shot"]) == key[1]
                    ]
                )
                for key in sorted(grouped)
            ]
        )
    )
    result = {
        "case": int(row["case"]),
        "case_order": int(row["case_order"]),
        "fold": int(row["case_order"]) % fold_count,
        "method": row["method"],
        "video_path": row["video_path"],
        "decoded_frames": int(row.get("decoded_frames", -1)),
        "sample_count_per_shot": [len(group) for group in row.get("sample_indices", [])],
        "conditional_style_consistency": score,
        "shot_pair_rank_std": float(np.std(shot_pair_values)),
        "minimum_shot_pair_rank": float(np.min(shot_pair_values)),
        "maximum_shot_pair_rank": float(np.max(shot_pair_values)),
        "direct_csd_ablation": direct,
        "calibration_support_fraction": supported / len(row["cross_pairs"]),
        "conditional_center_support_fraction": center_supported
        / len(row["cross_pairs"]),
        "cross_content_percentile_mean": float(np.mean(content_percentiles)),
        "cross_content_percentile_median": float(np.median(content_percentiles)),
        "cross_content_percentile_p90": float(np.quantile(content_percentiles, 0.9)),
        "cross_content_fraction_above_95th": float(
            np.mean(np.asarray(content_percentiles) > 0.95)
        ),
        "cross_content_fraction_above_99th": float(
            np.mean(np.asarray(content_percentiles) > 0.99)
        ),
        "shot_pair_scores": shot_pair_scores,
        **partition_coverage_diagnostic(row),
    }
    # Preserve a non-numeric provenance contract for deterministic synthetic
    # renders. It never enters the metric, but prevents resumable attack shards
    # from being summarized after mixing renderer implementations.
    if "attack_renderer_contract" in row:
        result["attack_renderer_contract"] = row["attack_renderer_contract"]
    return result


def heldout_null_diagnostic(row: dict, bank) -> dict:
    """Evaluate held-out within-shot ranks without feeding them into the score."""
    values = []
    supported = 0
    for pair in row["within_pairs"]:
        value, in_support = bank.survival(
            float(pair["content_distance"]), float(pair["style_distance"])
        )
        values.append(value)
        supported += int(in_support)
    if not values:
        raise RuntimeError(
            f"held-out null diagnostic has no within pairs: {row['method']} case={row['case']}"
        )
    histogram = np.zeros(10, dtype=np.float64)
    for value in values:
        histogram[min(int(value * 10.0), 9)] += 1.0
    histogram /= len(values)
    return {
        "heldout_within_rank_mean": float(np.mean(values)),
        "heldout_within_rank_median": float(np.median(values)),
        "heldout_within_support_fraction": supported / len(values),
        "heldout_within_pairs": len(values),
        "heldout_within_rank_decile_fractions": histogram.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bank-output",
        type=Path,
        help="write an all-calibration frozen bank for scoring future datasets",
    )
    parser.add_argument(
        "--bank-input",
        type=Path,
        help="score every row with an already frozen bank instead of cross-fitting",
    )
    parser.add_argument(
        "--calibration-methods",
        default="",
        help="optional comma-separated fixed calibration subset",
    )
    args = parser.parse_args()
    if args.fold_count < 2:
        raise ValueError("--fold-count must be at least 2")
    rows = load_jsonl(args.features)
    sampling_contracts = {
        (int(row["frames_per_shot"]), int(row["guard_frames"])) for row in rows
    }
    if len(sampling_contracts) != 1:
        raise RuntimeError(f"feature rows mix sampling contracts: {sampling_contracts}")
    frames_per_shot, guard_frames = next(iter(sampling_contracts))
    representation_contracts = {
        (row.get("representation"), row.get("image_preprocessing")) for row in rows
    }
    if len(representation_contracts) != 1:
        raise RuntimeError(
            f"feature rows mix representation contracts: {representation_contracts}"
        )
    representation, image_preprocessing = next(iter(representation_contracts))
    if not representation or not image_preprocessing:
        raise RuntimeError("feature rows lack a frozen representation/preprocessing contract")
    feature_keys = {
        (int(row["case"]), row["method"], row["video_path"]) for row in rows
    }
    coverage = None
    if args.inventory is not None:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        expected_keys = {
            (int(row["case"]), row["method"], row["video_path"])
            for row in inventory["rows"]
        }
        missing = sorted(expected_keys - feature_keys)
        extra = sorted(feature_keys - expected_keys)
        coverage = {
            "inventory_videos": len(expected_keys),
            "feature_rows": len(feature_keys),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing": [
                {"case": case, "method": method, "video_path": path}
                for case, method, path in missing
            ],
            "extra": [
                {"case": case, "method": method, "video_path": path}
                for case, method, path in extra
            ],
        }
    if args.require_complete and coverage is None:
        raise ValueError("--require-complete requires --inventory")
    if args.require_complete and (
        coverage["missing_count"] or coverage["extra_count"]
    ):
        # Refuse before fitting a bank or writing any score artifact.  An
        # incomplete-but-valid-looking JSON must never survive a failed
        # exhaustive run.
        raise RuntimeError(
            "feature coverage is incomplete: "
            f"missing={coverage['missing_count']} extra={coverage['extra_count']}"
        )
    calibration_methods = {
        value.strip() for value in args.calibration_methods.split(",") if value.strip()
    }
    available_methods = {row["method"] for row in rows}
    missing_calibration_methods = sorted(calibration_methods - available_methods)
    if missing_calibration_methods:
        raise RuntimeError(
            f"requested calibration methods are absent: {missing_calibration_methods}"
        )
    if args.bank_input is not None and args.bank_output is not None:
        raise ValueError("--bank-input and --bank-output are mutually exclusive")
    calibration_rows = [
        row for row in rows if not calibration_methods or row["method"] in calibration_methods
    ]
    banks = {}
    calibration_counts = {}
    if args.bank_input is not None:
        if calibration_methods:
            raise ValueError("--calibration-methods cannot be combined with --bank-input")
        frozen_payload = json.loads(args.bank_input.read_text(encoding="utf-8"))
        if int(frozen_payload.get("fold_count", -1)) != args.fold_count:
            raise RuntimeError(
                "requested fold metadata does not match frozen bank: "
                f"requested={args.fold_count} bank={frozen_payload.get('fold_count')}"
            )
        expected_sampling = frozen_payload.get("sampling_contract", {})
        if (
            frozen_payload.get("representation") != representation
            or
            int(expected_sampling.get("maximum_frames_per_shot", -1))
            != frames_per_shot
            or int(expected_sampling.get("minimum_frames_per_shot", -1)) != 2
            or int(expected_sampling.get("guard_rgb_frames_each_side", -1))
            != guard_frames
            or expected_sampling.get("image_preprocessing") != image_preprocessing
        ):
            raise RuntimeError(
                "feature sampling contract does not match frozen bank: "
                f"features={(representation, frames_per_shot, guard_frames, image_preprocessing)} "
                f"bank={(frozen_payload.get('representation'), expected_sampling)}"
            )
        reported_calibration_methods = frozen_payload.get(
            "calibration_methods", "frozen_bank_unspecified"
        )
        frozen_bank = ConditionalResidualStyleBank.from_dict(frozen_payload["bank"])
        banks = {query_fold: frozen_bank for query_fold in range(args.fold_count)}
        calibration_counts = {
            "frozen_bank": {
                "path": str(args.bank_input.resolve()),
                "within_pairs": frozen_payload["bank"]["within_pairs"],
                "robust_content_slope_kappa": frozen_bank.kappa,
                "huber_iterations": frozen_bank.huber_iterations,
                "content_bins": frozen_bank.content_bins,
            }
        }
    else:
        reported_calibration_methods = sorted(
            calibration_methods or {row["method"] for row in calibration_rows}
        )
        for query_fold in range(args.fold_count):
            fold_rows = [
                row
                for row in calibration_rows
                if int(row["case_order"]) % args.fold_count != query_fold
            ]
            pairs = balanced_calibration_pairs(fold_rows)
            banks[query_fold] = ConditionalResidualStyleBank(pairs)
            calibration_counts[str(query_fold)] = {
                "excluded_query_fold": query_fold,
                "rows": sum(
                    int(row["case_order"]) % args.fold_count != query_fold
                    for row in calibration_rows
                ),
                "within_pairs": len(pairs),
                "content_distance_min": banks[query_fold].minimum,
                "content_distance_max": banks[query_fold].maximum,
                "robust_content_slope_kappa": banks[query_fold].kappa,
                "huber_iterations": banks[query_fold].huber_iterations,
                "effective_sample_size": banks[query_fold].effective_sample_size,
                "content_bins": banks[query_fold].content_bins,
            }

        if args.bank_output is not None:
            deployment_pairs = balanced_calibration_pairs(calibration_rows)
            deployment_bank = ConditionalResidualStyleBank(deployment_pairs)
            bank_payload = {
                "status": "frozen_cross_dataset_deployment_bank",
                "representation": representation,
                "statistic": "content_conditional_empirical_survival_of_robust_style_residual",
                "weighting": "equal_method_then_equal_video_then_equal_shot_then_equal_within_pair",
                "case_split": f"original_dataset_order_mod_{args.fold_count}",
                "fold_count": args.fold_count,
                "content_residual_rule": "d_style-kappa*d_content",
                "kappa_rule": "through_origin_huber_irls_delta_1.345_initialized_weighted_median_ratio",
                "style_rank_rule": "interpolated_content_conditional_weighted_empirical_midranks",
                "conditional_resolution_rule": ConditionalResidualStyleBank.RESOLUTION_RULE,
                "conditional_interpolation_rule": ConditionalResidualStyleBank.INTERPOLATION_RULE,
                "sampling_contract": {
                    "maximum_frames_per_shot": frames_per_shot,
                    "minimum_frames_per_shot": 2,
                    "guard_rgb_frames_each_side": guard_frames,
                    "positions": "evenly_spaced_guarded_observed_interior_frames",
                    "image_preprocessing": image_preprocessing,
                },
                "calibration_methods": sorted(
                    calibration_methods or {row["method"] for row in calibration_rows}
                ),
                "calibration_video_counts": {
                    method: sum(row["method"] == method for row in calibration_rows)
                    for method in sorted({row["method"] for row in calibration_rows})
                },
                "calibration_rows": len(calibration_rows),
                "bank": deployment_bank.to_dict(),
            }
            args.bank_output.parent.mkdir(parents=True, exist_ok=True)
            bank_temporary = args.bank_output.with_suffix(args.bank_output.suffix + ".tmp")
            bank_temporary.write_text(json.dumps(bank_payload), encoding="utf-8")
            bank_temporary.replace(args.bank_output)

    scored = []
    for row in rows:
        bank = banks[int(row["case_order"]) % args.fold_count]
        scored.append(
            {
                **score_row(row, bank, args.fold_count),
                **heldout_null_diagnostic(row, bank),
            }
        )
    methods = sorted({row["method"] for row in scored})
    summary = []
    for method in methods:
        values = [
            row["conditional_style_consistency"]
            for row in scored
            if row["method"] == method
        ]
        support = [
            row["calibration_support_fraction"]
            for row in scored
            if row["method"] == method
        ]
        center_support = [
            row["conditional_center_support_fraction"]
            for row in scored
            if row["method"] == method
        ]
        direct = [
            row["direct_csd_ablation"] for row in scored if row["method"] == method
        ]
        null_ranks = [
            row["heldout_within_rank_mean"]
            for row in scored
            if row["method"] == method
        ]
        null_histograms = [
            row["heldout_within_rank_decile_fractions"]
            for row in scored
            if row["method"] == method
        ]
        pair_dispersion = [
            row["shot_pair_rank_std"] for row in scored if row["method"] == method
        ]
        worst_pair = [
            row["minimum_shot_pair_rank"] for row in scored if row["method"] == method
        ]
        content_percentile = [
            row["cross_content_percentile_median"]
            for row in scored
            if row["method"] == method
        ]
        extreme_content = [
            row["cross_content_fraction_above_99th"]
            for row in scored
            if row["method"] == method
        ]
        summary.append(
            {
                "method": method,
                "videos": len(values),
                "mean": float(np.mean(values)),
                "median": float(statistics.median(values)),
                "direct_csd_mean": float(np.mean(direct)),
                "direct_csd_median": float(statistics.median(direct)),
                "mean_calibration_support": float(np.mean(support)),
                "mean_conditional_center_support": float(np.mean(center_support)),
                "mean_heldout_within_rank": float(np.mean(null_ranks)),
                "mean_heldout_within_rank_decile_fractions": np.mean(
                    np.asarray(null_histograms, dtype=np.float64), axis=0
                ).tolist(),
                "mean_shot_pair_rank_std": float(np.mean(pair_dispersion)),
                "mean_minimum_shot_pair_rank": float(np.mean(worst_pair)),
                "mean_cross_content_percentile_median": float(
                    np.mean(content_percentile)
                ),
                "mean_cross_content_fraction_above_99th": float(
                    np.mean(extreme_content)
                ),
            }
        )
    result = {
        "status": (
            "frozen_bank_conditional_residual_style_rank"
            if args.bank_input is not None
            else "active_crossfit_conditional_residual_style_rank"
        ),
        "input_contract": ["generated_rgb_video", "shot_partition"],
        "formula": (
            "mean_shot_pairs(mean_frame_pairs(P_weighted(R_within>=R_cross|Q_content))),"
            "R=D_style-kappa*D_content"
        ),
        "direct_ablation_formula": (
            "mean_shot_pairs(mean_frame_pairs(1-D_style_cross))"
        ),
        "single_backbone": representation,
        "sampling_contract": {
            "maximum_frames_per_shot": frames_per_shot,
            "minimum_frames_per_shot": 2,
            "guard_rgb_frames_each_side": guard_frames,
            "positions": "evenly_spaced_guarded_observed_interior_frames",
            "image_preprocessing": image_preprocessing,
        },
        "calibration": {
            "case_split": (
                "frozen_bank_applied_to_all_rows"
                if args.bank_input is not None
                else f"original_dataset_order_mod_{args.fold_count}"
            ),
            "query_uses": (
                "precomputed_frozen_calibration_bank"
                if args.bank_input is not None
                else "all_calibration_cases_outside_query_fold"
            ),
            "debias_rule": "robust_global_content_residual_then_interpolated_equal_mass_conditional_midranks",
            "calibration_methods": reported_calibration_methods,
            "counts": calibration_counts,
        },
        "coverage": coverage,
        "rows": scored,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output}: rows={len(scored)}", flush=True)


if __name__ == "__main__":
    main()
