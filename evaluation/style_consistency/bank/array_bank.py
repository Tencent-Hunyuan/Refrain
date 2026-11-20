"""Array-backed implementation of the existing conditional residual SDC bank.

The estimator, stable weighted quantiles, tie handling, bins, and interpolation
match ConditionalResidualStyleBank. Only data representation and storage differ.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from evaluation.style_features.calibration import (
    ConditionalResidualStyleBank, weighted_quantile,
)


SCALARS = (
    "kappa", "huber_delta", "huber_iterations", "huber_scale", "huber_converged",
    "huber_fixed_point_relative_error", "huber_clipped_weight_fraction",
    "huber_mean_influence", "effective_sample_size", "minimum", "maximum",
    "requested_content_bins",
)


def score_pairs_many(pairs, banks, envs, fold):
    """The n02.score result schema, with all queries in one fold vectorized."""
    if not pairs:
        return []
    content = np.asarray([float(q["content_energy"]) for q in pairs], dtype=np.float64)
    styles = {head: np.asarray([float(q[key]) for q in pairs], dtype=np.float64)
              for head, key in (("csd", "csd_style_energy"), ("vgg", "vgg_style_energy"))}
    ranks, supports = {}, {}
    for head, style in styles.items():
        values, in_ranges = [], []
        for env in envs:
            bank = banks.get((head, fold, env))
            if bank is None:
                continue
            if hasattr(bank, "survival_many"):
                value, in_range = bank.survival_many(content, style)
            else:
                result = [bank.survival(c, s) for c, s in zip(content, style)]
                value = np.asarray([r[0] for r in result])
                in_range = np.asarray([r[1] for r in result])
            values.append(value)
            in_ranges.append(in_range)
        if not values:
            raise RuntimeError(f"no calibration environment for {head}, fold={fold}")
        ranks[head] = np.mean(values, axis=0)
        supports[head] = np.sum(in_ranges, axis=0)
    pooled = ranks["csd"] ** (2.0 / 3.0) * ranks["vgg"] ** (1.0 - 2.0 / 3.0)
    return [{"content_energy": float(content[i]),
             "csd_style_energy": float(styles["csd"][i]), "vgg_style_energy": float(styles["vgg"][i]),
             "R_csd": float(ranks["csd"][i]), "R_vgg": float(ranks["vgg"][i]),
             "n_env_in_minmax_csd": int(supports["csd"][i]),
             "n_env_in_minmax_vgg": int(supports["vgg"][i]), "pair_score": float(pooled[i])}
            for i in range(len(pairs))]


class ArrayConditionalResidualStyleBank(ConditionalResidualStyleBank):
    """Identical fitted bank without millions of temporary Python dictionaries."""

    @classmethod
    def fit(cls, content, style, weights, progress=None):
        self = cls.__new__(cls)
        content = np.asarray(content, dtype=np.float64)
        style = np.asarray(style, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if content.ndim != 1 or style.shape != content.shape or weights.shape != content.shape:
            raise ValueError("content, style, weights must be equal-length vectors")
        if len(content) < 32:
            raise RuntimeError(f"residual bank is too small: {len(content)}")
        if not (np.isfinite(content).all() and np.isfinite(style).all()):
            raise RuntimeError("bank energies must be finite")
        if np.any(content < 0) or np.any(style < 0):
            raise RuntimeError("bank energies must be nonnegative")
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise RuntimeError("residual bank weights must be finite and positive")
        self.effective_sample_size = float(np.sum(weights) ** 2 / np.sum(np.square(weights)))
        positive = content > np.finfo(np.float32).eps
        if int(np.sum(positive)) < 16 or float(np.sum(weights[positive])) <= 0:
            raise RuntimeError("too few nonzero-content pairs for robust residual slope")
        kappa = float(weighted_quantile(style[positive] / content[positive], weights[positive],
                                        np.asarray([0.5], dtype=np.float64))[0])
        del positive
        self.huber_delta = 1.345
        self.huber_converged = False
        for iteration in range(1, 101):
            residual = style - kappa * content
            center = float(weighted_quantile(residual, weights, np.asarray([0.5]))[0])
            scale = 1.4826 * float(weighted_quantile(np.abs(residual - center), weights,
                                                    np.asarray([0.5]))[0])
            self.huber_iterations, self.huber_scale = iteration, scale
            if progress is not None and (iteration == 1 or iteration % 5 == 0):
                progress("huber_iteration", iteration=iteration, kappa=kappa, scale=scale)
            if scale <= np.finfo(np.float64).eps:
                self.huber_converged = True
                break
            influence = np.minimum(1.0, self.huber_delta * scale /
                                   np.maximum(np.abs(residual), np.finfo(np.float64).eps))
            denominator = float(np.sum(weights * influence * np.square(content)))
            if denominator <= 0:
                raise RuntimeError("Huber residual bank has zero content energy")
            updated = float(np.sum(weights * influence * content * style) / denominator)
            if abs(updated - kappa) <= 1e-12 * (1.0 + abs(kappa)):
                kappa = updated
                self.huber_converged = True
                break
            kappa = updated
        if not self.huber_converged:
            raise RuntimeError("Huber residual slope did not converge in 100 iterations")
        self.kappa = kappa
        if not np.isfinite(kappa) or kappa < 0:
            raise RuntimeError(f"invalid robust content slope kappa={kappa}")
        residual = style - kappa * content
        center = float(weighted_quantile(residual, weights, np.asarray([0.5]))[0])
        self.huber_scale = 1.4826 * float(weighted_quantile(np.abs(residual - center), weights,
                                                          np.asarray([0.5]))[0])
        if self.huber_scale <= np.finfo(np.float64).eps:
            final_influence = np.ones_like(residual)
        else:
            final_influence = np.minimum(1.0, self.huber_delta * self.huber_scale /
                                         np.maximum(np.abs(residual), np.finfo(np.float64).eps))
        fixed_denominator = float(np.sum(weights * final_influence * np.square(content)))
        if fixed_denominator <= 0:
            raise RuntimeError("final Huber fixed-point audit has zero content energy")
        fixed_update = float(np.sum(weights * final_influence * content * style) / fixed_denominator)
        self.huber_fixed_point_relative_error = abs(fixed_update - kappa) / (1.0 + abs(kappa))
        if self.huber_fixed_point_relative_error > 1e-12 + 1e-15:
            raise RuntimeError("Huber fixed-point audit exceeds convergence tolerance")
        total_weight = float(np.sum(weights))
        self.huber_clipped_weight_fraction = float(np.sum(weights[final_influence < 1.0]) / total_weight)
        self.huber_mean_influence = float(np.sum(weights * final_influence) / total_weight)
        del final_influence
        if "influence" in locals():
            del influence
        self.minimum, self.maximum = float(content.min()), float(content.max())
        self.requested_content_bins = max(self.MINIMUM_BINS,
            min(self.MAXIMUM_BINS, int(round(np.sqrt(self.effective_sample_size)))))
        if progress is not None:
            progress("huber_complete", iterations=self.huber_iterations, kappa=kappa,
                     fixed_point_error=self.huber_fixed_point_relative_error)

        # Reuse the stable content ordering for quantiles, the global CDF, and
        # bin centres. This is the exact weighted_quantile operation order.
        order = np.argsort(content, kind="mergesort")
        self.sorted_content = content[order]
        sorted_weights = weights[order]
        self.content_weight_prefix = np.concatenate((np.zeros(1), np.cumsum(sorted_weights)))
        targets = np.linspace(0, 1, self.requested_content_bins + 1) * self.content_weight_prefix[-1]
        indices = np.searchsorted(self.content_weight_prefix[1:], targets, side="left")
        indices = np.clip(indices, 0, len(content) - 1)
        self.content_edges = np.unique(self.sorted_content[indices])
        del order, sorted_weights, targets, indices
        if len(self.content_edges) < 3:
            raise RuntimeError("conditional residual bank has insufficient content support")
        assignment = np.searchsorted(self.content_edges[1:-1], content, side="right")
        counts = np.bincount(assignment, minlength=len(self.content_edges) - 1)
        if np.any(counts == 0):
            raise RuntimeError("conditional residual bank contains an empty bin")
        self.bin_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
        # Stable integer grouping preserves original order inside each bin,
        # avoiding one complete N-item scan per bin without changing tie order.
        grouped = np.argsort(assignment, kind="stable")
        del assignment
        self.flat_residuals = np.empty(len(content), dtype=np.float64)
        self.flat_conditional_prefix = np.empty(len(content) + len(counts), dtype=np.float64)
        centers = []
        for index in range(len(counts)):
            lo, hi = self.bin_offsets[index:index + 2]
            selected = grouped[lo:hi]
            residual_values, selected_weights = residual[selected], weights[selected]
            order = np.argsort(residual_values, kind="mergesort")
            self.flat_residuals[lo:hi] = residual_values[order]
            prefix = self.flat_conditional_prefix[lo + index:hi + index + 1]
            prefix[0] = 0.0
            prefix[1:] = np.cumsum(selected_weights[order])
            center_content = float(weighted_quantile(content[selected], selected_weights,
                                                     np.asarray([0.5]))[0])
            centers.append(self.content_percentile(center_content))
        self.content_bin_percentile_centers = np.asarray(centers, dtype=np.float64)
        if np.any(np.diff(self.content_bin_percentile_centers) <= 0):
            raise RuntimeError("conditional content-bin centres are not increasing")
        self._make_bin_views()
        if progress is not None:
            progress("bank_complete", items=len(content), bins=self.content_bins,
                     effective_sample_size=self.effective_sample_size)
        return self

    def _make_bin_views(self):
        self.sorted_conditional_residuals = []
        self.conditional_weight_prefixes = []
        for b, (lo, hi) in enumerate(zip(self.bin_offsets[:-1], self.bin_offsets[1:])):
            self.sorted_conditional_residuals.append(self.flat_residuals[lo:hi])
            self.conditional_weight_prefixes.append(self.flat_conditional_prefix[lo + b:hi + b + 1])

    def survival_many(self, content, style):
        """Vectorized existing query rule; callers may chunk arbitrarily."""
        content, style = np.asarray(content, dtype=np.float64), np.asarray(style, dtype=np.float64)
        if content.shape != style.shape or content.ndim != 1:
            raise ValueError("query energies must be equal-length vectors")
        if not (np.isfinite(content).all() and np.isfinite(style).all()):
            raise ValueError("query energies must be finite")
        residual = style - self.kappa * content
        left = np.searchsorted(self.sorted_content, content, side="left")
        right = np.searchsorted(self.sorted_content, content, side="right")
        prefix = self.content_weight_prefix
        percentile = (prefix[left] + 0.5 * (prefix[right] - prefix[left])) / prefix[-1]
        centers = self.content_bin_percentile_centers
        right_bin = np.searchsorted(centers, percentile, side="right")
        left_bin = np.clip(right_bin - 1, 0, self.content_bins - 1)
        right_bin = np.clip(right_bin, 0, self.content_bins - 1)

        def ranks(which):
            out = np.empty(len(content), dtype=np.float64)
            for b in np.unique(which):
                mask = which == b
                values = self.sorted_conditional_residuals[b]
                weights = self.conditional_weight_prefixes[b]
                lo = np.searchsorted(values, residual[mask], side="left")
                hi = np.searchsorted(values, residual[mask], side="right")
                out[mask] = ((weights[-1] - weights[hi]) +
                             0.5 * (weights[hi] - weights[lo])) / weights[-1]
            return out

        lo_rank, hi_rank = ranks(left_bin), ranks(right_bin)
        fraction = np.zeros(len(content), dtype=np.float64)
        interior = left_bin != right_bin
        fraction[interior] = ((percentile[interior] - centers[left_bin[interior]]) /
                              (centers[right_bin[interior]] - centers[left_bin[interior]]))
        result = (1.0 - fraction) * lo_rank + fraction * hi_rank
        return result, (content >= self.minimum) & (content <= self.maximum)

    def save(self, directory, provenance=None):
        """Atomically publish a complete, memory-mappable cell directory."""
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            raise FileExistsError(directory)
        temporary = directory.with_name(f".{directory.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        temporary.mkdir()
        arrays = {name: getattr(self, name) for name in (
            "sorted_content", "content_weight_prefix", "flat_residuals",
            "flat_conditional_prefix", "bin_offsets", "content_edges",
            "content_bin_percentile_centers")}
        try:
            for name, value in arrays.items():
                np.save(temporary / f"{name}.npy", value, allow_pickle=False)
            metadata = {"format": "sdc_conditional_array_bank_v1", "complete": True,
                        "scalars": {key: getattr(self, key) for key in SCALARS},
                        "resolution_rule": self.RESOLUTION_RULE,
                        "interpolation_rule": self.INTERPOLATION_RULE,
                        "provenance": provenance or {},
                        "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str}
                                   for key, value in arrays.items()}}
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            os.rename(temporary, directory)
        except BaseException:
            shutil.rmtree(temporary)
            raise

    @classmethod
    def load(cls, directory, mmap_mode="r", verify=False):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("format") != "sdc_conditional_array_bank_v1" or not metadata.get("complete"):
            raise RuntimeError("incomplete or unsupported bank directory")
        if (metadata["resolution_rule"] != cls.RESOLUTION_RULE or
                metadata["interpolation_rule"] != cls.INTERPOLATION_RULE):
            raise RuntimeError("bank estimator rules differ")
        self = cls.__new__(cls)
        for key, value in metadata["scalars"].items():
            setattr(self, key, value)
        for key, spec in metadata["arrays"].items():
            value = np.load(directory / f"{key}.npy", mmap_mode=mmap_mode, allow_pickle=False)
            if list(value.shape) != spec["shape"] or value.dtype.str != spec["dtype"]:
                raise RuntimeError(f"bank array shape/type mismatch: {key}")
            setattr(self, key, value)
        if (len(self.content_weight_prefix) != len(self.sorted_content) + 1 or
                self.bin_offsets[-1] != len(self.flat_residuals) or
                len(self.flat_conditional_prefix) != len(self.flat_residuals) + len(self.bin_offsets) - 1):
            raise RuntimeError("bank array topology mismatch")
        self._make_bin_views()
        if verify:
            for values in [self.sorted_content, *self.sorted_conditional_residuals]:
                if not np.isfinite(values).all() or np.any(np.diff(values) < 0):
                    raise RuntimeError("bank values are not finite and sorted")
            for prefix in [self.content_weight_prefix, *self.conditional_weight_prefixes]:
                if prefix[0] != 0 or not np.isfinite(prefix).all() or np.any(np.diff(prefix) <= 0):
                    raise RuntimeError("bank empirical masses are invalid")
        self.provenance = metadata.get("provenance", {})
        return self

    def to_dict(self):
        raise RuntimeError("Use save(directory); JSON serialization of full bank arrays is prohibited")
