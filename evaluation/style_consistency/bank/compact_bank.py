"""Compact text form of the SDC real-video bank.

A full array bank stores every real-video shot pair (about 5M per cell). Scoring
only needs two kinds of one-dimensional distributions per cell: the content
energy CDF, and the residual-style CDF inside each content bin. This module
keeps each distribution as quantile knots and evaluates it with monotone cubic
interpolation, which reproduces full-bank SDC to about 1e-3 per video.

Each cell is one text file:
    line 1      content knots (energy at each content level)
    line 2      content-bin percentile centres (exact)
    line 3...   residual knots for each content bin, one line per bin

Export a full bank with:
    python -m evaluation.style_consistency.bank.compact_bank FULL_BANK_DIR OUTPUT_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

FORMAT = "sdc_quantile_bank_v1"
CONTENT_KNOTS, RESIDUAL_KNOTS = 256, 48
ENVS = ("real_env0", "real_env1", "real_env2")


def levels(k):
    """Cumulative-mass levels, denser in both tails (Chebyshev spacing)."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(k + 1) / k))


def _knots(values, prefix, p):
    """Values at mid-rank CDF levels p of a sorted, weighted sample."""
    values, prefix = np.asarray(values, dtype=np.float64), np.asarray(prefix, dtype=np.float64)
    cdf = (prefix[:-1] + prefix[1:]) / (2.0 * prefix[-1])
    x = np.interp(p, cdf, values)
    x[0], x[-1] = values[0], values[-1]
    return x


class _Cdf:
    """Monotone CDF through (x_k, p_k): 0 below the first knot, 1 above the last."""

    def __init__(self, x, p):
        x, first = np.unique(x, return_index=True)
        last = np.r_[first[1:], len(p)] - 1
        p = 0.5 * (p[first] + p[last])  # tied knots take the mid-rank of the tie
        self.lo, self.hi = x[0], x[-1]
        self.f = PchipInterpolator(x, p, extrapolate=False) if len(x) > 1 else None

    def __call__(self, q):
        out = np.where(q < self.lo, 0.0, 1.0)
        inside = (q >= self.lo) & (q <= self.hi)
        if self.f is not None and inside.any():
            out[inside] = np.clip(self.f(q[inside]), 0.0, 1.0)
        return out


class QuantileStyleBank:
    """Same query rule as ArrayConditionalResidualStyleBank.survival_many."""

    def __init__(self, kappa, minimum, maximum, content_knots, centers, residual_knots):
        self.kappa, self.minimum, self.maximum = float(kappa), float(minimum), float(maximum)
        self.content_knots = np.asarray(content_knots, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.residual_knots = np.asarray(residual_knots, dtype=np.float64)
        self.content_bins = len(self.centers)
        self._content = _Cdf(self.content_knots, levels(len(self.content_knots) - 1))
        p = levels(self.residual_knots.shape[1] - 1)
        self._residual = [_Cdf(x, p) for x in self.residual_knots]

    @classmethod
    def from_array_bank(cls, bank, content_knots=CONTENT_KNOTS, residual_knots=RESIDUAL_KNOTS):
        pc, pr = levels(content_knots), levels(residual_knots)
        return cls(bank.kappa, bank.minimum, bank.maximum,
                   _knots(bank.sorted_content, bank.content_weight_prefix, pc),
                   bank.content_bin_percentile_centers,
                   [_knots(v, w, pr) for v, w in zip(bank.sorted_conditional_residuals,
                                                     bank.conditional_weight_prefixes)])

    def survival_many(self, content, style):
        content, style = np.asarray(content, dtype=np.float64), np.asarray(style, dtype=np.float64)
        if content.shape != style.shape or content.ndim != 1:
            raise ValueError("query energies must be equal-length vectors")
        if not (np.isfinite(content).all() and np.isfinite(style).all()):
            raise ValueError("query energies must be finite")
        residual = style - self.kappa * content
        percentile = self._content(content)
        centers = self.centers
        right_bin = np.searchsorted(centers, percentile, side="right")
        left_bin = np.clip(right_bin - 1, 0, self.content_bins - 1)
        right_bin = np.clip(right_bin, 0, self.content_bins - 1)

        def ranks(which):
            out = np.empty(len(content), dtype=np.float64)
            for b in np.unique(which):
                mask = which == b
                out[mask] = 1.0 - self._residual[b](residual[mask])
            return out

        lo_rank, hi_rank = ranks(left_bin), ranks(right_bin)
        fraction = np.zeros(len(content), dtype=np.float64)
        interior = left_bin != right_bin
        fraction[interior] = ((percentile[interior] - centers[left_bin[interior]]) /
                              (centers[right_bin[interior]] - centers[left_bin[interior]]))
        result = (1.0 - fraction) * lo_rank + fraction * hi_rank
        return result, (content >= self.minimum) & (content <= self.maximum)

    def write(self, path):
        def line(values):
            return " ".join(f"{v:.7g}" for v in values)
        rows = [line(self.content_knots), line(self.centers)] + [line(x) for x in self.residual_knots]
        Path(path).write_text("\n".join(rows) + "\n")

    @classmethod
    def read(cls, path, kappa, minimum, maximum):
        rows = [np.array(r.split(), dtype=np.float64) for r in Path(path).read_text().splitlines()]
        return cls(kappa, minimum, maximum, rows[0], rows[1], np.stack(rows[2:]))


def load_compact_bank(path):
    """Return ({(head, fold, env): bank}, metadata) in the load_bank_set layout."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("format") != FORMAT:
        raise RuntimeError(f"{path} is not a {FORMAT} directory")
    banks = {}
    for cell in metadata["cells"]:
        bank = QuantileStyleBank.read(path / cell["file"], cell["kappa"], cell["minimum"], cell["maximum"])
        banks[(cell["head"], int(cell["fold"]), ENVS[int(cell["group"])])] = bank
    if len(banks) != 2 * len(ENVS) * metadata["folds"]:
        raise RuntimeError("compact bank is incomplete")
    return banks, metadata


def export(full, output):
    from evaluation.style_consistency.bank.array_bank import ArrayConditionalResidualStyleBank
    full, output = Path(full), Path(output)
    inventory = json.loads((full / "bank_inventory.json").read_text())
    if not inventory.get("complete") or len(inventory["cells"]) != 30:
        raise RuntimeError("full bank is not complete")
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for cell in sorted(inventory["cells"], key=lambda c: (c["head"], int(c["fold"]), int(c["group"]))):
        head, fold, group = cell["head"], int(cell["fold"]), int(cell["group"])
        bank = ArrayConditionalResidualStyleBank.load(full / cell["path"], mmap_mode=None)
        compact = QuantileStyleBank.from_array_bank(bank)
        name = f"{head}_fold{fold}_group{group}.txt"
        compact.write(output / name)
        cells.append(dict(head=head, fold=fold, group=group, file=name, kappa=bank.kappa,
                          minimum=bank.minimum, maximum=bank.maximum, pairs=len(bank.sorted_content)))
        print(json.dumps({"cell": name, "kappa": bank.kappa}), flush=True)
    metadata = {
        "format": FORMAT, "bank_id": inventory.get("bank_id"),
        "csd_head": inventory.get("csd_head", "module.last_layer_style"),
        "n_videos": inventory.get("n_videos"), "groups": 3, "folds": 5,
        "content_knots": CONTENT_KNOTS, "residual_knots": RESIDUAL_KNOTS,
        "levels": "0.5*(1-cos(pi*k/K)), k=0..K", "interpolation": "monotone cubic (PCHIP) CDF",
        "cells": cells,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("full", type=Path, help="full array bank directory (bank_inventory.json, cells/)")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.full, args.output)


if __name__ == "__main__":
    main()
