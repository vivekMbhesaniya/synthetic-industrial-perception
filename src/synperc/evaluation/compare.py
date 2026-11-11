"""Compare rendered and real data: mask overlap and depth residuals."""

from __future__ import annotations

import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks (NaN if both are empty)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Mean pixel position (u, v) of a boolean mask."""
    v, u = np.nonzero(mask)
    if u.size == 0:
        raise ValueError("Empty mask has no centroid")
    return float(u.mean()), float(v.mean())


def depth_residual_stats(real: np.ndarray, rendered: np.ndarray, region: np.ndarray) -> dict[str, float]:
    """Statistics of real - rendered depth (metres) inside region. Real depth 0 counts as missing."""
    region = np.asarray(region, dtype=bool)
    n = int(region.sum())
    if n == 0:
        raise ValueError("Empty region")
    valid = region & (real > 0) & np.isfinite(rendered)
    r = (real - rendered)[valid]
    nan = float("nan")
    med = float(np.median(r)) if r.size else nan
    return {
        "pixels": n,
        "missing_frac": float(1.0 - valid.sum() / n),
        "mean": float(r.mean()) if r.size else nan,
        "median": med,
        "std": float(r.std()) if r.size else nan,
        "robust_sigma": float(1.4826 * np.median(np.abs(r - med))) if r.size else nan,
        "p95_abs": float(np.percentile(np.abs(r), 95)) if r.size else nan,
    }