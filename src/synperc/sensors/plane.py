"""Robust plane fitting (RANSAC + total least squares) and per-pixel plane depth."""

from __future__ import annotations

import numpy as np

from synperc.processing.projection import normalized_coords


def fit_plane_ransac(points: np.ndarray, threshold: float, iterations: int = 300, seed: int = 0):
    """Fit n.p + d = 0 robustly. Returns (unit normal, d, inlier mask), normal facing the camera origin."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError(f"Expected (N >= 3, 3) points, got {pts.shape}")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        p0, p1, p2 = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue  # degenerate (collinear) sample
        n /= norm
        inliers = np.abs(pts @ n - n @ p0) < threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 3:
        raise RuntimeError("RANSAC found no plane")
    # Refit on all inliers with total least squares, then recompute inliers
    centroid = pts[best].mean(axis=0)
    n = np.linalg.svd(pts[best] - centroid, full_matrices=False)[2][-1]
    d = -float(n @ centroid)
    if d < 0:  # orient the normal towards the camera at the origin
        n, d = -n, -d
    return n, d, np.abs(pts @ n + d) < threshold


def plane_depth(normal: np.ndarray, d: float, K: np.ndarray, height: int, width: int) -> np.ndarray:
    """Depth z at which each pixel's ray z * (x~, y~, 1) meets the plane; NaN where it never does."""
    x, y = normalized_coords(K, height, width)
    denom = normal[0] * x + normal[1] * y + normal[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -d / denom
    z[~np.isfinite(z) | (z <= 0)] = np.nan
    return z