"""Pinhole camera geometry: depth images, ray distances and point clouds.

Convention: OpenCV camera frame (x right, y down, z forward), pixel (u, v) = (column, row).
"""

from __future__ import annotations

import numpy as np


def _check_inputs(image: np.ndarray, K: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, got shape {K.shape}")


def normalized_coords(K: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x~ = (u - cx) / fx and y~ = (v - cy) / fy for every pixel."""
    v, u = np.indices((height, width), dtype=np.float64)
    return (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]


def depth_to_distance(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Planar depth Z -> distance along the viewing ray, r = Z * sqrt(1 + x~^2 + y~^2)."""
    _check_inputs(depth, K)
    x, y = normalized_coords(K, *depth.shape)
    return depth * np.sqrt(1.0 + x**2 + y**2)


def distance_to_depth(distance: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Distance along the viewing ray -> planar depth Z."""
    _check_inputs(distance, K)
    x, y = normalized_coords(K, *distance.shape)
    return distance / np.sqrt(1.0 + x**2 + y**2)


def backproject_depth(depth: np.ndarray, K: np.ndarray, max_depth: float | None = None) -> np.ndarray:
    """Return an (N, 3) array of camera-frame points for all valid pixels.

    Pixels that are NaN, infinite, <= 0 or >= max_depth count as missing measurements.
    """
    _check_inputs(depth, K)
    x, y = normalized_coords(K, *depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth < max_depth
    z = depth[valid]
    return np.column_stack((x[valid] * z, y[valid] * z, z))


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane fit via SVD. Returns the unit normal and the RMS point-to-plane distance."""
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"Expected an (N >= 3, 3) array, got shape {points.shape}")
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    residuals = centered @ normal
    return normal, float(np.sqrt(np.mean(residuals**2)))