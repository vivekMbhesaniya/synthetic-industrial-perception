"""Annotation helpers: bounding boxes from masks and point-to-mesh distances."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bounding box (x, y, width, height) in pixels of a non-empty boolean mask."""
    v, u = np.nonzero(mask)
    if u.size == 0:
        raise ValueError("Empty mask has no bounding box")
    x0, y0 = int(u.min()), int(v.min())
    return x0, y0, int(u.max()) - x0 + 1, int(v.max()) - y0 + 1


def distances_to_mesh(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Unsigned distance from each (N, 3) point to the surface of a triangle mesh (same units)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got {pts.shape}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(pts)).numpy().astype(np.float64)