"""Annotation helpers checked on shapes with known answers."""

import numpy as np
import open3d as o3d
import pytest

from synperc.processing.annotations import bbox_from_mask, distances_to_mesh


def test_bbox_of_rectangle():
    m = np.zeros((10, 12), bool)
    m[2:5, 3:9] = True  # rows 2-4, columns 3-8
    assert bbox_from_mask(m) == (3, 2, 6, 3)


def test_bbox_of_empty_mask_raises():
    with pytest.raises(ValueError):
        bbox_from_mask(np.zeros((3, 3), bool))


def test_distances_to_unit_cube():
    cube = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)  # spans [0, 1]^3
    pts = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, -0.2], [0.5, 0.5, 0.5]])
    assert distances_to_mesh(pts, cube) == pytest.approx([0.0, 0.2, 0.5], abs=1e-6)