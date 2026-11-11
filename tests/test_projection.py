"""Projection geometry checked against closed-form results (no renderer involved)."""

import numpy as np
import pytest

from synperc.processing.projection import (
    backproject_depth,
    depth_to_distance,
    distance_to_depth,
    fit_plane,
)

H, W = 480, 640
K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def test_flat_depth_backprojects_to_a_flat_plane():
    points = backproject_depth(np.full((H, W), 1.0), K)
    assert points.shape == (H * W, 3)
    normal, rms = fit_plane(points)
    assert rms < 1e-9
    assert abs(abs(normal[2]) - 1.0) < 1e-9  # plane faces the camera


def test_distance_equals_depth_at_principal_point():
    distance = depth_to_distance(np.full((H, W), 2.0), K)
    assert distance[240, 320] == pytest.approx(2.0)


def test_corner_distance_matches_closed_form():
    distance = depth_to_distance(np.full((H, W), 1.0), K)
    x, y = (0 - 320) / 500, (0 - 240) / 500
    assert distance[0, 0] == pytest.approx(np.sqrt(1 + x**2 + y**2))


def test_depth_distance_roundtrip():
    depth = np.random.default_rng(0).uniform(0.5, 3.0, size=(H, W))
    assert np.allclose(distance_to_depth(depth_to_distance(depth, K), K), depth)


def test_distance_mistaken_for_depth_bends_a_plane():
    """The convention error on purpose: a ray-distance image back-projected as if it were depth."""
    wrong = backproject_depth(depth_to_distance(np.full((H, W), 1.0), K), K)
    _, rms = fit_plane(wrong)
    assert rms > 0.01  # centimetres of curvature on a plane 1 m away


def test_invalid_pixels_are_dropped():
    depth = np.full((4, 4), 1.0)
    depth[0, 0], depth[1, 1], depth[2, 2] = 0.0, np.nan, np.inf
    assert backproject_depth(depth, K).shape == (13, 3)