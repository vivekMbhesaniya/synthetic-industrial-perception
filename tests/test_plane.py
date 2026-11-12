"""RANSAC plane fit and plane depth checked on synthetic data with known answers."""

import numpy as np
import pytest

from synperc.processing.projection import backproject_depth
from synperc.sensors.plane import fit_plane_ransac, plane_depth

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def test_ransac_recovers_plane_despite_outliers():
    rng = np.random.default_rng(1)
    n_true = np.array([0.0, -0.6, -0.8])  # unit length, facing the camera
    d_true = 0.75
    xy = rng.uniform(-0.3, 0.3, size=(2000, 2))
    z = (-d_true - n_true[0] * xy[:, 0] - n_true[1] * xy[:, 1]) / n_true[2]
    plane_pts = np.column_stack((xy, z)) + rng.normal(scale=0.001, size=(2000, 3))
    outliers = rng.uniform([-0.3, -0.3, 0.3], [0.3, 0.3, 1.2], size=(800, 3))
    n, d, inliers = fit_plane_ransac(np.vstack((plane_pts, outliers)), threshold=0.005)
    assert abs(n @ n_true) == pytest.approx(1.0, abs=1e-3)
    assert d == pytest.approx(d_true, abs=2e-3)
    assert inliers[:2000].mean() > 0.95 and inliers[2000:].mean() < 0.1


def test_fronto_parallel_plane_depth_is_constant():
    z = plane_depth(np.array([0.0, 0.0, -1.0]), 1.0, K, 480, 640)  # plane z = 1
    assert np.allclose(z, 1.0)


def test_plane_depth_backprojects_onto_the_plane():
    n = np.array([0.0, -0.6, -0.8])
    d = 0.75
    pts = backproject_depth(plane_depth(n, d, K, 480, 640), K)
    assert np.allclose(pts @ n + d, 0.0, atol=1e-9)