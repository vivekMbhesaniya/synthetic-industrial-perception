"""Comparison metrics checked on hand-made masks and depth maps."""

import numpy as np
import pytest

from synperc.evaluation.compare import depth_residual_stats, mask_centroid, mask_iou


def test_iou_identical_disjoint_and_half():
    a = np.zeros((4, 4), bool)
    a[:, :2] = True
    b = np.zeros((4, 4), bool)
    b[:, :1] = True
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, ~a) == 0.0
    assert mask_iou(a, b) == pytest.approx(0.5)  # 4 shared pixels / 8 in the union


def test_iou_of_two_empty_masks_is_nan():
    empty = np.zeros((3, 3), bool)
    assert np.isnan(mask_iou(empty, empty))


def test_centroid_of_a_square():
    m = np.zeros((6, 8), bool)
    m[2:4, 4:6] = True  # rows 2-3, columns 4-5
    assert mask_centroid(m) == pytest.approx((4.5, 2.5))


def test_residual_bias_and_missing_fraction():
    rendered = np.full((2, 5), 1.0)
    real = np.full((2, 5), 1.002)  # constant 2 mm bias
    real[0, 0] = 0.0  # missing measurement
    s = depth_residual_stats(real, rendered, np.ones((2, 5), bool))
    assert s["pixels"] == 10
    assert s["missing_frac"] == pytest.approx(0.1)
    assert s["median"] == pytest.approx(0.002)
    assert s["std"] == pytest.approx(0.0, abs=1e-12)


def test_robust_sigma_ignores_a_single_outlier():
    rendered = np.zeros(11)
    real = np.full(11, 1.0)
    real[0] = 2.0  # one outlier, e.g. an edge pixel hitting the background
    s = depth_residual_stats(real, rendered, np.ones(11, bool))
    assert s["robust_sigma"] == pytest.approx(0.0, abs=1e-12)
    assert s["std"] > 0.2  # the plain standard deviation is inflated by the outlier