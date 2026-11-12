"""Edge distance checked on shapes with known answers."""

import numpy as np
import pytest

from synperc.evaluation.edges import distance_to_mask_edge


def test_square_boundary_and_centre():
    m = np.zeros((9, 9), bool)
    m[2:7, 2:7] = True  # 5 x 5 square
    d = distance_to_mask_edge(m)
    assert d[2, 2] == pytest.approx(1.0)  # boundary pixel
    assert d[4, 4] == pytest.approx(3.0)  # centre
    assert d[0, 0] == 0.0  # outside the mask


def test_image_border_counts_as_edge():
    d = distance_to_mask_edge(np.ones((3, 3), bool))
    assert d[0, 0] == pytest.approx(1.0)
    assert d[1, 1] == pytest.approx(2.0)