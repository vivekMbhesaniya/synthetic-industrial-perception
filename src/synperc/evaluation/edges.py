"""Distance of mask pixels to the mask boundary."""

import cv2
import numpy as np


def distance_to_mask_edge(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance in pixels from each mask pixel to the nearest pixel outside the mask.

    Pixels outside the mask get 0. The image border counts as outside, so a part cut off by the
    border has an edge there too.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {m.shape}")
    padded = np.pad(m, 1, constant_values=False).astype(np.uint8)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist[1:-1, 1:-1].astype(np.float64)