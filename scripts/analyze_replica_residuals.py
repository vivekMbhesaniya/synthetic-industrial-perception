"""Analyse pooled replica residuals: outlier sign, visibility-test asymmetry, per-instance bias. Runs in sip."""

import argparse
from pathlib import Path

import numpy as np

EDGE_BINS = [(0.0, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 4.5, "3-4"), (4.5, 9.5, "5-9"), (9.5, np.inf, ">=10")]
BIG_MM = 50.0


def robust_sigma(r: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("samples", type=Path)
    a = p.parse_args()

    S = np.load(a.samples)
    r = S["residual_m"].astype(np.float64) * 1000  # mm, NaN = missing real depth
    valid = np.isfinite(r)
    in_visib = S["in_visib"]
    edge = S["edge_px"]

    print(f"1) Residual sign, thresholds +-{BIG_MM:.0f} mm and -15 mm")
    for name, sel in [("all pixels", valid), ("inside mask_visib", valid & in_visib), ("outside mask_visib", valid & ~in_visib)]:
        rs = r[sel]
        print(f"   {name:<20} n={rs.size:>8}   r>+{BIG_MM:.0f}: {np.mean(rs > BIG_MM) * 100:6.2f}%"
              f"   r<-{BIG_MM:.0f}: {np.mean(rs < -BIG_MM) * 100:6.2f}%   r<-15: {np.mean(rs < -15) * 100:6.2f}%")

    print(f"\n2) Large residuals by distance to edge")
    for lo, hi, label in EDGE_BINS:
        rs = r[valid & (edge > lo) & (edge <= hi)]
        print(f"   edge {label:<5} n={rs.size:>8}   r>+{BIG_MM:.0f}: {np.mean(rs > BIG_MM) * 100:6.2f}%"
              f"   r<-{BIG_MM:.0f}: {np.mean(rs < -BIG_MM) * 100:6.2f}%")

    print("\n3) Per-instance bias vs within-instance noise (interior pixels, edge >= 10 px)")
    # In this scene every object appears once per frame, so (image, obj_id) identifies an instance.
    inst = S["image"].astype(np.int64) * 1000 + S["obj_id"].astype(np.int64)
    sel = valid & (edge > 9.5)
    ids, rr = inst[sel], r[sel]
    order = np.argsort(ids, kind="stable")
    ids, rr = ids[order], rr[order]
    _, starts = np.unique(ids, return_index=True)
    groups = np.split(rr, starts[1:])
    medians = np.array([np.median(g) for g in groups])
    within = np.concatenate([g - m for g, m in zip(groups, medians)])
    print(f"   instances: {len(groups)}")
    print(f"   instance medians: centre {np.median(medians):+.2f} mm, robust spread {robust_sigma(medians):.2f} mm, "
          f"range {medians.min():+.1f} .. {medians.max():+.1f} mm")
    print(f"   pooled robust sigma      {robust_sigma(rr):.2f} mm")
    print(f"   within-instance sigma    {robust_sigma(within):.2f} mm  (after removing each instance's median)")


if __name__ == "__main__":
    main()