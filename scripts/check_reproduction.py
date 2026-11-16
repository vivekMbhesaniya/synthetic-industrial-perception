"""Compare a fresh pipeline run with the committed results: a determinism and regression check. Runs in sip."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def csv_diff(a: Path, b: Path) -> float:
    A, B = pd.read_csv(a), pd.read_csv(b)
    if list(A.columns) != list(B.columns) or len(A) != len(B):
        return float("inf")
    num = A.select_dtypes("number").columns
    x, y = A[num].to_numpy(float), B[num].to_numpy(float)
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return float("inf")
    d = np.abs(x - y)
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def planes_diff(a: Path, b: Path, key: str) -> float:
    A = json.loads(a.read_text(encoding="utf-8"))[key]
    B = json.loads(b.read_text(encoding="utf-8"))[key]
    if A.keys() != B.keys():
        return float("inf")
    return max(max(float(np.max(np.abs(np.subtract(A[k]["normal"], B[k]["normal"])))), abs(A[k]["d"] - B[k]["d"])) for k in A)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("committed", type=Path, help="committed results folder, e.g. results")
    p.add_argument("fresh", type=Path, help="results folder of the fresh run")
    a = p.parse_args()
    checks = [
        ("replica per-object stats", csv_diff(a.committed / "phase2" / "scene01_replica_per_object.csv", a.fresh / "scene01_replica_per_object.csv"), 1e-3),
        ("support planes", planes_diff(a.committed / "phase4" / "scene01_support_planes.json", a.fresh / "scene01_support_planes.json", "planes"), 1e-9),
        ("consensus planes", planes_diff(a.committed / "phase4" / "scene01_consensus_plane.json", a.fresh / "scene01_consensus_plane.json", "camera_frame_planes"), 1e-9),
    ]
    ok = True
    for name, diff, tol in checks:
        ok &= diff <= tol
        print(f"{name:<26} max abs difference {diff:.3g} (tolerance {tol:g}): {'OK' if diff <= tol else 'DIFFERENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()