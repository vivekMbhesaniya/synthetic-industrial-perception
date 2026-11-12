"""Consensus support plane in BOP's world frame from per-frame RANSAC fits. Runs in sip.

Frames whose fit has too few inliers are rejected; the consensus is then transformed back into
every frame's camera with BOP's camera poses, including the rejected frames.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from synperc.processing.bop import load_scene_cameras

MIN_INLIER_FRACTION = 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("planes_json", type=Path)
    p.add_argument("--scene", type=int, default=1)
    a = p.parse_args()

    data = json.loads(a.planes_json.read_text(encoding="utf-8"))
    cams = load_scene_cameras(a.dataset_root / "test_primesense" / f"{a.scene:06d}")
    ids, n_world, d_world, good = [], [], [], []
    for key, pl in data["planes"].items():
        cam = cams[int(key)]
        R, t = cam.world2cam[:3, :3], cam.world2cam[:3, 3]
        n = np.array(pl["normal"])
        ids.append(int(key))
        n_world.append(R.T @ n)
        d_world.append(float(n @ t + pl["d"]))
        good.append(pl["inlier_fraction"] >= MIN_INLIER_FRACTION)
    ids, n_world, d_world, good = np.array(ids), np.array(n_world), np.array(d_world), np.array(good)

    n_cons = n_world[good].mean(axis=0)
    n_cons /= np.linalg.norm(n_cons)
    d_cons = float(np.median(d_world[good]))
    angle = np.degrees(np.arccos(np.clip(n_world @ n_cons, -1.0, 1.0)))
    offset_mm = (d_world - d_cons) * 1000

    print(f"frames {len(ids)}, accepted (inliers >= {MIN_INLIER_FRACTION:.0%}) {int(good.sum())}, rejected {ids[~good].tolist()}")
    print(f"consensus plane in world frame: normal {np.round(n_cons, 5)}, d_w {d_cons * 1000:+.2f} mm")
    print(f"accepted frames vs consensus: angle median {np.median(angle[good]):.3f} deg, max {angle[good].max():.3f} deg")
    print(f"                              offset median |.| {np.median(np.abs(offset_mm[good])):.2f} mm, max {np.abs(offset_mm[good]).max():.2f} mm")
    print(f"rejected frames vs consensus: angles {np.round(angle[~good], 1).tolist()} deg")

    planes_cam = {}
    for im in ids:
        cam = cams[int(im)]
        R, t = cam.world2cam[:3, :3], cam.world2cam[:3, 3]
        n_c = R @ n_cons
        planes_cam[int(im)] = {"normal": n_c.tolist(), "d": d_cons - float(n_c @ t)}
    d_cams = np.array([pl["d"] for pl in planes_cam.values()])
    print(f"consensus plane in camera frames: distance to camera {d_cams.min():.3f} .. {d_cams.max():.3f} m (must be > 0)")

    out = a.planes_json.with_name(a.planes_json.name.replace("support_planes", "consensus_plane"))
    out.write_text(
        json.dumps(
            {
                "min_inlier_fraction": MIN_INLIER_FRACTION,
                "world": {"normal": n_cons.tolist(), "d": d_cons},
                "accepted_frames": ids[good].tolist(),
                "rejected_frames": ids[~good].tolist(),
                "camera_frame_planes": planes_cam,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()