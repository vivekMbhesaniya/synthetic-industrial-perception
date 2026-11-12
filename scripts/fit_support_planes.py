"""Fit the real support plane in every frame of a T-LESS scene with RANSAC. Runs in sip.

Background pixels: valid real depth outside all parts' full masks, dilated to skip edge pixels.
Checks consistency: the same physical table must give the same plane in BOP's world frame.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from synperc.processing.bop import (
    find_image, list_image_ids, load_depth_m, load_mask, load_scene_annotations, load_scene_cameras,
)
from synperc.processing.projection import backproject_depth
from synperc.sensors.plane import fit_plane_ransac

THRESHOLD_M = 0.005
EDGE_MARGIN_PX = 5
MAX_POINTS = 20000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/phase4"))
    a = p.parse_args()

    scene_dir = a.dataset_root / "test_primesense" / f"{a.scene:06d}"
    cams = load_scene_cameras(scene_dir)
    anns = load_scene_annotations(scene_dir)
    kernel = np.ones((2 * EDGE_MARGIN_PX + 1, 2 * EDGE_MARGIN_PX + 1), np.uint8)
    rng = np.random.default_rng(0)

    planes, n_world, d_world = {}, [], []
    print(" image  bg_points  inliers  inlier_rms[mm]  dist_to_camera[m]  tilt_to_optical_axis[deg]")
    for im_id in list_image_ids(scene_dir):
        cam = cams[im_id]
        depth = load_depth_m(find_image(scene_dir / "depth", f"{im_id:06d}"), cam.depth_scale_mm)
        parts = np.zeros(depth.shape, np.uint8)
        for ann in anns[im_id]:
            parts |= load_mask(find_image(scene_dir / "mask", f"{im_id:06d}_{ann.gt_index:06d}")).astype(np.uint8)
        background = (cv2.dilate(parts, kernel) == 0) & (depth > 0)
        pts = backproject_depth(np.where(background, depth, 0.0), cam.K)
        if len(pts) > MAX_POINTS:
            pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]

        n, d, inl = fit_plane_ransac(pts, THRESHOLD_M, seed=im_id)
        rms = float(np.sqrt(np.mean((pts[inl] @ n + d) ** 2)) * 1000)
        tilt = float(np.degrees(np.arccos(abs(n[2]))))
        R, t = cam.world2cam[:3, :3], cam.world2cam[:3, 3]
        n_world.append(R.T @ n)
        d_world.append(float(n @ t + d))
        planes[im_id] = {"normal": n.tolist(), "d": d, "inlier_fraction": float(inl.mean()), "inlier_rms_mm": rms}
        print(f" {im_id:5d}  {len(pts):9d}  {inl.mean() * 100:6.1f}%  {rms:14.2f}  {d:17.3f}  {tilt:24.1f}")

    n_world, d_world = np.array(n_world), np.array(d_world)
    mean_n = n_world.mean(axis=0) / np.linalg.norm(n_world.mean(axis=0))
    angles = np.degrees(np.arccos(np.clip(n_world @ mean_n, -1, 1)))
    print(f"\nplane in BOP world frame over {len(planes)} frames:")
    print(f"  mean normal {np.round(mean_n, 4)}, angular spread: median {np.median(angles):.2f} deg, max {angles.max():.2f} deg")
    print(f"  offset d_w: mean {d_world.mean() * 1000:+.1f} mm, std {d_world.std() * 1000:.1f} mm")

    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"scene{a.scene:02d}_support_planes.json"
    out.write_text(json.dumps({"threshold_m": THRESHOLD_M, "edge_margin_px": EDGE_MARGIN_PX, "planes": planes}, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()