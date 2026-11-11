"""Inspect one real T-LESS frame and check that K, pose and depth units agree. Runs in sip."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synperc.processing.bop import (
    find_image,
    list_image_ids,
    load_depth_m,
    load_mask,
    load_rgb,
    load_scene_annotations,
    load_scene_cameras,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--split", default="test_primesense")
    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--image", type=int, default=None, help="image id (default: first existing id)")
    p.add_argument("--out", type=Path, default=Path("results/phase2"))
    a = p.parse_args()

    scene_dir = a.dataset_root / a.split / f"{a.scene:06d}"
    ids = list_image_ids(scene_dir)
    im_id = ids[0] if a.image is None else a.image
    print(f"scene {a.scene}: {len(ids)} images, first ids {ids[:5]}; using image {im_id}")

    cam = load_scene_cameras(scene_dir)[im_id]
    anns = load_scene_annotations(scene_dir)[im_id]
    rgb = load_rgb(find_image(scene_dir / "rgb", f"{im_id:06d}"))
    depth = load_depth_m(find_image(scene_dir / "depth", f"{im_id:06d}"), cam.depth_scale_mm)

    h, w = depth.shape
    valid = depth > 0
    print(f"image size : depth {w} x {h}, rgb {rgb.shape[1]} x {rgb.shape[0]}")
    print(f"K          : fx={cam.K[0, 0]:.2f} fy={cam.K[1, 1]:.2f} cx={cam.K[0, 2]:.2f} cy={cam.K[1, 2]:.2f}")
    print(f"depth      : {valid.mean() * 100:.1f} % valid, {(~valid).mean() * 100:.1f} % missing")
    print(f"             valid range {depth[valid].min():.3f} .. {depth[valid].max():.3f} m, median {np.median(depth[valid]):.3f} m")
    print(f"objects    : {len(anns)}")
    print("  gt  obj  visib  t_z[m]  depth@origin[m]  t_z-depth[mm]  origin on visible mask")

    overlay = rgb.astype(np.float64) / 255.0 * 0.4
    colors = np.random.default_rng(0).uniform(0.3, 1.0, size=(len(anns), 3))
    points = []
    for ann, color in zip(anns, colors):
        mask = load_mask(find_image(scene_dir / "mask_visib", f"{im_id:06d}_{ann.gt_index:06d}"))
        overlay[mask] = 0.4 * overlay[mask] + 0.6 * color
        t = ann.model2cam[:3, 3]
        u, v = (cam.K @ t)[:2] / t[2]
        points.append((u, v))
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            d = depth[vi, ui]
            d_str = f"{d:.3f}" if d > 0 else "missing"
            diff = f"{(t[2] - d) * 1000:+.1f}" if d > 0 else "-"
            on_mask = bool(mask[vi, ui])
        else:
            d_str, diff, on_mask = "outside", "-", False
        vis = f"{ann.visib_fract:.2f}" if ann.visib_fract is not None else "n/a"
        print(f"  {ann.gt_index:2d}  {ann.obj_id:3d}  {vis:>5}  {t[2]:.3f}  {d_str:>15}  {diff:>13}  {on_mask}")

    a.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Real RGB, scene {a.scene}, image {im_id}")
    im = axes[1].imshow(np.where(valid, depth, np.nan))
    axes[1].set_title("Real depth [m], white = missing")
    fig.colorbar(im, ax=axes[1])
    axes[2].imshow(overlay)
    for u, v in points:
        axes[2].plot(u, v, "r+", markersize=12, mew=2)
    axes[2].set_title("Visible masks + projected model origins")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = a.out / f"real_frame_s{a.scene:02d}_im{im_id:06d}.png"
    fig.savefig(out, dpi=110)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()