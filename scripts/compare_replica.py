"""Compare a rendered replica with its real BOP frame. Runs in sip.

Reports mask IoU, centroid offset and depth residuals (real - rendered) per object.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synperc.evaluation.compare import depth_residual_stats, mask_centroid, mask_iou
from synperc.processing.bop import find_image, load_depth_m, load_mask, load_rgb, load_scene_cameras

SEG_KEY = "category_id_segmaps"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("job_dir", type=Path)
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/phase2"))
    a = p.parse_args()

    job = json.loads((a.job_dir / "job.json").read_text(encoding="utf-8"))
    src = job["source"]
    scene_dir = a.dataset_root / src["split"] / f"{src['scene']:06d}"
    im_id = src["image"]
    cam = load_scene_cameras(scene_dir)[im_id]
    real_rgb = load_rgb(find_image(scene_dir / "rgb", f"{im_id:06d}"))
    real_depth = load_depth_m(find_image(scene_dir / "depth", f"{im_id:06d}"), cam.depth_scale_mm)

    with h5py.File(a.job_dir / "0.hdf5", "r") as f:
        if SEG_KEY not in f:
            raise KeyError(f"'{SEG_KEY}' not found, file contains {list(f.keys())}")
        ren_rgb = np.array(f["colors"])
        ren_depth = np.array(f["depth"], dtype=np.float64)
        seg = np.array(f[SEG_KEY])
    if ren_depth.shape != real_depth.shape:
        raise ValueError(f"Shape mismatch: rendered {ren_depth.shape}, real {real_depth.shape}")

    print("  gt  obj    IoU   du[px]  dv[px]  pixels  missing  median[mm]  std[mm]  robust_sigma[mm]")
    residual = np.full(real_depth.shape, np.nan)
    real_any = np.zeros(real_depth.shape, bool)
    ren_any = np.zeros(real_depth.shape, bool)
    all_both = np.zeros(real_depth.shape, bool)
    for o in job["objects"]:
        real_mask = load_mask(find_image(scene_dir / "mask_visib", f"{im_id:06d}_{o['gt_index']:06d}"))
        ren_mask = seg == o["gt_index"] + 1
        iou = mask_iou(real_mask, ren_mask)
        (ur, vr), (us, vs) = mask_centroid(real_mask), mask_centroid(ren_mask)
        both = real_mask & ren_mask
        s = depth_residual_stats(real_depth, ren_depth, both)
        ok = both & (real_depth > 0)
        residual[ok] = (real_depth - ren_depth)[ok]
        real_any |= real_mask
        ren_any |= ren_mask
        all_both |= both
        print(
            f"  {o['gt_index']:2d}  {o['obj_id']:3d}  {iou:5.3f}  {ur - us:+6.2f}  {vr - vs:+6.2f}"
            f"  {s['pixels']:6d}  {s['missing_frac'] * 100:5.1f}%  {s['median'] * 1000:+9.2f}"
            f"  {s['std'] * 1000:7.2f}  {s['robust_sigma'] * 1000:9.2f}"
        )

    total = depth_residual_stats(real_depth, ren_depth, all_both)
    print(
        f"  all objects: {total['pixels']} px, median {total['median'] * 1000:+.2f} mm, "
        f"std {total['std'] * 1000:.2f} mm, robust sigma {total['robust_sigma'] * 1000:.2f} mm, "
        f"95% of |residual| below {total['p95_abs'] * 1000:.2f} mm"
    )

    agree = np.zeros(real_depth.shape + (3,))
    agree[real_any & ren_any] = (1.0, 1.0, 1.0)
    agree[real_any & ~ren_any] = (1.0, 0.2, 0.2)
    agree[~real_any & ren_any] = (0.2, 1.0, 0.2)
    lim = float(np.nanpercentile(np.abs(residual), 98)) * 1000

    a.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].imshow(real_rgb)
    axes[0, 0].set_title("Real RGB")
    axes[0, 1].imshow(ren_rgb)
    axes[0, 1].set_title("Rendered RGB (same K, ground-truth poses)")
    axes[1, 0].imshow(agree)
    axes[1, 0].set_title("Masks: white = both, red = real only, green = rendered only")
    im = axes[1, 1].imshow(residual * 1000, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[1, 1].set_title("Depth residual real - rendered [mm]")
    fig.colorbar(im, ax=axes[1, 1])
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    out = a.out / f"replica_s{src['scene']:02d}_im{im_id:06d}.png"
    fig.savefig(out, dpi=100)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()