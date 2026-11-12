"""Replica vs real statistics over all frames of a scene. Runs in sip.

Geometry: rendered visible masks vs real full masks, only for unoccluded instances.
Depth: real - rendered residuals on the rendered visible mask, which is not filtered by the
benchmark's depth-based visibility test. The mask_visib subset is reported to show that bias.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synperc.evaluation.compare import depth_residual_stats, mask_centroid, mask_iou
from synperc.evaluation.edges import distance_to_mask_edge
from synperc.processing.bop import find_image, load_depth_m, load_mask, load_scene_cameras

SEG_KEY = "category_id_segmaps"
UNOCCLUDED_RATIO = 0.999  # rendered visible pixels / full silhouette pixels
EDGE_BINS = [(0.0, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 4.5, "3-4"), (4.5, 9.5, "5-9"), (9.5, np.inf, ">=10")]
NAN = float("nan")


def robust_sigma(r: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else NAN


def summarize(res_mm: np.ndarray, missing: np.ndarray) -> dict:
    """res_mm: residuals of valid pixels in mm; missing: one flag per region pixel."""
    return {
        "pixels": int(missing.size),
        "missing_pct": float(100 * missing.mean()) if missing.size else NAN,
        "median_mm": float(np.median(res_mm)) if res_mm.size else NAN,
        "robust_sigma_mm": robust_sigma(res_mm),
        "std_mm": float(res_mm.std()) if res_mm.size else NAN,
        "p95_abs_mm": float(np.percentile(np.abs(res_mm), 95)) if res_mm.size else NAN,
    }


def print_row(label: str, s: dict) -> None:
    print(
        f"  {label:<26} {s['pixels']:>9}  {s['missing_pct']:6.2f}%  {s['median_mm']:+8.2f}"
        f"  {s['robust_sigma_mm']:8.2f}  {s['std_mm']:8.2f}  {s['p95_abs_mm']:8.2f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jobs_root", type=Path)
    p.add_argument("pattern")
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--name", default="scene01")
    p.add_argument("--out", type=Path, default=Path("results/phase2"))
    p.add_argument("--samples-out", type=Path, default=Path("data/processed"))
    a = p.parse_args()

    job_dirs = sorted(d for d in a.jobs_root.glob(a.pattern) if (d / "0.hdf5").exists())
    if not job_dirs:
        raise FileNotFoundError(f"No rendered jobs matching {a.pattern} in {a.jobs_root}")

    rows = []
    keys = ("residual_m", "missing", "rendered_z_m", "edge_px", "in_visib", "obj_id", "image")
    pix = {k: [] for k in keys}
    for d in job_dirs:
        job = json.loads((d / "job.json").read_text(encoding="utf-8"))
        src = job["source"]
        scene_dir = a.dataset_root / src["split"] / f"{src['scene']:06d}"
        im_id = src["image"]
        cam = load_scene_cameras(scene_dir)[im_id]
        real = load_depth_m(find_image(scene_dir / "depth", f"{im_id:06d}"), cam.depth_scale_mm)
        with h5py.File(d / "0.hdf5", "r") as f:
            ren_depth = np.array(f["depth"], dtype=np.float64)
            seg = np.array(f[SEG_KEY])

        for o in job["objects"]:
            k = o["gt_index"]
            ren = seg == k + 1
            full = load_mask(find_image(scene_dir / "mask", f"{im_id:06d}_{k:06d}"))
            visib = load_mask(find_image(scene_dir / "mask_visib", f"{im_id:06d}_{k:06d}"))
            n_ren, n_full = int(ren.sum()), int(full.sum())
            row = {
                "image": im_id, "gt_index": k, "obj_id": o["obj_id"],
                "rendered_px": n_ren, "full_px": n_full, "visib_px": int(visib.sum()),
                "unoccluded": bool(n_full > 0 and n_ren / n_full >= UNOCCLUDED_RATIO),
                "iou_full": mask_iou(full, ren), "iou_visib": mask_iou(visib, ren),
            }
            if n_ren and n_full:
                (uf, vf), (ur, vr) = mask_centroid(full), mask_centroid(ren)
                row.update(du_full=uf - ur, dv_full=vf - vr)
            if n_ren:
                s = depth_residual_stats(real, ren_depth, ren)
                row.update(
                    missing_pct=100 * s["missing_frac"], median_mm=1000 * s["median"],
                    robust_sigma_mm=1000 * s["robust_sigma"], std_mm=1000 * s["std"],
                )
                miss = real[ren] <= 0
                pix["residual_m"].append(np.where(miss, np.nan, real[ren] - ren_depth[ren]).astype(np.float32))
                pix["missing"].append(miss)
                pix["rendered_z_m"].append(ren_depth[ren].astype(np.float32))
                pix["edge_px"].append(distance_to_mask_edge(ren)[ren].astype(np.float32))
                pix["in_visib"].append(visib[ren])
                pix["obj_id"].append(np.full(n_ren, o["obj_id"], np.int16))
                pix["image"].append(np.full(n_ren, im_id, np.int32))
            rows.append(row)

    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    csv_path = a.out / f"{a.name}_replica_per_object.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    P = {k: np.concatenate(v) for k, v in pix.items()}
    a.samples_out.mkdir(parents=True, exist_ok=True)
    npz_path = a.samples_out / f"{a.name}_replica_pixels.npz"
    np.savez_compressed(npz_path, **P)

    unocc = df[df["unoccluded"]]
    print(f"frames: {len(job_dirs)}, object instances: {len(df)}, unoccluded: {len(unocc)}, "
          f"not visible at all: {int((df['rendered_px'] == 0).sum())}")
    if len(unocc):
        print(f"geometry, unoccluded instances: full-mask IoU min {unocc['iou_full'].min():.3f}, "
              f"median {unocc['iou_full'].median():.3f}; max |du| {unocc['du_full'].abs().max():.2f} px, "
              f"max |dv| {unocc['dv_full'].abs().max():.2f} px")

    valid = ~P["missing"]
    res_mm = P["residual_m"] * 1000
    print("\n  depth residual real - rendered      pixels  missing    median  rob.sigma      std   p95|r|")
    print_row("all rendered-mask pixels", summarize(res_mm[valid], P["missing"]))
    v = P["in_visib"]
    print_row("only inside BOP mask_visib", summarize(res_mm[valid & v], P["missing"][v]))
    print("\n  by distance to edge [px]")
    edge_rows = []
    for lo, hi, label in EDGE_BINS:
        sel = (P["edge_px"] > lo) & (P["edge_px"] <= hi)
        s = summarize(res_mm[sel & valid], P["missing"][sel])
        edge_rows.append((label, s))
        print_row(f"edge {label}", s)
    print("\n  by object")
    for obj in sorted(np.unique(P["obj_id"])):
        sel = P["obj_id"] == obj
        print_row(f"obj {obj}", summarize(res_mm[sel & valid], P["missing"][sel]))

    labels = [lab for lab, _ in edge_rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(labels, [s["robust_sigma_mm"] for _, s in edge_rows], "o-", label="robust sigma")
    ax1.plot(labels, [s["std_mm"] for _, s in edge_rows], "s--", label="std")
    ax1.set_xlabel("distance to mask edge [px]")
    ax1.set_ylabel("depth residual spread [mm]")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.bar(labels, [s["missing_pct"] for _, s in edge_rows])
    ax2.set_xlabel("distance to mask edge [px]")
    ax2.set_ylabel("missing real depth [%]")
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle(f"T-LESS {a.name}: real vs rendered depth, {len(job_dirs)} frames")
    fig.tight_layout()
    fig_path = a.out / f"{a.name}_edge_profile.png"
    fig.savefig(fig_path, dpi=110)
    print(f"\nsaved: {csv_path}, {fig_path}, {npz_path}")


if __name__ == "__main__":
    main()