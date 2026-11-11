"""Check BlenderProc's depth convention on the rendered plane. Runs in the sip environment."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synperc.processing.projection import backproject_depth, depth_to_distance, fit_plane


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("--fig", type=Path, default=Path("results/phase1/plane_depth_check.png"))
    args = parser.parse_args()

    meta = json.loads((args.render_dir / "camera.json").read_text(encoding="utf-8"))
    K = np.array(meta["K"])
    true_depth = meta["camera_height_m"]

    with h5py.File(args.render_dir / "0.hdf5", "r") as f:
        print("datasets in file:", list(f.keys()))
        depth = np.array(f["depth"], dtype=np.float64)
        rgb = np.array(f["colors"])

    h, w = depth.shape
    print(f"depth image  : shape={depth.shape}, min={depth.min():.6f} m, max={depth.max():.6f} m")
    print(f"centre pixel : {depth[h // 2, w // 2]:.6f} m")
    print(f"corner pixel : {depth[0, 0]:.6f} m")
    max_err = float(np.abs(depth - true_depth).max())
    print(f"max |depth - {true_depth} m| = {max_err:.2e} m")

    _, rms_as_depth = fit_plane(backproject_depth(depth, K))
    _, rms_if_distance = fit_plane(backproject_depth(depth_to_distance(depth, K), K))
    print(f"plane-fit RMS, image used as planar depth        : {rms_as_depth * 1000:.4f} mm")
    print(f"plane-fit RMS, if a ray-distance image were used : {rms_if_distance * 1000:.1f} mm")
    print("verdict:", "PLANAR DEPTH" if max_err < 1e-3 else "NOT CONSTANT, investigate the convention")

    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    im1 = axes[1].imshow(depth)
    axes[1].set_title("Rendered depth [m]")
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(depth_to_distance(depth, K))
    axes[2].set_title("Ray distance computed from intrinsics [m]")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.fig, dpi=120)
    print(f"saved figure : {args.fig}")


if __name__ == "__main__":
    main()