"""Compare grey-level statistics of real T-LESS frames and synthetic scenes. Runs in sip."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from synperc.processing.bop import find_image, list_image_ids, load_mask, load_rgb, load_scene_annotations


def grey(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def stats(values: np.ndarray) -> str:
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return f"mean {values.mean():6.1f}   p5 {p5:6.1f}   median {p50:6.1f}   p95 {p95:6.1f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("synthetic_root", type=Path)
    p.add_argument("pattern")
    p.add_argument("--scene", type=int, default=1)
    a = p.parse_args()

    scene_dir = a.dataset_root / "test_primesense" / f"{a.scene:06d}"
    anns = load_scene_annotations(scene_dir)
    real_parts, real_bg = [], []
    for im_id in list_image_ids(scene_dir):
        g = grey(load_rgb(find_image(scene_dir / "rgb", f"{im_id:06d}")))
        parts = np.zeros(g.shape, bool)
        for ann in anns[im_id]:
            parts |= load_mask(find_image(scene_dir / "mask_visib", f"{im_id:06d}_{ann.gt_index:06d}"))
        real_parts.append(g[parts])
        real_bg.append(g[~parts])

    syn_parts, syn_bg = [], []
    for d in sorted(x for x in a.synthetic_root.glob(a.pattern) if (x / "0.hdf5").exists()):
        with h5py.File(d / "0.hdf5", "r") as f:
            g = grey(np.array(f["colors"]))
            parts = np.array(f["category_id_segmaps"]) > 0
        syn_parts.append(g[parts])
        syn_bg.append(g[~parts])

    print(f"grey level 0-255   ({len(real_parts)} real frames, {len(syn_parts)} synthetic scenes)")
    print(f"  real parts       {stats(np.concatenate(real_parts))}")
    print(f"  synthetic parts  {stats(np.concatenate(syn_parts))}")
    print(f"  real background  {stats(np.concatenate(real_bg))}")
    print(f"  synth background {stats(np.concatenate(syn_bg))}")


if __name__ == "__main__":
    main()