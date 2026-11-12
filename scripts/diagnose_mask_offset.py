"""Rendered masks vs real full and visible masks for one replica: IoU, centroid offset, size. Runs in sip."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from synperc.evaluation.compare import mask_centroid, mask_iou
from synperc.processing.bop import find_image, load_mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("job_dir", type=Path)
    p.add_argument("dataset_root", type=Path)
    a = p.parse_args()

    job = json.loads((a.job_dir / "job.json").read_text(encoding="utf-8"))
    src = job["source"]
    scene_dir = a.dataset_root / src["split"] / f"{src['scene']:06d}"
    im_id = src["image"]
    with h5py.File(a.job_dir / "0.hdf5", "r") as f:
        seg = np.array(f["category_id_segmaps"])

    print("  gt  obj | full mask:    IoU   du[px]  dv[px]  pixels | visible mask: IoU   du[px]  dv[px]  pixels | rendered pixels")
    for o in job["objects"]:
        ren = seg == o["gt_index"] + 1
        us, vs = mask_centroid(ren)
        cells = []
        for kind in ("mask", "mask_visib"):
            real = load_mask(find_image(scene_dir / kind, f"{im_id:06d}_{o['gt_index']:06d}"))
            ur, vr = mask_centroid(real)
            cells.append(f"{mask_iou(real, ren):.3f}  {ur - us:+6.2f}  {vr - vs:+6.2f}  {int(real.sum()):6d}")
        print(f"  {o['gt_index']:2d}  {o['obj_id']:3d} |          {cells[0]} |          {cells[1]} | {int(ren.sum()):6d}")


if __name__ == "__main__":
    main()