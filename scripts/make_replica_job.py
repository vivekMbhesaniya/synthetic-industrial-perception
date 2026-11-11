"""Write a renderer-independent job file that replicates one real BOP frame. Runs in sip."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from synperc.processing.bop import (
    find_image,
    list_image_ids,
    load_depth_m,
    load_scene_annotations,
    load_scene_cameras,
)

MESH_UNIT_TO_M = 0.001  # BOP meshes are in millimetres; verified by the printed extents


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--split", default="test_primesense")
    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--image", type=int, default=None)
    p.add_argument("--models", default="models_cad")
    p.add_argument("--out-root", type=Path, default=Path("data/synthetic"))
    a = p.parse_args()

    scene_dir = a.dataset_root / a.split / f"{a.scene:06d}"
    im_id = list_image_ids(scene_dir)[0] if a.image is None else a.image
    cam = load_scene_cameras(scene_dir)[im_id]
    anns = load_scene_annotations(scene_dir)[im_id]
    height, width = load_depth_m(find_image(scene_dir / "depth", f"{im_id:06d}"), cam.depth_scale_mm).shape

    objects = []
    for ann in anns:
        mesh_path = (a.dataset_root / a.models / f"obj_{ann.obj_id:06d}.ply").resolve()
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        extent = mesh.get_axis_aligned_bounding_box().get_extent()
        print(f"obj {ann.obj_id:2d}: {len(mesh.triangles):6d} triangles, bbox extent {np.round(extent, 1)} (file units)")
        objects.append(
            {
                "gt_index": ann.gt_index,
                "obj_id": ann.obj_id,
                "mesh": mesh_path.as_posix(),
                "mesh_unit_to_m": MESH_UNIT_TO_M,
                "model2cam": ann.model2cam.tolist(),
            }
        )

    job = {
        "camera_frame": "opencv",
        "width": width,
        "height": height,
        "K": cam.K.tolist(),
        "source": {"dataset": a.dataset_root.name, "split": a.split, "scene": a.scene, "image": im_id},
        "objects": objects,
    }
    out_dir = a.out_root / f"replica_s{a.scene:02d}_im{im_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    print(f"image {width} x {height}, {len(objects)} objects")
    print(f"wrote {out_dir / 'job.json'}")


if __name__ == "__main__":
    main()