"""Write renderer-independent job files that replicate real BOP frames. Runs in sip."""

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
    which = p.add_mutually_exclusive_group()
    which.add_argument("--image", type=int, default=None, help="one image id (default: the first)")
    which.add_argument("--all", action="store_true", help="every image of the scene")
    p.add_argument("--models", default="models_cad")
    p.add_argument("--out-root", type=Path, default=Path("data/synthetic"))
    a = p.parse_args()

    scene_dir = a.dataset_root / a.split / f"{a.scene:06d}"
    ids = list_image_ids(scene_dir)
    im_ids = ids if a.all else ([a.image] if a.image is not None else ids[:1])
    cams = load_scene_cameras(scene_dir)
    anns_all = load_scene_annotations(scene_dir)

    mesh_paths: dict[int, Path] = {}
    for im_id in im_ids:
        cam = cams[im_id]
        height, width = load_depth_m(find_image(scene_dir / "depth", f"{im_id:06d}"), cam.depth_scale_mm).shape
        objects = []
        for ann in anns_all[im_id]:
            if ann.obj_id not in mesh_paths:
                path = (a.dataset_root / a.models / f"obj_{ann.obj_id:06d}.ply").resolve()
                if not path.exists():
                    raise FileNotFoundError(path)
                mesh = o3d.io.read_triangle_mesh(str(path))
                extent = mesh.get_axis_aligned_bounding_box().get_extent()
                print(f"obj {ann.obj_id:2d}: {len(mesh.triangles):6d} triangles, bbox extent {np.round(extent, 1)} (file units)")
                mesh_paths[ann.obj_id] = path
            objects.append(
                {
                    "gt_index": ann.gt_index,
                    "obj_id": ann.obj_id,
                    "mesh": mesh_paths[ann.obj_id].as_posix(),
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
    print(f"wrote {len(im_ids)} job(s) for scene {a.scene} into {a.out_root}")


if __name__ == "__main__":
    main()