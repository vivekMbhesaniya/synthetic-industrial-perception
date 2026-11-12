"""Validate rendered random scenes and write annotations.json per scene. Runs in sip.

For every visible instance, rendered depth pixels are back-projected with the job's (OpenCV) K,
mapped into the model frame with the written pose, and their distance to the mesh is measured.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d

from synperc.processing.annotations import bbox_from_mask, distances_to_mesh
from synperc.processing.projection import normalized_coords

MAX_POINTS = 3000
OFF_SURFACE_MM = 2.0
PASS_MEDIAN_MM = 1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jobs_root", type=Path)
    p.add_argument("pattern")
    a = p.parse_args()

    scene_dirs = sorted(d for d in a.jobs_root.glob(a.pattern) if (d / "result.json").exists())
    if not scene_dirs:
        raise FileNotFoundError(f"No rendered scenes matching {a.pattern} in {a.jobs_root}")

    rng = np.random.default_rng(0)
    meshes: dict[str, o3d.geometry.TriangleMesh] = {}
    medians, failures, n_instances, n_visible, n_off_table = [], [], 0, 0, 0
    for d in scene_dirs:
        job = json.loads((d / "job.json").read_text(encoding="utf-8"))
        result = json.loads((d / "result.json").read_text(encoding="utf-8"))
        with h5py.File(d / "0.hdf5", "r") as f:
            depth = np.array(f["depth"], dtype=np.float64)
            seg = np.array(f["category_id_segmaps"])
        K = np.array(job["K"])
        x_n, y_n = normalized_coords(K, *depth.shape)
        mesh_of = {o["instance"]: (o["mesh"], o["mesh_unit_to_m"]) for o in job["objects"]}

        annotations, line = [], []
        for o in result["objects"]:
            n_instances += 1
            n_off_table += not o["on_table"]
            mask = seg == o["category_id"]
            n_px = int(mask.sum())
            if n_px == 0:
                line.append(f"{o['instance']}:hidden")
                continue
            n_visible += 1
            mesh_path, s = mesh_of[o["instance"]]
            if mesh_path not in meshes:
                m = o3d.io.read_triangle_mesh(mesh_path)
                m.scale(s, center=(0.0, 0.0, 0.0))  # mm -> m about the model origin
                meshes[mesh_path] = m
            v, u = np.nonzero(mask)
            if v.size > MAX_POINTS:
                keep = rng.choice(v.size, MAX_POINTS, replace=False)
                v, u = v[keep], u[keep]
            z = depth[v, u]
            pts_cam = np.column_stack((x_n[v, u] * z, y_n[v, u] * z, z))
            T = np.array(o["model2cam"])
            pts_model = (pts_cam - T[:3, 3]) @ T[:3, :3]  # R^T (p - t), row-vector form
            dist_mm = distances_to_mesh(pts_model, meshes[mesh_path]) * 1000
            med = float(np.median(dist_mm))
            off = float(np.mean(dist_mm > OFF_SURFACE_MM) * 100)
            medians.append(med)
            if med > PASS_MEDIAN_MM:
                failures.append(f"{d.name}/instance {o['instance']}: median {med:.2f} mm")
            annotations.append(
                {
                    "instance": o["instance"], "obj_id": o["obj_id"], "category_id": o["category_id"],
                    "bbox_xywh": list(bbox_from_mask(mask)), "visible_px": n_px,
                    "model2cam": o["model2cam"], "surface_median_mm": med, "off_surface_pct": off,
                }
            )
            line.append(f"{o['instance']}:{med:.3f}mm/{off:.1f}%")
        (d / "annotations.json").write_text(
            json.dumps({"scene_index": result["scene_index"], "K": job["K"], "instances": annotations}, indent=2),
            encoding="utf-8",
        )
        print(f"{d.name}: " + "  ".join(line))

    print(f"\nscenes {len(scene_dirs)}, instances {n_instances}, visible {n_visible}, fell off table {n_off_table}")
    if medians:
        print(f"surface distance median per instance: max {max(medians):.3f} mm, median {np.median(medians):.3f} mm")
    print("FAILURES:" if failures else f"all visible instances below {PASS_MEDIAN_MM} mm median")
    for f in failures:
        print("  " + f)


if __name__ == "__main__":
    main()