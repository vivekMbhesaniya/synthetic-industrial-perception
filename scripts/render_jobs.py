import blenderproc as bproc  # must be the very first line, BlenderProc checks this

# Render many job files in one Blender session.
# Runs inside Blender's Python:  blenderproc run scripts/render_jobs.py <jobs_root> <folder_pattern>
# Only blenderproc, numpy and the standard library are available here.

import argparse
import json
from pathlib import Path

import numpy as np

# Pixel-centre convention, measured on T-LESS scene 1 frame 1: see the comment in render_job.py
PIXEL_CENTRE_SHIFT = -0.5

parser = argparse.ArgumentParser()
parser.add_argument("jobs_root")
parser.add_argument("pattern", help='folder pattern, e.g. "replica_s01_im*"')
args = parser.parse_args()

job_dirs = sorted(d for d in Path(args.jobs_root).glob(args.pattern) if (d / "job.json").exists())
if not job_dirs:
    raise FileNotFoundError(f"No job folders matching {args.pattern} in {args.jobs_root}")
jobs = [json.loads((d / "job.json").read_text(encoding="utf-8")) for d in job_dirs]

# Static scene: every job must contain the same objects, so meshes are loaded once and re-posed.
signature = [(o["gt_index"], o["obj_id"], o["mesh"]) for o in jobs[0]["objects"]]
for d, job in zip(job_dirs, jobs):
    if job["camera_frame"] != "opencv":
        raise ValueError(f"{d}: unsupported camera_frame {job['camera_frame']}")
    if [(o["gt_index"], o["obj_id"], o["mesh"]) for o in job["objects"]] != signature:
        raise ValueError(f"{d}: object list differs from {job_dirs[0]}")

bproc.init()
meshes = []
for o in jobs[0]["objects"]:
    loaded = bproc.loader.load_obj(o["mesh"])
    if len(loaded) != 1:
        raise RuntimeError(f"Expected one mesh in {o['mesh']}, got {len(loaded)}")
    loaded[0].set_cp("category_id", o["gt_index"] + 1)  # 0 = background
    meshes.append(loaded[0])

# World frame = OpenCV camera frame; Blender camera looks along -z with y up
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(np.eye(4), ["X", "-Y", "-Z"])
light = bproc.types.Light()
light.set_location([0.0, 0.0, 0.0])
light.set_energy(50)
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
bproc.renderer.set_max_amount_of_samples(64)

for i, (d, job) in enumerate(zip(job_dirs, jobs), start=1):
    for mesh, o in zip(meshes, job["objects"]):
        s = o["mesh_unit_to_m"]
        mesh.set_local2world_mat(np.array(o["model2cam"]) @ np.diag([s, s, s, 1.0]))
    K = np.array(job["K"], dtype=np.float64)
    K[0, 2] += PIXEL_CENTRE_SHIFT
    K[1, 2] += PIXEL_CENTRE_SHIFT
    bproc.camera.set_intrinsics_from_K_matrix(K, job["width"], job["height"])
    bproc.utility.reset_keyframes()
    bproc.camera.add_camera_pose(cam2world)
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(str(d), data)
    print(f"[{i}/{len(job_dirs)}] rendered {d.name}")