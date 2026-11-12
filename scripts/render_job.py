import blenderproc as bproc  # must be the very first line, BlenderProc checks this

# Render a job file written by make_replica_job.py.
# Runs inside Blender's Python:  blenderproc run scripts/render_job.py <job_dir>
# Only blenderproc, numpy and the standard library are available here.

import argparse
import json
from pathlib import Path

import numpy as np

# Pixel-centre convention: OpenCV/BOP put pixel centres at integer coordinates. Rendering with
# the unchanged K placed every part +0.5 px right and down of the full real masks (measured on
# T-LESS scene 1, frame 1, all four parts). Shifting the principal point by -0.5 px compensates.
PIXEL_CENTRE_SHIFT = -0.5

parser = argparse.ArgumentParser()
parser.add_argument("job_dir")
args = parser.parse_args()
job_dir = Path(args.job_dir)
job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
if job["camera_frame"] != "opencv":
    raise ValueError(f"Unsupported camera_frame: {job['camera_frame']}")

bproc.init()

for o in job["objects"]:
    loaded = bproc.loader.load_obj(o["mesh"])
    if len(loaded) != 1:
        raise RuntimeError(f"Expected one mesh in {o['mesh']}, got {len(loaded)}")
    obj = loaded[0]
    s = o["mesh_unit_to_m"]
    # Fold the mm -> m scale into the pose: a model point X lands at R (s X) + t
    obj.set_local2world_mat(np.array(o["model2cam"]) @ np.diag([s, s, s, 1.0]))
    obj.set_cp("category_id", o["gt_index"] + 1)  # 0 = background

# World frame = OpenCV camera frame, so the camera sits at the origin.
# A Blender camera looks along -z with y up: flip y and z.
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(np.eye(4), ["X", "-Y", "-Z"])
K_blender = np.array(job["K"], dtype=np.float64)
K_blender[0, 2] += PIXEL_CENTRE_SHIFT
K_blender[1, 2] += PIXEL_CENTRE_SHIFT
bproc.camera.set_intrinsics_from_K_matrix(K_blender, job["width"], job["height"])
bproc.camera.add_camera_pose(cam2world)

light = bproc.types.Light()
light.set_location([0.0, 0.0, 0.0])  # light at the camera, like a sensor's own illumination
light.set_energy(50)

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
bproc.renderer.set_max_amount_of_samples(64)

data = bproc.renderer.render()
bproc.writer.write_hdf5(str(job_dir), data)
print(f"Rendered {len(job['objects'])} objects with principal-point shift {PIXEL_CENTRE_SHIFT} px")
print(f"Wrote {job_dir / '0.hdf5'}")