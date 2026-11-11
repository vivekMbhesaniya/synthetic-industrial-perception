import blenderproc as bproc  # must be the very first line, BlenderProc checks this

# Render a flat plane seen straight on, to check BlenderProc's depth convention.
# Runs inside Blender's Python:  blenderproc run scripts/render_plane_check.py <output_dir>
# Only blenderproc, numpy and the standard library are available here.

import argparse
import json
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 640, 480
FX = FY = 500.0
CX, CY = WIDTH / 2, HEIGHT / 2
CAMERA_HEIGHT_M = 1.0

parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
args = parser.parse_args()
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

bproc.init()

# 4 m x 4 m plane in z = 0, large enough to fill the whole view from 1 m
bproc.object.create_primitive("PLANE", size=4.0)

light = bproc.types.Light()
light.set_location([1.0, -1.0, 2.0])
light.set_energy(200)

K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
bproc.camera.set_intrinsics_from_K_matrix(K, WIDTH, HEIGHT)

# Camera 1 m above the plane with zero rotation: a Blender camera then looks along -Z
cam2world = bproc.math.build_transformation_mat([0.0, 0.0, CAMERA_HEIGHT_M], [0.0, 0.0, 0.0])
bproc.camera.add_camera_pose(cam2world)

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(64)

data = bproc.renderer.render()
bproc.writer.write_hdf5(str(out_dir), data)

meta = {
    "K": K.tolist(),
    "width": WIDTH,
    "height": HEIGHT,
    "camera_height_m": CAMERA_HEIGHT_M,
    "cam2world": np.asarray(cam2world).tolist(),
}
(out_dir / "camera.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"Wrote {out_dir / '0.hdf5'} and {out_dir / 'camera.json'}")