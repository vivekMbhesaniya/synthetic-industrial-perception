import blenderproc as bproc  # must be the very first line, BlenderProc checks this

# Build, simulate and render random scene jobs (from make_random_jobs.py) in one Blender session.
# Runs inside Blender's Python:  blenderproc run scripts/render_random_scenes.py <jobs_root> <folder_pattern>
# Writes 0.hdf5 (colors, depth, segmentation) and result.json (final poses) into each job folder.

import argparse
import json
from pathlib import Path

import numpy as np

PIXEL_CENTRE_SHIFT = -0.5  # OpenCV/BOP pixel-centre convention, measured in Phase 2
OPENCV_TO_BLENDER_CAMERA = ["X", "-Y", "-Z"]

parser = argparse.ArgumentParser()
parser.add_argument("jobs_root")
parser.add_argument("pattern", help='folder pattern, e.g. "scene_*"')
args = parser.parse_args()

job_dirs = sorted(d for d in Path(args.jobs_root).glob(args.pattern) if (d / "job.json").exists())
if not job_dirs:
    raise FileNotFoundError(f"No job folders matching {args.pattern} in {args.jobs_root}")

bproc.init()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
bproc.renderer.set_max_amount_of_samples(64)


def grey_material(name, grey, roughness, metallic=0.0):
    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", [grey, grey, grey, 1.0])
    mat.set_principled_shader_value("Roughness", roughness)
    mat.set_principled_shader_value("Metallic", metallic)
    return mat


for i, d in enumerate(job_dirs, start=1):
    job = json.loads((d / "job.json").read_text(encoding="utf-8"))
    if job.get("type") != "random_scene" or job["camera_frame"] != "opencv":
        raise ValueError(f"{d}: not an OpenCV random_scene job")

    bproc.clean_up(clean_up_camera=False)
    bproc.utility.reset_keyframes()

    # Table: a passive rigid body in the plane z = 0
    table = bproc.object.create_primitive("PLANE", size=2.0)
    table.set_cp("category_id", 0)
    tm = job["table_material"]
    table.replace_materials(grey_material("table", tm["gray"], tm["roughness"]))
    table.enable_rigidbody(active=False, collision_shape="BOX", friction=100.0)

    parts = []
    for o in job["objects"]:
        loaded = bproc.loader.load_obj(o["mesh"])
        if len(loaded) != 1:
            raise RuntimeError(f"Expected one mesh in {o['mesh']}, got {len(loaded)}")
        obj = loaded[0]
        s = o["mesh_unit_to_m"]
        obj.set_scale([s, s, s])
        obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)  # mesh now in metres
        start = np.eye(4)
        start[:3, :3] = np.array(o["initial_rotation"])
        start[:3, 3] = o["initial_position_m"]
        obj.set_local2world_mat(start)
        obj.set_cp("category_id", o["instance"] + 1)
        m = o["material"]
        obj.replace_materials(grey_material(f"part_{o['instance']}", m["gray"], m["roughness"], m["metallic"]))
        obj.enable_rigidbody(active=True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        parts.append((o, obj))

    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=10, check_object_interval=1)

    for k, lj in enumerate(job["lights"]):
        light = bproc.types.Light()
        light.set_location(lj["position_m"])
        light.set_energy(lj["energy_w"])

    K = np.array(job["K"], dtype=np.float64)
    K[0, 2] += PIXEL_CENTRE_SHIFT
    K[1, 2] += PIXEL_CENTRE_SHIFT
    bproc.camera.set_intrinsics_from_K_matrix(K, job["width"], job["height"])
    cam2world_cv = np.array(job["cam2world"], dtype=np.float64)
    bproc.camera.add_camera_pose(
        bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world_cv, OPENCV_TO_BLENDER_CAMERA)
    )

    data = bproc.renderer.render()
    for key in ("colors", "depth", "category_id_segmaps"):
        if key not in data:
            raise KeyError(f"{d}: render output '{key}' missing, got {sorted(data.keys())}")
    bproc.writer.write_hdf5(str(d), data)

    world2cam_cv = np.linalg.inv(cam2world_cv)
    result = {"scene_index": job["scene_index"], "camera_frame": "opencv", "units": "m", "objects": []}
    for o, obj in parts:
        model2world = np.array(obj.get_local2world_mat(), dtype=np.float64)
        result["objects"].append(
            {
                "instance": o["instance"],
                "obj_id": o["obj_id"],
                "category_id": o["instance"] + 1,
                "model2cam": (world2cam_cv @ model2world).tolist(),
                "model2world": model2world.tolist(),
                "on_table": bool(model2world[2, 3] > -0.01),
            }
        )
    (d / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{i}/{len(job_dirs)}] {d.name}: {len(parts)} parts, {len(job['lights'])} lights")