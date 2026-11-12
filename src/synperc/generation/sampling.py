"""Sample renderer-independent random scene jobs.

World frame: table plane z = 0, z up, metres. Camera poses use the OpenCV convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def look_at_opencv(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 cam2world of an OpenCV camera (x right, y down, z forward) at eye, looking at target."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = target - eye
    dist = np.linalg.norm(z)
    if dist == 0:
        raise ValueError("eye and target coincide")
    z /= dist
    x = np.cross(z, up)
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        raise ValueError("viewing direction is parallel to the up vector")
    x /= nx
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation: a normalised 4D Gaussian is a uniform unit quaternion."""
    w, x, y, z = (q := rng.normal(size=4)) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _uniform(rng: np.random.Generator, bounds) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _spherical(distance: float, elevation_rad: float, azimuth_rad: float) -> np.ndarray:
    return distance * np.array(
        [np.cos(elevation_rad) * np.cos(azimuth_rad), np.cos(elevation_rad) * np.sin(azimuth_rad), np.sin(elevation_rad)]
    )


def sample_scene(cfg: dict, scene_index: int, base_seed: int) -> dict:
    """One random scene job. The same (cfg, scene_index, base_seed) always gives the same job."""
    rng = np.random.default_rng([base_seed, scene_index])
    models_dir = Path(cfg["dataset_root"]) / cfg["models"]
    lo, hi = cfg["objects_per_scene"]
    obj_ids = rng.choice(cfg["objects"], size=int(rng.integers(lo, hi + 1)), replace=True)

    half = cfg["placement"]["area_m"] / 2
    objects = []
    for i, oid in enumerate(obj_ids):
        objects.append(
            {
                "instance": i,
                "obj_id": int(oid),
                "mesh": (models_dir / f"obj_{int(oid):06d}.ply").resolve().as_posix(),
                "mesh_unit_to_m": cfg["mesh_unit_to_m"],
                "initial_position_m": [
                    _uniform(rng, (-half, half)),
                    _uniform(rng, (-half, half)),
                    _uniform(rng, cfg["placement"]["drop_height_m"]) + i * cfg["placement"].get("stack_spacing_m", 0.0),
                ],
                "initial_rotation": random_rotation(rng).tolist(),
                "material": {k: _uniform(rng, v) for k, v in cfg["object_material"].items()},
            }
        )

    cam = cfg["camera"]
    distance = _uniform(rng, cam["distance_m"])
    elevation = _uniform(rng, cam["elevation_deg"])
    azimuth = _uniform(rng, cam["azimuth_deg"])
    jitter = cam["target_jitter_m"]
    target = np.array([_uniform(rng, (-jitter, jitter)), _uniform(rng, (-jitter, jitter)), 0.0])
    eye = target + _spherical(distance, np.radians(elevation), np.radians(azimuth))

    lc = cfg["lights"]
    lights = []
    for _ in range(int(rng.integers(lc["count"][0], lc["count"][1] + 1))):
        pos = _spherical(
            _uniform(rng, lc["distance_m"]), np.radians(_uniform(rng, lc["elevation_deg"])), np.radians(_uniform(rng, (0, 360)))
        )
        lights.append({"position_m": pos.tolist(), "energy_w": _uniform(rng, lc["energy_w"])})

    intr = cfg["intrinsics"]
    return {
        "schema_version": cfg["schema_version"],
        "type": "random_scene",
        "scene_index": scene_index,
        "seed": [base_seed, scene_index],
        "camera_frame": "opencv",
        "width": cfg["image"]["width"],
        "height": cfg["image"]["height"],
        "K": [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        "cam2world": look_at_opencv(eye, target).tolist(),
        "camera_params": {"distance_m": distance, "elevation_deg": elevation, "azimuth_deg": azimuth, "target_m": target.tolist()},
        "objects": objects,
        "lights": lights,
        "table_material": {k: _uniform(rng, v) for k, v in cfg["table_material"].items()},
        "world_background_strength": _uniform(rng, cfg["world_background_strength"]) if "world_background_strength" in cfg else None,
    }