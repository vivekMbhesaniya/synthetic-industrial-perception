"""Scene sampling: reproducibility, camera geometry, rotations and parameter ranges."""

import numpy as np
import pytest

from synperc.generation.sampling import look_at_opencv, random_rotation, sample_scene

CFG = {
    "schema_version": 1, "dataset_root": "fake", "models": "models_cad", "mesh_unit_to_m": 0.001,
    "objects": [2, 25, 29, 30], "objects_per_scene": [3, 6], "image": {"width": 720, "height": 540},
    "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 360.0, "cy": 270.0},
    "camera": {"distance_m": [0.65, 0.9], "elevation_deg": [45, 85], "azimuth_deg": [0, 360], "target_jitter_m": 0.05},
    "placement": {"area_m": 0.3, "drop_height_m": [0.05, 0.25]},
    "lights": {"count": [1, 3], "energy_w": [20, 150], "distance_m": [0.5, 1.5], "elevation_deg": [20, 85]},
    "object_material": {"gray": [0.2, 0.9], "roughness": [0.1, 0.9], "metallic": [0.0, 0.5]},
    "table_material": {"gray": [0.05, 0.6], "roughness": [0.3, 1.0]},
}


def test_same_seed_and_index_give_identical_scene():
    assert sample_scene(CFG, 17, 2026) == sample_scene(CFG, 17, 2026)


def test_different_indices_give_different_scenes():
    assert sample_scene(CFG, 0, 2026)["cam2world"] != sample_scene(CFG, 1, 2026)["cam2world"]


def test_look_at_projects_target_to_principal_point():
    T = look_at_opencv([0.5, -0.3, 0.6], [0.0, 0.1, 0.0])
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    p = np.linalg.inv(T) @ np.array([0.0, 0.1, 0.0, 1.0])  # target in camera coordinates
    assert np.allclose(p[:2], 0.0)  # on the optical axis
    assert p[2] == pytest.approx(np.linalg.norm([0.5, -0.4, 0.6]))  # in front, at the right distance


def test_look_at_image_y_points_down_in_the_world():
    T = look_at_opencv([1.0, 0.0, 0.5], [0.0, 0.0, 0.0])
    assert T[2, 1] < 0  # OpenCV y axis has a negative world z component


def test_random_rotations_are_proper():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = random_rotation(rng)
        assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_sampled_values_respect_configured_ranges():
    for i in range(50):
        s = sample_scene(CFG, i, 2026)
        assert 3 <= len(s["objects"]) <= 6
        cp = s["camera_params"]
        assert 0.65 <= cp["distance_m"] <= 0.9 and 45 <= cp["elevation_deg"] <= 85
        eye = np.array(s["cam2world"])[:3, 3]
        assert np.linalg.norm(eye - np.array(cp["target_m"])) == pytest.approx(cp["distance_m"])
        for o in s["objects"]:
            assert o["obj_id"] in CFG["objects"]
            assert 0.2 <= o["material"]["gray"] <= 0.9 and 0.0 <= o["material"]["metallic"] <= 0.5

def test_stacked_drop_heights_and_world_strength():
    cfg = {**CFG, "placement": {"area_m": 0.3, "drop_height_m": [0.08, 0.09], "stack_spacing_m": 0.16},
           "world_background_strength": [0.0, 0.3]}
    for i in range(30):
        s = sample_scene(cfg, i, 2026)
        z = [o["initial_position_m"][2] for o in s["objects"]]
        assert all(b - a >= 0.15 for a, b in zip(z, z[1:]))  # at least 0.16 - 0.01 m apart
        assert 0.0 <= s["world_background_strength"] <= 0.3


def test_config_without_new_keys_is_unchanged():
    s = sample_scene(CFG, 0, 2026)
    assert s["world_background_strength"] is None
    assert s["objects"][0]["initial_position_m"][2] <= 0.25  # no stacking added