"""BOP loader tests on a tiny hand-made scene, built to catch specific bugs."""

import json

import cv2
import numpy as np
import pytest

from synperc.processing.bop import (
    find_image,
    list_image_ids,
    load_depth_m,
    load_scene_annotations,
    load_scene_cameras,
)

R90Z = [0, -1, 0, 1, 0, 0, 0, 0, 1]  # 90 deg about z; not symmetric, so a transposed reshape is caught


@pytest.fixture
def scene(tmp_path):
    cam = {
        "3": {"cam_K": [1000.0, 0, 300.0, 0, 900.0, 200.0, 0, 0, 1], "cam_R_w2c": R90Z,
              "cam_t_w2c": [0, 0, 500.0], "depth_scale": 0.1},
        "10": {"cam_K": [1000.0, 0, 310.0, 0, 900.0, 210.0, 0, 0, 1], "cam_R_w2c": R90Z,
               "cam_t_w2c": [0, 0, 500.0], "depth_scale": 0.1},
    }
    gt = {"3": [{"obj_id": 5, "cam_R_m2c": R90Z, "cam_t_m2c": [10.0, 20.0, 800.0]}], "10": []}
    info = {"3": [{"bbox_visib": [1, 2, 30, 40], "visib_fract": 0.75}], "10": []}
    for name, data in [("scene_camera.json", cam), ("scene_gt.json", gt), ("scene_gt_info.json", info)]:
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "depth").mkdir()
    depth = np.full((4, 5), 8000, dtype=np.uint16)
    depth[0, 0] = 0  # missing measurement
    cv2.imwrite(str(tmp_path / "depth" / "000003.png"), depth)
    return tmp_path


def test_image_ids_sorted_numerically_with_gaps(scene):
    assert list_image_ids(scene) == [3, 10]  # a string sort would give ["10", "3"]


def test_intrinsics_read_row_major_and_per_image(scene):
    cams = load_scene_cameras(scene)
    assert cams[3].K[0, 2] == 300.0 and cams[3].K[1, 2] == 200.0 and cams[3].K[1, 1] == 900.0
    assert cams[10].K[0, 2] == 310.0  # principal point differs between images


def test_pose_in_metres_and_rotation_not_transposed(scene):
    cam = load_scene_cameras(scene)[3]
    assert np.allclose(cam.world2cam[:3, 3], [0.0, 0.0, 0.5])
    assert cam.world2cam[0, 1] == -1.0  # row-major: first row is (0, -1, 0)


def test_annotations_merge_gt_and_info(scene):
    anns = load_scene_annotations(scene)
    (ann,) = anns[3]
    assert ann.obj_id == 5 and ann.gt_index == 0
    assert np.allclose(ann.model2cam[:3, 3], [0.01, 0.02, 0.8])
    assert ann.bbox_visib == (1, 2, 30, 40) and ann.visib_fract == 0.75
    assert anns[10] == []


def test_depth_in_metres_and_missing_stays_zero(scene):
    depth = load_depth_m(find_image(scene / "depth", "000003"), depth_scale_mm=0.1)
    assert depth[1, 1] == pytest.approx(0.8)  # 8000 * 0.1 mm = 800 mm
    assert depth[0, 0] == 0.0


def test_mismatched_gt_and_info_raise(scene):
    (scene / "scene_gt_info.json").write_text(json.dumps({"3": [], "10": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scene_annotations(scene)