"""Loader for BOP-format datasets such as T-LESS.

Project conventions: lengths in metres, 4x4 homogeneous transforms, OpenCV camera frame.
BOP stores lengths in millimetres; the conversion happens only in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MM_TO_M = 1e-3


@dataclass(frozen=True)
class FrameCamera:
    K: np.ndarray  # 3x3 intrinsics, pixels
    world2cam: np.ndarray | None  # 4x4 rigid transform, metres
    depth_scale_mm: float  # stored depth value * depth_scale_mm = depth in mm


@dataclass(frozen=True)
class ObjectAnnotation:
    gt_index: int  # position in scene_gt.json, used in mask file names
    obj_id: int
    model2cam: np.ndarray  # 4x4 rigid transform, metres
    bbox_visib: tuple[int, int, int, int] | None  # x, y, width, height in pixels
    visib_fract: float | None


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rigid_transform(R_flat, t_mm) -> np.ndarray:
    """4x4 transform from a row-major 3x3 rotation and a translation in millimetres."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_flat, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t_mm, dtype=np.float64).reshape(3) * MM_TO_M
    return T


def list_image_ids(scene_dir: Path) -> list[int]:
    """Image IDs present in a scene, sorted numerically. IDs can have gaps."""
    return sorted(int(k) for k in _read_json(Path(scene_dir) / "scene_camera.json"))


def load_scene_cameras(scene_dir: Path) -> dict[int, FrameCamera]:
    raw = _read_json(Path(scene_dir) / "scene_camera.json")
    cameras = {}
    for key, c in raw.items():
        K = np.asarray(c["cam_K"], dtype=np.float64).reshape(3, 3)
        world2cam = rigid_transform(c["cam_R_w2c"], c["cam_t_w2c"]) if "cam_R_w2c" in c else None
        cameras[int(key)] = FrameCamera(K=K, world2cam=world2cam, depth_scale_mm=float(c["depth_scale"]))
    return cameras


def load_scene_annotations(scene_dir: Path) -> dict[int, list[ObjectAnnotation]]:
    scene_dir = Path(scene_dir)
    gt = _read_json(scene_dir / "scene_gt.json")
    info_path = scene_dir / "scene_gt_info.json"
    info = _read_json(info_path) if info_path.exists() else {}
    annotations = {}
    for key, objects in gt.items():
        if key in info and len(info[key]) != len(objects):
            raise ValueError(f"Image {key}: {len(objects)} poses but {len(info[key])} info entries")
        infos = info.get(key, [{}] * len(objects))
        anns = []
        for i, (o, inf) in enumerate(zip(objects, infos)):
            bbox = tuple(int(v) for v in inf["bbox_visib"]) if "bbox_visib" in inf else None
            anns.append(
                ObjectAnnotation(
                    gt_index=i,
                    obj_id=int(o["obj_id"]),
                    model2cam=rigid_transform(o["cam_R_m2c"], o["cam_t_m2c"]),
                    bbox_visib=bbox,
                    visib_fract=inf.get("visib_fract"),
                )
            )
        annotations[int(key)] = anns
    return annotations


def find_image(folder: Path, stem: str) -> Path:
    """Return the single file named <stem>.<any extension> in folder."""
    matches = sorted(Path(folder).glob(f"{stem}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one file '{stem}.*' in {folder}, found {len(matches)}")
    return matches[0]


def load_depth_m(path: Path, depth_scale_mm: float) -> np.ndarray:
    """16-bit BOP depth image -> depth in metres. Missing measurements stay 0."""
    raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise FileNotFoundError(f"Could not read depth image {path}")
    if raw.dtype != np.uint16:
        raise ValueError(f"Expected a 16-bit depth image, got {raw.dtype}: {path}")
    return raw.astype(np.float64) * depth_scale_mm * MM_TO_M


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask {path}")
    return mask > 0