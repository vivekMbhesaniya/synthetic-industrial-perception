"""Smoke tests: each core library imports and performs one real operation."""

import numpy as np


def test_numpy_linear_algebra():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))
    assert np.allclose(a @ np.linalg.inv(a), np.eye(3))


def test_opencv_connected_components():
    import cv2

    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (32, 32), 10, 255, -1)
    n_labels, _ = cv2.connectedComponents(img)
    assert n_labels == 2  # background + one filled disc


def test_open3d_voxel_downsampling():
    """Open3D starts its voxel grid half a voxel below the minimum point."""
    import open3d as o3d

    pts = np.random.default_rng(0).uniform(size=(1000, 3))
    voxel = 0.2
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    down = pcd.voxel_down_sample(voxel_size=voxel)

    origin = pts.min(axis=0) - voxel / 2
    expected = len(np.unique(np.floor((pts - origin) / voxel).astype(int), axis=0))
    assert len(down.points) == expected


def test_pandas_groupby():
    import pandas as pd

    df = pd.DataFrame({"obj_id": [1, 1, 2], "area": [10.0, 20.0, 5.0]})
    assert df.groupby("obj_id")["area"].mean().loc[1] == 15.0


def test_matplotlib_headless_save(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "plot.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.stat().st_size > 0


def test_yaml_roundtrip():
    import yaml

    cfg = {"camera": {"fx": 500.0, "width": 640}}
    assert yaml.safe_load(yaml.safe_dump(cfg)) == cfg