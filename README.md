# Synthetic-to-Real Data Pipeline for Industrial Perception

**Status: work in progress.** Phase 1 complete (environment, tested camera geometry, verified renderer depth convention). No dataset or model results yet.

Research question: how can synthetic image and 3D sensor data generated in simulation
be made realistic enough to support industrial object perception models on real data?

Renderer: BlenderProc (CPU rendering, Blender 4.2.1). Render scripts are kept separate from
processing and evaluation code, so the renderer can be replaced.

## Verified so far

- BlenderProc's depth output is planar depth (z-buffer): a plane rendered at 1.000 m reads
  1.000000 m in every pixel. If a ray-distance image were mistaken for depth in this camera
  setup (640 x 480, f = 500 px), the plane would bend with 63.0 mm RMS.
  See `results/phase1/plane_depth_check.png`.
- Open3D's voxel downsampling places its grid origin half a voxel below the minimum point
  (checked against a NumPy reimplementation in `tests/test_libraries.py`).

## Run the checks

```
python -m pytest
blenderproc run scripts/render_plane_check.py data/synthetic/phase1_plane
python scripts/check_plane_render.py data/synthetic/phase1_plane
```