# Synthetic-to-Real Data Pipeline for Industrial Perception

**Status: work in progress.** Phases 1 and 2 complete (environment, verified renderer conventions,
real-vs-rendered analysis on T-LESS). No randomized dataset or model results yet.

Research question: how can synthetic image and 3D sensor data generated in simulation
be made realistic enough to support industrial object perception models on real data?

Renderer: BlenderProc (CPU rendering, Blender 4.2.1). Scenes are described in renderer-independent
job files; render scripts only read jobs and write images, so the renderer can be replaced.

## Phase 1: verified tools

- BlenderProc's depth output is planar depth: a plane rendered at 1.000 m reads 1.000000 m in every
  pixel. Mistaking a ray-distance image for depth in this setup (640 x 480, f = 500 px) would bend
  the plane with 63.0 mm RMS. See `results/phase1/plane_depth_check.png`.
- Open3D's voxel downsampling places its grid origin half a voxel below the minimum point.

## Phase 2: real T-LESS frames vs renders at ground-truth poses

T-LESS test scene 1 (PrimeSense sensor): 50 frames, 200 object instances, about 2.0 million pixels.
Each frame is re-rendered with its own intrinsics and the ground-truth poses, then compared pixel by pixel.

**Geometry.** Rendering with the benchmark's K placed every silhouette 0.5 px right and down
(pixel-centre convention). After shifting the principal point by -0.5 px in the Blender backend,
all 131 unoccluded instances match the benchmark's full masks exactly: IoU min 1.000, largest
centroid offset 0.03 px.

**Depth residual, real minus rendered**, on the rendered visible masks:

| Distance to mask edge | Missing depth | Median | Robust sigma | Residual > +50 mm |
|---|---|---|---|---|
| 1 px | 4.97 % | -3.65 mm | 9.68 mm | 19.48 % |
| 2 px | 5.00 % | -3.20 mm | 7.54 mm | 14.20 % |
| 3-4 px | 4.53 % | -2.59 mm | 6.17 mm | 7.69 % |
| 5-9 px | 3.57 % | -1.33 mm | 5.24 mm | 1.99 % |
| >= 10 px | 0.75 % | +0.20 mm | 3.90 mm | 0.08 % |

Findings:
- Near part boundaries the real sensor frequently reports the background: one in five boundary
  pixels lies more than 50 mm behind the part surface. These outliers dominate the standard
  deviation (81.7 mm overall vs a robust sigma of 4.58 mm), so robust statistics are reported.
- The benchmark's visible masks (`mask_visib`) come from a one-sided depth test: inside them only
  0.03 % of residuals are below -15 mm, while 98 % of the excluded pixels are. Residuals far behind
  the model are kept. Evaluating depth only inside `mask_visib` therefore hides one kind of error.
- Per-instance median offsets spread by 2.29 mm (robust), range -7.3 to +5.3 mm, consistent with
  small ground-truth pose errors. Removing them lowers the interior robust sigma from 3.90 to
  2.63 mm. That value is an upper bound on sensor noise at about 0.75 m: it still contains pose
  rotation errors and CAD-vs-part differences.

See `results/phase2/` for figures and the per-instance table.

## Reproduce

```
python -m pytest
python scripts/make_replica_job.py data/raw/tless --all
blenderproc run scripts/render_jobs.py data/synthetic "replica_s01_im*"
python scripts/scene_replica_stats.py data/synthetic "replica_s01_im*" data/raw/tless
python scripts/analyze_replica_residuals.py data/processed/scene01_replica_pixels.npz
```

Dataset source, license and file hashes: `docs/datasets.md`.