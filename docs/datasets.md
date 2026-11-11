# Datasets

## T-LESS (BOP format)

Real RGB-D test images and CAD models of 30 texture-less industrial parts.

- **Source:** https://huggingface.co/datasets/bop-benchmark/tless (BOP benchmark mirror)
- **License:** the Hugging Face dataset card lists CC BY 4.0. Use requires attribution.
- **Not in this repository:** the data is downloaded to `data/raw/tless/`, which Git ignores.

### Files used

| File | Bytes | SHA-256 |
|---|---|---|
| `tless_base.zip` | 49597 | `DD70CA884B7C471A530A952F70C5AB2C212F3D2C2F371BE86397442B97D70A7E` |
| `tless_models.zip` | 33468762 | `6A29C59766B8D2AF05E62E71739F0CA7243AD81BC3C7F9A24925504A8CB37928` |
| `tless_test_primesense_bop19.zip` | 825276992 | `1A18F6BBFB5AC4CED8529F7A35225ADFED88C0F62EF38067933E2B541EF1D00B` |

### Local layout

```
data/raw/tless/
  models_cad/        CAD models (rendered in simulation)
  models_eval/       models used for evaluation
  models_reconst/    3D reconstructions from real data
  test_primesense/   20 real test scenes: rgb/, depth/, mask/, mask_visib/ + JSON annotations
```

Note: in T-LESS the principal point differs between images, so intrinsics are read per image
from each scene's `scene_camera.json`, never assumed global.

### Citations

- T. Hodan, P. Haluza, S. Obdrzalek, J. Matas, M. Lourakis, X. Zabulis.
  *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects.* WACV 2017.
- T. Hodan et al. *BOP: Benchmark for 6D Object Pose Estimation.* ECCV 2018.