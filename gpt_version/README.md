# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering — Educational Reproduction

Clean, faithful PyTorch reproduction of Wu et al., CVPR 2024
(`arXiv:2310.08528`, official code
[`hustvl/4DGaussians`](https://github.com/hustvl/4DGaussians)) for study
purposes. It follows the official code path closely — same equations, same
conventions, same defaults — reorganized for readability, with every
non-obvious choice traced to an official file and covered by parity tests.

Reference implementation lives in `gpt_version/` only. `my_version/` is the
student's manual reimplementation space and is never touched here.

## A. Pipeline

```text
Canonical Gaussians S = {X, s_raw, r_raw, alpha_raw, C_SH}
        |
        +-- (X, t) --> HexPlane (6 planes x multires) --> f_voxel
        |                    |
        |              tiny decoder --> (dX, ds, dr)   [Eqs. 11-12]
        |                    |
        +-- X' = X + dX, s' = s_raw + ds, r' = r_raw + dr   [Eq. 9]
        |
        +-- scaling = exp(s'), rotation = normalize(r'),
            opacity = sigmoid(alpha), SH as-is
        |
        +-- faithful CUDA 3DGS rasterizer --> image   [Eqs. 4, 8]
```

One canonical Gaussian set is shared across all timestamps; a compact
spatio-temporal deformation field carries the motion. Static rendering is
the same path with deformation bypassed (coarse stage).

## B. Directory map

- `data/` — `cameras.py` (pinhole + time), readers `dnerf.py`,
  `dynerf.py`, `hypernerf.py`, `scene.py` (dispatch, NeRF++ norm, AABB).
- `gaussians/` — static 3D-GS foundation: `sh.py`, `geometry.py`,
  `gaussian_model.py` (canonical params, PLY IO), `densify.py` (population
  control + optimizer surgery), `rasterizer.py` (faithful CUDA binding).
- `deformation/` — 4D novelty: `hexplane.py`, `decoder.py`, `field.py`,
  `regularize.py`, `render_4d.py` (time-conditioned rendering).
- `training/` — `losses.py`, `schedules.py`, `optim.py` (named Adam groups),
  `trainer_static.py` / `train_static.py`, `trainer_4d.py` / `train_4d.py`
  (coarse-to-fine), `step_4d.py` (minimal 4D step), `checkpoints.py`,
  `render_static.py`.
- `evaluation/` — `metrics.py` (PSNR/SSIM/LPIPS), `evaluate.py` (CLI),
  `_lpips/` (exact official LPIPS fork).
- `configs/` — per-dataset training configs (`dnerf/dynerf/hypernerf`).
- `tests/` — CPU math tests + real-CUDA integration tests.
- `requirements.txt`, `setup_rasterizer.md` (CUDA extension policy/build).

## C. Raw vs activated parameters (read this first)

Canonical stored parameters (everything the optimizer updates):

| param | storage | activation | where applied |
|---|---|---|---|
| `xyz` | direct, `[N,3]` | identity | — |
| `scaling` | log-space, `[N,3]` | `exp` | renderer, after deformation |
| `rotation` | raw quat, `[N,4]` | `normalize` | renderer, after deformation |
| `opacity` | logits, `[N,1]` | `sigmoid` | renderer, after deformation |
| SH DC/rest | raw coeffs | identity | — |

Deformation consumes and returns RAW `xyz / scaling / rotation` (plus
passthrough opacity/SH). The rasterizer consumes deformed `xyz`,
`exp(scaling)`, normalized quaternions, `sigmoid(opacity)`, raw SH.

Historical note (do not reimplement the bug): Commit 6 originally passed
raw `_scaling`/`_rotation` straight into the CUDA kernel. Commit 14 traced
the official boundary (`gaussian_renderer/__init__.py`: `scaling_activation`
/ `rotation_activation` / `opacity_activation` AFTER deformation; the kernel's
`computeCov3D` applies no `exp`) and corrected the wrapper. Manual
reimplementations should use the corrected Commit-14 semantics from the
start.

## D. HexPlane semantics (verified)

- Six planes over `(x, y, z, t)`: `xy, xz, xt, yz, yt, zt` (fixed order).
- Per level: bilinear `grid_sample` of all six, elementwise PRODUCT;
  levels concatenated (`F = out_dim x #levels`; D-NeRF `32x2 = 64`).
- Spatial coords normalized to `[-1, 1]` via the scene AABB
  (`set_aabb(xyz_max, xyz_min)`); **time stays in `[0, 1]`** (upper grid
  half, border padding).
- Temporal planes initialize to ones (neutral modulation); spatial to
  `U(0.1, 0.5)`. D-NeRF uses `multires = [1, 2]`.

## E. Regularization semantics (verified)

The official live path uses **second-order squared finite differences**
(`mean((t[i+1] - 2t[i] + t[i-1])^2)`), NOT first-order TV — despite the
`plane_tv_weight` name. (`compute_plane_tv` exists officially but is dead
code; it is ported and marked non-live.)

- spatial smoothness on `xy/xz/yz` (weight `plane_tv`),
- temporal smoothness on `xt/yt/zt` (weight `time_smoothness`),
- L1 toward ONE on `xt/yt/zt` (weight `l1_time_planes`; one = the
  neutral init), summed over levels. Fine stage only.

## F. Coarse/fine training

- Coarse (default 3000): static rendering, deformation bypassed (its
  optimizer groups exist but receive no gradients and stay frozen).
- Fine (D-NeRF 20000): time-conditioned rendering, `L1 + λ(1-SSIM)` loss
  (official form — full L1 kept, NOT the canonical `(1-λ)` form) plus the
  regularization above.
- Stage iteration counters are independent (fine restarts at 1); the
  optimizer/stats are rebuilt per stage; SH degree persists. Checkpoints
  preserve stage-local counters; resume continues at `N + 1`.

## G. Densification note (intentional deviation)

Adaptive clone/split/prune/reset and Adam state surgery (`exp_avg` /
`exp_avg_sq` grow/slice/reset in lockstep) mirror the official code, with
one deviation: official `train.py` only prunes when `N > 200000`; we prune
whenever scheduled (opacity-driven, same rule otherwise).

## H. Datasets

| dataset | layout | bg | notes |
|---|---|---|---|
| D-NeRF | `transforms_{train,test}.json` + PNGs | white | monocular synthetic; random-cube init or `fused.ply` |
| DyNeRF | `poses_bounds.npy` + `camXX.mp4` markers + `camXX/images/*.png` + `points3D_downsample2.ply` | black | `eval_index=0` held out; 1352x1014, focal/2 (official quirk); frames must be pre-extracted |
| HyperNeRF | `scene/dataset/metadata.json` + `camera/*.json` + `rgb/{R}x/*.png` + `points3D_downsample2.ply` | white | `warp_id` times; video = test views; masks omitted (unused) |

Auto-detection by sentinel file, or explicit `--dataset_type`.

## I. Training commands

```bash
pip install -r requirements.txt
# CUDA rasterizer (see setup_rasterizer.md):
# TORCH_CUDA_ARCH_LIST="<your-arch>" pip install --no-build-isolation <rasterizer-checkout>

python -m training.train_4d --source data/dnerf/lego --output ./output/lego \
    --dataset_type dnerf
python -m training.train_4d --source data/dynerf/cut_roasted_beef --output ./output/beef \
    --dataset_type dynerf
python -m training.train_4d --source data/hypernerf/broom --output ./output/broom \
    --dataset_type hypernerf
```

`--dataset_type` selects BOTH the reader and the faithful per-dataset
hyperparameters (Commit 20 configs). Overrides: `--coarse_iterations`,
`--fine_iterations`, `--seed`, `--lambda_dssim`, `--background`.

## J. Resume

```bash
python -m training.train_4d --source ... --output ./output/again \
    --dataset_type dnerf --resume ./output/lego/checkpoint_fine_0006000.pth
python -m training.train_4d --source ... --output ... --checkpoint_interval 1000
```

Checkpoints (`checkpoint_{coarse,fine}_NNNNNN.pth`) hold raw Gaussians,
stats, field weights, full Adam state and RNG streams; cross-architecture
resume fails loudly. Rendering artifacts (`point_cloud.ply`,
`deformation.pth`) stay separate and sufficient for inference.

## K. Rendering

```bash
python -m training.render_static --source <scene> --model point_cloud.ply \
    --output ./renders --split test        # canonical static path
```

(The 4D render path is exercised by training and direct evaluation; test
cameras carry their timestamps into `DeformationField`.)

## L. Evaluation

```bash
# File mode (official workflow: render -> evaluate):
python -m evaluation.evaluate --renders <renders_dir> --gt <gt_dir> \
    --output ./output/metrics.json --device cuda
# Direct mode (PLY + deformation.pth + scene):
python -m evaluation.evaluate --source <scene> --model_path <model_dir> \
    --split test --dataset_type hypernerf --output ./output/metrics.json
```

Reports mean PSNR / SSIM / LPIPS-vgg / LPIPS-alex (+ per-image values) in
`metrics.json`.

## M. CUDA extension setup

Faithful source: `ingra14m/depth-diff-gaussian-rasterization` @ `9055fcf`
(the exact submodule pinned by `hustvl/4DGaussians`), plus one build-only
patch — `#include <cstdint>` in `cuda_rasterizer/rasterizer_impl.h`
(newer nvcc/GCC no longer provide `uint32_t` transitively; no math change).
Verified on RTX A6000, Python 3.12, PyTorch 2.8+cu128, CUDA 12.8. Without
the extension, math modules import but render/train calls raise
`RuntimeError` (no simplified fallback, by design).

## N. 20-commit implementation map

1. scaffold · 2. SH/covariance math · 3. Camera · 4. canonical Gaussian
   params + PLY · 5. densification/split/clone/prune · 6. faithful CUDA
   rasterizer + GPU gate · 7. D-NeRF reader + Scene · 8. losses/LR
   schedules/optimizer groups · 9. static trainer · 10. static
   rendering · 11. HexPlane encoder · 12. deformation decoder ·
   13. DeformationField coupling · 14. dynamic renderer (static->4D seam;
   rasterizer activation fix) · 15. spatial-temporal regularization ·
   16. coarse-to-fine 4D training · 17. checkpoint save/resume ·
   18. DyNeRF + HyperNeRF readers · 19. PSNR/SSIM/LPIPS evaluation ·
   20. per-dataset configs + docs.

## O. Parity / deviation matrix

Parity-checked (bit-exact or 0.0-diff unless noted): SH polynomials +
DC conversions; camera W2V/proj/center (~1e-7); PLY layout; HexPlane
init/query/fusion; decoder trunk + 5 heads; field forward; quaternion
multiply (1e-6); rasterizer boundary + pixels (1e-6); regularization
primitives/aggregation; LR schedules; L1/SSIM (0.0); PSNR (bit-exact);
LPIPS (byte-identical fork); D-NeRF/DyNeRF/HyperNeRF readers (R/T/FoV/
times/pixels); coarse/fine loss, schedules, thresholds.

Intentional deviations: prune without the official `N > 200k` guard;
`RuntimeError` on NaN loss (official re-execs); structured checkpoint dict
+ RNG preservation (official: positional tuple, no RNG); explicit
`--dataset_type` (official infers only); DyNeRF frames pre-extracted (no
mp4 decoding); HyperNeRF covisible masks omitted (unused downstream);
`metrics.json` instead of `results.json` + `per_view.json`; D-NeRF video
views carry `image=None` (official reuses a train image); determinism is
best-effort (CUDA backward atomics are order-nondeterministic — see the
Commit 14 report).

## P. Known limitations

- DyNeRF multi-view batching (`dataloader`, `batch_size = 4`) not
  reproduced (single-view sampling).
- PanopticSports / MultipleView / COLMAP readers not ported.
- Viewer (SIBR/GUI) not included.
- `dynerf`/`hypernerf` temporal grid resolution 150 and width 128 raise
  VRAM use vs D-NeRF; tiny fixtures only exercise the path.
