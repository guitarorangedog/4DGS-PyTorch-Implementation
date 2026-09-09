# CUDA rasterizer policy

Real rendering and training must use a faithful CUDA Gaussian rasterizer
compatible with 3D-GS / the official 4DGaussians fork
(`depth-diff-gaussian-rasterization` family).

Rules:

1. No simplified pure-PyTorch rasterizer that changes the algorithm.
2. The `gaussians/rasterizer.py` binding (Commit 6) imports the compiled
   extension lazily and raises a clear `RuntimeError` when it is missing.
3. Python modules stay importable without CUDA; CPU-only math tests cover
   projection/encoding shapes, never image formation.
4. Commit 6 is an integration gate: build/install the extension on this
   RunPod machine and run a minimal GPU smoke render on the RTX A6000
   before proceeding to Commit 7+.

## Verified build record (Commit 6 gate, PASSED)

- Source: `https://github.com/ingra14m/depth-diff-gaussian-rasterization`
  @ `9055fcf` (2023-11-09) — the exact submodule pinned by
  `hustvl/4DGaussians` — cloned with `--recursive` (needs `third_party/glm`).
- Environment: RTX A6000 (sm_86), Python 3.12.3, PyTorch 2.8.0+cu128,
  CUDA toolkit 12.8.
- Prerequisite: `pip install ninja` (parallel extension build).
- Build (the stock `setup.py` imports torch, so build isolation must be off):
  `TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation .`
- Compatibility patch (build-only, no rendering-math change): add
  `#include <cstdint>` to `cuda_rasterizer/rasterizer_impl.h` — CUDA 12.8 /
  GCC 13 no longer provide `uint32_t` / `std::uintptr_t` transitively.
  First attempt without it fails with `namespace "std" has no member
  "uintptr_t"`; with it the wheel builds and installs as
  `diff_gaussian_rasterization-0.0.0`.
- Gate test: `python3 -m tests.test_rasterizer_smoke` on the A6000 —
  8/8 Gaussians visible at 64x64, finite image + depth, `backward()` from
  `mean(image)` yields finite nonzero grads on `_xyz` (via 2D means),
  `_features_dc`, `_scaling`, `_opacity`.

Nothing is installed by Commit 1.
