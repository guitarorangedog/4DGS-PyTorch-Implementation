# Light 4DGS — Educational Implementation

Small, readable demo of the core idea from:

> 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering (Wu et al., CVPR 2024)

This is NOT a full reproduction. It exists so you can study the
paper's pipeline line by line and reimplement it yourself in `my_version/`.
See `gpt_version/` for the faithful, full-scale reproduction.

## Simplified pipeline

```text
static 3D Gaussians
        ↓
time-conditioned deformation (HexPlane-lite + tiny MLP)
        ↓
deformed Gaussian parameters
        ↓
Gaussian rendering (pure PyTorch splatting)
        ↓
image reconstruction loss
        ↓
optimization
```

One canonical set of Gaussians is shared across time. A small
spatio-temporal field predicts per-Gaussian offsets from `(X, t)`.
Static rendering is the same path with deformation bypassed.

## 10-step progression (one concept per commit)

1. scaffold project (this commit)
2. pinhole camera and projection
3. canonical Gaussian parameters + activations
4. pure-PyTorch static splat renderer
5. static fit sanity check (one image)
6. tiny synthetic dynamic dataset (moving blobs)
7. HexPlane-lite + deformation decoder (`xy,xz,xt,yz,yt,zt`)
8. time-conditioned dynamic rendering
9. minimal 4DGS training loop (coarse → fine)
10. novel-time demo + PSNR

## Intentionally omitted

* CUDA rasterizer (we use brute-force PyTorch at 64x64)
* SH colors (we use flat RGB), densification / clone / split / prune (fixed N)
* Real datasets (synthetic blobs only), SSIM/LPIPS/regularizers/schedules/checkpoints
* Multi-GPU, viewers, config systems, logging frameworks

## Verify Commit 1

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import cameras, gaussians, render, dataset, deformation, train, demo; print('imports ok')"
```

Run from inside `light_version/`.
