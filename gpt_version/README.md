# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering — Educational Reproduction

Clean PyTorch reproduction of Wu et al., CVPR 2024, for study purposes.

Reference implementation lives in `gpt_version/` only.
`my_version/` is the student's manual reimplementation space and is never touched here.

## Method map (paper → code)

- Eqs.1–4, static 3D-GS: `gaussians/` (SH, covariance/projection, canonical model, densification, CUDA rasterizer binding).
- Eqs.9–12, 4D novelty: `deformation/` (HexPlane encoder, tiny decoder, field coupling, regularization).
- Infrastructure: `data/` (cameras, readers, scene), `training/` (losses, schedules, trainer, checkpoints), `configs/`, `train.py`, `render.py`, `evaluate.py`.

Separation rule:

- `[STATIC]` inherited from 3D-GS.
- `[4D]` introduced by 4D-GS.
- `[INFRA]` needed only to train/evaluate.

## Roadmap (20 commits)

1. scaffold (this commit)
2. SH + covariance/projection math
3. Camera model
4. Canonical Gaussian model + PLY IO
5. Adaptive densification / prune
6. Faithful CUDA rasterizer binding + GPU smoke gate (RTX A6000)
7. D-NeRF reader + Scene
8. Static losses + schedules
9. Static 3D-GS trainer
10. Static render path
11. HexPlane encoder
12. Tiny decoder heads
13. DeformationField coupling
14. Static→4D integration (renderer + training step)
15. Spatial-temporal regularization
16. Coarse-to-fine 4D-GS schedule
17. Checkpoint save/restore
18. DyNeRF + HyperNeRF readers
19. Evaluation script
20. Per-dataset configs + docs

See `setup_rasterizer.md` for the CUDA dependency policy:
real rendering/training uses the faithful CUDA rasterizer only and fails fast when it is missing.

## Status

Commit 1: scaffold only. No math, no training, no rasterizer code yet.
