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

Nothing is installed by Commit 1.
