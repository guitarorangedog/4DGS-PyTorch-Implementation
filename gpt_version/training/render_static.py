"""Static novel-view / video rendering for trained canonical models [STATIC].

Final "before 4D" snapshot (Commit 10): renders the SAME canonical Gaussian
set for every camera. ``Camera.time`` is never read — see
:func:`render_views` and the timestamp-independence test.

Layout (per split)::

    <output>/<split>/renders/{idx:05d}_{image_name}.png
    <output>/<split>/gt/{idx:05d}_{image_name}.png   (train/test only)
    <output>/video/video.mp4                          (video split, best-effort)

MP4 assembly uses ``imageio`` when importable and is skipped gracefully
otherwise; PNG frames are the required artifact.
"""

import copy
import os

import numpy as np
import torch
from PIL import Image

from data.scene import Scene
from gaussians.gaussian_model import CanonicalGaussianModel
from gaussians.rasterizer import render_view

__all__ = [
    "SPLITS",
    "infer_sh_degree",
    "load_static_model",
    "render_views",
    "save_render_set",
    "render_splits",
]

SPLITS = ("train", "test", "video")


def infer_sh_degree(ply_path: str) -> int:
    """Infer ``max_sh_degree`` from the PLY ``f_rest_*`` attribute count."""
    from plyfile import PlyData

    names = [p.name for p in PlyData.read(ply_path).elements[0].properties
             if p.name.startswith("f_rest_")]
    k = len(names) // 3 + 1
    degree = int(round(k ** 0.5)) - 1
    assert degree in (0, 1, 2, 3) and (degree + 1) ** 2 == k, \
        f"cannot infer SH degree from {len(names)} f_rest attrs"
    return degree


def load_static_model(ply_path: str, device: torch.device | str = "cuda",
                      max_sh_degree: int | None = None) -> CanonicalGaussianModel:
    """Load a Commit 9 PLY bit-for-bit (raw log-scales/logits/SH preserved).

    Args:
        ply_path: ``point_cloud.ply`` written by ``save_static_model``.
        device: target CUDA device.
        max_sh_degree: explicit degree; inferred from the file when ``None``.
    """
    if max_sh_degree is None:
        max_sh_degree = infer_sh_degree(ply_path)
    model = CanonicalGaussianModel(max_sh_degree=max_sh_degree).to(device)
    model.load_ply(ply_path, device=device)
    return model


@torch.no_grad()
def render_views(model: CanonicalGaussianModel, views,
                 device: torch.device | str = "cuda",
                 bg_color=(1.0, 1.0, 1.0)) -> list[torch.Tensor]:
    """Render views with the canonical Gaussians (STATIC: time ignored).

    Only ``camera`` geometry (R/T/FoV/W/H) conditions the rasterizer; the
    identical parameter set is splatted for every timestamp.

    Returns:
        List of ``[3, H, W]`` float32 CPU tensors in ``[0, 1]`` (clamped).
    """
    device = torch.device(device)
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    images = []
    for view in views:
        pkg = render_view(view.camera, model, bg_color=bg, device=device)
        images.append(pkg["render"].detach().cpu().clamp(0.0, 1.0))
    return images


def _to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def save_render_set(split_dir: str, views, images: list[torch.Tensor],
                    save_gt: bool = True, mp4_name: str | None = None) -> dict:
    """Write ``renders/`` (+ ``gt/`` for views carrying images) PNGs.

    Filenames ``{idx:05d}_{image_name}.png`` are deterministic in view order.
    Returns paths of written dirs/files.
    """
    render_dir = os.path.join(split_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    written = {"renders": render_dir}
    for idx, (view, image) in enumerate(zip(views, images)):
        _to_pil(image).save(os.path.join(
            render_dir, f"{idx:05d}_{view.image_name}.png"))
    gt_views = [v for v in views if v.image is not None]
    if save_gt and gt_views:
        gt_dir = os.path.join(split_dir, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        for idx, view in enumerate(views):
            if view.image is not None:
                _to_pil(view.image).save(os.path.join(
                    gt_dir, f"{idx:05d}_{view.image_name}.png"))
        written["gt"] = gt_dir
    if mp4_name is not None:
        try:
            import imageio.v2 as imageio
            mp4_path = os.path.join(split_dir, mp4_name)
            frames = [np.asarray(_to_pil(im)) for im in images]
            imageio.mimsave(mp4_path, frames, fps=30, codec="libx264")
            written["mp4"] = mp4_path
        except ImportError:
            print("imageio unavailable: skipping MP4 assembly (PNG frames kept)")
    return written


def render_splits(scene: Scene, model: CanonicalGaussianModel, output_dir: str,
                  splits=("test",), device: torch.device | str = "cuda",
                  bg_color=(1.0, 1.0, 1.0), save_gt: bool = True) -> dict:
    """Render requested splits; returns ``{split: written_paths}``."""
    out: dict = {}
    for split in splits:
        assert split in SPLITS, split
        views = {"train": scene.train_views, "test": scene.test_views,
                 "video": scene.video_views}[split]
        images = render_views(model, views, device=device, bg_color=bg_color)
        out[split] = save_render_set(
            os.path.join(output_dir, split), views, images, save_gt=save_gt,
            mp4_name="video.mp4" if split == "video" else None)
    return out


def with_time(view, time: float):
    """Copy a view with only ``camera.time`` changed (independence probe)."""
    import dataclasses

    cam = copy.copy(view.camera)
    cam.time = float(time)
    return dataclasses.replace(view, camera=cam)


def main(argv=None) -> None:
    """CLI: ``python -m training.render_static --source DIR --model PLY ...``."""
    import argparse

    from data.scene import load_scene

    parser = argparse.ArgumentParser(description="Static rendering (Commit 10)")
    parser.add_argument("--source", required=True, help="D-NeRF scene directory")
    parser.add_argument("--model", required=True, help="trained point_cloud.ply")
    parser.add_argument("--output", default="./output/static_renders")
    parser.add_argument("--split", default="test",
                        choices=["train", "test", "video", "all"])
    parser.add_argument("--background", choices=["white", "black"], default="white")
    parser.add_argument("--extension", default=".png")
    parser.add_argument("--no_gt", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device("cuda")
    scene = load_scene(args.source, white_background=(args.background == "white"),
                       eval_mode=True, extension=args.extension)
    model = load_static_model(args.model, device=device)
    splits = list(SPLITS) if args.split == "all" else [args.split]
    bg = (1.0, 1.0, 1.0) if args.background == "white" else (0.0, 0.0, 0.0)
    out = render_splits(scene, model, os.path.abspath(args.output), splits,
                        device=device, bg_color=bg, save_gt=(not args.no_gt))
    for split, paths in out.items():
        print(f"{split}: {paths}")


if __name__ == "__main__":
    main()
