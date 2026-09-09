"""Evaluation drivers: renders/GT directories + direct model eval [INFRA].

Official workflow reproduced (``render.py`` -> ``metrics.py``):

- ``render.py`` writes ``<model>/<split>/ours_<iter>/{renders,gt}/`` PNGs;
  ``metrics.py::readImages`` pairs them BY FILENAME and compares
  ``[1, 3, H, W]`` ``[0, 1]`` tensors (alpha dropped).
- Aggregates are arithmetic means of per-image values; ``results.json`` /
  ``per_view.json`` per scene (we write a single ``metrics.json``).

Two modes:

A. Directory mode (primary, official-style): ``--renders DIR --gt DIR``.
B. Direct mode: ``--source DIR --model_path DIR`` loads the scene test
   split + ``point_cloud.ply`` + ``deformation.pth`` and renders each test
   camera time-conditionally (Commit 14 semantics — never the static
   canonical at every timestamp), comparing to ``View.image``.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from evaluation.metrics import evaluate_pair, lpips_value

__all__ = [
    "pair_files",
    "load_image_tensor",
    "evaluate_dirs",
    "evaluate_model",
    "write_metrics_json",
    "METRIC_KEYS",
]

METRIC_KEYS = ("psnr", "ssim", "lpips_vgg", "lpips_alex")


def load_image_tensor(path: str) -> torch.Tensor:
    """PNG -> ``[1, 3, H, W]`` float32 ``[0, 1]`` (official ``readImages``).

    Contiguous (official ``to_tensor`` outputs are contiguous, which the
    official ``psnr`` ``.view`` relies on).
    """
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def pair_files(renders_dir: str, gt_dir: str) -> list[tuple[str, str, str]]:
    """Deterministically pair renders with GT by FILENAME (sorted).

    Official pairs by iterating ``os.listdir(renders_dir)`` (arbitrary
    order) and opening the same name under ``gt/``; we sort for
    determinism. Missing/unmatched files and shape mismatches raise
    clear errors instead of failing mid-loop.
    """
    r_files = sorted(f for f in os.listdir(renders_dir) if f.lower().endswith(".png"))
    g_files = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(".png"))
    if not r_files:
        raise FileNotFoundError(f"No renders in {renders_dir}.")
    missing_gt = [f for f in r_files if f not in g_files]
    if missing_gt:
        raise FileNotFoundError(
            f"{len(missing_gt)} renders lack GT mates in {gt_dir} "
            f"(e.g. {missing_gt[:3]}).")
    extra_gt = [f for f in g_files if f not in r_files]
    if extra_gt:
        raise FileNotFoundError(
            f"{len(extra_gt)} GT files lack renders in {renders_dir} "
            f"(e.g. {extra_gt[:3]}).")
    return [(f, os.path.join(renders_dir, f), os.path.join(gt_dir, f)) for f in r_files]


@torch.no_grad()
def evaluate_dirs(renders_dir: str, gt_dir: str, device="cpu") -> dict:
    """Evaluate paired PNG directories. Returns the result dict (no JSON)."""
    from evaluation._lpips.modules.lpips import LPIPS

    device = torch.device(device)
    models = {net: LPIPS(net, "0.1").to(device).eval() for net in ("vgg", "alex")}
    per_image = {}
    for name, r_path, g_path in pair_files(renders_dir, gt_dir):
        pred = load_image_tensor(r_path).to(device)
        gt = load_image_tensor(g_path).to(device)
        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {name}: render {tuple(pred.shape)} vs "
                f"gt {tuple(gt.shape)}.")
        per_image[name] = evaluate_pair(pred, gt, models)
    return _aggregate(per_image, {"mode": "dirs", "renders_dir": renders_dir,
                                  "gt_dir": gt_dir, "device": str(device)})


def _aggregate(per_image: dict, meta: dict) -> dict:
    import math

    result = {"n_images": len(per_image), "per_image": per_image, "meta": meta}
    for key in METRIC_KEYS:
        vals = [v[key] for v in per_image.values()]
        finite = [v for v in vals if math.isfinite(v)]
        result[key] = float(sum(vals) / len(vals)) if finite and len(finite) == len(vals) \
            else (float(sum(finite) / len(finite)) if finite else float("inf"))
    return result


@torch.no_grad()
def evaluate_model(source: str, model_path: str, split: str = "test",
                   dataset_type=None, ratio: float = 0.5,
                   device="cuda", bg_color=None) -> dict:
    """Direct evaluation: render test views time-conditionally, compare to GT.

    Loads ``<model_path>/point_cloud.ply`` + ``deformation.pth`` (field arch
    inferred from weights), renders ``scene.<split>_views`` with
    :func:`render_deformed_view`, and scores against ``View.image``.
    """
    from data.scene import load_scene
    from deformation.field import DeformationField, field_config_from_state_dict
    from deformation.render_4d import render_deformed_view
    from training.render_static import infer_sh_degree, load_static_model

    device = torch.device(device)
    scene = load_scene(source, white_background=None, eval_mode=True,
                       dataset_type=dataset_type, ratio=ratio)
    views = {"train": scene.train_views, "test": scene.test_views,
             "video": scene.video_views}[split]
    if not views:
        raise ValueError(f"No {split} views in {source}.")
    if bg_color is None:
        from data.scene import DATASET_BG
        white = DATASET_BG[scene.dataset_type]
        bg_color = (1.0, 1.0, 1.0) if white else (0.0, 0.0, 0.0)
    ply = os.path.join(model_path, "point_cloud.ply")
    dph = os.path.join(model_path, "deformation.pth")
    for p in (ply, dph):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Direct eval needs {p}.")
    model = load_static_model(ply, device=device)
    sd = torch.load(dph, map_location=device, weights_only=False)
    field = DeformationField(field_config_from_state_dict(sd)).to(device)
    field.load_state_dict(sd)
    from evaluation._lpips.modules.lpips import LPIPS
    models = {net: LPIPS(net, "0.1").to(device).eval() for net in ("vgg", "alex")}
    per_image = {}
    for view in views:
        if view.image is None:
            raise ValueError(f"View {view.image_name} has no GT image.")
        pred = render_deformed_view(view.camera, model, field,
                                    bg_color=bg_color, device=device)["render"]
        pred = pred.clamp(0.0, 1.0).unsqueeze(0)
        gt = view.image.to(device).unsqueeze(0)
        per_image[view.image_name + ".png"] = evaluate_pair(pred, gt, models)
    return _aggregate(per_image, {"mode": "model", "source": source,
                                  "model_path": model_path, "split": split,
                                  "dataset_type": scene.dataset_type,
                                  "device": str(device)})


def write_metrics_json(result: dict, path: str) -> str:
    """Write ``metrics.json`` (aggregates, per-image values, metadata)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def print_summary(result: dict) -> None:
    """Concise console summary (official ``metrics.py`` print style)."""
    print(f"Images : {result['n_images']}")
    print(f"PSNR   : {result['psnr']:.7f}")
    print(f"SSIM   : {result['ssim']:.7f}")
    print(f"LPIPS-vgg : {result['lpips_vgg']:.7f}")
    print(f"LPIPS-alex: {result['lpips_alex']:.7f}")


def main(argv=None) -> None:
    """CLI: ``python -m evaluation.evaluate (--renders DIR --gt DIR | --source DIR --model_path DIR) ...``."""
    import argparse

    parser = argparse.ArgumentParser(description="4DGS evaluation (Commit 19)")
    parser.add_argument("--renders", default=None)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--dataset_type", default="auto",
                        choices=["auto", "dnerf", "dynerf", "hypernerf"])
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--output", default="./output/metrics.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.renders is not None:
        if args.gt is None:
            raise ValueError("--gt is required with --renders.")
        result = evaluate_dirs(args.renders, args.gt, device=args.device)
    elif args.source is not None and args.model_path is not None:
        result = evaluate_model(args.source, args.model_path, split=args.split,
                                dataset_type=None if args.dataset_type == "auto" else args.dataset_type,
                                ratio=args.ratio, device=args.device)
    else:
        raise ValueError("Provide --renders + --gt (file mode) or --source + --model_path (direct mode).")
    print_summary(result)
    print("Wrote", write_metrics_json(result, args.output))


if __name__ == "__main__":
    main()
