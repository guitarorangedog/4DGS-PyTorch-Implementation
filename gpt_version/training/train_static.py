"""CLI: static 3D-GS training on a D-NeRF scene (Commit 9).

Usage (from ``gpt_version/``)::

    python -m training.train_static --source <dnerf_dir> --output ./output/static \\
        --iterations 3000 --seed 0

STATIC only: timestamps are loaded but never condition rendering.
"""

import argparse
import os

import torch

from data.scene import load_scene
from training.trainer_static import (
    StaticTrainerConfig,
    init_static_model,
    save_static_model,
    set_seed,
    train_static_scene,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Static 3D-GS training (Commit 9)")
    parser.add_argument("--source", required=True, help="D-NeRF scene directory")
    parser.add_argument("--output", default="./output/static")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_dssim", type=float, default=0.0)
    parser.add_argument("--extension", default=".png")
    parser.add_argument("--background", choices=["white", "black"], default="white")
    parser.add_argument("--no_eval", action="store_true",
                        help="merge test views into train (official eval=False)")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    scene = load_scene(args.source, white_background=(args.background == "white"),
                       eval_mode=(not args.no_eval), extension=args.extension, seed=args.seed)
    cfg = StaticTrainerConfig(iterations=args.iterations, seed=args.seed)
    cfg.optim.lambda_dssim = args.lambda_dssim
    if args.background == "black":
        cfg.background = (0.0, 0.0, 0.0)

    device = torch.device("cuda")
    model = init_static_model(scene, cfg, device=device)
    print(f"Training views: {len(scene.train_views)}, "
          f"init points: {model.num_points}, extent: {scene.scene_extent:.4f}")
    history = train_static_scene(scene, model, cfg, device=device)
    paths = save_static_model(model, os.path.abspath(args.output), cfg, history)
    print(f"Saved: {paths['ply']} ({model.num_points} points)")


if __name__ == "__main__":
    main()
