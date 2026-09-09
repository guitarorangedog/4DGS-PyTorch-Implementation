"""CLI: coarse-to-fine 4D-GS training (Commit 16; configs wired in Commit 20).

Usage (from ``gpt_version/``)::

    python -m training.train_4d --source <scene_dir> --output ./output/4d \\
        --dataset_type dnerf --coarse_iterations 3000 --fine_iterations 20000
"""

import argparse
import os
from dataclasses import replace

import torch

from configs import get_trainer_config
from data.scene import detect_dataset, load_scene
from training.trainer_4d import init_4d_model, save_4d_model, train_4d
from training.trainer_static import set_seed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Coarse-to-fine 4D-GS training")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="./output/4d")
    parser.add_argument("--dataset_type", choices=["auto", "dnerf", "dynerf", "hypernerf"],
                        default="auto",
                        help="selects BOTH the data reader and the faithful "
                             "per-dataset training config (auto = detect).")
    parser.add_argument("--coarse_iterations", type=int, default=None)
    parser.add_argument("--fine_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_dssim", type=float, default=None)
    parser.add_argument("--extension", default=".png")
    parser.add_argument("--background", choices=["white", "black"], default=None,
                        help="default: per-dataset official value "
                             "(dnerf white, dynerf black, hypernerf white).")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--resume", default=None,
                        help="resume from checkpoint_*.pth (continues at N+1)")
    parser.add_argument("--checkpoint_interval", type=int, default=0,
                        help="save end-of-iteration checkpoints every N iters (0=off)")
    args = parser.parse_args(argv)

    dtype = args.dataset_type
    if dtype == "auto":
        dtype = detect_dataset(args.source)
    print(f"Dataset: {dtype}")

    set_seed(args.seed)
    bg = None if args.background is None else (args.background == "white")
    scene = load_scene(args.source, white_background=bg,
                       eval_mode=(not args.no_eval), extension=args.extension,
                       seed=args.seed, dataset_type=dtype)
    over = {"seed": args.seed}
    if args.coarse_iterations is not None:
        over["coarse_iterations"] = args.coarse_iterations
    if args.fine_iterations is not None:
        over["fine_iterations"] = args.fine_iterations
    cfg = replace(get_trainer_config(dtype), **over)
    if args.lambda_dssim is not None:
        cfg.optim.lambda_dssim = args.lambda_dssim
    if args.background is not None:
        cfg.background = (1.0, 1.0, 1.0) if args.background == "white" else (0.0, 0.0, 0.0)

    device = torch.device("cuda")
    model, field = init_4d_model(scene, cfg, device=device)
    print(f"Training views: {len(scene.train_views)}, init points: {model.num_points}, "
          f"extent: {scene.scene_extent:.4f}, feat_dim: {field.hexplane.feat_dim}")
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
    outdir = os.path.abspath(args.output)
    history = train_4d(scene, model, field, cfg, device=device,
                       resume=args.resume, output_dir=outdir,
                       checkpoint_interval=args.checkpoint_interval)
    paths = save_4d_model(model, field, os.path.abspath(args.output), cfg, history)
    print(f"Saved: {paths['ply']} ({model.num_points} points), {paths['deformation']}")


if __name__ == "__main__":
    main()
