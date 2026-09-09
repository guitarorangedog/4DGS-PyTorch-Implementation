"""Tests for evaluation/metrics.py + evaluate.py. Run via `python3 -m tests.test_evaluate`."""

import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch
from PIL import Image

from evaluation.evaluate import evaluate_dirs, pair_files, write_metrics_json
from evaluation.metrics import evaluate_pair, lpips_value, psnr
from training.losses import ssim as ours_ssim


def _official_loss_utils():
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    sys.modules.setdefault("lpips", types.ModuleType("lpips"))
    spec = importlib.util.spec_from_file_location(
        "official_loss", "/tmp/opencode/4DGaussians/utils/loss_utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _official_image_utils():
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    spec = importlib.util.spec_from_file_location(
        "official_img", "/tmp/opencode/4DGaussians/utils/image_utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_psnr_parity_and_semantics() -> None:
    official = _official_image_utils()
    torch.manual_seed(0)
    a = torch.rand(2, 3, 16, 16)
    b = torch.rand(2, 3, 16, 16)
    assert torch.allclose(psnr(a, b), official.psnr(a, b), atol=0.0), \
        (psnr(a, b) - official.psnr(a, b)).abs().max()
    # Identical -> inf; -20dB-ish hand case: constant 0.5 vs 0.6 over 1px.
    one = torch.full((1, 1, 1, 1), 0.5)
    assert torch.isinf(psnr(one, one)).all()
    got = psnr(torch.full((1, 1, 1, 1), 0.5), torch.full((1, 1, 1, 1), 0.6)).item()
    assert abs(got - 20.0) < 1e-4, got  # mse=0.01 -> 20*log10(10)
    print("test_psnr_parity_and_semantics: passed (bit-exact, inf, 20dB)")


def test_ssim_parity() -> None:
    official = _official_loss_utils()
    torch.manual_seed(1)
    a = torch.rand(1, 3, 24, 24)
    b = torch.rand(1, 3, 24, 24)
    assert abs(ours_ssim(a, b).item() - official.ssim(a, b).item()) == 0.0
    assert abs(ours_ssim(a, a).item() - 1.0) < 1e-4
    print("test_ssim_parity: passed (bit-exact vs official loss_utils)")


def test_lpips_parity_and_inputs() -> None:
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    spec = importlib.util.spec_from_file_location(
        "official_lpips", "/tmp/opencode/4DGaussians/lpipsPyTorch/__init__.py")
    # The vendored package uses relative imports; load ours instead and
    # compare module-by-module against the official files on disk.
    import filecmp
    for rel in ("__init__.py", "modules/lpips.py", "modules/networks.py",
                "modules/utils.py"):
        assert filecmp.cmp(f"/tmp/opencode/4DGaussians/lpipsPyTorch/{rel}",
                           f"evaluation/_lpips/{rel}", shallow=False), rel
    from evaluation._lpips.modules.lpips import LPIPS
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.rand(1, 3, 64, 64, device=device)
    b = torch.rand(1, 3, 64, 64, device=device)
    for net in ("alex", "vgg"):
        model = LPIPS(net, "0.1").to(device).eval()
        with torch.no_grad():
            same = model(a, a).item()
            diff = model(a, b).item()
        assert same == 0.0, (net, same)
        assert diff > 0, (net, diff)
        ours = lpips_value(a, b, net, model).item()
        assert abs(ours - diff) < 1e-9
    print("test_lpips_parity_and_inputs: passed (files identical, [0,1] in, 0 on identical)")


def test_sanity_ordering() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = torch.rand(1, 3, 64, 64, device=device)
    small = (base + 0.02 * torch.randn_like(base)).clamp(0, 1)
    large = (base + 0.4 * torch.randn_like(base)).clamp(0, 1)
    ms = evaluate_pair(base, small)
    ml = evaluate_pair(base, large)
    mi = evaluate_pair(base, base)
    assert mi["psnr"] == float("inf") and mi["ssim"] > 0.999 and mi["lpips_alex"] == 0.0
    assert ms["psnr"] > ml["psnr"] and ms["ssim"] > ml["ssim"]
    assert ms["lpips_alex"] < ml["lpips_alex"] and ms["lpips_vgg"] < ml["lpips_vgg"]
    print(f"test_sanity_ordering: passed "
          f"(psnr {ml['psnr']:.1f} < {ms['psnr']:.1f} < inf)")


def _write_png(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def test_directory_fixture_roundtrip() -> None:
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory() as root:
        rdir, gdir = os.path.join(root, "renders"), os.path.join(root, "gt")
        names = ["00000_a.png", "00001_b.png"]
        for n in names:
            base = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
            _write_png(os.path.join(gdir, n), base)
            noisy = np.clip(base.astype(np.float32) + np.random.normal(0, 5, base.shape),
                            0, 255).astype(np.uint8)
            _write_png(os.path.join(rdir, n), noisy)
        pairs = pair_files(rdir, gdir)
        assert [p[0] for p in pairs] == names  # sorted deterministic pairing
        result = evaluate_dirs(rdir, gdir, device="cpu")
        assert result["n_images"] == 2
        for key in ("psnr", "ssim", "lpips_vgg", "lpips_alex"):
            vals = [result["per_image"][n][key] for n in names]
            assert abs(result[key] - sum(vals) / len(vals)) < 1e-9, key
        assert result["per_image"][names[0]]["psnr"] > 20.0
        mp = write_metrics_json(result, os.path.join(root, "metrics.json"))
        back = json.load(open(mp))
        assert back["n_images"] == 2 and set(back["per_image"]) == set(names)
        # Mismatches fail clearly.
        os.remove(os.path.join(gdir, names[0]))
        try:
            evaluate_dirs(rdir, gdir, device="cpu")
        except FileNotFoundError as e:
            assert "lack GT mates" in str(e)
        else:
            raise AssertionError("missing GT should raise")
        _write_png(os.path.join(gdir, names[0]),
                   np.zeros((16, 16, 3), dtype=np.uint8))
        try:
            evaluate_dirs(rdir, gdir, device="cpu")
        except ValueError as e:
            assert "Shape mismatch" in str(e)
        else:
            raise AssertionError("shape mismatch should raise")
    print("test_directory_fixture_roundtrip: passed")


def test_cuda_dynamic_smoke() -> None:
    if not torch.cuda.is_available():
        print("test_cuda_dynamic_smoke: skipped (no CUDA)")
        return
    sys.path.insert(0, "tests")
    import tempfile as tf
    from test_hypernerf import _write_fixture
    from data.scene import load_scene
    from deformation.field import DeformationField, FieldConfig
    from deformation.hexplane import HexPlaneConfig
    from deformation.decoder import DecoderConfig
    from deformation.render_4d import render_deformed_view
    from gaussians.gaussian_model import CanonicalGaussianModel
    from evaluation.evaluate import evaluate_model

    device = torch.device("cuda")
    with tf.TemporaryDirectory() as root:
        _write_fixture(root)
        scene = load_scene(root, seed=0, ratio=1.0)
        model = CanonicalGaussianModel(max_sh_degree=0).to(device)
        model.create_from_pointcloud(scene.points, scene.colors, device=device)
        fcfg = FieldConfig(
            hexplane=HexPlaneConfig(output_coordinate_dim=4,
                                    resolution=[8, 8, 8, 5], multires=[1, 2]),
            decoder=DecoderConfig(width=16, depth=1))
        field = DeformationField(fcfg)
        field.set_aabb(scene.aabb_max, scene.aabb_min)
        field = field.to(device)
        view = scene.test_views[0]
        img = render_deformed_view(view.camera, model, field, device=device)["render"]
        assert torch.isfinite(img).all()
        torch.save(field.state_dict(), os.path.join(root, "deformation.pth"))
        model.save_ply(os.path.join(root, "point_cloud.ply"))
        result = evaluate_model(root, root, split="test", dataset_type="hypernerf",
                                ratio=1.0, device=device)
        assert result["n_images"] == 1
        for key in ("psnr", "ssim", "lpips_vgg", "lpips_alex"):
            assert np.isfinite(result[key]), (key, result[key])
    print(f"test_cuda_dynamic_smoke: passed "
          f"(psnr={result['psnr']:.2f} ssim={result['ssim']:.4f})")


if __name__ == "__main__":
    test_psnr_parity_and_semantics()
    test_ssim_parity()
    test_lpips_parity_and_inputs()
    test_sanity_ordering()
    test_directory_fixture_roundtrip()
    test_cuda_dynamic_smoke()
    print("test_evaluate.py: done")
