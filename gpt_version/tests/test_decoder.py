"""CPU tests for deformation/decoder.py. Run via `python3 -m tests.test_decoder`."""

import math
import sys
import types
from types import SimpleNamespace

import torch
import torch.nn as nn

from deformation.decoder import (
    AuxDeformationHeads,
    DecoderConfig,
    DeformationDecoder,
)


def _stub_official_imports() -> None:
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    _tk = types.ModuleType("tkinter")
    _tk.W = None
    sys.modules.setdefault("tkinter", _tk)
    _sk = types.ModuleType("simple_knn")
    _sk_C = types.ModuleType("simple_knn._C")
    _sk_C.distCUDA2 = None
    _sk._C = _sk_C
    sys.modules.setdefault("simple_knn", _sk)
    sys.modules.setdefault("simple_knn._C", _sk_C)


def test_shapes_and_batches() -> None:
    dec = DeformationDecoder(feature_dim=8)
    for n in (1, 5):
        out = dec(torch.randn(n, 8))
        assert out.d_xyz.shape == (n, 3)
        assert out.d_scaling.shape == (n, 3)
        assert out.d_rotation.shape == (n, 4)
    aux = AuxDeformationHeads()
    do, dshs = aux(torch.randn(4, 64))
    assert do.shape == (4, 1) and dshs.shape == (4, 16, 3)


def test_architecture_matches_official() -> None:
    dec = DeformationDecoder(feature_dim=8, config=DecoderConfig(width=64, depth=1))
    assert len(dec.trunk) == 1 and isinstance(dec.trunk[0], nn.Linear)
    for head in (dec.pos_head, dec.scale_head, dec.rot_head):
        kinds = [type(m).__name__ for m in head]
        assert kinds == ["ReLU", "Linear", "ReLU", "Linear"], kinds
    deep = DeformationDecoder(feature_dim=8, config=DecoderConfig(width=32, depth=3))
    assert len(deep.trunk) == 5  # Linear + 2x(ReLU, Linear)


def test_initialization_is_xavier_not_zero() -> None:
    torch.manual_seed(0)
    dec = DeformationDecoder(feature_dim=8)
    for name, mod in dec.named_modules():
        if isinstance(mod, nn.Linear):
            fan_in, fan_out = mod.weight.shape[1], mod.weight.shape[0]
            xavier_bound = math.sqrt(6.0 / (fan_in + fan_out))
            assert mod.weight.abs().max() <= xavier_bound + 1e-6, name
            default_bound = 1.0 / math.sqrt(fan_in)  # untouched PyTorch bias init
            assert mod.bias.abs().max() <= default_bound + 1e-6, name
    # Heads are NOT zero-initialized: fresh decoder deforms (official behavior).
    out = dec(torch.randn(4, 8) * 0.1)
    assert out.d_xyz.abs().sum() > 0 and out.d_scaling.abs().sum() > 0


def test_head_independence() -> None:
    torch.manual_seed(1)
    dec = DeformationDecoder(feature_dim=8)
    feat = torch.randn(3, 8)
    base = dec(feat)
    old_bias = dec.pos_head[-1].bias.detach().clone()
    with torch.no_grad():
        dec.pos_head[-1].bias.fill_(5.0)
    moved = dec(feat)
    assert torch.allclose(moved.d_xyz - base.d_xyz,
                          (5.0 - old_bias).expand(3, 3), atol=1e-5)
    assert torch.equal(moved.d_scaling, base.d_scaling)
    assert torch.equal(moved.d_rotation, base.d_rotation)


def test_gradients() -> None:
    torch.manual_seed(2)
    dec = DeformationDecoder(feature_dim=8)
    feat = torch.randn(6, 8, requires_grad=True)
    out = dec(feat)
    (out.d_xyz.sum() + out.d_scaling.sum() + out.d_rotation.sum()).backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert feat.grad.abs().sum() > 0
    for p in dec.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def _official_deformation():
    _stub_official_imports()
    from scene.deformation import Deformation

    args = SimpleNamespace(
        no_grid=False, empty_voxel=False, static_mlp=False, bounds=1.6,
        kplanes_config={"grid_dimensions": 2, "input_coordinate_dim": 4,
                        "output_coordinate_dim": 4, "resolution": [4, 4, 4, 3]},
        multires=[1, 2], no_dx=False, no_ds=False, no_dr=False,
        no_do=True, no_dshs=True, apply_rotation=False,
    )
    return Deformation(D=1, W=64, grid_pe=0, args=args)


def test_official_parity() -> None:
    torch.manual_seed(4)
    ref = _official_deformation()  # feat_dim = 4 * 2 levels = 8
    ours = DeformationDecoder(feature_dim=8)
    aux = AuxDeformationHeads()
    # Transplant official weights into ours (identical layer shapes).
    ours.trunk[0].weight.data.copy_(ref.feature_out[0].weight)
    ours.trunk[0].bias.data.copy_(ref.feature_out[0].bias)
    for ours_head, ref_head in ((ours.pos_head, ref.pos_deform),
                               (ours.scale_head, ref.scales_deform),
                               (ours.rot_head, ref.rotations_deform),
                               (aux.opacity_head, ref.opacity_deform),
                               (aux.shs_head, ref.shs_deform)):
        for o, r in zip(ours_head, ref_head):
            if isinstance(o, nn.Linear):
                o.weight.data.copy_(r.weight)
                o.bias.data.copy_(r.bias)
    feat = torch.randn(7, 8)
    out = ours(feat)
    hidden = ref.feature_out(feat)
    assert torch.equal(out.d_xyz, ref.pos_deform(hidden))
    assert torch.equal(out.d_scaling, ref.scales_deform(hidden))
    assert torch.equal(out.d_rotation, ref.rotations_deform(hidden))
    do, dshs = aux(hidden)
    assert torch.equal(do, ref.opacity_deform(hidden))
    assert torch.equal(dshs, ref.shs_deform(hidden).reshape(7, 16, 3))
    print("test_official_parity: passed (bit-exact trunk + 5 heads)")


if __name__ == "__main__":
    test_shapes_and_batches()
    test_architecture_matches_official()
    test_initialization_is_xavier_not_zero()
    test_head_independence()
    test_gradients()
    test_official_parity()
    print("test_decoder.py: all 6 tests passed")
