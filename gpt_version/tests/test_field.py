"""CPU tests for deformation/field.py. Run via `python3 -m tests.test_field`."""

import sys
import types
from types import SimpleNamespace

import torch
import torch.nn as nn

from deformation.decoder import DecoderConfig
from deformation.field import (
    DeformationField,
    FieldConfig,
    quaternion_multiply,
)
from deformation.hexplane import HexPlaneConfig


def _tiny_field(**over) -> DeformationField:
    kw = dict(
        hexplane=HexPlaneConfig(bounds=1.6, output_coordinate_dim=4,
                                resolution=[4, 4, 4, 3], multires=[1, 2]),
        decoder=DecoderConfig(width=16, depth=1),
    )
    kw.update(over)
    return DeformationField(FieldConfig(**kw))


def _inputs(n: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    return (torch.randn(n, 3), torch.randn(n, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
            torch.full((n, 1), 0.4))


def test_shapes_and_time_forms() -> None:
    field = _tiny_field()
    xyz, s, r, t = _inputs()
    for time in (t, t.squeeze(-1), 0.4):
        out = field(xyz, s, r, time)
        assert out.xyz.shape == (6, 3)
        assert out.scaling.shape == (6, 3)
        assert out.rotation.shape == (6, 4)
        assert out.opacity is None and out.shs is None


def test_zero_deltas_is_identity() -> None:
    field = _tiny_field()
    with torch.no_grad():
        for p in field.decoder.parameters():
            p.zero_()
    xyz, s, r, t = _inputs()
    out = field(xyz, s, r, t)
    assert torch.equal(out.xyz, xyz)
    assert torch.equal(out.scaling, s)
    assert torch.equal(out.rotation, r)


def test_manual_delta_values() -> None:
    field = _tiny_field()
    with torch.no_grad():
        for p in field.decoder.parameters():
            p.zero_()
        field.decoder.pos_head[-1].bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        field.decoder.scale_head[-1].bias.copy_(torch.tensor([0.5, -0.5, 0.0]))
    xyz, s, r, t = _inputs()
    out = field(xyz, s, r, t)
    assert torch.allclose(out.xyz - xyz, torch.tensor([[1.0, 2.0, 3.0]]).expand(6, 3), atol=1e-6)
    assert torch.allclose(out.scaling - s, torch.tensor([[0.5, -0.5, 0.0]]).expand(6, 3), atol=1e-6)
    assert torch.equal(out.rotation, r)  # untouched head -> exact passthrough


def test_time_sensitivity_with_trained_planes() -> None:
    torch.manual_seed(0)
    field = _tiny_field()
    with torch.no_grad():  # at init time planes are ones: install trained-like values
        for level in field.hexplane.grids:
            for i in (2, 4, 5):
                level[i].copy_(torch.rand_like(level[i]))
    xyz, s, r, _ = _inputs()
    a = field(xyz, s, r, torch.full((6, 1), 0.2))
    b = field(xyz, s, r, torch.full((6, 1), 0.8))
    assert not torch.equal(a.xyz, b.xyz)
    assert not torch.equal(a.scaling, b.scaling)


def test_raw_spaces_preserved() -> None:
    """Activations must NOT happen inside the field: scaling stays log-space."""
    field = _tiny_field()
    xyz, s, r, t = _inputs()
    out = field(xyz, s, r, t)
    assert (out.scaling <= 0).any(), "raw log-scales should often be negative"
    assert torch.allclose(out.scaling.mean().exp().log(), out.scaling.mean(), atol=1e-5)


def test_quaternion_multiply_matches_official() -> None:
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    from utils.graphics_utils import batch_quaternion_multiply as ref

    torch.manual_seed(0)
    q1 = torch.randn(9, 4)
    q2 = torch.randn(9, 4)
    assert torch.allclose(quaternion_multiply(q1, q2), ref(q1, q2), atol=1e-6)


def test_apply_rotation_option() -> None:
    field = _tiny_field(apply_rotation=True)
    xyz, s, r, t = _inputs()
    out = field(xyz, s, r, t)
    assert torch.allclose(out.rotation.norm(dim=1), torch.ones(6), atol=1e-5)


def test_parameter_groups_disjoint_and_complete() -> None:
    field = _tiny_field()
    grid = field.get_grid_parameters()
    mlp = field.get_mlp_parameters()
    grid_ids, mlp_ids = {id(p) for p in grid}, {id(p) for p in mlp}
    assert not grid_ids & mlp_ids
    all_ids = {id(p) for p in field.parameters()}
    assert grid_ids | mlp_ids == all_ids
    assert len(grid) == 1 + 2 * 6  # aabb + 2 levels x 6 planes
    assert len(mlp) == 2 + 3 * 4  # trunk Linear + 3 heads x 2 Linears (w+b each)


def test_gradients_end_to_end() -> None:
    torch.manual_seed(1)
    field = _tiny_field()
    xyz = torch.randn(5, 3, requires_grad=True)
    s = torch.randn(5, 3, requires_grad=True)
    r = torch.randn(5, 4, requires_grad=True)
    t = torch.full((5, 1), 0.5, requires_grad=True)
    out = field(xyz, s, r, t)
    (out.xyz.sum() + out.scaling.sum() + out.rotation.sum()).backward()
    for name, g in (("xyz", xyz.grad), ("scaling", s.grad),
                    ("rotation", r.grad), ("time", t.grad)):
        assert g is not None and torch.isfinite(g).all(), name
    assert xyz.grad.abs().sum() > 0 and s.grad.abs().sum() > 0 and r.grad.abs().sum() > 0
    for p in field.parameters():
        if not p.requires_grad:
            continue  # frozen AABB
        assert p.grad is not None and torch.isfinite(p.grad).all()
    grid_mass = sum(p.grad.abs().sum().item() for p in field.get_grid_parameters()
                    if p.grad is not None)
    mlp_mass = sum(p.grad.abs().sum().item() for p in field.get_mlp_parameters())
    assert grid_mass > 0 and mlp_mass > 0, (grid_mass, mlp_mass)


def _official_field():
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
    from scene.deformation import Deformation

    args = SimpleNamespace(
        no_grid=False, empty_voxel=False, static_mlp=False, bounds=1.6,
        kplanes_config={"grid_dimensions": 2, "input_coordinate_dim": 4,
                        "output_coordinate_dim": 4, "resolution": [4, 4, 4, 3]},
        multires=[1, 2], no_dx=False, no_ds=False, no_dr=False,
        no_do=True, no_dshs=True, apply_rotation=False,
    )
    return Deformation(D=1, W=16, grid_pe=0, args=args)


def test_official_forward_parity() -> None:
    torch.manual_seed(5)
    ref = _official_field()  # feat 8, width 16, depth 1
    ours = _tiny_field()
    ours.hexplane.load_state_dict(ref.grid.state_dict())
    pairs = ((ours.decoder.trunk[0], ref.feature_out[0]),
             (ours.decoder.pos_head, ref.pos_deform),
             (ours.decoder.scale_head, ref.scales_deform),
             (ours.decoder.rot_head, ref.rotations_deform))
    for o_mod, r_mod in pairs:
        if isinstance(o_mod, nn.Linear):
            o_mod.weight.data.copy_(r_mod.weight)
            o_mod.bias.data.copy_(r_mod.bias)
        else:
            for o, r in zip(o_mod, r_mod):
                if isinstance(o, nn.Linear):
                    o.weight.data.copy_(r.weight)
                    o.bias.data.copy_(r.bias)
    xyz, s, r, t = _inputs(n=7, seed=6)
    out = ours(xyz, s, r, t)
    # Official takes pos-encoded embs but slices raw prefixes: pad with zeros.
    pad3 = torch.zeros(7, 5)
    pad4 = torch.zeros(7, 6)
    pts, scl, rot, op, sh = ref.forward_dynamic(
        torch.cat([xyz, pad3], -1), torch.cat([s, pad3], -1),
        torch.cat([r, pad4], -1), torch.zeros(7, 1), torch.zeros(7, 16, 3),
        None, t)
    assert torch.equal(out.xyz, pts)
    assert torch.equal(out.scaling, scl)
    assert torch.equal(out.rotation, rot)
    print("test_official_forward_parity: passed (bit-exact xyz/scaling/rotation)")


if __name__ == "__main__":
    test_shapes_and_time_forms()
    test_zero_deltas_is_identity()
    test_manual_delta_values()
    test_time_sensitivity_with_trained_planes()
    test_raw_spaces_preserved()
    test_quaternion_multiply_matches_official()
    test_apply_rotation_option()
    test_parameter_groups_disjoint_and_complete()
    test_gradients_end_to_end()
    test_official_forward_parity()
    print("test_field.py: all 10 tests passed")
