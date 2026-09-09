"""CPU tests for deformation/hexplane.py. Run via `python3 -m tests.test_hexplane`."""

import importlib.util
import sys

import torch

from deformation.hexplane import (
    PLANE_NAMES,
    HexPlaneConfig,
    HexPlaneField,
    interpolate_ms_features,
    normalize_aabb,
)


def _tiny_config() -> HexPlaneConfig:
    return HexPlaneConfig(bounds=1.6, output_coordinate_dim=4,
                          resolution=[4, 4, 4, 3], multires=[1, 2])


def test_output_shape_and_determinism() -> None:
    torch.manual_seed(0)
    field = HexPlaneField(_tiny_config())
    xyz = torch.randn(7, 3)
    out = field(xyz, torch.rand(7, 1))
    assert out.shape == (7, 8), out.shape  # 4 dims x 2 levels
    assert field.feat_dim == 8
    t = torch.full((7, 1), 0.3)
    assert torch.equal(field(xyz, t), field(xyz, t))  # deterministic
    assert torch.equal(field(xyz, 0.3), field(xyz, t))  # scalar broadcast


def test_plane_ordering() -> None:
    assert PLANE_NAMES == ("xy", "xz", "xt", "yz", "yt", "zt")


def test_aabb_normalization_formula() -> None:
    field = HexPlaneField(_tiny_config())
    field.set_aabb(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1.0, 0.0, 1.0]))
    got = normalize_aabb(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]), field.aabb)
    # Official mapping: aabb[0] (=max) -> -1, aabb[1] (=min) -> +1.
    assert torch.allclose(got, torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), atol=1e-6)
    assert torch.equal(field.get_aabb[0], torch.tensor([1.0, 2.0, 3.0]))


def test_hand_filled_bilinear() -> None:
    cfg = HexPlaneConfig(bounds=1.0, output_coordinate_dim=1,
                         resolution=[2, 2, 2, 2], multires=[1])
    field = HexPlaneField(cfg)
    with torch.no_grad():
        for plane in field.grids[0]:
            plane.fill_(1.0)
        # xy plane (index 0), layout [1,1,res_y=2,res_x=2], corners a/b/c/d.
        field.grids[0][0].copy_(torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]]))
    field.set_aabb(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([-1.0, -1.0, -1.0]))
    # With aabb=[max,min], world +1 -> -1 (grid row/col 0) and -1 -> +1.
    # Grid rows index y, cols index x: [[y0x0, y0x1], [y1x0, y1x1]].
    q = torch.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    out = field(q, torch.zeros(3, 1))
    assert torch.allclose(out.squeeze(-1), torch.tensor([2.0, 8.0, 6.0]), atol=1e-5)


def _trained_like(field: HexPlaneField, seed: int = 0) -> None:
    """Imprint varying values on time planes (at init they are all ones,
    so timestamps legitimately have no effect until training moves them)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for level in field.grids:
            for i in (2, 4, 5):
                level[i].copy_(torch.rand_like(level[i]))


def test_all_six_planes_used_and_time_sensitivity() -> None:
    torch.manual_seed(1)
    field = HexPlaneField(_tiny_config())
    _trained_like(field)
    xyz = torch.randn(5, 3)
    t = torch.full((5, 1), 0.4)
    base = field(xyz, t)
    for i in range(6):
        with torch.no_grad():
            saved = field.grids[0][i].clone()
            field.grids[0][i].zero_()
        changed = not torch.equal(base, field(xyz, t))
        with torch.no_grad():
            field.grids[0][i].copy_(saved)
        assert changed, f"plane {i} ({PLANE_NAMES[i]}) has no effect"
    assert not torch.equal(base, field(xyz, torch.full((5, 1), 0.9)))
    assert not torch.equal(base, field(xyz + 0.5, t))


def test_multires_concat_order() -> None:
    torch.manual_seed(2)
    field = HexPlaneField(_tiny_config())
    xyz = torch.randn(4, 3)
    t = torch.full((4, 1), 0.5)
    with torch.no_grad():
        for plane in field.grids[1]:
            plane.zero_()
    out = field(xyz, t)
    assert (out[:, 4:] == 0).all() and out[:, :4].abs().sum() > 0


def test_gradients_to_inputs_and_params() -> None:
    torch.manual_seed(3)
    field = HexPlaneField(_tiny_config())
    _trained_like(field, seed=3)
    xyz = torch.randn(6, 3, requires_grad=True)
    t = torch.full((6, 1), 0.5, requires_grad=True)
    field(xyz, t).sum().backward()
    for name, g in (("xyz", xyz.grad), ("time", t.grad)):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name
    for p in field.get_grid_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), "grid param"
    assert sum(p.grad.abs().sum().item() for p in field.get_grid_parameters()) > 0


def test_official_parity() -> None:
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    spec = importlib.util.spec_from_file_location(
        "official_hexplane", "/tmp/opencode/4DGaussians/scene/hexplane.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    torch.manual_seed(11)
    cfg = HexPlaneConfig(bounds=1.6, output_coordinate_dim=8,
                         resolution=[8, 8, 8, 5], multires=[1, 2])
    ours = HexPlaneField(cfg)
    ref = official.HexPlaneField(
        1.6,
        {"grid_dimensions": 2, "input_coordinate_dim": 4,
         "output_coordinate_dim": 8, "resolution": [8, 8, 8, 5]},
        [1, 2],
    )
    ref.load_state_dict(ours.state_dict())  # identical layout -> bit-shared weights
    xyz = torch.randn(16, 3)
    t = torch.rand(16, 1)
    a = ref(xyz, t)
    b = ours(xyz, t)
    assert a.shape == b.shape == (16, 16)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
    # Reverse direction with official-initialized weights + custom AABB.
    ref2 = official.HexPlaneField(
        1.6,
        {"grid_dimensions": 2, "input_coordinate_dim": 4,
         "output_coordinate_dim": 8, "resolution": [8, 8, 8, 5]},
        [1, 2],
    )
    ours2 = HexPlaneField(cfg)
    ours2.load_state_dict(ref2.state_dict())
    xyz_max = [1.2, 0.8, 2.0]
    xyz_min = [-1.1, -0.9, 0.5]
    ref2.set_aabb(xyz_max, xyz_min)
    ours2.set_aabb(torch.tensor(xyz_max), torch.tensor(xyz_min))
    a2 = ref2(xyz, t)
    b2 = ours2(xyz, t)
    assert torch.allclose(a2, b2, atol=1e-6), (a2 - b2).abs().max()
    print(f"test_official_parity: passed (max diff {(a2 - b2).abs().max().item():.2e})")


if __name__ == "__main__":
    test_output_shape_and_determinism()
    test_plane_ordering()
    test_aabb_normalization_formula()
    test_hand_filled_bilinear()
    test_all_six_planes_used_and_time_sensitivity()
    test_multires_concat_order()
    test_gradients_to_inputs_and_params()
    test_official_parity()
    print("test_hexplane.py: all 8 tests passed")
