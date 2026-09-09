"""CPU tests for gaussians/densify.py. Run via `python3 -m tests.test_densify`."""

import torch

from gaussians.densify import (
    OPTIM_GROUP_NAMES,
    add_densification_stats,
    attach_optimizer,
    densify,
    densify_and_clone,
    densify_and_split,
    init_densify_stats,
    mean_viewspace_grads,
    prune_by_opacity_and_size,
    prune_points,
    reset_opacity,
)
from gaussians.gaussian_model import CanonicalGaussianModel, inverse_sigmoid


def _model(n: int = 12, seed: int = 0, degree: int = 1) -> CanonicalGaussianModel:
    torch.manual_seed(seed)
    model = CanonicalGaussianModel(max_sh_degree=degree)
    model.create_from_pointcloud(torch.randn(n, 3), torch.rand(n, 3))
    attach_optimizer(model, lr=1e-3)
    init_densify_stats(model)
    # One dummy step so Adam exp_avg/exp_avg_sq exist (populated-state path).
    loss = (
        model._xyz.pow(2).sum()
        + model._scaling.pow(2).sum()
        + model._rotation.pow(2).sum()
        + model._opacity.pow(2).sum()
        + model._features_dc.pow(2).sum()
        + model._features_rest.pow(2).sum()
    )
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    return model


def _optim_shapes(model) -> dict[str, tuple]:
    return {
        g["name"]: tuple(g["params"][0].shape)
        for g in model.optimizer.param_groups
    }


def _assert_fully_aligned(model) -> None:
    n = model.num_points
    assert model._xyz.shape == (n, 3)
    assert model._scaling.shape == (n, 3)
    assert model._rotation.shape == (n, 4)
    assert model._opacity.shape == (n, 1)
    assert model._features_dc.shape[0] == n
    assert model._features_rest.shape[0] == n
    assert model.xyz_gradient_accum.shape == (n, 1)
    assert model.denom.shape == (n, 1)
    assert model.max_radii2D.shape == (n,)
    for g in model.optimizer.param_groups:
        assert g["params"][0].shape[0] == n, g["name"]
        state = model.optimizer.state.get(g["params"][0], None)
        if state is not None:
            assert state["exp_avg"].shape == g["params"][0].shape, g["name"]
            assert state["exp_avg_sq"].shape == g["params"][0].shape, g["name"]


def test_group_names_match_official() -> None:
    assert OPTIM_GROUP_NAMES == ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")


def test_stats_accumulation() -> None:
    model = _model(n=6)
    grads = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0],
                          [0.0, 2.0], [5.0, 12.0], [0.0, 0.0]])
    filt = torch.tensor([True, False, True, True, False, True])
    add_densification_stats(model, grads, filt)
    # Norms over :2 -> 5, 0, 1, 2, 13, 0; only filtered rows accumulate.
    assert torch.allclose(
        model.xyz_gradient_accum.squeeze(-1),
        torch.tensor([5.0, 0.0, 1.0, 2.0, 0.0, 0.0]),
        atol=1e-5,
    )
    assert torch.allclose(
        model.denom.squeeze(-1), torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    )
    assert torch.allclose(
        mean_viewspace_grads(model).squeeze(-1),
        torch.tensor([5.0, 0.0, 1.0, 2.0, 0.0, 0.0]),
        atol=1e-5,
    )


def test_clone_preserves_and_grows() -> None:
    model = _model(n=10)
    with torch.no_grad():
        model._scaling.copy_(torch.full((10, 3), -6.0))  # tiny -> clone-eligible
    grads = torch.full((10, 1), 1.0)
    selected = torch.tensor([True] * 4 + [False] * 6)
    grads[~selected] = 0.0
    before = {k: getattr(model, k).detach().clone() for k in
              ("_xyz", "_scaling", "_rotation", "_opacity")}
    n = densify_and_clone(model, grads, grad_threshold=0.5,
                          scene_extent=10.0, percent_dense=0.01)
    assert n == 4 and model.num_points == 14
    for k in ("_xyz", "_scaling", "_rotation", "_opacity"):
        assert torch.allclose(getattr(model, k).detach()[10:], before[k][selected], atol=1e-6), k
    _assert_fully_aligned(model)


def test_split_children_parents_parity() -> None:
    torch.manual_seed(0)
    model = _model(n=8, seed=3)
    with torch.no_grad():
        model._scaling.copy_(torch.full((8, 3), 2.0))  # huge -> split-eligible
        model._rotation.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1))
    torch.manual_seed(1234)
    grads = torch.full((8, 1), 1.0)
    parent_scale = model.get_scaling.detach().clone()
    m = densify_and_split(model, grads, grad_threshold=0.5,
                          scene_extent=10.0, percent_dense=0.01, num_children=2)
    assert m == 8  # all selected
    assert model.num_points == 16  # 8 parents removed, 16 children added
    # Shrink rule parity: child log-scale = log(parent_scale / (0.8 * 2)).
    expected = torch.log(parent_scale.repeat(2, 1) / (0.8 * 2))
    assert torch.allclose(model._scaling.detach(), expected, atol=1e-5)
    _assert_fully_aligned(model)


def test_split_deterministic_under_seed() -> None:
    def run_once() -> torch.Tensor:
        torch.manual_seed(0)
        model = _model(n=6, seed=5)
        with torch.no_grad():
            model._scaling.copy_(torch.full((6, 3), 2.0))
        torch.manual_seed(999)
        densify_and_split(model, torch.full((6, 1), 1.0), 0.5, 10.0, 0.01)
        return model._xyz.detach().clone()

    assert torch.allclose(run_once(), run_once(), atol=0)


def test_prune_consistency() -> None:
    model = _model(n=10)
    with torch.no_grad():
        model._opacity.copy_(inverse_sigmoid(torch.linspace(0.001, 0.5, 10).unsqueeze(-1)))
    removed = prune_by_opacity_and_size(model, min_opacity=0.05, scene_extent=5.0)
    assert removed == int((torch.linspace(0.001, 0.5, 10) < 0.05).sum()) == 1
    assert model.num_points == 9
    assert (model.get_opacity >= 0.05 - 1e-6).all()
    _assert_fully_aligned(model)


def test_prune_big_points_rule() -> None:
    model = _model(n=6)
    with torch.no_grad():
        model._scaling.copy_(torch.full((6, 3), 3.0))  # max scale e^3 >> 0.1*extent
    model.max_radii2D = torch.tensor([1.0, 1.0, 50.0, 1.0, 60.0, 1.0])
    removed = prune_by_opacity_and_size(model, min_opacity=1e-9,
                                        scene_extent=5.0, max_screen_size=20.0)
    # rows 2, 4 via screen size; all rows via world size (e^3 > 0.5) -> all 6.
    assert removed == 6 and model.num_points == 0
    _assert_fully_aligned(model)


def test_reset_opacity_consistent() -> None:
    model = _model(n=8)
    old_param = model._opacity
    reset_opacity(model)
    assert model._opacity is not old_param  # replaced, not mutated
    assert (model.get_opacity <= 0.01 + 1e-6).all()
    state = model.optimizer.state[model._opacity]
    assert (state["exp_avg"] == 0).all() and (state["exp_avg_sq"] == 0).all()
    _assert_fully_aligned(model)


def test_full_densify_pass() -> None:
    torch.manual_seed(0)
    model = _model(n=10, seed=8)
    with torch.no_grad():
        model._scaling.copy_(torch.cat((
            torch.full((5, 3), -6.0),  # small -> clone candidates
            torch.full((5, 3), 2.0),   # large -> split candidates
        )))
    model.xyz_gradient_accum = torch.full((10, 1), 1.0)
    model.denom = torch.ones(10, 1)
    torch.manual_seed(77)
    n_cloned, n_split = densify(model, grad_threshold=0.5, scene_extent=10.0)
    assert (n_cloned, n_split) == (5, 5)
    # 10 + 5 clones = 15, then 5 parents -> 10 children: 15 - 5 + 10 = 20.
    assert model.num_points == 20
    _assert_fully_aligned(model)


if __name__ == "__main__":
    test_group_names_match_official()
    test_stats_accumulation()
    test_clone_preserves_and_grows()
    test_split_children_parents_parity()
    test_split_deterministic_under_seed()
    test_prune_consistency()
    test_prune_big_points_rule()
    test_reset_opacity_consistent()
    test_full_densify_pass()
    print("test_densify.py: all 9 tests passed")
