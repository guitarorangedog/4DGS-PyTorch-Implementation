"""CPU tests for deformation/regularize.py. Run via `python3 -m tests.test_regularize`."""

import importlib.util
import sys

import torch

from deformation.decoder import DecoderConfig
from deformation.field import DeformationField, FieldConfig
from deformation.hexplane import HexPlaneConfig
from deformation.regularize import (
    SPATIAL_PLANE_IDS,
    TIME_PLANE_IDS,
    RegWeights,
    l1_time_planes_term,
    plane_total_variation,
    regularization_terms,
    second_difference_smoothness,
    spatial_smoothness_term,
    temporal_smoothness_term,
    weighted_regularization,
)


def _tiny_field(**over) -> DeformationField:
    kw = dict(
        hexplane=HexPlaneConfig(bounds=1.6, output_coordinate_dim=2,
                                resolution=[5, 5, 5, 4], multires=[1, 2]),
        decoder=DecoderConfig(width=8, depth=1),
    )
    kw.update(over)
    return DeformationField(FieldConfig(**kw))


def test_known_values() -> None:
    # First-order TV: [[0,1],[2,3]] -> 2*((8/2)+(2/2)) = 10.
    tv = plane_total_variation(torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]))
    assert abs(tv.item() - 10.0) < 1e-5, tv.item()
    # Second-difference: constant and linear -> 0; quadratic i^2 -> 4.
    assert second_difference_smoothness(torch.ones(1, 1, 6, 3)).item() == 0.0
    ramp = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).expand(1, 1, 6, 3)
    assert second_difference_smoothness(ramp).item() == 0.0
    quad = (torch.arange(5, dtype=torch.float32) ** 2).view(1, 1, 5, 1).expand(1, 1, 5, 2)
    assert abs(second_difference_smoothness(quad).item() - 4.0) < 1e-5
    # L1 reference: ones -> 0, zeros -> 1.
    field = _tiny_field()
    with torch.no_grad():
        for level in field.hexplane.grids:
            for i in TIME_PLANE_IDS:
                level[i].fill_(1.0)
    assert l1_time_planes_term(field.hexplane.grids).item() == 0.0
    with torch.no_grad():
        for level in field.hexplane.grids:
            for i in TIME_PLANE_IDS:
                level[i].zero_()
    assert abs(l1_time_planes_term(field.hexplane.grids).item() - 6.0) < 1e-6  # 3 planes x 2 levels


def test_plane_selectivity() -> None:
    torch.manual_seed(0)
    field = _tiny_field()
    with torch.no_grad():
        for level in field.hexplane.grids:
            for i in range(6):
                level[i].copy_(torch.rand_like(level[i]))
        for level in field.hexplane.grids:
            for i in SPATIAL_PLANE_IDS:
                level[i].zero_()
    terms = regularization_terms(field)
    assert terms.spatial.item() == 0.0
    assert terms.temporal.item() > 0 and terms.l1_time.item() > 0
    with torch.no_grad():
        for level in field.hexplane.grids:
            for i in TIME_PLANE_IDS:
                level[i].fill_(1.0)
    terms = regularization_terms(field)
    assert terms.temporal.item() == 0.0 and terms.l1_time.item() == 0.0
    assert terms.spatial.item() == 0.0  # spatial still zeroed


def test_all_levels_participate() -> None:
    torch.manual_seed(1)
    field = _tiny_field()
    with torch.no_grad():  # time planes are ones at init -> install variation
        for level in field.hexplane.grids:
            for i in TIME_PLANE_IDS:
                level[i].copy_(torch.rand_like(level[i]))
    full = regularization_terms(field)
    with torch.no_grad():
        for plane in field.hexplane.grids[1]:
            plane.zero_()
    dropped = regularization_terms(field)
    assert dropped.spatial < full.spatial and dropped.temporal < full.temporal
    # Single-level manual sum matches the level-0 contribution.
    manual = sum(second_difference_smoothness(field.hexplane.grids[0][i]).item()
                 for i in SPATIAL_PLANE_IDS)
    assert abs(dropped.spatial.item() - manual) < 1e-6


def test_weighted_sum_and_finiteness() -> None:
    torch.manual_seed(2)
    field = _tiny_field()
    terms = regularization_terms(field)
    for t in (terms.spatial, terms.temporal, terms.l1_time):
        assert t.dim() == 0 and torch.isfinite(t)
    w = RegWeights(plane_tv=0.5, time_smoothness=2.0, l1_time_planes=0.25)
    total = weighted_regularization(field, w)
    expected = 0.5 * terms.spatial + 2.0 * terms.temporal + 0.25 * terms.l1_time
    assert torch.allclose(total, expected, atol=1e-9)
    dflt = RegWeights()
    assert (dflt.plane_tv, dflt.time_smoothness, dflt.l1_time_planes) == (0.0001, 0.01, 0.0001)


def test_gradient_isolation() -> None:
    torch.manual_seed(3)
    field = _tiny_field()
    total = weighted_regularization(field)
    total.backward()
    grid_with_grad = sum(1 for p in field.get_grid_parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
    assert grid_with_grad > 0
    for p in field.get_mlp_parameters():
        assert p.grad is None, "decoder must not be regularized"
    # Frozen AABB carries no grad.
    assert field.hexplane.aabb.grad is None


def test_official_parity() -> None:
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    spec = importlib.util.spec_from_file_location(
        "official_regulation", "/tmp/opencode/4DGaussians/scene/regulation.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    torch.manual_seed(4)
    field = _tiny_field()
    levels = field.hexplane.grids
    # Primitive parity on random planes.
    probe = torch.randn(1, 2, 5, 4)
    assert abs(official.compute_plane_smoothness(probe).item()
               - second_difference_smoothness(probe).item()) < 1e-9
    # Aggregation parity: mirror the official float32-tensor accumulation
    # (total = 0; total += term) — NOT a float64 sum of .item()s.
    ref_spatial = 0
    for level in levels:
        for i in (0, 1, 3):
            ref_spatial = ref_spatial + official.compute_plane_smoothness(level[i])
    ref_temporal = 0
    for level in levels:
        for i in (2, 4, 5):
            ref_temporal = ref_temporal + official.compute_plane_smoothness(level[i])
    ref_l1 = 0
    for level in levels:
        for i in (2, 4, 5):
            ref_l1 = ref_l1 + torch.abs(1 - level[i]).mean()
    terms = regularization_terms(field)
    assert torch.equal(terms.spatial, ref_spatial)
    assert torch.equal(terms.temporal, ref_temporal)
    assert torch.equal(terms.l1_time, ref_l1)
    print("test_official_parity: passed (primitives + index sets + aggregation)")


if __name__ == "__main__":
    test_known_values()
    test_plane_selectivity()
    test_all_levels_participate()
    test_weighted_sum_and_finiteness()
    test_gradient_isolation()
    test_official_parity()
    print("test_regularize.py: all 6 tests passed")
