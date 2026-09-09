"""REAL GPU integration test for the STATIC -> 4D seam (Commit 14).

Covers: representation boundary, no-mutation, time broadcast, dynamic
forward/backward into canonical+decoder+grids, viewspace grads, one
optimizer step, different-time renders, official pre-rasterizer + pixel
parity. Requires CUDA + the extension.

Run via `python3 -m tests.test_4d_integration`.
"""

import json
import os
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from data.scene import load_scene
from deformation.decoder import DecoderConfig
from deformation.field import DeformationField, FieldConfig
from deformation.hexplane import HexPlaneConfig
from deformation.render_4d import render_deformed_view
from gaussians.rasterizer import render_view
from training.step_4d import train_step_4d
from training.trainer_static import StaticTrainerConfig, init_static_model, set_seed


def _write_scene(root: str, size: int = 64) -> None:
    def c2w(tx=0.0):
        m = np.eye(4)
        m[:3, 3] = [tx, 0, 0]
        return m.tolist()

    train_frames = [
        {"file_path": f"train/r_{i}", "transform_matrix": c2w(0.1 * i), "time": t}
        for i, t in enumerate([0, 1])
    ]
    test_frames = [{"file_path": "test/r_0", "transform_matrix": c2w(0.0), "time": 1}]
    for name, frames in (("transforms_train.json", train_frames),
                         ("transforms_test.json", test_frames)):
        with open(os.path.join(root, name), "w") as f:
            json.dump({"camera_angle_x": 0.9, "frames": frames}, f)
    rng = np.random.default_rng(4)
    for rel in ("train/r_0", "train/r_1", "test/r_0"):
        arr = (rng.random((size, size, 4)) * 255).astype(np.uint8)
        arr[..., 3] = 255
        os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
        Image.fromarray(arr, "RGBA").save(os.path.join(root, rel + ".png"))


def _field(device, seed: int = 0) -> DeformationField:
    torch.manual_seed(seed)
    field = DeformationField(FieldConfig(
        hexplane=HexPlaneConfig(bounds=1.6, output_coordinate_dim=4,
                                resolution=[8, 8, 8, 5], multires=[1, 2]),
        decoder=DecoderConfig(width=16, depth=1),
    )).to(device)
    return field


def _trained_like_time_planes(field: DeformationField, seed: int = 9) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for level in field.hexplane.grids:
            for i in (2, 4, 5):
                level[i].copy_(torch.rand_like(level[i]) * 2 - 1)


def _near_identity_decoder(field: DeformationField, seed: int = 10) -> None:
    """Scale decoder weights down (NOT zero): deformations stay cm-scale so
    the cloud stays framed, while every Jacobian stays nonzero and gradients
    reach both decoder and HexPlane grids. (Zeroing the trunk would give
    d(delta)/d(feature) = 0 and starve the grids of gradients.)"""
    _ = seed
    with torch.no_grad():
        for p in field.decoder.parameters():
            p.mul_(0.05)


def _setup():
    device = torch.device("cuda")
    tmp = tempfile.TemporaryDirectory()
    _write_scene(tmp.name)
    set_seed(0)
    scene = load_scene(tmp.name, seed=0)
    cfg = StaticTrainerConfig(iterations=1, seed=0, max_sh_degree=0,
                              densify_from_iter=100)
    model = init_static_model(scene, cfg, device=device)
    with torch.no_grad():  # small cloud placed where D-NeRF cameras look (-z)
        model._xyz.copy_(torch.randn_like(model._xyz) * 0.5 + torch.tensor([0, 0, -2.5], device=device))
        model._scaling.copy_(torch.full_like(model._scaling, float(np.log(0.15))))
    field = _field(device)
    field.set_aabb(model._xyz.detach().max(dim=0).values + 0.5,
                   model._xyz.detach().min(dim=0).values - 0.5)
    return tmp, scene, model, field, device


def test_no_mutation_and_broadcast() -> None:
    tmp, scene, model, field, device = _setup()
    try:
        before = [p.detach().clone() for p in
                  (model._xyz, model._scaling, model._rotation, model._opacity)]
        view = scene.test_views[0]
        a = render_deformed_view(view.camera, model, field, device=device)
        for p, b in zip((model._xyz, model._scaling, model._rotation, model._opacity), before):
            assert torch.equal(p.detach(), b), "canonical params mutated by forward"
        assert a["state"].xyz.shape == (model.num_points, 3)
        assert a["state"].scaling.shape == (model.num_points, 3)
        assert a["state"].rotation.shape == (model.num_points, 4)
        # Scalar time == explicit [N, 1] time.
        t1 = view.camera.time
        s1 = field(model.get_xyz, model._scaling, model._rotation, t1).xyz
        s2 = field(model.get_xyz, model._scaling, model._rotation,
                   torch.full((model.num_points, 1), t1, device=device)).xyz
        assert torch.equal(s1, s2)
        print("test_no_mutation_and_broadcast: passed")
    finally:
        tmp.cleanup()


def test_dynamic_forward_backward_and_step() -> None:
    tmp, scene, model, field, device = _setup()
    try:
        _trained_like_time_planes(field)
        _near_identity_decoder(field)
        view = scene.test_views[0]
        gt = view.image.to(device)
        opt = torch.optim.Adam(
            [model._xyz, model._scaling, model._rotation, model._opacity,
             model._features_dc, model._features_rest]
            + field.get_grid_parameters() + field.get_mlp_parameters(),
            lr=1e-3, eps=1e-15)
        opt.zero_grad(set_to_none=True)
        out = train_step_4d(view.camera, gt, model, field, device=device)
        assert torch.isfinite(out["loss"]) and torch.isfinite(out["l1"])
        assert out["pkg"]["visibility_filter"].any(), "nothing visible"
        assert torch.isfinite(out["pkg"]["viewspace_points"].grad).all()
        for name, p in (("xyz", model._xyz), ("scaling", model._scaling),
                        ("rotation", model._rotation)):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
        for group, ps in (("grid", field.get_grid_parameters()),
                          ("mlp", field.get_mlp_parameters())):
            mass = sum(p.grad.abs().sum().item() for p in ps if p.grad is not None)
            assert mass > 0, group
        shapes_before = {n: p.shape for n, p in
                         (("xyz", model._xyz), ("grid0", field.hexplane.grids[0][0]))}
        before = (model._xyz.detach().clone(),
                  field.decoder.trunk[0].weight.detach().clone())
        opt.step()
        assert model._xyz.shape == shapes_before["xyz"]
        assert not torch.equal(model._xyz.detach(), before[0]), "canonical xyz unchanged"
        assert not torch.equal(field.decoder.trunk[0].weight.detach(), before[1])
        print(f"test_dynamic_forward_backward_and_step: passed "
              f"(loss={out['loss'].item():.4f}, "
              f"|dL/dxyz|={model._xyz.grad.abs().sum().item():.3e})")
    finally:
        tmp.cleanup()


def test_different_times_differ() -> None:
    tmp, scene, model, field, device = _setup()
    try:
        _trained_like_time_planes(field)
        _near_identity_decoder(field)
        view = scene.test_views[0]
        import copy
        late = copy.copy(view.camera)
        late.time = 0.05
        a = render_deformed_view(view.camera, model, field, device=device)["render"]
        b = render_deformed_view(late, model, field, device=device)["render"]
        assert not torch.equal(a, b), "renders identical across times"
        c = render_deformed_view(view.camera, model, field, device=device)["render"]
        assert torch.allclose(a, c, atol=1e-6), "same time not deterministic"
        print(f"test_different_times_differ: passed "
              f"(mean|a-b|={(a - b).abs().mean().item():.3e})")
    finally:
        tmp.cleanup()


def _official_modules():
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
    from scene.deformation import deform_network
    from gaussian_renderer import render as official_render
    return deform_network, official_render


def test_official_parity() -> None:
    deform_network, official_render = _official_modules()
    tmp, scene, model, field, device = _setup()
    try:
        import torch.nn as nn
        import torch.nn.functional as Fmod

        args = SimpleNamespace(
            no_grid=False, empty_voxel=False, static_mlp=False, bounds=1.6,
            kplanes_config={"grid_dimensions": 2, "input_coordinate_dim": 4,
                            "output_coordinate_dim": 4, "resolution": [8, 8, 8, 5]},
            multires=[1, 2], no_dx=False, no_ds=False, no_dr=False,
            no_do=True, no_dshs=True, apply_rotation=False,
            net_width=16, timebase_pe=4, defor_depth=1, posebase_pe=10,
            scale_rotation_pe=2, opacity_pe=2, timenet_width=64,
            timenet_output=32, grid_pe=0)
        ref = deform_network(args).to(device)
        wrapper = ref  # official render() calls the wrapper with 6 raw args
        ref = ref.deformation_net  # inner module (wrapper only adds dead encodings)
        ref.grid.load_state_dict(field.hexplane.state_dict())
        pairs = ((field.decoder.trunk[0], ref.feature_out[0]),
                 (field.decoder.pos_head, ref.pos_deform),
                 (field.decoder.scale_head, ref.scales_deform),
                 (field.decoder.rot_head, ref.rotations_deform))
        for o_mod, r_mod in pairs:
            if isinstance(o_mod, nn.Linear):
                r_mod.weight.data.copy_(o_mod.weight.data)
                r_mod.bias.data.copy_(o_mod.bias.data)
            else:
                for o, r in zip(o_mod, r_mod):
                    if isinstance(o, nn.Linear):
                        r.weight.data.copy_(o.weight.data)
                        r.bias.data.copy_(o.bias.data)
        _trained_like_time_planes(field)
        ref.grid.load_state_dict(field.hexplane.state_dict())

        view = scene.test_views[0]
        n = model.num_points
        t = torch.full((n, 1), float(view.camera.time), device=device)
        state = field(model.get_xyz, model._scaling, model._rotation, t,
                      opacity=model._opacity, shs=model.get_features)
        # Official wrapper slices raw prefixes out of its (dead) encodings,
        # so passing raw tensors must reproduce our field bit-exactly.
        pts, scl, rot, op, _ = wrapper(
            model.get_xyz, model._scaling, model._rotation,
            model._opacity, model.get_features, t)
        assert torch.equal(state.xyz, pts)
        assert torch.equal(state.scaling, scl)
        assert torch.equal(state.rotation, rot)

        # Pixel parity: official render() with a thin fake pc namespace.
        bg = torch.ones(3, dtype=torch.float32, device=device)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=False)
        fake_pc = SimpleNamespace(
            get_xyz=model.get_xyz, _opacity=model._opacity,
            get_features=model.get_features, _scaling=model._scaling,
            _rotation=model._rotation, _deformation=wrapper,
            _deformation_table=torch.ones(n, dtype=torch.bool, device=device),
            scaling_activation=torch.exp,
            rotation_activation=Fmod.normalize,
            opacity_activation=torch.sigmoid,
            active_sh_degree=0, max_sh_degree=0)
        ref_pkg = official_render(view.camera, fake_pc, pipe, bg,
                                  stage="fine", cam_type="blender")
        ours_pkg = render_deformed_view(view.camera, model, field,
                                        bg_color=(1.0, 1.0, 1.0), device=device)
        assert torch.allclose(ref_pkg["render"], ours_pkg["render"], atol=1e-6), \
            (ref_pkg["render"] - ours_pkg["render"]).abs().max()
        print("test_official_parity: passed (bit-exact pre-rasterizer + pixel 1e-6)")
    finally:
        tmp.cleanup()


if __name__ == "__main__":
    test_no_mutation_and_broadcast()
    test_dynamic_forward_backward_and_step()
    test_different_times_differ()
    test_official_parity()
    print("test_4d_integration.py: done")
