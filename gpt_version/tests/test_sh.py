"""CPU tests for gaussians/sh.py. Runnable via `python3 tests/test_sh.py`."""

import torch

from gaussians.sh import C0, eval_sh, num_sh_coeffs, rgb_to_sh, sh_to_rgb


def test_num_sh_coeffs() -> None:
    assert [num_sh_coeffs(d) for d in range(4)] == [1, 4, 9, 16]


def test_dc_roundtrip() -> None:
    rgb = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]])
    assert torch.allclose(sh_to_rgb(rgb_to_sh(rgb)), rgb, atol=1e-6)


def test_eval_sh_degree0() -> None:
    sh = torch.tensor([[[0.5], [1.0], [-0.25]]])  # [1, 3, 1]
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    out = eval_sh(0, sh, dirs)
    assert out.shape == (1, 3)
    assert torch.allclose(out, C0 * sh[..., 0], atol=1e-6)


def test_eval_sh_shapes_and_dc_consistency() -> None:
    torch.manual_seed(0)
    N = 5
    dirs = torch.randn(N, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    for deg, k in [(1, 4), (2, 9), (3, 16)]:
        sh = torch.randn(N, 3, k)
        out = eval_sh(deg, sh, dirs)
        assert out.shape == (N, 3)
        # Zeroing higher bands must reduce to the DC term.
        sh_dc_only = torch.zeros_like(sh)
        sh_dc_only[..., 0] = sh[..., 0]
        assert torch.allclose(
            eval_sh(deg, sh_dc_only, dirs), C0 * sh[..., 0], atol=1e-5
        )


def test_eval_sh_rejects_degree4() -> None:
    sh = torch.zeros(1, 3, 25)
    dirs = torch.zeros(1, 3)
    try:
        eval_sh(4, sh, dirs)
    except AssertionError:
        return
    raise AssertionError("eval_sh(4, ...) should be rejected (max degree 3)")


if __name__ == "__main__":
    test_num_sh_coeffs()
    test_dc_roundtrip()
    test_eval_sh_degree0()
    test_eval_sh_shapes_and_dc_consistency()
    test_eval_sh_rejects_degree4()
    print("test_sh.py: all 5 tests passed")
