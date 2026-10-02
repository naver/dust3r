"""Dense exponential point regression has an identity Jacobian at zero."""

import pytest
import torch

from dust3r.heads.postprocess import postprocess, reg_dense_depth

EXP = ("exp", -float("inf"), float("inf"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_radial_exp_origin_jacobian_is_identity(dtype):
    value = torch.zeros(3, dtype=dtype, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda x: reg_dense_depth(x, EXP), value
    )
    torch.testing.assert_close(jacobian, torch.eye(3, dtype=dtype))


def test_radial_exp_gradcheck_includes_zero():
    value = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: reg_dense_depth(x, EXP), (value,))


def test_small_vectors_keep_the_identity_limit():
    value = torch.tensor(
        [[1e-12, -2e-12, 3e-12]], dtype=torch.float64, requires_grad=True
    )
    result = reg_dense_depth(value, EXP)
    torch.testing.assert_close(result, value, atol=0, rtol=1e-10)
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value), atol=1e-10, rtol=0)


def test_nonzero_radial_value_and_jacobian_follow_closed_form():
    value = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64, requires_grad=True)
    radius = value.detach().norm()
    scale = radius.expm1() / radius
    coefficient = (radius * radius.exp() - radius.expm1()) / radius**3
    expected_jacobian = scale * torch.eye(
        3, dtype=value.dtype
    ) + coefficient * torch.outer(value.detach(), value.detach())
    torch.testing.assert_close(reg_dense_depth(value, EXP), value.detach() * scale)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(lambda x: reg_dense_depth(x, EXP), value),
        expected_jacobian,
    )


def test_postprocess_can_update_zero_dense_predictions():
    raw = torch.nn.Parameter(torch.zeros((1, 4, 2, 2), dtype=torch.float64))
    target = torch.tensor([1.0, -2.0, 3.0], dtype=raw.dtype).reshape(1, 1, 1, 3)
    optimizer = torch.optim.SGD([raw], lr=0.05)
    predictions = postprocess(raw, EXP, ("exp", 1, float("inf")))
    initial_loss = (predictions["pts3d"] - target).square().mean()
    initial_loss.backward()
    expected = (-2 * target / predictions["pts3d"].numel()).expand_as(
        predictions["pts3d"]
    )
    torch.testing.assert_close(raw.grad[:, :3], expected.permute(0, 3, 1, 2))
    optimizer.step()
    updated = postprocess(raw, EXP, ("exp", 1, float("inf")))["pts3d"]
    assert (updated - target).square().mean() < initial_loss


def test_linear_and_square_modes_remain_unchanged():
    value = torch.tensor([[0.2, -0.3, 0.4], [0, 0, 0]], dtype=torch.float64)
    torch.testing.assert_close(reg_dense_depth(value, ("linear", *EXP[1:])), value)
    torch.testing.assert_close(
        reg_dense_depth(value, ("square", *EXP[1:])),
        value * value.norm(dim=-1, keepdim=True),
    )
