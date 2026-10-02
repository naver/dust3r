"""Signed pose parameterizations must retain their slope at zero."""

import pytest
import torch

from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.cloud_opt.commons import signed_expm1, signed_log1p


@pytest.mark.parametrize("function", [signed_expm1, signed_log1p])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_transforms_have_the_analytic_first_derivative(function, dtype):
    values = torch.tensor([-5, -0.1, 0, 0.1, 5], dtype=dtype, requires_grad=True)
    result = function(values)
    magnitude = values.detach().abs()
    if function is signed_expm1:
        expected = values.detach().sign() * magnitude.expm1()
        derivative = magnitude.exp()
    else:
        expected = values.detach().sign() * magnitude.log1p()
        derivative = (1 + magnitude).reciprocal()
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(values.grad, derivative)


@pytest.mark.parametrize("function", [signed_expm1, signed_log1p])
def test_signed_transform_gradcheck_at_zero(function):
    values = torch.tensor([-0.1, 0, 0.1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(function, (values,))


def test_inverse_roundtrip_preserves_values_and_derivatives():
    values = torch.tensor(
        [-20, -0.1, 0, 0.1, 20], dtype=torch.float64, requires_grad=True
    )
    reconstructed = signed_log1p(signed_expm1(values))
    torch.testing.assert_close(reconstructed, values)
    reconstructed.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_optimizer_can_move_a_zero_translation():
    points = torch.zeros((1, 2, 2, 3))
    confidence = torch.ones((1, 2, 2))
    model = BasePCOptimizer(
        {"idx": [0]},
        {"idx": [1]},
        {"pts3d": points, "conf": confidence},
        {"pts3d_in_other_view": points, "conf": confidence},
        verbose=False,
    )
    with torch.no_grad():
        model.pw_poses.zero_()
        model.pw_poses[:, 3] = 1  # identity unit quaternion, xyzw
    target = torch.tensor([[1.0, -2.0, 3.0]])
    optimizer = torch.optim.SGD([model.pw_poses], lr=0.05)
    translation = model._get_poses(model.pw_poses)[:, :3, 3]
    initial_loss = (translation - target).square().sum()
    initial_loss.backward()
    torch.testing.assert_close(model.pw_poses.grad[:, 4:7], -2 * target)
    optimizer.step()
    updated = model._get_poses(model.pw_poses)[:, :3, 3]
    assert (updated - target).square().sum() < initial_loss
