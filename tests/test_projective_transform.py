import unittest

import numpy as np
import torch

from dust3r.utils.geometry import geotrf


class ProjectiveTransformTest(unittest.TestCase):
    def inputs(self):
        points = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3) / 7 + 0.1
        transforms = np.array([[[1.2, 0.1, 0, 1], [0, 0.8, 0.2, -2], [0.3, 0, 1, 0.5],
                                [0.1, -0.05, 0.2, 1]],
                               [[0.9, 0, 0.1, -1], [0.1, 1.1, 0, 3], [0, 0.2, 1, 0.25],
                                [0.05, 0.1, -0.1, 1]]])
        return points, transforms

    def test_batched_image_path_matches_explicit_homogeneous_projection(self):
        points, transforms = self.inputs()
        expected = np.empty_like(points)
        for batch in range(2):
            for row in range(2):
                for column in range(2):
                    homogeneous = transforms[batch] @ np.append(points[batch, row, column], 1)
                    expected[batch, row, column] = homogeneous[:3] / homogeneous[3]
        actual = geotrf(torch.from_numpy(transforms), torch.from_numpy(points), norm=True)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12)
        np.testing.assert_allclose(geotrf(transforms, points, norm=True), expected, rtol=1e-12)

    def test_affine_homogeneous_projection_does_not_divide_by_depth(self):
        points, transforms = self.inputs()
        transforms[:, 3] = [0, 0, 0, 1]
        tensor_points = torch.from_numpy(points)
        tensor_transforms = torch.from_numpy(transforms)
        actual = geotrf(tensor_transforms, tensor_points, norm=True)
        expected = geotrf(tensor_transforms, tensor_points, norm=False)
        torch.testing.assert_close(actual, expected)

    def test_projective_gradients_match_homogeneous_matrix_oracle(self):
        points, transforms = self.inputs()
        points = torch.tensor(points, requires_grad=True)
        transforms = torch.tensor(transforms, requires_grad=True)
        actual = geotrf(transforms, points, norm=2.0)
        homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
        projected = torch.einsum("bij,bhwj->bhwi", transforms, homogeneous)
        expected = 2 * projected[..., :3] / projected[..., 3:]
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.square().sum(), (points, transforms), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (points, transforms))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
