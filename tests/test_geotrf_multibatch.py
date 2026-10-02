"""Geometric transforms must preserve every leading batch axis."""

import unittest

import numpy as np
import torch

from dust3r.utils.geometry import geotrf


class TestGeoTransformMultiBatch(unittest.TestCase):
    def test_torch_multibatch_output_and_gradients(self):
        if torch is None:
            self.skipTest("torch unavailable in exact-source NumPy runner")
        rng = np.random.default_rng(42)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        for point_shape in ((), (4,), (2, 4)):
            with self.subTest(point_shape=point_shape):
                transforms_np = np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy()
                transforms_np[..., :3, :3] = rotation
                shifts = rng.normal(size=(2, 3, 3))
                transforms_np[..., :3, 3] = shifts
                points_np = rng.normal(size=(2, 3) + point_shape + (3,))
                expected = points_np @ rotation.T + shifts.reshape(
                    (2, 3) + (1,) * len(point_shape) + (3,)
                )
                transforms = (
                    torch.tensor(transforms_np.swapaxes(0, 1).copy())
                    .transpose(0, 1)
                    .detach()
                    .requires_grad_(True)
                )
                points = (
                    torch.tensor(points_np.swapaxes(0, 1).copy())
                    .transpose(0, 1)
                    .detach()
                    .requires_grad_(True)
                )
                self.assertFalse(transforms.is_contiguous())
                self.assertFalse(points.is_contiguous())
                actual = geotrf(transforms, points)
                np.testing.assert_allclose(actual.detach().numpy(), expected)
                actual.square().sum().backward()
                np.testing.assert_allclose(
                    points.grad.numpy(), 2 * expected @ rotation, rtol=1e-12, atol=1e-12
                )
                self.assertTrue(torch.isfinite(transforms.grad).all())
                self.assertTrue(
                    torch.autograd.gradcheck(
                        geotrf, (transforms, points), fast_mode=True
                    )
                )

    def test_affine_transform_with_arbitrary_batch_and_point_axes(self):
        rng = np.random.default_rng(16)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        for batch_shape in ((2,), (2, 3), (2, 1, 3), (0,), (0, 2)):
            for point_shape in ((), (4,), (2, 4), (0,)):
                with self.subTest(batch_shape=batch_shape, point_shape=point_shape):
                    transforms = np.broadcast_to(np.eye(4), batch_shape + (4, 4)).copy()
                    transforms[..., :3, :3] = rotation
                    shifts = rng.normal(size=batch_shape + (3,))
                    transforms[..., :3, 3] = shifts
                    points = rng.normal(size=batch_shape + point_shape + (3,))
                    expected = points @ rotation.T + shifts.reshape(
                        batch_shape + (1,) * len(point_shape) + (3,)
                    )
                    original_points, original_transforms = (
                        points.copy(),
                        transforms.copy(),
                    )
                    actual = geotrf(transforms, points)
                    self.assertEqual(actual.shape, points.shape)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(points, original_points)
                    np.testing.assert_array_equal(transforms, original_transforms)

    def test_homogeneous_projection_with_multiple_batch_axes(self):
        transforms = np.broadcast_to(np.eye(3), (2, 3, 3, 3)).copy()
        transforms[..., 2, 2] = 2.0
        points = np.arange(18, dtype=float).reshape(2, 3, 3)
        points[..., 2] = 1.0
        actual = geotrf(transforms, points, ncol=2, norm=1.0)
        np.testing.assert_allclose(actual, points[..., :2] / 2.0)

    def test_mismatched_batch_axes_still_rejected(self):
        with self.assertRaisesRegex(AssertionError, "batch size does not match"):
            geotrf(np.zeros((2, 3, 4, 4)), np.zeros((2, 4, 3)))


if __name__ == "__main__":
    unittest.main()
