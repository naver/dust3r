import unittest

import torch

from dust3r.utils.geometry import normalize_pointcloud


class UnmaskedPointCountTest(unittest.TestCase):
    def test_implicit_and_explicit_all_valid_masks_agree(self):
        points = torch.arange(1, 25, dtype=torch.float64).reshape(2, 2, 2, 3)
        for mode in ("avg_dis", "avg_log1p", "avg_warp-log1p"):
            with self.subTest(mode=mode):
                implicit = normalize_pointcloud(points, None, norm_mode=mode)
                explicit = normalize_pointcloud(points, None, norm_mode=mode, valid1=torch.ones(2, 2, 2, dtype=torch.bool))
                torch.testing.assert_close(implicit, explicit)

    def test_joint_average_counts_points_from_both_maps(self):
        first = torch.tensor([[[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]]], dtype=torch.float64)
        second = torch.tensor([[[[0.0, 0.0, 8.0]]]], dtype=torch.float64)
        actual_first, actual_second = normalize_pointcloud(first, second)
        factor = (3 + 4 + 8) / 3
        torch.testing.assert_close(actual_first, first / factor)
        torch.testing.assert_close(actual_second, second / factor)

    def test_mixed_mask_and_unmasked_view_use_same_units(self):
        first = torch.tensor([[[[3.0, 0.0, 0.0], [0.0, 100.0, 0.0]]]], dtype=torch.float64)
        second = torch.tensor([[[[0.0, 0.0, 9.0]]]], dtype=torch.float64)
        mask = torch.tensor([[[True, False]]])
        actual_first, actual_second = normalize_pointcloud(first, second, valid1=mask)
        torch.testing.assert_close(actual_first, first / 6)
        torch.testing.assert_close(actual_second, second / 6)

    def test_maskless_normalization_gradient_matches_distance_average(self):
        points = torch.arange(1, 13, dtype=torch.float64).reshape(1, 2, 2, 3).requires_grad_()
        actual = normalize_pointcloud(points, None)
        expected = points / points.norm(dim=-1).mean()
        actual_gradient = torch.autograd.grad(actual.square().sum(), points, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected.square().sum(), points)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
