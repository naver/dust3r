import unittest

import torch

from dust3r.utils.geometry import normalize_pointcloud


class SingleMapFactorTest(unittest.TestCase):
    def test_single_map_returns_cloud_and_factor(self):
        points = torch.arange(1, 13, dtype=torch.float64).reshape(1, 2, 2, 3)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        for mode in ("avg_dis", "median_dis", "sqrt_dis"):
            with self.subTest(mode=mode):
                normalized, factor = normalize_pointcloud(points, None, mode, valid1=mask, ret_factor=True)
                self.assertEqual(factor.shape, (1, 1, 1, 1))
                torch.testing.assert_close(normalized * factor, points)
                torch.testing.assert_close(normalized, normalize_pointcloud(points, None, mode, valid1=mask))

    def test_paired_maps_keep_three_element_return(self):
        first = torch.arange(1, 13, dtype=torch.float64).reshape(1, 2, 2, 3)
        second = first + 1
        first_mask = torch.ones(1, 2, 2, dtype=torch.bool)
        result = normalize_pointcloud(first, second, valid1=first_mask, valid2=first_mask, ret_factor=True)
        self.assertEqual(len(result), 3)
        torch.testing.assert_close(result[0] * result[2], first)
        torch.testing.assert_close(result[1] * result[2], second)

    def test_single_map_factor_preserves_autograd(self):
        points = torch.arange(1, 13, dtype=torch.float64).reshape(1, 2, 2, 3).requires_grad_()
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        normalized, factor = normalize_pointcloud(points, None, valid1=mask, ret_factor=True)
        gradient = torch.autograd.grad((normalized * factor).sum(), points)[0]
        torch.testing.assert_close(gradient, torch.ones_like(points))


if __name__ == "__main__":
    unittest.main()
