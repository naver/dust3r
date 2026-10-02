import unittest

import numpy as np
import torch

from dust3r.utils.geometry import xy_grid


class HomogeneousGridAxisTest(unittest.TestCase):
    def test_numpy_unsqueeze_preserves_all_homogeneous_coordinates(self):
        grid = xy_grid(3, 2, origin=(5, -2), homogeneous=True, unsqueeze=0)
        expected = np.array([[[[5, -2, 1], [6, -2, 1], [7, -2, 1]],
                              [[5, -1, 1], [6, -1, 1], [7, -1, 1]]]])
        np.testing.assert_array_equal(grid, expected)

    def test_torch_grid_retains_homogeneous_coordinate_at_each_axis(self):
        reference = torch.tensor([[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]]])
        for axis in (0, 1, 2, -1):
            with self.subTest(axis=axis):
                grid = xy_grid(2, 2, device="cpu", homogeneous=True, unsqueeze=axis, cat_dim=None)
                self.assertEqual(len(grid), 3)
                actual = torch.stack(grid, -1)
                torch.testing.assert_close(actual, reference.float().unsqueeze(axis if axis >= 0 else axis + 3))

    def test_nonhomogeneous_numpy_and_torch_agree(self):
        for axis in (0, 1, -1):
            with self.subTest(axis=axis):
                numpy_grid = xy_grid(3, 2, unsqueeze=axis)
                torch_grid = xy_grid(3, 2, device="cpu", unsqueeze=axis)
                np.testing.assert_array_equal(numpy_grid, torch_grid.numpy())


if __name__ == "__main__":
    unittest.main()
