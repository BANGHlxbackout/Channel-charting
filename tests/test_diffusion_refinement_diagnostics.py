import unittest

import numpy as np

from diagnostics.diagnose_diffusion_refinement import finite_corrcoef, knn_indices, median_neighbor_lengths, row_overlap


class DiffusionRefinementDiagnosticsTests(unittest.TestCase):
    def test_row_overlap_returns_fraction_of_shared_neighbors(self):
        left = np.array([[1, 2, 3], [0, 2, 3]], dtype=np.int32)
        right = np.array([[3, 2, 4], [4, 5, 6]], dtype=np.int32)

        overlap = row_overlap(left, right, k=3)

        np.testing.assert_allclose(overlap, np.array([2 / 3, 0.0], dtype=np.float32))

    def test_median_neighbor_lengths_uses_per_row_neighbors(self):
        points = np.array(
            [
                [0.0, 0.0],
                [3.0, 4.0],
                [6.0, 8.0],
            ],
            dtype=np.float32,
        )
        neighbors = np.array([[1, 2], [0, 2], [1, 0]], dtype=np.int32)

        lengths = median_neighbor_lengths(points, neighbors)

        np.testing.assert_allclose(lengths, np.array([7.5, 5.0, 7.5], dtype=np.float32))

    def test_knn_indices_excludes_self(self):
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float32,
        )

        neighbors = knn_indices(points, k=1)

        np.testing.assert_array_equal(neighbors.ravel(), np.array([1, 0, 1], dtype=np.int32))

    def test_finite_corrcoef_ignores_non_finite_values(self):
        x = np.array([1.0, 2.0, 3.0, np.nan])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        corr = finite_corrcoef(x, y)

        self.assertAlmostEqual(corr, 1.0)


if __name__ == "__main__":
    unittest.main()
