import unittest

import numpy as np

from run_euclidean_knn_refined_charting import compute_embedding_distances


class EuclideanKnnRefinementTests(unittest.TestCase):
    def test_compute_embedding_distances_returns_pairwise_euclidean_distances(self):
        embedding = np.array(
            [
                [0.0, 0.0],
                [3.0, 4.0],
                [6.0, 8.0],
            ],
            dtype=np.float32,
        )

        distances = compute_embedding_distances(embedding)

        expected = np.array(
            [
                [0.0, 5.0, 10.0],
                [5.0, 0.0, 5.0],
                [10.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(distances, expected)
        self.assertEqual(distances.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
