import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = ROOT / "held-out"
for path in (ROOT, HELDOUT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from run_diffusion_refined_oos import (  # noqa: E402
    compute_oos_geodesic_block,
    compute_train_sigma,
    local_tikhonov_oos,
    select_sorted_topk,
)


class DiffusionRefinedOosTests(unittest.TestCase):
    def test_oos_geodesic_extends_through_entry_nodes(self):
        geodesic_train = np.array(
            [
                [0.0, 2.0, 5.0, 9.0],
                [2.0, 0.0, 1.0, 4.0],
                [5.0, 1.0, 0.0, 3.0],
                [9.0, 4.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        entry_local = np.array([[0, 2], [3, 1]], dtype=np.int64)
        entry_dist = np.array([[0.5, 4.0], [1.0, 3.0]], dtype=np.float32)

        extended = compute_oos_geodesic_block(
            geodesic_train,
            entry_local,
            entry_dist,
        )

        expected_first = np.minimum(
            0.5 + geodesic_train[0],
            4.0 + geodesic_train[2],
        )
        expected_second = np.minimum(
            1.0 + geodesic_train[3],
            3.0 + geodesic_train[1],
        )
        np.testing.assert_allclose(extended[0], expected_first)
        np.testing.assert_allclose(extended[1], expected_second)

    def test_sorted_topk_returns_indices_and_distances_in_ascending_order(self):
        distances = np.array(
            [[5.0, 1.0, 4.0, 2.0], [0.2, 0.5, 0.1, 0.3]],
            dtype=np.float32,
        )

        indices, values = select_sorted_topk(distances, k=2)

        np.testing.assert_array_equal(indices, np.array([[1, 3], [2, 0]]))
        np.testing.assert_allclose(values, np.array([[1.0, 2.0], [0.1, 0.2]]))

    def test_train_sigma_uses_graph_kth_neighbor(self):
        geodesic_train = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 4.0],
                [2.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )
        neighbors = np.array(
            [
                [1, 2],
                [0, 2],
                [0, 1],
            ],
            dtype=np.int64,
        )

        sigma = compute_train_sigma(geodesic_train, neighbors, graph_k=2)

        np.testing.assert_allclose(sigma, np.array([2.0, 4.0, 4.0], dtype=np.float32))

    def test_local_tikhonov_oos_matches_closed_form_solution(self):
        y_test = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        z_train = np.array(
            [[2.0, 0.0], [4.0, 0.0], [100.0, 0.0]],
            dtype=np.float32,
        )
        neighbor_local = np.array([[0, 1], [1, 2]], dtype=np.int64)
        weights = np.array([[1.0, 3.0], [0.5, 0.5]], dtype=np.float32)

        z_oos = local_tikhonov_oos(
            y_test,
            z_train,
            neighbor_local,
            weights,
            lambda_oos=0.5,
        )

        expected_first = (y_test[0] + 0.5 * (1.0 * z_train[0] + 3.0 * z_train[1])) / (
            1.0 + 0.5 * 4.0
        )
        expected_second = (
            y_test[1] + 0.5 * (0.5 * z_train[1] + 0.5 * z_train[2])
        ) / (1.0 + 0.5 * 1.0)
        np.testing.assert_allclose(z_oos[0], expected_first)
        np.testing.assert_allclose(z_oos[1], expected_second)


if __name__ == "__main__":
    unittest.main()
