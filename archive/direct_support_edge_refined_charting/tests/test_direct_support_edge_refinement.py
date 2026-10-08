import unittest

import numpy as np
import scipy.sparse as sp

from archive.direct_support_edge_refined_charting.run_direct_support_edge_refined_charting import build_direct_support_graph


class DirectSupportEdgeRefinementTests(unittest.TestCase):
    def test_direct_support_reliability_downweights_path_mediated_edges(self):
        graph = sp.coo_matrix(
            (
                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                (
                    np.array([0, 1, 1, 2]),
                    np.array([1, 0, 2, 1]),
                ),
            ),
            shape=(3, 3),
        )
        geodesic = np.array(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 1.0],
                [3.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        fused = np.array(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 4.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )

        refined_graph, stats = build_direct_support_graph(
            graph,
            geodesic,
            fused,
            rho=0.6,
            tau_min=1e-6,
        )

        weights = {(i, j): w for i, j, w in zip(refined_graph.row, refined_graph.col, refined_graph.data)}
        self.assertAlmostEqual(weights[(0, 1)], 1.0, places=6)
        self.assertLess(weights[(1, 2)], 1.0)
        self.assertGreaterEqual(weights[(1, 2)], 0.6)
        self.assertEqual(stats["num_edges"], 4)
        self.assertLess(stats["raw_weight_ratio"], 1.0)

    def test_energy_preserving_rescales_total_weight(self):
        graph = sp.coo_matrix(
            (
                np.array([2.0, 2.0], dtype=np.float32),
                (np.array([0, 1]), np.array([1, 0])),
            ),
            shape=(2, 2),
        )
        geodesic = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        fused = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float32)

        refined_graph, stats = build_direct_support_graph(
            graph,
            geodesic,
            fused,
            rho=0.6,
            preserve_energy=True,
        )

        self.assertAlmostEqual(float(refined_graph.data.sum()), 4.0, places=5)
        self.assertGreater(stats["energy_scale"], 1.0)
        self.assertTrue(stats["energy_preserving"])


if __name__ == "__main__":
    unittest.main()
