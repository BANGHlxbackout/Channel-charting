import unittest

import numpy as np
import scipy.sparse as sp
import torch

from methods.sc2_stage2_adaptive_fidelity.run_pilot import (
    build_symmetric_union,
    compute_second_order_scores,
    percentile_fidelity,
    solve_adaptive_fidelity,
)
from run_diffusion_refined_charting import solve_graph_harmonic_refinement


class Sc2Stage2AdaptiveFidelityTests(unittest.TestCase):
    def test_symmetric_union_keeps_one_value_per_reciprocal_pair(self):
        rows = np.array([0, 0, 1, 2], dtype=np.int64)
        cols = np.array([1, 2, 0, 0], dtype=np.int64)
        values = np.array([0.2, 0.4, 0.2, 0.4], dtype=np.float64)
        union, checks = build_symmetric_union(rows, cols, values, 3)

        np.testing.assert_allclose(union.toarray(), union.toarray().T)
        np.testing.assert_allclose(union.toarray(), np.array([[0, 0.2, 0.4], [0.2, 0, 0], [0.4, 0, 0]]))
        self.assertEqual(checks["undirected_union_edge_count"], 2)
        self.assertEqual(checks["diagonal_nnz"], 0)
        self.assertEqual(checks["max_directional_disagreement"], 0.0)

    def test_h_and_s_use_sparse_union_witnesses(self):
        # Symmetric weighted triangle.  H_01 is the cosine of rows 0 and 1;
        # S_01 multiplies that cosine by the directed C_01 value.
        union = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.5],
                    [1.0, 0.0, 0.25],
                    [0.5, 0.25, 0.0],
                ],
                dtype=np.float64,
            )
        )
        rows = np.array([0, 0, 1, 1, 2, 2])
        cols = np.array([1, 2, 0, 2, 0, 1])
        c_directed = np.array([1.0, 0.5, 1.0, 0.25, 0.5, 0.25])
        q_fo, h, q_sc2 = compute_second_order_scores(c_directed, union, rows, cols, 3)
        expected_h01 = (0.5 * 0.25) / (
            np.sqrt(1.0**2 + 0.5**2) * np.sqrt(1.0**2 + 0.25**2) + 1e-8
        )
        self.assertAlmostEqual(h[0], expected_h01, places=7)
        self.assertAlmostEqual(q_fo[0], 0.75, places=7)
        self.assertAlmostEqual(q_sc2[0], (1.0 * h[0] + 0.5 * h[1]) / 2.0, places=7)

    def test_rankdata_average_ties_and_constant_scores(self):
        percentile, fidelity = percentile_fidelity(np.array([0.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(percentile, [1 / 6, 1 / 6, 2 / 3, 1.0])
        np.testing.assert_allclose(fidelity, [2 / 3, 2 / 3, 7 / 6, 1.5])
        _, constant_fidelity = percentile_fidelity(np.zeros(5))
        np.testing.assert_allclose(constant_fidelity, np.ones(5))

    def test_uniform_fidelity_matches_original_solver_update(self):
        graph = sp.coo_matrix(
            (
                np.array([0.3, 0.4, 0.3, 0.4], dtype=np.float32),
                (np.array([0, 1, 1, 2]), np.array([1, 0, 2, 1])),
            ),
            shape=(3, 3),
        )
        y0 = np.array([[0.2, -0.1], [0.4, 0.8], [-0.3, 0.5]], dtype=np.float32)
        uniform = solve_adaptive_fidelity(
            y0, graph, np.ones(3, dtype=np.float64), 0.15, 7, torch.device("cpu"), verbose=False
        )
        original = solve_graph_harmonic_refinement(
            y0, graph, lambda_smooth=0.15, iterations=7, device=torch.device("cpu")
        )
        np.testing.assert_array_equal(uniform, original)


if __name__ == "__main__":
    unittest.main()
