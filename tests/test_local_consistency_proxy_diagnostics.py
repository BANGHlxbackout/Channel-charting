import unittest

import numpy as np

from diagnostics.diagnose_local_consistency_proxies import (
    anti_repair_gate,
    binary_auc,
    multi_k_stability,
    mutual_support,
    rank_weighted_support,
    strict_gate_report,
)


class LocalConsistencyProxyDiagnosticsTests(unittest.TestCase):
    def test_binary_auc_returns_one_for_perfect_ranking(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        labels = np.array([False, False, True, True])

        self.assertAlmostEqual(binary_auc(scores, labels), 1.0)

    def test_mutual_support_requires_bidirectional_y0_neighbors(self):
        physical = np.array(
            [
                [1, 2],
                [0, 2],
                [0, 1],
            ],
            dtype=np.int32,
        )
        y0_neighbors = np.array(
            [
                [1, 2],
                [0, 2],
                [1, 0],
            ],
            dtype=np.int32,
        )

        support = mutual_support(physical, y0_neighbors, k=2)

        np.testing.assert_allclose(support, np.array([1.0, 1.0, 1.0], dtype=np.float32))

    def test_mutual_support_rejects_one_way_neighbor(self):
        physical = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        y0_neighbors = np.array([[1, 2], [2, 0], [1, 0]], dtype=np.int32)

        support = mutual_support(physical, y0_neighbors, k=1)

        np.testing.assert_allclose(support, np.array([0.0, 0.0, 0.0], dtype=np.float32))

    def test_rank_weighted_support_uses_lower_rank_as_stronger_support(self):
        physical = np.array([[1, 3]], dtype=np.int32)
        y0_rank_neighbors = np.array([[1, 2, 3, 4]], dtype=np.int32)

        support = rank_weighted_support(physical, y0_rank_neighbors, k=2, tau=2.0)

        expected = np.mean([np.exp(-1 / 2.0), np.exp(-3 / 2.0)])
        self.assertAlmostEqual(float(support[0]), float(expected), places=6)

    def test_multi_k_stability_uses_worst_overlap_across_scales(self):
        physical = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.int32)
        y0 = np.array([[1, 2, 9, 8], [1, 9, 8, 7]], dtype=np.int32)

        stability = multi_k_stability(physical, y0, ks=(1, 2, 4))

        np.testing.assert_allclose(stability, np.array([0.5, 0.25], dtype=np.float32))

    def test_anti_repair_gate_downweights_high_repair_proxy(self):
        scale_stress = np.array([0.0, 0.0, 3.0, 4.0], dtype=np.float32)
        residual = np.array([1.0, 1.0, 10.0, 12.0], dtype=np.float32)

        gate = anti_repair_gate(scale_stress, residual, rho=0.25)

        self.assertLess(float(gate[-1]), float(gate[0]))
        self.assertTrue(np.all(gate <= 1.0))
        self.assertTrue(np.all(gate > 0.0))

    def test_strict_gate_requires_relative_gain_and_outlier_control(self):
        raw_report = {
            "auc_good_initial_high_score_positive": 0.52,
            "auc_strong_worsened_high_score_positive": 0.51,
        }
        candidate_report = {
            "auc_good_initial_high_score_positive": 0.66,
            "auc_strong_worsened_high_score_positive": 0.63,
            "spearman_error_delta": 0.18,
        }
        top_report = {
            "good_initial_enrichment": 1.30,
            "strong_worsened_enrichment": 1.00,
            "outlier_initial_enrichment": 1.05,
        }

        gate = strict_gate_report(raw_report, candidate_report, top_report)

        self.assertTrue(gate["passes"])


if __name__ == "__main__":
    unittest.main()
