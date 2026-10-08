import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from diagnostics.phase2c_h1_h2.run_h1_counterfactual import (
    constrained_optimal_sequences,
    counterfactual_decision,
    frozen_isotonic_predict,
    remove_logical_edges,
)


class Phase2CH1CounterfactualTests(unittest.TestCase):
    def test_frozen_isotonic_prediction_is_clipped_and_monotone(self):
        values = frozen_isotonic_predict(np.asarray([-100.0, 0.0, 100.0]))
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_remove_logical_edge_deletes_both_directions_only(self):
        graph = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                (
                    np.asarray([0, 1, 1, 2]),
                    np.asarray([1, 0, 2, 1]),
                ),
            ),
            shape=(3, 3),
        )
        rows = np.repeat(np.arange(3, dtype=np.int32), np.diff(graph.indptr))
        keys = np.minimum(rows, graph.indices).astype(np.int64) * 3 + np.maximum(
            rows, graph.indices
        )
        edge_frame = pd.DataFrame(
            {
                "u": [0, 1],
                "v": [1, 2],
                "directed_support_u_to_v": [True, True],
                "directed_support_v_to_u": [True, True],
            }
        )
        modified = remove_logical_edges(
            graph,
            rows,
            keys,
            np.asarray([0], dtype=np.int32),
            edge_frame,
        )
        self.assertEqual(modified.nnz, 2)
        self.assertEqual(float(modified[0, 1]), 0.0)
        self.assertEqual(float(modified[1, 0]), 0.0)
        self.assertEqual(float(modified[1, 2]), 3.0)
        self.assertEqual(float(modified[2, 1]), 4.0)

    def test_constrained_sequences_are_unique_and_from_pool(self):
        covariates = pd.DataFrame(
            {
                "a": np.tile(np.arange(10, dtype=float), 3),
                "b": np.tile(np.arange(10, dtype=float) / 2, 3),
            }
        )
        targets = np.asarray([0, 1, 2], dtype=np.int32)
        pool = np.arange(10, 30, dtype=np.int32)
        sequences, _, _ = constrained_optimal_sequences(
            targets,
            pool,
            covariates,
            ("a",),
            ("b",),
            np.random.default_rng(11),
        )
        self.assertEqual(sequences.shape[1], 3)
        for sequence in sequences:
            self.assertEqual(len(np.unique(sequence)), 3)
            self.assertTrue(np.all(np.isin(sequence, pool)))

    def test_decision_requires_two_supported_batch_sizes(self):
        rows = []
        for count in (25, 50, 100):
            rows.append(
                {
                    "family": "targeted",
                    "replicate": -1,
                    "edge_count": count,
                    "repair_rate": 0.8 if count != 100 else 0.1,
                    "median_delta_r_baseline_severe": 0.4 if count != 100 else 0.05,
                    "new_overestimation_fraction_increase": 0.01,
                    "median_log_inflation_well_estimated": 0.01,
                    "connected_components": 1,
                    "giant_component_ratio": 1.0,
                    "unreachable_test_pairs": 0,
                }
            )
            for family in ("matched_random", "suspicious_low_impact"):
                for replicate in range(50):
                    rows.append(
                        {
                            "family": family,
                            "replicate": replicate,
                            "edge_count": count,
                            "repair_rate": 0.2,
                            "median_delta_r_baseline_severe": 0.1,
                            "new_overestimation_fraction_increase": 0.02,
                            "median_log_inflation_well_estimated": 0.02,
                            "connected_components": 1,
                            "giant_component_ratio": 1.0,
                            "unreachable_test_pairs": 0,
                        }
                    )
        decision, details = counterfactual_decision(pd.DataFrame(rows))
        self.assertEqual(decision, "true")
        self.assertEqual(details["counts"]["fully_supported_X_count"], 2)


if __name__ == "__main__":
    unittest.main()
