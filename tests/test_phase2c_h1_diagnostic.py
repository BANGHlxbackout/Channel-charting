import unittest

import numpy as np
import scipy.sparse as sp

from diagnostics.phase2c_h1_h2.run_h1_diagnostic import (
    edge_set_path_hits,
    extract_undirected_edges,
    incident_median_scale,
    physical_local_scale,
    robust_mismatch_scores,
    sequence_concentration,
    stratified_source_split,
)


class Phase2CH1DiagnosticTests(unittest.TestCase):
    def test_directed_graph_is_reduced_to_weight_preserving_union(self):
        graph = sp.csr_matrix(
            (
                np.array([1.0, 1.0, 2.0], dtype=np.float32),
                (
                    np.array([0, 1, 2]),
                    np.array([1, 0, 1]),
                ),
            ),
            shape=(3, 3),
        )

        u, v, weight, support_uv, support_vu, weight_uv, weight_vu = extract_undirected_edges(graph)

        np.testing.assert_array_equal(u, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(v, np.array([1, 2], dtype=np.int32))
        np.testing.assert_array_equal(weight, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(support_uv, np.array([True, False]))
        np.testing.assert_array_equal(support_vu, np.array([True, True]))
        np.testing.assert_array_equal(weight_uv, np.array([1.0, np.inf], dtype=np.float32))
        np.testing.assert_array_equal(weight_vu, np.array([1.0, 2.0], dtype=np.float32))

    def test_undirected_effective_weight_uses_smaller_reciprocal_entry(self):
        graph = sp.csr_matrix(
            (
                np.array([2.0, 1.5], dtype=np.float32),
                (
                    np.array([0, 1]),
                    np.array([1, 0]),
                ),
            ),
            shape=(2, 2),
        )

        _, _, weight, _, _, weight_uv, weight_vu = extract_undirected_edges(graph)

        np.testing.assert_array_equal(weight, np.array([1.5], dtype=np.float32))
        np.testing.assert_array_equal(weight_uv, np.array([2.0], dtype=np.float32))
        np.testing.assert_array_equal(weight_vu, np.array([1.5], dtype=np.float32))

    def test_incident_median_scale_uses_only_stage1_union_edges(self):
        u = np.array([0, 0, 1], dtype=np.int32)
        v = np.array([1, 2, 2], dtype=np.int32)
        weight = np.array([1.0, 3.0, 5.0], dtype=np.float32)

        scale, degree = incident_median_scale(3, u, v, weight)

        np.testing.assert_allclose(scale, np.array([2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(degree, np.array([2, 2, 2], dtype=np.int32))

    def test_robust_mismatch_score_increases_with_physical_ratio(self):
        wireless = np.ones(5, dtype=np.float32)
        physical = np.array([1.0, 2.0, 3.0, 5.0, 20.0], dtype=np.float32)

        _, score, _, robust_scale = robust_mismatch_scores(wireless, physical)

        self.assertGreater(robust_scale, 0)
        self.assertTrue(np.all(np.diff(score) > 0))

    def test_physical_scale_fallback_uses_positive_distances_in_same_q_set(self):
        positions = np.array(
            [[0.0, 0.0]] * 4 + [[1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )

        scale, fallback_count = physical_local_scale(positions, q=4)

        self.assertGreater(fallback_count, 0)
        self.assertTrue(np.all(scale > 0))

    def test_source_split_is_disjoint_and_balanced_by_quartile(self):
        physical_scale = np.arange(400, dtype=np.float32)
        rng = np.random.default_rng(10)

        discovery, validation, quartiles, _ = stratified_source_split(
            physical_scale,
            rng,
            per_quartile_per_split=8,
        )

        self.assertEqual(len(np.intersect1d(discovery, validation)), 0)
        np.testing.assert_array_equal(
            np.bincount(quartiles[discovery], minlength=4),
            np.full(4, 8),
        )
        np.testing.assert_array_equal(
            np.bincount(quartiles[validation], minlength=4),
            np.full(4, 8),
        )

    def test_concentration_counts_path_if_any_selected_edge_is_present(self):
        flat_edges = np.array([0, 1, 2, 3, 1], dtype=np.int32)
        offsets = np.array([0, 2, 4, 5], dtype=np.int64)
        severe = np.array([True, True, False])
        ranked = np.array([1, 3, 0, 2], dtype=np.int32)

        curve = sequence_concentration(
            flat_edges,
            offsets,
            severe,
            ranked,
            np.array([1, 2], dtype=np.int32),
            edge_count=4,
        )
        selected = np.array([False, True, False, False])
        hits = edge_set_path_hits(flat_edges, offsets, selected)

        np.testing.assert_allclose(curve, np.array([0.5, 1.0]))
        np.testing.assert_array_equal(hits, np.array([True, False, True]))


if __name__ == "__main__":
    unittest.main()
