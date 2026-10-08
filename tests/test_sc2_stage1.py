import unittest

import numpy as np
import scipy.sparse as sp

from methods.sc2_stage1.run_pilot import (
    build_variant_graph,
    common_neighbor_support,
    shuffled_multipliers,
)


class Sc2Stage1SemanticsTests(unittest.TestCase):
    def test_common_support_uses_undirected_union_and_frozen_normalization(self):
        # Directed edges: 0->1, 1->2, 2->0, plus 2->3.  The undirected
        # triangle gives one common neighbor for each triangle edge.
        graph = sp.csr_matrix(
            (
                np.ones(4, dtype=np.float32),
                ([0, 1, 2, 2], [1, 2, 0, 3]),
            ),
            shape=(4, 4),
        )
        counts, scores, degree, _ = common_neighbor_support(graph)
        np.testing.assert_array_equal(counts, np.array([1, 1, 1, 0], dtype=np.int32))
        # deg=[2,2,3,1], so the first three denominators are 1,1,1 and the
        # 2->3 denominator is max(1,min(2,0))=1, with zero common support.
        np.testing.assert_allclose(scores, np.array([1, 1, 1, 0], dtype=np.float32))
        np.testing.assert_array_equal(degree, np.array([2, 2, 3, 1], dtype=np.int32))

    def test_variant_keeps_directed_csr_support_and_only_changes_weights(self):
        graph = sp.csr_matrix(
            (np.array([1, 2, 3], dtype=np.float32), ([0, 1, 2], [1, 2, 0])),
            shape=(3, 3),
        )
        variant = build_variant_graph(graph, np.array([2, 4, 6], dtype=np.float32))
        np.testing.assert_array_equal(variant.indices, graph.indices)
        np.testing.assert_array_equal(variant.indptr, graph.indptr)
        np.testing.assert_array_equal(variant.data, np.array([2, 4, 6], dtype=np.float32))
        self.assertEqual(variant.diagonal().sum(), 0.0)

    def test_shuffled_assignment_preserves_multipliers_and_is_seeded(self):
        rows = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        cols = np.array([1, 2, 0, 3, 0, 3, 1, 2])
        weights = np.arange(1, 9, dtype=np.float32)
        multipliers = np.arange(1, 9, dtype=np.float64) / 10 + 1
        degree = np.array([4, 4, 4, 4], dtype=np.int32)
        shuffled_a, info_a = shuffled_multipliers(weights, multipliers, rows, cols, degree, seed=10)
        shuffled_b, info_b = shuffled_multipliers(weights, multipliers, rows, cols, degree, seed=10)
        np.testing.assert_array_equal(shuffled_a, shuffled_b)
        np.testing.assert_array_equal(np.sort(shuffled_a), np.sort(multipliers))
        self.assertEqual(info_a["seed"], 10)
        self.assertEqual(info_a["group_sizes"], info_b["group_sizes"])


if __name__ == "__main__":
    unittest.main()
