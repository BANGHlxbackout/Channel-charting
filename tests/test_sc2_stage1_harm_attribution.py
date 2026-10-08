import unittest

import numpy as np
import scipy.sparse as sp

from diagnostics.sc2_stage1_harm_attribution.run_pilot import summarize, variant_graph


class HarmAttributionTests(unittest.TestCase):
    def test_variant_graph_changes_only_selected_edges_and_preserves_support(self):
        graph = sp.csr_matrix(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]], dtype=np.float32))
        original = graph.data.copy()
        multiplier = np.array([1.5, 1.25, 1.5, 1.25], dtype=np.float32)
        mask = np.array([True, False, True, False])
        variant = variant_graph(graph, original, multiplier, mask)
        np.testing.assert_array_equal(variant.indices, graph.indices)
        np.testing.assert_array_equal(variant.indptr, graph.indptr)
        np.testing.assert_allclose(variant.data, original * np.where(mask, multiplier, 1.0))

    def test_complementary_groups_reconstruct_full_penalty(self):
        graph = sp.csr_matrix(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]], dtype=np.float32))
        original = graph.data.copy()
        multiplier = np.array([1.5, 1.25, 1.5, 1.25], dtype=np.float32)
        t = np.array([True, False, True, False])
        t_graph = variant_graph(graph, original, multiplier, t)
        r_graph = variant_graph(graph, original, multiplier, ~t)
        full_graph = variant_graph(graph, original, multiplier, np.ones(4, dtype=bool))
        np.testing.assert_allclose(full_graph.data - original, (t_graph.data - original) + (r_graph.data - original))

    def test_summary_has_fixed_quantiles(self):
        result = summarize(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 3.0)


if __name__ == "__main__":
    unittest.main()