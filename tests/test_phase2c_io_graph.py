import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from diagnostics.phase2c_h1_h2.phase2c_io_graph import (
    boundary_tie_statistics,
    graph_sanity_statistics,
    reconstruct_stage1_graph,
)


def pairwise_line_distances(n: int) -> np.ndarray:
    points = np.arange(n, dtype=np.float32)
    return np.abs(points[:, None] - points[None, :]).astype(np.float32)


class Phase2CStage1GraphTests(unittest.TestCase):
    def test_k20_graph_has_twenty_nonself_outgoing_neighbors(self):
        fused = pairwise_line_distances(24)

        graph = reconstruct_stage1_graph(fused, k=20)
        stats = graph_sanity_statistics(graph, k=20)

        self.assertTrue(sp.isspmatrix_csr(graph))
        self.assertEqual(graph.nnz, 24 * 20)
        self.assertEqual(stats["self_edge_count"], 0)
        self.assertTrue(stats["all_rows_have_exactly_k_nonself_neighbors"])
        self.assertEqual(stats["nonself_outgoing_degree"], {"min": 20, "median": 20.0, "max": 20})

    def test_directed_csr_and_fused_edge_weights_are_preserved(self):
        fused = pairwise_line_distances(4)

        graph = reconstruct_stage1_graph(fused, k=1)

        self.assertTrue(sp.isspmatrix_csr(graph))
        self.assertGreater((graph != graph.T).nnz, 0)
        for row in range(graph.shape[0]):
            start, stop = graph.indptr[row : row + 2]
            for offset in range(start, stop):
                col = graph.indices[offset]
                self.assertEqual(graph.data[offset], fused[row, col])

    def test_directed_false_shortest_path_traverses_reverse_edges(self):
        graph = sp.csr_matrix(
            (
                np.array([1.0, 2.0], dtype=np.float32),
                (
                    np.array([0, 2], dtype=np.int32),
                    np.array([1, 1], dtype=np.int32),
                ),
            ),
            shape=(3, 3),
        )

        distances = shortest_path(
            graph,
            method="auto",
            directed=False,
            indices=np.array([0]),
        )

        self.assertEqual(distances[0, 2], 3.0)

    def test_boundary_tie_detection_excludes_self(self):
        fused = np.array(
            [
                [0.0, 1.0, 2.0, 2.0],
                [1.0, 0.0, 3.0, 4.0],
                [2.0, 3.0, 0.0, 5.0],
                [2.0, 4.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        )

        stats = boundary_tie_statistics(fused, k=2, block_rows=2)

        self.assertEqual(stats["boundary_tie_row_count"], 1)
        self.assertEqual(stats["boundary_tie_row_ids"], [0])
        self.assertEqual(stats["max_boundary_tie_multiplicity"], 2)
        self.assertEqual(stats["kth_distance"]["min"], 2.0)


if __name__ == "__main__":
    unittest.main()
