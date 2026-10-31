import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy import sparse

from methods.multiview_edge_uncertainty.gate0 import (
    apply_risk_cost,
    kth_nonself,
    load_scoring_inputs,
    score_edges,
    view_adp_block,
    view_mapping,
)
from run_adp_fused import compute_adp_matrix, normalize_csi_vectors


ROOT = Path(__file__).resolve().parents[1]


class Gate0Tests(unittest.TestCase):
    def test_stored_views_match_official_array_sets(self):
        self.assertEqual(
            view_mapping(ROOT / "preprocess_adp_old_shard.py", ROOT / "datasets/specv2.0.json"),
            ["C", "B", "A", "D"],
        )

    def test_four_view_adp_reconstructs_aggregate_on_small_csi(self):
        rng = np.random.default_rng(7)
        h = (rng.normal(size=(6, 4, 8, 13))
             + 1j * rng.normal(size=(6, 4, 8, 13))).astype(np.complex64)
        z = normalize_csi_vectors(h).reshape(6, 4, 13, 8)
        reconstructed = np.zeros((6, 6), dtype=np.float32)
        for view in range(4):
            reconstructed += view_adp_block(torch.from_numpy(z[:, view].copy()), 0, 6)
        reference = compute_adp_matrix(h, block_size=6, device="cpu")
        np.testing.assert_allclose(reconstructed, reference, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(np.diag(reconstructed), np.zeros(6, dtype=np.float32))

    def test_mad_invariants_and_source_flag(self):
        d = np.array([[1, 1, 1, 1], [1, 1, 1, 10], [1, 1, 10, 10]], dtype=np.float32)
        scales = np.ones((4, 4), dtype=np.float32)
        left = np.array([0, 0, 1])
        right = np.array([1, 2, 3])
        adp = np.zeros((4, 4), dtype=np.float32)
        adp[left, right] = [2, 2, 20]
        timestamps = np.array([0, 1, 2, 3], dtype=np.float32)
        q, s, u, source = score_edges(d, scales, left, right, adp, timestamps, gamma=5)
        self.assertEqual(s[0], 0)
        self.assertEqual(u[0], 0)
        self.assertEqual(s[1], 0)
        self.assertGreater(s[2], 0)
        self.assertGreater(float(np.ptp(q[1])), 0)
        self.assertEqual(int(np.count_nonzero(q[0] <= 1)), 4)
        self.assertEqual(int(np.count_nonzero(q[1] <= 1)), 3)
        self.assertTrue(np.array_equal(source, [True, True, False]))
        _, s_perm, u_perm, _ = score_edges(d[:, ::-1], scales[:, ::-1], left, right,
                                          adp, timestamps, gamma=5)
        np.testing.assert_allclose(s_perm, s)
        np.testing.assert_allclose(u_perm, u)
        self.assertTrue(np.all(np.isfinite(q)))

    def test_kth_scale_excludes_self(self):
        distances = np.array([[0, 4, 1, 3], [4, 0, 2, 1]], dtype=np.float32)
        np.testing.assert_array_equal(kth_nonself(distances, start=0, k=2), [3, 2])

    def test_scoring_input_loader_does_not_need_positions(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "H.npy", np.zeros((2, 32, 13), dtype=np.complex64))
            np.save(path / "timestamps.npy", np.array([1, 2], dtype=np.float32))
            h, timestamps = load_scoring_inputs(path)
            self.assertEqual(h.shape, (2, 32, 13))
            np.testing.assert_array_equal(timestamps, [1, 2])
            del h, timestamps

    def test_cost_preserves_directed_entries_and_reciprocal_multiplier(self):
        graph = sparse.csr_matrix(
            (np.array([2, 2.00001, 3], dtype=np.float32),
             (np.array([0, 1, 1]), np.array([1, 0, 2]))), shape=(3, 3)
        )
        pair_index = np.array([0, 0, 1])
        u = np.array([0.5, 0.8])
        source = np.array([True, False])
        baseline = apply_risk_cost(graph, pair_index, u, source, 0)
        changed = apply_risk_cost(graph, pair_index, u, source, 0.5)
        np.testing.assert_array_equal(baseline.data, graph.data)
        np.testing.assert_array_equal(changed.indptr, graph.indptr)
        np.testing.assert_array_equal(changed.indices, graph.indices)
        np.testing.assert_allclose(changed.data, [2.5, 2.00001 * 1.25, 3.0])
        self.assertGreater(changed.data[1], changed.data[0])


if __name__ == "__main__":
    unittest.main()
