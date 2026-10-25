from __future__ import annotations

import unittest

import numpy as np
import torch

from methods.g1_acf_graph_fcf.diagnose_graph_propagation import probe_forward, state_digest
from methods.g1_acf_graph_fcf.graph_utils import identity_adjacency
from methods.g1_acf_graph_fcf.run import GraphFCFWithExistingACF


class PropagationDiagnosticTests(unittest.TestCase):
    def test_hooks_preserve_prediction_and_frozen_buffers(self):
        torch.manual_seed(19)
        model = GraphFCFWithExistingACF(input_dim=16, hidden=12, layers=2, dropout=0.1).eval()
        x = torch.randn(20, 16)
        adj = identity_adjacency(len(x))
        before = state_digest(model)
        with torch.no_grad():
            direct = model(x, adj).numpy()
        observed, arrays = probe_forward(model, x, adj, None)
        np.testing.assert_array_equal(direct, observed)
        self.assertEqual(before, state_digest(model))
        for layer in range(2):
            np.testing.assert_array_equal(arrays[f'layer{layer}_message_minus_self_over_h'], np.zeros(20))
            # A residual update can be large even with no neighbor aggregation.
            self.assertGreater(np.median(arrays[f'layer{layer}_delta_over_h']), 0)

    def test_streamed_eval_and_direct_eval_agree(self):
        torch.manual_seed(20)
        model = GraphFCFWithExistingACF(input_dim=16, hidden=12).eval()
        x = np.random.default_rng(20).normal(size=(20, 16)).astype(np.float32)
        adj = identity_adjacency(len(x))
        direct, _ = probe_forward(model, torch.from_numpy(x), adj, None)
        streamed, _ = probe_forward(model, x, adj, 5)
        np.testing.assert_allclose(direct, streamed, atol=1e-5, rtol=1e-5)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))

    def test_training_mode_is_rejected(self):
        model = GraphFCFWithExistingACF(input_dim=16, hidden=12)
        with self.assertRaises(ValueError):
            probe_forward(model, torch.randn(10, 16), identity_adjacency(10), None)


if __name__ == '__main__':
    unittest.main()
