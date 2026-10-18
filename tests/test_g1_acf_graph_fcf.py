from __future__ import annotations

import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra

from methods.g1_acf_graph_fcf.graph_utils import (audit_shortest_paths, identity_adjacency,
                                propagation_adjacency, rebuild_directed_graph,
                                train_pairs, validation_pairs)
from run_adp_siamese import AdpFeatureLayer, siamese_loss
from methods.g1_acf_graph_fcf.run import GraphFCFWithExistingACF
from methods.g1_acf_graph_fcf.compare import compare


class G1Tests(unittest.TestCase):
    def test_feature_shape_and_common_phase(self):
        rng = np.random.default_rng(7)
        h = (rng.normal(size=(3, 4, 8, 13)) + 1j * rng.normal(size=(3, 4, 8, 13))).astype(np.complex64)
        layer = AdpFeatureLayer()
        original = layer(torch.from_numpy(h))
        rotated = layer(torch.from_numpy(h * np.exp(1j * 0.71).astype(np.complex64)))
        self.assertEqual(tuple(original.shape), (3, 26624))
        torch.testing.assert_close(original, rotated, atol=2e-5, rtol=2e-5)
        h[:, 1] *= np.complex64(np.exp(1j * 0.5))
        self.assertFalse(torch.allclose(original, layer(torch.from_numpy(h))))

    def test_graph_provenance_and_symmetry(self):
        x = np.arange(24, dtype=np.float32)
        fused = np.abs(x[:, None] - x[None, :]).astype(np.float32)
        fused[3, 4] += 1e-6
        graph = rebuild_directed_graph(fused, 4)
        geo = dijkstra(graph, directed=False).astype(np.float32)
        self.assertEqual(audit_shortest_paths(graph, geo, n_sources=8)["mismatch_count"], 0)
        adj, info = propagation_adjacency(graph, fused)
        dense = adj.to_dense()
        torch.testing.assert_close(dense, dense.T, atol=1e-7, rtol=1e-7)
        self.assertTrue(torch.all(dense.diagonal() > 0))
        self.assertGreater(info["undirected_edges"], 0)
        ident = identity_adjacency(len(x))
        h = torch.randn(len(x), 4)
        torch.testing.assert_close(torch.sparse.mm(ident, h), h)

    def test_pairs_exclude_validation(self):
        n = 80
        vl, vr, keys = validation_pairs(n, 400, 123)
        left, right = train_pairs(n, 2000, np.random.default_rng(456), keys)
        self.assertTrue(np.all(vl < vr))
        self.assertTrue(np.all(left != right))
        generated = np.minimum(left, right) * n + np.maximum(left, right)
        self.assertFalse(np.isin(generated, keys).any())

    def test_direct_and_stream_projection_gradients(self):
        torch.manual_seed(9)
        features = np.random.default_rng(9).normal(size=(32, 26624)).astype(np.float32)
        model = GraphFCFWithExistingACF(hidden=16, layers=2, dropout=0.0)
        other = copy.deepcopy(model)
        adj = identity_adjacency(32)
        direct = model(torch.from_numpy(features), adj)
        streamed = other(features, adj, chunk_size=8)
        torch.testing.assert_close(direct, streamed, atol=1e-5, rtol=1e-5)
        left = torch.tensor([0, 2, 4, 6])
        right = torch.tensor([1, 3, 5, 7])
        target = torch.tensor([1.0, 1.2, 0.9, 1.1])
        siamese_loss(direct[left], direct[right], target, 0.1).backward()
        siamese_loss(streamed[left], streamed[right], target, 0.1).backward()
        for (name, p), (other_name, q) in zip(model.named_parameters(), other.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(p.grad)
            self.assertIsNotNone(q.grad)
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-5)

    def test_common_protocol_four_panel_output(self):
        rng = np.random.default_rng(21)
        pos = rng.normal(size=(100, 2)).astype(np.float32)
        idx = np.arange(100)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            charts = []
            for name, scale in (("b0", 0.1), ("identity", 0.2), ("graph", 0.05)):
                folder = root / name
                folder.mkdir()
                path = folder / "embedding.npz"
                np.savez(path, indices=idx, positions=pos,
                         embedding=pos + scale * rng.normal(size=pos.shape))
                charts.append(path)
            output = root / "comparison"
            result = compare(Namespace(baseline_embedding=charts[0],
                                       identity_embedding=charts[1],
                                       graph_embedding=charts[2], seed=10,
                                       out_dir=output))
            self.assertEqual(set(result), {"B0", "G1-Identity", "G1-Graph"})
            self.assertTrue((output / "four_column_raw_channel_chart.png").exists())
            self.assertTrue((output / "four_column_affine_channel_chart.png").exists())
            self.assertTrue((output / "aligned_charts.npz").exists())
            self.assertTrue((output / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
