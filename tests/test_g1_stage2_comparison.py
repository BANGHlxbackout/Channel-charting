from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy import sparse

from methods.g1_acf_graph_fcf.compare import compare_paths
from methods.g1_acf_graph_fcf.run_stage2_comparison import S2, refine_charts
from run_diffusion_refined_charting import solve_graph_harmonic_refinement


class Stage2ComparisonTests(unittest.TestCase):
    def test_shared_graph_and_original_solver_equivalence(self):
        n = 12
        graph = sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n)).tocoo()
        y = np.random.default_rng(10).normal(size=(n, 2)).astype(np.float32)
        inputs = {"b0": y, "identity": y + 1, "graph": y * 2}
        before = {key: value.copy() for key, value in inputs.items()}
        with patch('methods.g1_acf_graph_fcf.run_stage2_comparison.solve_graph_harmonic_refinement',
                   wraps=solve_graph_harmonic_refinement) as solver:
            outputs = refine_charts(inputs, graph, torch.device('cpu'))
            self.assertEqual(solver.call_count, 3)
            for call in solver.call_args_list:
                self.assertIs(call.args[1], graph)
                self.assertEqual(call.args[2:4], (0.15, 200))
        expected = solve_graph_harmonic_refinement(y, graph, 0.15, 200, torch.device('cpu'))
        np.testing.assert_array_equal(outputs['b0'], expected)
        repeat = refine_charts({'b0': y}, graph, torch.device('cpu'))['b0']
        np.testing.assert_array_equal(outputs['b0'], repeat)
        for key in inputs:
            np.testing.assert_array_equal(inputs[key], before[key])
        self.assertEqual(S2, {'top_k': 512, 'graph_k': 128, 'lambda_smooth': 0.15, 'iterations': 200})

    def test_six_method_export_uses_saved_common_subset(self):
        rng = np.random.default_rng(3)
        pos = rng.normal(size=(100, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subset = np.arange(20, dtype=np.int64)
            np.save(root / 'subset.npy', subset)
            paths = {}
            for name in ('B0', 'G1-Identity', 'G1-Graph', 'B0+S2', 'G1-Identity+S2', 'G1-Graph+S2'):
                folder = root / name
                folder.mkdir()
                paths[name] = folder / 'embedding.npz'
                np.savez(paths[name], indices=np.arange(100), positions=pos,
                         embedding=pos + rng.normal(size=pos.shape).astype(np.float32) * 0.1)
                (folder / 'result.json').write_text('{}')
            output = root / 'comparison'
            result = compare_paths(paths, output, subset_path=root / 'subset.npy')
            self.assertEqual(list(result), list(paths))
            np.testing.assert_array_equal(np.load(output / 'metric_subset_seed10.npy'), subset)
            for suffix in ('raw', 'affine'):
                self.assertTrue((output / f'7_column_{suffix}_channel_chart.png').exists())
            for row in result.values():
                self.assertIn('p90_m', row)
                self.assertIn('max_m', row)
                self.assertTrue(Path(row['source_json']).is_file())
            self.assertEqual(json.loads((output / 'eval_protocol.json').read_text())['metric_subset_size'], 20)


if __name__ == '__main__':
    unittest.main()
