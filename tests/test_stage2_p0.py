from __future__ import annotations

import unittest
import numpy as np
from scipy import sparse

from diagnostics.stage2_p0.preflight import incidence_and_laplacian, solve_same_target_cg


class Stage2P0Tests(unittest.TestCase):
    def test_each_undirected_edge_once_and_laplacian_identity(self):
        adjacency = sparse.csr_matrix(np.array([[0., 2., 0.], [2., 0., 3.], [0., 3., 0.]]))
        d, weights, left, right, laplacian, degree = incidence_and_laplacian(adjacency)
        self.assertEqual(d.shape, (2, 3))
        np.testing.assert_array_equal(d.toarray(), [[-1., 1., 0.], [0., -1., 1.]])
        np.testing.assert_array_equal(weights, [2., 3.])
        np.testing.assert_array_equal(degree, [2., 5., 3.])
        self.assertTrue(np.all(left < right))
        np.testing.assert_allclose((d.T @ sparse.diags(weights) @ d).toarray(), laplacian.toarray(), atol=1e-14)
        np.testing.assert_allclose(laplacian @ np.ones(3), 0, atol=1e-14)

    def test_two_node_objective_has_no_factor_two_error(self):
        adjacency = sparse.csr_matrix([[0., 2.], [2., 0.]])
        _, _, _, _, laplacian, _ = incidence_and_laplacian(adjacency)
        y = np.array([[0., 0.], [1., 2.]])
        before = y.copy()
        z, info = solve_same_target_cg(laplacian, y, 0.15)
        np.testing.assert_allclose(z, [[0.1875, 0.375], [0.8125, 1.625]], atol=1e-12)
        np.testing.assert_array_equal(y, before)
        self.assertEqual(info['info'], [0, 0])
        self.assertLess(max(info['raw_relative_residuals']), 1e-7)

    def test_gauge_cg_matches_dense_raw_system(self):
        rng = np.random.default_rng(10)
        adjacency = sparse.diags([np.linspace(0.1, 1, 14)] * 2, [-1, 1], shape=(15, 15))
        _, _, _, _, laplacian, _ = incidence_and_laplacian(adjacency)
        y = rng.normal(size=(15, 2)) * 100 + np.array([50., -70.])
        z, info = solve_same_target_cg(laplacian, y, 0.15)
        expected = np.linalg.solve(np.eye(15) + 0.15 * laplacian.toarray(), y)
        np.testing.assert_allclose(z, expected, atol=1e-7, rtol=1e-9)
        self.assertLess(max(info['normalized_relative_residuals']), 1e-7)


if __name__ == '__main__':
    unittest.main()
