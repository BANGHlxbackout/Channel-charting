"""Targeted independent checks for the frozen P1 numerical implementations."""
import unittest

import numpy as np

from diagnostics.stage2_p0.preflight import incidence_and_laplacian
from diagnostics.stage2_p1.cases import synthetic_case
from diagnostics.stage2_p1.reference import ic_reference, framelet_reference, edge_s_reference
from methods.stage2_modern_graph.ic_solver import solve_ic, shrink
from methods.stage2_modern_graph.framelet_operator import FrameletOperator, operator_checks
from methods.stage2_modern_graph.framelet_solver import solve_framelet
from methods.stage2_modern_graph.edge_small import solve_edge_small


class P1MathTests(unittest.TestCase):
    def setUp(self):
        self.adjacency, self.x, _, _ = synthetic_case(16)
        self.d, self.w, *_ = incidence_and_laplacian(self.adjacency)
        degree = np.asarray(self.adjacency.sum(axis=1)).ravel()
        lap = np.eye(16) - self.adjacency.toarray() / np.sqrt(degree[:, None] * degree[None, :])
        self.op = FrameletOperator(self.adjacency, 4, float(np.linalg.eigvalsh(lap).max()))

    def test_ic8_huber(self):
        g = np.sqrt(self.w)[:, None] * (self.d @ self.x)
        p = shrink(g, .5)
        actual = .5 * np.abs(p) + .5 * (g - p) ** 2
        reference = np.where(np.abs(g) <= .5, .5 * g ** 2, .5 * np.abs(g) - .125)
        np.testing.assert_allclose(actual, reference, atol=1e-12)

    def test_ic_convex_references_and_eta_zero(self):
        weights = self.w / self.w.mean()
        for kind, group in [('IC8', False), ('IC9', False), ('IC9', True)]:
            x = self.x if group else self.x[:, :1]
            result = solve_ic(x, self.d, weights, kind=kind, group=group)
            ref = ic_reference(x, self.d, weights, .5, .095, kind, group)
            self.assertLess(abs(result['objective'] / ref['reference_objective'] - 1), 1e-3)
            self.assertLess(result['feasibility'], 1e-4)
            zero = solve_ic(x, self.d, weights, kind=kind, group=group, eta=0.)
            np.testing.assert_allclose(zero['U'], x, atol=1e-12)

    def test_framelet_zero_mode_and_adjoint(self):
        checks = operator_checks(self.op, np.random.default_rng(81))
        self.assertEqual(self.op.active_bands, ('low2', 'high1', 'high2'))
        self.assertLess(checks['adjoint_relative_error'], 1e-6)
        self.assertLess(checks['zero_mode_residual'], 1e-8)

    def test_framelet_exact_u_step_and_reference(self):
        for group in (False, True):
            result = solve_framelet(self.x, self.op, .1, group, audit_u_step=True)
            ref = framelet_reference(self.x, self.op.reference_matrices(), self.op.degree, .1, group)
            self.assertLess(abs(result['objective'] / ref['reference_objective'] - 1), 1e-3)
            self.assertLess(result['u_step_dense_relative_error'], 1e-5)
            self.assertLess(result['pcg_max_relative_residual'], 1e-5)
        zero = solve_framelet(self.x, self.op, 0.)
        np.testing.assert_allclose(zero['U'], self.x, atol=1e-8)

    def test_sef_convex_reference(self):
        result = solve_edge_small(self.x, self.adjacency, fixed_u=True)
        ref = edge_s_reference(self.x, self.op.degree, row_sum=False)
        self.assertLess(abs(result['feasible_objective'] / ref['reference_objective'] - 1), 1e-3)
        self.assertLess(result['equality_residual'], 1e-3)
        self.assertEqual(result['diagonal_max'], 0.)


if __name__ == '__main__':
    unittest.main()
