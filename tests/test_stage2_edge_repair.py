"""Only Edge solver repair; no IC or Framelet Node solver execution."""
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from diagnostics.stage2_p1.edge_repair import augmented_reference_checks, inputs
from methods.stage2_modern_graph.edge_small import balance_penalty, solve_edge_small


class EdgeRepairTests(unittest.TestCase):
    def test_penalty_change_preserves_multiplier(self):
        vector=np.random.default_rng(41).normal(size=(32,2))
        for primal,dual,expected in [(1.,.001,20.),(.001,1.,5.),(1.,1.,10.)]:
            mu,scaled=balance_penalty(10.,vector,primal,dual)
            self.assertEqual(mu,expected)
            np.testing.assert_array_equal(mu*scaled,10.*vector)

    def test_nonzero_dual_augmented_subproblems(self):
        with threadpool_limits(1):
            checks=augmented_reference_checks(32)
        for check in checks:
            self.assertEqual(check['status'],'PASS',check)

    def test_complete_trace_and_zero_parameter_changes(self):
        x,adj,op=inputs(32)
        with threadpool_limits(1):
            result=solve_edge_small(x,adj,operator=op,nu=.1,max_iter=100,penalty_strategy='residual_balance')
        self.assertFalse(result['converged'])
        final=result['history'][-1]
        self.assertEqual(result['penalties']['mu2'],float(op.degree.mean()))
        for key in ['self_Y_raw','Y_S_raw','Y_row_raw','S_row_raw','frame_raw','primal_residual','dual_residual','prox_S','prox_E','prox_Q']:
            self.assertTrue(np.isfinite(final[key]))
        for change in result['penalty_history']:
            self.assertLess(change['unscaled_multiplier_relative_error'],1e-14)


if __name__=='__main__':
    unittest.main()
