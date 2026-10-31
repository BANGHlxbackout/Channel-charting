"""Frozen G3 thresholds and separate J/CG references."""
import unittest
from methods.stage2_modern_graph.evaluate_ic_fulln import g3


class G3Tests(unittest.TestCase):
    def test_thresholds_and_deterioration_caps(self):
        base=dict(ks=100.,mean_m=100.,p95_m=100.)
        self.assertTrue(g3(dict(ks=99.,mean_m=99.,p95_m=101.),base,base)['TARGET_HIT'])
        self.assertFalse(g3(dict(ks=99.,mean_m=99.,p95_m=101.001),base,base)['TARGET_HIT'])
        self.assertTrue(g3(dict(ks=99.,mean_m=102.,p95_m=100.),base,base)['SURVIVOR'])
        self.assertFalse(g3(dict(ks=99.,mean_m=102.001,p95_m=100.),base,base)['SURVIVOR'])

    def test_both_baselines_required(self):
        jacobi=dict(ks=100.,mean_m=100.,p95_m=100.)
        cg=dict(ks=98.,mean_m=98.,p95_m=98.)
        out=g3(dict(ks=99.,mean_m=99.,p95_m=99.),jacobi,cg)
        self.assertTrue(out['ADVISOR_HIT'])
        self.assertFalse(out['SURVIVOR'])
        self.assertTrue(out['SOLVER_LEVEL_GAIN_ONLY'])
        self.assertEqual(g3(jacobi,jacobi,jacobi)['label'],'NO_GO')


if __name__=='__main__':unittest.main()
