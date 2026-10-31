"""Same-objective CUDA PDHG validation for P2A; no Edge/FN execution."""
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from diagnostics.stage2_p0.preflight import incidence_and_laplacian
from diagnostics.stage2_p1.cases import synthetic_case
from diagnostics.stage2_p1.reference import ic_reference
from methods.stage2_modern_graph.ic_solver import solve_ic
from methods.stage2_modern_graph.ic_cuda import CudaICGraph,solve_ic_cuda


class CudaICTests(unittest.TestCase):
    def test_same_iterates_including_active_auxiliary(self):
        with threadpool_limits(1):
            adjacency,x,*_=synthetic_case(32);x=4*x
            d,w,*_=incidence_and_laplacian(adjacency);w=w/w.mean();g=CudaICGraph(d,w)
            for kind,group in [('IC8',False),('IC9',False),('IC9',True)]:
                for alpha in (.2,.8):
                    out=solve_ic_cuda(x,g,alpha=alpha,eta=.09517167367990964,kind=kind,group=group,max_iter=100,tolerance=0,balance_steps=False)
                    if group:
                        cpu=solve_ic(x,d,w,alpha=alpha,eta=.09517167367990964,kind=kind,group=True,max_iter=100,tolerance=0)['U']
                    else:
                        cpu=np.column_stack([solve_ic(x[:,i:i+1],d,w,alpha=alpha,eta=.09517167367990964,kind=kind,max_iter=100,tolerance=0)['U'].ravel() for i in range(2)])
                    np.testing.assert_allclose(out['U'],cpu,atol=1e-10,rtol=1e-10)

    def test_balanced_steps_same_convex_objective(self):
        with threadpool_limits(1):
            adjacency,x,*_=synthetic_case(32);d,w,*_=incidence_and_laplacian(adjacency);w=w/w.mean();g=CudaICGraph(d,w)
            for kind,group in [('IC8',False),('IC9',False),('IC9',True)]:
                out=solve_ic_cuda(x,g,alpha=.5,eta=.09517167367990964,kind=kind,group=group,max_iter=30000)
                self.assertTrue(out['converged'])
                self.assertLess(out['step_product_times_norm_bound'],1.)
                for i,target in enumerate([x] if group else [x[:,:1],x[:,1:]]):
                    ref=ic_reference(target,d,w,.5,.09517167367990964,kind,group)
                    self.assertLess(abs(out['objective'][i]/ref['reference_objective']-1),1e-3)

    def test_fulln_sized_step_ratio_against_reference(self):
        # Small matrices with large signal norm cover the step ratio used at full N.
        with threadpool_limits(1):
            adjacency,x,*_=synthetic_case(32);x=20*x
            d,w,*_=incidence_and_laplacian(adjacency);w=w/w.mean();g=CudaICGraph(d,w)
            for kind,group in [('IC8',False),('IC9',False),('IC9',True)]:
                out=solve_ic_cuda(x,g,alpha=.5,eta=.09517167367990964,kind=kind,group=group,max_iter=30000)
                self.assertGreater(out['step_scale'],10.)
                self.assertTrue(out['converged'])
                for i,target in enumerate([x] if group else [x[:,:1],x[:,1:]]):
                    ref=ic_reference(target,d,w,.5,.09517167367990964,kind,group)
                    self.assertLess(abs(out['objective'][i]/ref['reference_objective']-1),1e-3)


if __name__=='__main__':unittest.main()
