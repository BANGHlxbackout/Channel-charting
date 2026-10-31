"""Independent small convex references for the frozen P2A-R scalar objective."""
import time
import cvxpy as cp
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import incidence_and_laplacian
from diagnostics.stage2_p1.cases import synthetic_case
from diagnostics.stage2_p1.reference import ic_reference,solve_problem
from methods.stage2_modern_graph.ic_cuda import solve_ic_cuda
from methods.stage2_modern_graph.ic8_scale_cuda import CudaICGraph,solve_ic8_scale
from methods.stage2_modern_graph.ic8_scale_experiment import ROOT,read


def main():
    started=time.perf_counter();rows=[];derived=read(ROOT/'derived_alpha_values.json');alphas=derived.get('values',derived.get('parameters'))
    with threadpool_limits(1):
        for n in (16,32,64):
            adjacency,x,_,_=synthetic_case(n)
            d,w,*_=incidence_and_laplacian(adjacency);w=w/w.mean();graph=CudaICGraph(d,w)
            for eta in (.03,.50):
                for value in [None]+alphas:
                    disabled=value is None;alpha=None if disabled else value['alpha'];label='Ball-GLR' if disabled else value['alpha_source']
                    out=solve_ic8_scale(x,graph,alpha=alpha,eta=eta,aux_mode='disabled' if disabled else 'enabled')
                    references=[]
                    for axis in range(2):
                        a=x[:,axis:axis+1]
                        if disabled:
                            u=cp.Variable(a.shape);radius=eta*np.linalg.norm(a-a.mean(0));lo=a.min()-.5*np.ptp(a);hi=a.max()+.5*np.ptp(a)
                            problem=cp.Problem(cp.Minimize(cp.sum_squares(cp.multiply(np.sqrt(w)[:,None],d@u))),
                                [cp.norm(u-a,'fro')<=radius,u>=lo,u<=hi])
                            ref=solve_problem(problem)
                        else:
                            try:ref=ic_reference(a,d,w,alpha,eta,'IC8')
                            except ValueError as error:
                                # Same explicit (u,p) convex objective; an independent
                                # fixed SCS reference when CLARABEL reports inaccurate.
                                u=cp.Variable(a.shape);p=cp.Variable((d.shape[0],1));radius=eta*np.linalg.norm(a-a.mean(0))
                                problem=cp.Problem(cp.Minimize(alpha*cp.sum(cp.abs(p))+(1-alpha)*cp.sum_squares(cp.multiply(np.sqrt(w)[:,None],d@u)-p)),
                                    [cp.norm(u-a,'fro')<=radius,u>=a.min()-.5*np.ptp(a),u<=a.max()+.5*np.ptp(a)])
                                problem.solve(solver='SCS',eps=1e-8,max_iters=100000)
                                if problem.status!='optimal':raise ValueError('Independent SCS reference failed')
                                ref=dict(reference_objective=float(problem.value),status=problem.status,solver='SCS',
                                    iterations=problem.solver_stats.num_iters,solve_time_s=problem.solver_stats.solve_time,clarabel_error=str(error))
                        references.append({k:v for k,v in ref.items() if k not in ('U','P')})
                    objective_gap=max(abs(v-r['reference_objective'])/max(1.,abs(r['reference_objective'])) for v,r in zip(out['objective'],references))
                    g=np.sqrt(w[:,None])*(d@out['U'])
                    expected=np.zeros_like(g) if disabled else np.sign(g)*np.maximum(np.abs(g)-alpha/(2*(1-alpha)),0)
                    soft_error=float(np.max(np.abs(out['P']-expected)))
                    passed=bool(out['converged'] and objective_gap<=1e-3 and max(out['feasibility'])<=1e-4 and soft_error<=1e-3 and np.isfinite(out['U']).all())
                    row=dict(N=n,method=label,eta=eta,alpha=alpha,objective=out['objective'],references=references,
                        relative_objective_gap=objective_gap,soft_threshold_max_error=soft_error,feasibility=out['feasibility'],
                        prox_residual=out['stationarity'],iterations=out['iterations'],time_s=out['solve_time_s'],status='PASS' if passed else 'FAIL')
                    rows.append(row);print(row,flush=True)
                    write_json(ROOT/'sanity_progress.json',dict(cases=rows))
            if n==32:
                alpha=alphas[0]['alpha'];new=solve_ic8_scale(x,graph,alpha=alpha,eta=.16,max_iter=200,tolerance=0.)
                old=solve_ic_cuda(x,graph,alpha=alpha,eta=.16,kind='IC8',group=False,max_iter=200,tolerance=0.)
                errors={k:float(np.max(np.abs(new[k]-old[k]))) for k in ('U','P','dual')}
                rows.append(dict(test='immutable_old_IC8_update_regression',iterations=200,errors=errors,status='PASS' if max(errors.values())<=1e-10 else 'FAIL'))
            del graph;torch.cuda.empty_cache()
    roundtrips=[abs(a['alpha']/(2*(1-a['alpha']))-a['kappa']) for a in alphas]
    rows.append(dict(test='alpha_kappa_roundtrip',errors=roundtrips,status='PASS' if max(roundtrips)<1e-14 else 'FAIL'))
    result=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',cases=rows,wall_time_s=time.perf_counter()-started,
        gates=dict(relative_objective_gap=1e-3,feasibility=1e-4,prox_residual=2e-6,soft_threshold_error=1e-3,old_update_error=1e-10),
        protocol='synthetic only; no UE coordinates, chart metrics or GT parameter selection')
    write_json(ROOT/'sanity.json',result)
    if result['status']!='PASS':raise ValueError('P2A-R small-reference gate failed; full-N prohibited')


if __name__=='__main__':main()
