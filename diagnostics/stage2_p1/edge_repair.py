"""P1-Edge Solver Repair: numerical traces/references only; never chart metrics."""
import argparse
import csv
import difflib
import json
import subprocess
import sys
import platform
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import file_record
from diagnostics.stage2_p1.cases import synthetic_case
from diagnostics.stage2_p1.reference import edge_s_reference, edge_u_reference
from methods.stage2_modern_graph.framelet_operator import FrameletOperator
from methods.stage2_modern_graph.edge_small import solve_edge_small, edge_objective_components

ROOT=Path('results/stage2_modern_graph_20261019/p1_edge_solver_repair')
OLD=Path('results/stage2_modern_graph_20261019/p1_math_sanity')
CASES=[('SEF',True,'author'),('FE',False,'author'),('FE',False,'prior'),('DoT',False,'author'),('DoT',False,'prior')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def inputs(n):
    if n==256:
        case=np.load(OLD/'edge256/case.npz');x=case['X'];adj=sparse.csr_matrix(case['adjacency'])
        spec=read(OLD/'edge256/operator_attempts.json')[-1]
    else:
        adj=synthetic_case(n)[0]
        x=np.load(OLD/'ic'/f'n{n}'/'case.npz')['X']
        spec=read(OLD/'framelet'/f'n{n}'/'operator_attempts.json')[-1]
    op=FrameletOperator(adj,spec['n'],spec['lambda_max'])
    return x,adj,op


def save_result(folder,result):
    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    for key in ('Q','Lambda2'):
        arrays.update({f'{key}_{i}':a for i,a in enumerate(result[key])})
    np.savez(folder/'state.npz',**arrays)
    write_json(folder/'solver.json',{k:v for k,v in result.items()
        if k not in arrays and k not in ('history','penalty_history','Q','Lambda2','last_sweep')})
    write_csv(folder/'history.csv',result['history'])
    write_json(folder/'penalty_history.json',result['penalty_history'])


def save_reference(folder,name,ref):
    np.savez(folder/(name+'_arrays.npz'),**{k:v for k,v in ref.items() if isinstance(v,np.ndarray)})
    write_json(folder/(name+'.json'),{k:v for k,v in ref.items() if not isinstance(v,np.ndarray)})


def augmented_reference_checks(n):
    """Independent convex block minimizers at a nonzero-dual third sweep."""
    def solve_problem(problem):
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12,max_iter=500)
        if problem.status!='optimal':raise ValueError('augmented reference failed: '+problem.status)
        return dict(reference_objective=float(problem.value),solver='CLARABEL',status=problem.status,
                    iterations=problem.solver_stats.num_iters,tolerance=1e-12)
    x,adj,op=inputs(n);degree=op.degree;checks=[]
    for label,fixed_u,operator in [('SEF',True,None),('FE',False,None),('DoT',False,op)]:
        result=solve_edge_small(x,adj,operator=operator,nu=.1 if operator else 0.,fixed_u=fixed_u,
                                max_iter=3,audit_last_sweep=True)
        before=result['last_sweep'];u0,e0,s0=before['U'],before['E'],before['S']
        m1,m2,m3,m4=[before[k] for k in ('mu1','mu2','mu3','mu4')]
        l1,l3,l4=before['Lambda1'],before['Lambda3'],before['Lambda4']
        y=cp.Variable((n,n));r1=u0-y@u0-e0;r4=y-s0
        cost=.5*m1*cp.sum_squares(r1)+cp.sum(cp.multiply(l1,r1))+.5*m4*cp.sum_squares(r4)+cp.sum(cp.multiply(l4,r4))
        if not fixed_u:
            r3=y@np.ones((n,1))-1
            cost+=.5*m3*cp.sum_squares(r3)+cp.sum(cp.multiply(l3,r3))
        yr=solve_problem(cp.Problem(cp.Minimize(cost)))
        errors={'Y':float(np.linalg.norm(result['Y']-y.value)/max(1.,np.linalg.norm(y.value)))}
        sv=cp.Variable((n,n));r4=result['Y']-sv
        sr=solve_problem(cp.Problem(cp.Minimize(cp.sum(cp.abs(sv))+.5*m4*cp.sum_squares(r4)+cp.sum(cp.multiply(l4,r4))),[cp.diag(sv)==0]))
        errors['S']=float(np.linalg.norm(result['S']-sv.value)/max(1.,np.linalg.norm(sv.value)))
        native_s_cost=float(np.abs(result['S']).sum()+.5*m4*np.sum((result['Y']-result['S'])**2)+np.sum(l4*(result['Y']-result['S'])))
        sr['native_objective']=native_s_cost
        sr['relative_objective_gap']=float(abs(native_s_cost-sr['reference_objective'])/max(1.,abs(sr['reference_objective'])))
        if not fixed_u:
            uv=cp.Variable(x.shape);r1=(np.eye(n)-result['Y'])@uv-e0
            cost=.5*cp.sum(cp.multiply(degree[:,None],cp.square(uv-x)))+.5*m1*cp.sum_squares(r1)+cp.sum(cp.multiply(l1,r1))
            if operator:
                for t,q,l in zip(op.reference_matrices(),before['Q'],before['Lambda2']):
                    r2=t@uv-q;cost+=.5*m2*cp.sum_squares(r2)+cp.sum(cp.multiply(l,r2))
            ur=solve_problem(cp.Problem(cp.Minimize(cost)))
            errors['U']=float(np.linalg.norm(result['U']-uv.value)/max(1.,np.linalg.norm(uv.value)))
        else:ur=None
        ev=cp.Variable(x.shape);r1=result['U']-result['Y']@result['U']-ev
        cost=cp.sum(cp.multiply(degree,cp.norm(ev,2,axis=1)))+.5*m1*cp.sum_squares(r1)+cp.sum(cp.multiply(l1,r1))
        er=solve_problem(cp.Problem(cp.Minimize(cost)))
        errors['E']=float(np.linalg.norm(result['E']-ev.value)/max(1.,np.linalg.norm(ev.value)))
        qr=[]
        if operator:
            for i,(t,l,c) in enumerate(zip(op.reference_matrices(),before['Lambda2'],(0.,.1/1.25,.025/1.25))):
                qv=cp.Variable(x.shape);r2=t@result['U']-qv
                cost=c*cp.sum(cp.multiply(degree[:,None],cp.abs(qv)))+.5*m2*cp.sum_squares(r2)+cp.sum(cp.multiply(l,r2))
                qr.append(solve_problem(cp.Problem(cp.Minimize(cost))))
                errors[f'Q{i}']=float(np.linalg.norm(result['Q'][i]-qv.value)/max(1.,np.linalg.norm(qv.value)))
        errors['dual1_update']=float(np.max(np.abs(result['Lambda1']-l1-m1*(result['U']-result['Y']@result['U']-result['E']))))
        errors['dual4_update']=float(np.max(np.abs(result['Lambda4']-l4-m4*(result['Y']-result['S']))))
        checks.append(dict(N=n,method=label,relative_solution_errors=errors,threshold=1e-6,
            status='PASS' if max(errors.values())<=1e-6 else 'FAIL',references=dict(Y=yr,S=sr,U=ur,E=er,Q=qr)))
    return checks


def run_case(n,label,fixed_u,initialization,strategy,deadline):
    x,adj,op=inputs(n);operator=op if label=='DoT' else None
    folder=ROOT/f'n{n}'/(label+'-'+initialization)/strategy;folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    result=solve_edge_small(x,adj,operator=operator,nu=.1 if operator else 0.,fixed_u=fixed_u,
                           initialization=initialization,max_iter=15000,penalty_strategy=strategy,wall_deadline=deadline)
    solver_time=time.perf_counter()-started
    save_result(folder,result)
    ref=edge_s_reference(result['U'],op.degree,not fixed_u);save_reference(folder,'Sblock_reference',ref)
    s_native=float(np.abs(result['S']).sum()+np.sum(op.degree*np.linalg.norm(result['U']-result['S']@result['U'],axis=1)))
    sgap=float(abs(s_native-ref['reference_objective'])/max(1e-12,abs(ref['reference_objective'])))
    ugap=0.;uref=None
    if not fixed_u:
        uref=edge_u_reference(x,result['S'],op.degree,op.reference_matrices() if operator else None,.1 if operator else 0.)
        save_reference(folder,'Ublock_reference',uref)
        ugap=float(abs(result['feasible_objective']-uref['reference_objective'])/max(1e-12,abs(uref['reference_objective'])))
    finite=all(np.isfinite(float(value)) for row in result['history'] for value in row.values())
    spread=float(np.linalg.norm(result['U']-result['U'].mean(axis=0))/np.linalg.norm(x-x.mean(axis=0)))
    constraints=max(result[k] for k in ('equality_residual','row_sum_residual','Y_row_max','self_Y_relative','split_residual','frame_primal_residual','diagonal_max'))
    gate=dict(constraints_pass=constraints<=1e-3,solver_completion_pass=result['converged'],
        block_reference_pass=max(sgap,ugap)<=1e-3,finite_noncollapsed_pass=bool(finite and spread>=.05),
        stationarity_pass=result['stationarity']<=5e-3)
    legacy_diff=None
    if n==256 and strategy=='fixed':
        legacy=np.load(OLD/'edge256'/(label.lower()+'-p-small-'+initialization)/'attempt2_arrays.npz')
        legacy_diff={k:float(np.linalg.norm(result[k]-legacy[k])/max(1.,np.linalg.norm(legacy[k]))) for k in ('U','S','E')}
    summary=dict(N=n,method=label,initialization=initialization,strategy=strategy,iterations=result['iterations'],
        status='PASS' if all(gate.values()) else 'NUMERICAL_BLOCKER',**gate,
        solver_time_s=solver_time,total_time_with_reference_s=time.perf_counter()-started,
        objective=result['objective'],equality_eliminated_objective=result['feasible_objective'],objective_components=result['objective_components'],
        Sblock_native_objective=s_native,Sblock_reference_objective=ref['reference_objective'],Sblock_gap=sgap,
        Ublock_reference_objective=None if uref is None else uref['reference_objective'],Ublock_gap=ugap,
        equality=result['equality_residual'],self_Y=result['self_Y_relative'],Y_row=result['Y_row_max'],S_row=result['row_sum_residual'],
        Y_S=result['split_residual'],frame=result['frame_primal_residual'],primal=result['primal_residual'],dual=result['dual_residual'],
        stationarity=result['stationarity'],stopping_error=result['stopping_error'],spread_ratio=spread,
        pcg_residual=result['pcg_max_relative_residual'],penalties=result['penalties'],legacy_relative_differences=legacy_diff,
        source_json=str((folder/'result.json').resolve()),degree_min=float(op.degree.min()),degree_mean=float(op.degree.mean()),
        framelet_order=None if operator is None else op.n,nu=.1 if operator else 0.)
    write_json(folder/'result.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


def run(stage):
    manifest=read(ROOT/'manifest.json');started=time.perf_counter()
    manifest['status']='running_'+stage
    manifest['code']=[file_record(Path(p)) for p in ('methods/stage2_modern_graph/edge_small.py',__file__)]
    manifest.setdefault('commands',[]).append(['D:/anaconda3/envs/diffusion_charting/python.exe','-m','diagnostics.stage2_p1.edge_repair','--stage',stage])
    old_source=subprocess.check_output(['git','cat-file','blob',manifest['original_git_blob']],text=True)
    new_source=Path('methods/stage2_modern_graph/edge_small.py').read_text(encoding='utf-8')
    (ROOT/'edge_small.diff').write_text(''.join(difflib.unified_diff(old_source.splitlines(True),new_source.splitlines(True),fromfile='edge_small.py P1 original',tofile='edge_small.py repair')),encoding='utf-8')
    write_json(ROOT/'manifest.json',manifest)
    with threadpool_limits(1):
        if stage=='small':
            for n in (32,64):
                checks=augmented_reference_checks(n);write_json(ROOT/f'augmented_reference_n{n}.json',checks)
                if any(c['status']!='PASS' for c in checks):
                    raise ValueError('independent augmented subproblem check failed; stop before execution')
        else:
            for n in (32,64):
                if any(c['status']!='PASS' for c in read(ROOT/f'augmented_reference_n{n}.json')):
                    raise ValueError('small-graph algebra gate failed')
        for n in ((32,64) if stage=='small' else (256,)):
            for label,fixed_u,initialization in CASES:
                for strategy in ('fixed','residual_balance'):
                    elapsed=sum(manifest.get('stage_times_s',{}).values())+time.perf_counter()-started
                    if elapsed>=3600:
                        manifest.update(status='NUMERICAL_BLOCKER',stop='fixed one-hour budget reached');write_json(ROOT/'manifest.json',manifest);return
                    run_case(n,label,fixed_u,initialization,strategy,started+3600-sum(manifest.get('stage_times_s',{}).values()))
    manifest.setdefault('stage_times_s',{})[stage]=time.perf_counter()-started
    for item in manifest['protected_inputs']:
        if file_record(Path(item['path']))!=item:raise ValueError('protected artifact changed')
    manifest.update(status=stage+'_completed',protected_inputs_unchanged=True)
    write_json(ROOT/'manifest.json',manifest)


def initialize():
    """Bootstrap a fresh output in an independent checkout; never overwrite."""
    ROOT.mkdir(parents=True,exist_ok=False)
    protected=[Path('AGENTS.md'),Path('methods/stage2_modern_graph/ic_solver.py'),
        Path('methods/stage2_modern_graph/framelet_solver.py'),Path('methods/stage2_modern_graph/framelet_operator.py'),
        Path('diagnostics/stage2_p1/cases.py'),Path('diagnostics/stage2_p1/reference.py')]
    protected += [p for p in OLD.rglob('*') if p.is_file()]
    protected += list(Path('references/stage2_p1/gnn_dot').glob('*.py'))
    edge=Path('methods/stage2_modern_graph/edge_small.py')
    write_json(ROOT/'manifest.json',dict(task='P1-Edge Solver Repair',status='initialized',
        protected_inputs=[file_record(p) for p in protected],original_edge_source=file_record(edge),
        original_git_blob=subprocess.check_output(['git','hash-object','-w',str(edge)],text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        budget=dict(max_iterations=15000,config_retries=0,wall_time_cap_s=3600),
        strategies=dict(A='fixed10',B='normalized residual balance; ratio10/factor2/interval50/iterations100..3000/bounds[.001,10000]; mu2 fixed'),
        stop_tolerance=2e-5,lambda1=1,lambda2=1,nu=.1,
        environment=dict(python=sys.version,numpy=np.__version__,cvxpy=cp.__version__,platform=platform.platform(),blas_threads=1),
        scope='numerical only; no Channel Chart or performance metrics'))
    x,adj,op=inputs(256);legacy=[]
    for label,fixed,init in CASES:
        folder=OLD/'edge256'/(label.lower()+'-p-small-'+init);state=np.load(folder/'attempt2_arrays.npz')
        legacy.append(dict(method=label,initialization=init,components=edge_objective_components(state['U'],state['S'],state['E'],x,op.degree,op if label=='DoT' else None,.1 if label=='DoT' else 0.,fixed)))
    write_json(ROOT/'legacy_diagnosis.json',legacy)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('init','small','full'),required=True)
    stage=parser.parse_args().stage
    initialize() if stage=='init' else run(stage)
