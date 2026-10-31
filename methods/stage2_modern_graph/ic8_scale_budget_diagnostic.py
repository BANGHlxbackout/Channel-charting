"""Explain the eta=.50 iteration cutoff using saved iterates, without GT."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common_metrics import write_json
from methods.stage2_modern_graph.ic8_scale_experiment import ROOT,load_inputs,read
from methods.stage2_modern_graph.ic_solver import project_observation


def history(path):
    rows=list(csv.DictReader(path.open()))
    return np.array([int(r['iteration']) for r in rows]),np.array([json.loads(r['objective']) for r in rows]),np.array([json.loads(r['prox_residual']) for r in rows])


def main():
    folder=ROOT/'ball_glr_eta050';solver=read(folder/'attempt1/solver.json');accepted=read(folder/'attempt2/solver.json')
    t,objective,residual=history(folder/'attempt1/history.csv');t2,o2,r2=history(folder/'attempt2/history.csv')
    tail=t>=15000;rates=[];predictions=[]
    for axis in range(2):
        slope,intercept=np.polyfit(t[tail],np.log(residual[tail,axis]),1)
        rates.append(float(-slope));predictions.append(float((np.log(2e-6)-intercept)/slope))
    gauge,d,w=load_inputs();x=gauge['X'];state=np.load(folder/'attempt1/state.npz');u,p,q=state['U'],state['P'],state['dual']
    gradient=np.sqrt(w[:,None])*(d@u);gu=d.T@(np.sqrt(w[:,None])*q);tau,sigma=solver['tau'],solver['sigma']
    projected=np.column_stack([project_observation((u-tau*gu)[:,i:i+1],x[:,i:i+1],solver['radius'][i],
        (solver['box'][0][i],solver['box'][1][i])).ravel() for i in range(2)])
    norm=lambda a:np.linalg.norm(a,axis=0)
    primal=norm(u-projected)/tau/(1+norm(u)+norm(gu))
    dual=norm(q-(q+sigma*gradient)/(1+sigma/2))/sigma/(1+norm(q)+norm(gradient))
    accepted_u=np.load(folder/'attempt2/state.npz')['U']
    true_gu=2*(d.T@(np.sqrt(w[:,None])*gradient))
    true_projected=np.column_stack([project_observation((u-tau*true_gu)[:,i:i+1],x[:,i:i+1],solver['radius'][i],
        (solver['box'][0][i],solver['box'][1][i])).ravel() for i in range(2)])
    direct_primal_KKT=norm(u-true_projected)/tau/(1+norm(u)+norm(true_gu))
    objective_change=np.max(np.abs(objective[t==5000][0]-objective[-1])/np.maximum(1,np.abs(objective[-1])))
    rows=[]
    for eta,label in [(.03,'003'),(.08,'008'),(.16,'016'),(.32,'032'),(.50,'050')]:
        s=read(ROOT/f'ball_glr_eta{label}/attempt1/solver.json')
        rows.append(dict(eta=eta,tau=s['tau'],sigma=s['sigma'],step_scale=s['step_scale'],iterations=s['iterations'],
            converged=s['converged'],final_prox=s['stationarity']))
    result=dict(GT_used=False,diagnosis='BUDGET_LIMIT_AND_SLOW_DUAL_RELAXATION; not divergence or a demonstrated objective failure',
        initial_cap=30000,accepted_iteration=accepted['iterations'],max_prox_at_initial_cap=max(solver['stationarity']),
        accepted_max_prox=max(accepted['stationarity']),primal_prox_xy=primal.tolist(),dual_prox_xy=dual.tolist(),
        direct_primal_KKT_using_exact_quadratic_gradient_xy=direct_primal_KKT.tolist(),
        U_relative_difference_30000_vs_accepted=float(np.linalg.norm(u-accepted_u)/np.linalg.norm(accepted_u)),
        fitted_decay_rate_xy=rates,theoretical_dual_relaxation_log_rate=float(np.log1p(sigma/2)),
        predicted_threshold_iteration_xy=predictions,objective_relative_change_5000_to_30000=float(objective_change),
        step_product_times_operator_norm_bound=solver['step_product_times_norm_bound'],step_rule='scale=max(1,min(radius)/beta); tau=base*scale; sigma=base/scale',
        interpretation='When scale>1, sigma/(2*beta)=base/(2*min(radius)); increasing eta slows dual relaxation. Final actual certificate is still required.',
        ball_budget_scaling=rows,amendment=read(ROOT/'solver_budget_amendment.json'),
        plan_judgment='Objective, matched-eta control and input-only thresholds are coherent. A uniform 30000 cap from the smaller old eta range is insufficient at eta=.50; repeated cold restarts waste work. No evidence here requires changing alpha, graph or objective.')
    write_json(ROOT/'solver_budget_diagnosis.json',result)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4))
    for axis in range(2):
        axes[0].plot(t,objective[:,axis],label=f'axis {axis}')
        axes[1].semilogy(t,residual[:,axis],label=f'initial axis {axis}')
        axes[1].semilogy(t2,r2[:,axis],linestyle='--',label=f'accepted cold retry {axis}')
    axes[1].axhline(2e-6,color='black',linestyle=':',label='prox tolerance')
    axes[1].axvline(30000,color='red',linestyle=':',label='initial cap')
    axes[2].semilogy([r['eta'] for r in rows],[r['sigma'] for r in rows],marker='o')
    for ax,title in zip(axes,['Ball-GLR eta=.50 objective','Proximal certificate decay','Dual step vs eta']):
        ax.set_title(title);ax.grid(alpha=.25);ax.set_xlabel('iteration' if ax is not axes[2] else 'eta')
    axes[0].legend();axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'solver_budget_diagnosis.png',dpi=180);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['primal_prox_xy','dual_prox_xy','fitted_decay_rate_xy','theoretical_dual_relaxation_log_rate','predicted_threshold_iteration_xy','objective_relative_change_5000_to_30000']}))


if __name__=='__main__':main()
