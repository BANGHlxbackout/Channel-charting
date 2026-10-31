"""P2A-R: input-only scale calibration and frozen 5+15 full-N runs."""
import argparse
import csv
import gc
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import file_record
from methods.stage2_modern_graph.ic8_scale_cuda import CudaICGraph,solve_ic8_scale
from methods.stage2_modern_graph.run_ic_fulln import rss_bytes

ROOT=Path('results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation')
FAMILY=ROOT.parent;P0=FAMILY/'p0_preflight'
B0=Path('results/adp_old_siamese/stride1_n20997_seed10/embedding.npz')
SUBSET=P0/'comparison/metric_subset_seed10.npy'
TASK=Path('docs/tasks/P2A_R_Robust_IC_天线数据尺度适配_Codex执行指令_2026-10-20.md')
ETAS=(.03,.08,.16,.32,.50)
QUANTILES=sorted(set(np.linspace(0,1,101).round(8).tolist()+[.005,.995]))


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def distribution(a):
    a=np.asarray(a,dtype=np.float64).ravel()
    if len(a)==0:return dict(count=0,min=None,max=None,mean=None,median=None,quantiles={})
    values=np.quantile(a,QUANTILES)
    return dict(count=len(a),min=float(a.min()),max=float(a.max()),mean=float(a.mean()),median=float(np.median(a)),
                quantiles={f'{q:.3f}':float(v) for q,v in zip(QUANTILES,values)})


def load_inputs():
    gauge=np.load(P0/'chart_gauge.npz');d=sparse.load_npz(P0/'graph/incidence_D.npz')
    raw=np.load(P0/'graph/edges_and_degrees.npz')['edge_weights'];w=raw/raw.mean()
    return gauge,d,w


def grid():
    derived=read(ROOT/'derived_alpha_values.json');alphas=derived.get('values',derived.get('parameters'));configs=[]
    for eta in ETAS:
        configs.append(dict(method='Ball-GLR',alpha_source='disabled',alpha=None,kappa=None,eta=eta,
            aux_mode='disabled',slug=f'ball_glr_eta{round(eta*100):03d}'))
        for a in alphas:
            configs.append(dict(method='IC8-P',alpha_source=a['alpha_source'],alpha=a['alpha'],kappa=a['kappa'],eta=eta,
                aux_mode='enabled',slug=f"ic8_{a['alpha_source']}_eta{round(eta*100):03d}"))
    return configs


def prepare():
    if (ROOT/'manifest.json').exists():
        manifest=read(ROOT/'manifest.json')
        if 'frozen_parameters' in manifest:return
        # The input-only calibration was already saved before implementation.
        # Preserve its bytes and extend only this run's execution manifest.
        for item in manifest['protected_inputs']:
            if file_record(Path(item['path']))!=item:raise ValueError('protected artifact changed: '+item['path'])
        manifest.update(frozen_parameters=[file_record(ROOT/'input_edge_gradient_distribution.json'),file_record(ROOT/'derived_alpha_values.json')],
            grid=grid(),environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),platform=platform.platform()))
        write_json(ROOT/'manifest.json',manifest)
        for label,source in [('b0',B0),('s2_j',P0/'s2_j/embedding.npz'),('s2_cg',P0/'s2_cg/embedding.npz')]:
            folder=ROOT/'references'/label;folder.mkdir(parents=True,exist_ok=False);shutil.copyfile(source,folder/'embedding.npz')
            write_json(folder/'result.json',dict(phase='P2A_R_REUSED_REFERENCE',method=label,status='FROZEN_REFERENCE',
                source=file_record(source),original_result=str((source.parent/'result.json').resolve()),train_time_s=None,peak_vram_bytes=None))
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    old=read(P0/'manifest.json')
    for item in old['output_hashes']:
        if file_record(Path(item['path']))!=item:raise ValueError('P0 artifact changed: '+item['path'])
    protected=[]
    for folder in [FAMILY/'p0_preflight',FAMILY/'p1_math_sanity',FAMILY/'p1_edge_solver_repair',FAMILY/'p2a_b0_ic_fulln',
                   B0.parent,Path('results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10'),
                   Path('results/g1_acf_graph_fcf_followup/e1_stage2')]:
        protected.extend(p for p in folder.rglob('*') if p.is_file())
    protected += [Path('AGENTS.md'),Path('common_metrics.py'),Path('run_adp_siamese.py'),
        Path('run_diffusion_refined_charting.py'),Path('methods/g1_acf_graph_fcf/compare.py')]
    protected += [p for p in Path('methods/stage2_modern_graph').glob('*.py') if not p.name.startswith('ic8_scale')]
    protected=list(dict.fromkeys(protected));hashes=[file_record(p) for p in protected]
    gauge,d,w=load_inputs();x=gauge['X'];grad=np.abs(np.sqrt(w[:,None])*(d@x))
    inputs=[file_record(p) for p in [B0,P0/'chart_gauge.npz',P0/'graph/G128.npz',P0/'graph/incidence_D.npz',P0/'graph/edges_and_degrees.npz']]
    values=[]
    for q,label in [(.95,'q95'),(.99,'q99'),(.995,'q995')]:
        k=float(np.quantile(grad,q));alpha=2*k/(1+2*k)
        values.append(dict(alpha_source=label,quantile=q,kappa=k,alpha=alpha,
            expected_pool_entry_fraction=float(np.mean(grad>k)),expected_edge_union_fraction=float(np.mean(np.any(grad>k,axis=1)))))
    write_json(ROOT/'input_edge_gradient_distribution.json',dict(definition='pooled abs(sqrt(W_IC) D X) across both coordinates',
        N=len(x),undirected_edges=len(w),pool=distribution(grad),x_coordinate=distribution(grad[:,0]),y_coordinate=distribution(grad[:,1]),
        raw_weight_mean=float(np.load(P0/'graph/edges_and_degrees.npz')['edge_weights'].mean()),W_IC_mean=float(w.mean()),
        input_hashes=inputs,X_array_sha256=hashlib.sha256(x.tobytes()).hexdigest(),quantile_interpolation='NumPy linear'))
    write_json(ROOT/'derived_alpha_values.json',dict(formula='alpha=2*kappa/(1+2*kappa)',values=values,input_hashes=inputs,
        frozen_before_GT_evaluation=True,old_alpha_thresholds=[dict(alpha=a,kappa=a/(2*(1-a)),input_entry_exceedance=float(np.mean(grad>a/(2*(1-a))))) for a in (.2,.5,.8)]))
    manifest=dict(task='P2A-R Robust-IC antenna-data scale adaptation',status='PARAMETERS_FROZEN',N=20997,
        task_source=file_record(TASK),protected_inputs=hashes,calibration_inputs=inputs,
        frozen_parameters=[file_record(ROOT/'input_edge_gradient_distribution.json'),file_record(ROOT/'derived_alpha_values.json')],
        grid=grid(),eta_source='fixed dimensionless [.03,.08,.16,.32,.50]; delta0 is not used',
        numerical_protocol=dict(initial_max_iter=30000,retry_max_iter=120000,retries=1,prox_tolerance=2e-6,feasibility_stop=1e-9,dtype='float64',
            step_rule='same P2A radius/beta balancing; Ball beta=1; conservative common norm bound',CUDA_graph_steps=50),
        mechanism_protocol=dict(nonzero='P != 0 exactly; additional abs(P)>1e-12 counts reported',P_active_edge_fraction=1e-4,
            IC_distinct_relative_raw_RMS=1e-4,relative_raw_RMS_denominator='centered B0 RMS; save Ball-centered denominator too',
            direction_cosine='flattened global movement Frobenius inner product; save x/y values too'),
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),platform=platform.platform()),
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),completed=[])
    write_json(ROOT/'manifest.json',manifest)
    for label,source in [('b0',B0),('s2_j',P0/'s2_j/embedding.npz'),('s2_cg',P0/'s2_cg/embedding.npz')]:
        folder=ROOT/'references'/label;folder.mkdir(parents=True,exist_ok=False);shutil.copyfile(source,folder/'embedding.npz')
        write_json(folder/'result.json',dict(phase='P2A_R_REUSED_REFERENCE',method=label,status='FROZEN_REFERENCE',
            source=file_record(source),original_result=str((source.parent/'result.json').resolve()),train_time_s=None,peak_vram_bytes=None))


def attempt(config,number,graph,x):
    folder=ROOT/config['slug']/f'attempt{number}';folder.mkdir(parents=True,exist_ok=False)
    max_iter=30000 if number==1 else 120000;stop=threading.Event();rss=[rss_bytes()]
    def sample():
        while not stop.wait(.1):rss.append(rss_bytes())
    worker=threading.Thread(target=sample,daemon=True);worker.start();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with (folder/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=None
        def log(row):
            nonlocal writer
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()});f.flush()
            if row['iteration']%5000==0:print(json.dumps(dict(config=config['slug'],attempt=number,**row)),flush=True)
        try:
            out=solve_ic8_scale(x,graph,alpha=config['alpha'],eta=config['eta'],aux_mode=config['aux_mode'],max_iter=max_iter,callback=log)
        finally:stop.set();worker.join();rss.append(rss_bytes())
    out.update(attempt_wall_time_s=time.perf_counter()-started,peak_ram_bytes=max(rss),peak_vram_bytes=torch.cuda.max_memory_allocated(),peak_reserved_vram_bytes=torch.cuda.max_memory_reserved())
    np.savez_compressed(folder/'state.npz',U=out['U'],P=out['P'],dual=out['dual'])
    write_json(folder/'solver.json',{k:v for k,v in out.items() if k not in ('U','P','dual','history')})
    return out


def run():
    prepare();manifest=read(ROOT/'manifest.json')
    for item in manifest['frozen_parameters']:
        if file_record(Path(item['path']))!=item:raise ValueError('frozen parameter file changed')
    sanity=read(ROOT/'sanity.json')
    if sanity['status']!='PASS':raise ValueError('small reference failed; stop before full-N')
    manifest.update(status='RUNNING_FULL_N_20',numerical_code=[file_record(Path(p)) for p in
        ['methods/stage2_modern_graph/ic8_scale_cuda.py',__file__]])
    write_json(ROOT/'manifest.json',manifest)
    gauge,d,w=load_inputs();x=gauge['X'];data=np.load(B0);graph=CudaICGraph(d,w);started=time.perf_counter()
    with threadpool_limits(1):
        for config in grid():
            folder=ROOT/config['slug']
            if (folder/'result.json').exists():
                if read(folder/'result.json')['config']!=config:raise ValueError('resume config mismatch')
                continue
            if folder.exists():raise ValueError('incomplete configuration; inspect without adding retry: '+str(folder))
            folder.mkdir();write_json(folder/'config.json',config);outputs=[]
            for number in (1,2):
                out=attempt(config,number,graph,x);outputs.append(out)
                if out['converged']:break
            out=outputs[-1];u,p=out['U'],out['P'];z=gauge['center']+float(gauge['scale'])*u
            finite=all(np.isfinite(a).all() for a in (u,p,out['dual']))
            spread=float(np.linalg.norm(u-u.mean(0))/np.linalg.norm(x-x.mean(0)))
            duplicates=int(np.unique(u,axis=0,return_counts=True)[1].max())/len(x)
            history=np.array([a['objective'] for a in out['history'][-10:]])
            rising=np.diff(history,axis=0)>1e-7*np.maximum(1,np.abs(history[:-1]));sustained=bool(len(rising)>=4 and np.any(np.all(rising[-4:],axis=0)))
            valid=bool(out['converged'] and finite and max(out['feasibility'])<=1e-4 and spread>=.05 and duplicates<.001 and not sustained)
            g=np.sqrt(w[:,None])*(d@u);beta=1. if config['aux_mode']=='disabled' else 1-config['alpha']
            l1=np.zeros(2) if config['aux_mode']=='disabled' else config['alpha']*np.abs(p).sum(0)
            quadratic=beta*np.sum((g-p)**2,axis=0);cpu_objective=(l1+quadratic).tolist()
            objective_error=max(abs(a-b)/max(1.,abs(a)) for a,b in zip(cpu_objective,out['objective']))
            if objective_error>1e-8:raise ValueError('CPU objective disagrees with CUDA')
            if config['aux_mode']=='disabled' and np.count_nonzero(p):raise ValueError('disabled P is nonzero')
            nz=p!=0;edge_nz=np.any(nz,axis=1);edge_fraction=float(edge_nz.mean());delta_axes=np.linalg.norm(u-x,axis=0)/np.linalg.norm(x-x.mean(0),axis=0)
            mechanism=dict(P_nonzero_entries=int(nz.sum()),P_nonzero_edges=int(edge_nz.sum()),P_nonzero_fraction=edge_fraction,
                P_nonzero_entry_fraction=float(nz.mean()),P_L1=float(np.abs(p).sum()),P_absolute_distribution=distribution(np.abs(p)),
                P_nonzero_absolute_distribution=distribution(np.abs(p[nz])),P_nonzero_edges_above_1e_minus12=int(np.any(np.abs(p)>1e-12,axis=1).sum()),
                final_abs_weighted_edge_gradient_distribution=distribution(np.abs(g)),final_gradient_x=distribution(np.abs(g[:,0])),final_gradient_y=distribution(np.abs(g[:,1])),
                P_active=bool(edge_fraction>=1e-4),regime='QUADRATIC_REFERENCE' if config['aux_mode']=='disabled' else 'QUADRATIC_REGIME' if not nz.any() else 'P_ACTIVE' if edge_fraction>=1e-4 else 'SUBTHRESHOLD_ACTIVITY',
                objective_L1_components=l1.tolist(),objective_quadratic_components=quadratic.tolist(),P_nonzero_entries_xy=nz.sum(0).tolist())
            record=dict(phase='P2A_R_FULL_N',N=20997,config=config,status='VALID' if valid else 'NONCONVERGED' if not out['converged'] or sustained else 'DEGENERATE',
                numerically_valid=valid,attempts=len(outputs),selected_attempt=len(outputs),checkpoint_selection='only frozen numerical convergence; no GT',
                iterations=out['iterations'],converged=out['converged'],objective=out['objective'],stationarity=out['stationarity'],feasibility=out['feasibility'],
                movement_x=float(delta_axes[0]),movement_y=float(delta_axes[1]),ball_usage_ratio_xy=(delta_axes/config['eta']).tolist(),
                raw_chart_movement_RMS_xy=(float(gauge['scale'])*np.sqrt(np.mean((u-x)**2,axis=0))).tolist(),
                delta=float(np.linalg.norm(u-x)/np.linalg.norm(x-x.mean(0))),spread_ratio=spread,largest_identical_point_fraction=duplicates,
                G2=dict(finite=bool(finite),objective_no_sustained_worsening=not sustained,CPU_objective=cpu_objective,CPU_CUDA_objective_relative_error=objective_error),
                solve_time_s=sum(a['solve_time_s'] for a in outputs),total_runtime_s=sum(a['attempt_wall_time_s'] for a in outputs),
                peak_ram_bytes=max(a['peak_ram_bytes'] for a in outputs),peak_vram_bytes=max(a['peak_vram_bytes'] for a in outputs),train_time_s=None,
                evaluation_status='NOT_EVALUATED' if valid else 'EXCLUDED_BY_G2',mechanism=mechanism)
            np.savez_compressed(folder/'embedding.npz',indices=data['indices'],positions=data['positions'],embedding=z)
            np.savez_compressed(folder/'normalized_output.npz',U=u);write_json(folder/'result.json',record)
            write_json(folder/'mechanism.json',mechanism)
            manifest['completed'].append(config['slug']);write_json(ROOT/'manifest.json',manifest)
            print(json.dumps(dict(completed=len(manifest['completed']),config=config['slug'],status=record['status'],iterations=out['iterations'],P_edge_fraction=edge_fraction,delta=record['delta'])),flush=True)
            del outputs,out;gc.collect();torch.cuda.empty_cache()
    for item in manifest['protected_inputs']:
        if file_record(Path(item['path']))!=item:raise ValueError('historical artifact changed')
    manifest.update(status='NUMERICAL_20_COMPLETED',run_wall_time_s=time.perf_counter()-started,protected_inputs_unchanged=True,
                    stop_after='unified P2A-R evaluation/report only')
    write_json(ROOT/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','run'),default='run');args=parser.parse_args()
    prepare() if args.phase=='prepare' else run()
