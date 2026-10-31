"""Frozen P2A: 27 B0 Robust-IC configurations, G2 first, then G3."""
import argparse
import csv
import ctypes
from ctypes import wintypes
import gc
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
from methods.stage2_modern_graph.ic_cuda import CudaICGraph,solve_ic_cuda

ROOT=Path('results/stage2_modern_graph_20261019/p2a_b0_ic_fulln')
P0=ROOT.parent/'p0_preflight'
B0=Path('results/adp_old_siamese/stride1_n20997_seed10/embedding.npz')
SUBSET=P0/'comparison/metric_subset_seed10.npy'


class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(key,ctypes.c_size_t) for key in
        ('peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]


def rss_bytes():
    counters=MemoryCounters();counters.cb=ctypes.sizeof(counters)
    current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=wintypes.HANDLE
    get=ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
    if not get(current(),ctypes.byref(counters),counters.cb):raise ctypes.WinError()
    return int(counters.working_set)


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def grid():
    rows=[]
    delta0=read(P0/'delta0.json')['delta0']
    for method,kind,group in [('IC8-P','IC8',False),('IC9-P','IC9',False),('IC9-G','IC9',True)]:
        for alpha in (.2,.5,.8):
            for factor in (.5,1.,1.5):
                rows.append(dict(method=method,kind=kind,group=group,alpha=alpha,eta_factor=factor,eta=factor*delta0,
                    slug=f"{method.lower()}_a{int(alpha*100):02d}_eta{int(factor*100):03d}"))
    # Numerical full-run midpoint first for each model; no GT-based ordering.
    return [r for r in rows if r['alpha']==.5 and r['eta_factor']==1.]+[r for r in rows if not (r['alpha']==.5 and r['eta_factor']==1.)]


def initialize():
    if (ROOT/'manifest.json').exists():return
    ROOT.mkdir(parents=True,exist_ok=True)
    old=read(P0/'manifest.json')
    for record in old['output_hashes']:
        if file_record(Path(record['path']))!=record:raise ValueError('P0 saved output hash mismatch: '+record['path'])
    records=list(csv.DictReader((P0/'input_file_hashes.csv').open(encoding='utf-8')))
    for record in records:
        if record['path'].endswith(('geodesic.npy','distance_data.npz','adp_old_siamese\\stride1_n20997_seed10\\embedding.npz')):
            actual=file_record(Path(record['path']))
            if actual['sha256']!=record['sha256']:raise ValueError('P0 source changed: '+record['path'])
    baseline=np.load(B0);gauge=np.load(P0/'chart_gauge.npz')
    assert baseline['embedding'].shape==(20997,2) and np.isfinite(baseline['embedding']).all()
    center=baseline['embedding'].astype(np.float64).mean(axis=0)
    scale=np.sqrt(np.mean(np.sum((baseline['embedding'].astype(np.float64)-center)**2,axis=1)))
    np.testing.assert_allclose((baseline['embedding']-center)/scale,gauge['X'],atol=1e-13,rtol=1e-13)
    protected=[Path('AGENTS.md'),B0,SUBSET,Path('methods/stage2_modern_graph/ic_solver.py'),
        Path('methods/stage2_modern_graph/framelet_solver.py'),Path('methods/stage2_modern_graph/edge_small.py'),
        Path('common_metrics.py'),Path('run_adp_siamese.py'),Path('methods/g1_acf_graph_fcf/compare.py')]
    protected += [p for p in P0.rglob('*') if p.is_file()]
    protected += [ROOT.parent/'p1_math_sanity/manifest.json',ROOT.parent/'p1_edge_solver_repair/manifest.json']
    manifest=dict(phase='P2A_FULL_N',status='prepared',N=20997,grid=grid(),
        delta0=read(P0/'delta0.json')['delta0'],protected_inputs=[file_record(p) for p in dict.fromkeys(protected)],
        numerical_protocol=dict(initial_max_iter=30000,retry_max_iter=120000,retries=1,tolerance=2e-6,feasibility_stop=1e-9,
            dtype='float64',algorithm='same-objective PDHG',step_balance='s=max(1,min_observation_radius/beta); tau=t0*s,sigma=t0/s; tau*sigma*norm_bound=.9025',
            cuda_graph='50 identical steps with exact guarded replay for box or zero-aux certificate failure'),
        metric_protocol=read(P0/'comparison/eval_protocol.json'),environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
            platform=platform.platform(),gpu=torch.cuda.get_device_name(0)),
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),completed=[],commands=[])
    write_json(ROOT/'manifest.json',manifest)
    for name,source in [('b0',B0),('s2_j',P0/'s2_j/embedding.npz'),('s2_cg',P0/'s2_cg/embedding.npz')]:
        folder=ROOT/'references'/name;folder.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(source,folder/'embedding.npz')
        write_json(folder/'result.json',dict(phase='P2A_REUSED_FROZEN_REFERENCE',method=name,
            embedding_source=file_record(source),original_result=str((source.parent/'result.json').resolve()),
            status='FROZEN_REFERENCE',train_time_s=None,peak_vram_bytes=None))
    edges=np.load(P0/'graph/edges_and_degrees.npz');weights=edges['edge_weights'];d=sparse.load_npz(P0/'graph/incidence_D.npz')
    normalized=weights/weights.mean()
    rng=np.random.default_rng(10);probe=rng.normal(size=(20997,2))
    l=sparse.csr_matrix(d.T@sparse.diags(normalized)@d)
    left=d.T@(normalized[:,None]*(d@probe));right=l@probe
    error=float(np.linalg.norm(left-right)/np.linalg.norm(left))
    if error>=1e-12:raise ValueError('IC sparse operator normalization mismatch')
    write_json(ROOT/'graph_and_gauge_checks.json',dict(N=20997,undirected_edges=d.shape[0],
        graph_source=file_record(P0/'graph/G128.npz'),incidence_source=file_record(P0/'graph/incidence_D.npz'),
        raw_weights=dict(min=float(weights.min()),mean=float(weights.mean()),max=float(weights.max())),
        ic_weights=dict(min=float(normalized.min()),mean=float(normalized.mean()),max=float(normalized.max())),
        transformation='W_IC = original W / original undirected positive-edge mean',
        laplacian_incidence_relative_error=error,center=center.tolist(),scale=float(scale),no_GT_in_solver=True))


def attempt(config,number,graph,x):
    folder=ROOT/config['slug']/f'attempt{number}';folder.mkdir(parents=True,exist_ok=False)
    limit=30000 if number==1 else 120000
    stop=threading.Event();rss=[rss_bytes()]
    def sample_memory():
        while not stop.wait(.1):rss.append(rss_bytes())
    worker=threading.Thread(target=sample_memory,daemon=True);worker.start()
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with (folder/'history.csv').open('w',newline='',encoding='utf-8') as file:
        writer=None
        def log(row):
            nonlocal writer
            if writer is None:writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader()
            writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()});file.flush()
            if row['iteration']%5000==0:print(json.dumps(dict(config=config['slug'],attempt=number,**row)),flush=True)
        try:
            out=solve_ic_cuda(x,graph,alpha=config['alpha'],eta=config['eta'],kind=config['kind'],group=config['group'],
                max_iter=limit,callback=log,balance_steps=True)
        finally:
            stop.set();worker.join();rss.append(rss_bytes())
    out['total_attempt_time_s']=time.perf_counter()-started
    out['peak_ram_bytes']=max(rss);out['peak_vram_bytes']=torch.cuda.max_memory_allocated()
    out['peak_reserved_vram_bytes']=torch.cuda.max_memory_reserved()
    np.savez_compressed(folder/'state.npz',U=out['U'],P=out['P'],dual=out['dual'])
    write_json(folder/'solver.json',{k:v for k,v in out.items() if k not in ('U','P','dual','history')})
    return out


def run():
    initialize();manifest=read(ROOT/'manifest.json');manifest['status']='RUNNING_FULL_N_GRID'
    manifest['code']=[file_record(p) for p in Path('methods/stage2_modern_graph').glob('*ic*.py')]
    manifest['commands'].append([sys.executable,'-m','methods.stage2_modern_graph.run_ic_fulln','--phase','run'])
    write_json(ROOT/'manifest.json',manifest)
    data=np.load(B0);gauge=np.load(P0/'chart_gauge.npz');x=gauge['X'];d=sparse.load_npz(P0/'graph/incidence_D.npz')
    raw=np.load(P0/'graph/edges_and_degrees.npz')['edge_weights'];graph=CudaICGraph(d,raw/raw.mean())
    started=time.perf_counter()
    with threadpool_limits(1):
        for config in grid():
            folder=ROOT/config['slug']
            if (folder/'result.json').exists():
                prior=read(folder/'result.json')
                if prior['config']!=config:raise ValueError('resume config mismatch')
                continue
            if folder.exists():raise ValueError('incomplete attempt directory; inspect before rerun: '+str(folder))
            folder.mkdir();write_json(folder/'config.json',config)
            outputs=[]
            for number in (1,2):
                out=attempt(config,number,graph,x);outputs.append(out)
                if out['converged']:break
            out=outputs[-1];u=out['U'];z=gauge['center']+float(gauge['scale'])*u
            finite=bool(np.isfinite(u).all() and np.isfinite(out['P']).all() and np.isfinite(out['dual']).all())
            delta=float(np.linalg.norm(u-x)/np.linalg.norm(x-x.mean(axis=0)))
            spread=float(np.linalg.norm(u-u.mean(axis=0))/np.linalg.norm(x-x.mean(axis=0)))
            max_repeat=int(np.unique(u,axis=0,return_counts=True)[1].max())
            valid=bool(out['converged'] and finite and max(out['feasibility'])<=1e-4 and spread>=.05 and max_repeat/len(x)<.001)
            status='VALID' if valid else 'NONCONVERGED' if not out['converged'] else 'DEGENERATE'
            np.savez_compressed(folder/'embedding.npz',indices=data['indices'],positions=data['positions'],embedding=z)
            np.savez_compressed(folder/'normalized_output.npz',U=u)
            summary=dict(phase='P2A_FULL_N',N=20997,config=config,status=status,numerically_valid=valid,
                attempts=len(outputs),selected_attempt=len(outputs),selection='only numerical convergence; never GT',
                objective=out['objective'],stationarity=out['stationarity'],feasibility=out['feasibility'],
                iterations=out['iterations'],converged=out['converged'],finite=finite,delta=delta,
                delta_axes=(np.linalg.norm(u-x,axis=0)/np.linalg.norm(x-x.mean(axis=0),axis=0)).tolist(),
                spread_ratio=spread,largest_identical_point_fraction=max_repeat/len(x),
                solve_time_s=sum(o['solve_time_s'] for o in outputs),total_runtime_s=sum(o['total_attempt_time_s'] for o in outputs),
                train_time_s=None,peak_ram_bytes=max(o['peak_ram_bytes'] for o in outputs),peak_vram_bytes=max(o['peak_vram_bytes'] for o in outputs),
                gpu='cuda:0',checkpoint='attempt'+str(len(outputs))+'/state.npz',
                data_source=str(B0.resolve()),evaluation_status='NOT_YET_EVALUATED' if valid else 'EXCLUDED_BY_G2')
            write_json(folder/'result.json',summary)
            manifest['completed'].append(config['slug']);manifest['last_completed']=config['slug']
            write_json(ROOT/'manifest.json',manifest)
            print(json.dumps(dict(completed=len(manifest['completed']),config=config['slug'],status=status,iterations=out['iterations'],delta=delta,prox=out['stationarity'])),flush=True)
            del outputs,out;gc.collect();torch.cuda.empty_cache()
    manifest.update(status='NUMERICAL_GRID_COMPLETED',run_wall_time_s=time.perf_counter()-started,stop_after='P2A evaluation/G3 only')
    write_json(ROOT/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','run'),default='run')
    args=parser.parse_args();initialize() if args.phase=='prepare' else run()
