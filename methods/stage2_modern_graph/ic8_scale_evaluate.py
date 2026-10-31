"""Frozen P2A-R audit, same-eta mechanism comparisons and standard chart outputs."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import file_record
from methods.g1_acf_graph_fcf.compare import compare_paths
from methods.stage2_modern_graph.evaluate_ic_fulln import hit,g3,METRICS
from methods.stage2_modern_graph.ic8_scale_experiment import ROOT,P0,B0,SUBSET,ETAS,grid,load_inputs,read
from run_adp_siamese import plot_chart


def name(c):
    return 'Ball-GLR' if c['aux_mode']=='disabled' else f"IC8-P {c['alpha_source']}"


def write_table(folder,rows,title):
    folder.mkdir(parents=True,exist_ok=True);keys=list(rows[0])
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    lines=['# '+title,'','Same 4199-node seed10 CT/TW/KS sample; affine errors on all 20997 identical nodes. GT evaluation only.','',
        '| '+' | '.join(keys)+' |','|'+'---|'*len(keys)]
    for row in rows:
        values=[]
        for k,v in row.items():
            if k=='source_json':v=f'[JSON]({Path(v).as_posix()})'
            elif v is None:v='unavailable'
            elif isinstance(v,float):v=f'{v:.6f}'
            values.append(str(v))
        lines.append('| '+' | '.join(values)+' |')
    (folder/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def cosine(a,b):
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator else None


def main():
    started=time.perf_counter();manifest=read(ROOT/'manifest.json');configs=grid()
    if manifest['status']!='NUMERICAL_20_COMPLETED':raise ValueError('All 20 numerical configurations must finish before evaluation')
    for item in manifest['protected_inputs']+manifest['frozen_parameters']:
        if file_record(Path(item['path']))!=item:raise ValueError('Protected input changed: '+item['path'])
    gauge,d,w=load_inputs();x=gauge['X'];data=np.load(B0);records={};gates=[]
    # All checks and raw output differences precede the first GT quality metric.
    for c in configs:
        folder=ROOT/c['slug'];r=read(folder/'result.json');saved=np.load(folder/'embedding.npz')
        if not np.array_equal(saved['indices'],data['indices']) or not np.array_equal(saved['positions'],data['positions']):raise ValueError('node identity changed')
        u=np.load(folder/'normalized_output.npz')['U'];state=np.load(folder/f"attempt{r['selected_attempt']}/state.npz");p=state['P']
        expected=gauge['center']+float(gauge['scale'])*u
        if not np.array_equal(expected,saved['embedding']):raise ValueError('raw/gauge mismatch')
        beta=1. if c['aux_mode']=='disabled' else 1-c['alpha'];a=0. if c['aux_mode']=='disabled' else c['alpha']
        weighted_gradient=np.sqrt(w[:,None])*(d@u)
        objective=a*np.abs(p).sum(0)+beta*np.sum((weighted_gradient-p)**2,axis=0)
        objective_error=float(np.max(np.abs(objective-r['objective'])/np.maximum(1.,np.abs(objective))))
        if objective_error>1e-8:raise ValueError('independent objective mismatch')
        r['G2'].update(node_identity=True,raw_gauge_identity=True,CPU_objective_relative_error=objective_error)
        if c['aux_mode']=='enabled':
            ball=next(a for a in configs if a['eta']==c['eta'] and a['aux_mode']=='disabled')
            bu=np.load(ROOT/ball['slug']/'normalized_output.npz')['U'];diff=u-bu;relative=float(np.linalg.norm(diff)/np.linalg.norm(x-x.mean(0)))
            absolute=float(gauge['scale'])*float(np.sqrt(np.mean(np.sum(diff**2,axis=1))))
            threshold_p=np.sign(weighted_gradient)*np.maximum(np.abs(weighted_gradient)-c['kappa'],0.)
            m=r['mechanism'];m.update(relative_raw_RMS_difference_B0=relative,absolute_raw_RMS_difference=absolute,
                final_P_soft_threshold_max_error=float(np.max(np.abs(p-threshold_p))),
                final_U_threshold_predicted_nonzero_edges=int(np.any(threshold_p!=0,axis=1).sum()),
                relative_raw_RMS_difference_Ball=float(np.linalg.norm(diff)/np.linalg.norm(bu-bu.mean(0))),
                movement_direction_cosine=cosine(u-x,bu-x),movement_direction_cosine_xy=[cosine(u[:,i]-x[:,i],bu[:,i]-x[:,i]) for i in range(2)],
                IC_distinct=bool(m['P_active'] and relative>=1e-4),matched_ball_slug=ball['slug'])
            write_json(folder/'mechanism.json',m)
        records[c['slug']]=r;write_json(folder/'result.json',r)
        gates.append(dict(config=c['slug'],status='PASS' if r['numerically_valid'] else 'FAIL',iterations=r['iterations'],attempts=r['attempts'],
            converged=r['converged'],prox_residual=r['stationarity'],feasibility=r['feasibility'],spread=r['spread_ratio'],
            max_duplicate_fraction=r['largest_identical_point_fraction'],objective_error=objective_error,source_json=str((folder/'result.json').resolve())))
    write_json(ROOT/'G2.json',dict(cases=gates,valid=sum(r['numerically_valid'] for r in records.values()),total=20))
    references={label:ROOT/'references'/key/'embedding.npz' for label,key in [('B0','b0'),('B0+S2-J','s2_j'),('B0+S2-CG','s2_cg')]}
    old={r['method']:r for r in csv.DictReader((P0/'comparison/comparison.csv').open(encoding='utf-8'))}
    all_rows=[];mechanism_rows=[];decisions={};baseline_diffs={};metric_cache={}
    for eta in ETAS:
        group=[c for c in configs if c['eta']==eta];valid=[c for c in group if records[c['slug']]['numerically_valid']]
        folder=ROOT/'comparison'/f'eta{round(eta*100):03d}'
        paths={**references,**{name(c):ROOT/c['slug']/'embedding.npz' for c in valid}}
        with threadpool_limits(1):metrics=compare_paths(paths,folder,seed=10,subset_path=SUBSET)
        protocol=read(folder/'eval_protocol.json')
        protocol.update(colors='same GT-normalized x/y RGB in every panel; blue=0.25',
            GT_use='evaluation/plots only; no solver, parameter or numerical checkpoint selection',
            metric_functions=['common_metrics.chart_metrics','common_metrics.localization_error_summary'],
            CT_TW_neighbors=max(1,int(.05*len(np.load(SUBSET)))),node_identity='exact indices and positions checked before metrics')
        write_json(folder/'eval_protocol.json',protocol)
        prefix='four_column' if len(paths)+1==4 else f'{len(paths)+1}_column'
        for suffix in ('raw','affine'):(folder/f'{prefix}_{suffix}_channel_chart.png').rename(folder/f'{suffix}_channel_chart.png')
        for label,path in paths.items():
            values=metrics[label];r=read(path.parent/'result.json')
            r.update(metrics={k:values[k] for k in METRICS[:3]},localization_errors_after_affine={k:values[k] for k in METRICS[3:]},evaluation_status='EVALUATED_FROZEN_PROTOCOL')
            write_json(path.parent/'result.json',r)
            if label in references:
                diffs={k:abs(values[k]-float(old[label][k])) for k in METRICS}
                if max(diffs.values())>1e-10:raise ValueError('Historical baseline evaluation mismatch')
                baseline_diffs[label]=diffs
            if label not in metric_cache or label not in references:
                saved=np.load(path);plot_chart(path.parent/'chart.png',saved['positions'],saved['embedding'],
                    f"{label}"+('' if label in references else f' eta={eta:g}')+f"\nCT={values['ct']:.3f}, TW={values['tw']:.3f}, KS={values['ks']:.3f}")
            metric_cache[label]=values
        rows=[]
        for label,path in references.items():
            rows.append(dict(method=label,alpha_source=None,alpha=None,kappa=None,eta=None,**{k:metrics[label][k] for k in METRICS},
                P_nonzero_edges=None,P_nonzero_fraction=None,movement_x=None,movement_y=None,status='FROZEN_REFERENCE',
                source_json=str((path.parent/'result.json').resolve()),solve_time_s=None,peak_ram_bytes=None,peak_vram_bytes=None))
        if not all_rows:all_rows.extend(rows)
        for c in group:
            r=records[c['slug']];is_valid=r['numerically_valid'];values=metrics[name(c)] if is_valid else {k:None for k in METRICS}
            decision=g3(values,metrics['B0+S2-J'],metrics['B0+S2-CG']) if is_valid else dict(label='EXCLUDED_NONCONVERGED',ADVISOR_HIT=False,SURVIVOR=False,TARGET_HIT=False)
            decisions[c['slug']]=decision;r.update(G3=decision)
            if c['aux_mode']=='enabled':
                m=r['mechanism'];ball_record=next(records[a['slug']] for a in group if a['aux_mode']=='disabled');pair_valid=is_valid and ball_record['numerically_valid']
                signal,improvement=hit(values,metrics['Ball-GLR'],1,.02) if pair_valid else (False,{k:None for k in ('ks','mean_m','p95_m')})
                positive=bool(pair_valid and m['P_active'] and m['IC_distinct'] and signal)
                m.update(pair_numerically_valid=pair_valid,relative_improvement_vs_Ball=improvement,MECHANISM_POSITIVE_SIGNAL=positive,
                    BALL_CONSTRAINT_GAIN=bool(is_valid and decision['SURVIVOR'] and (m['regime']=='QUADRATIC_REGIME' or not m['IC_distinct'])))
                mechanism_rows.append(dict(config=c['slug'],eta=eta,alpha_source=c['alpha_source'],alpha=c['alpha'],kappa=c['kappa'],
                    P_nonzero_edges=m['P_nonzero_edges'],P_nonzero_fraction=m['P_nonzero_fraction'],P_nonzero_entry_fraction=m['P_nonzero_entry_fraction'],P_L1=m['P_L1'],
                    regime=m['regime'],P_active=m['P_active'],relative_raw_RMS_difference_B0=m['relative_raw_RMS_difference_B0'],absolute_raw_RMS_difference=m['absolute_raw_RMS_difference'],
                    relative_raw_RMS_difference_Ball=m['relative_raw_RMS_difference_Ball'],movement_direction_cosine=m['movement_direction_cosine'],IC_distinct=m['IC_distinct'],
                    pair_numerically_valid=pair_valid,KS_improvement_vs_Ball=improvement['ks'],Mean_improvement_vs_Ball=improvement['mean_m'],P95_improvement_vs_Ball=improvement['p95_m'],
                    MECHANISM_POSITIVE_SIGNAL=positive,BALL_CONSTRAINT_GAIN=m['BALL_CONSTRAINT_GAIN'],G3=decision['label'],source_json=str((ROOT/c['slug']/'result.json').resolve())))
                write_json(ROOT/c['slug']/'mechanism.json',m)
            # Merge metrics just saved by compare_paths loop with numerical facts.
            updated=read(ROOT/c['slug']/'result.json');updated.update(G3=decision,mechanism=r['mechanism']);write_json(ROOT/c['slug']/'result.json',updated)
            row=dict(method=name(c),alpha_source=c['alpha_source'],alpha=c['alpha'],kappa=c['kappa'],eta=eta,**{k:values[k] for k in METRICS},
                P_nonzero_edges=r['mechanism']['P_nonzero_edges'],P_nonzero_fraction=r['mechanism']['P_nonzero_fraction'],movement_x=r['movement_x'],movement_y=r['movement_y'],status=r['status'],
                source_json=str((ROOT/c['slug']/'result.json').resolve()),solve_time_s=r['solve_time_s'],peak_ram_bytes=r['peak_ram_bytes'],peak_vram_bytes=r['peak_vram_bytes'])
            rows.append(row);all_rows.append(row)
        write_table(folder,rows,f'P2A-R eta={eta:g}')
        print(json.dumps(dict(evaluated_eta=eta,valid=len(valid),elapsed_s=time.perf_counter()-started)),flush=True)
    write_table(ROOT/'comparison',all_rows,'P2A-R complete frozen parameter comparison')
    save_csv(ROOT/'parameter_scan.csv',all_rows[3:]);save_csv(ROOT/'mechanism_summary.csv',mechanism_rows)
    write_json(ROOT/'G3.json',dict(configurations=decisions,mechanism_positive_count=sum(r['MECHANISM_POSITIVE_SIGNAL'] for r in mechanism_rows),
        IC8_active_count=sum(r['P_active'] for r in mechanism_rows),IC8_distinct_count=sum(r['IC_distinct'] for r in mechanism_rows),
        IC8_survivor_count=sum(decisions[c['slug']]['SURVIVOR'] for c in configs if c['aux_mode']=='enabled'),
        IC8_advisor_count=sum(decisions[c['slug']]['ADVISOR_HIT'] for c in configs if c['aux_mode']=='enabled'),
        IC8_target_count=sum(decisions[c['slug']]['TARGET_HIT'] for c in configs if c['aux_mode']=='enabled'),
        IC8_task_status='SURVIVOR_EXPLORATORY' if any(decisions[c['slug']]['SURVIVOR'] for c in configs if c['aux_mode']=='enabled') else 'NO_GO_EXPLORATORY',
        mechanism_status='POSITIVE_EXPLORATORY_SIGNAL' if any(r['MECHANISM_POSITIVE_SIGNAL'] for r in mechanism_rows) else
            'MECHANISM_ACTIVE_NO_POSITIVE_SIGNAL' if any(r['pair_numerically_valid'] and r['IC_distinct'] for r in mechanism_rows) else 'IC_MECHANISM_NOT_ACTIVATED',
        stop='20 configurations completed; no new grid, branch, perturbation or training authorized'))
    write_json(ROOT/'baseline_evaluation_checks.json',baseline_diffs)
    for item in manifest['protected_inputs']+manifest['frozen_parameters']:
        if file_record(Path(item['path']))!=item:raise ValueError('Historical artifact changed')
    manifest.update(status='EVALUATED_AWAITING_REPORT',evaluation_wall_time_s=time.perf_counter()-started,protected_inputs_unchanged=True,
        evaluation_code=file_record(Path(__file__)),GT_boundary='all parameters, solvers and numerical selection frozen before metrics; GT only evaluation/plots')
    write_json(ROOT/'manifest.json',manifest)


if __name__=='__main__':main()
