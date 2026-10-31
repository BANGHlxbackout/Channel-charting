"""P2A-only numerical audit, frozen shared evaluation and G3 decisions."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import file_record
from methods.g1_acf_graph_fcf.compare import compare_paths
from methods.stage2_modern_graph.ic_solver import ic_objective
from methods.stage2_modern_graph.run_ic_fulln import ROOT,P0,B0,SUBSET,grid,read
from run_adp_siamese import plot_chart

METRICS=['ct','tw','ks','mean_m','median_m','p90_m','p95_m','max_m']


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def name(config):return f"{config['method']} a={config['alpha']:.1f} eta={config['eta_factor']:g}d0"


def hit(candidate,reference,count,limit):
    improvements={key:(reference[key]-candidate[key])/reference[key] for key in ('ks','mean_m','p95_m')}
    return sum(v>=.01 for v in improvements.values())>=count and min(improvements.values())>=-limit,improvements


def g3(candidate,jacobi,cg):
    advisor,jimp=hit(candidate,jacobi,2,.01)
    sj,_=hit(candidate,jacobi,1,.02);sc,cimp=hit(candidate,cg,1,.02)
    tj,_=hit(candidate,jacobi,2,.01);tc,_=hit(candidate,cg,2,.01)
    survivor=sj and sc;target=tj and tc
    return dict(ADVISOR_HIT=advisor,SURVIVOR=survivor,TARGET_HIT=target,
        SOLVER_LEVEL_GAIN_ONLY=sj and not sc,
        label='TARGET_HIT' if target else 'SURVIVOR' if survivor else 'ADVISOR_HIT' if advisor else 'NO_GO',
        relative_improvement_vs_J= jimp,relative_improvement_vs_CG=cimp)


def audit(config,data,x,d,w):
    folder=ROOT/config['slug'];record=read(folder/'result.json');saved=np.load(folder/'embedding.npz')
    if not np.array_equal(saved['indices'],data['indices']) or not np.array_equal(saved['positions'],data['positions']):
        raise ValueError('node identity changed: '+config['slug'])
    state=np.load(folder/f"attempt{record['selected_attempt']}/state.npz");u,p=state['U'],state['P']
    cpu=[ic_objective(u,p,d,w,config['alpha'],config['kind'],True)] if config['group'] else [ic_objective(u[:,i:i+1],p[:,i:i+1],d,w,config['alpha'],config['kind']) for i in range(2)]
    objective_error=max(abs(a-b)/max(1.,abs(b)) for a,b in zip(cpu,record['objective']))
    history=list(csv.DictReader((folder/f"attempt{record['selected_attempt']}/history.csv").open()))
    tail=np.array([json.loads(row['objective']) for row in history[-10:]])
    increases=np.diff(tail,axis=0)>1e-7*np.maximum(1.,np.abs(tail[:-1]))
    sustained=bool(len(increases)>=4 and np.any(np.all(increases[-4:],axis=0)))
    checks=dict(indices_positions_identical=True,finite=bool(np.isfinite(saved['embedding']).all() and np.isfinite(u).all() and np.isfinite(p).all()),
        cpu_objective=cpu,cuda_cpu_objective_relative_error=objective_error,objective_no_sustained_worsening=not sustained,
        prox=max(record['stationarity']),feasibility=max(record['feasibility']),spread=record['spread_ratio'],
        auxiliary_nonzero_rows=int(np.count_nonzero(np.any(p!=0,axis=1))),
        auxiliary_max_abs=float(np.max(np.abs(p))),
        largest_identical_point_fraction=record['largest_identical_point_fraction'])
    valid=record['numerically_valid'] and objective_error<=1e-8 and not sustained and checks['finite']
    if not valid and record['status']=='VALID':record.update(status='NONCONVERGED',numerically_valid=False,evaluation_status='EXCLUDED_BY_G2')
    record['G2']=checks;write_json(folder/'result.json',record)
    return record


def render_group(paths,rows,folder,root_protocol):
    """Reuse saved common metrics; plotting always uses the baseline helper."""
    folder.mkdir(parents=True,exist_ok=True)
    names=list(paths);data=np.load(next(iter(paths.values())));pos=data['positions'];indices=data['indices']
    aligned=np.load(ROOT/'comparison/aligned_charts.npz')
    for suffix,align in [('raw',False),('affine',True)]:
        fig,axes=plt.subplots(1,len(names)+1,figsize=(4.5*(len(names)+1),4))
        for i,label in enumerate(names,1):
            values=rows[label];z=np.load(paths[label])['embedding']
            title=f"{label}\nCT={values['ct']:.3f}, TW={values['tw']:.3f}, KS={values['ks']:.3f}"
            plot_chart(None,pos,z,title,axes=(axes[0] if i==1 else None,axes[i]),align=align)
        fig.tight_layout();prefix='four_column' if len(names)+1==4 else f'{len(names)+1}_column'
        fig.savefig(folder/f'{prefix}_{suffix}_channel_chart.png',dpi=180);plt.close(fig)
    selected=[dict(method=label,**rows[label]) for label in names]
    write_csv(folder/'comparison.csv',selected)
    lines=['# P2A Channel Chart comparison','','Same fixed 4199-node metric subset; all-N affine errors; GT evaluation only.','',
        '| method | '+' | '.join(METRICS)+' | source_json | train_time_s | peak_vram_bytes |','|---|'+'---:|'*len(METRICS)+'---|---:|---:|']
    for label in names:
        values=rows[label];source=Path(values['source_json']).as_posix()
        time_text='unavailable' if values['train_time_s'] is None else str(values['train_time_s'])
        vram='unavailable' if values['peak_vram_bytes'] is None else str(values['peak_vram_bytes'])
        lines.append('| '+label+' | '+' | '.join(f'{values[k]:.6f}' for k in METRICS)+f' | [JSON]({source}) | {time_text} | {vram} |')
    (folder/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    np.save(folder/'metric_subset_seed10.npy',np.load(SUBSET))
    keys=root_protocol['aligned_keys'];np.savez_compressed(folder/'aligned_charts.npz',indices=indices,positions=pos,**{keys[k]:aligned[keys[k]] for k in names})
    write_json(folder/'eval_protocol.json',{**root_protocol,'embeddings':{k:str(v.resolve()) for k,v in paths.items()},
        'metric_subset_path':str((folder/'metric_subset_seed10.npy').resolve()),'aligned_keys':{k:keys[k] for k in names}})


def main():
    manifest=read(ROOT/'manifest.json')
    configs=sorted(grid(),key=lambda c:({'IC8-P':0,'IC9-P':1,'IC9-G':2}[c['method']],c['alpha'],c['eta_factor']))
    if not all((ROOT/c['slug']/'result.json').exists() for c in configs):raise ValueError('27-config numerical grid is incomplete; do not evaluate partial grid silently')
    for item in manifest['protected_inputs']:
        if file_record(Path(item['path']))!=item:raise ValueError('protected P0/source changed: '+item['path'])
    data=np.load(B0);x=np.load(P0/'chart_gauge.npz')['X'];d=sparse.load_npz(P0/'graph/incidence_D.npz')
    w=np.load(P0/'graph/edges_and_degrees.npz')['edge_weights'];w=w/w.mean()
    with threadpool_limits(1):records={c['slug']:audit(c,data,x,d,w) for c in configs}
    paths={label:ROOT/'references'/key/'embedding.npz' for label,key in [('B0','b0'),('B0+S2-J','s2_j'),('B0+S2-CG','s2_cg')]}
    paths.update({name(c):ROOT/c['slug']/'embedding.npz' for c in configs if records[c['slug']]['numerically_valid']})
    with threadpool_limits(1):metrics=compare_paths(paths,ROOT/'comparison',seed=10,subset_path=SUBSET)
    old_metrics={row['method']:row for row in csv.DictReader((P0/'comparison/comparison.csv').open(encoding='utf-8'))}
    baseline_checks={}
    for label in ('B0','B0+S2-J','B0+S2-CG'):
        differences={key:abs(metrics[label][key]-float(old_metrics[label][key])) for key in METRICS}
        if max(differences.values())>1e-10:raise ValueError('frozen baseline evaluation mismatch: '+label)
        baseline_checks[label]=differences
    write_json(ROOT/'baseline_evaluation_checks.json',baseline_checks)
    for label,path in paths.items():
        record=read(path.parent/'result.json');values=metrics[label]
        record.update(metrics={k:values[k] for k in ('ct','tw','ks')},localization_errors_after_affine={k:values[k] for k in METRICS[3:]},evaluation_status='EVALUATED_FROZEN_PROTOCOL')
        write_json(path.parent/'result.json',record)
        saved=np.load(path);title=f"{label}\nCT={values['ct']:.3f}, TW={values['tw']:.3f}, KS={values['ks']:.3f}"
        plot_chart(path.parent/'chart.png',saved['positions'],saved['embedding'],title)
    decisions={};scan=[];gates=[]
    for c in configs:
        record=read(ROOT/c['slug']/'result.json');label=name(c);valid=record['numerically_valid']
        decision=g3(metrics[label],metrics['B0+S2-J'],metrics['B0+S2-CG']) if valid else dict(label='EXCLUDED_NONCONVERGED',ADVISOR_HIT=False,SURVIVOR=False,TARGET_HIT=False,SOLVER_LEVEL_GAIN_ONLY=False)
        decision.update(config=c,numerically_valid=valid,source_json=str((ROOT/c['slug']/'result.json').resolve()))
        decisions[c['slug']]=decision;record['G3']=decision
        write_json(ROOT/c['slug']/'result.json',record)
        row=dict(method=label,**{k:metrics[label][k] if valid else None for k in METRICS},
            source_json=decision['source_json'],alpha=c['alpha'],eta_factor=c['eta_factor'],eta=c['eta'],
            phase='FULL_N',N=20997,status=record['status'],delta=record['delta'],iterations=record['iterations'],attempts=record['attempts'],
            prox=max(record['stationarity']),feasibility=max(record['feasibility']),G3=decision['label'],
            solve_time_s=record['solve_time_s'],total_runtime_s=record['total_runtime_s'],peak_ram_bytes=record['peak_ram_bytes'],peak_vram_bytes=record['peak_vram_bytes'])
        scan.append(row);gates.append(dict(config=c,valid=valid,status=record['status'],checks=record['G2'],source_json=decision['source_json']))
        if not valid:
            saved=np.load(ROOT/c['slug']/'embedding.npz')
            plot_chart(ROOT/c['slug']/'chart.png',saved['positions'],saved['embedding'],label+'\nNONCONVERGED — not evaluated')
    write_csv(ROOT/'parameter_scan.csv',scan);write_json(ROOT/'G2.json',gates);write_json(ROOT/'G3.json',decisions)
    pairs=[]
    for alpha in (.2,.5,.8):
        for factor in (.5,1.,1.5):
            a=next(c for c in configs if c['method']=='IC9-P' and c['alpha']==alpha and c['eta_factor']==factor)
            b=next(c for c in configs if c['method']=='IC9-G' and c['alpha']==alpha and c['eta_factor']==factor)
            ar,br=records[a['slug']],records[b['slug']];valid=ar['numerically_valid'] and br['numerically_valid']
            pair=dict(alpha=alpha,eta_factor=factor,eta=a['eta'],valid_pair=valid,delta_P=ar['delta'],delta_G=br['delta'],delta_G_minus_P=br['delta']-ar['delta'])
            for key in METRICS:
                pair[key+'_P']=metrics[name(a)][key] if valid else None;pair[key+'_G']=metrics[name(b)][key] if valid else None
            pair['source_json_P']=str((ROOT/a['slug']/'result.json').resolve());pair['source_json_G']=str((ROOT/b['slug']/'result.json').resolve());pairs.append(pair)
            group_paths={label:path for label,path in paths.items() if label in ('B0','B0+S2-J','B0+S2-CG') or any(label==name(c) for c in configs if c['alpha']==alpha and c['eta_factor']==factor)}
            render_group(group_paths,metrics,ROOT/'comparison'/f'a{int(alpha*100):02d}_eta{int(factor*100):03d}',read(ROOT/'comparison/eval_protocol.json'))
    write_csv(ROOT/'ic9_group_paired_comparison.csv',pairs)
    family={}
    for method in ('IC8-P','IC9-P','IC9-G','Robust-IC-all'):
        selected=[d for d in decisions.values() if method=='Robust-IC-all' or d['config']['method']==method]
        family[method]=dict(valid=sum(d['numerically_valid'] for d in selected),total=len(selected),survivors=sum(d['SURVIVOR'] for d in selected),targets=sum(d['TARGET_HIT'] for d in selected),
            label='G3_INCOMPLETE_NUMERICAL' if not all(d['numerically_valid'] for d in selected) else 'TARGET_HIT' if any(d['TARGET_HIT'] for d in selected) else 'SURVIVOR' if any(d['SURVIVOR'] for d in selected) else 'NO_GO_EXPLORATORY')
    write_json(ROOT/'family_decisions.json',family)
    manifest.update(status='P2A_COMPLETED' if all(r['numerically_valid'] for r in records.values()) else 'P2A_COMPLETED_WITH_NUMERICAL_FAILURES',
        valid_configurations=sum(r['numerically_valid'] for r in records.values()),G3=family,stop='P2A only; await user review; P2B/P3/P4/P5 not executed')
    write_json(ROOT/'manifest.json',manifest)
    report(manifest,scan,metrics,pairs)
    print(json.dumps(dict(status=manifest['status'],valid=manifest['valid_configurations'],G3=family),indent=2),flush=True)


def report(manifest,scan,metrics,pairs):
    def link(path,text=None):return f'[{text or path.name}]({path.resolve().as_posix()})'
    lines=['# P2A：B0 Robust-IC full-N（2026-10-20）','','## 结论与停止','',
        f"实际阶段FULL-N，N=20997；27个冻结配置全部执行，数值有效{manifest['valid_configurations']}个。P2A已停止；没有执行P2B/P3/P4/P5、Stage-1重训或Edge调参。Edge仍NUMERICAL_BLOCKER，IC/FN既有G1结论保留。",'',
        '| 方法 | 数值有效/总数 | SURVIVOR | TARGET_HIT | 家族判定 |','|---|---:|---:|---:|---|']
    for key,value in manifest['G3'].items():lines.append(f"| {key} | {value['valid']}/{value['total']} | {value['survivors']} | {value['targets']} | {value['label']} |")
    lines+=['','## 固定输入、目标、数值与协议','',
        '- 输入为已核验P0的B0、G128、delta0=0.09517167367990964、S2-J/CG；原图1544167无向边，只读复用。IC权重仅除以无向正边均值0.5592081927929078。没有变更图/相对权重或使用UE GT训练。',
        '- 固定alpha={0.2,0.5,0.8} × eta={0.5,1,1.5}delta0 × IC8-P/IC9-P/IC9-G，共27。标量模型x/y独立球与box；group二维共球无分轴box。原始IC8平方与IC9非平方D.T(WDU-P)保持不变。',
        '- 原P1合成入口不适合full-N及独立阶段；新增ic_cuda.py和run_ic_fulln.py/evaluate_ic_fulln.py。仅执行相同目标PDHG，float64稀疏CUDA。步长固定按观测球半径/beta平衡比例，乘积tau*sigma*norm_bound=.9025不变；每配置记录实际tau/sigma。没有基于GT/性能改动数值参数。',
        '- CUDA graph批处理50步；只有数学充分证书保证辅助P仍为0时才跳过其冗余计算。证书/box失效则恢复并按完整原迭代重放该chunk，不限制P=0。GPU与CPU同一步长逐迭代误差约1e-16；含active aux及独立凸reference测试均通过。',
        '- 初次30000步；仅同一目标一次冷启动重试120000步。prox<2e-6、feas停止<1e-9；G2硬门槛ball<=1e-4、finite/节点一致、无持续目标恶化、中心化RMS>=输入0.05、最大同点比例<0.1%。没有因达到上限标收敛。',
        '- CT/TW/KS使用P0同一固定4199节点；误差指标使用相同全部20997点，各方法独立仿射拟合/评价。直接复用common_metrics和run_adp_siamese.plot_chart。GT仅评价/绘图；全数据传导式结果，不代表held-out部署或边权扰动鲁棒性。',
        '- 只有数值有效配置进入性能评价；未收敛配置保留状态/日志/诊断图，指标留空。网格数值不完整时标G3_INCOMPLETE_NUMERICAL，不能解释成方法性能NO_GO。', '',
        f"数值与全参数扫描：{link(ROOT/'parameter_scan.csv')}；G2检查{link(ROOT/'G2.json')}；实际数值资源/每个attempt状态与完整轨迹见各配置目录。Sanity100/1000步检查单独标SANITY，不是正式完成结果。",'',
        '## 全部有效结果指标','', (ROOT/'comparison/comparison.md').read_text(encoding='utf-8'),'',
        '## G3 与配对解释','',
        'KS/Mean/P95均越小越好；改善率=(baseline-method)/baseline。ADVISOR_HIT要求对J至少两项>=1%、第三项不劣>1%；SURVIVOR要求分别对J和CG至少一项>=1%、其余不劣>2%；TARGET_HIT分别对两者至少两项>=1%、第三项不劣>1%。阈值用全精度，不按显示舍入判断。全部有效且没有SURVIVOR才NO_GO_EXPLORATORY；仅超过J的结果标SOLVER_LEVEL_GAIN_ONLY。',
        f"每配置布尔判据和对J/CG的实际改善率：{link(ROOT/'G3.json')}；IC9-G/P同alpha/eta、实际delta及全指标：{link(ROOT/'ic9_group_paired_comparison.csv')}。没有选择不同最优参数冒充几何适配消融。",'',
        '| alpha | eta/delta0 | 两者有效 | delta_P | delta_G | KS_P | KS_G | Mean_P | Mean_G | P95_P | P95_G |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        keys=['delta_P','delta_G','ks_P','ks_G','mean_m_P','mean_m_G','p95_m_P','p95_m_G']
        lines.append(f"| {p['alpha']} | {p['eta_factor']} | {p['valid_pair']} | "+' | '.join('—' if p[k] is None else f'{p[k]:.6f}' for k in keys)+' |')
    lines+=['','## 图谱、资源与复现','',
        'comparison/根目录保留全部有效配置大表/原始及仿射图；每组固定alpha/eta的子目录另有清晰的GT / B0 / S2-J / S2-CG / IC8-P / IC9-P / IC9-G对比（数值无效者不混入有效比较）。各配置保留标准chart.png、embedding.npz、normalized_output.npz、result.json及每次attempt状态/轨迹。','']
    for alpha in (.2,.5,.8):
        for factor in (.5,1.,1.5):
            f=ROOT/'comparison'/f'a{int(alpha*100):02d}_eta{int(factor*100):03d}'
            raw=next(f.glob('*_raw_channel_chart.png'));affine=next(f.glob('*_affine_channel_chart.png'))
            lines.append(f"- alpha={alpha}, eta={factor}delta0：{link(f/'comparison.md','表')} / {link(f/'comparison.csv','CSV')} / {link(raw,'raw')} / {link(affine,'affine')}。")
    lines+=['','| 配置 | 状态 | attempt | iterations | prox max | delta | 求解loop秒 | attempt墙钟秒 | RAM峰值bytes | VRAM峰值bytes |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in scan:lines.append(f"| {r['method']} | {r['status']} | {r['attempts']} | {r['iterations']} | {r['prox']:.6e} | {r['delta']:.6f} | {r['solve_time_s']:.3f} | {r['total_runtime_s']:.3f} | {r['peak_ram_bytes']} | {r['peak_vram_bytes']} |")
    lines+=['',
        '以上是后处理求解耗时，非训练耗时。solver loop包含数值日志/状态传回；attempt墙钟包含CUDA graph准备，不含最终压缩写盘；RAM为求解期间100ms采样working set，VRAM为PyTorch allocator峰值（含常驻图），非整机其它进程显存。完整网格墙钟包含写盘，见manifest.run_wall_time_s。baseline历史资源未知，不填造值。',
        f"代码与复现：{link(Path('methods/stage2_modern_graph/run_ic_fulln.py'))} / {link(Path('methods/stage2_modern_graph/ic_cuda.py'))} / {link(Path('methods/stage2_modern_graph/evaluate_ic_fulln.py'))}；环境、输入hash、冻结配置与命令见{link(ROOT/'manifest.json')}。未修改AGENTS、原baseline、P0输入/缓存、P1/FN/Edge；当前分支{manifest['git_branch']}、HEAD {manifest['git_head']}，未提交/上传。",'',
        '```powershell','D:/anaconda3/envs/diffusion_charting/python.exe -m unittest tests.test_stage2_ic_cuda -v',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.run_ic_fulln --phase prepare',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.run_ic_fulln --phase run',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.evaluate_ic_fulln','```','',
        '已完成配置按config与source记录恢复，不重跑；独立复现应使用空输出目录/副本。若存在未完成attempt目录，入口主动报错，避免擅自追加数值重试。', '',
        '**STOP：仅交付P2A真实B0结果，等待用户审核，不自动进入其它阶段。**']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
