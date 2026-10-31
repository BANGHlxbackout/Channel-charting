"""Audit and summarize saved P1 outputs; no real-data method experiments."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import file_record, incidence_and_laplacian
from diagnostics.stage2_p1.reference import edge_u_reference
from diagnostics.stage2_p1.cases import synthetic_case
from diagnostics.stage2_p1.run import ROOT, save_reference, evaluate_group


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def link(path, name=None):
    return f'[{name or path.name}]({path.resolve().as_posix()})'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(manifest, gates, rows, controls):
    report = ['# Stage-2 P1 数学 sanity（2026-10-19）', '',
        '**范围：P1 合成小图；已停止，未执行 P2A。** P0 已获用户确认，原实验事实不变。本轮不重新训练 G1，不拟合真实 B0/G1，不做真实数据性能结论。', '',
        '## 1. 结果与停止边界', '',
        '| 家族 | 状态 | 解释 |', '|---|---|---|']
    for family, status in manifest['families'].items():
        report.append(f'| {family} | {status} | '+('参考目标、可行性及数值 gate 均通过' if status=='PASS' else '存在未通过的约束/求解器完成 gate，原始失败结果保留，禁止进入该分支 full-N')+' |')
    report += ['', f"总计 {len(gates)} 项检查，{manifest['gate_fail_count']} 项 FAIL。完整逐项实际值、门槛及证据：{link(ROOT/'gates.md')} / {link(ROOT/'gates.csv')}。",
        'SEF 的数学参考门槛与求解器自身停止状态分开报告；即使前者通过，也不将预算用尽的 NONCONVERGED 标为完整收敛成功。联合 FE/DoT 没有全局凸 reference，只检查固定 U 的 S/E 块和固定 S 的 U 块。', '',
        '## 2. 固定对象、公式与实现', '',
        '- 图：固定 RNG=20261019+N 的加权 path+two-hop，IC/FN N=16/32/64，Edge N=256。合成干净弧线仅离线评价和绘图；求解器只读加噪观测 X 和图。未从真实 UE 坐标生成输入。',
        '- IC：无向正边均值归一化 W；alpha=0.5，beta=0.5，eta=P0 delta0=0.09517167367990964。IC8 保留平方项；IC9 保留 D.T(WDU-P) 的非平方 L2/Frobenius 范数。标量有 box，group 仅共用观测球。',
        '- Framelet：nu=0.1 仅数学 smoke。active=[low2,high1,high2]；高通尺度权重1和0.25，低通惩罚0但 Q_low=T_low U 约束存在。保存 objective 为 per-degree/per-band 定义。每次 U-step PCG 解 D_G+rho*sum(T.T T)，未用 D_G+rho*I 近似。',
        '- FN-G-delta 最多8次新求解调用（缓存 same-nu 不重算），按实际移动量无标签匹配。匹配是有限预算近似，误差见下表；没有据 GT 选 nu。',
        '- SEF/FE/DoT 保留 signed S、diag=0；FE/DoT 保留 S1=1、lambda1=lambda2=1，DoT 只额外加 Framelet 项。P1 使用 nu=0.1 的合成数学 smoke，不能冒充尚未授权的 B0 nu_P(mid) 或 P3 512/1024 gate。',
        '- PDHG 最大30000次，唯一数值重试120000次；FN ADMM 最大5000次，唯一重试10000次；Edge 最大5000次，唯一重试15000次。alpha/eta/nu 与数据规模不因结果改变。',
        '- 独立参考：CVXPY/CLARABEL，gap/feas tolerance=1e-9，status 必须 optimal；原始参考数组、objective、solver状态和迭代数均保存。IC论文附件本地未找到，因此本轮核验的是任务冻结公式，不声称逐行复现其 Algorithm1。',
        f"- 代码：{link(Path('methods/stage2_modern_graph/ic_solver.py'))}、{link(Path('methods/stage2_modern_graph/framelet_operator.py'))}、{link(Path('methods/stage2_modern_graph/framelet_solver.py'))}、{link(Path('methods/stage2_modern_graph/edge_small.py'))}；{link(Path('diagnostics/stage2_p1/reference.py'))} 是独立 CVXPY 目标。", '',
        '作者滤波器代码固定到 commit 4db39904e4099d82cef686d4732c33150814ece4；仅临时将已移除的 np.trapz 指向等价 np.trapezoid 以执行比较，未修改作者源文件。算子与作者返回的三个实际 active bands 做数值一致性比较。来源 URL/hash 见 manifest。', '',
        '## 3. Reference objective、可行性与最优条件', '',
        f"全精度明细：{link(ROOT/'reference_objectives.csv')}；IC-P 两个坐标是两个独立标量目标，不与 group objective 直接相加排名。", '',
        '| N | 方法 / 坐标 | native objective | reference objective | relative gap | feasibility | stationarity |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['family']=='edge':
            continue
        report.append(f"| {row['N']} | {row['method']} / {row['coordinate']} | {row['objective']:.9g} | {row['reference_objective']:.9g} | {row['relative_gap']:.3e} | {row['feasibility']:.3e} | {row['stationarity']:.3e} |")
    report += ['', '## 4. Framelet 近似、零模态与移动量匹配', '',
        '| N | 最小 n | reconstruction error | highpass true-zero max | matched delta relative error |',
        '|---:|---:|---:|---:|---:|']
    for n in (16,32,64):
        op=read(ROOT/'framelet'/f'n{n}'/'operator_attempts.json')[-1]
        match=read(ROOT/'framelet'/f'n{n}'/'delta_matching.json')
        report.append(f"| {n} | {op['n']} | {op['reconstruction_relative_error']:.6e} | {max(op['highpass_zero_mode_responses']):.6e} | {match['relative_delta_error']:.6e} |")
    report += ['', 'n=2 失败是滤波近似精度不足；N=64 的 n=3 也未过 reconstruction 门槛，最小通过阶为4。零模态使用 sqrt(degree)*1；普通1响应仅作辅助，未参与 gate。PCG 每步真实残差及与稠密同目标 U-step 的误差保存在 checks.json/history.csv。IC9-G、FN-G-nu/FN-G-delta 的四种 gauge 等变性均单独保存。', '',
        '## 5. Edge/DoT 数值状态及 BLOCKER', '',
        '| 方法/初始化 | iterations | converged | equality | row sum | stationarity | S-block ref / gap | U-block ref / gap |',
        '|---|---:|---|---:|---:|---:|---|---|']
    for path in sorted((ROOT/'edge256').glob('*/checks.json')):
        c=read(path)
        u='N/A（固定U）' if c['Ublock_relative_gap'] is None else f"{c['fixed_S_Ublock_reference_objective']:.9g} / {c['Ublock_relative_gap']:.3e}"
        report.append(f"| {path.parent.name} | {c['iterations']} | {c['converged']} | {c['equality_residual']:.6e} | {c['row_sum_residual']:.6e} | {c['stationarity']:.6e} | {c['fixed_U_Sblock_reference_objective']:.9g} / {c['Sblock_relative_gap']:.3e} | {u} |")
    report += ['', '约束门槛1e-3；native objective 和将 E 消去后的 objective 分列保存。后者只使自表达等式精确，**不自动使原 S 的行和精确**，不能称全约束可行值。固定 U 的凸参考解强制 diag/row-sum，固定 S 的凸参考继承该 S 的行和误差；目标接近不能消除最终约束 FAIL。',
        f"两个初始化的输出差异见 {link(ROOT/'edge256/initialization_sensitivity.json')}，冻结S0/学习S的数值对照见 {link(ROOT/'edge256/fixed_vs_learned_S.json')}。这只是实现诊断，不构成学到无线物理边或性能提升的证据。SEF 固定U，不进图谱性能表。",
        '诊断：在固定5000+15000迭代预算下约束收敛慢；没有 OOM/NaN、PCG失败或参考解失败证据。两次预算和全部轨迹保留，未追加第三次、修改目标、放宽阈值或缩小数据。当前失败只说明该求解实现/数值配置未通过，不否定论文数学方法。末尾1000步的残差波动范围见 edge256/convergence_diagnostics.json。下降的比较基准统一为可行的先验 S0、U=X、E=X-S0X，并非把不满足行和的作者零初始化当成可行初值。', '',
        '## 6. 日志、Chart 和完整指标', '',
        '以下均为 **P1 synthetic math**，非真实天线阵列性能。字段保留米制命名以复用规范，但合成位置单位是人为设定的坐标单位，不可解读成真实定位米数。CT/TW/KS 用 seed10 的20%节点，N=16时仅3个指标节点，仅保留工程一致性，不作质量结论。误差在同一全部节点仿射拟合/评价。', '']
    for folder in [ROOT/'ic'/f'n{n}' for n in (16,32,64)]+[ROOT/'framelet'/f'n{n}' for n in (16,32,64)]+[ROOT/'edge256']:
        comp=folder/'comparison'
        raw=next(comp.glob('*_raw_channel_chart.png'))
        affine=next(comp.glob('*_affine_channel_chart.png'))
        report.append(f"- {folder.relative_to(ROOT).as_posix()}：{link(comp/'comparison.md','表')} / {link(comp/'comparison.csv','CSV')} / {link(raw,'raw图')} / {link(affine,'affine图')}。")
    report += ['', '图谱直接复用 run_adp_siamese.plot_chart 和原 compare_paths；GT首列，同一样本与颜色，未另写散点格式。Edge 未收敛图明确标记 NONCONVERGED，仅作失败诊断，不进行有效方法排名。',
        f"收敛轨迹：各方法目录 attempt*_history.csv（IC为 axis*_attempt*_history.csv）；控制台 {link(ROOT/'ic_console.log')}、{link(ROOT/'framelet_console.log')}、{link(ROOT/'edge_console.log')}。完整JSON/CSV一致性核对：{link(ROOT/'artifact_consistency.json')}。", '',
        '## 7. 资源、测试、复现与保护', '',
        f"正式调用实际墙钟时间（包含参考求解与出图）依次为 {manifest['run_times_s']} 秒；不能作为单独算法训练耗时。CPU float64，本轮未使用GPU，GPU显存分配0；全流程最终CPU峰值RAM未完整记录，标未知。Edge运行中观察到peak working set 868552704 bytes（仅截至采样时刻的下界），见 {link(ROOT/'resource_observation.json')}。无模型训练/checkpoint。",
        f"5项相关单元测试通过：{link(ROOT/'unit_tests.log')}。运行环境、输入/代码/作者文件hash与命令：{link(ROOT/'manifest.json')}。AGENTS及P0 delta0/manifest前后hash一致；原baseline源与历史结果未写入。", '',
        '复现（仓库根目录，使用空的本轮输出目录/独立副本；入口拒绝覆盖已存在家族结果）：', '',
        '```powershell', 'D:/anaconda3/envs/diffusion_charting/python.exe -m unittest tests.test_stage2_p1_math -v',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.run --family ic',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.run --family framelet',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.run --family edge',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.report', '```', '',
        f"当前分支 {manifest['git_branch']}，HEAD {manifest['git_head']}，本轮未提交/上传。{link(ROOT/'git_diff_stat.txt')} 是全工作区既有差异，包含本轮之前的删除/修改；本轮新增未跟踪代码不会显示在该stat中，以 manifest.final_code/test_source 为准，不把既有改动记为本轮工作。", '',
        '**阶段状态：P0 PASS（用户已确认）；P1 已执行并交付，Edge 存在 BLOCKER；P2A/P2B/P3/P4/P5 均 STOP（未授权/未执行）。等待用户审查，本轮不继续实验。**']
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')


def main():
    manifest = read(ROOT / 'manifest.json')
    gates = [g for g in read(ROOT / 'gates.json')
             if not g['gate'].endswith(('_finite_objective_decrease', '_solver_completion'))]
    rows = []
    descent = []
    for family in ('ic', 'framelet'):
        for path in sorted((ROOT / family).glob('n*/*/checks.json')):
            check = read(path)
            n = int(path.parent.parent.name[1:])
            cases = check.get('coordinate_results', [check])
            for coordinate, case in enumerate(cases):
                gap = case.get('relative_objective_gap', check.get('max_objective_gap'))
                rows.append(dict(family=family, N=n, method=path.parent.name, coordinate=coordinate,
                    objective=case['objective'], reference_objective=case['reference_objective'],
                    relative_gap=gap, feasibility=case.get('feasibility', case.get('primal_residual')),
                    stationarity=case['stationarity'], evidence=str(path.resolve())))
                initial = case.get('initial_optimal_aux_objective', case.get('initial_objective'))
                descent.append(dict(family=family, case=f'{n}_{path.parent.name}_axis{coordinate}',
                    initial=initial, final=case['objective'], decrease=initial-case['objective'],
                    status='PASS' if np.isfinite(case['objective']) and initial > case['objective'] else 'FAIL',
                    evidence=str(path.resolve())))
    edge = ROOT / 'edge256'
    controls = {}
    case = np.load(edge / 'case.npz')
    x, adjacency = case['X'], case['adjacency']
    degree = adjacency.sum(axis=1)
    for path in sorted(edge.glob('*/checks.json')):
        check = read(path)
        arrays = np.load(sorted(path.parent.glob('attempt*_arrays.npz'))[-1])
        structure_objective = float(np.abs(arrays['S']).sum() + np.sum(degree * np.linalg.norm(arrays['U'] - arrays['S'] @ arrays['U'], axis=1)))
        if path.parent.name.startswith('sef'):
            gates.append(dict(family='Edge', gate='SEF-P-small-author_solver_completion',
                actual={'converged':check['converged'], 'iterations':check['iterations'], 'stationarity':check['stationarity']},
                threshold='native stopping criterion met within frozen initial + one retry budgets',
                status='PASS' if check['converged'] else 'FAIL', evidence=str(path.resolve())))
        for block, key in [('S', 'fixed_U_Sblock_reference_objective'), ('U', 'fixed_S_Ublock_reference_objective')]:
            if check[key] is not None:
                rows.append(dict(family='edge', N=256, method=path.parent.name, coordinate=block,
                    objective=check['feasible_objective'] if block == 'U' else structure_objective,
                    reference_objective=check[key], relative_gap=check[block+'block_relative_gap'],
                    feasibility=check['equality_residual'], stationarity=check['stationarity'], evidence=str(path.resolve())))
        descent.append(dict(family='edge', case=path.parent.name, initial=check['feasible_initial_objective'],
            final=check['feasible_objective'], decrease=check['finite_objective_decrease'],
            status='PASS' if check['finite_objective_decrease'] > 0 and np.isfinite(check['feasible_objective']) else 'FAIL',
            evidence=str(path.resolve())))
    # Independent fixed-S0 reference, required to detect a vacuous learned-S update.
    prior = adjacency / degree[:, None]
    for name in ('fe-p-small', 'dot-p-small'):
        matrices = None
        if name.startswith('dot'):
            from methods.stage2_modern_graph.framelet_operator import FrameletOperator
            spec = read(edge / 'operator_attempts.json')[-1]
            matrices = FrameletOperator(sparse.csr_matrix(adjacency), spec['n'], spec['lambda_max']).reference_matrices()
        with threadpool_limits(1):
            reference = edge_u_reference(x, prior, degree, matrices, .1 if matrices is not None else 0.)
        save_reference(edge, reference, name+'_fixed_S0_reference')
        values = {}
        for init in ('author', 'prior'):
            folder = edge / (name+'-'+init)
            attempts = sorted(folder.glob('attempt*_arrays.npz'))
            result = np.load(attempts[-1])
            values[init] = dict(output_relative_difference=float(np.linalg.norm(result['U']-reference['U']) / np.linalg.norm(x)),
                                structure_frobenius_difference=float(np.linalg.norm(result['S']-prior)))
        controls[name] = dict(reference_objective=reference['reference_objective'], comparisons=values)
    write_json(edge / 'fixed_vs_learned_S.json', controls)
    tails=[]
    for path in sorted(edge.glob('*/attempt*_history.csv')):
        history=list(csv.DictReader(path.open(encoding='utf-8')))
        tail=history[-100:]
        tails.append(dict(path=str(path.resolve()),last_iteration=int(tail[-1]['iteration']),
            finite=all(np.isfinite(float(value)) for row in history for value in row.values()),
            final_window={key:dict(min=min(float(row[key]) for row in tail),max=max(float(row[key]) for row in tail))
                          for key in ('objective','equality_residual','row_sum_residual','stationarity')}))
    write_json(edge/'convergence_diagnostics.json',tails)
    old_protocol = read(edge / 'comparison/eval_protocol.json')
    paths = {}
    for row in csv.DictReader((edge / 'comparison/comparison.csv').open(encoding='utf-8')):
        name = row['method'].removesuffix(' [NONCONVERGED]')
        path = Path(old_protocol['embeddings'][row['method']])
        record = read(path.parent / 'result.json')
        paths[name + (' [NONCONVERGED]' if not record.get('converged', True) else '')] = path
    evaluate_group(paths, edge)
    for item in descent:
        gates.append(dict(family={'ic':'IC','framelet':'Framelet','edge':'Edge'}[item['family']],
            gate=item['case']+'_finite_objective_decrease', actual=item,
            threshold='finite objective; final < feasible initial for nonzero update', status=item['status'], evidence=item['evidence']))
    write_json(ROOT / 'objective_decrease.json', descent)
    write_json(ROOT / 'gates.json', gates)
    write_csv(ROOT / 'reference_objectives.csv', rows)
    write_csv(ROOT / 'gates.csv', [{**g, 'actual': json.dumps(g['actual'], ensure_ascii=False)} for g in gates])
    lines = ['# P1 Gate actual values', '', '| family | gate | actual | threshold | status | evidence |',
             '|---|---|---|---|---|---|']
    for gate in gates:
        actual = json.dumps(gate['actual'], ensure_ascii=False).replace('|','/')
        lines.append(f"| {gate['family']} | {gate['gate']} | {actual} | {gate['threshold']} | {gate['status']} | {link(Path(gate['evidence']))} |")
    (ROOT / 'gates.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Ensure every comparison agrees with the actual source JSON and saved arrays.
    checks = []
    for path in ROOT.rglob('comparison.csv'):
        protocol = read(path.parent / 'eval_protocol.json')
        sample = np.load(protocol['metric_subset_path'])
        for row in csv.DictReader(path.open(encoding='utf-8')):
            source = read(Path(row['source_json']))
            values = {**source['metrics'], **source['localization_errors_after_affine']}
            for key in ('ct','tw','ks','mean_m','median_m','p90_m','p95_m','max_m'):
                if float(row[key]) != values[key]:
                    raise ValueError(f'CSV/source disagreement: {path}, {key}')
            saved = np.load(protocol['embeddings'][row['method']])
            if not np.isfinite(saved['embedding']).all() or len(saved['indices']) != protocol['num_samples']:
                raise ValueError('invalid chart')
        checks.append(dict(path=str(path.resolve()), N=protocol['num_samples'], metric_subset_size=len(sample), status='PASS'))
    write_json(ROOT / 'artifact_consistency.json', checks)
    graph_audit = []
    for n in (16,32,64,256):
        adjacency, x, clean, observation = synthetic_case(n)
        d, weights, *_ = incidence_and_laplacian(adjacency)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        if (adjacency-adjacency.T).nnz or np.any(adjacency.diagonal()) or np.any(degree<=0):
            raise ValueError('synthetic graph consistency failed')
        if n<256:
            saved=np.load(ROOT/'ic'/f'n{n}'/'case.npz')
            np.testing.assert_array_equal(saved['incidence'],d.toarray())
            np.testing.assert_array_equal(saved['weights_ic'],weights/weights.mean())
            np.testing.assert_array_equal(saved['X'],x)
        hashes={name:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                for name,value in dict(csr_indptr=adjacency.indptr,csr_indices=adjacency.indices,csr_data=adjacency.data,
                                       X=x,positions=clean,observation=observation).items()}
        graph_audit.append(dict(N=n,symmetric=True,self_loops=0,positive_degree_min=float(degree.min()),
            undirected_edges=len(weights),original_weight_mean=float(weights.mean()),ic_weight_mean=float((weights/weights.mean()).mean()),
            hashes=hashes,seed=20261019+n))
    write_json(ROOT/'synthetic_graph_audit.json',graph_audit)
    manifest['families'] = {family: 'FAIL' if any(g['family']==family and g['status']=='FAIL' for g in gates) else 'PASS'
                             for family in ('IC','Framelet','Edge')}
    manifest['status'] = 'P1_COMPLETED_PASS' if all(v=='PASS' for v in manifest['families'].values()) else 'P1_COMPLETED_WITH_BLOCKER'
    manifest['gate_count'] = len(gates)
    manifest['gate_fail_count'] = sum(g['status']=='FAIL' for g in gates)
    manifest['final_code'] = [file_record(p) for folder in ('methods/stage2_modern_graph','diagnostics/stage2_p1')
                              for p in Path(folder).glob('*.py')]
    final_by_path={record['path']:record for record in manifest['final_code']}
    for record in manifest['code']:
        if 'methods\\stage2_modern_graph' in record['path'] and final_by_path[record['path']] != record:
            raise ValueError('solver changed during formal P1 execution')
    manifest['solver_code_unchanged_during_run']=True
    manifest['reporting_code_change']='run.py only changed completed vs NONCONVERGED artifact status; report.py added after math runs began. Solver/cases/reference sources unchanged.'
    manifest['test_source'] = file_record(Path('tests/test_stage2_p1_math.py'))
    manifest['phase_status'] = {'P0':'PASS accepted by user', 'P1':manifest['status'], 'P2A':'STOP not authorized',
                                'P2B':'STOP not authorized', 'P3':'STOP not authorized', 'P4':'STOP not authorized', 'P5':'STOP not authorized'}
    manifest['git_head'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    write_json(ROOT / 'manifest.json', manifest)
    (ROOT / 'git_diff_stat.txt').write_text(subprocess.check_output(['git','diff','--stat'],text=True),encoding='utf-8')
    write_report(manifest, gates, rows, controls)
    print(json.dumps(dict(families=manifest['families'], gates=len(gates), failures=manifest['gate_fail_count'],
                         comparisons=checks, fixed_controls=controls),indent=2))


if __name__ == '__main__':
    main()
