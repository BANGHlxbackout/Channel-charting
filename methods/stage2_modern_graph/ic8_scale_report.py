"""Generate the single P2A-R report from saved numerical and evaluation evidence."""
import argparse
import csv
from pathlib import Path
import numpy as np

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import file_record
from methods.stage2_modern_graph.ic8_scale_experiment import ROOT,P0,grid,read
from methods.stage2_modern_graph.evaluate_ic_fulln import METRICS


def link(p,label=None):return f'[{label or Path(p).name}]({Path(p).resolve().as_posix()})'


def table(rows,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'---|'*len(columns)]
    for row in rows:
        values=[]
        for k in columns:
            v=row[k]
            if k=='source_json':v=link(v,'JSON')
            elif v is None or v=='':v='unavailable'
            elif isinstance(v,(float,np.floating)):v=f'{v:.6e}' if k in ('prox','feas') else f'{v:.6f}'
            values.append(str(v))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def main():
    manifest=read(ROOT/'manifest.json')
    if manifest['status'] not in ('EVALUATED_AWAITING_REPORT','COMPLETED_STOP'):raise ValueError('Evaluation incomplete')
    configs=grid();records={c['slug']:read(ROOT/c['slug']/'result.json') for c in configs}
    scan=list(csv.DictReader((ROOT/'comparison/comparison.csv').open(encoding='utf-8')))
    for row in scan:
        for k in METRICS+['alpha','kappa','eta','P_nonzero_fraction','movement_x','movement_y','solve_time_s','peak_ram_bytes','peak_vram_bytes']:
            row[k]=float(row[k]) if row[k] else None
    mechanism=list(csv.DictReader((ROOT/'mechanism_summary.csv').open(encoding='utf-8')))
    for row in mechanism:
        for k in ('P_nonzero_fraction','relative_raw_RMS_difference_B0','KS_improvement_vs_Ball','Mean_improvement_vs_Ball','P95_improvement_vs_Ball'):
            row[k]=float(row[k]) if row[k] else None
    g3=read(ROOT/'G3.json');g2=read(ROOT/'G2.json');sanity=read(ROOT/'sanity.json')
    diagnostic=read(ROOT/'solver_budget_diagnosis.json') if (ROOT/'solver_budget_diagnosis.json').exists() else None
    review=read(ROOT/'visual_review.json') if (ROOT/'visual_review.json').exists() else None
    active=[r for r in mechanism if r['P_active']=='True'];distinct=[r for r in mechanism if r['IC_distinct']=='True'];positive=[r for r in mechanism if r['MECHANISM_POSITIVE_SIGNAL']=='True']
    ic_rows=[r for r in scan if r['method'].startswith('IC8') and r['mean_m'] is not None]
    best=min(ic_rows,key=lambda r:r['mean_m']);bestrecord=read(Path(best['source_json']));bestmech=bestrecord['mechanism']
    bestmatched=next(r for r in scan if r['method']=='Ball-GLR' and r['eta']==best['eta'])
    low=min((r['P_nonzero_fraction'] for r in active),default=0.);high=max((r['P_nonzero_fraction'] for r in active),default=0.)
    zero=sum(r['regime']=='QUADRATIC_REGIME' for r in mechanism)
    maxprox=max(max(r['stationarity']) for r in records.values());maxfeas=max(max(r['feasibility']) for r in records.values())
    minspread=min(r['spread_ratio'] for r in records.values());maxdupes=max(r['largest_identical_point_fraction'] for r in records.values())
    total_time=sum(r['solve_time_s'] for r in records.values());peakvram=max(r['peak_vram_bytes'] for r in records.values());peakram=max(r['peak_ram_bytes'] for r in records.values())
    qs=manifest['parameters'] if 'parameters' in manifest else read(ROOT/'derived_alpha_values.json')['values']
    lines=['# P2A-R：B0 Robust-IC 天线数据尺度适配','','## 状态与问题','',
        f"**正式 full-N {'已完成并STOP，等待用户审核' if review else '数值与评价完成，交付核验待完成'}**：N=20997，20/20配置执行，G2数值有效{g2['valid']}/20。小图{len(sanity['cases'])}项{sanity['status']}。数值执行日期2026-10-20；中断恢复后的评价与交付日期2026-10-25。",
        '本轮区分异常边分解相对同eta Ball-GLR的增量，与相对S2-J/S2-CG的最终任务收益。全部配置预先冻结；本报告中的最小Mean仅用于完整网格的描述，不构成独立验证集选优。',
        '',f"**结论**：P-active {len(active)}/15、IC-distinct {len(distinct)}/15；{len(positive)}个同eta机制正向信号均在eta=.03。IC8相对S2-J/CG的ADVISOR_HIT/SURVIVOR/TARGET_HIT全部为0，任务层面为NO_GO_EXPLORATORY。eta=.03的Mean/P95改善伴随Max显著恶化，不能概括为全面质量提升。",
        '', '## 冻结目标与实现','',
        '`IC8-P: alpha*|P|_1 + (1-alpha)*|sqrt(W_IC)DU-P|²`；x/y独立观测球与原宽box。Ball-GLR使用同入口`aux_mode=disabled`，P固定为0、beta=1，目标严格为`|sqrt(W_IC)DU|²`。没有新增fidelity，也未用Huber近似替换求解目标。',
        'B0、P0缓存D/G128、节点与相对权重不变；W_IC=W/mean(W_e)。eta固定为0.03/0.08/0.16/0.32/0.50，完全不由delta0生成。delta0只保留为历史对照。',
        '运行中用户补充允许灵活调整eta，原方案为初步设计；此前禁止修改eta的限制不再作为后续硬边界。本批已在多个eta找到P-active，故没有为追求激活追加点，完成原20配置的匹配Ball比较。若未来调整必须记录时点和依据，不能把后验GT选优冒充预注册验证。',
        'alpha由池化两坐标输入边梯度的q95/q99/q995计算；5%/1%/0.5%指坐标条目率，边union率另存。参数在本轮GT指标前冻结。',
        '',table(qs,['alpha_source','quantile','kappa','alpha']),
        '', '使用原PDHG的IC8更新及投影/图存储；新入口隔离旧27网格。最初使用30000步/至多一次同目标120000步冷启动。用户随后授权调整步数，诊断后仅剩余3个eta=.50 IC8改为单次60000步，无额外重试；已完成17配置不重跑，原核函数/步长/容差不变。当前未完成的q95日志归档为人工预算调整中断，无有效embedding；其已记录耗时计入资源。float64，prox<2e-6、feas<1e-9才停止；不根据GT或迭代上限宣布收敛。',
        'relative raw RMS的主分母是中心化B0 RMS；另存绝对raw RMS、Ball中心化RMS分母、全局与分轴移动余弦。绝对raw差异是原图谱坐标单位，不能当作米。P-active要求非零边union比例>=1e-4，IC-distinct再要求relative raw RMS>=1e-4。',
        '', '## 数学与数值验收','',
        f"小图16/32/64节点覆盖两端eta及3个alpha和Ball；最大凸参考相对目标差{max(r.get('relative_objective_gap',0.) for r in sanity['cases']):.6e}，与旧IC8的200步U/P/dual误差均为0。一次CLARABEL optimal_inaccurate改用同显式目标的固定SCS参考通过；原失败日志保留。",
        f"20配置最大prox={maxprox:.6e}，最大feas={maxfeas:.6e}；最小中心化spread/input={minspread:.6f}，最大同点比例={maxdupes:.6e}。详见{link(ROOT/'G2.json')}。",
        '',table([dict(config=c['slug'],status=records[c['slug']]['status'],iterations=records[c['slug']]['iterations'],attempts=records[c['slug']]['attempts'],
            prox=max(records[c['slug']]['stationarity']),feas=max(records[c['slug']]['feasibility']),time_s=records[c['slug']]['solve_time_s'],
            G3=records[c['slug']]['G3']['label']) for c in configs],['config','status','iterations','attempts','prox','feas','time_s','G3']),
        '', '## 全部评价指标','',
        'CT/TW/KS复用P0 seed10的4199节点（20%，CT/TW邻居209）；误差在相同20997节点上分别做全数据仿射拟合/评价，单位米。GT仅用于此处评价与绘图；不是held-out或部署泛化。三基线与P0所有8指标复算差均<=1e-10。',
        '',table(scan,['method','alpha_source','eta']+METRICS+['source_json']),
        '', '## 方法机制收益：与匹配eta的Ball-GLR','',
        f'15个IC配置中P-active={len(active)}，IC-distinct={len(distinct)}，精确P=0的QUADRATIC_REGIME={zero}；预注册正向信号={len(positive)}。正向信号需数值有效、P-active、IC-distinct，并且KS/Mean/P95至少一项改善>=1%，其它不恶化>2%。表中收益为比例，正值表示改善。',
        '',table(mechanism,['config','P_nonzero_fraction','relative_raw_RMS_difference_B0','KS_improvement_vs_Ball','Mean_improvement_vs_Ball','P95_improvement_vs_Ball','MECHANISM_POSITIVE_SIGNAL']),
        '', '## 最终任务收益：与S2-J/S2-CG','',
        f"IC8的ADVISOR_HIT={g3['IC8_advisor_count']}/15，SURVIVOR={g3['IC8_survivor_count']}/15，TARGET_HIT={g3['IC8_target_count']}/15。ADVISOR_HIT/SURVIVOR/TARGET_HIT继续使用原G3，不把数值成功当性能成功。",
        f"描述性最小Mean配置：{best['method']} eta={best['eta']:g}，Mean={best['mean_m']:.6f}、P95={best['p95_m']:.6f}、KS={best['ks']:.6f}；匹配Ball的Mean={bestmatched['mean_m']:.6f}、P95={bestmatched['p95_m']:.6f}。该配置的G3={bestrecord['G3']['label']}，P-active={bestmech['P_active']}，IC-distinct={bestmech['IC_distinct']}。",
        '', '## 必须回答的研究问题','',
        f'1. **是否覆盖遗漏激活区间？** 输入尺度上已覆盖：alpha约0.244131/0.311103/0.344191，处于旧0.2与0.5之间；不等于最终P必然激活。最终P-active共{len(active)}/15，精确零P为{zero}/15。',
        f'2. **哪些配置激活、比例是否合理？** P-active：q95的全部5个eta；q99的.03/.08/.16；q995的.03/.08。有效激活边比例范围{100*low:.6f}%–{100*high:.6f}%。13/15配置精确P非零，但其中3个低于有效激活门槛。完整P幅度、L1、分位数、条目/边比例均在各mechanism.json。激活稀疏，没有大面积吞噬全部边差异；稀疏本身不能证明这些边就是无线异常边。',
        f'3. **是否区别于匹配eta Ball？** {len(distinct)}/15通过IC-distinct；其余不作为异常边机制实际参与的证据。raw差异计算未仿射对齐、未用GT。',
        f'4. **同eta是否优于Ball？** {len(positive)}/15通过预注册机制正向信号；保留全部配对改善/退化。不能只因超越S2就宣称机制收益。',
        f"5. **是否超过S2-J/CG？** 依据原G3，IC8 SURVIVOR {g3['IC8_survivor_count']}、TARGET_HIT {g3['IC8_target_count']}；每个配置分别相对J/CG的全精度变化在G3.json。P=0或不可区分Ball的任务收益单列BALL_CONSTRAINT_GAIN。",
        '6. **平均还是尾部？** eta=.03三个IC配置相对匹配Ball：Mean改善1.173870%–1.329649%，Median改善1.565333%–1.899983%，P90改善1.383613%–1.648915%，P95改善1.223998%–1.514022%；Max却恶化38.876209%–50.125364%。收益覆盖典型误差和P90/P95，同时有极端误差损害，不能笼统说“尾部都改善”。',
        f'7. **大eta结构退化或坍缩？** 数值硬门槛最小spread={minspread:.6f}，最大同点比例={maxdupes:.6e}；全部eta原始/仿射图保留。未把仿射缩放恢复误写为原始尺度保持；具体图谱观察见人工复核段。',
        f'8. **旧IC8接近S2的问题解决了吗？** 已解决“无法证明机制参与”的覆盖缺口：P-active={len(active)}、IC-distinct={len(distinct)}。没有获得超越S2的任务收益。机制正向信号仅在eta=.03的小修正区；质量接近S2的eta=.08处，q95实际激活但无增益，q99/q995输出接近Ball且无>=1%增益。高质量区域的增量价值仍未建立。eta=.50的q99/q995为精确二次区域。旧P2A的数值与结论不回写。',
        ('9. **值得继续研究IC边权鲁棒性？** 仅足够支持一次另行审核的小范围机制核验，尚不足以支持继续大规模性能优化。3个正向信号是探索依据，Max损害和0个S2 SURVIVOR是明确反证边界。当前没有边权扰动证据，不能声称权重鲁棒性；本轮按任务层面NO_GO_EXPLORATORY停止，不自动追加实验。' if positive else '9. **值得继续研究IC边权鲁棒性？** 本轮没有达到预注册机制正向信号，当前证据不足以支持继续投入边权鲁棒性实验；不追加参数或扰动。'),
        '', '## 资源、复现与产物','',
        f"GPU：{manifest['environment']['gpu']}；float64。累计已记录求解循环{total_time:.3f}s（含未达标尝试及人工中断前最后记录的耗时）；数值运行段累计墙钟{manifest['run_wall_time_s']:.3f}s（含新产物保存/审计，未计人工诊断暂停间隔）；评价{manifest['evaluation_wall_time_s']:.3f}s。峰值Torch allocated VRAM={peakvram}bytes，采样进程RAM={peakram}bytes；采样100ms，不能视为系统精确RAM峰值。没有训练，train_time_s为空；实际GPU总占用另见gpu_snapshot_during_fulln.csv，包含桌面/其它进程，不能冒充方法独占显存。",
        '所有配置保留显式P/dual/U、solver.json/history.csv、运行配置与资源；不涉及神经网络checkpoint。旧1122个保护文件及冻结参数hash核验保持不变。',
        '', '```powershell',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_experiment --phase prepare',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_sanity',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_experiment --phase run',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_evaluate',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_report',
        '```',
        '上述代码在新输出目录重放；现目录运行入口只复用已完成配置，不覆盖历史结果。全新重放需在隔离checkout保留相同相对输入路径与固定常量。',
        '',link(ROOT/'parameter_scan.csv')+' · '+link(ROOT/'mechanism_summary.csv')+' · '+link(ROOT/'G3.json')+' · '+link(ROOT/'manifest.json'),
        '', '### 各eta标准comparison','']
    for eta in (.03,.08,.16,.32,.5):
        folder=ROOT/'comparison'/f'eta{round(eta*100):03d}'
        lines.append(f"- eta={eta:g}：{link(folder/'comparison.md','完整表')} · {link(folder/'comparison.csv','CSV')} · {link(folder/'raw_channel_chart.png','原始图')} · {link(folder/'affine_channel_chart.png','仿射图')}")
    lines+=['','## 人工复核与结论边界','',
        '\n\n'.join(review['observations']) if review else '待本轮图像与表格复核后填写。未完成此项时不得标记交付完成。','',
        f"实际分支`{manifest['git_branch']}`，HEAD`{manifest['git_head']}`；本轮尚未提交或上传。旧有工作区改动未清理。",
        '**STOP**：不执行P2B/P3/P4/P5、Edge、Stage-1训练、额外网格或边权扰动，等待用户审核。']
    if diagnostic:
        position=lines.index('## 全部评价指标')-1
        lines[position:position]=['','## 为什么30000步未达标：预算与对偶暂态诊断','',
            f"Ball-GLR eta=.50首轮prox={diagnostic['max_prox_at_initial_cap']:.6e}；分解后primal最大{max(diagnostic['primal_prox_xy']):.6e}，dual最大{max(diagnostic['dual_prox_xy']):.6e}。残差来自对偶暂态，不是约束违例/NaN或持续发散。",
            f"15000–30000步的实测指数衰减率约{diagnostic['fitted_decay_rate_xy'][0]:.9f}/step，理论对偶relaxation log(1+sigma/(2beta))={diagnostic['theoretical_dual_relaxation_log_rate']:.9f}/step，预测达到2e-6约32219–32229步，实际32250步。5000到30000步目标相对变化仅{diagnostic['objective_relative_change_5000_to_30000']:.6e}。",
            f"首轮30000步与接受解的U相对差{diagnostic['U_relative_difference_30000_vs_accepted']:.6e}；直接用精确二次梯度的primal KKT最大{max(diagnostic['direct_primal_KKT_using_exact_quadratic_gradient_xy']):.6e}。为保持原完整停止协议，本轮仍不把未收敛的PDHG状态算作VALID。",
            '方案判断：输入尺度alpha、同eta Ball对照和原凸目标合理；旧较小eta网格继承的统一30000步上限不足以覆盖eta=.50。step scale随半径增加导致sigma变小，完整对偶证书变慢；机械冷启动会重复已完成的迭代。这是执行预算适配问题，尚无证据说明目标本身难解。60000约为观测所需步数的1.86倍安全余量，调整在本轮GT评价前完成。未修改步长规则或停止阈值。',
            link(ROOT/'solver_budget_diagnosis.json','诊断数值')+' · '+link(ROOT/'solver_budget_diagnosis.png','收敛图')+' · '+link(ROOT/'solver_budget_amendment.json','预算变更')]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(ROOT/'report_build.json',dict(code=file_record(Path(__file__)),status='REVIEWED' if review else 'AWAITING_VISUAL_AND_TABLE_REVIEW'))


def audit():
    if '待本轮图像与表格复核后填写' in (ROOT/'REPORT.md').read_text(encoding='utf-8'):raise ValueError('Visual review incomplete')
    maximum=0.;rowcount=0
    for path in (ROOT/'comparison').rglob('comparison.csv'):
        rows=list(csv.DictReader(path.open(encoding='utf-8')))
        for row in rows:
            r=read(Path(row['source_json']))
            if row['ct']:
                for key in METRICS:
                    actual=r['metrics'][key] if key in METRICS[:3] else r['localization_errors_after_affine'][key]
                    maximum=max(maximum,abs(float(row[key])-actual))
            rowcount+=1
        if path.parent!=ROOT/'comparison':
            if not np.array_equal(np.load(path.parent/'metric_subset_seed10.npy'),np.load(P0/'comparison/metric_subset_seed10.npy')):raise ValueError('Metric subset changed')
            for suffix in ('raw','affine'):
                if not (path.parent/f'{suffix}_channel_chart.png').exists():raise ValueError('Comparison image missing')
            aligned=np.load(path.parent/'aligned_charts.npz');reference=np.load(P0/'s2_cg/embedding.npz')
            if not np.array_equal(aligned['indices'],reference['indices']) or not np.array_equal(aligned['positions'],reference['positions']):raise ValueError('Aligned node identity changed')
    if maximum>1e-12:raise ValueError('CSV/JSON disagreement')
    manifest=read(ROOT/'manifest.json')
    for item in manifest['protected_inputs']+manifest['frozen_parameters']:
        if file_record(Path(item['path']))!=item:raise ValueError('Protected source changed')
    manifest.update(status='COMPLETED_STOP',protected_inputs_unchanged=True,
        final_code=[file_record(p) for p in Path('methods/stage2_modern_graph').glob('ic8_scale*.py')],
        main_report=file_record(ROOT/'REPORT.md'),stop='Await user review. No follow-on experiment executed.')
    write_json(ROOT/'manifest.json',manifest)
    write_json(ROOT/'delivery_audit.json',dict(status='PASS',comparison_rows_checked=rowcount,max_CSV_JSON_metric_difference=maximum,
        protected_files=len(manifest['protected_inputs']),protected_files_unchanged=True,all_five_metric_subsets_identical=True,
        images_manually_reviewed=True,branch=manifest['git_branch'],HEAD=manifest['git_head'],committed=False,uploaded=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    audit() if args.audit else main()
