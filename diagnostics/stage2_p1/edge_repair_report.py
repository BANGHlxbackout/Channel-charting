"""Read saved Edge numerical states; make no Channel Chart or GT metrics."""
import csv
import difflib
import json
import subprocess
import sys
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common_metrics import write_json
from diagnostics.stage2_p0.preflight import file_record

ROOT=Path('results/stage2_modern_graph_20261019/p1_edge_solver_repair')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def link(path,label=None):
    return f'[{label or path.name}]({path.resolve().as_posix()})'


def plot_history(folder):
    fig,axes=plt.subplots(3,2,figsize=(13,11));colors={'fixed':'#555555','residual_balance':'#0072B2'}
    for strategy in ('fixed','residual_balance'):
        path=folder/strategy/'history.csv'
        rows=list(csv.DictReader(path.open(encoding='utf-8')))
        x=np.array([float(a['iteration']) for a in rows])
        values=lambda key: np.maximum([float(a[key]) for a in rows],1e-18)
        for key,style in [('objective','-'),('l1_S',':'),('group_E','--'),('fidelity','-.'),('framelet',(0,(1,1)))]:
            axes[0,0].plot(x,values(key),linestyle=style,color=colors[strategy],label=strategy+': '+key)
        for key,style in [('self_Y_relative','-'),('Y_S_relative','--'),('frame_primal_residual',':')]:
            axes[0,1].semilogy(x,values(key),linestyle=style,color=colors[strategy],label=strategy+': '+key)
        for key,style in [('Y_row_max','-'),('row_sum_residual','--')]:
            axes[1,0].semilogy(x,values(key),linestyle=style,color=colors[strategy],label=strategy+': '+key)
        for key,style in [('primal_residual','-'),('dual_residual','--')]:
            axes[1,1].semilogy(x,values(key),linestyle=style,color=colors[strategy],label=strategy+': '+key)
        for key,style in [('stationarity','-'),('prox_S','--'),('prox_E',':'),('prox_Q','-.')]:
            axes[2,0].semilogy(x,values(key),linestyle=style,color=colors[strategy],label=strategy+': '+key)
        for key,style in [('mu1','-'),('mu2',':'),('mu3','--'),('mu4','-.')]:
            axes[2,1].semilogy(x,values(key),linestyle=style,color=colors[strategy],label=strategy+': '+key)
    for ax,title in zip(axes.ravel(),['Objective components','Self-expression / split / frame','Y and S row-sum constraints','Primal / linearized dual residual','KKT / proximal residual','Penalty history']):
        ax.set(title=title,xlabel='iteration');ax.grid(alpha=.25);ax.legend(fontsize=6)
    axes[1,0].axhline(1e-3,color='red',alpha=.5,linewidth=.7)
    fig.suptitle(folder.parent.name+' '+folder.name+' — numerical solver comparison')
    fig.tight_layout();fig.savefig(folder/'convergence.png',dpi=160);plt.close(fig)


def main():
    manifest=read(ROOT/'manifest.json')
    method_order={'SEF':0,'FE':1,'DoT':2}
    paths=sorted(ROOT.glob('n*/*/*/result.json'),key=lambda p:(int(p.parts[-4][1:]),method_order[p.parts[-3].split('-')[0]],p.parts[-3],p.parts[-2]))
    results=[read(p) for p in paths]
    if len(results)!=30:
        raise ValueError('expected 20 small and 10 original256 cases; preserve partial outputs, do not report completion')
    comparison=ROOT/'comparison';comparison.mkdir(exist_ok=True)
    fields=['N','method','initialization','strategy','status','iterations','primal','dual','stationarity','equality','self_Y','Y_row','S_row','Y_S','frame','Sblock_gap','Ublock_gap','solver_time_s','source_json']
    write_csv(comparison/'comparison.csv',[{key:r[key] for key in fields} for r in results])
    references=[]
    for r in results:
        references.append(dict(N=r['N'],method=r['method'],initialization=r['initialization'],strategy=r['strategy'],
            block='fixed_U_S/E',native_objective=r['Sblock_native_objective'],reference_objective=r['Sblock_reference_objective'],
            relative_gap=r['Sblock_gap'],source_json=r['source_json']))
        if r['method']!='SEF':
            references.append(dict(N=r['N'],method=r['method'],initialization=r['initialization'],strategy=r['strategy'],
                block='fixed_S_U',native_objective=r['equality_eliminated_objective'],reference_objective=r['Ublock_reference_objective'],
                relative_gap=r['Ublock_gap'],source_json=r['source_json']))
    write_csv(ROOT/'reference_objectives.csv',references)
    components=[]
    for r in results:
        components.append(dict(phase='repair',N=r['N'],method=r['method'],initialization=r['initialization'],
            strategy=r['strategy'],**r['objective_components'],total=r['objective'],source_json=r['source_json']))
    for r in read(ROOT/'legacy_diagnosis.json'):
        source=ROOT.parent/'p1_math_sanity/edge256'/(r['method'].lower()+'-p-small-'+r['initialization'])/'checks.json'
        components.append(dict(phase='original_P1',N=256,method=r['method'],initialization=r['initialization'],strategy='fixed',
            **r['components'],total=sum(r['components'].values()),source_json=str(source.resolve())))
    write_csv(ROOT/'objective_components.csv',components)
    lines=['# P1-Edge numerical solver comparison','','Only numerical convergence; fixed objective and inputs. No GT or Channel Chart metrics.','',
           '| N | method/init | policy | status | iterations | primal | dual | KKT | max constraint | max ref gap | time(s) | source |',
           '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    gates=[];tails=[];identities=[]
    for r,path in zip(results,paths):
        maxconstraint=max(r[k] for k in ['equality','self_Y','Y_row','S_row','Y_S','frame'])
        lines.append(f"| {r['N']} | {r['method']}/{r['initialization']} | {r['strategy']} | {r['status']} | {r['iterations']} | {r['primal']:.6e} | {r['dual']:.6e} | {r['stationarity']:.6e} | {maxconstraint:.6e} | {max(r['Sblock_gap'],r['Ublock_gap']):.6e} | {r['solver_time_s']:.6f} | {link(path,'JSON')} |")
        for name,actual,threshold,passed in [
            ('constraints',maxconstraint,'<=1e-3',r['constraints_pass']),
            ('complete_stop',r['stopping_error'],'<2e-5 AND converged',r['solver_completion_pass']),
            ('convex_block_objective',max(r['Sblock_gap'],r['Ublock_gap']),'<=1e-3',r['block_reference_pass']),
            ('stationarity',r['stationarity'],'<=5e-3',r['stationarity_pass']),
            ('finite_noncollapsed',r['spread_ratio'],'finite AND centered spread ratio>=0.05',r['finite_noncollapsed_pass']),
            ('PCG',r['pcg_residual'],'<1e-5 (SEF no U-step)',r['pcg_residual']<1e-5)]:
            gates.append(dict(N=r['N'],method=r['method'],initialization=r['initialization'],strategy=r['strategy'],
                gate=name,actual=actual,threshold=threshold,status='PASS' if passed else 'FAIL',evidence=r['source_json']))
        history=list(csv.DictReader((path.parent/'history.csv').open(encoding='utf-8')))
        tail=[row for row in history if int(row['iteration'])>r['iterations']-1000]
        tails.append(dict(source_json=r['source_json'],final_window_iterations=[int(tail[0]['iteration']),int(tail[-1]['iteration'])],
            ranges={key:dict(min=min(float(a[key]) for a in tail),max=max(float(a[key]) for a in tail))
                    for key in ['self_Y_relative','Y_S_relative','Y_row_max','row_sum_residual','frame_primal_residual','primal_residual','dual_residual','stationarity']}))
        state=np.load(path.parent/'state.npz');y,s=state['Y'],state['S']
        if not all(np.isfinite(state[key]).all() for key in state.files):raise ValueError('nonfinite saved state')
        row_difference=(s-y)@np.ones(r['N'])
        identity_error=float(np.max(np.abs((s.sum(axis=1)-1)-(y.sum(axis=1)-1)-row_difference)))
        changes=read(path.parent/'penalty_history.json')
        multiplier_error=max([c['unscaled_multiplier_relative_error'] for c in changes] or [0.])
        if identity_error>1e-10 or multiplier_error>1e-12:raise ValueError('row identity or multiplier invariance failed')
        identities.append(dict(source_json=r['source_json'],row_identity_error=identity_error,
            row_error_split_gain=float(np.linalg.norm(row_difference)/max(1e-30,np.linalg.norm(s-y))),
            scaled_dual_preservation_error=multiplier_error,penalty_change_count=sum(c['new_mu']!=c['old_mu'] for c in changes)))
    (comparison/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_csv(ROOT/'gates.csv',gates);write_json(ROOT/'gates.json',gates)
    write_json(ROOT/'tail_residual_diagnosis.json',tails);write_json(ROOT/'constraint_and_dual_identities.json',identities)
    for folder in sorted({p.parent.parent for p in paths}):plot_history(folder)
    for item in manifest['protected_inputs']:
        if file_record(Path(item['path']))!=item:raise ValueError('frozen IC/FN/oldP1/source changed')
    full=[r for r in results if r['N']==256];b=[r for r in full if r['strategy']=='residual_balance']
    manifest.update(status='REPAIR_PASS' if all(r['status']=='PASS' for r in b) else 'NUMERICAL_BLOCKER',
        stopped=True,stop='P1-Edge only; await user review; no additional retry or P2A',protected_inputs_unchanged=True,
        configurations_completed=len(results),full256_B_pass_count=sum(r['status']=='PASS' for r in b),
        final_code=[file_record(Path(p)) for p in ['methods/stage2_modern_graph/edge_small.py','diagnostics/stage2_p1/edge_repair.py',__file__,'tests/test_stage2_edge_repair.py']])
    manifest['environment']=dict(python=sys.version,numpy=np.__version__,cvxpy=version('cvxpy'),clarabel=version('clarabel'),platform=platform.platform(),blas_threads=1,
        python_executable=sys.executable,reference_environment='same diffusion_charting environment as accepted P1')
    manifest['entry_reporting_changes']='After formal start only init bootstrap was added; after all runs Lambda2 export became ndarray for legacy NPZ/JSON serializer compatibility. Update equations, strategy, stopping rules unchanged. No new configuration or run.'
    old_source=subprocess.check_output(['git','cat-file','blob',manifest['original_git_blob']],text=True)
    current_source=Path('methods/stage2_modern_graph/edge_small.py').read_text(encoding='utf-8')
    (ROOT/'edge_small.diff').write_text(''.join(difflib.unified_diff(old_source.splitlines(True),current_source.splitlines(True),fromfile='edge_small.py original P1',tofile='edge_small.py repaired diagnostics')),encoding='utf-8')
    write_json(ROOT/'manifest.json',manifest)
    (ROOT/'git_diff_stat.txt').write_text(subprocess.check_output(['git','diff','--stat'],text=True),encoding='utf-8')
    report=['# P1-Edge Solver Repair','','**数值求解诊断；无 Channel Chart、仿射图、CT/TW/KS 或定位指标。IC 和 Framelet Node 代码/结果未改、未重跑。**','',
        '## 1. 验收与停止','',f"原256节点 B 策略 {sum(r['status']=='PASS' for r in b)}/5 配置 PASS；最终状态 **{manifest['status']}**。仅完成本轮固定A/B，没有数值重试。P2A/P2B/P3/SparseDoT/full-N均未执行，已停止等待审核。",'',
        '| 方法/初始化 | A状态 / 次数 | B状态 / 次数 | B primal | B dual | B KKT | B最大约束 | B最大block gap | A/B秒 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in b:
        a=next(t for t in full if t['method']==r['method'] and t['initialization']==r['initialization'] and t['strategy']=='fixed')
        report.append(f"| {r['method']}/{r['initialization']} | {a['status']} / {a['iterations']} | {r['status']} / {r['iterations']} | {r['primal']:.6e} | {r['dual']:.6e} | {r['stationarity']:.6e} | {max(r[k] for k in ['equality','self_Y','Y_row','S_row','Y_S','frame']):.6e} | {max(r['Sblock_gap'],r['Ublock_gap']):.6e} | {a['solver_time_s']:.3f}/{r['solver_time_s']:.3f} |")
    report+=['',f"逐项实数/阈值/证据：{link(ROOT/'gates.csv')}；全部32/64/256配置：{link(comparison/'comparison.md')} / {link(comparison/'comparison.csv')}；reference目标全精度见reference_objectives.csv。每配置独立保存solver.json、state.npz（U/S/E/Y/Q/所有unscaled multiplier）、逐10步history.csv和penalty_history.json，以及凸reference objective/solver状态/数组。",'',
        '## 2. 原目标与增广拉格朗日核查','',
        '用S表示论文Z。冻结目标为 `||S||1 + sum_i d_i||E_i||2 + 0.5 sum_i d_i||U_i-X_i||² + (nu/1.25) sum_h c_h sum_i d_i |T_h U_i|1`；SEF固定U且去掉fidelity/Framelet，FE去掉Framelet，DoT保留四项。d为原始加权邻接行和，不除以均值、N或d_sum；lambda1=lambda2=1，DoT nu=.1，c=(1,.25)，active=(low2,high1,high2)，low2权重0但保留约束。',
        '令 `c1=U-YU-E, c3=Y1-1, c4=Y-S, c2_b=T_b U-Q_b`，增广项为 `sum_j <Lambda_j,c_j> + mu_j/2 ||c_j||²`。SEF没有c3，S始终diag=0且可有负值。Y不单独强制diag，Y=S在极限传递此约束。消去E使自表达精确，但不自动使S行和可行；equality_eliminated_objective不能冒充全约束可行值。','',
        '更新方程：','',
        '- `Y (mu4 I + mu1 UU.T + mu3 11.T) = (mu1(U-E)+Lambda1)U.T + (mu3 1-Lambda3)1.T + mu4 S-Lambda4`，SEF去掉行和项；Woodbury与独立凸解一致。',
        '- `S = offdiag soft(Y+Lambda4/mu4, 1/mu4)`。',
        '- `U` 由PCG精确解 `D + mu1(I-Y).T(I-Y) + mu2 sum_b T_b.T T_b`，RHS为 `DX+(I-Y).T(mu1 E-Lambda1)+sum_b T_b.T(mu2 Q_b-Lambda2_b)`。FE无Framelet项，SEF不更新U；没有tight-frame近似。',
        '- `E_i = group_soft(((I-Y)U+Lambda1/mu1)_i, d_i/mu1)`。',
        '- `Q_b = soft(T_b U+Lambda2_b/mu2, nu*c_b*d/(1.25*mu2))`，低通阈值0；该c2方向与论文Q-TU相反，所以Lambda2的符号也相反，数学等价。',
        '- 按Y→S→U→E→Q的现有Gauss-Seidel顺序完成后，`Lambda_j += mu_j*c_j`。此顺序各步仍最小化同一增广目标；不把多块非凸ADMM写成全局最优保证。','',
        '核查结论：没有发现本地更新公式、矩阵逆、soft-threshold或对偶符号的实现错误。作者代码的E阈值未乘d、部分对偶更新使用S而非Y，不能照抄来替代本轮冻结目标。相应论文目标参见 [Eq5/Eq6/Eq17](https://arxiv.org/html/2111.03264v1)，[固定作者源码](https://github.com/bzho3923/GNN_DoT/blob/4db39904e4099d82cef686d4732c33150814ece4/denoising_filters.py)。','',
        '## 3. 独立reference与修改边界','',
        '32/64节点，在第三次sweep的非零对偶状态，独立CVXPY逐块构造并求解Y/S/U/E/Q增广子问题。初次tol1e-9在64节点soft-threshold附近参考解的向量误差为几e-6；仅收紧reference至1e-12，原始证据仍在initial_reference_tolerance.json，未放宽1e-6向量误差门槛。最终全部通过，最大向量差2.413588e-7，S增广objective相对差<=8.208e-14。32/64完整20个A/B配置也全部PASS。联合FE/DoT最终只比较固定U的S/E凸块和固定S的U凸块，不宣称全局凸reference。',
        f"证据：{link(ROOT/'augmented_reference_n32.json')} / {link(ROOT/'augmented_reference_n64.json')} / {link(ROOT/'unit_tests.log')}。3项Edge-only单元测试通过；未调用IC/FN Node测试。",'',
        f"实际源码差异：{link(ROOT/'edge_small.diff')}；当前代码{link(Path('methods/stage2_modern_graph/edge_small.py'))}。原源码git blob/hash在manifest，diff可恢复修改前内容。修改仅含分量日志、完整残差/停止条件、单一penalty策略及预算截止，不改变原更新顺序或目标。A/B共享所有诊断和停止条件。",'',
        '## 4. Residual balancing与完整停止条件','',
        'A固定mu1=mu3=mu4=10。B初值相同，每50次、仅第100..3000次比较归一化primal和linearized dual drift：r>10s则mu乘2；s>10r则除2；范围[1e-3,1e4]，之后冻结。mu2始终原degree.mean。变mu时scaled dual乘old_mu/new_mu，保留unscaled multiplier；全批次最大保持误差见constraint_and_dual_identities.json。此规则在正式A/B前已固定，没有GT/性能调参。','',
        '双线性多块问题没有唯一的两块ADMM对偶残差。本轮明确记录：s4=mu4||Delta S||；s2=mu2||sum T.T Delta Q||；s1同时记录对Y及U stationarity的线性化block drift。行约束只有Y这一个变量块，canonical两块s3会恒零，故使用 `mu3||(Delta Y)11.T||` 作为**行约束image-drift代理**。它仅用于预先固定的数值平衡，不能当作独立收敛证明。各项按相应lambda stationarity尺度归一化，具体公式见源码history记录处。',
        '最终收敛必须满足原equality/S-row/split/frame/progress/stationarity全部<2e-5，并增加Y-self、Y-row、完整prox(S/E/Q)、U/Y迭代变化及primal/linearized-dual<2e-5。prox检查对应S=prox(S+Lambda4)、E=prox(E+Lambda1)、Q=prox(Q+Lambda2)。硬G1所有约束<=1e-3、block gap<=1e-3、stationarity<=5e-3不变；finite且中心化spread>=输入0.05，PCG<1e-5。达到15000上限不算收敛。','',
        '## 5. 根因证据与分量量级','',
        '旧日志没有保存Y、multiplier、Q，因此不能凭空重建缺失轨迹。本轮A从同一输入/初始化重放固定penalty，补齐全部轨迹，并将最终U/S/E与原P1 attempt2逐项比较（差异见各result.json的legacy_relative_differences）。旧P1实际耗时包含两个attempt及reference，不能与本轮纯solver_time直接计算加速比；A/B本轮相同日志开销可比较。',
        '旧FE/DoT的L1项约256，FE fidelity约0.33–0.37、DoT fidelity约1.56、Framelet约2.30，E项为0。目标总值与凸block参考接近时，行和仍可有百分之几误差，故relative objective不能代替可行性。SEF的L1约190.18、group_E约0.52。实际全精度分量见legacy_diagnosis.json和各history.csv。',
        '结构上 `(S1-1)-(Y1-1)=(S-Y)1`：Frobenius相对split很小也不自动保证S每行和很小，需单独监控。A/B末尾1000步残差范围见tail_residual_diagnosis.json，逐项约束身份/penalty rescale数值误差见constraint_and_dual_identities.json。这些检查用于判断哪条约束停滞；不会将泛化或图谱质量变化混入数值诊断。','',
        '本轮能检验的是该固定数据/初始化/预算下，完整更新正确后，一种penalty平衡能否消除阻塞。若B通过，说明该个案不需要改目标即可求解；若B仍FAIL，只能定位为当前数值策略/预算下的NUMERICAL_BLOCKER，不能证明目标本身不可求解。没有追加第三策略或重试。','',
        '## 6. 数值收敛图与资源','']
    for folder in sorted({p.parent.parent for p in paths}):report.append('- '+link(folder/'convergence.png',folder.parent.name+'/'+folder.name))
    report+=['',f"正式阶段墙钟（含独立reference与IO）：{manifest.get('stage_times_s')} 秒。每配置纯solver时间见comparison。CPU float64，未用GPU；RAM峰值未完整记录，不冒称已测。每配置最多15000次、无重试，全批次计算上限3600秒，A/B无lambda/nu/图/数据规模搜索。",'',
        f"{link(ROOT/'manifest.json')} 保存冻结策略、清单中旧IC/FN/Edge结果和作者源码的前后hash、环境/代码与复现信息；所有保护文件hash一致。{link(ROOT/'git_diff_stat.txt')} 是已有整个工作区差异，含历史改动，不将其当成本轮修改；本轮edge改动见专用diff。分支{manifest['git_branch']}，HEAD {manifest['git_head']}，未提交/上传。",'',
        '复现（独立空输出目录/副本，入口拒绝覆盖当前结果）：','',
        '```powershell','D:/anaconda3/envs/diffusion_charting/python.exe -m unittest tests.test_stage2_edge_repair -v',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.edge_repair --stage init',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.edge_repair --stage small',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.edge_repair --stage full',
        'D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.edge_repair_report','```','',
        '**STOP：本轮数值诊断交付后等待用户审核；不执行P2A/P2B/P3/SparseDoT或新的full-N。**']
    diagnosis=['', '| 原256配置 | policy | U-YU-E相对 | Y-S相对 | Y行和max | S行和max | Frame split |',
               '|---|---|---:|---:|---:|---:|---:|']
    for r in full:
        diagnosis.append(f"| {r['method']}/{r['initialization']} | {r['strategy']} | {r['self_Y']:.6e} | {r['Y_S']:.6e} | {r['Y_row']:.6e} | {r['S_row']:.6e} | {r['frame']:.6e} |")
    a_fe=next(r for r in full if r['method']=='FE' and r['initialization']=='author' and r['strategy']=='fixed')
    prior=next(r for r in b if r['method']=='FE' and r['initialization']=='prior')
    diagnosis+=['',f"具体定位：A的FE默认初始化，Y行和仅{a_fe['Y_row']:.6e}、Y-self仅{a_fe['self_Y']:.6e}，但S行和{a_fe['S_row']:.6e}。其split传播至S约束是主要可见阻塞；不能将它说成Y行约束本身完全没有收敛。B的FE先验初始化状态为{prior['status']}，说明单一penalty策略能影响该个案；并不保证其默认初始化或整家族通过。",'',
        'L1(S)>=N 对精确行和约束成立（逐行三角不等式），非负系数达到等号；这里没有添加非负约束。二维自表达可有许多结构近似等价解，因此L1大项/局部目标接近不足以认证收敛。该事实只解释数值目标的平坦性，未被用来更改目标或设计新方法。', '',
        '根因边界：独立增广子问题和A的原轨迹重放排除了已核验范围内的更新代数错误；旧停止条件的诊断覆盖不完整已补齐，但旧S行和检查已正确阻止了假成功，所以原P1 FAIL结论仍有效。残差比例平衡本身不能保证绝对残差趋零。SEF是凸问题且CVXPY给出optimal参考，不能将当前ADMM未达停止门槛称为目标不可解；FE/DoT整体双线性非凸，本轮仍不足以唯一归因于penalty、变量更新顺序或非唯一解。保留NUMERICAL_BLOCKER，不强行给出单一因果结论。','']
    index=report.index('## 6. 数值收敛图与资源');report[index:index]=diagnosis
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],configurations=len(results),B_pass=manifest['full256_B_pass_count'],gates=len(gates),failures=sum(g['status']=='FAIL' for g in gates)),indent=2))


if __name__=='__main__':main()
