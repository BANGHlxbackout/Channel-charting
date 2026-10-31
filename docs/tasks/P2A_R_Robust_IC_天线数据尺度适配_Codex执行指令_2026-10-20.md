# P2A-R：Robust-IC 天线数据尺度适配实验

## 接收方与任务性质

**接收方：Codex，在仓库根目录 `D:\diffusion_refined_charting` 执行。**

这是一次新的、有限范围的正式实验。旧 P2A 正确实现了论文 IC8，但获胜配置使用：

```text
alpha = 0.5 / 0.8
eta = delta0
P = 0
```

因此论文新增的异常边分解没有参与，结果自然接近 S2。

旧实验还存在两个范围问题：

1. `eta={0.5,1,1.5}delta0` 始终以 S2 移动量为中心；
2. `alpha={0.2,0.5,0.8}` 从大量激活直接跳到完全不激活，没有覆盖适度激活区间。

本任务重新回答：

> 当 IC8 的阈值按照天线阵列 Chart 边差异尺度设置，并允许其使用独立于 S2 的移动范围时，能否得到真正区别于普通 Laplacian、且优于 B0+S2 的结果？

不得覆盖旧 P2A 结果。

## 任务边界

仅执行：

- Ball-GLR；
- IC8-P 论文形式；
- B0 full-N，`N=20997`；
- 固定 G128；
- 数据尺度适配后的有限参数矩阵；
- 统一评价与 Channel Chart。

不执行：

- IC8-G 或任何旋转/group 适配；
- IC9-P/G 重跑；
- P2B Framelet；
- Edge/DoT；
- G1-Identity/G1-Graph 扩展；
- 学习新图；
- GT 引导的边权、阈值或参数选择；
- 结果出来后的追加网格。

旧 P2A、P1、P0、B0、S2 及其结果全部只读。

输出目录：

```text
results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/
```

## 一、保留的论文公式

IC8-P 继续严格使用：

\[
\min_{u,p}
\alpha\lVert p\rVert_1+
(1-\alpha)\lVert\sqrt{W_{\mathrm{IC}}}Du-p\rVert_2^2
\]

满足：

\[
\lVert u-x\rVert_2
\le
\eta\lVert x-\operatorname{mean}(x)\rVert_2
\]

并保留当前宽 box。

要求：

- x、y 按照论文标量问题分别求解；
- 不添加 S2 的二次 fidelity；
- 不把 IC8 替换成手写 Huber 近似；
- 保留并实际优化 \(P\)；
- 使用已经验证的 PDHG/CUDA 求解器；
- `W_IC=W/mean(W_e)`；
- 不修改图拓扑和相对边权；
- GT 不进入求解器、参数生成、停止条件或 checkpoint 选择。

## 二、新增 Ball-GLR 必要对照

实现：

\[
\min_u
\lVert\sqrt{W_{\mathrm{IC}}}Du\rVert_2^2
\]

满足与 IC8 完全相同的：

- observation ball；
- box；
- x/y 分轴；
- 输入规范化；
- 图和权重；
- 数值容差；
- \(\eta\)。

Ball-GLR 只删除 \(P\) 和 L1 项。

优先复用 IC8 求解器，增加一个明确的 `aux_mode=disabled` 分支，避免重复实现。必须通过小图 CVXPY 参考。

Ball-GLR 的作用不是限制 IC，而是判断 IC 是否真的产生了不同于普通 Laplacian 的解。

## 三、按照天线数据确定 IC 激活阈值

不再直接使用：

```text
alpha = 0.2, 0.5, 0.8
```

先在规范化 B0 上计算全部无向边、两个坐标的：

\[
g_{e,c}
=
\left|
\sqrt{w_e}(DX_c)_e
\right|.
\]

将两个坐标的 \(g_{e,c}\) 合并成一个 pool，只使用输入 B0 和 G128，不使用 GT。

固定三个阈值：

```text
kappa_95  = pooled 95% quantile
kappa_99  = pooled 99% quantile
kappa_995 = pooled 99.5% quantile
```

因为 IC8 的激活阈值满足：

\[
\kappa
=
\frac{\alpha}{2(1-\alpha)},
\]

所以计算：

\[
\alpha
=
\frac{2\kappa}{1+2\kappa}.
\]

运行前保存：

```text
input_edge_gradient_distribution.json
derived_alpha_values.json
```

必须记录：

- 全部分位数；
- 三个 \(\kappa\)；
- 三个实际 \(\alpha\)；
- 每个阈值下输入边差异预计超过阈值的比例；
- 输入 hash。

这些参数在计算任何 GT 指标前冻结。

三个配置分别代表：

- 约 5% 的初始边差异有机会进入 \(P\)；
- 约 1%；
- 约 0.5%。

不得根据结果改变分位数或增加第四个阈值。

## 四、使用独立于 S2 的移动范围

不再使用：

```text
eta = factor * delta0
```

固定以下无量纲移动范围：

```text
eta = 0.03
eta = 0.08
eta = 0.16
eta = 0.32
eta = 0.50
```

定义仍然是：

\[
r_c
=
\eta\lVert X_c-\operatorname{mean}(X_c)\rVert_2.
\]

这些值构成从小修正到大修正的预注册路径，不由 S2 决定。

旧 `delta0=0.095171...` 只在报告中画一条参考线，不能参与新参数生成或筛选。

禁止在结果出来后：

- 缩小区间追最优；
- 插入额外 eta；
- 围绕最佳点继续搜索；
- 根据 GT 更改移动范围。

## 五、正式配置矩阵

### Ball-GLR

```text
5 eta
```

共 5 组。

### IC8-P

```text
3 derived alpha × 5 eta
```

共 15 组。

总计 20 个 full-N 配置。

全部配置必须运行完成后再统一评价。不得因为前几个结果不好而提前修改设计。

## 六、Sanity 与数值验收

正式 full-N 前，在原 P1 同类的 16/32/64 节点图上检查。

### Ball-GLR

- 与独立 CVXPY 目标相对差 `<=1e-3`；
- observation-ball feasibility `<=1e-4`；
- box 满足；
- stationarity/proximal residual 达到原标准。

### IC8-P

- 与独立 CVXPY 目标相对差 `<=1e-3`；
- \(P\) soft-threshold 正确；
- 输入阈值和派生 alpha 计算正确；
- `sqrt(W)Du-P` 实现正确；
- observation-ball 与 box 满足；
- 辅助变量不能被代码固定为零。

小图未通过则停止，不进入 full-N。

full-N 沿用旧 P2A 数值 Gate：

- finite；
- observation-ball feasibility 通过；
- proximal residual 通过；
- 无持续目标恶化；
- 节点索引完全一致；
- 无 collapse；
- 达到正式停止条件；
- 不因达到迭代上限宣布收敛。

每配置只允许旧 P2A 规定的一次同目标数值重试，不调整 alpha 或 eta。

## 七、必须记录的机制数据

每个 IC8 配置保存：

- 最终非零 \(P\) entry 数量；
- 非零 \(P\) edge 数量；
- 非零比例；
- \(\lVert P\rVert_1\)；
- `max/mean/median/nonzero quantiles of |P|`；
- 最终 \(\lvert\sqrt WDU\rvert\) 分布；
- observation-ball 实际使用比例；
- x/y 实际移动量；
- 与相同 eta Ball-GLR 的 raw RMS 差异；
- 与相同 eta Ball-GLR 的移动方向余弦；
- 目标中的 L1 分量和二次分量；
- 收敛步数、耗时和显存。

定义：

```text
P-active:
    nonzero P edge fraction >= 1e-4

IC-distinct:
    P-active
    and relative raw RMS difference from matched Ball-GLR >= 1e-4
```

这些门槛只判断论文机制是否实际参与，不使用 GT。

如果一个配置 \(P=0\)，必须标记：

```text
QUADRATIC_REGIME
```

不得将其作为 infimal-convolution 机制成功。

## 八、统一真实数据评价

所有数值有效配置使用完全相同的：

- B0；
- indices 和 positions；
- seed=10 固定 4199 节点 CT/TW/KS 子集；
- 全部 20997 节点仿射误差；
- `common_metrics.chart_metrics`；
- `common_metrics.localization_error_summary`；
- `run_adp_siamese.plot_chart`。

完整字段：

```text
method,
alpha_source,
alpha,
kappa,
eta,
ct,
tw,
ks,
mean_m,
median_m,
p90_m,
p95_m,
max_m,
P_nonzero_edges,
P_nonzero_fraction,
movement_x,
movement_y,
status,
solve_time_s,
peak_vram_bytes,
source_json
```

统一比较顺序：

```text
GT
B0
S2-J
S2-CG
matched-eta Ball-GLR
IC8-P
```

每个 eta 生成一组简洁 comparison：

```text
comparison/eta003/
comparison/eta008/
comparison/eta016/
comparison/eta032/
comparison/eta050/
```

每组必须包含：

- `comparison.csv`
- `comparison.md`
- `raw_channel_chart.png`
- `affine_channel_chart.png`
- `metric_subset_seed10.npy`
- `eval_protocol.json`
- `aligned_charts.npz`

如果三种 alpha 列数过多，可为每个 eta 输出一张包含全部方法的多列图，同时保持方法顺序清晰。不得只绘制最佳点。

## 九、结论判定

分别回答三个问题。

### Q1：IC 是否真正不同于普通 Laplacian？

要求至少一个数值有效配置同时满足：

```text
P-active
IC-distinct
```

否则结论：

```text
IC_MECHANISM_NOT_ACTIVATED
```

### Q2：适度激活是否比 Ball-GLR 更好？

对同 eta、同评价协议比较。

探索性机制信号要求：

- KS、Mean、P95 至少一项改善 `>=1%`；
- 另外两项不恶化超过 `2%`；
- `P-active=true`；
- `IC-distinct=true`。

否则不得把性能差异归因于 IC。

### Q3：是否优于 S2-J 和 S2-CG？

沿用原 G3：

- SURVIVOR；
- ADVISOR_HIT；
- TARGET_HIT。

但最终报告必须把：

- 相对 Ball-GLR；
- 相对 S2-J/S2-CG；

分开报告。

一个方法即使超过 S2，如果与 Ball-GLR 相同且 \(P=0\)，也只能称：

```text
BALL_CONSTRAINT_GAIN
```

不能称为 Robust-IC 增益。

## 十、停止规则

完成 20 个配置和统一评价后立即停止。

如果出现以下情况：

- 所有 IC8 配置 \(P=0\)；
- 有 \(P\) 激活但全部不优于 Ball-GLR；
- 有配置超过 S2 但与 Ball-GLR 一致；
- 大移动范围导致 collapse；
- 数值不能通过；

均完整保留结果，不增加新 alpha、eta、权重或图。

本任务不执行：

- 边权扰动；
- 天线专用新边权；
- P2B；
- P3；
- G1 扩展。

边权扰动只在本任务出现“P-active 且相对 Ball-GLR 有正向信号”后，另行决定，不自动执行。

## 十一、报告必须明确回答

1. 新 alpha 是否覆盖了旧实验遗漏的适度激活区间？
2. 哪些配置 \(P\neq0\)？
3. 非零 \(P\) 是否只出现在少量边，而非大面积吞噬全部差异？
4. IC8 是否实际产生了不同于 Ball-GLR 的解？
5. IC8 是否在相同 eta 下优于 Ball-GLR？
6. IC8 是否优于 S2-J/S2-CG？
7. 改善集中在 Mean、Median 还是尾部？
8. 大 eta 是否带来结构坍缩？
9. 旧 P2A 中“IC8 接近 S2”的原因是否被新实验解除？
10. 结果是否值得继续做边权不确定性实验？

交付：

```text
REPORT.md
parameter_scan.csv
mechanism_summary.csv
G2.json
G3.json
manifest.json
comparison/
所有配置的 embedding.npz
result.json
config.json
solver 日志和 history
```

完成后停止，等待用户审核。

## 十二、与旧 P2A 的本质区别

| 旧 P2A | 新 P2A-R |
|---|---|
| 移动范围围绕 S2 | 移动范围独立于 S2 |
| alpha 粗略取 0.2/0.5/0.8 | alpha 由天线 Chart 边差异分位数产生 |
| 从大量激活直接跳到零激活 | 明确覆盖 5%、1%、0.5% 候选异常边 |
| 只和 S2 比较 | 同时和相同移动范围的 Ball-GLR 比较 |
| 获胜点 P=0 仍标 survivor | P=0 明确标为二次区域 |
| group/旋转适配 | 完全移除 |
| 不能判断新机制是否参与 | 直接记录 P 激活和与 Ball-GLR 的差异 |

本任务不再要求 IC 迁就 S2 的移动量，也不围绕旋转做适配。它直接检查论文的核心能力是否能在天线 Chart 的实际边差异尺度上工作。

## 恢复检查点（2026-10-25；本任务唯一）

- 授权：仅P2A-R，旧P2A/P1/P0/B0/S2结果及原求解代码只读。固定N=20997、缓存G128与相对边权。用户在运行中补充：eta可灵活调整以寻找激活，原方案为初步设计；旧“eta不可更改”限制因此失效，alpha/图/目标及GT使用边界不变。当前已找到多个激活点，暂不追加eta，先完成现有20配置和同eta Ball比较，交付后停止。
- 输入：`results/stage2_modern_graph_20261019/p0_preflight/{chart_gauge.npz,graph/}`，B0为`results/adp_old_siamese/stride1_n20997_seed10/embedding.npz`。
- 新代码：`methods/stage2_modern_graph/ic8_scale_cuda.py`共享IC8/Ball开关；`ic8_scale_experiment.py`负责输入尺度冻结、资源记录与20配置。新增原因：旧入口固定旧27网格；本轮禁止修改旧代码/结果。
- 输出：`results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/`，与旧实验完全分开。
- 数学验证：16/32/64共24凸参考案例+更新回归+阈值回代共26项PASS，最大目标差4.44e-10，最大近端残差1.51e-6，与旧IC8的200步U/P/dual逐元素一致。参考一次CLARABEL inaccurate，固定SCS同目标独立复核通过，失败日志保留。alpha=.2441313527262478/.31110304399606037/.3441908057988971；池化条目率与边union率分别保存。relative raw RMS主分母为中心化B0 RMS，另存Ball中心化分母；参数生成无GT。
- 运行进度：前16配置全部首轮VALID；Ball eta=.50首轮30000不达prox(4.59e-6)，原重试于32250 PASS，累计17 VALID。用户随后明确允许调整步数并要求审查原因。独立状态分解：primal3.00e-16、dual4.59e-6；U相对接受解差2.38e-13；残差衰减预测约32229步，实际32250。因此是旧30000预算不覆盖大eta、对偶暂态慢，不是目标发散。详见`solver_budget_diagnosis.json/png`。
- 预算修正：只将剩余3个eta=.50 IC8单次上限改60000（约观测所需1.86倍余量），无额外重试；核函数、步长/容差、alpha/eta/图/目标均不变，尚无GT质量评价。已完成17结果不动。原正在求解q95于15550步人工停止归档`interrupted_runs/ic8_q95_eta050_before_budget_update/`，不是数值故障或有效结果；旧执行代码快照在`methods/stage2_modern_graph/archive/p2ar_initial_budget/`，变更依据及耗时在`solver_budget_amendment.json`。
- 完成事实：20/20 full-N均VALID，最后3配置于32700/32850/32950步通过；不存在运行进程。数值运行段累计6274.218926s，已记录求解循环6222.973625s（含初次失败及人工中断已记录耗时），峰值VRAM468262400bytes、采样RAM1367719936bytes。20日求解完成，25日中断恢复后统一评价146.521585s；未重跑求解或Stage-1。
- 研究结果：P-active10/15、IC-distinct8/15、13/15精确P非零；3个机制正向信号均为eta=.03。q95相对匹配Ball的Mean/P95改善1.329649%/1.514022%，Max恶化50.125364%；其余两个信号也有Max损害。15个IC相对J/CG均0 ADVISOR_HIT/SURVIVOR/TARGET_HIT，NO_GO_EXPLORATORY。描述性最小Mean q995 eta=.08为.319195292m，S2-J为.318747967m；没有按GT补扫eta。
- 交付核验：主`REPORT.md`逐项回答9问题，包含全部8指标；`parameter_scan.csv`、`mechanism_summary.csv`、G2/G3、5组标准comparison/raw/affine/协议/抽样/对齐数组及原生chart、每配置state/solver/history齐全。实际查看10张比较图、4张原生图及诊断图；`delivery_audit.json`核对58比较行与JSON差0，1122旧保护文件/派生参数hash未变。旧代码快照与预算调整中断保留。分支codex/new-stage1，HEAD f63652f，未提交或上传。
- 用户追加问题（2026-10-25）：以B0+S2为主要参照，诊断是参数覆盖不当还是方法增量不足。原20配置及报告继续完整保留。只读发现S2-CG真实eta_x=.08708173602446727、eta_y=.10228255353994962，而原共同eta .08/.16跨过此工作区；原旧alpha=.5/.8 common-delta0的SURVIVOR均P=0。
- 已撤回的追加方案：agent曾提出4个两轴基线预算匹配诊断，用户随后明确纠正CG的用途并要求先思考、不要无用执行。此方案不生效；`p2ar_baseline_budget_diagnostic/`与`diagnostics/stage2_p2ar_baseline_match/run.py`尚未创建，4个实验均未启动，不作为授权中的下一动作。
- 当前边界：原20配置/旧结果保留；后续唯一性能基线为B0+S2-J，CG只作已完成的历史求解核验。先基于现有结果分析参数覆盖是否足以解释差距，以及P的独立增量是否存在；不自动补网格、对齐预算或追加运行。
- 只读代码/机制核对（2026-10-25）：Robust-IC论文的核心思想是把图梯度差异在二次平滑与稀疏P之间分配。IC8是此机制的逐边形式，不能称它“不是核心”；旧Stage2规格将含Dᵀ耦合的IC9列为主候选。已实现的冻结Eq8/Eq9目标在16/32/64独立CVXPY参考上目标差最大约3e-9，20,997节点数值已收敛；不能据此声称逐行复现尚未直接核对的论文Algorithm1。
- 核心审计证据：旧IC9-P/G的18个已保存full-N状态P全部精确为0。按冻结Eq9在P=0的KKT条件，从这些状态直接计算的临界alpha见`results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic9_zero_aux_parameter_audit.json`；eta≈0.095时IC9-P两个坐标约.0132、IC9-G约.01057，而旧网格alpha最小.2。18/18旧配置的alpha均高于对应临界值；这是旧IC9参数覆盖失活的确定证据，不是“IC9激活后仍失败”的证据。未运行新求解或使用GT。
- 结论边界：本轮IC8确实激活P但未超过B0+S2-J，支持“逐边IC8直接替换当前S2未获任务收益”；旧IC9分支的鲁棒分解尚未得到有效激活检验。原论文全文PDF/作者Algorithm1未在本地直接核对，冻结公式与原文是否逐项一致仍待原件确认；不能凭当前实验断言论文整体不适合天线阵列。
- 下一具体动作：提交现有证据支持的判断和不确定性，等待用户确认任何后续实验；当前不执行P2B/P3/G1/边权扰动。
- 停止：小图不通过即阻塞；未收敛不入性能排名。用户授权步数调整已取代旧30k硬上限及机械冷启动策略，但不授权改目标或放宽门槛。若60000仍失败，先诊断不盲目加步。eta若后续确需调整，单独记录依据和时点；不把后验GT挑选冒充预注册结果。全部交付后等待用户审核。
