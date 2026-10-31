# Stage-2 P1 数学 sanity（2026-10-19）

**范围：P1 合成小图；已停止，未执行 P2A。** P0 已获用户确认，原实验事实不变。本轮不重新训练 G1，不拟合真实 B0/G1，不做真实数据性能结论。

## 1. 结果与停止边界

| 家族 | 状态 | 解释 |
|---|---|---|
| Edge | FAIL | 存在未通过的约束/求解器完成 gate，原始失败结果保留，禁止进入该分支 full-N |
| Framelet | PASS | 参考目标、可行性及数值 gate 均通过 |
| IC | PASS | 参考目标、可行性及数值 gate 均通过 |

总计 82 项检查，9 项 FAIL。完整逐项实际值、门槛及证据：[gates.md](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/gates.md) / [gates.csv](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/gates.csv)。
SEF 的数学参考门槛与求解器自身停止状态分开报告；即使前者通过，也不将预算用尽的 NONCONVERGED 标为完整收敛成功。联合 FE/DoT 没有全局凸 reference，只检查固定 U 的 S/E 块和固定 S 的 U 块。

## 2. 固定对象、公式与实现

- 图：固定 RNG=20261019+N 的加权 path+two-hop，IC/FN N=16/32/64，Edge N=256。合成干净弧线仅离线评价和绘图；求解器只读加噪观测 X 和图。未从真实 UE 坐标生成输入。
- IC：无向正边均值归一化 W；alpha=0.5，beta=0.5，eta=P0 delta0=0.09517167367990964。IC8 保留平方项；IC9 保留 D.T(WDU-P) 的非平方 L2/Frobenius 范数。标量有 box，group 仅共用观测球。
- Framelet：nu=0.1 仅数学 smoke。active=[low2,high1,high2]；高通尺度权重1和0.25，低通惩罚0但 Q_low=T_low U 约束存在。保存 objective 为 per-degree/per-band 定义。每次 U-step PCG 解 D_G+rho*sum(T.T T)，未用 D_G+rho*I 近似。
- FN-G-delta 最多8次新求解调用（缓存 same-nu 不重算），按实际移动量无标签匹配。匹配是有限预算近似，误差见下表；没有据 GT 选 nu。
- SEF/FE/DoT 保留 signed S、diag=0；FE/DoT 保留 S1=1、lambda1=lambda2=1，DoT 只额外加 Framelet 项。P1 使用 nu=0.1 的合成数学 smoke，不能冒充尚未授权的 B0 nu_P(mid) 或 P3 512/1024 gate。
- PDHG 最大30000次，唯一数值重试120000次；FN ADMM 最大5000次，唯一重试10000次；Edge 最大5000次，唯一重试15000次。alpha/eta/nu 与数据规模不因结果改变。
- 独立参考：CVXPY/CLARABEL，gap/feas tolerance=1e-9，status 必须 optimal；原始参考数组、objective、solver状态和迭代数均保存。IC论文附件本地未找到，因此本轮核验的是任务冻结公式，不声称逐行复现其 Algorithm1。
- 代码：[ic_solver.py](D:/diffusion_refined_charting/methods/stage2_modern_graph/ic_solver.py)、[framelet_operator.py](D:/diffusion_refined_charting/methods/stage2_modern_graph/framelet_operator.py)、[framelet_solver.py](D:/diffusion_refined_charting/methods/stage2_modern_graph/framelet_solver.py)、[edge_small.py](D:/diffusion_refined_charting/methods/stage2_modern_graph/edge_small.py)；[reference.py](D:/diffusion_refined_charting/diagnostics/stage2_p1/reference.py) 是独立 CVXPY 目标。

作者滤波器代码固定到 commit 4db39904e4099d82cef686d4732c33150814ece4；仅临时将已移除的 np.trapz 指向等价 np.trapezoid 以执行比较，未修改作者源文件。算子与作者返回的三个实际 active bands 做数值一致性比较。来源 URL/hash 见 manifest。

## 3. Reference objective、可行性与最优条件

全精度明细：[reference_objectives.csv](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/reference_objectives.csv)；IC-P 两个坐标是两个独立标量目标，不与 group objective 直接相加排名。

| N | 方法 / 坐标 | native objective | reference objective | relative gap | feasibility | stationarity |
|---:|---|---:|---:|---:|---:|---:|
| 16 | ic8-p / 0 | 1.21521379 | 1.21521379 | 5.092e-11 | 1.893e-17 | 7.042e-10 |
| 16 | ic8-p / 1 | 0.949275661 | 0.949275661 | 1.590e-11 | 0.000e+00 | 1.480e-06 |
| 16 | ic9-g / 0 | 0.859808469 | 0.859808471 | 2.470e-09 | 0.000e+00 | 5.805e-07 |
| 16 | ic9-p / 0 | 0.727237988 | 0.727237988 | 1.162e-10 | 1.893e-17 | 6.673e-07 |
| 16 | ic9-p / 1 | 0.494210571 | 0.494210571 | 5.105e-11 | 0.000e+00 | 3.757e-09 |
| 32 | ic8-p / 0 | 0.574767976 | 0.574767976 | 6.717e-11 | 0.000e+00 | 1.260e-06 |
| 32 | ic8-p / 1 | 0.561248297 | 0.561248298 | 2.016e-10 | 1.401e-17 | 1.271e-06 |
| 32 | ic9-g / 0 | 0.460121582 | 0.460121584 | 2.995e-09 | 0.000e+00 | 4.836e-08 |
| 32 | ic9-p / 0 | 0.315144211 | 0.315144211 | 5.749e-11 | 0.000e+00 | 1.474e-08 |
| 32 | ic9-p / 1 | 0.344482829 | 0.344482829 | 2.469e-10 | 1.401e-17 | 1.111e-06 |
| 64 | ic8-p / 0 | 0.597117594 | 0.597117594 | 2.311e-10 | 0.000e+00 | 1.381e-06 |
| 64 | ic8-p / 1 | 0.358410201 | 0.358410201 | 7.743e-11 | 1.996e-17 | 1.202e-06 |
| 64 | ic9-g / 0 | 0.854007166 | 0.854007167 | 1.439e-09 | 0.000e+00 | 2.697e-07 |
| 64 | ic9-p / 0 | 0.822515246 | 0.822515246 | 7.286e-11 | 0.000e+00 | 1.411e-07 |
| 64 | ic9-p / 1 | 0.367815664 | 0.367815664 | 5.753e-11 | 1.996e-17 | 3.653e-07 |
| 16 | fn-g-delta / 0 | 0.0105991283 | 0.0105991283 | 5.299e-09 | 1.933e-07 | 3.855e-08 |
| 16 | fn-g-nu / 0 | 0.00820379797 | 0.00820379797 | 8.166e-10 | 8.439e-08 | 1.756e-07 |
| 16 | fn-p / 0 | 0.010019842 | 0.0100198405 | 1.479e-07 | 1.870e-07 | 4.920e-08 |
| 32 | fn-g-delta / 0 | 0.00689156282 | 0.00689156112 | 2.466e-07 | 1.967e-07 | 1.144e-08 |
| 32 | fn-g-nu / 0 | 0.00580748799 | 0.00580748699 | 1.732e-07 | 1.910e-07 | 3.695e-08 |
| 32 | fn-p / 0 | 0.00681455859 | 0.00681455583 | 4.040e-07 | 1.993e-07 | 5.225e-08 |
| 64 | fn-g-delta / 0 | 0.00717520985 | 0.00717520806 | 2.492e-07 | 1.916e-07 | 1.867e-08 |
| 64 | fn-g-nu / 0 | 0.0060713779 | 0.0060713779 | 5.159e-10 | 1.784e-07 | 2.779e-08 |
| 64 | fn-p / 0 | 0.00700372224 | 0.00700372011 | 3.040e-07 | 1.748e-07 | 8.623e-08 |

## 4. Framelet 近似、零模态与移动量匹配

| N | 最小 n | reconstruction error | highpass true-zero max | matched delta relative error |
|---:|---:|---:|---:|---:|
| 16 | 3 | 1.796435e-02 | 1.516063e-02 | 4.845203e-04 |
| 32 | 3 | 1.697682e-02 | 1.516063e-02 | 1.787935e-03 |
| 64 | 4 | 3.261385e-03 | 1.244775e-03 | 1.416872e-03 |

n=2 失败是滤波近似精度不足；N=64 的 n=3 也未过 reconstruction 门槛，最小通过阶为4。零模态使用 sqrt(degree)*1；普通1响应仅作辅助，未参与 gate。PCG 每步真实残差及与稠密同目标 U-step 的误差保存在 checks.json/history.csv。IC9-G、FN-G-nu/FN-G-delta 的四种 gauge 等变性均单独保存。

## 5. Edge/DoT 数值状态及 BLOCKER

| 方法/初始化 | iterations | converged | equality | row sum | stationarity | S-block ref / gap | U-block ref / gap |
|---|---:|---|---:|---:|---:|---|---|
| dot-p-small-author | 15000 | False | 2.807908e-03 | 2.374957e-02 | 4.185457e-04 | 256.031185 / 8.873e-04 | 260.069649 / 1.987e-04 |
| dot-p-small-prior | 15000 | False | 3.275172e-03 | 3.061898e-02 | 6.858617e-04 | 256.030769 / 6.374e-04 | 260.012343 / 1.715e-04 |
| fe-p-small-author | 15000 | False | 2.573480e-03 | 1.953216e-02 | 3.424611e-04 | 256.003435 / 7.706e-04 | 256.489017 / 1.553e-04 |
| fe-p-small-prior | 15000 | False | 3.773182e-04 | 3.098536e-03 | 3.459784e-04 | 256.000821 / 1.538e-04 | 256.39067 / 6.964e-05 |
| sef-p-small-author | 15000 | False | 3.867752e-04 | 0.000000e+00 | 3.943185e-05 | 190.698794 / 5.338e-04 | N/A（固定U） |

约束门槛1e-3；native objective 和将 E 消去后的 objective 分列保存。后者只使自表达等式精确，**不自动使原 S 的行和精确**，不能称全约束可行值。固定 U 的凸参考解强制 diag/row-sum，固定 S 的凸参考继承该 S 的行和误差；目标接近不能消除最终约束 FAIL。
两个初始化的输出差异见 [initialization_sensitivity.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/initialization_sensitivity.json)，冻结S0/学习S的数值对照见 [fixed_vs_learned_S.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/fixed_vs_learned_S.json)。这只是实现诊断，不构成学到无线物理边或性能提升的证据。SEF 固定U，不进图谱性能表。
诊断：在固定5000+15000迭代预算下约束收敛慢；没有 OOM/NaN、PCG失败或参考解失败证据。两次预算和全部轨迹保留，未追加第三次、修改目标、放宽阈值或缩小数据。当前失败只说明该求解实现/数值配置未通过，不否定论文数学方法。末尾1000步的残差波动范围见 edge256/convergence_diagnostics.json。下降的比较基准统一为可行的先验 S0、U=X、E=X-S0X，并非把不满足行和的作者零初始化当成可行初值。

## 6. 日志、Chart 和完整指标

以下均为 **P1 synthetic math**，非真实天线阵列性能。字段保留米制命名以复用规范，但合成位置单位是人为设定的坐标单位，不可解读成真实定位米数。CT/TW/KS 用 seed10 的20%节点，N=16时仅3个指标节点，仅保留工程一致性，不作质量结论。误差在同一全部节点仿射拟合/评价。

- ic/n16：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n16/comparison/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n16/comparison/comparison.csv) / [raw图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n16/comparison/5_column_raw_channel_chart.png) / [affine图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n16/comparison/5_column_affine_channel_chart.png)。
- ic/n32：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n32/comparison/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n32/comparison/comparison.csv) / [raw图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n32/comparison/5_column_raw_channel_chart.png) / [affine图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n32/comparison/5_column_affine_channel_chart.png)。
- ic/n64：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n64/comparison/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n64/comparison/comparison.csv) / [raw图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n64/comparison/5_column_raw_channel_chart.png) / [affine图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic/n64/comparison/5_column_affine_channel_chart.png)。
- framelet/n16：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n16/comparison/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n16/comparison/comparison.csv) / [raw图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n16/comparison/5_column_raw_channel_chart.png) / [affine图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n16/comparison/5_column_affine_channel_chart.png)。
- framelet/n32：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n32/comparison/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n32/comparison/comparison.csv) / [raw图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n32/comparison/5_column_raw_channel_chart.png) / [affine图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n32/comparison/5_column_affine_channel_chart.png)。
- framelet/n64：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n64/comparison/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n64/comparison/comparison.csv) / [raw图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n64/comparison/5_column_raw_channel_chart.png) / [affine图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n64/comparison/5_column_affine_channel_chart.png)。
- edge256：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/comparison/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/comparison/comparison.csv) / [raw图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/comparison/7_column_raw_channel_chart.png) / [affine图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/comparison/7_column_affine_channel_chart.png)。

图谱直接复用 run_adp_siamese.plot_chart 和原 compare_paths；GT首列，同一样本与颜色，未另写散点格式。Edge 未收敛图明确标记 NONCONVERGED，仅作失败诊断，不进行有效方法排名。
收敛轨迹：各方法目录 attempt*_history.csv（IC为 axis*_attempt*_history.csv）；控制台 [ic_console.log](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/ic_console.log)、[framelet_console.log](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet_console.log)、[edge_console.log](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge_console.log)。完整JSON/CSV一致性核对：[artifact_consistency.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/artifact_consistency.json)。

## 7. 资源、测试、复现与保护

正式调用实际墙钟时间（包含参考求解与出图）依次为 [9.545380399969872, 26.8746872000047, 735.9366339999833] 秒；不能作为单独算法训练耗时。CPU float64，本轮未使用GPU，GPU显存分配0；全流程最终CPU峰值RAM未完整记录，标未知。Edge运行中观察到peak working set 868552704 bytes（仅截至采样时刻的下界），见 [resource_observation.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/resource_observation.json)。无模型训练/checkpoint。
5项相关单元测试通过：[unit_tests.log](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/unit_tests.log)。运行环境、输入/代码/作者文件hash与命令：[manifest.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/manifest.json)。AGENTS及P0 delta0/manifest前后hash一致；原baseline源与历史结果未写入。

复现（仓库根目录，使用空的本轮输出目录/独立副本；入口拒绝覆盖已存在家族结果）：

```powershell
D:/anaconda3/envs/diffusion_charting/python.exe -m unittest tests.test_stage2_p1_math -v
D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.run --family ic
D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.run --family framelet
D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.run --family edge
D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.report
```

当前分支 codex/new-stage1，HEAD f63652fc3bec725f4fddd1105d13e288cbf233dc，本轮未提交/上传。[git_diff_stat.txt](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/git_diff_stat.txt) 是全工作区既有差异，包含本轮之前的删除/修改；本轮新增未跟踪代码不会显示在该stat中，以 manifest.final_code/test_source 为准，不把既有改动记为本轮工作。

**阶段状态：P0 PASS（用户已确认）；P1 已执行并交付，Edge 存在 BLOCKER；P2A/P2B/P3/P4/P5 均 STOP（未授权/未执行）。等待用户审查，本轮不继续实验。**
