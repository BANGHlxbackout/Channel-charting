# P1-Edge Solver Repair

**数值求解诊断；无 Channel Chart、仿射图、CT/TW/KS 或定位指标。IC 和 Framelet Node 代码/结果未改、未重跑。**

## 1. 验收与停止

原256节点 B 策略 1/5 配置 PASS；最终状态 **NUMERICAL_BLOCKER**。仅完成本轮固定A/B，没有数值重试。P2A/P2B/P3/SparseDoT/full-N均未执行，已停止等待审核。

| 方法/初始化 | A状态 / 次数 | B状态 / 次数 | B primal | B dual | B KKT | B最大约束 | B最大block gap | A/B秒 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SEF/author | NUMERICAL_BLOCKER / 15000 | NUMERICAL_BLOCKER / 15000 | 5.654260e-05 | 3.627826e-05 | 3.627825e-05 | 4.129758e-04 | 5.713346e-04 | 13.881/15.352 |
| FE/author | NUMERICAL_BLOCKER / 15000 | NUMERICAL_BLOCKER / 15000 | 6.038890e-04 | 1.047649e-03 | 2.644243e-04 | 2.025511e-02 | 1.588036e-03 | 66.308/39.876 |
| FE/prior | NUMERICAL_BLOCKER / 15000 | PASS / 8180 | 3.961215e-07 | 1.672208e-06 | 9.058880e-07 | 1.776009e-05 | 6.666556e-07 | 62.417/19.285 |
| DoT/author | NUMERICAL_BLOCKER / 15000 | NUMERICAL_BLOCKER / 15000 | 6.046447e-04 | 2.290797e-03 | 3.329532e-04 | 2.113828e-02 | 1.643445e-03 | 209.247/111.432 |
| DoT/prior | NUMERICAL_BLOCKER / 15000 | NUMERICAL_BLOCKER / 15000 | 8.330034e-04 | 8.948223e-04 | 2.043443e-04 | 3.338486e-02 | 2.046810e-03 | 205.970/111.687 |

逐项实数/阈值/证据：[gates.csv](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/gates.csv)；全部32/64/256配置：[comparison.md](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/comparison/comparison.md) / [comparison.csv](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/comparison/comparison.csv)；reference目标全精度见reference_objectives.csv。每配置独立保存solver.json、state.npz（U/S/E/Y/Q/所有unscaled multiplier）、逐10步history.csv和penalty_history.json，以及凸reference objective/solver状态/数组。

按每个方法要求的全部初始化汇总：**SEF FAIL / FE FAIL / DoT FAIL**。FE仅先验初始化通过，不构成整个FE分支通过。

## 2. 原目标与增广拉格朗日核查

用S表示论文Z。冻结目标为 `||S||1 + sum_i d_i||E_i||2 + 0.5 sum_i d_i||U_i-X_i||² + (nu/1.25) sum_h c_h sum_i d_i |T_h U_i|1`；SEF固定U且去掉fidelity/Framelet，FE去掉Framelet，DoT保留四项。d为原始加权邻接行和，不除以均值、N或d_sum；lambda1=lambda2=1，DoT nu=.1，c=(1,.25)，active=(low2,high1,high2)，low2权重0但保留约束。
令 `c1=U-YU-E, c3=Y1-1, c4=Y-S, c2_b=T_b U-Q_b`，增广项为 `sum_j <Lambda_j,c_j> + mu_j/2 ||c_j||²`。SEF没有c3，S始终diag=0且可有负值。Y不单独强制diag，Y=S在极限传递此约束。消去E使自表达精确，但不自动使S行和可行；equality_eliminated_objective不能冒充全约束可行值。

更新方程：

- `Y (mu4 I + mu1 UU.T + mu3 11.T) = (mu1(U-E)+Lambda1)U.T + (mu3 1-Lambda3)1.T + mu4 S-Lambda4`，SEF去掉行和项；Woodbury与独立凸解一致。
- `S = offdiag soft(Y+Lambda4/mu4, 1/mu4)`。
- `U` 由PCG精确解 `D + mu1(I-Y).T(I-Y) + mu2 sum_b T_b.T T_b`，RHS为 `DX+(I-Y).T(mu1 E-Lambda1)+sum_b T_b.T(mu2 Q_b-Lambda2_b)`。FE无Framelet项，SEF不更新U；没有tight-frame近似。
- `E_i = group_soft(((I-Y)U+Lambda1/mu1)_i, d_i/mu1)`。
- `Q_b = soft(T_b U+Lambda2_b/mu2, nu*c_b*d/(1.25*mu2))`，低通阈值0；该c2方向与论文Q-TU相反，所以Lambda2的符号也相反，数学等价。
- 按Y→S→U→E→Q的现有Gauss-Seidel顺序完成后，`Lambda_j += mu_j*c_j`。此顺序各步仍最小化同一增广目标；不把多块非凸ADMM写成全局最优保证。

核查结论：没有发现本地更新公式、矩阵逆、soft-threshold或对偶符号的实现错误。作者代码的E阈值未乘d、部分对偶更新使用S而非Y，不能照抄来替代本轮冻结目标。相应论文目标参见 [Eq5/Eq6/Eq17](https://arxiv.org/html/2111.03264v1)，[固定作者源码](https://github.com/bzho3923/GNN_DoT/blob/4db39904e4099d82cef686d4732c33150814ece4/denoising_filters.py)。

## 3. 独立reference与修改边界

32/64节点，在第三次sweep的非零对偶状态，独立CVXPY逐块构造并求解Y/S/U/E/Q增广子问题。初次tol1e-9在64节点soft-threshold附近参考解的向量误差为几e-6；仅收紧reference至1e-12，原始证据仍在initial_reference_tolerance.json，未放宽1e-6向量误差门槛。最终全部通过，最大向量差2.413588e-7，S增广objective相对差<=8.208e-14。32/64完整20个A/B配置也全部PASS。联合FE/DoT最终只比较固定U的S/E凸块和固定S的U凸块，不宣称全局凸reference。
证据：[augmented_reference_n32.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/augmented_reference_n32.json) / [augmented_reference_n64.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/augmented_reference_n64.json) / [unit_tests.log](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/unit_tests.log)。3项Edge-only单元测试通过；未调用IC/FN Node测试。

实际源码差异：[edge_small.diff](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/edge_small.diff)；当前代码[edge_small.py](D:/diffusion_refined_charting/methods/stage2_modern_graph/edge_small.py)。原源码git blob/hash在manifest，diff可恢复修改前内容。修改仅含分量日志、完整残差/停止条件、单一penalty策略及预算截止，不改变原更新顺序或目标。A/B共享所有诊断和停止条件。

## 4. Residual balancing与完整停止条件

A固定mu1=mu3=mu4=10。B初值相同，每50次、仅第100..3000次比较归一化primal和linearized dual drift：r>10s则mu乘2；s>10r则除2；范围[1e-3,1e4]，之后冻结。mu2始终原degree.mean。变mu时scaled dual乘old_mu/new_mu，保留unscaled multiplier；全批次最大保持误差见constraint_and_dual_identities.json。此规则在正式A/B前已固定，没有GT/性能调参。

双线性多块问题没有唯一的两块ADMM对偶残差。本轮明确记录：s4=mu4||Delta S||；s2=mu2||sum T.T Delta Q||；s1同时记录对Y及U stationarity的线性化block drift。行约束只有Y这一个变量块，canonical两块s3会恒零，故使用 `mu3||(Delta Y)11.T||` 作为**行约束image-drift代理**。它仅用于预先固定的数值平衡，不能当作独立收敛证明。各项按相应lambda stationarity尺度归一化，具体公式见源码history记录处。
最终收敛必须满足原equality/S-row/split/frame/progress/stationarity全部<2e-5，并增加Y-self、Y-row、完整prox(S/E/Q)、U/Y迭代变化及primal/linearized-dual<2e-5。prox检查对应S=prox(S+Lambda4)、E=prox(E+Lambda1)、Q=prox(Q+Lambda2)。硬G1所有约束<=1e-3、block gap<=1e-3、stationarity<=5e-3不变；finite且中心化spread>=输入0.05，PCG<1e-5。达到15000上限不算收敛。

## 5. 根因证据与分量量级

旧日志没有保存Y、multiplier、Q，因此不能凭空重建缺失轨迹。本轮A从同一输入/初始化重放固定penalty，补齐全部轨迹，并将最终U/S/E与原P1 attempt2逐项比较（差异见各result.json的legacy_relative_differences）。旧P1实际耗时包含两个attempt及reference，不能与本轮纯solver_time直接计算加速比；A/B本轮相同日志开销可比较。
旧FE/DoT的L1项约256，FE fidelity约0.33–0.37、DoT fidelity约1.56、Framelet约2.30，E项为0。目标总值与凸block参考接近时，行和仍可有百分之几误差，故relative objective不能代替可行性。SEF的L1约190.18、group_E约0.52。实际全精度分量见legacy_diagnosis.json和各history.csv。
结构上 `(S1-1)-(Y1-1)=(S-Y)1`：Frobenius相对split很小也不自动保证S每行和很小，需单独监控。A/B末尾1000步残差范围见tail_residual_diagnosis.json，逐项约束身份/penalty rescale数值误差见constraint_and_dual_identities.json。这些检查用于判断哪条约束停滞；不会将泛化或图谱质量变化混入数值诊断。

本轮能检验的是该固定数据/初始化/预算下，完整更新正确后，一种penalty平衡能否消除阻塞。若B通过，说明该个案不需要改目标即可求解；若B仍FAIL，只能定位为当前数值策略/预算下的NUMERICAL_BLOCKER，不能证明目标本身不可求解。没有追加第三策略或重试。


| 原256配置 | policy | U-YU-E相对 | Y-S相对 | Y行和max | S行和max | Frame split |
|---|---|---:|---:|---:|---:|---:|
| SEF/author | fixed | 2.905916e-06 | 5.261348e-05 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| SEF/author | residual_balance | 7.819917e-07 | 5.654260e-05 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| FE/author | fixed | 1.965576e-05 | 5.300757e-04 | 7.708706e-05 | 1.953216e-02 | 0.000000e+00 |
| FE/author | residual_balance | 3.196600e-04 | 6.038891e-04 | 8.048345e-05 | 2.025511e-02 | 0.000000e+00 |
| FE/prior | fixed | 4.251411e-06 | 7.437056e-05 | 1.220162e-05 | 3.098536e-03 | 0.000000e+00 |
| FE/prior | residual_balance | 3.961215e-07 | 3.779872e-07 | 1.387545e-07 | 1.776009e-05 | 0.000000e+00 |
| DoT/author | fixed | 2.129049e-05 | 5.774934e-04 | 9.427939e-05 | 2.374957e-02 | 6.811806e-05 |
| DoT/author | residual_balance | 3.702832e-04 | 6.046448e-04 | 8.365077e-05 | 2.113828e-02 | 1.374632e-04 |
| DoT/prior | fixed | 2.066034e-05 | 6.165351e-04 | 1.204325e-04 | 3.061898e-02 | 6.447749e-05 |
| DoT/prior | residual_balance | 4.768016e-04 | 8.330037e-04 | 1.316048e-04 | 3.338486e-02 | 4.477990e-05 |

具体定位：A的FE默认初始化，Y行和仅7.708706e-05、Y-self仅1.965576e-05，但S行和1.953216e-02。其split传播至S约束是主要可见阻塞；不能将它说成Y行约束本身完全没有收敛。B的FE先验初始化状态为PASS，说明单一penalty策略能影响该个案；并不保证其默认初始化或整家族通过。

L1(S)>=N 对精确行和约束成立（逐行三角不等式），非负系数达到等号；这里没有添加非负约束。二维自表达可有许多结构近似等价解，因此L1大项/局部目标接近不足以认证收敛。该事实只解释数值目标的平坦性，未被用来更改目标或设计新方法。

根因边界：独立增广子问题和A的原轨迹重放排除了已核验范围内的更新代数错误；旧停止条件的诊断覆盖不完整已补齐，但旧S行和检查已正确阻止了假成功，所以原P1 FAIL结论仍有效。残差比例平衡本身不能保证绝对残差趋零。SEF是凸问题且CVXPY给出optimal参考，不能将当前ADMM未达停止门槛称为目标不可解；FE/DoT整体双线性非凸，本轮仍不足以唯一归因于penalty、变量更新顺序或非唯一解。保留NUMERICAL_BLOCKER，不强行给出单一因果结论。

## 6. 数值收敛图与资源

- [n256/DoT-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n256/DoT-author/convergence.png)
- [n256/DoT-prior](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n256/DoT-prior/convergence.png)
- [n256/FE-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n256/FE-author/convergence.png)
- [n256/FE-prior](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n256/FE-prior/convergence.png)
- [n256/SEF-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n256/SEF-author/convergence.png)
- [n32/DoT-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n32/DoT-author/convergence.png)
- [n32/DoT-prior](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n32/DoT-prior/convergence.png)
- [n32/FE-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n32/FE-author/convergence.png)
- [n32/FE-prior](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n32/FE-prior/convergence.png)
- [n32/SEF-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n32/SEF-author/convergence.png)
- [n64/DoT-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n64/DoT-author/convergence.png)
- [n64/DoT-prior](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n64/DoT-prior/convergence.png)
- [n64/FE-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n64/FE-author/convergence.png)
- [n64/FE-prior](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n64/FE-prior/convergence.png)
- [n64/SEF-author](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/n64/SEF-author/convergence.png)

正式阶段墙钟（含独立reference与IO）：{'full': 892.7399932999979, 'small': 130.89132649998646} 秒。每配置纯solver时间见comparison。CPU float64，未用GPU；RAM峰值未完整记录，不冒称已测。每配置最多15000次、无重试，全批次计算上限3600秒，A/B无lambda/nu/图/数据规模搜索。

[manifest.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/manifest.json) 保存冻结策略、清单中旧IC/FN/Edge结果和作者源码的前后hash、环境/代码与复现信息；所有保护文件hash一致。[git_diff_stat.txt](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/git_diff_stat.txt) 是已有整个工作区差异，含历史改动，不将其当成本轮修改；本轮edge改动见专用diff。分支codex/new-stage1，HEAD f63652fc3bec725f4fddd1105d13e288cbf233dc，未提交/上传。

复现（独立空输出目录/副本，入口拒绝覆盖当前结果）：

```powershell
D:/anaconda3/envs/diffusion_charting/python.exe -m unittest tests.test_stage2_edge_repair -v
D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.edge_repair --stage init
D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.edge_repair --stage small
D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.edge_repair --stage full
D:/anaconda3/envs/diffusion_charting/python.exe -m diagnostics.stage2_p1.edge_repair_report
```

**STOP：本轮数值诊断交付后等待用户审核；不执行P2A/P2B/P3/SparseDoT或新的full-N。**

交付复核：[delivery_checks.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_edge_solver_repair/delivery_checks.json)：保护hash、固定A原结果逐元素重放、CSV/JSON数值、仅数值图及导出兼容性均通过。
