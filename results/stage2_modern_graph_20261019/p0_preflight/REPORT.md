# P0 — Repo & Baseline Preflight

日期：2026-10-19。**P0 完成，G0 的 14 项全部 PASS，无 BLOCKER。已停止，等待用户审核；P1–P5 未执行。**

唯一规格：`docs/tasks/Stage2_2026-10-19.md`，以用户本轮“仅 P0、完成即停止”覆盖旧的自动进入 P1 表述。适用 AGENTS 仅仓库根目录，已读取且运行前后 SHA256 一致，没有修改。未实现/运行 Robust-IC、Framelet、Edge-aware DoT；S2-CG 仅是原二次目标的数值控制。

## G0 逐项验收

| G0 项 | 状态 | 实际值 | 冻结门槛 | 证据 | 复现命令 |
|---|---|---|---|---|---|
| N_and_shapes | PASS | N=20997; 3 embeddings=(20997,2); D_geo=(20997,20997) | N=20997; all embeddings=(20997,2); D_geo=(20997,20997) | [input_checks.json](input_checks.json) | R0 |
| indices_and_positions | PASS | 3组 indices/positions 与 distance_data 逐元素相同 | all arrays elementwise identical to distance_data | [input_checks.json](input_checks.json) | R0 |
| finite_noncollapsed_inputs | PASS | 所有输入 finite; 各坐标 std 最小156.998428 | finite embeddings/D_geo/positions; nonzero embedding std | [input_checks.json](input_checks.json) | R0 |
| original_topk_self_precondition | PASS | 对角=0；非对角零值=0 | zero diagonal; no non-self zero ties before original topk[:,1:] | [input_checks.json](input_checks.json) | R0 |
| evaluation_subset | PASS | 4199 个节点，原 seed10 子集逐元素相同 | same fixed 4199 seed10 evaluation indices | [input_checks.json](input_checks.json) | R0 |
| G128_structure | PASS | 对称差0；自环0；min(w)=0.0826393；min degree=128；m=1544167 | symmetric; no self loops; finite positive weights/degrees; each undirected edge once in D | [graph_checks.json](graph_checks.json) | R0 |
| DTWD_matches_CSR_L | PASS | 5.344658883611e-16 | <1e-6 (float64 relative matvec) | [graph_checks.json](graph_checks.json) | R0 |
| L_times_one | PASS | 0.000000000000e+00 | <1e-8 max absolute | [graph_checks.json](graph_checks.json) | R0 |
| S2_CG_linear_residual | PASS | 9.950824665323e-11; info=[0,0] | <=1e-7; CG exits converged | [solver_checks.json](solver_checks.json) | R0 |
| delta0_auditable | PASS | delta0=0.09517167367991; 原始/归一化差=2.776e-17 | finite scalar; saved raw-J anchor, input/output SHA256, formula and gauge | [delta0.json](delta0.json) | R0 |
| Jacobi_history_ks | PASS | 1.398393956187e-09 | absolute difference <= 0.0002 | [historical_comparison.json](historical_comparison.json) | R0 |
| Jacobi_history_mean_m | PASS | 0.000000000000e+00 m | absolute difference <= 0.002 | [historical_comparison.json](historical_comparison.json) | R0 |
| Jacobi_history_p95_m | PASS | 5.960464477539e-07 m | absolute difference <= 0.003 | [historical_comparison.json](historical_comparison.json) | R0 |
| source_inputs_and_AGENTS_unchanged | PASS | 29 个文件 SHA256 全相同；源目录列表不变 | all SHA256 equal; no source distance-run entries added | [immutability_checks.json](immutability_checks.json) | R0 |

每项完整实际值、原执行命令和复现命令均在 [G0.json](G0.json) / [G0.csv](G0.csv)，上表 R0 为下方同一 P0 入口（使用新目录，不覆盖已有产物）：

```powershell
& 'D:\anaconda3\envs\diffusion_charting\python.exe' -m diagnostics.stage2_p0.preflight --device cuda:0 --out-dir results/stage2_modern_graph_20261019/p0_preflight_reproduce
```

目录必须尚不存在。原实际命令保留在 [run_command.txt](run_command.txt)。本轮正式 full-N P0 仅运行一次，无失败、重跑或门槛放宽。

## 输入、图与数学对应

B0、G1-Identity、G1-Graph 的真实绝对路径、shape、dtype、索引与 positions 一致性见 [input_checks.json](input_checks.json)。D_geo=(20997,20997) float32；distance_data 中 indices=(20997,)、positions=(20997,2)。原始结果与训练配置来源保存在 [input_provenance.json](input_provenance.json)，没有用 GT 构图或求解。只对 B0 重新求解两种数值 baseline，G1 仅做输入检查。

全部29个只读输入/来源/代码文件的完整 SHA256 见 [input_file_hashes.csv](input_file_hashes.csv) 和 [manifest.json](manifest.json)，含 AGENTS、三份 embedding/原结果、两份 G1 best 权重与配置、D_geo、distance_data、共同评价子集、历史 baseline JSON 和原 Stage-2/指标/绘图代码。前后逐文件内容核验见 [immutability_checks.json](immutability_checks.json)。原 distance-run 未新增或覆盖缓存；新 top512 缓存在 `graph/`。

调用原 `compute_neighbor_indices` / `build_local_scaled_graph`，得到 G128=3,088,334 个对称 CSR entry；**无向边只有 1,544,167 条，每条在 D 中只记一次**。D.shape=(1544167,20997)，每行两个元素 -1/+1；无自环、边权严格为正、图有1个连通分量。权重 min/median/mean/max=0.082639/0.511642/0.559208/0.999999；无权节点度 min/median/max=128/140/397；加权 degree min/median/max=56.869054/80.212837/177.251900。

原 W 以 float32 保存，检验时原样转 float64；不重缩放或归一化图权。CSR 图、关联矩阵、未归一化 L 及 hash/统计分别在 `graph/G128.npz`、`graph/incidence_D.npz`、`graph/L_csr_float64.npz`、`graph/edges_and_degrees.npz`、[graph_checks.json](graph_checks.json)。四个固定 seed10 随机向量检验 DᵀWD 与独立 CSR L 的 matvec，误差 5.345e-16；L1 最大差0。

小例子：两节点边权2时，D=[-1,+1]，L=[[2,-2],[-2,2]]；lambda0.15 下 Y=[0,1] 的解为 [0.1875,0.8125]。本次按实际 Jacobi 更新对应的 `(I+0.15L)Z=Y` 求解，目标为 `0.5||U-X||² + 0.15/2 ||sqrt(W)DU||²`；不因邻接双向保存而给正则多乘2，也不修改原函数以匹配历史数值。

## S2-J 与 S2-CG：同一二次目标

S2-J 原函数不变：top_k512、graph_k128、lambda0.15、float32 Jacobi200。S2-CG：CPU SciPy float64，在无标签 gauge `X=(Y-c)/a` 上解同一个系统，再保存 `Z=c+aU` 为 float64；c=mean(Y)，a=233.503696422285。预条件为系统正对角的逆，rtol1e-10、atol0、maxiter5000，没有新正则项、训练或 GT 调参。配置与日志见各自 `run_config.json`、[solver_checks.json](solver_checks.json)、[cg_residual_history.json](cg_residual_history.json)。

CG 两坐标均 info=0，各42次；原始与归一化坐标的显式真实相对线性残差最大 9.950824665323e-11，严于1e-7。Jacobi200 对同一个 float64 系统的相对残差为 2.345189742450e-06。CG 相对 J 的 raw 最大差0.000203436、每节点位移 RMS=0.000042516，都是原始 chart 单位而非米。

历史参考使用原 `results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/result.json`（SHA256在 manifest 中）。历史与本次 J 的 KS/Mean/P95 差分别为 1.398393956187e-09、0.000000000000e+00 m、5.960464477539e-07 m，均通过固定复现阈值。

## 统一 Channel Chart 指标

CT/TW/KS 均复用同一 seed10 的4199个节点；误差在全部20997节点分别按原函数仿射拟合并评价。GT 只在此处及图片中使用。所有指标从当前保存 embedding 重算，source_json 均存在；完整表格/CSV和协议位于 `comparison/`。

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [result.json](D:/diffusion_refined_charting/results/adp_old_siamese/stride1_n20997_seed10/result.json) | unavailable | unavailable |
| B0+S2-J | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558675 | 0.634640 | 1.141431 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p0_preflight/s2_j/result.json) | unavailable | unavailable |
| B0+S2-CG | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558674 | 0.634639 | 1.141432 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p0_preflight/s2_cg/result.json) | unavailable | unavailable |

CG 相对 J 的 Mean 改变仅约 -2.91e-8 m，P95 约 -7.45e-7 m，属于当前数值求解差异，不能包装为新方法收益。两者都保留，后续是否启动任何方法等待用户审核。

[原始四列图](comparison/four_column_raw_channel_chart.png) · [仿射四列图](comparison/four_column_affine_channel_chart.png) · [完整 CSV](comparison/comparison.csv) · [比较表](comparison/comparison.md)。图片直接复用原 `plot_chart` 格式，已实际检查。

## B0 的固定修正预算 δ₀

**δ₀=0.09517167367991**，由本轮保存的原 Jacobi200 raw 输出计算，不由 CG 或 GT 评价改选：

- numerator=||Z_J−Y_B0||F=3220.179574061183；
- denominator=||Y_B0−mean(Y_B0)||F=33835.483285621252；
- 在 gauge 中重算 delta0=0.09517167367991，与 raw 结果差3e-17。

[delta0.json](delta0.json) 包含完整公式、源 B0 和 J 输出 hash、路径、c/a、冻结参数；[chart_gauge.npz](chart_gauge.npz) 保存 c、a、X、U_J 和 U_CG。后续锚点仍为原 J，未因 CG 数字改变。

## 运行与停止状态

正式 P0 全部计算、输入hash及评价约 31.30 s；J 求解0.477 s，CG0.663 s。构图/neighbor阶段实际 PyTorch 峰值分配1.805 GiB；J稀疏求解阶段0.104 GiB；CG在CPU，不把GPU缓存显存写成CG成本。GPU为RTX3050 Laptop4GB；Python/NumPy/SciPy/Torch及平台版本见 manifest。模型没有训练或参数更新。

新增代码仅 `diagnostics/stage2_p0/{__init__.py,preflight.py}` 和 `tests/test_stage2_p0.py`。3项P0测试全部通过：incidence/Laplacian、两节点解析解系数、gauge-CG与稠密原坐标解等价；没有运行P1公式/reference测试。命令：

```powershell
& 'D:\anaconda3\envs\diffusion_charting\python.exe' -m unittest tests.test_stage2_p0 -v
```

原 baseline、历史 embedding/checkpoint/result、source distance cache 与 AGENTS 未改。`git_diff_stat.txt` 是工作树整体快照，主要包含运行前已经存在的修改/删除；新P0代码尚未跟踪，不会计入普通 `git diff --stat`。本轮文件清单见下，不能把既有用户改动计作本轮改动：

- 新增P0诊断包和3项测试；
- 在当前 Stage2 task 检查点记录 P0 scope/事实，并同步 research_state；
- 记录P0基线与审计产物的 log/registry；新结果仅写入本轮根目录。

当前分支 `codex/new-stage1`、起始HEAD `f63652f`，本轮未提交或上传。这里的“提交P0结果”为本地报告交付，不执行Git提交。

| 阶段 | 状态 |
|---|---|
| P0 / G0 | PASS；14/14验收通过 |
| P1 | NOT RUN；等待用户审核 |
| P2A / P2B | NOT RUN |
| P3 / P4 / P5 | NOT RUN |

本轮只检查未归一化 L 的 L1=0。没有构造 Framelet、L_sym 滤波器或执行 P1；将来 P1 的零模态按 D_G^(1/2)1（多连通分量时逐分量）核验，普通常数向量不能作为唯一硬门槛。

**停止：仅P0结果交付完成，等待用户审核，不自动进入P1。**
