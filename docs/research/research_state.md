# 当前研究状态（Research State）

- 版本：v3.14
- 更新日期：2026-10-27
- 当前范围：毕业论文第一个工作——信道图谱（Channel Charting）表示构建与改进。
- 当前进度：已具备基线、消融、诊断与多轮优化探索；具体创新机制尚未确定，尚未形成有充分机制证据和稳定收益的改进方法。
- P0已确认：2026-10-19 Stage-2 P0 完成，G0 14/14 PASS：输入/图 hash 可审计，原 J200 复现通过，DᵀWD/CSR matvec 相对误差5.344659e-16，L1=0，CG同二次目标真实相对残差9.950825e-11，δ0=0.09517167367991。结果见 [P0报告](../../results/stage2_modern_graph_20261019/p0_preflight/REPORT.md)。
- 当前任务：用户确认对 G1 Stage-1 的 `D_geo` 目标偏差、Graph/Identity 模型残差与最终误差做只读定位，见[当前任务与恢复检查点](../tasks/G1_graph_affinity_next_cycle.md)和[误差来源报告](../../results/g1_acf_graph_fcf/error_source_diagnostic/REPORT.md)。同一 399983 随机点对上 `D_geo` 与 GT 距离排序 Spearman 0.982725，但按全局比例换算的近/远端偏差分别 +0.4536/-0.3904 m；Graph/Identity 原生图谱对 `D_geo` 的残差方向部分抵消这类偏差。邻边目标误差与 Graph 节点误差的相关 -0.0257，随机点对目标误差相关 0.1705，模型目标拟合误差相关 0.0597；未找到可直接指导新 loss 的单一主要原因。GT 仅用于离线归因，未改变 ACF 预处理、fused 图、模型或 Stage 2。旧局部扩散 loss 和 ADP-affinity 的完整训练均暂停；如继续，应先判断 GT-free 的输入量能否在匹配距离与空间后识别目标偏差，再决定是否有可靠性加权的依据，不预设新公式。既有[梯度诊断](../../results/g1_acf_graph_fcf/loss_direction_diagnostic/REPORT.md)与[ADP-affinity 阶段报告](../../results/g1_acf_graph_fcf/adp_affinity/REPORT.md)保留各自边界。P2A-R 已完成并 STOP 的数值和结论不因本次范围变化而改写，见[旧报告](../../results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/REPORT.md)。
- 运行中授权与处置：用户允许eta灵活调整和调整步数。已找到多档激活点，未追加eta；eta=.50原30000步未达标来自慢对偶暂态，非发散。仅剩余3配置上限改60000，原完整停止阈值不变，实际32700/32850/32950步通过；失败尝试及人工预算更新中断均保留。旧P2A/P1/P0/B0/S2的1122保护文件hash未变，IC/FN G1与Edge NUMERICAL_BLOCKER既有结论不变。未执行P2B/P3或Stage-1训练。
- 最新用户纠正（2026-10-25）：后续性能参照仅B0+S2-J；CG只保留已完成的Jacobi数值核验历史，不再作为后续性能比较对象。用户要求先思考清楚，agent提出的4个基线预算匹配追加实验已撤回，尚未新增该诊断代码或启动任何运行。当前仅根据已完成结果判断参数覆盖与方法增量，不自动追加实验；原20配置和历史事实完整保留。
- 随后的只读核对：旧IC9-P/G共18个full-N状态P均为0；冻结Eq9的P=0最优条件显示在eta≈0.095时IC9-P临界alpha约.0132、IC9-G约.01057，旧网格最小alpha=.2；全部18个alpha超过各自临界值。证据见[只读审计](../../results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic9_zero_aux_parameter_audit.json)。因此旧IC9对该核心分解的激活测试不足，不能从其负性能结果断言论文方法在天线领域不适用。P2A-R的IC8实际激活且未超S2-J之事实不变；论文原件逐式对应尚未直接核对。

## 1. 已确认范围与历史安排

毕业论文题目为《信道图谱表示学习及其在无线感知中的应用研究》。用户于 2026-10-07 明确当前项目承担第一个工作，于 2026-10-08 明确第二个无线应用工作和毕业论文整体工作将在另外的项目开展。因此，本项目围绕精修基线（diffusion-refined channel charting baseline）寻找表示改进与创新点；其余工作仅作为背景。

技术流程的第一阶段（Stage 1）与第二阶段（Stage 2）都属于第一个工作。已确定的是研究范围，尚未确定的是具体创新机制；不把基线优化本身或工作流调整描述成新增科研成果。

[历史 v2.1](archive/research_state_v2.1_2026-09-03.md) 的多轮跨方向筛选与独立模型审批不是当前安排。本次修正 v2.2 停留在 9 月 13 日“下一任务尚未启动”的表述，纳入日志和任务已记载的后续探索，不重新解释其冻结结果。

## 2. 基线事实与核心证据

- Stage 1：信道状态信息（Channel State Information，CSI）→ 角度时延谱（Angle-Delay Profile，ADP）/时间戳融合 → 20 近邻图 → 测地距离 `D_geo` → 孪生前向图谱映射函数（Siamese Forward Charting Function，FCF）→ `Y0`。
- Stage 2：`Y0 + D_geo` → 局部尺度测地邻居图 → 图拉普拉斯（graph Laplacian）/ Tikhonov 精修 → `Z`。当前记录配置为 `k=128`、`lambda=0.15`、200 次迭代，近似求解 `(I + lambda L)Z=Y0`。
- 同源 `D_geo` 同时用于训练目标和精修图，是已确认事实；误差强化（error reinforcement）仍无支持结论。
- 当前可比组（comparable group）`CG-A-transductive-stride1-current` 的保存结果：参考 Siamese 平均误差 0.4093 m，baseline（Z）为 0.3187 m。它们属于传导式评估（transductive evaluation）、仿射对齐误差（affine-aligned error），不是未见样本或无线应用性能。
- 精修诊断报告记录约 68.25% 样本误差改善、31.75% 恶化，原本误差较小的组更容易恶化。这是评价协议下的观察；不能直接推断唯一原因，也不能据此认定已经找到有效保护机制。

精确运行数值与可比性以 [experiment_registry.md](experiment_registry.md) 和源文件为准，不用状态文档维护跨协议排行榜。

## 3. 已探索方向及证据状态

代码文件名、目录名和文件路径保持仓库中的原始英文写法，不作中文翻译；研究说明使用中文。

以下按实际完成范围归纳；不同协议和可比组的数值仅用于说明各自结果，不直接跨组排名。真实位置（ground truth，GT）辅助的诊断不构成可部署方法输入。

| 方向 | 已有结果与边界 | 当前定位 / 来源 |
|---|---|---|
| baseline 与欧氏图消融（Euclidean graph ablation） | 欧氏图精修平均误差 0.3944 m，弱于原图配置 0.3187 m；支持图选择影响效果，尚未确定具体原因 | 核心基线与历史消融；[experiment_registry.md](experiment_registry.md) |
| 精修收益与损害诊断 | 记录收益、恶化样本与局部压缩；整体改善和局部损害并存 | 问题线索；[diagnostic_summary.md](../../results/diffusion_refinement_diagnostics/stride1_n20997_seed10_k32/diagnostic_summary.md) |
| 节点保护与局部一致性代理（local consistency proxy） | 修订后的保护代理未通过预设判据；残差对异常点有一定辨别能力，但不等于能可靠决定保护对象 | 诊断资产，完整历史协议仍待核验；[local_consistency_proxy_diagnostics.md](../../results/local_consistency_proxy_diagnostics/stride1_n20997_seed10_k32_phys128_revised/local_consistency_proxy_diagnostics.md) |
| 直接支持感知边过滤（direct-support edge filtering） | 历史报告判定未通过验收；原图降低平滑强度的对照优于该过滤方案 | 保留负结果，历史协议待核验；[summary.md](../../results/direct_support_edge_refined_charting/summary.md) |
| 门控残差修正（repair-gated residual correction） | 保存的最佳门控平均误差 0.3234 m，未超过该实验的 baseline / 全局插值对照；邻域指标评估设置与原 baseline 文件有差异 | 保留负结果，不能直接跨文件排名全部指标；[result_summary.md](../../results/repair_gated_residual_correction/stride1_n20997_seed10_phys128/result_summary.md) |
| 平滑参数敏感性 | 已有 13 组 λ 运行；参数变化不等于机制创新，选择协议尚未复核 | 历史参数对照；[结果家族](../../results/lambda_sensitivity_refined_charting/) |
| 样本外扩展（out-of-sample extension，OOS）与交叉拟合谐波残差蒸馏（Cross-Fitted Harmonic Residual Distillation，CF-HRD） | 已有代码和留出评估（held-out evaluation）结果；保存的测试平均误差分别为 0.4250 m、0.4871 m，划分与信息使用等协议仍待核验 | 表示能力探索，标为 `UNVERIFIED / verification_required`；不属于第二个无线应用工作；[OOS](../../results/heldout/seed10/diffusion_oos/result.json)、[CF-HRD](../../results/CF_HRD/seed10/stage5_evaluation/result.json) |
| Stage-1 图溯源与 H1 删边反事实（counterfactual） | 已存溯源报告通过抽样距离复现；目标删边局部修复与附带损害（collateral degradation）并存，冻结低损害判据未通过 | 上游侧诊断，未证明能改进最终 Z；[H1_COUNTERFACTUAL_REPORT.md](../../results/phase2c_h1_h2/H1_COUNTERFACTUAL_REPORT.md) |
| SC²：Stage-1 共同邻居边长惩罚 | 固定 2100 样本多维尺度分析（Multidimensional Scaling，MDS）门槛实验中平均误差 0.5825 m，baseline 为 0.4581 m；停止于 FCF 之前 | 完成、负结果；只检验新增边长惩罚，不否定共同支持思想整体；[REPORT.md](../../results/sc2_stage1_consensus_pilot/REPORT.md) |
| SC²：Stage-2 自适应保真（adaptive fidelity） | 二阶一致性得分决定保留 Y0 的力度；平均误差 0.322221 m，均匀保真对照 0.318748 m，冻结判据未通过 | 完成、负结果；原 worktree 代码及结果已整合 master；[REPORT.md](../../results/sc2_stage2_adaptive_fidelity_pilot/worktree_run/REPORT.md) |
| SC²：Stage-2 边权干预与随机复验 | 定向选边 mean 0.318021 m，但 P95 恶化且未超过随机对照；兼容边随机降权 20 次的 mean 中位数 0.317755 m、P95 中位数 0.633537 m | 同一数据和 Y0 下的小幅稳定观察，未证明确定性选边机制或独立数据泛化；[REPORT.md](../../results/sc2_stage2_edge_consensus/REPORT.md)、[complete_metrics_followup.json](../../results/sc2_stage2_edge_consensus/complete_metrics_followup.json) |
| 多视角边不确定性 Gate 0 | 四阵列 ADP 求和、原 20-NN 图与 32 源点测地距离复现通过；`kappa=0` 图逐项一致 | 实现与数据回归，未检验边分数或 `Z` 收益；[REPORT.md](../../results/multiview_edge_uncertainty/REPORT.md) |

SC² 此处沿用实验名称，指二阶空间兼容性启发的共同支持思想；保留实验的对象和干预不同，不能把一次规则失败扩大为整个思想无效。

保留 SC² 实验的 [任务规格](../tasks/) 和 [experiment_log.md](experiment_log.md) 已记录完成状态。其局限包括单一随机种子、固定干预及全数据评估；原始复现误差、数值容差说明和对照限制以各报告为准，不静默省略或改写。

## 4. 未完成验证与停止边界

- H1 保留冻结标签：`H1_counterfactual_supported=false`、`H1_supported=null`。它不是“无作用”或“完全证伪”；H2 同源误差保留/放大诊断仍未执行，不自动补跑 H2、敏感性实验或修改阈值。
- 保留 SC² 实验的预设筛选判据均未通过；依原任务收尾，不因某个指标略好而翻转结论，不自动调整分数方向、相容阈值或干预强度。
- 多视角边不确定性目前只完成 Gate 0；Gate 1–3、Siamese 与最终 `Z` 的效果均未执行。其他多视图稳健边与 Huber 图总变差（graph total variation）精修、拓扑感知修复等材料主要保存在 `codex_idea/`、`notes/`，尚未定位完整方案已验证的结果。
- 早期训练充分性检查、直接蒸馏尝试在日志中仍为 `missing-records`（记录缺失），不能作为正式实验结论；不与已有完整 CF-HRD 资产混同。
- 文献、评价指标和已有优化失败提供了背景与排除性证据，尚未闭合“明确表示问题—有依据的机制—受控验证—创新贡献”的链条。当前没有自动选定新的优化路线。

## 5. 工作流与下一行动

每轮只处理一个明确的表示问题：

1. 说明问题的重要性、现象来源、与当前基线的直接关系，以及什么结果会改变方法决定。
2. 先检查已探索方案是否覆盖该问题；区分“问题存在”“评分能识别问题”和“干预能改进表示”，避免从诊断直接跳到方法。
3. 针对关键缺口核对最接近的已有工作（closest prior work），提出最小对照及可否定条件，不重新进行广泛赛道筛选。
4. 正式实验前固定输入、干预、对照、主要指标、损害指标、预算和停止条件；在授权范围内执行必要验证。
5. 完成后记录结论与限制，更新日志和索引；用户确认研究决定后同步本文件，避免任务已完成而状态滞后。

G1阶段历史任务为 [G1_S2_2026-10-17.md](../tasks/G1_S2_2026-10-17.md)，检查点记录 E1/E2 的授权、审查发现、输入、结果和停止边界。原 G1 训练任务 [G1_GraphFCF.md](../tasks/G1_GraphFCF.md) 保留原结论；本轮未改阶段结论文档 `G1_ACF_2026-10-17.md` 的事实。图传播收益以已训练 Graph/Identity 同构对照为依据，冻结权重邻接替换仅是敏感性诊断。下一投入由用户决定。

2026-10-17：用户确认 G1 方向继续保留；本次冻结设置下的消融结论不能扩大为终止整个方向。代码已整理到 `methods/g1_acf_graph_fcf/`，两版四列图直接复用 `run_adp_siamese.plot_chart`。B0 原 `result.json` 和 `chart.png` 已从 master 提交 `61178b5` 恢复；当前 B0 明确为 `results/adp_old_siamese/stride1_n20997_seed10/`，与 Stage-2 Z 区分。

用户指定的多视角边不确定性 Gate 0 已完成，结果和代码见 [REPORT.md](../../results/multiview_edge_uncertainty/REPORT.md)。它只确认实现与数据一致，不证明边分数有效或最终 `Z` 改进。下一步是否进入 Gate 1 由用户决定；不自动启动后续 Gate。

2026-10-08：跨时间匹配实验已按用户要求清除专用代码、测试、任务和产物，退出当前工作清单。此前“识别与处理的作用尚未分开”的分析仅为待解问题，不代表已选定下一实验或获得新的执行授权。

优先复用既有代码；新增入口需说明必要性。默认一份任务、一份主报告和必要机器产物，再向日志/索引追加记录。研究讨论与代码执行共用仓库证据，不要求在不同应用间反复审批；指标增益、机制证据、创新性（novelty）分别判断。

## 6. 资产与待核验事项

核心 CSI、融合距离、测地距离、Y0、Z 和 Stage-1 图已有保存资产；既往已检查存在性和关键数组头，完整输入哈希、环境与历史协议未全部复核。文件用途见 [PROJECT_GUIDE.md](../PROJECT_GUIDE.md)；其 9 月 13 日快照需结合本状态阅读。

- 历史 MDS 和部分 Siamese 文本数值与源 JSON 有差异，不能静默合并；总体 MDS 记录仍待核验，不与新的固定子集 MDS 门槛结果混用。
- 历史 OOS 实现与主基线架构应区分；仅在复用时核验划分、样本顺序、信息可用性、模型选择和评价设置，不因文件存在就提升为已验证证据。
- 旧状态引用的两份阶段交接总结本地未找到，用户也不清楚位置；直接复用有原始报告的证据，不靠重写聊天总结填补来源。
- 先前文献审计“完成”属于历史记录；具体创新判断依赖某条结论时核验原文，不因此重做全部调研。
- 小型代码与治理文件此前存在未跟踪情况，大型数据/结果被 Git 忽略；外部备份尚未核验。需要时独立处理，不与科研进展混计。
