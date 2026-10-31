# Baseline Optimization — Next Cycle

- 更新：2026-10-09；关联 `docs/research/research_state.md` v2.5。
- 阶段：多视角边不确定性方案 Gate 0 已完成；尚无方法效果结论。
- 本轮授权：用户于 2026-10-09 明确要求完成 Gate 0；已按范围收尾。Gate 1–3、训练及效果实验不在本轮授权内。

## 目标与分工

围绕第一个图谱表示工作，判断现有精修证据中哪一个缺口值得继续解决，给出明确推荐及一个可审阅的最小方案。agent 负责思考、证据核对、技术设计与建议；用户主要决定研究取舍和资源投入，不负责替 agent 寻找正确技术答案。

## 先用的证据

- `results/sc2_stage2_edge_consensus/REPORT.md`、`complete_metrics_followup.json`：定向 SC² 选边未超过随机；20 次随机掩码在同一数据上有小幅收益，不能当作独立泛化。
- `results/diffusion_refinement_diagnostics/stride1_n20997_seed10_k32/diagnostic_summary.md`：整体精修收益与局部损害并存。
- 原节点保真结果 `results/sc2_stage2_adaptive_fidelity_pilot/worktree_run/REPORT.md`；边过滤、门控旧结果及 λ 对照按需读取。失败方案已存在，不能换名字重复。
- 代码来源：`methods/sc2_stage2/run_sc2_adaptive_fidelity.py`、`methods/sc2_stage2_edge_consensus/`、`diagnostics/sc2_edge_posthoc/audit.py`。原 worktree 来源已记录在 master 提交 `efb5a32`；当前代码/README 基准为 `a916c69`。
- Stage-1/H1 属于相关诊断背景，只有直接影响当前判断时读取；已删除的时间片段匹配不恢复、不继续。

## 本阶段必须交付

1. 用普通语言解释要解决的现象、已有证据、最强反解释以及解决后会改变哪个精修决定。
2. 比较至多两个可区分的解释/解决思路，明确推荐一个，并说明另一项为何暂不投入；若现有证据不足或研究价值低，明确建议暂缓并列出关键缺口。
3. 在一个具体小例子上检查对象与计算含义一致：节点/边/关系是什么，评分实际判断什么，干预改在哪个阶段。异常评分相关性不是干预收益。
4. 针对推荐机制核对必要的 closest prior work，不重做完整赛道调研，也不先承诺 novelty。
5. 最小方案由 agent 写清：复用的代码与输入、主要干预、必要对照、评价端点、损害约束、计算预算、支持/否定条件、精确命令及结果位置；在正式执行前检查端点能否识别目标效应。
6. 用一个推荐和依据供用户决定是否投入；不把“该怎么做”作为开放问题退回用户。

本阶段只在本任务内维护准备结果，并在对话中交付建议；不新增 plan/synthesis/handoff 文件。获准执行后先在本任务固定方案，再复用现有入口；没有授权不运行训练、改图、参数扫描或新诊断。

## 恢复检查点（覆盖更新，不追加聊天历史）

- 目标/授权：仅完成 [多视角方案](../../codex_idea/02_multiview_edge_uncertainty_geodesic_plan.md) 的 Gate 0；不运行 Gate 1–3。
- 已完成：`methods/multiview_edge_uncertainty/gate0.py` 全量运行 `confirmed`，13/13 机器检查通过；`tests/test_multiview_adp_gate0.py` 6/6 通过。四视角 ADP 对缓存全矩阵最大差 `9.536743e-6`，32×20997 个基线测地距离最大差 0，`kappa=0` 图逐项一致。原图 2759 对 reciprocal 边有微小两向差，保留原值并共享乘因子；首次不合理的绝对相等检查已纠正并完整重跑。
- 代码/结果来源：[主报告](../../results/multiview_edge_uncertainty/REPORT.md)、`results/multiview_edge_uncertainty/edge_risk/{gate0_regression.json,edge_observations.npz}`；输入为 `datasets/adp_old_shard4/` 与 `results/adp_old_fused/stride1_n20997_seed10/`。当前分支 `codex/sc2-stage2-edge-consensus`；本轮改动尚未提交或上传。
- 未解问题：四阵列分歧是否在原候选边上识别可操作的错误关系，以及是否改善 `Y0`/`Z`，均未测试。
- 下一动作：停在 Gate 0；仅在新的明确授权下按冻结方案进入 Gate 1，不重跑已完成 Gate 0。

方案固定、验证完成、失败/阻塞、用户改变范围和结束一轮前更新本检查点。继续任务或上下文恢复时先读检查点，再核验文件/分支/运行状态，不能盲信旧记录或重跑已完成项。模型记忆仍可能遗漏，检查点只提供可恢复、可核验的依据，不保证推理正确。
