# Research Agent Rules

## Role and source of truth

- 本仓库用于已有baseline的优化与创新研究。具体范围、当前任务和用户确认事项见 `docs/research/research_state.md`，不要从旧 idea 推断方向。
- 明确的当前用户指令优先；如与旧 state 冲突，标明哪些旧决策已失效。
- 研究工作依次读取 `docs/research/Research_Context.md`、`research_state.md`、state 链接的 active task；按需检索 `experiment_log.md` 和 `experiment_registry.md`。
- context 管长期原则，state 管当前决策，task 管本轮执行，log 管历史，registry 管结果索引。历史文档中的 next step 不自动生效。
- 任务或 artifact 有冲突时停止依赖该冲突的实验，报告具体缺口；继续已授权且不依赖缺口的只读检查、导航和文档纠错。
- `notes/`、`codex_idea/`、旧 README、archive、聊天意见均不是当前科研决策。

## Scientific constraints

- Stage 1 是 inherited CSI → ADP/time fused → 20-NN → D_geo → Siamese FCF → Y0。
- Advisor Stage 2 是 Y0 + D_geo → geodesic-neighbor graph → locally scaled Laplacian/Tikhonov → Z。
- 区分 Stage-1 的 k=20 与当前记录 Stage-2 的 k=128；Stage-2 baseline lambda=0.15、iterations=200，近似求解 (I + lambda L)Z=Y0。
- 相同 D_geo 用于 FCF target 与 Stage-2 图是事实；error reinforcement 等因果解释必须有独立证据。
- GT 仅用于明确标记的离线诊断、评估与绘图。禁止用于 deployable graph、weights、gate、loss、features、targets 或推理规则。
- 仅同 comparable_group 且协议兼容的结果可直接排名。核验 split、subset、stride/N、seed、alignment、CT/TW/KS sampling、调参和 GT 使用；未知时标 verification_required。
- 冻结的分组、端点、阈值和停止规则不因结果而改变。局部修复与 collateral harm 分开报告；负结果不等于没有作用。


## Work one bounded question at a time

- 长任务在当前 task 内维护唯一的“恢复检查点”：目标与授权、代码/结果来源、已完成步骤及证据、未解问题、下一具体动作。方案冻结、实验/验证完成、失败或阻塞、用户改范围，以及结束本轮前更新；不另建同义 handoff 或 memory 文件。
- 新会话或上下文压缩后先读 state 指定的 task 检查点，核验相关文件、分支、运行/提交状态再继续；完成项不因忘记对话而重跑。旧检查点与现状不一致时先纠正事实，不能把记录当成验证。

## Reuse code and keep files discoverable

- 新增入口脚本前说明现有入口不能复用的具体原因。优先在对应实验家族中实现最小改动；不修改无关代码，不建 speculative abstraction。
- 不为每个 idea/参数值复制一个根目录 run_*.py。确需新诊断代码放 diagnostics/<question>/；新方法代码按任务放 methods/<method>/。
- 结果与 REPORT.md 放 results/<experiment_or_run>/，配置/种子/对照使用同一实验家族管理；不在 results 放代码，不在 docs 放大型结果。
- 不覆盖 baseline，不删除负结果，不因整洁而移动历史代码、数据或实验目录。改已有入口时保证旧运行可追溯。
- 默认每轮一个 task、一个主报告和必要机器产物；不重复生成同义 plan/synthesis/handoff/summary。
- 对未知用途文件先标注用途待核验，不能仅凭名称认定可删除。

## Records and completion

- 每个完成实验在 docs/research/experiment_log.md 追加：ID、日期、问题、代码入口、精确配置、种子、输入/输出、协议、数值、解释、限制、状态。
- 日志状态：confirmed / preliminary / negative / invalid / missing-records。负结果必须保存。
- docs/research/experiment_registry.md 登记需保留结果：角色、协议、comparable_group、split、dataset/subset、stride/N、seed、状态、摘要和路径。
- Registry 状态：active / historical / verification_required / negative / invalid；诊断和失败探索不混入主方法排名。临时检查不登记为实验。
- 完成时用简短结论说明：新增证据、支持/不支持什么、关键产物、一个下一步。用证据减少的不确定性衡量进展，不用文件数量。
- 完成验收包含代码、测试、结果和报告的对应关系、实际分支/提交以及记录更新；上传前逐文件检查，不能只上传结果遗漏已有代码。未完成或未上传必须明确标记。
- 已验证的执行事实和文件位置可直接同步 state/registry；研究路线、范围和下一实验的投入由用户确认。state 不保存原始数据或未经审阅的临时解释。

## Channel Chart 默认交付规范（用户确认：2026-10-17）

- **自动执行**：用于评价方法效果的 Channel Chart 实验，默认交付图谱、完整指标 CSV 和报告，不等用户提醒；正式结果与负结果均适用。功能 Sanity、资源 Pilot 等小测试按检查目的选择必要输出，不因能够生成 embedding 就要求绘图或完整效果指标；若测试本身需要判断图谱质量，则按本规范交付。各阶段均须标明阶段及完成状态，不混用或预填未完成结果。补全报告不扩大实验授权或改变冻结的判据。
- **完整指标表**：每个方法/对照一行，字段固定为 `method, ct, tw, ks, mean_m, median_m, p90_m, p95_m, max_m, source_json`。允许增加 seed、N、阶段、状态、`train_time_s`、`peak_vram_bytes`，不能省略 P90/Max 或来源。JSON 保存全精度数值，Markdown 默认显示六位小数；误差单位为米。`source_json` 指向该行真实存在的结果 JSON，报告与最终回复提供可点击来源。
- **复用指标实现**：使用 `common_metrics.chart_metrics` 和 `common_metrics.localization_error_summary`，不得另写近似版本。相互可比的方法须核对相同数据索引并使用同一评价样本；冻结协议优先。当前传导式协议默认 CT/TW/KS 使用 seed=10 固定抽取的 20% 节点，保存抽样索引；误差指标在相同全部节点上分别仿射对齐后计算。留出实验按其既定 train/test 及对齐协议执行，不把全数据仿射评价写成 held-out 能力。保存 `eval_protocol.json`，注明指标样本、对齐拟合/评价范围及 GT 用途。
- **复用当前绘图格式**：直接调用 `run_adp_siamese.plot_chart`，多列图使用其外部 `axes` 接口；原始图 `align=False`、仿射图 `align=True`。保留当前格式：散点大小 5、无点边线、GT 的 x/y 归一化 RGB 映射且蓝通道 0.25、`axis('equal')`、网格 alpha=0.25、180 dpi；两列 9×4 英寸，多列按每列 4.5 英寸宽、4 英寸高扩展。标题含真实方法名及 CT/TW/KS。不得另起一套散点、颜色或布局代码。
- **图谱组织**：需要交付图谱的运行保存标准 `chart.png`（左 GT、右仿射图谱）；方法比较同时输出原始与仿射多列图，GT 在首列，后面明确列出 B0/其他 baseline、必要消融和候选方法。G1 的顺序为 GT / B0 / G1-Identity / G1-Graph。所有列使用相同样本及同一 GT 颜色映射，原始坐标与仿射坐标不混称。只从保存的 embedding 重画时无需重训。
- **产物位置与可追溯性**：沿用 `results/<experiment_or_run>/`，保留包含 `indices/positions/embedding` 的 `embedding.npz`、`result.json`、运行配置与日志；训练实验还应保留实际 best/last checkpoint。主 `REPORT.md` 覆盖问题、配置、实际阶段、完整指标表、资源耗时/显存、图谱观察、结论与限制，并链接代码及产物；复用已有报告，不另建同义总结。
- **验收与汇报**：完成前核验 CSV、来源 JSON、报告表格和图标题数值一致，实际查看输出图片，说明相对 baseline 与必要消融分别改善/退化什么。历史资源未留存时标未知，不用总流程时间冒充训练时间。baseline 原记录本地缺失时先查 Git 历史/已有来源，不能仅凭当前目录判定实验没跑过，也不能用新算结果冒充历史原件。

### comparison/固定格式

- **目录**：比较产物统一放在 `results/<experiment_family>/comparison/`；参考当前 `results/g1_acf_graph_fcf/comparison/` 和 `methods/g1_acf_graph_fcf/compare.py`，按实验家族复用，不为一次比较另建一套格式。
- **表格**：`comparison.csv` 与 `comparison.md` 的方法顺序、字段和值一致，一行一个方法，baseline 在前、必要消融随后、候选方法在后。沿用上面的完整指标字段；耗时/显存放末列，未知值明确标注。Markdown 保持“比较标题 + 一句评价协议 + 指标表”的简洁形式，数值六位小数、`source_json` 可点击；详细分析写入主报告。
- **图片**：同目录保存 raw 和 affine 两版对比图，直接复用上述 baseline 绘图格式；GT 首列，其他列与表格方法顺序一致。四列时沿用 `four_column_raw_channel_chart.png` 和 `four_column_affine_channel_chart.png`，列数按实际比较对象调整，文件名明确区分原始与仿射图。
- **复核文件**：同时保存 `metric_subset_seed<seed>.npy`、`eval_protocol.json` 和 `aligned_charts.npz`，明确同一评价样本、真实输入路径和对齐方式。最终回复提供 comparison 表格、CSV、两版图片和主报告的链接。
