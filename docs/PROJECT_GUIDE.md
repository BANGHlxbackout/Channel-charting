# 项目导航与工作流

整理日期：2026-09-13。范围：当前本地仓库的文档、代码入口、文件清单、选定 JSON/报告及数组头，并按本轮用户补充重设工作流。没有重跑科研实验或修改历史实验结论，没有核验全部论文原文或历史聊天。当前研究安排由 [research_state.md](research/research_state.md) v2.2 定义。

## 1. 先看这四处

| 你现在想知道什么 | 入口 | 用法 |
|---|---|---|
| 我们做到哪一步了 | [research_state.md](research/research_state.md) | 围绕导师给定 baseline 寻找优化与创新点；当前机制未确定，下一任务已明确 |
| 哪些实验和结果值得保留 | [experiment_registry.md](research/experiment_registry.md) | 按角色、协议和源文件查找，不能跨 comparable_group 排名 |
| 最近一次诊断究竟说明什么 | [H1_COUNTERFACTUAL_REPORT.md](../results/phase2c_h1_h2/H1_COUNTERFACTUAL_REPORT.md) | 看冻结结果、局部修复与附带损害，不从单一指标推断论文方向 |
| 长期研究边界是什么 | [Research_Context.md](research/Research_Context.md) | 区分 Stage 1/2、GT 诊断/部署输入、研究目标与候选方法 |

[experiment_log.md](research/experiment_log.md) 用于追查执行历史，不必每次从头通读。旧 [Phase 2C 任务](tasks/phase2c_empirical_diagnosis.md) 已加历史标记。

## 2. 对当前进展的判断

已有可复用的工程与排除性证据，但尚未闭合“问题—已有工作差距—方法机制—验证”的论文链条。用户已明确导师给定 baseline 和方向；欠缺的是具体优化机制与实验闭环，不应继续默认重选赛道。本地记录不足以认定停滞由某个模型或应用造成。

- **baseline 资产可保留**：现存 Stage-1 JSON 的 affine-aligned mean 为 0.409347 m，Stage-2 为 0.318748 m；两者在索引中属于 `CG-A-transductive-stride1-current`。这支持该协议内的改善，不支持 unseen-CSI 结论，也不自动构成创新。
- **图溯源有价值**：保存的 Gate 0.5 报告记录 32 个源、671904 个距离的精确 float32 匹配。本轮检查了报告及图文件头，没有再次运行 shortest paths。
- **H1 不是“什么都没发现”**：删除 25/50/100 条目标边时，严重低估关系的修复率记录为 17.30%/18.44%/23.60%，同时产生 collateral degradation；冻结低损害判据未满足。保留 `H1_counterfactual_supported=false`、`H1_supported=null`，不自动补跑 H2。
- **历史 OOS/CF-HRD 值得先盘点**：样本外扩展代码与结果已存在。它们不能证明当前 Candidate B 已被选定或具有 novelty，却能避免下一轮从零重复实现。
- **一些探索结果适合防止重复试错**：direct-support 过滤报告未通过验收，repair gate 报告未超过最佳 global-control mean。其协议尚未重新审计，不扩展为对整个 reliability 方向的否定。

## 3. 文件按用途整理

“历史”表示不作为当前指令，不表示可以删除。“待核验”表示不能直接用于论文结论，不表示文件没有价值。

| 分类 | 路径 / 文件 | 保留与阅读策略 |
|---|---|---|
| 当前决策与治理 | `docs/research/`、`AGENTS.md` | 保留；以 state 定位当前阶段，以 context 定义长期原则 |
| 当前/历史任务 | `docs/tasks/baseline_optimization_next_cycle.md`、旧 `phase2c_empirical_diagnosis.md` | 前者是唯一下一任务，后者保留冻结设计与溯源 |
| Stage-1 主链 | `run_adp_fused.py`、`run_adp_siamese.py` | 核心复用入口；除非任务明确要求，不重新训练或计算 APSP |
| Stage-2 baseline | `run_diffusion_refined_charting.py` | 核心比较对象；与 Stage-1 图分开 |
| 数据与指标工具 | `common_dichasus.py`、`common_metrics.py`、两个 `preprocess_*.py` | 核心依赖；后续协议审计需读 |
| 最近的诊断代码 | `diagnostics/phase2c_h1_h2/`、三个 `tests/test_phase2c_*.py` | 保留 Gate/H1/counterfactual 可复现链条；不是待继续开发的方法 |
| 历史 held-out/OOS | `held-out/`、对应 `tests/` | 优先复用候选资产；先核验 train/test、信息可用性和调参历史 |
| 历史蒸馏探索 | `CF_HRD/` | 保留 teacher/context/student/evaluation；旧 README 的后续指令不自动生效 |
| 图消融 | `run_euclidean_knn_refined_charting.py` | 有已登记的历史 CG-A 消融结果 |
| 历史过滤/修复实验 | `run_direct_support_edge_refined_charting.py`、`run_repair_gated_residual_correction.py` | 保留失败结果、对照与测试，避免以新名字重做同一机制 |
| 历史其他探索 | `run_smoothness_audited_refined_charting.py`、`run_multiscale_stress_charting.py` | 保留；本次未定位同名独立结果目录，不猜测其完成度或有效性 |
| 历史诊断 | `diagnose_diffusion_refinement.py`、`diagnose_local_consistency_proxies.py` | 有对应结果；按具体问题检索，不作为当前实验计划 |
| 外部方法参考 | `methods/P2/`、`ChannelCharting.ipynb` | 保留；P2、triplet 与当前主协议分组不同 |
| 思考和文献过程 | `notes/`、`codex_idea/`、`docs/superpowers/` | 按候选问题检索；idea、聊天解释及旧“next step”不是已确认结论 |
| 展示材料 | `ppt_assets/` | 供组会/论文制图复用，不作为实验原始证据 |
| 演示构建中间件 | `.codex-finalizer/`、`.codex-finalizer-pipeline/`、`.codex-ppt-build/` | 有 PPTX 候选稿和构建脚本；不能仅因名字像临时文件就删除 |
| 可重建缓存候选 | `__pycache__/`、子目录 `__pycache__/` | 不影响研究理解，可在独立清理任务中处理；本轮未删除 |
| 提取媒体 | `.tmp_docx_media_20260710/` | 先确认原文档/最终汇报已保存；本轮未删除 |

### 结果目录全家族导航

| 结果目录（均在 `results/` 下） | 定位 |
|---|---|
| `adp_old_fused/` | 核心 ADP/fused/geodesic 与样本索引；其中 MDS 指标为 UNVERIFIED，不能据目录位置默认全家族可比 |
| `adp_old_siamese/`、`diffusion_refined_charting/` | 当前 CG-A 的 Y0 / Z baseline |
| `phase2c_h1_h2/` | 最近诊断证据：audit、图、配置、冻结设计、parquet、报告与图表应成套保留 |
| `euclidean_knn_refined_charting/` | 已登记 CG-A 历史消融 |
| `heldout/`、`CF_HRD/` | 历史样本外与蒸馏资产，已补入待核验索引 |
| `direct_support_edge_refined_charting/`、`repair_gated_residual_correction/` | 历史未通过验收/对照的探索，已补索引 |
| `lambda_sensitivity_refined_charting/` | 13 组历史参数运行；用于追查调参，不包装成创新 |
| `diffusion_refinement_diagnostics/`、`local_consistency_proxy_diagnostics/` | 历史诊断，后者含 original/revised 两套，保留差异 |
| `p2_baseline_shard4/`、`triplet_baseline_notebook/` | 已登记的其他协议参考 |
| `archive/` | 旧 context/state 快照，不覆盖 `docs/research/` 当前决策 |
| `method_summary.md`、`method_summary_5methods.csv` | 旧汇总；不能用作跨协议排行榜 |

盘点时 `results/` 有 196 个文件，约 6.42 GiB；`datasets/` 约 20.67 GiB。大部分空间来自数据和距离矩阵。目录数量主要造成认知负担，当前应先改善入口与证据连接；删除小型笔记不能解决研究方向问题。

## 4. 核心资产核验快照

以下形状来自当前文件的 NPY/NPZ 头，不需要将大型矩阵读入内存。文件头可读不等于完整数值或历史内容一致性校验。

| 路径 | 数组 / 形状 / dtype | 用途 |
|---|---|---|
| `datasets/adp_old_shard4/H.npy` | `(20997,32,13)` complex64 | 当前预处理 CSI |
| `datasets/adp_old_shard4/UE_pos.npy` | `(20997,3)` float64 | GT，部署方法不得使用；loader 取前两列 |
| `datasets/adp_old_shard4/timestamps.npy` | `(20997,)` float32 | 时间戳 |
| `results/adp_old_fused/stride1_n20997_seed10/fused.npy` | `(20997,20997)` float32，约 1.64 GiB | Stage-1 融合距离 |
| 同一 fused run 的 `geodesic.npy` | `(20997,20997)` float32，约 1.64 GiB | D_geo |
| 同一 fused run 的 `distance_data.npz` | `indices (20997,)` int64；`positions (20997,2)` float32 | 实际样本顺序与诊断位置 |
| 同一 fused run 的 `geodesic_top512.npy` | `(20997,512)` int32 | Stage-2 邻居缓存，不能当作 Stage-1 图 |
| `results/adp_old_siamese/stride1_n20997_seed10/embedding.npz` | `embedding (20997,2)` float32；另含 indices/positions | Y0 |
| `results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz` | `embedding (20997,2)` float32；另含 indices/positions | Z |
| `results/phase2c_h1_h2/stage1_graph.npz` | CSR；data/indices 长 419940，indptr 长 20998 | 溯源图；audit 记录 shape `(20997,20997)` |

代码核对：Stage 1 使用 `k=20`、原始 fused 边长、保存 directed weighted CSR，shortest paths 使用 `directed=False`；不显式修改成对称 CSR。Stage 2 取 D_geo 的 `k=128` 邻居，局部尺度指数权重，`maximum(W,W.T)` 对称化，`lambda=0.15`、200 次迭代。两者语义不同。

本次整理不需要重算。未来运行实验应按任务核验输入哈希、样本顺序、环境和完整协议；避免重新生成全矩阵来“整理文件”。

## 5. 发现的断点与处理状态

| 断点 | 当前证据 | 本次处理 / 剩余问题 |
|---|---|---|
| 主入口过时 | README 指向早期 idea，并以“本工作”介绍 advisor baseline | 已在 README 顶部加当前入口和历史范围提示，并折叠旧工程说明 |
| state 与任务脱节 | 整理前 state v2.1 进入 Phase 03；tasks 中只有 Phase 2C | 已按用户新说明更新为 v2.2 和明确下一任务；旧任务加历史标记 |
| 交接材料缺失 | 旧 state 引用 `phase2c_research_synthesis.md`、`phase2c_empirical_handoff.md`；包括隐藏/被忽略文件的仓库文件名检索未找到 | 用户也不清楚位置；暂不追索，不代写成“原始总结”；有原始报告的工作直接复用原始证据 |
| 文献结论难追溯 | 旧 state 记录 Phase 2A/2B complete，整理前 docs 无对应独立审计报告 | 旧 notes 可作线索；仅在具体创新判断依赖该结论时核验 closest work，不重做全部调研 |
| 结果索引遗漏 | OOS、CF-HRD 等 JSON 存在，原索引未覆盖 | 已单列补登记，统一标 `UNVERIFIED` / `verification_required`，未新增排名组 |
| OOS 叙述范围混淆 | 旧 state 描述主 Stage 2 缺少样本外映射；历史 `held-out/run_diffusion_refined_oos.py` 和结果已存在 | v2.2 区分“主 baseline 架构”与“历史扩展”；存在性不等于新机制已验证 |
| 旧 README 未闭环 | `CF_HRD/README.md` 仍有 “Missing diffusion-refined OOS baseline”，但 OOS JSON 已存在 | 在本导航标明过时，未擅自改写历史研究结论 |
| MDS 汇总冲突 | 旧 state/README 的历史 mean 约 0.458；当前 fused `result.json` 为 0.527488 | 保留索引 `UNVERIFIED`；不把任一数值静默替换为另一运行结果 |
| Siamese 汇总漂移 | 旧 state 的 max 约 7.986；当前 JSON 为 7.893630，median/P95 也有小差异 | v2.2 引用 registry/JSON，旧数值保存在原文快照，不静默改写历史 |
| H1 日志输入路径抄录错误 | 日志指向不存在的 dataset NPZ/config；现代码使用 fused run 的 distance_data/result 及 dataset timestamps | 已向 experiment_log 追加勘误，保留原始条目 |
| 版本保存不完整 | `git ls-files` 未列出 AGENTS、docs/research、docs/tasks、diagnostics；`git status` 显示未跟踪；results/datasets 被 ignore | 本轮未提交或推送；建议之后为小型代码/治理文档建立 Git 快照，为大型数据/结果单独备份。当前未核验外部备份 |

不能把 Git 未跟踪解释为“文件丢失”，也不能假定 GPT、Codex 与本地仓库已经自动同步。当前确认的是本地证据链存在缺口。

## 6. 已生效的工作流重设

用户在本轮补充：导师已给定 baseline 和方向，需要思考优化与创新点；不再把从头选择研究赛道作为下一步。因此已按这一最新范围更新 [research_state.md](research/research_state.md)，保留旧版作历史。

[AGENTS.md](../AGENTS.md) 已精简为稳定规则，移除了重复的结果数值和冗长目录说明。保留图语义、GT 边界、协议可比性、冻结判据、负结果、稀疏计算和可复现记录要求。新增的核心约束是：**先查旧证据、优先复用入口、一个问题一个实验闭环、新增脚本必须有必要性**。

OpenAI 官方文档说明，Codex 启动时会组合适用的全局与项目 AGENTS.md 指令；将稳定行为规则放在此处、让具体状态通过文件引用读取，符合这种配置方式。[官方文档](https://learn.chatgpt.com/docs/agent-configuration/agents-md)

下面是本项目采用的协作流程，不是官方科研流程要求：

1. **你确定范围和关键取舍**：以导师 baseline 为起点；不用每次在 GPT 与 Codex 之间来回转交批准。
2. **先读已有证据**：找一个具体未解决的问题，检查是否已经有相同机制或失败对照；不重新遍历全部 idea。
3. **准备最小对照实验**：固定假设、处理因素、对照、主要指标、损害指标、预算和停止条件。端点能否识别目标效应应在正式冻结前检查。
4. **执行与解释闭环**：已授权任务内完成代码、必要验证、实验和解释；有效负结果收尾，不自动堆模块。
5. **决定继续或停止**：指标改善、机制支持和论文 novelty 分别判断；下一次投入由新增证据决定。

当前 [下一任务](tasks/baseline_optimization_next_cycle.md) 已落地为 ready，尚未执行。先从现存结果挑一个问题并形成具体最小实验方案；此时还没有选定机制和数值判据，不能假装已经具备实验计划。它不要求重新找回全部 GPT 聊天或完成三轮广泛选题。

### 文件产出约束

- 优先修改/复用对应家族代码；不要为每个参数、idea 复制一个根目录 run 脚本。
- 确需独立诊断才放 `diagnostics/<question>/`；方法代码按任务放 `methods/<method>/`。
- 每轮默认一个任务规格、一个主报告和必要机器产物；结果归入同一实验家族。
- log 与 registry 用短条目索引产物，不另写多个同义 plan/synthesis/handoff。
- 现有历史文件按本导航查询；本轮没有为了目录美观移动它们，避免破坏硬编码路径。
- 未执行实验时不创建实验结果、不报新性能；文件数量不计作科研进展。

跨应用协作如果仍有需要，只交接当前 state 版本、active task、相关主报告/结果及唯一待解决问题。另一模型提供批评，不自动充当研究批准人；无法访问仓库时应显式提供相应文件，不假定聊天与本地状态同步。

## 7. 本次改动与验证

- 新增本导航；README 增加当前入口并折叠旧工程说明；旧 Phase 2C 任务加历史标记。
- 原 registry 的 9 个来源文件均存在；新增 9 个历史结果/家族条目，保持待核验身份。
- 修复日志的 registry 路径，并追加 H1 输入路径勘误。
- 根据本轮用户新说明，精简根目录 AGENTS.md，将 state 更新为 v2.2，并添加唯一下一任务入口。
- 原 AGENTS 和 state v2.1 保存在 [archive](research/archive/)，复制时已比较 SHA256，确认逐字节一致。
- 核验核心数组头、选定结果 JSON、引用路径和文档差异；没有重新验证模型性能或全历史协议。
- 未删除/移动实验资产、修改实验代码、改变 H1 冻结结论、运行新实验或提交 Git。
