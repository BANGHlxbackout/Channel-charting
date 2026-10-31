# Stage 1 共同邻居实验：Gitee 同步前清单

状态：**仅完成本地整理，未提交、未上传、未推送**。导师反馈所指的是 2026-09-25 的 Stage 1 共同邻居边长实验；原实验及停止判据不改。本清单与事后分析准备于 2026-10-08。

## 仓库与版本

- 本地 Git 远端 `origin`：`https://gitee.com/cactus-jacque/diffusion-charting.git`，已记录的远端分支为 `origin/master`。
- 当前本地基础提交：`bf7acc1f7b39354a733b85e037f917c7f2704732`（2026-07-03），**不包含**本次 SC² 文件。同步时应创建独立的 `codex/sc2-stage1-review` 提交，再报告实际提交 SHA 和 `https://gitee.com/cactus-jacque/diffusion-charting` 对应链接；现在不能把基础提交冒充本次版本。
- 只读访问检查 `git ls-remote --symref origin HEAD` 未成功：当前主机的 Git 凭据提示脚本无法启动，未取得 Gitee 远端最新状态。正式提交/推送前必须重新检查远端并解决认证；本轮没有尝试上传。
- 工作树还有与本实验无关的 `README.md` 修改和许多未跟踪文件。同步时只能逐文件选择下表，不能执行 `git add .`、不能顺带提交旧 idea、其他实验或缓存。

## 拟同步的精确范围

| 角色 | 文件或目录 | 是否拟进入本次提交 |
|---|---|---|
| 原实验代码与语义测试 | `methods/sc2_stage1/run_pilot.py`、`tests/test_sc2_stage1.py` | 是 |
| 本轮事后核查代码 | `diagnostics/sc2_stage1_review.py` | 是；明确为 GT-assisted 评价脚本，不是选边方法 |
| 原设计 | `docs/tasks/sc2_stage1_consensus_pilot.md` | 是 |
| 累积研究日志与索引 | `docs/research/experiment_log.md`、`docs/research/experiment_registry.md` | 可选；它们包含其他实验，若同步应另行审阅，不必混进本次最小提交 |
| 供导师阅读的两份报告 | `results/sc2_stage1_consensus_pilot/REPORT.md`、`POSTHOC_REVIEW.md` | 是；前者是原冻结结果，后者是新事后分析 |
| 小型配置与指标 | 同目录 `config.json`、`variant_summaries.json`、`mds_gate.json`、`geodesic_gate.json`、`POSTHOC_REVIEW.json`，以及 `baseline/`、`uniform/`、`shuffled/`、`consensus/` 下各自的 `mds_result.json`、`graph_summary.json` | 是 |
| 复查调整了哪些连接 | 同目录 `edge_scores.npz`；四组 `stage1_graph.npz`；以及原始 `results/phase2c_h1_h2/stage1_graph.npz` | 建议是；每个文件数 MB，含边和权重，不含 GT 标签 |
| 复查抽样对与样本收益/损害 | 同目录 `gate_pairs.npz`、`POSTHOC_SAMPLE_ERRORS.csv` | 待上传前检查：CSV 含 GT 位置和离线误差，仅用于评价/事后分析，不得作为方法输入 |

`results/` 被仓库 `.gitignore` 整体忽略。上述需要进入提交的结果文件必须**逐一明确选择并按需 `git add -f`**；不能解除整个目录的忽略规则。原始数据集、`fused.npy`、全量 `geodesic.npy` 和四个约 1.76 GB/个的重算距离矩阵留在本地，不能当作普通小文件推送 Gitee；报告须写明它们是精确重算所需的外部输入，并提供来源/哈希或另行决定受控存储。仅上传小型结果**不构成从空白 clone 完整复现训练和最短路的自包含包**。

本地输入/评分文件的 SHA-256，可用于将来核对导师机器上的文件版本：

| 文件 | SHA-256 |
|---|---|
| `results/adp_old_fused/stride1_n20997_seed10/fused.npy` | `4EC665404FBB2BE7508743AA4BBFCB523E45FEA33A85CBC2234B6CE6CDE6EBD7` |
| `results/adp_old_fused/stride1_n20997_seed10/geodesic.npy` | `DE464EF9598757A9893E78A4A5335AC9EBBD363BE14863E316E816005E73439D` |
| `results/phase2c_h1_h2/stage1_graph.npz` | `D9D58BB0C266A8F797F6C6E265F0FB82DBB0776675C60D9C37E7404D9CEC10FF` |
| `results/sc2_stage1_consensus_pilot/edge_scores.npz` | `B561E5D0BAB8B14A601BC6DECEAF81A0032DA7543A04C85FF3252D1E60282D2B` |

## 上传前的最后核对

1. 确认导师指的是 Stage 1 的 MDS 预检查：`consensus` mean/P95 为 `0.582461/1.173664 m`，打乱对照为 `0.559366/1.156367 m`；**这不是最终 `Z`**。
2. 运行 `diagnostics/sc2_stage1_review.py`，核对 `POSTHOC_REVIEW.json` 与报告表格，并执行已有 `tests/test_sc2_stage1.py`；确认 GT 只在离线分析和评价中出现。
3. 确认 `POSTHOC_SAMPLE_ERRORS.csv` 的位置坐标是否适合当前 Gitee 仓库的可见范围；如不适合，提交不含坐标的样本误差表，而将完整 CSV 留在本地。
4. 检查远端最新提交与认证、逐文件审阅 staging diff，形成独立提交。**只有实际推送成功且远端可读后**，才在汇报里附仓库链接和该提交 SHA。当前不做这一步。

本地分析入口：[事后审查](../../results/sc2_stage1_consensus_pilot/POSTHOC_REVIEW.md)。后续已完成[损害归因检查](../../results/sc2_stage1_harm_attribution/REPORT.md)：只加长跨时间 ADP 连接便几乎重现旧实验的 MDS 恶化；对应代码、冻结方案、测试及小型 JSON 应在上传前另行审阅并逐文件选择。另有[两视角数值一致性诊断](../../results/sc2_stage1_multiview_check/REPORT.md)，它仅检验两个距离是否相等，不能作为“双方都认为相近”或 SC² 第三方支持的验证。上述新文件尚未提交或上传，当前没有可提供的本次提交版本。
