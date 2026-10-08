# Graph-Harmonic Diffusion Refined Channel Charting（图扩散精修）

> 2026-10-08 更新：本项目负责第一个图谱表示工作。第 8 节列出已同步到 Gitee master 的实验入口；仅本地存在的文件另行注明。

这是信道图谱表示学习的 baseline、消融与优化探索代码。核心思路：先用 ADP+时间戳测地相异性
训练出一张自监督的 channel chart（Siamese FCF），再用**图拉普拉斯的闭式平滑**
`(I + λL) Z = Y0` 对这张图做去噪精修——图 `L` 只用 CSI 推出来的 ADP/时间测地邻居
构建，**不用任何真实坐标**，保持自监督。

在同一份 DICHASUS 实测数据（20997 样本）的 transductive 协议下，保存的参考结果如下：

| 方法 | CT | TW | KS | 均值误差 | P95 | 最大误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 参考 Siamese（Y0） | 0.9957 | 0.9960 | 0.0840 | 0.4093 m | 0.8558 m | 7.8936 m |
| **baseline（Z）** | **0.9982** | **0.9982** | **0.0651** | **0.3187 m** | **0.6346 m** | **1.1414 m** |

> 以上为仿射对齐后的图谱评价，不是未见 CSI 的应用性能；历史 MDS 使用不同评价设置，不在此表混排。

---

## 目录说明

```
run_diffusion_refined_charting.py   ← baseline：图扩散精修
archive/multiscale_stress_charting/run_multiscale_stress_charting.py ← 旧多尺度训练探索；完整结果记录缺失
run_adp_fused.py                    ← 产出测地距离（精修的输入之一）
run_adp_siamese.py                  ← 产出初始 chart embedding（精修的输入之一）
preprocess_adp_old_shard.py         ← 从原始 tfrecords 生成 datasets/adp_old_shard4
preprocess_dichasus_numpy.py        ← 预处理的解码辅助
common_dichasus.py / common_metrics.py  ← 数据加载 + CT/TW/KS、仿射定位误差等指标
notes/                              ← 想法说明 + 全部复现记录
datasets/adp_old_shard4/            ← 本地预处理输入；未上传 Gitee
results/                            ← 已同步报告和选定产物；大型矩阵留在本地
```

`notes/new_idea_diffusion_refined_channel_charting.md` 是早期想法说明；旧笔记的待办不作为当前执行指令。已同步代码与结果入口见第 8 节。

---

## 数据（Gitee 不含原始数据与大型矩阵）

本地已有数据与结果；Gitee 只保留选定结果，未上传原始数据、模型权重和大型距离矩阵。运行需要准备
**DICHASUS dataset v2.0，`dichasus-cf02 / cf03 / cf04`**。

准备方式二选一：
1. 下载原始 tfrecords，然后跑 `preprocess_adp_old_shard.py` 生成
   `datasets/adp_old_shard4/`（里面应有 `H.npy / UE_pos.npy / AP_pos.npy / timestamps.npy`）；
2. 或直接向我们要已预处理好的 `adp_old_shard4/` 四个 .npy 放进该目录。

> 协议：拼接 cf02/cf03/cf04 后取 `full_dataset.shard(4, 0)`，共 20997 样本；
> STO/CPO 校准、IFFT 到时域、taps `507:520`、固定四阵列天线排序。

---

## 环境

Python 3，依赖：`numpy scipy scikit-learn torch matplotlib`；预处理另需 `tfrecord tensorboard`，H1 表格诊断另需 `pandas pyarrow`。
有 GPU 更好（精修用 GPU Jacobi 迭代解线性系统，几秒钟）。

---

## 运行流程（4 步）

```bash
# 1. 预处理：生成 datasets/adp_old_shard4（已有数据可跳过）
python3 preprocess_adp_old_shard.py

# 2. ADP+时间融合测地距离 + MDS（产出 results/adp_old_fused/...，含测地距离）
python3 run_adp_fused.py --data-path datasets/adp_old_shard4 \
  --stride 1 --out-dir results/adp_old_fused

# 3. Siamese FCF 学初始 chart
python3 run_adp_siamese.py --data-path datasets/adp_old_shard4 \
  --distance-run results/adp_old_fused/stride1_n20997_seed10 \
  --stride 1 --seed 10 --out-dir results/adp_old_siamese

# 4. baseline 图扩散精修
python3 run_diffusion_refined_charting.py \
  --data-path datasets/adp_old_shard4 \
  --distance-run results/adp_old_fused/stride1_n20997_seed10 \
  --init-embedding results/adp_old_siamese/stride1_n20997_seed10/embedding.npz \
  --out-dir results/diffusion_refined_charting \
  --stride 1 --graph-k 128 --lambda-smooth 0.15 --iterations 200
```

输出在 `results/diffusion_refined_charting/.../`，含 `result.json`（指标）、
`embedding.npz`、`chart.png`。保存 baseline 使用 `graph_k=128, lambda=0.15`，并非脚本默认值；请显式传参。

---

## 早期验证计划（历史记录，不是当前任务）

目前这版是 transductive（没留测试集、单种子、20% 子集算指标），而且数学本身是
标准的图拉普拉斯/流形正则化平滑——直接当"新方法"投会被审稿人质疑。要补：

1. **方法学硬伤**：留 held-out 测试集；跑 5 个种子报均值±方差；主打位置误差/尾部。
2. **决定性消融：物理图 vs 欧氏图** —— 把 ADP/物理测地图换成普通欧氏 kNN 图，
   证明换了就不灵。这是区分"通用平滑"和"信道物理方法"的命根子实验。
3. **λ / 迭代次数的过平滑分析** + 一个有原则的停止准则（能量平台）。
4. **跨数据集 / 跨基础 chart**：在 Fraunhofer 5G/UWB 或 P2 上、在 triplet/AE chart 上也试。
5. **inductive / 样本外扩展**（Nyström 或蒸馏回 FCF），堵"只能离线"的批评。
6. 写作时**主动引用并承认**算子是标准的（Belkin–Niyogi 流形正则化、GFHF、Diffusion Maps、
   GSP 图 Tikhonov 去噪），把贡献定位成系统性、物理驱动、免训练的 CC 精修研究。

`notes/` 里有更详细的复现记录和待办。

## 新增文件说明

基础仓库主要包含全数据版本的预处理、ADP/time 融合距离、Siamese FCF 和图扩散精修；多尺度训练脚本现已归档。当前版本在此基础上新增了以下内容，方便区分哪些是后续补充工作。

### 1. Held-out 验证流程

新增目录：`held-out/`

- `held-out/common_heldout.py`：held-out 实验共用工具，包括固定划分、仿射对齐、误差统计和输出目录管理。
- `held-out/run_adp_fused_heldout.py`：只在训练集上构建 ADP/time 融合距离和 geodesic 图，同时保存 train/test 索引，避免全数据 transductive 评估。
- `held-out/run_adp_siamese_heldout.py`：按 held-out 划分训练 Siamese FCF，并分别导出训练集和测试集 embedding。
- `held-out/run_diffusion_refined_heldout.py`：在训练图上运行图扩散精修，输出 `Z_train`，作为 train-only 诊断结果。
- `held-out/run_diffusion_refined_oos.py`：新增的 diffusion-refined out-of-sample baseline。它用测试样本到训练样本的 ADP/time 物理距离，把训练图精修结果扩展到测试样本，不使用 test-test 图，也不使用测试坐标参与扩展。
- `held-out/diffusion_refined_oos_plan.md`：OOS baseline 的设计文档，说明公式、输入输出、泄漏约束和评价规则。
- `held-out/diffusion_refined_oos_seed10_result.md`：seed 10 的 OOS 结果记录。当前结果显示 diffusion-refined OOS 在 held-out test 上优于 FCF 和当前 CF-HRD `y0_q` 学生模型。

### 2. 欧氏 kNN 消融

- `run_euclidean_knn_refined_charting.py`：把图扩散精修中的 ADP/time 物理图替换为初始 embedding 上的欧氏 kNN 图，用来验证收益是否来自物理图，而不是普通平滑。
- `tests/test_euclidean_knn_refinement.py`：覆盖欧氏距离矩阵计算等基础逻辑。

### 3. CF-HRD 探索线

新增目录：`CF_HRD/`

- `CF_HRD/README.md`：CF-HRD 各阶段的验证记录和当前结论。
- `CF_HRD/closed_loop_validation.md`：闭环验证方案和通过/失败标准。
- `CF_HRD/run_cross_fitted_teacher.py`：生成 cross-fitted no-self-anchor harmonic teacher，用于得到泄漏更低的 residual target。
- `CF_HRD/run_landmark_context.py`：从训练集物理图中选 landmark，构造 train/test 的物理上下文特征 `q`。
- `CF_HRD/run_residual_student.py`：训练 ridge residual student，包括 `y0_only`、`q_only`、`y0_q` 三个变体。
- `CF_HRD/run_stage5_evaluation.py`：用训练集拟合仿射变换，在 held-out test 上评估 residual student。
- `CF_HRD/plot_residual_student_charts.py`：画 residual student 的 chart 对比图。

相关 idea 文档：

- `codex_idea/cross_fitted_harmonic_residual_distillation.md`：CF-HRD 的完整想法、公式、与 baseline 的关系、风险和可证伪预测。

当前定位：CF-HRD 是探索性优化线，不是替代 baseline 的主结论。已有 seed 10 结果说明 `y0_q` 可以优于 held-out FCF，但还没有超过 diffusion-refined OOS baseline。

### 4. P2 方法复现与协议检查

新增目录：`methods/P2/`

- `methods/P2/main.py`：P2 方法的本地复现入口。
- `methods/P2/check_p2_protocol.py`：检查 P2 数据划分、预处理和实验协议是否一致。
- `methods/P2/preprocess_dichasus_v2.0.py`：P2 相关的 DICHASUS v2.0 预处理脚本。
- `methods/P2/utils/`：P2 复现所需的模型、loss、参数和 channel charting helper。

这部分主要用于横向参考和后续跨方法/跨数据协议对齐，不是当前主实验入口。

### 5. 测试文件

新增目录：`tests/`

- `tests/test_common_heldout.py`：held-out 划分、仿射对齐等共用逻辑测试。
- `tests/test_adp_fused_heldout.py`：训练集 ADP/time 距离与 split 保存逻辑测试。
- `tests/test_adp_siamese_heldout.py`：held-out Siamese FCF 相关辅助函数测试。
- `tests/test_diffusion_refined_heldout.py`：train-only diffusion refinement 输入校验和保存逻辑测试。
- `tests/test_diffusion_refined_oos.py`：OOS geodesic 扩展、top-k 选择、local Tikhonov 闭式解测试。

### 6. 文献、想法和过程记录

新增或扩展目录：`notes/`、`docs/superpowers/`

- `notes/channel_charting_light_literature_search.md`、`notes/graph_laplacian_regularization_light_literature_search.md`：channel charting 与图拉普拉斯正则相关文献调研。
- `notes/diffusion_refined_reference_papers_learning_report.md`、`notes/paper_learning_extracts/`、`notes/paper_deep_extracts/`：参考论文摘录和深度阅读记录。
- `notes/mentor_idea_baseline_alignment_audit.md`：当前工作与 diffusion-refined baseline 的对齐审查。
- `notes/refinement_repair_*.md`、`notes/physics_chart_disagreement_*.md`：围绕 refinement/repair 方向的机制、假设、风险和 idea 推演。
- `docs/superpowers/specs/2026-06-25-heldout-diffusion-charting-design.md`、`docs/superpowers/plans/2026-06-25-heldout-diffusion-charting-implementation.md`：held-out 工作流的设计和实施计划记录。

这些文件主要是研究过程材料，便于追溯 held-out、OOS 和 CF-HRD 的探索过程；真正运行主流程时优先看根目录脚本、`held-out/` 和 `CF_HRD/README.md`。

### 7. 其他新增文件

- `ChannelCharting.ipynb`：triplet-based channel charting baseline notebook，包含 triplet loss、anchor/positive/negative 采样、embedding 训练和结果导出。
- `run_euclidean_knn_refined_charting.py`：也属于消融脚本，已在上面列出。


## 8. 已同步实验代码与结果入口

下表路径均相对仓库根目录；代码和选定结果已进入 Gitee master。Stage 1 生成 Y0，Stage 2 精修 Y0→Z，不能与论文的两个工作混同。

| 实验 / 诊断 | 代码位置 | 结果入口 |
|---|---|---|
| baseline / 欧氏图对照 | `run_diffusion_refined_charting.py`、`run_euclidean_knn_refined_charting.py` | `results/diffusion_refined_charting/`、`results/euclidean_knn_refined_charting/` 下的 `result.json` |
| 精修收益与损害 | [diagnostics/diagnose_diffusion_refinement.py](diagnostics/diagnose_diffusion_refinement.py) | [diagnostic_summary.md](results/diffusion_refinement_diagnostics/stride1_n20997_seed10_k32/diagnostic_summary.md) |
| 局部一致性代理 | [diagnostics/diagnose_local_consistency_proxies.py](diagnostics/diagnose_local_consistency_proxies.py) | [修订报告](results/local_consistency_proxy_diagnostics/stride1_n20997_seed10_k32_phys128_revised/local_consistency_proxy_diagnostics.md) |
| 图溯源 / H1 删边诊断 | [diagnostics/phase2c_h1_h2/](diagnostics/phase2c_h1_h2/) | [H1_COUNTERFACTUAL_REPORT.md](results/phase2c_h1_h2/H1_COUNTERFACTUAL_REPORT.md)；H2 未执行 |
| SC² Stage-1 边长惩罚 | [methods/sc2_stage1/run_pilot.py](methods/sc2_stage1/run_pilot.py) | [REPORT.md](results/sc2_stage1_consensus_pilot/REPORT.md)；MDS 判据未通过，未训练 FCF |
| Stage-1 事后复查 / 损害归因 | [diagnostics/sc2_stage1_review.py](diagnostics/sc2_stage1_review.py)、[diagnostics/sc2_stage1_harm_attribution/](diagnostics/sc2_stage1_harm_attribution/) | [POSTHOC_REVIEW.md](results/sc2_stage1_consensus_pilot/POSTHOC_REVIEW.md)、[REPORT.md](results/sc2_stage1_harm_attribution/REPORT.md) |
| Stage-1 两视角一致性 | [diagnostics/sc2_stage1_multiview_check/](diagnostics/sc2_stage1_multiview_check/) | [REPORT.md](results/sc2_stage1_multiview_check/REPORT.md)；未通过判据 |
| Stage-2 节点保真 | [methods/sc2_stage2/run_sc2_adaptive_fidelity.py](methods/sc2_stage2/run_sc2_adaptive_fidelity.py) | [worktree_run/REPORT.md](results/sc2_stage2_adaptive_fidelity_pilot/worktree_run/REPORT.md)、同目录 `pilot_metrics.json`；未通过判据 |
| Stage-2 边权实验 | [methods/sc2_stage2_edge_consensus/](methods/sc2_stage2_edge_consensus/) | [REPORT.md](results/sc2_stage2_edge_consensus/REPORT.md)、同目录 `complete_metrics_followup.json` |
| 边权事后分析 | [diagnostics/sc2_edge_posthoc/audit.py](diagnostics/sc2_edge_posthoc/audit.py) | [posthoc_edge_and_sample_audit.json](results/sc2_stage2_edge_consensus/posthoc_edge_and_sample_audit.json) |

`methods/sc2_stage2_edge_consensus/` 中：`run.py` 为原实验，`random_followup.py` 与 `best_lambda_followup.py` 为复验，`complete_metrics.py` 汇总完整指标，`plot_representative_chart.py` 生成代表图谱。节点保真原 worktree 结果放在 `worktree_run/`，避免与另一份本地运行记录混用。随机边降权有小幅收益，但 SC² 定向选边未超过随机对照；20 次随机掩码不是 20 套独立数据。

归档代码：

- [archive/direct_support_edge_refined_charting/](archive/direct_support_edge_refined_charting/)：原边过滤脚本与测试。
- [archive/repair_gated_residual_correction/](archive/repair_gated_residual_correction/)：原门控残差修正脚本与测试。
- [archive/multiscale_stress_charting/run_multiscale_stress_charting.py](archive/multiscale_stress_charting/run_multiscale_stress_charting.py)：Stage-1 多尺度采样训练；完整结果记录缺失，不作为当前方案。

已同步测试位于 [tests/](tests/)；归档方案测试位于各自 `archive/<experiment>/tests/`。结果目录只同步选定报告、指标和复核产物，不能假定包含所有运行输入。

本地 `AGENTS.md`、`docs/research/`、`docs/tasks/`、`docs/PROJECT_GUIDE.md`、`run_smoothness_audited_refined_charting.py` 和另一份 `methods/sc2_stage2_adaptive_fidelity/` 实现尚未统一审阅同步；Gitee 克隆不包含这些文件。汇报素材、构建目录和缓存不上传。跨时间匹配专用实验已删除，不列入当前代码清单。
