# 本次 SC² 启发的 Stage-2 边级实验复现

本节是检查本次实验使用的入口；下方原 README 是历史说明，**本次请以这里的路径和命令为准**。第 1 阶段的 Y0 与 D_geo、第 2 阶段的原图基线均保持原配置；本实验只把部分 Stage-2 图边权减半。GT 位置只用于评价和事后分析。全部 20,997 个样本参与原图建图，结果属于 transductive 评估。

## 1. 数据放在哪里

从 [DICHASUS cf0x 官方数据 v2.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-2854&version=2.0)下载**旧版而非 rev2**的三个 TFRecord，并把以下六个文件放在仓库根目录的 `datasets/`。各自的 REFTX offset JSON 下载方式见[官方 Channel Charting 教程](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/channelcharting/)：

```text
datasets/
  dichasus-cf02.tfrecords
  dichasus-cf03.tfrecords
  dichasus-cf04.tfrecords
  reftx-offsets-dichasus-cf02.json
  reftx-offsets-dichasus-cf03.json
  reftx-offsets-dichasus-cf04.json
  adp_old_shard4/             # 预处理输出，初始只有 .gitkeep
```

官方 TFRecord MD5：cf02 `95ab14e40536089b2365f76d72345403`；cf03 `12e8d8458809400ef1e0524a0254c61c`；cf04 `847efb9caef7d3d997d9e23c68db375f`。本机这三份 MD5 已按用户提供的核对记录与官方页面一致。官方 `spec.json` 的 MD5 为 `cce58edb6d40998b929488a204ec4415`；本地曾命名为 `specv2.0.json`。**预处理脚本不读取 spec 文件**，它只读取上面六个准确文件名。

`preprocess_adp_old_shard.py` 默认从 `datasets/` 读取原始文件，按 cf02→cf03→cf04 顺序拼接后全局每 4 条取 1 条，生成 `datasets/adp_old_shard4/` 中的 `H.npy`、`UE_pos.npy`、`timestamps.npy`、`AP_pos.npy`，应有 20,997 个样本。**不要添加 `--sort-by-time`**，否则样本顺序与本次实验不符。仓库只提交两个空占位 `datasets/.gitkeep` 和 `datasets/adp_old_shard4/.gitkeep`；TFRecord、offset JSON、预处理数据均不上传。

## 2. 从数据跑到汇报结果

在仓库根目录执行。已记录环境为 Python 3.11.15，NumPy 2.4.6，SciPy 1.17.1，scikit-learn 1.8.0，matplotlib 3.10.9，tfrecord 1.14.6，PyTorch 2.11.0+cu128。全量距离矩阵和 MDS 较耗时、耗内存；`geodesic.npy` 单文件约 1.76 GB。命令里的 `python` 应指向安装这些依赖的环境。

```powershell
python preprocess_adp_old_shard.py --raw-dir datasets --out-dir datasets/adp_old_shard4 --expect-samples 20997

python run_adp_fused.py --data-path datasets/adp_old_shard4 --out-dir results/adp_old_fused --stride 1 --seed 10 --neighbors 20 --mds-iter 80 --mds-n-init 4 --mds-reduction 1

python run_adp_siamese.py --data-path datasets/adp_old_shard4 --distance-run results/adp_old_fused/stride1_n20997_seed10 --out-dir results/adp_old_siamese --stride 1 --seed 10

python run_diffusion_refined_charting.py --data-path datasets/adp_old_shard4 --distance-run results/adp_old_fused/stride1_n20997_seed10 --init-embedding results/adp_old_siamese/stride1_n20997_seed10/embedding.npz --out-dir results/diffusion_refined_charting --stride 1 --seed 10 --top-k 512 --graph-k 128 --lambda-smooth 0.15 --iterations 200

python methods/sc2_stage2_edge_consensus/run.py
python methods/sc2_stage2_edge_consensus/random_followup.py --pool eligible
python methods/sc2_stage2_edge_consensus/random_followup.py --pool all
python methods/sc2_stage2_edge_consensus/random_followup.py --pool incompatible
python methods/sc2_stage2_edge_consensus/complete_metrics.py
python diagnostics/sc2_edge_posthoc/audit.py
python methods/sc2_stage2_edge_consensus/plot_representative_chart.py
```

边级脚本目前在 `SOURCE = Path(...)` 一行使用原实验机路径 `D:\diffusion_refined_charting`。**需将该行改成克隆后的仓库绝对路径**；算法公式、输入文件相对位置和随机种子不需改。需要复现报告里 `lambda=0.1344` 的已有最优原图对照时，先用相同的 `run_diffusion_refined_charting.py` 命令把 `--lambda-smooth` 改为 `0.1344`，并将 `--out-dir` 改为 `results/lambda_sensitivity_refined_charting`，再执行 `python methods/sc2_stage2_edge_consensus/best_lambda_followup.py`。

预期原图平均误差约 0.318748 m；SC² 选边约 0.318021 m（P95 约 0.636267 m）；一阶兼容边内分层随机选边的 20 次指标中位数约 0.317755 m。20 次只是固定同一数据与 Y0 后，更换 seed 10–29 的选边掩码。指标与图见 [REPORT.md](results/sc2_stage2_edge_consensus/REPORT.md)；精确参数见 [config.json](results/sc2_stage2_edge_consensus/config.json)。CUDA 累加顺序可能导致坐标最后几位不同，应核对样本顺序、配置与指标，不要求逐字节相同。

**上传范围：**代码、README、配置、小型 JSON/图及必要的小数组；不上传上述数据集、生成的 ADP/fused/geodesic 矩阵、模型权重或可再生的 20 MB `edge_scores_and_masks.npz`。

---

## 以下为原 README 历史内容
# Graph-Harmonic Diffusion Refined Channel Charting（图扩散精修）

这是我们在 channel charting 上的改进工作的代码。核心思路：先用 ADP+时间戳测地相异性
训练出一张自监督的 channel chart（Siamese FCF），再用**图拉普拉斯的闭式平滑**
`(I + λL) Z = Y0` 对这张图做去噪精修——图 `L` 只用 CSI 推出来的 ADP/时间测地邻居
构建，**不用任何真实坐标**，保持自监督。

在同一份 DICHASUS 实测数据（20997 样本）上，它比所有基线都好：

| 方法 | CT | TW | KS | 均值误差 | P95 | 最大误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADP/time fused MDS | 0.9917 | 0.9934 | 0.0907 | 0.458 m | 0.961 m | 8.90 m |
| ADP/time Siamese FCF（基线） | 0.9957 | 0.9960 | 0.0841 | 0.409 m | 0.854 m | 7.99 m |
| **图扩散精修（本工作）** | **0.9982** | **0.9982** | **0.0651** | **0.319 m** | **0.635 m** | **1.14 m** |

> 最能打的点不是 CT/TW 的第三位小数，而是**定位误差 / 尾部**：最大误差从 7.99 m → 1.14 m。

---

## 目录说明

```
run_diffusion_refined_charting.py   ← 本工作（idea 1）：图扩散精修
run_multiscale_stress_charting.py   ← idea 2（多尺度采样），目前没超过基线，留作对照/消融
run_adp_fused.py                    ← 产出测地距离（精修的输入之一）
run_adp_siamese.py                  ← 产出初始 chart embedding（精修的输入之一）
preprocess_adp_old_shard.py         ← 从原始 tfrecords 生成 datasets/adp_old_shard4
preprocess_dichasus_numpy.py        ← 预处理的解码辅助
common_dichasus.py / common_metrics.py  ← 数据加载 + CT/TW/KS、仿射定位误差等指标
notes/                              ← 想法说明 + 全部复现记录
datasets/adp_old_shard4/            ← 【空】数据放这里（见下）
results/                            ← 【空】运行输出放这里
```

`notes/new_idea_diffusion_refined_channel_charting.md` 是这个想法的完整说明（数学推导、
目标函数、为什么可发表、待办），**先读它**。

---

## 数据（目录留空，需自行准备）

`datasets/` 和 `results/` 是空的（大文件没打包）。需要的数据是公开的
**DICHASUS dataset v2.0，`dichasus-cf02 / cf03 / cf04`**。

准备方式二选一：
1. 下载原始 tfrecords，然后跑 `preprocess_adp_old_shard.py` 生成
   `datasets/adp_old_shard4/`（里面应有 `H.npy / UE_pos.npy / AP_pos.npy / timestamps.npy`）；
2. 或直接向我们要已预处理好的 `adp_old_shard4/` 四个 .npy 放进该目录。

> 协议：拼接 cf02/cf03/cf04 后取 `full_dataset.shard(4, 0)`，共 20997 样本；
> STO/CPO 校准、IFFT 到时域、taps `507:520`、固定四阵列天线排序。

---

## 环境

Python 3，依赖：`numpy scipy scikit-learn torch matplotlib`；预处理另需 `tfrecord`。
有 GPU 更好（精修用 GPU Jacobi 迭代解线性系统，几秒钟）。

---

## 运行流程（4 步）

```bash
# 1. 预处理：生成 datasets/adp_old_shard4（已有数据可跳过）
python3 preprocess_adp_old_shard.py

# 2. ADP+时间融合测地距离 + MDS（产出 results/adp_old_fused/...，含测地距离）
python3 run_adp_fused.py --data-path datasets/adp_old_shard4 \
  --stride 1 --out-dir results/adp_old_fused

# 3. Siamese FCF 学初始 chart（产出 results/adp_old_siamese/.../embedding.npz）
python3 run_adp_siamese.py --data-path datasets/adp_old_shard4 \
  --stride 1 --out-dir results/adp_old_siamese
python run_adp_siamese.py 
  --data-path datasets/adp_old_shard4 
  --distance-run results/adp_old_fused/stride1_n20997_seed10 
  --stride 1 
  --out-dir results/adp_old_siamese

# 4. ★ 图扩散精修（本工作）
python3 run_diffusion_refined_charting.py \
  --data-path datasets/adp_old_shard4 \
  --distance-run results/adp_old_fused/stride1_n20997_seed10 \
  --init-embedding results/adp_old_siamese/stride1_n20997_seed10/embedding.npz \
  --out-dir results/diffusion_refined_charting \
  --stride 1 --graph-k 128 --lambda-smooth 0.15 --iterations 200
```

输出在 `results/diffusion_refined_charting/.../`，含 `result.json`（指标）、
`embedding.npz`、`chart.png`。`graph_k=128, lambda=0.15` 是目前搜到的最佳设置。

---

## 接下来要做的（把"内部最好"做成"能投的"）

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

基础仓库主要包含全数据版本的预处理、ADP/time 融合距离、Siamese FCF、图扩散精修和多尺度对照脚本。当前版本在此基础上新增了以下内容，方便区分哪些是后续补充工作。

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

- `codex_idea/cross_fitted_harmonic_residual_distillation.md`：CF-HRD 的完整想法、公式、与导师 baseline 的关系、风险和可证伪预测。

当前定位：CF-HRD 是探索性优化线，不是替代导师 baseline 的主结论。已有 seed 10 结果说明 `y0_q` 可以优于 held-out FCF，但还没有超过 diffusion-refined OOS baseline。

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
- `tests/test_cross_fitted_gated_refinement.py`：早期 cross-fitted gated refinement 探索的测试文件。目前对应主脚本未放入仓库，上传前可视情况删除或补回脚本。

### 6. 文献、想法和过程记录

新增或扩展目录：`notes/`、`docs/superpowers/`

- `notes/channel_charting_light_literature_search.md`、`notes/graph_laplacian_regularization_light_literature_search.md`：channel charting 与图拉普拉斯正则相关文献调研。
- `notes/diffusion_refined_reference_papers_learning_report.md`、`notes/paper_learning_extracts/`、`notes/paper_deep_extracts/`：参考论文摘录和深度阅读记录。
- `notes/mentor_idea_baseline_alignment_audit.md`：当前工作与 diffusion-refined baseline 的对齐审查。
- `notes/refinement_repair_*.md`、`notes/physics_chart_disagreement_*.md`：围绕 refinement/repair 方向的机制、假设、风险和 idea 推演。
- `docs/superpowers/specs/2026-06-25-heldout-diffusion-charting-design.md`、`docs/superpowers/plans/2026-06-25-heldout-diffusion-charting-implementation.md`：held-out 工作流的设计和实施计划记录。

这些文件主要是研究过程材料，便于导师追溯我们为什么补 held-out、OOS 和 CF-HRD；真正运行主流程时优先看根目录脚本、`held-out/` 和 `CF_HRD/README.md`。

### 7. 其他新增文件

- `ChannelCharting.ipynb`：triplet-based channel charting baseline notebook，包含 triplet loss、anchor/positive/negative 采样、embedding 训练和结果导出。
- `run_euclidean_knn_refined_charting.py`：也属于消融脚本，已在上面列出。

