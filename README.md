# Graph-Harmonic Diffusion Refined Channel Charting（图扩散精修）

本项目负责信道图谱表示学习的第一个工作，包含 baseline、消融与优化探索代码。核心思路：先用 ADP+时间戳测地相异性
训练出一张自监督的 channel chart（Siamese FCF），再用**图拉普拉斯的闭式平滑**
`(I + λL) Z = Y0` 对这张图做去噪精修——图 `L` 只用 CSI 推出来的 ADP/时间测地邻居
构建，**不用任何真实坐标**，保持自监督。

在同一份 DICHASUS 实测数据（20997 样本）的 transductive 协议下，保存的参考结果如下：

| 方法 | CT | TW | KS | 均值误差 | P95 | 最大误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 参考 Siamese（Y0） | 0.9957 | 0.9960 | 0.0840 | 0.4093 m | 0.8558 m | 7.8936 m |
| **baseline（Z）** | **0.9982** | **0.9982** | **0.0651** | **0.3187 m** | **0.6346 m** | **1.1414 m** |

> 以上为仿射对齐后的图谱评价，不是未见 CSI 的应用性能；历史 MDS 使用不同评价设置，不在此表混排。

## 代码与结果导航

以下代码及链接产物已同步到 Gitee master；每项只在此列出入口。Stage 1 生成 Y0，Stage 2 精修 Y0→Z，两者都属于第一个工作。

| 用途 | 代码位置 | 结果 / 说明 |
|---|---|---|
| 数据预处理 | [preprocess_adp_old_shard.py](preprocess_adp_old_shard.py)、[preprocess_dichasus_numpy.py](preprocess_dichasus_numpy.py) | `datasets/adp_old_shard4/`；数据仅在本地 |
| 数据与指标工具 | [common_dichasus.py](common_dichasus.py)、[common_metrics.py](common_metrics.py) | 数据加载、CT/TW/KS 与仿射误差 |
| Stage-1 融合距离与 MDS | [run_adp_fused.py](run_adp_fused.py) | `results/adp_old_fused/`；大型距离矩阵仅在本地 |
| Siamese 初始图谱 | [run_adp_siamese.py](run_adp_siamese.py) | [result.json](results/adp_old_siamese/stride1_n20997_seed10/result.json) |
| Stage-2 baseline | [run_diffusion_refined_charting.py](run_diffusion_refined_charting.py) | [result.json](results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/result.json) |
| 欧氏图消融 | [run_euclidean_knn_refined_charting.py](run_euclidean_knn_refined_charting.py) | [result.json](results/euclidean_knn_refined_charting/stride1_n20997_euclidean_k128_lambda0.15_seed10/result.json) |
| 精修收益与损害诊断 | [diagnostics/diagnose_diffusion_refinement.py](diagnostics/diagnose_diffusion_refinement.py) | [diagnostic_summary.md](results/diffusion_refinement_diagnostics/stride1_n20997_seed10_k32/diagnostic_summary.md) |
| 局部一致性代理诊断 | [diagnostics/diagnose_local_consistency_proxies.py](diagnostics/diagnose_local_consistency_proxies.py) | [修订报告](results/local_consistency_proxy_diagnostics/stride1_n20997_seed10_k32_phys128_revised/local_consistency_proxy_diagnostics.md) |
| 图溯源与 H1 删边诊断 | [diagnostics/phase2c_h1_h2/](diagnostics/phase2c_h1_h2/) | [H1_COUNTERFACTUAL_REPORT.md](results/phase2c_h1_h2/H1_COUNTERFACTUAL_REPORT.md)；H2 未执行 |
| SC² Stage-1 边长惩罚 | [methods/sc2_stage1/run_pilot.py](methods/sc2_stage1/run_pilot.py) | [REPORT.md](results/sc2_stage1_consensus_pilot/REPORT.md)；MDS 判据未通过，未训练 FCF |
| Stage-1 事后复查 | [diagnostics/sc2_stage1_review.py](diagnostics/sc2_stage1_review.py) | [POSTHOC_REVIEW.md](results/sc2_stage1_consensus_pilot/POSTHOC_REVIEW.md) |
| Stage-1 损害归因 | [diagnostics/sc2_stage1_harm_attribution/](diagnostics/sc2_stage1_harm_attribution/) | [REPORT.md](results/sc2_stage1_harm_attribution/REPORT.md) |
| Stage-1 两视角一致性 | [diagnostics/sc2_stage1_multiview_check/](diagnostics/sc2_stage1_multiview_check/) | [REPORT.md](results/sc2_stage1_multiview_check/REPORT.md)；未通过判据 |
| Stage-2 节点保真 | [methods/sc2_stage2/run_sc2_adaptive_fidelity.py](methods/sc2_stage2/run_sc2_adaptive_fidelity.py) | [worktree_run/REPORT.md](results/sc2_stage2_adaptive_fidelity_pilot/worktree_run/REPORT.md)、同目录 `pilot_metrics.json`；原 worktree 版本，未通过判据 |
| Stage-2 边权实验与复验 | [methods/sc2_stage2_edge_consensus/](methods/sc2_stage2_edge_consensus/)：`run.py`、`random_followup.py`、`best_lambda_followup.py`、`complete_metrics.py`、`plot_representative_chart.py` | [REPORT.md](results/sc2_stage2_edge_consensus/REPORT.md)、同目录 `complete_metrics_followup.json` |
| 边权事后分析 | [diagnostics/sc2_edge_posthoc/audit.py](diagnostics/sc2_edge_posthoc/audit.py) | [posthoc_edge_and_sample_audit.json](results/sc2_stage2_edge_consensus/posthoc_edge_and_sample_audit.json) |
| 留出与样本外扩展（OOS） | [held-out/](held-out/)：`run_adp_fused_heldout.py`、`run_adp_siamese_heldout.py`、`run_diffusion_refined_heldout.py`、`run_diffusion_refined_oos.py` | [方案](held-out/diffusion_refined_oos_plan.md)、[历史结果说明](held-out/diffusion_refined_oos_seed10_result.md) |
| CF-HRD 残差蒸馏 | [CF_HRD/](CF_HRD/)：teacher、landmark context、residual student、evaluation | [CF_HRD/README.md](CF_HRD/README.md) 列出各阶段入口与历史记录 |
| 参考方法 | [methods/P2/](methods/P2/)、[ChannelCharting.ipynb](ChannelCharting.ipynb) | P2 与 triplet 对照；协议不同，不与开头的基线指标表混排 |
| 归档：direct-support 边过滤 | [archive/direct_support_edge_refined_charting/](archive/direct_support_edge_refined_charting/) | 未通过验收；原脚本与对应测试保留 |
| 归档：门控残差修正 | [archive/repair_gated_residual_correction/](archive/repair_gated_residual_correction/) | 未超过 baseline 对照；原脚本与对应测试保留 |
| 归档：多尺度采样训练 | [archive/multiscale_stress_charting/run_multiscale_stress_charting.py](archive/multiscale_stress_charting/run_multiscale_stress_charting.py) | Stage-1 旧探索，完整结果记录缺失 |

SC² 探索不等于原论文方法的完整复现。边权实验的随机降权有小幅收益，但定向 SC² 选边未超过随机对照；20 次掩码复验不是 20 套独立数据。第二个无线应用工作由其他项目负责。

## 数据与环境

数据来自 DICHASUS cf02/cf03/cf04。按当前预处理协议拼接后取 `shard(4,0)`，共 20,997 样本，进行 STO/CPO 校准、IFFT、天线排序并保留 taps `507:520`。可以从原始 TFRecords 预处理，或准备相同协议的预处理文件。

运行主流程所需的本地输入：

- `datasets/adp_old_shard4/`：`H.npy`、`UE_pos.npy`、`AP_pos.npy`、`timestamps.npy`。
- `results/adp_old_fused/stride1_n20997_seed10/`：`adp.npy`、`fused.npy`、`geodesic.npy`、`distance_data.npz`；Stage-2 可使用 `geodesic_top512.npy` 邻居缓存，它不是 Stage-1 图。
- 精修还需 `results/adp_old_siamese/stride1_n20997_seed10/embedding.npz`。

Gitee 保存选定报告、指标和复核产物；原始数据、模型权重及大型距离矩阵未上传，仅克隆仓库不能直接重现完整训练。真实位置用于评价和明确标记的离线诊断，不进入可部署方法。

Python 3，依赖 `numpy scipy scikit-learn torch matplotlib`；预处理另需 `tfrecord tensorboard`，H1 表格诊断另需 `pandas pyarrow`。GPU 可用于训练和 Jacobi 精修。

## 主流程运行

从仓库根目录执行；已有输入可跳过相应步骤。

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

保存的 baseline 使用 `graph_k=128`、`lambda=0.15`、200 次迭代，参数与脚本默认值不同，需显式传入。运行输出包含指标、图谱及坐标文件；已有 baseline 产物应保留，新的实验使用独立输出目录。

## 测试与仓库范围

[tests/](tests/) 包含已同步实验的相关测试，归档方案测试在各自 `archive/<experiment>/tests/`。例如：`python -m unittest tests.test_sc2_stage1 -v`。支持参数的入口可用 `--help` 查看；固定实验脚本直接执行会启动计算，应先读对应报告。

当前仅本地、尚未统一审阅同步的内容：`AGENTS.md`、`docs/research/`、`docs/tasks/`、`docs/PROJECT_GUIDE.md`、`run_smoothness_audited_refined_charting.py`，以及另一份 `methods/sc2_stage2_adaptive_fidelity/` 实现及其测试和本地运行记录。不要与上表原 worktree 版本混用。

`notes/`、`codex_idea/` 和旧 `docs/superpowers/` 是历史过程材料，其旧计划不是当前任务。跨时间匹配专用实验已删除。汇报素材、构建目录与缓存不上传；本地未同步的其他结果仍保留原目录。
