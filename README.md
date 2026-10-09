# SC² 启发的 Stage-2 边级实验

本分支记录一个已完成的边级实验。原始 `D_geo`、Siamese `Y0` 和 Stage-2 图构造保持不变；先按一阶距离兼容性与共同邻居数给原图的边评分，再将选中边的权重减半，运行原 Stage-2 求解器。真实位置（GT）只用于评价和事后分析，不参与选边或调参。全部 20,997 个样本参与建图，结果属于 **transductive** 评估。

**当前发现：**SC² 定向选边的平均误差比原 baseline 低约 0.73 毫米，但 P95 变差；沿相同 `D_geo` 距离层配额随机选边的 20 次运行，平均误差中位数比 baseline 低约 0.99 毫米，P95 也降低。共同支持分数倾向选中物理跨度偏大的边，却未证明“将这些边减半”比随机更好。随机对照选中边的 73.2% 集中在最长四个距离层，因此目前也无法单独解释随机对照为何改善。完整设计、全部指标、三张图及其含义见 [实验报告](results/sc2_stage2_edge_consensus/REPORT.md)。

## 1. 数据集下载

从 [DICHASUS cf0x 官方数据 v2.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-2854&version=2.0)下载 cf02、cf03、cf04 的三个 TFRecord，并把以下六个文件放在仓库根目录的 `datasets/`。各自的 REFTX offset JSON 下载方式见[官方 Channel Charting 教程](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/channelcharting/)：

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

三个 TFRecord 的已核对 MD5：cf02 `95ab14e40536089b2365f76d72345403`；cf03 `12e8d8458809400ef1e0524a0254c61c`；cf04 `847efb9caef7d3d997d9e23c68db375f`。本地 `specv2.0.json` 与官方 `spec.json` 只是文件名不同，核对记录中的 MD5 都是 `cce58edb6d40998b929488a204ec4412`。**本预处理脚本不读取 spec 文件**，只读取上面六个准确文件名。

`preprocess_adp_old_shard.py` 默认从 `datasets/` 读取原始文件，按 cf02→cf03→cf04 顺序拼接后全局每 4 条取 1 条，生成 `datasets/adp_old_shard4/` 中的 `H.npy`、`UE_pos.npy`、`timestamps.npy`、`AP_pos.npy`，应有 20,997 个样本。**不要添加 `--sort-by-time`**，否则样本顺序与本次实验不符。仓库只提交两个空占位 `datasets/.gitkeep` 和 `datasets/adp_old_shard4/.gitkeep`；TFRecord、offset JSON、预处理数据均不上传。

## 2. 运行命令和结果

在仓库根目录执行。已记录环境为 Python 3.11.15，NumPy 2.4.6，SciPy 1.17.1，scikit-learn 1.8.0，matplotlib 3.10.9，tfrecord 1.14.6，PyTorch 2.11.0+cu128。`geodesic.npy` 单文件约 1.76 GB。命令里的 `python` 应指向安装这些依赖的环境。
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

边级脚本目前在 `SOURCE = Path(...)` 一行使用原实验机路径 `D:\diffusion_refined_charting`。**需将该行改成克隆后的仓库绝对路径**；算法公式、输入文件相对位置和随机种子不需改。需要复现报告里 `lambda=0.1344` 的已有原图对照时，先用相同的 `run_diffusion_refined_charting.py` 命令把 `--lambda-smooth` 改为 `0.1344`，并将 `--out-dir` 改为 `results/lambda_sensitivity_refined_charting`，再执行 `python methods/sc2_stage2_edge_consensus/best_lambda_followup.py`。

预期原图平均误差约 0.318748 m；SC² 选边约 0.318021 m；一阶兼容边内分层随机选边的 20 次平均误差中位数约 0.317755 m。20 次只是固定同一数据与 `Y0` 后，更换 seed 10–29 的选边掩码，并非 20 次独立数据实验。精确参数见 [配置](results/sc2_stage2_edge_consensus/config.json)。CUDA 累加顺序可能导致坐标最后几位不同，应核对样本顺序、配置与指标，不要求逐字节相同。

## 3. 结果怎么看、当前卡在哪里

- [REPORT.md](results/sc2_stage2_edge_consensus/REPORT.md)：实验设计、主表、对照范围与结论；主表的“20 次中位数”是各指标分别取中位数，并不对应一张实际图谱。
- [选边诊断图](results/sc2_stage2_edge_consensus/posthoc_selected_edge_diagnostic.png)：SC² 更常选中物理跨度偏大的边；GT 仅用于选完后的判断，跨度偏大不等于降权有益。
- [修复和损害图](results/sc2_stage2_edge_consensus/posthoc_repair_harm_map.png)：SC² 使一些区域改善，也使另一些区域恶化；颜色表示相对原 baseline 的逐点误差变化。
- [Channel Chart 对照](results/sc2_stage2_edge_consensus/channel_chart_seed18_vs_baseline.png)：seed-18 随机方案的整体形状与 baseline 几乎一样，毫米级误差差别须看指标表。

当前卡点是**识别出的关系与正确的处理动作尚未接上**：SC² 分数可富集某类边，却没有证据表明给它们统一乘 0.5 是合理规则；随机收益还与距离层配额混在一起。后续应先定位误差从原始关系、`D_geo`、`Y0` 还是精修目标进入，再决定改哪个环节，不预设继续使用当前 Tikhonov 算子。

仓库保留实验代码、配置、报告、必要 JSON 和展示图片。数据集、生成的 ADP/fused/geodesic 矩阵、模型权重及可再生的约 20 MB `edge_scores_and_masks.npz` 不上传。旧 README 内容仍可从本分支的 Git 历史查看。
