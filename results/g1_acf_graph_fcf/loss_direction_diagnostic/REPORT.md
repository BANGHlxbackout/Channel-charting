# G1 新 loss：已训练图谱上的梯度方向诊断

日期：2026-10-26。**状态：离线 embedding 梯度诊断完成；没有重训模型，也没有验证训练后收益。** 目标是在已训练 G1-Graph 上检验候选 `L_new=L_geo+L_diff` 会把点推向哪里、该方向是否修复现有误差。原 B0/G1 checkpoint 和核心训练代码未改。可复算入口为 [`diagnostics/g1_loss_direction/run.py`](../../../diagnostics/g1_loss_direction/run.py)（用 `python -m diagnostics.g1_loss_direction.run --out-dir <新目录>` 避免覆盖本次产物），精确配置见 [run_config.json](run_config.json)。

## 方法和评价边界

- `L_geo` 直接沿用原 `run_adp_siamese.siamese_loss` 公式。`L_diff=mean_i[||z_i-(Pz)_i||²/(s_i+m)]`；`P` 从原 fused-k20 图的同一无向 support、Gaussian 权重及自环行归一化得到，`s_i²=sum_j P_ij D_geo(i,j)²`，`m` 为原训练 margin。这里只检验固定 `lambda=1`。
- `L_geo` 梯度用 seed `20261026` 固定抽取的 399980 个非自身随机点对估计，索引保存于 [loss_pair_sample.npz](loss_pair_sample.npz)；`L_diff` 梯度在 20997 个节点上精确计算。两个解析梯度通过玩具输入中心差分核验，最大绝对差分别约 `1.04e-11` 与 `1.85e-12`。
- 从保存的 Graph/Identity embedding 出发，分别沿 `-∇L_geo`、`-∇L_diff`、`-∇(L_geo+L_diff)` **仅移动一次坐标**。每条方向单独归一化到相同的 raw 坐标 RMS 位移 `0.654238`，即原 fused 邻边测地距离中位数的 5%。这个幅度和 `lambda` 没有根据 GT 选择。没有更新模型权重、优化器或 checkpoint；`diff-step` 只是观察新增项本身的方向，不是独立可训练 loss 的建议。
- 全部 variant 保持与旧结果相同的节点顺序和 4199 个 CT/TW/KS 评价节点。误差在全部相同节点上分别仿射拟合和评价。GT 仅用于**移动完成后**的评价与方向核对。完整 [CSV](comparison/comparison.csv)、[简表](comparison/comparison.md)、[评价协议](comparison/eval_protocol.json)、[梯度与逐点统计](direction_analysis.json) 均已保存；表中 `source_json` 指向真实来源。

## loss 实际推动方向

Graph 起点的固定样本 `L_geo=1.995201`、全节点 `L_diff=0.196716`。令 `Q=I-P`、`S=diag(1/(s_i+m))`，新增项对坐标的精确梯度为 `2 Qᵀ S Qz/N`；它抑制节点与邻域加权质心的残差，同时每个节点也受到相邻节点残差的反向作用，**并非逐点简单移向自身质心**。两个损失梯度的夹角余弦仅 `0.075664`，说明新增项提供的方向与原 stress 基本不同。一次等位移更新后，`L_new` 从 `2.191917` 降为 `2.122706`；原 loss 单独更新后的 `L_new=2.148509`。原 stress 仍提供距离尺度约束。

把移动向量投到原图谱的最优仿射坐标系，Graph 合成方向约有 **55.50%** 的节点在一阶意义上朝减小真实位置误差的方向；实际移动并重新拟合仿射后，**54.43% 改善、45.57% 恶化**。单步物理坐标移动中位数约 `0.008455 m`，Mean 误差只降低 `0.000887 m`。初始误差最高的 20% 节点平均降低 `0.001907 m`，最低的 20% 仅降低 `0.000050 m`；但逐点误差变化的 P95 是 **增加 `0.011682 m`**，存在附带损害。详细分位组见 [方向统计](graph_new_step/result.json)，全部节点的前后坐标与误差见 [pointwise_change.npz](graph_new_step/pointwise_change.npz)。

Identity 的合成方向同样约有 55.07% 节点在一阶意义上朝 GT，实际 54.19% 节点改善，Mean 降低 `0.000915 m`。因此这个局部方向不是 Graph 独有。

## 完整指标：同一评价样本

所有 `*-step` 都是**保存 embedding 的一次离线移动**，其训练耗时与训练峰值显存均不适用；原 G1 的历史训练资源仍在各来源 JSON。误差单位为米，Markdown 保留六位小数；CSV 保留全精度。

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [result.json](../../adp_old_siamese/stride1_n20997_seed10/result.json) |
| G1-Identity | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 | [result.json](../stride1_n20997_seed10_identity_full/result.json) |
| G1-Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 | [result.json](../stride1_n20997_seed10_graph_full/result.json) |
| G1-Identity/geo-step | 0.996976 | 0.996881 | 0.079327 | 0.398324 | 0.361987 | 0.712919 | 0.842530 | 1.428611 | [result.json](identity_geo_step/result.json) |
| G1-Graph/geo-step | 0.996959 | 0.996863 | 0.079259 | 0.399704 | 0.365140 | 0.713207 | 0.845223 | 1.460867 | [result.json](graph_geo_step/result.json) |
| G1-Identity/diff-step | 0.997001 | 0.996912 | 0.079211 | 0.397239 | 0.361124 | 0.712484 | 0.840558 | 1.402385 | [result.json](identity_diff_step/result.json) |
| G1-Graph/diff-step | 0.996982 | 0.996894 | 0.079159 | 0.398712 | 0.363619 | 0.713783 | 0.843205 | 1.432677 | [result.json](graph_diff_step/result.json) |
| G1-Identity/new-step | 0.997000 | 0.996910 | 0.079202 | 0.397428 | 0.360852 | 0.712296 | 0.840959 | 1.407999 | [result.json](identity_new_step/result.json) |
| G1-Graph/new-step | 0.996982 | 0.996895 | 0.079146 | 0.398884 | 0.364134 | 0.713734 | 0.843249 | 1.438673 | [result.json](graph_new_step/result.json) |

与 G1-Graph 原图相比，合成方向的单步 Mean 降 `0.000887 m`、P95 降 `0.003089 m`；同样位移的原 stress 控制仅降 `0.000066/0.001114 m`。CT/TW 略升、KS 略降，Max 降 `0.024506 m`。不过 G1-Graph/new-step 的 Mean/P95 仍高于 G1-Identity/new-step；当前诊断**不支持传播的额外正向作用**。只看这一步，纯 `L_diff` 方向比组合方向略好，说明 `lambda=1` 并非已证实最优，也不应凭 GT 在此调参。

## Channel Chart 与判断

主要四列图直接复用 `run_adp_siamese.plot_chart`： [Graph 原始](comparison/graph_focus/four_column_raw_channel_chart.png) / [Graph 仿射](comparison/graph_focus/four_column_affine_channel_chart.png)，以及 [Identity 原始](comparison/identity_focus/four_column_raw_channel_chart.png) / [Identity 仿射](comparison/identity_focus/four_column_affine_channel_chart.png)。另有包含 B0 和全部对照的 [完整 raw 图](comparison/10_column_raw_channel_chart.png)、[完整 affine 图](comparison/10_column_affine_channel_chart.png)。已目视检查：一次小步对整体轮廓和颜色梯度几乎不可见；结论来自逐点和完整精度指标，而非声称看见明显结构修复。

**判断**：新项在当前 Graph 图谱附近有局部有利方向，尤其对高误差节点；但作用很小、约 46% 节点受损，Identity 同样受益。一次自由移动节点坐标可实现的变化，不等于通过同一个 G1 网络参数更新可实现，更不等于重新训练有收益。后续的无 GT 随机初始化前向检查显示 Graph 的 `L_geo≈250.321`、`L_diff≈0.0165`（Identity 为 `250.076/0.0333`），新增项在起点的数值影响很弱；这仍不能预测后期训练。原 G1 在 100/400/800/2000 step 的验证 geodesic loss 约为 144.28/4.30/2.03/1.87，故 5–100 step 只能核验实现和资源，不能判断最终 Chart 是否改善。若继续检验该候选，必须从随机初始化训练，短功能检查后按已确认的新预算完成最多2000 step 的 Graph；它未超过原有最好 Identity 的主要指标即停，达标才同条件训练新 Identity 以分离 loss 与传播贡献。此次离线结果本身不支持宣称改进成功或直接扩大训练预算。
