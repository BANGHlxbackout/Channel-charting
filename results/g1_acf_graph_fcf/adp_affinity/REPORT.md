# G1-Graph/ADP-affinity：Sanity 与资源 Pilot

日期：2026-10-26。**正式 full-N 实验未运行；没有方法效果结论。** 本轮只问：保留现有 ACF 预处理、fused-k20 图边集合和原 geodesic stress 时，改用 ADP 距离决定 GCN 传播权重是否值得完整训练。论文仅提供图传播启发；未使用 raw CSI，也未改变模型层数或 loss。

## 实际改动和核验

入口为 [`run.py`](../../../methods/g1_acf_graph_fcf/run.py) 的 `--mode graph --affinity adp`；[`graph_utils.py`](../../../methods/g1_acf_graph_fcf/graph_utils.py) 在原 fused 图的同一无向 support 上，以 ADP cost 及两端第 k 个非自身 ADP 邻居的局部尺度计算传播权重，保留自环和对称归一化。原 `--affinity fused` 为默认。训练仍调用 `run_adp_siamese.siamese_loss`，目标仍为原 `D_geo`。GT 仅在 best checkpoint 依验证 loss 选出后用于绘图和指标。

- full-N Pilot 与原 full-N G1-Graph：20997 个索引及其位置相同；CT/TW/KS 使用相同的 4199 个 seed10 节点；首个训练点对 hash、200000 验证点对 hash 和 14162434 参数均相同。
- 原始有向图 419940 项、无向 support 228773 边；32 个源的保存最短路复验 `mismatch_count=0`、`max_abs_diff=0`。原 fused 双向边有 2759 对微小不对称，最大差 `1.1444091796875e-05`；未修改其最短路语义。新传播邻接经测试为有限、对称，且 support 不变、权重确实变化。
- 已复用原 full-N ACF 特征缓存；Sanity 与 Pilot 的首步梯度均有限，Loss 有限。`python -m unittest tests.test_g1_acf_graph_fcf -v` 的 5 项测试通过；CSV 与两份来源 JSON 逐项一致。相关证据见 [Sanity 图元数据](../stride10_n2100_seed10_graph_adp_sanity/graph_metadata.json)、[Pilot 图元数据](../stride1_n20997_seed10_graph_adp_pilot/graph_metadata.json)、[Pilot 特征元数据](../stride1_n20997_seed10_graph_adp_pilot/feature_metadata.json)。

## 阶段结果

下表的 Sanity 使用 stride10、N=2100、k=35、100 step；Pilot 使用 full-N、k=20，但仅 **5 step**。两者都不是与已完整训练 B0/Identity/Graph 可排名的正式效果比较。误差单位为米；每阶段 CT/TW/KS 在其固定 20% 节点计算，误差在该阶段全部节点上各自仿射拟合与评价。完整数值见 [phase_metrics.csv](phase_metrics.csv) 和各行 JSON。

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | train_time_s | peak_vram_bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| G1-Graph/ADP-affinity-Sanity | 0.577244 | 0.710966 | 0.609835 | 3.473172 | 3.382268 | 5.903951 | 6.478653 | 8.696918 | [result.json](../stride10_n2100_seed10_graph_adp_sanity/result.json) | 8.819892 | 531593216 |
| G1-Graph/ADP-affinity-Pilot | 0.704074 | 0.614653 | 0.598968 | 3.492231 | 3.214147 | 6.270279 | 7.216507 | 10.748053 | [result.json](../stride1_n20997_seed10_graph_adp_pilot/result.json) | 27.510492 | 707292672 |

[Sanity Chart](../stride10_n2100_seed10_graph_adp_sanity/chart.png) / [Pilot Chart](../stride1_n20997_seed10_graph_adp_pilot/chart.png) 均直接调用 `run_adp_siamese.plot_chart`，左 GT、右仿射 chart，统一 GT 位置颜色映射。实际查看后，Pilot 图谱仍挤成宽带，说明 5 step 尚未充分训练；不据此判断新邻接好坏。每个运行保留 [Sanity 评价协议](../stride10_n2100_seed10_graph_adp_sanity/eval_protocol.json)、[Pilot 评价协议](../stride1_n20997_seed10_graph_adp_pilot/eval_protocol.json)、embedding、best/last checkpoint、训练历史、运行命令；完整控制台日志分别为 [Sanity](../adp_affinity_sanity_console.log) 和 [Pilot](../adp_affinity_pilot_console.log)。

## 资源和下一步边界

Pilot 5 step 的训练耗时 27.51 s，峰值 PyTorch 已分配显存 707292672 bytes（约 0.659 GiB）；总流程 47.97 s。旧同配置 full-N G1-Graph 的 2000 step 训练耗时 6224.64 s，峰值 707461120 bytes。新方法 full-N 单臂的粗略资源估计为 **2–3 GPU 小时**，实际早停、缓存和系统负载会改变耗时；这不是实测完整训练耗时。

正式 full-N 尚未获得本次预算确认。若确认，保持 seed10、N20997、k20、原 2000-step 上限和原早停协议，只训练新的 ADP-affinity Graph；B0、G1-Identity 与原 G1-Graph 直接复用已保存 embedding，在同一评价样本上统一重算完整指标、raw/affine 多列图及 collateral harm。旧训练中的验证点对采样有已知排序截取偏向；本轮为可比性保留并明示，不能声称未见节点泛化。若新权重未胜同构 Identity，也不推断所有图传播无效。
