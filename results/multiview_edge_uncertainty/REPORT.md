# 多视角边不确定性 Gate 0：实现与数据回归

日期：2026-10-09。状态：**通过 Gate 0**；未执行 Gate 1、路径效果实验、MDS、Siamese 训练或 Stage-2 精修。

## 问题与固定输入

本关只判断：当前 `H.npy` 能否按四个物理阵列拆出 ADP，是否复现原 aggregate ADP、原 Stage-1 图与测地距离，以及边分数和零干预图是否满足实现不变量。它不判断分数能否识别错误边或改善最终 `Z`。

- 数据：`datasets/adp_old_shard4/{H.npy,timestamps.npy}`，stride=1，`N=20997`；四组顺序 C/B/A/D，每组 8 天线、13 taps。
- 基线：`results/adp_old_fused/stride1_n20997_seed10/` 的 `adp.npy`、`fused.npy`、`geodesic.npy`、`distance_data.npz` 和 `result.json`；实际 `neighbors=20`、`gamma=12.983333587646484`。
- 实现：`methods/multiview_edge_uncertainty/gate0.py`；分块大小 64，RTX 3050 Laptop GPU 4 GB；每 view 精确计算非自身第 20 邻居尺度，只保存原候选边的四份 ADP，不生成四张完整矩阵。
- 复现命令：`D:\anaconda3\envs\diffusion_charting\python.exe -m methods.multiview_edge_uncertainty.gate0`。
- 测试命令：`D:\anaconda3\envs\diffusion_charting\python.exe -m unittest tests.test_multiview_adp_gate0 -v`（6 项通过）。

## 结果

| Gate 0 检查 | 实测结果 |
|---|---:|
| 原始 CSI/时间与预处理文件的前两条保留样本 | 完全相同；最大差 0 |
| 四份 ADP 之和对缓存 aggregate ADP | 全矩阵无超容差元素；最大绝对差 `9.536743e-6` |
| Stage-1 图 | `419,940` 条有向边；每节点 20 条非自身出边；1 个弱连通分量 |
| 固定 32 源点到所有节点的测地距离 | `671,904/671,904` 在容差内，最大绝对差 `0` |
| 单视角数值对称性 | 固定 1,024 条边抽查，最大两向差 `1.907349e-6` |
| 唯一无向候选边 | `228,773` 条，其中 ADP-source `49,626`，time-source `179,147` |
| source flag 与原 fused cost | 反向 source 不一致 `0`；重算 fused cost 最大差 `0` |
| `kappa=0` 稀疏图 | `data/indices/indptr` 与原图逐项一致 |
| `kappa=0.5` 结构检查 | 连通且无新/删边、非有限值或代价下降；`77,762` 条有向 entry 代价增加 |
| 互为邻居的两向乘因子 | 最大数值差 `2.384186e-7` |

原始 `fused.npy`/kNN 图已有 **2,759 对**互为邻居的边值存在浮点级两向差，最大 `1.144409e-5`。最初的 Gate 0 实现误把“风险图两向绝对代价完全相同”设为通过条件，因此报失败；该条件与 `kappa=0` 原图逐项复现互相冲突。修正为“同一端点对共享乘因子、保留各向原始代价”，补充非对称测试后完整重跑，13 项机器检查全部通过。未对原图作对称化。

评分及建图代码只读取 `H.npy`、`timestamps.npy` 和基线距离/索引；原始 TFRecord 抽查也只解码 CSI 与时间字段。单元测试在没有 `UE_pos.npy` 的目录中验证了评分输入加载。GT 未用于本关分数、图、权重或判断。

## 产物与边界

- 机器结果：`results/multiview_edge_uncertainty/edge_risk/gate0_regression.json`，状态 `confirmed`，记录输入、软件、全部检查和数值。
- 候选边观察：`results/multiview_edge_uncertainty/edge_risk/edge_observations.npz`，包含无向边端点、四视角 ADP、节点局部尺度、`q`、MAD 分歧、`u`、source flag 和 directed entry 映射。
- 数值对称性检查抽取 1,024 条边；四视角求和回归覆盖完整 `N×N`。原始 TFRecord 顺序抽查覆盖 cf02 的前两条保留样本，没有重处理全部三条轨迹。
- `kappa=0.5` 只用于图不变量检查，没有生成新测地矩阵或评价效果。Gate 0 通过不构成多视角风险有效、`Y0` 或 `Z` 改善的证据。

下一步仅在另行授权时按冻结方案进入 Gate 1，检验分数在原候选边上的条件增量信息。
