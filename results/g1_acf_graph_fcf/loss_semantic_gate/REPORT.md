# G1 Stage-1 loss 语义 Gate（离线诊断）

- 日期：2026-10-25；状态：**Gate 未通过；没有训练新模型，也没有产生新 Channel Chart**。
- 问题锚点：G1-Graph 已使用 fused 图传播，但沿用 B0 的端点 geodesic stress；Graph 未明显优于同构 Identity。判断改变 Stage-1 loss 是否有数据与机制依据，不进入 Stage-2。
- 流程：参考 ARIS `research-refine` 的问题锚定、文献机制核对、最小证伪检查和独立审查；审查代理为 `gpt-5.6-sol`、`xhigh`。本轮候选未通过 Gate，未形成可执行的训练方案，也不把该诊断称为方法成功。

## 已核对的事实与候选

G1-Graph 和 G1-Identity 均调用 B0 的 `siamese_loss`：将 chart 端点直线距离拟合 `D_geo`。旧 G1 同构比较中，Graph 的 Mean/P95 为 0.399771/0.846337 m，Identity 为 0.398343/0.843197 m；Graph 的传播尚未形成可辨认的净几何收益。原论文 [2609](https://arxiv.org/html/2609.06204) 的纯图平滑目标若直接替换 geodesic stress，常数 chart 就能使目标为零，缺少防坍缩条件；其 Sionna ray-traced 数据和本项目实测天线数据也不能直接等同。[2412 的 geodesic path loss](https://arxiv.org/html/2412.01715) 用 chart 路径折线长度拟合图最短路长度，研究对象与本项目的 DICHASUS 数据更接近，但其效果不能移植为本项目结果；直接把旧任务文档里的 B0 FCF 路径方案当作 G1 改进也不成立。

## 离线 Gate 协议

执行 `python -m diagnostics.g1_loss_semantics.run`。使用 full-N `N=20,997` 的既有 fused、`D_geo` 和 G1-Graph `embedding.npz`；首先核对索引一致，重建原 k=20 图，并用 `scipy.sparse.csgraph.dijkstra(..., directed=False)` 抽样复现保存的距离。seed=10 随机选 64 个源，每源 256 个目标；保留物理端点距离大于 0.5 m 的 16,222 对。前 32 源只用于分组阈值和尺度标定，后 32 源只用于检验。短而较直的标定组为 hop≤16 且 GT 路径曲率≤1.099254；长而弯曲的检验组为 hop≥32 且曲率≥1.279918。曲率定义为同一最短路径在 GT 中的折线长度除以物理端点距离。

真实 UE 坐标只用于**离线诊断及已训练 chart 的仿射评价**，没有参与 G1 训练、checkpoint、图、loss 或推理规则。这里的 `D_geo→米` 系数仅用来检验距离语义，不进入任何模型。没有使用 GT 选择新训练配置。

## 结果与判断

| 离线量 | 数值 |
|---|---:|
| 复算路径距离与保存 `D_geo` 最大差 | 0.000031 |
| 短直路径标定 `α`（米 / `D_geo`） | 0.018233 |
| 长曲路径检验点对数 | 283 |
| 长曲路径 `α·D_geo` 中位数 | 7.539 m |
| 同组 GT 折线 / GT 端点中位数 | 9.611 / 7.107 m |
| `α·D_geo` 到 GT 折线 / 端点的绝对误差中位数 | 2.142 / 0.417 m |
| 每对“更接近折线”的比例 | 8.83% |
| 每对折线相对端点的误差优势中位数；按源聚类 bootstrap 95% 区间 | −1.773 m；[−1.993, −1.460] m |
| G1 长曲 / 长直端点相对 GT 的有符号拉伸中位数 | +2.79% / −2.91% |

**尺度敏感性：**若改用 662 个“长而较直”的标定点对，`α=0.020694`，比短直标尺高 13.5%；长曲路径的折线/端点误差中位数变为 1.110/1.481 m，比较方向翻转。这表明当前 `D_geo` 到物理长度的单尺度映射依赖路径长度。旧图谱确有一些与曲率相关的端点拉伸（长路径中 Spearman 0.345），但不足以证明把其 chart 折线压向现有 `D_geo` 会改善真实几何。

按研究审查的证伪标准，**当前 path loss 的“`D_geo` 可作稳定物理路径长度目标”前提未通过**。不启动 G1 Graph/Identity 的 path-vs-endpoint 四臂训练，也不把 2609 的纯图平滑损失直接塞进当前训练。这个结论只针对这些已有图、缓存、单 seed 和抽样路径；不能否定所有路径型目标。若将来继续研究 Stage-1 loss，应先确定无需 GT 参与训练的多尺度距离目标，并重新冻结对照与预算。

## 2026-10-25 补充：下一候选 loss 的定位证据

在 seed=17 的 59,995 个均匀随机非自身点对中，`D_geo`/G1-Graph 仿射图谱距离与真实端点距离的 Spearman 分别为 0.982649/0.991562；真实端点距离除以 `D_geo` 的四分位分组中位数从 0.014467 增至 0.019101。因此，简单用全局序关系替换当前 metric stress，既不能修复单尺度问题，也可能丢失图谱已有的排序优势。

按 timestamp 排序并使用 `0<Δt≤1 s` 后有 20,955 个连续三点组。真实位置、Graph、Identity 仿射图谱的相邻速度变化范数中位数分别为 0.056382、0.328982、0.338135；P95 分别为 0.190822、1.053172、1.166058。这只定位到图谱时间轨迹抖动，不证明其是 Mean/P95 误差的因果来源。新候选是保留原 geodesic stress、加入鲁棒时间速度变化项；完整对照及停止判据见 [ARIS experiment-plan](../../../refine-logs/EXPERIMENT_PLAN.md)。GT 未用于 loss、权重或 checkpoint。

代码与机器结果：[诊断入口](../../../diagnostics/g1_loss_semantics/run.py)、[result.json](result.json)。原 G1 完整评价及 Channel Chart：[主报告](../REPORT.md)、[原始四列图](../comparison/four_column_raw_channel_chart.png)、[仿射四列图](../comparison/four_column_affine_channel_chart.png)。
