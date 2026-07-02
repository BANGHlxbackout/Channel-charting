# 物理图与 Chart 分歧驱动的双向弹性精修

更新时间：2026-06-25

> **基线对齐修正（2026-06-25）**
>
> 重新核对导师原始 idea、Siamese FCF 实现和欧氏图消融后，本笔记原先有两项关键假设不成立：
>
> 1. ADP/time 物理图不是独立于 `Y0` 训练监督的第二套信息。`Y0` 本身就是用同一份 ADP/time geodesic distance 做 pair regression 得到的；更准确的关系是“显式源流形图”与“该流形的神经坐标近似”。
> 2. 当前 local-scaled graph 并没有丢弃物理距离大小，`W_ij=exp(-d_ij^2/(sigma_i sigma_j))` 已使用距离大小决定边权。普通 GLR 没有做的是“匹配非零目标边长”，而不是“只保留邻接”。
>
> 因此，原先把双图分歧直接解释为 chart 错误、把 `E_-` 解释为应拆除的伪邻居，超出了导师 baseline 和现有欧氏图实验能够支持的范围。尤其是 ADP/time geodesic metric 与二维 chart Euclidean metric 并非天然可直接做负边判定。**本方案当前撤回“有条件通过”结论，降级为待重新论证的探索方向，不能作为已经对齐导师方法的正式优化 idea。**
>
> 下文保留原方案仅用于审计“哪里跑偏”，不应作为当前 idea 继续引用或实施。

## 1. 需求边界与基本假设

本 idea 只优化当前主线 2：对已经训练好的 channel chart 做 post-hoc refinement / repair。

假设：

- 初始 chart 为 Siamese FCF 输出 `Y0`；
- 显式源流形证据为 ADP/time fused geodesic graph `G_phys`，它与 FCF 的训练目标同源；
- 不修改前端 metric，不重新训练 FCF，不使用真实坐标构造修复约束；
- 目标是把当前“物理图 + 普通 GLR”升级成方法级贡献，而不是继续调 `k`、`lambda` 或迭代次数。

## 2. 一句话 Idea

> 比较 ADP/time 物理图与初始 chart 图的局部邻域，只在两者发生强且稳定分歧的位置执行双向几何修复：拉近“物理上近但 chart 中远”的样本，推开“chart 中近但物理上远”的伪邻居，并用一致邻域约束一个平滑、稀疏的坐标位移场。

暂定名称：

**Physics-Chart Disagreement Guided Bidirectional Elastic Repair**

简称：

**PCD-BER**

## 3. 当前方法真正没有利用的信息

当前 refinement 为

```math
Z^\star
=
\arg\min_Z
\|Z-Y_0\|_F^2
+
\lambda\operatorname{tr}(Z^\top L_{\mathrm{phys}}Z).
```

`L_phys` 来自 ADP/time geodesic distance，并通过局部尺度核把距离大小编码为边权：

```text
哪些样本相连，以及不同物理近邻约束的相对强度。
```

普通 GLR 对所有物理边只执行一种操作：

```text
让 Zi 和 Zj 更接近。
```

它没有显式匹配目标边长，也没有区分三种“源流形约束与神经坐标实现的残差状态”：

1. 物理图和 chart 图都认为两点近：当前局部结构大概率已经一致；
2. 源图中近、chart 中远：FCF 可能没有充分拟合该局部流形关系；
3. chart 中近、源图中远：两种度量发生不一致，但仅凭该不一致不能判断 chart 邻接错误。

第三种情况原先被本笔记直接解释成伪邻居，这是不充分的。它也可能来自 geodesic distance 与二维 Euclidean distance 的度量差异、不可避免的降维失真，或 ADP/time metric 自身误差。欧氏图 max error 变差只能说明 chart 自图平滑不能稳定控制极端误差，不能证明具体某条欧氏边就是错误边。

## 4. 从双图分歧构造修复证据

分别构造多尺度近邻：

```math
\mathcal N_i^{p}(k)
=
\operatorname{kNN}_{G_{\mathrm{phys}}}(i),
\qquad
\mathcal N_i^{c}(k)
=
\operatorname{kNN}_{Y_0}(i).
```

使用多个邻域尺度 `k in K`，避免单个 `k` 的偶然性。

### 4.1 一致边

```math
E_0
=
\{(i,j):j\in\mathcal N_i^p(k)\cap\mathcal N_i^c(k)\}.
```

这些边不直接代表真值，只表示 source graph 与其神经坐标实现的局部关系一致，可能适合：

- 保持初始 chart 已经正确的局部几何；
- 估计物理距离与 chart 局部尺度之间的单调校准函数；
- 约束修复位移在可靠区域内连续。

### 4.2 吸引修复边

```math
E_+
=
\{(i,j):j\in\mathcal N_i^p(k),\;
j\notin\mathcal N_i^c(K_c)\}.
```

它表示：

```text
物理证据持续认为两点接近，但初始 chart 把它们放得过远。
```

这类边负责修复遗漏的物理邻域。

### 4.3 排斥修复边

```math
E_-
=
\{(i,j):j\in\mathcal N_i^c(k),\;
j\notin\mathcal N_i^p(K_p)\}.
```

它表示：

```text
初始 chart 把两点放得很近，但较宽范围的物理邻域仍不支持该关系。
```

原方案把这类边用于排斥，但目前缺少足够依据。即使跨多个 `k` 保持分歧，也不能自动解决“源图错、chart 图错，还是两种度量不可直接比较”的不可辨识性。因此 `E_-` 在当前基线下不能直接进入正式方法。

## 5. 无监督物理尺度校准

ADP/time geodesic distance 不等于真实米制距离，因此不能直接要求

```math
\|Z_i-Z_j\|=d_{ij}^{\mathrm{phys}}.
```

利用一致边 `E0`，拟合一个单调映射：

```math
g:
d_{ij}^{\mathrm{phys}}
\mapsto
\|Y_{0,i}-Y_{0,j}\|.
```

`g` 只负责把物理距离排序和局部尺度映射到当前 chart 的尺度，可使用 isotonic regression 或低自由度单调样条。它不使用真实坐标。

这样做的意义不是获得绝对距离，而是补充当前 GLR 不显式匹配非零目标边长的性质：

```text
修复后的邻居应符合物理距离所暗示的相对尺度，
而不是只满足“越近越好”。
```

## 6. 双向弹性修复目标

令

```math
Z=Y_0+\Delta,
```

其中 `Delta` 是待估计的坐标位移场。建议优化：

```math
\min_\Delta\;
\alpha\sum_i a_i\|\Delta_i\|_2^2
+
\beta\operatorname{tr}(\Delta^\top L_0\Delta)
+
\lambda_+
\sum_{(i,j)\in E_+}
c_{ij}^+
\rho\left(
\|Z_i-Z_j\|_2-g(d_{ij}^{p})
\right)
+
\lambda_-
\sum_{(i,j)\in E_-}
c_{ij}^-
\left[
m_{ij}-\|Z_i-Z_j\|_2
\right]_+^2
+
\eta
\sum_{(i,j)\in E_0}
c_{ij}^0
\rho\left(
\|Z_i-Z_j\|_2-\|Y_{0,i}-Y_{0,j}\|_2
\right).
```

各项含义：

- `||Delta_i||^2`：没有充分分歧证据时不移动节点；
- `tr(Delta^T L0 Delta)`：平滑的是修复位移场，而不是直接压平 chart 坐标；
- `E+`：把遗漏物理邻域拉回到校准后的目标距离；
- `E-`：把物理证据不支持的 chart 伪邻居推到安全间隔之外；
- `E0`：保护原本一致的局部几何；
- `rho`：使用 Huber、Geman-McClure 等鲁棒损失，降低错误物理边的破坏力。

这里的核心变化不是多加几个 loss，而是修复语义发生改变：

```text
当前方法：给定物理图，对全部坐标做单向低通平滑。

PCD-BER：先用双图分歧定位结构矛盾，再估计最小必要位移，
同时执行邻域补全、伪邻域拆除和可靠几何保护。
```

## 7. 为什么这比可靠性加权 GLR 更强

简单 reliability weighting 仍然只有吸引作用：

```math
\sum_{ij}c_{ij}w_{ij}\|Z_i-Z_j\|^2.
```

它最多回答“多相信这条物理边”，但不能回答：

- 哪些源图与 chart 的度量残差只是降维必然结果，哪些才需要修复；
- 为什么欧氏图会强化某些错误邻域；
- 已经正确的局部结构为什么不应被整体压缩；
- ADP/time geodesic distance 的相对大小如何进入修复。

PCD-BER 增加的不是一个置信度系数，而是三项方法级能力：

1. 双图分歧诊断；
2. 吸引与排斥并存的双向拓扑修复；
3. 对修复位移而非最终坐标施加弹性正则。

## 8. 与参考论文的关系

### ADP/time fused metric

提供 FCF 训练和 refinement 共同依赖的 radio-geometric source manifold。它不是独立于 `Y0` 训练监督的信息源，但在 refinement 阶段以显式图算子的形式重新施加同源流形先验。

### Uncertainty-aware geodesic loss

在 FCF 训练阶段建模 dissimilarity 和 geodesic path uncertainty，并匹配 chart geodesic length；PCD-BER 不重新训练 FCF，而是使用物理图与 chart 图的分歧构造后训练的正、负修复约束。

### graphLa+ / 普通 GLR

支撑“初始估计 + 图正则”的数学框架，但只有给定图上的吸引式平滑，不显式处理双图矛盾和错误 chart 邻域。

### Robust graph filter identification

说明 supporting graph 可能有 link perturbation；PCD-BER 不把任一图视为真值，而要求极端、跨尺度、局部支持一致的分歧才触发修复。

### GGLR / framelet / smoothness verification

共同提醒普通坐标低通不一定适合连续几何。PCD-BER 不直接换成更复杂频域算子，而是把平滑对象改成位移场，并保护一致区域的局部距离。

## 9. 当前数据给出的初步支持

使用现有 `20997` 个样本做了只读诊断：

- 诊断分数定义为：物理 top-128 邻居在 chart 中的归一化距离 `90%` 分位数，乘以 `1 + chart top-128 中未进入 physical top-512 的比例`；
- 物理 top-128 与 chart top-128 邻域平均重合率约为 `0.530`；
- 简单的全局邻域分歧与 baseline 定位误差几乎不单调相关，Spearman 约为 `-0.015`；
- 这说明不能把所有图分歧连续地解释成错误，也反对简单的全局 reliability weighting；
- 但最极端的 `1%` 双向分歧节点上，baseline 平均误差约为 `0.909 m`，当前物理图 GLR 后约为 `0.257 m`，平均改善约为 `0.652 m`；
- 最低 `1%` 分歧节点的改善约为 `0.124 m`。

因此当前证据更支持：

```text
图分歧不是全局误差分数，
但极端、稳定的双图矛盾可能定位当前物理精修收益最集中的结构区域。
```

这正好支持“局部触发、最小位移、双向修复”，而不支持对所有节点统一增强扩散。

上述数字只用于判断 idea 是否有数据基础，不等于该方法已经被验证。
其中真实坐标只用于事后计算定位误差和检查诊断分数是否有意义，不参与分歧分数、边集或修复目标的构造。

## 10. 可证伪的核心假设

该 idea 的核心假设不是“复杂方法一定更好”，而是：

> 当前 physical-graph refinement 的主要收益可能集中在 FCF 对源流形关系拟合最不充分的区域；但需要先找到与导师同源流形假设一致的残差定义，不能直接把 physical-graph 与 chart-graph 的邻域差集解释成错误拓扑。

如果后续发现：

- 极端双图分歧在不同种子、不同基础 chart 或不同数据集上不能稳定定位修复收益；
- 排斥边主要来自物理 metric 自身的歧义；
- 显式排斥没有优于仅吸引的鲁棒 stress；
- 一致区域保护没有降低过平滑或结构损伤；

那么该 idea 的核心机制不成立，应停止，而不是继续堆模块。

## 11. 创新边界与撞车判断

截至 2026-06-25 的当前参考论文和补充检索中，没有发现一篇工作同时完成：

```text
ADP/time physical graph
+ 已训练 channel chart
+ 双图分歧诊断
+ 吸引/排斥双向 post-hoc repair
+ displacement-field elastic regularization
```

但以下组成部分分别已有成熟邻居：

- dissimilarity/geodesic matching；
- signed graph 的吸引与排斥；
- graph Laplacian / graph signal restoration；
- robust graph filtering；
- manifold stress optimization。

因此不能声称各个数学组件首次提出。可防守的新颖性应限定为：

> 原拟议新颖性是将 source radio-physical graph 与 learned chart graph 的结构残差定义为 channel-chart repair signal；基线复核后，这一主张必须重新证明，因为两者共享训练监督且度量不天然等价。

这仍需在正式立项前继续做至少两库的系统撞车检索和最近邻论文逐项对比。

## 12. 审稿人最可能的三条拒稿理由

### 12.1 “图不一致不代表 chart 错，可能是物理图错”

成立。方法不能选择性相信物理图。防守必须依靠：

- 多尺度一致性；
- 局部路径或共同邻域支持；
- 鲁棒损失；
- 只处理极端分歧；
- 对吸引和排斥分别做机制消融。

### 12.2 “这只是 signed graph embedding 的 CC 应用”

若方法只写成正边拉近、负边推远，这条批评成立。真正的差异必须包括：

- 两类边由物理图与 learned chart 图的定向矛盾产生；
- 一致边用于无监督尺度校准和几何保护；
- 优化对象是最小弹性位移场，而不是从头做 signed embedding；
- 目标是修复已训练 FCF 的结构错误，而不是重新学习 chart。

### 12.3 “当前普通 GLR 已经很强，新增复杂度不值得”

这是最大现实风险。该方法只有在以下贡献成立时才值得：

- 保持或进一步降低 P95/max tail error；
- 明显减少一致区域的不必要位移；
- 对欧氏图强化伪邻域的现象给出直接、可视化、可量化解释；
- 在不同基础 chart 上保持同一修复机制。

## 13. 当前判断

当前基线对齐后的结论是：原版 PCD-BER **暂不通过**。吸引式“修复 FCF 对源流形的局部拟合残差”仍可能有研究价值，但排斥式伪邻居拆除没有从导师原始 idea 或欧氏图消融中得到充分推导。若后续保留此方向，必须从“同一源流形的坐标近似误差”重新推导，而不是从“两套独立图相互审计”出发。

它最值得保留的核心只有三件事：

```text
双图结构分歧；
吸引与排斥双向修复；
最小弹性位移场。
```

尺度校准、鲁棒损失和多尺度支持是使这三件事成立的必要技术细节，不应扩张成更多并列创新点。
