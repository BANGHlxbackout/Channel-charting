# Refinement / Repair 中 `L_phys` 可靠性的论文学习

更新时间：2026-06-25

本笔记继续学习参考论文，只吸收知识，不正式提出优化 idea，不安排新实验。它聚焦当前主线2里最关键的一层：物理图 `L_phys` 为什么比欧氏图更有修复力，以及为什么它仍然不能被当成绝对真值。

当前 refinement 基线：

```math
Z^\star=\arg\min_Z \|Z-Y_0\|_F^2+\lambda\operatorname{tr}(Z^\top L_{\mathrm{phys}}Z)
```

这里的核心问题不是“怎么改 `L`”，而是先理解：

```text
L_phys 是什么证据？
这个证据什么时候可靠？
什么时候会把错误边变成错误坐标约束？
```

## 1. ADP/time 论文：物理图的价值来自显式源流形，但 metric 有病理场景

对应论文：`05_ADP_Timestamp_Dissimilarity_CC`

这篇论文是导师当前物理图的直接前置。它比较 CSI-based metric、timestamp-based metric、ADP metric、geodesic version 和 fused metric。对当前 refinement 来说，最重要的不是“又有一个 metric”，而是它证明 `L_phys` 可以来自 learned chart 之外的 radio-geometry evidence。

PDF 正文里有一个关键图示：近距离用户可能有“vastly different channel conditions”，使 CSI-based metrics 失效；空间上近但时间上远的样本会让 timestamp-based metric 失效。这说明 ADP/time fused metric 的合理性来自互补，而不是单一 metric 永远可靠。

对 `L_phys` 的学习：

- ADP/time graph 与 `Y0` 的欧氏邻域不同源，因此它能解释为什么物理图 refinement 比欧氏图 refinement 更有 tail repair 能力。
- fused metric 的意义是降低单一证据源的病理风险；它不是证明每条边都可信。
- geodesic graph 可以把局部 dissimilarity 扩展为全局结构，但 geodesic 路径本身也可能受错误局部边影响。

不能推出：

- 不能把 ADP/time metric 写成当前工作的原创贡献。
- 不能把 `L_phys` 写成 ground-truth graph。
- 不能因为物理图结果好，就忽略 metric pathology。

## 2. Uncertainty-aware geodesic loss：距离/边不是确定标签，而是带不确定性的监督证据

对应论文：`10_Uncertainty_Aware_Geodesic_Loss`

这篇论文的关键机制是：geodesic distance 或 dissimilarity 不应只作为一个确定数值进入 loss，而应考虑条件分布和不确定性。它还讨论通过图采样/路径采样生成可能的距离 realization。

这对当前 refinement 的含义很直接：

```text
W_ij 在 tr(Z^T LZ) 中不只是 similarity，也隐含 confidence。
```

如果一条边的 dissimilarity value 很小，但其物理不确定性很大，那么它不应和“相似且可信”的边拥有同样约束强度。当前 `L_phys` 把相似度和可信度压缩进一个边权，这在数学上简单，但在论文机制上是一个近似。

对 `L_phys` 的学习：

- 图边应被理解为 noisy physical evidence。
- 全局 `lambda` 本质上是“相信 `Y0`”和“相信物理图”的全局证据权衡。
- 若不同边的不确定性不同，固定图 + 单一 `lambda` 只是同质化近似。

不能推出：

- 不能简单说“加 uncertainty 权重”就是足够强的优化 idea。
- 该论文主要作用在训练 loss，不直接提供 post-hoc refinement 算法。
- 当前结果还没有直接测量每条 ADP/time 边的不确定性。

## 3. TDoA/NLoS 与 UNILocPro：传播条件改变物理证据可靠性

对应论文：`12_TDoA_Self_Supervised_NLoS`、`13_UNILocPro_Unified_Localization`

TDoA/NLoS 论文明确指出，现实场景中 LoS path 可能被遮挡，TDoA estimates 会变成 noisy/corrupted measurements，因此引入 binary masking coefficient 丢弃 NLoS-corrupted measurements。

UNILocPro 则把 model-based localization 和 channel charting 放进一个统一框架，并根据 LoS/NLoS identification 在 model-based 和 CC-based 方法之间 adaptive activation。它说明前沿工作并不是把所有物理约束一视同仁地塞进 loss，而是先判断传播条件，再决定使用哪类证据。

对 `L_phys` 的学习：

- NLoS 不是普通高斯噪声，而是系统性证据失效。
- 如果某些物理边来自 NLoS 或被遮挡路径污染，那么强扩散会把错误约束传播到 chart。
- “全局统一使用物理图”隐含了传播条件同质的假设。

不能推出：

- 当前数据没有 NLoS 标签或 mask，不能声称已经处理 NLoS。
- 不能把当前 tail error 直接归因于 NLoS。
- 不能用 UNILocPro 的复杂训练框架包装当前闭式后处理。

## 4. LoS Channel Charting：metric 错误可以是结构性歧义，不是随机噪声

对应论文：`30_LoS_Channel_Charting`

这篇论文的重要性在于提出 identifiability 视角：某个 channel distance 是否能让空间邻域被识别出来，不是默认成立的。PDF 中讨论了 phase-insensitive distance 的 ambiguity，并展示“空间上更近的用户，反而在距离度量下看起来更远”的现象。论文还给出 identifiable neighborhood 和 thresholding 的分析。

对 `L_phys` 的学习：

- 错误物理边可能来自 channel distance 的周期性、振荡性或系统参数诱发的歧义。
- 这类错误不是“边权轻微扰动”，而可能是邻域关系被反转。
- 如果这种歧义进入 `L_phys`，扩散会把错误邻域写成坐标接近约束。

不能推出：

- LoS/free-space 下的 PI distance 分析不能直接等价到 ADP/time fused metric。
- 但它强力提醒：无线物理距离不是自动可信，metric identifiability 是前置问题。

## 5. Physical Channel Fingerprints：物理证据有层次，ADP/time 不是唯一深度

对应论文：`31_Physical_Fingerprints_CC`

这篇论文强调 conventional channel charting 的 latent embedding 通常缺少直接物理解释，而 physical channel fingerprints 从底层物理信道模型中提取 compact、interpretable 的参数表示。论文还通过统计平均和模型拟合过滤 uncorrelated noise，保留 stable large-scale scattering characteristics。

对 `L_phys` 的学习：

- 当前 ADP/time graph 是一种物理证据，但仍偏“浅层物理表示”。
- 更深的物理表示可以来自 cluster、MPC、GBSM、PCF 等可解释信道参数。
- 物理图的可信度不仅由图算法决定，也由前端物理表示决定。

不能推出：

- 当前没有提取 PCF，不能声称使用了 physical fingerprint graph。
- PCF 论文主要服务 channel acquisition 和 physical channel knowledge，不是 post-hoc chart repair。

## 6. Robust graph filter：`L_phys` 本身也是观测对象，不是真图

对应论文：`33_Robust_Graph_Filter_Identification`

这篇论文从 GSP 角度指出，很多图信号处理任务默认 supporting graph 已知，但实际图可能有 link perturbations。论文讨论了 link creation/destruction、edge weight uncertainty，以及在图扰动下联合估计 graph filter 和 denoised graph。

这对当前方法非常关键。当前 refinement 不是只在 noisy `Y0` 上做去噪，还在默认 `L_phys` 是支持图。如果 `L_phys` 有错误边，扩散解不是“去噪”，而是在传播错误图结构。

对 `L_phys` 的学习：

- supporting graph should be treated as evidence, not truth。
- 错误边对高阶图滤波、多步扩散、强正则尤其危险。
- “`Y0` 自图可能传播已有邻域误差”是欧氏图 max error 变差的一种可能解释，但当前消融没有直接识别错误边；图尺度、密度和谱性质差异也可能造成该现象。

不能推出：

- 当前方法没有联合估计真实图，不能声称解决 graph perturbation。
- 不能把 robust graph filter 论文直接等价成 channel chart repair 方法。

## 7. Verifying smoothness：`L_phys` 可信还不够，chart 坐标还必须在图上平滑

对应论文：`34_Verifying_Smoothness_Graph_Signals`

这篇论文把 graph signal smoothness 作为需要验证的统计假设，而不是 GSP 方法的默认前提。它讨论低频图滤波、频域能量和 smoothness detector。

对当前 refinement 来说，即使 `L_phys` 的边是合理的，也还需要另一个条件：

```text
Y0[:,0] 和 Y0[:,1] 这两个 chart 坐标信号，在 L_phys 上确实应当是平滑/低频的。
```

如果物理图连接的是真实空间近邻，那么 chart 坐标应当接近，这个假设合理。但如果局部存在真实转弯、非均匀采样、边界、路径分叉或 graph shortcut，普通 GLR 会把真实几何变化也当成高频噪声。

不能推出：

- 不能因为 refinement 指标变好，就说 smoothness 假设已经被证明。
- 不能直接用 smoothness verification 论文当作 repair algorithm；它是诊断视角。

## 8. 把这些论文合成到 `L_phys` 的四层判断

现在可以把 `L_phys` 的可靠性拆成四层，而不是只问“物理图好不好”：

| 层级 | 需要回答的问题 | 相关论文 | 对当前 refinement 的含义 |
|---|---|---|---|
| 证据来源 | 这条边来自什么物理表示？ | ADP/time、PCF | ADP/time 图有物理来源，但不是唯一可能的物理证据 |
| 证据不确定性 | 相似度是否等于可信度？ | uncertainty-aware geodesic loss | `W_ij` 同时承担 similarity/confidence 是近似 |
| 传播条件 | 这条边是否受 LoS/NLoS 或遮挡影响？ | TDoA/NLoS、UNILocPro | 全局同质物理图可能忽略局部传播差异 |
| 图结构扰动 | supporting graph 是否有错边/缺边？ | LoS identifiability、robust graph filter | 错误边会被 diffusion 变成错误坐标约束 |
| 图信号前提 | chart 坐标是否应在这个图上低频？ | verifying smoothness、GGLR/framelet | `tr(Z^T LZ)` 是可检验假设，不是自动真理 |

这张表给出的学习结论是：

```text
物理图 refinement 的关键不只是 `L_phys` 比欧氏图“更好”，而是 `L_phys` 重用了不依赖 `Y0` 输出坐标构造的 ADP/time source manifold；但它与 FCF 训练监督同源，不能被称为统计独立的 radio-geometry evidence；
但论文同时要求我们把这份 evidence 看成带来源、带不确定性、带传播条件、带图扰动风险的证据。
```

## 9. 当前学习状态

到目前为止，我对 `L_phys` 的理解已经从“ADP/time 物理图”推进到：

```text
L_phys = 由无线物理表示构造的 supporting graph，
它可作为 learned chart 的外部修复证据，
但它本身仍是 noisy, condition-dependent, and hypothesis-dependent graph evidence。
```

这仍然不是正式优化 idea。它只是后续构思时必须遵守的文献约束：如果之后要讨论改 `L`、改边权、局部化、迭代或学习化，必须先说明它对应上述哪一层可靠性问题。
