# Graph Laplacian Regularization / 图正则论文轻量检索

更新时间：2026-06-25

本笔记按 `light-literature-search` 流程重新检索 Graph Laplacian Regularization、graph signal denoising、graph total variation、GGLR、unrolled graph regularization 等图正则相关论文。目标是为 `diffusion_refined` 的 graph-based refinement / repair 主线提供正式期刊文献支撑。

## 1. 检索约束

- 主题：Graph Laplacian Regularization、Graph Signal Denoising、Graph Total Variation、Gradient Graph Laplacian Regularizer、graph smoothness prior、unrolled graph regularization。
- 时间：2022-2026。
- 来源：Crossref + OpenAlex 元数据交叉检索。
- 文献类型：优先正式期刊论文；不把 arXiv-only 作为核心引用。
- 期刊层级：优先 SCI / 中科院 2区以上高置信期刊，如 IEEE TSP、TIP、TPAMI、TGRS、TCSVT、IJCV、Pattern Recognition、Information Fusion、SIAM JSC、Expert Systems with Applications、Signal Processing 等。
- 分区说明：中科院分区需要用学校/机构数据库最终核验。本表先按期刊层级做高置信初筛。

## 2. 检索留痕

### Crossref

使用的 query：

```text
graph Laplacian regularization
gradient graph Laplacian regularizer
graph signal denoising
graph total variation regularization
weighted graph Laplacian regularization
graph Laplacian image restoration
graph Laplacian hyperspectral denoising
data-dependent graph Laplacian regularization
graph smoothness regularization signal processing
graph denoising framelet regularizers
unrolling graph Laplacian regularizer
graph Laplacian depth completion
```

结果：

```text
Raw hits: 240
Title-filtered and deduplicated relevant records: 58
```

标题过滤关键词：

```text
graph laplacian
graph signal
graph total variation
graph denoising
graph smoothness
framelet regularizer
```

### OpenAlex

使用的 query：

```text
graph Laplacian regularization
gradient graph Laplacian regularizer
graph signal denoising
graph total variation regularization
graph smoothness regularization
unrolled graph Laplacian regularization
```

结果：

```text
Raw hits: 150
Title-filtered and deduplicated relevant records: 45
```

## 3. SCI2+ 高置信核心候选（15 篇）

| # | 论文 | 年份 | 期刊 | DOI | 核心作用 |
|---:|---|---:|---|---|---|
| 1 | Manifold Graph Signal Restoration Using Gradient Graph Laplacian Regularizer | 2024 | IEEE Transactions on Signal Processing | https://doi.org/10.1109/TSP.2023.3343560 | GGLR 核心论文，直接解释普通 GLR 过平滑 / staircase 问题，是 refinement operator 升级的关键支撑。 |
| 2 | Unrolling Plug-and-Play Gradient Graph Laplacian Regularizer for Image Restoration | 2026 | IEEE Transactions on Image Processing | https://doi.org/10.1109/TIP.2025.3562425 | 将 GGLR 展开为 plug-and-play / trainable restoration 框架，支撑后续把 graph repair 做成可学习层。 |
| 3 | Graph Denoising With Framelet Regularizers | 2024 | IEEE Transactions on Pattern Analysis and Machine Intelligence | https://doi.org/10.1109/TPAMI.2024.3393131 | 多尺度 graph denoising，不只依赖低频平滑；支撑“保结构修复而非盲目低通”。 |
| 4 | Robust Graph Filter Identification and Graph Denoising From Signal Observations | 2023 | IEEE Transactions on Signal Processing | https://doi.org/10.1109/TSP.2023.3300632 | 图滤波器识别 + 图信号去噪，启发自适应滤波强度和图可靠性估计。 |
| 5 | Verifying the Smoothness of Graph Signals: A Graph Signal Processing Approach | 2024 | IEEE Transactions on Signal Processing | https://doi.org/10.1109/TSP.2024.3439554 | 将“图上平滑性是否成立”变成可检验问题；最贴近 chart physical consistency audit。 |
| 6 | A New Approach for Graph Signal Separation Based on Smoothness | 2024 | IEEE Transactions on Signal Processing | https://doi.org/10.1109/TSP.2024.3361712 | 按 smoothness 分离图信号成分，启发 chart error 分解：真实几何变化 vs 错误高频。 |
| 7 | Graph Signal Restoration Using Nested Deep Algorithm Unrolling | 2022 | IEEE Transactions on Signal Processing | https://doi.org/10.1109/TSP.2022.3180546 | graph signal restoration + deep unrolling，支撑把固定线性求解扩展为可解释可学习求解器。 |
| 8 | Fast Hyperspectral Image Denoising and Destriping Method Based on Graph Laplacian Regularization | 2023 | IEEE Transactions on Geoscience and Remote Sensing | https://doi.org/10.1109/TGRS.2023.3272906 | 高维数据上的 GLR 去噪/结构恢复，适合类比 CSI/chart 中的结构性错误修复。 |
| 9 | Superpixel-Based Graph Laplacian Regularized and Weighted Robust Sparse Unmixing | 2024 | IEEE Transactions on Geoscience and Remote Sensing | https://doi.org/10.1109/TGRS.2024.3440309 | weighted GLR + robust sparse modeling，支持边权/局部结构可靠性建模。 |
| 10 | Graph Signal Processing for Heterogeneous Change Detection | 2022 | IEEE Transactions on Geoscience and Remote Sensing | https://doi.org/10.1109/TGRS.2022.3221489 | 图信号处理用于异质数据变化检测，启发从图域识别结构异常。 |
| 11 | Deep Unrolled Weighted Graph Laplacian Regularization for Depth Completion | 2025 | International Journal of Computer Vision | https://doi.org/10.1007/s11263-024-02188-3 | weighted GLR + unrolling，与“带权物理图 refinement”结构接近。 |
| 12 | Graph Laplacian regularization for fast infrared small target detection | 2025 | Pattern Recognition | https://doi.org/10.1016/j.patcog.2024.111077 | GLR 用于突出小目标/异常结构，启发 chart outlier island / tail error repair。 |
| 13 | A Data-Dependent Regularization Method Based on the Graph Laplacian | 2025 | SIAM Journal on Scientific Computing | https://doi.org/10.1137/23M162750X | “初值 -> 数据依赖图 -> Laplacian regularization”数学结构最接近导师原方法。 |
| 14 | Graph-based semi-supervised learning with non-convex graph total variation regularization | 2024 | Expert Systems with Applications | https://doi.org/10.1016/j.eswa.2024.124709 | 非凸 GTV 比二次 GLR 更边缘保持，启发 topology-aware / edge-preserving repair。 |
| 15 | Dual adaptive graph laplacian regularization for high-fidelity hyperspectral image fusion | 2026 | Information Fusion | https://doi.org/10.1016/j.inffus.2026.104309 | adaptive / dual graph Laplacian，用于支持多图、多证据、自适应图正则。 |

## 4. 可作为补充但分区或相关性需核验的候选

| 论文 | 年份 | 期刊 | DOI | 备注 |
|---|---:|---|---|---|
| Depth Map Super-Resolution Based on Dual Normal-Depth Regularization and Graph Laplacian Prior | 2022 | IEEE Transactions on Circuits and Systems for Video Technology | https://doi.org/10.1109/TCSVT.2021.3104151 | 高水平期刊，图拉普拉斯作为深度图先验；可作为应用补充。 |
| Multi-dimensional visual data completion via weighted hybrid graph-Laplacian | 2024 | Signal Processing | https://doi.org/10.1016/j.sigpro.2023.109305 | weighted hybrid GL，适合支持多图修复；分区需核验。 |
| Fractional graph Laplacian for image reconstruction | 2024 | Applied Numerical Mathematics | https://doi.org/10.1016/j.apnum.2023.05.007 | fractional GL，方法相关；期刊层级相对核心表弱。 |
| Large Graph Signal Denoising With Application to Differential Privacy | 2022 | IEEE Transactions on Signal and Information Processing over Networks | https://doi.org/10.1109/TSIPN.2022.3205555 | 图信号去噪相关，但 TSIPN 分区需核验。 |
| Large-scale graph signal denoising: A heuristic approach | 2025 | Digital Signal Processing | https://doi.org/10.1016/j.dsp.2024.104914 | 大规模图信号去噪；期刊分区和创新强度需核验。 |
| Graph total variation and low-rank regularization for heterogeneous change detection | 2025 | Digital Signal Processing | https://doi.org/10.1016/j.dsp.2024.104825 | GTV + low-rank，适合作为 edge-preserving prior 补充。 |
| Deep Unrolled Graph Laplacian Regularization for Robust Time-of-Flight Depth Denoising | 2025 | IEEE Signal Processing Letters | https://doi.org/10.1109/LSP.2025.3539908 | 主题相关，但 Letters 层级不作为核心引用。 |
| Graph Signal Processing: History, development, impact, and outlook | 2023 | IEEE Signal Processing Magazine | https://doi.org/10.1109/MSP.2023.3262906 | 综述/背景文献，高引用，适合写图信号处理背景，不作为方法创新证据。 |

## 5. 对 diffusion_refined 的方法启发

### 5.1 普通 Laplacian smoothing 不是创新，只能是数学支架

最相关文献：#13。

`A Data-Dependent Regularization Method Based on the Graph Laplacian` 已经明确给出类似结构：

```text
initial estimate -> construct graph -> graph Laplacian regularized refinement
```

因此导师原始方法中的

```text
min_Z ||Z - Y0||_F^2 + lambda Tr(Z^T L Z)
```

不能写成新图正则方法。它更适合被定位为：**在 channel charting 中引入物理图证据的可解释修复算子**。

### 5.2 需要防止过平滑

最相关文献：#1, #3, #14。

这些论文共同说明：

- 二次 Laplacian 是低通平滑，容易压掉真实结构变化。
- GGLR / framelet / GTV 可以更好保留边缘、多尺度结构或非平滑细节。
- 对 chart repair 来说，不能把所有高频都当噪声；有些高频可能是真实几何转弯、遮挡边界、轨迹分段。

### 5.3 可以把“修图”升级为“诊断 + 选择性修复”

最相关文献：#4, #5, #6, #10。

可迁移的问题表述：

```text
Given an initial learned chart Y0 and a physical graph G,
first verify where Y0 is physically smooth/consistent on G,
then selectively repair inconsistent regions.
```

这比“全图扩散一下”更像论文贡献。

### 5.4 可学习化是后续升级，不是当前第一创新点

最相关文献：#2, #7, #11。

unrolling / plug-and-play / weighted GLR 可以支持后续版本：

```text
physical graph diffusion solver
-> unrolled repair layer
-> learnable reliability-weighted repair
```

但如果当前实验还在后处理 refinement 阶段，建议不要把“深度展开”写成主贡献，先作为 future extension 或 ablation motivation。

## 6. 最值得精读的 8 篇

围绕当前主线 `Y0 -> physical consistency audit -> graph-based repair -> Z`，优先级如下：

1. Verifying the Smoothness of Graph Signals: A Graph Signal Processing Approach.
2. Manifold Graph Signal Restoration Using Gradient Graph Laplacian Regularizer.
3. Graph Denoising With Framelet Regularizers.
4. Robust Graph Filter Identification and Graph Denoising From Signal Observations.
5. A Data-Dependent Regularization Method Based on the Graph Laplacian.
6. Graph Signal Restoration Using Nested Deep Algorithm Unrolling.
7. Deep Unrolled Weighted Graph Laplacian Regularization for Depth Completion.
8. Graph-based semi-supervised learning with non-convex graph total variation regularization.

这 8 篇分别覆盖：

- 图平滑性诊断；
- GLR/GGLR 修复算子；
- 多尺度保结构去噪；
- robust graph denoising；
- data-dependent graph regularization；
- unrolled graph restoration；
- weighted graph refinement；
- edge-preserving GTV prior。

## 7. 写作时的边界

可以写：

> Graph regularization and graph signal denoising provide mature mathematical tools for smoothing, restoring, and separating graph-supported signals. However, these works do not address the radio-geometry-specific failure modes of learned channel charts. Our work uses graph regularization as an interpretable repair operator, while placing the novelty on physical-consistency auditing and topology-aware chart repair.

不要写：

- 我们首次提出 Graph Laplacian regularization。
- 只要 Laplacian smoothing 就能解决 chart failure。
- 所有 chart 高频都是噪声。
- 图正则文献已经解决了 channel charting 的 shortcut / fold / tear / NLoS bridge 问题。

