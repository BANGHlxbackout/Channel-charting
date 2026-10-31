# Refinement / Repair 核心论文机制卡片

更新时间：2026-06-25

本笔记继续学习参考论文，只做知识吸收，不正式提出优化 idea，不安排新实验。和 `refinement_repair_paper_grounded_mechanism_matrix.md` 的区别是：前者是 34 篇总表，本笔记只把最关键的论文读成可复用的“机制卡片”。

阅读原则：

```text
不是：L 怎么改、lambda 怎么调、是否迭代。
而是：哪篇论文证明了哪个假设不总成立；如果这个假设在当前 refinement 中存在，才有资格讨论怎么改。
```

当前 refinement 基线：

```math
Z^\star=\arg\min_Z \|Z-Y_0\|_F^2+\lambda\operatorname{tr}(Z^\top L_{\mathrm{phys}}Z)
```

其中每个对象都对应一个论文问题：

- `Y0`：已训练 chart 是怎样的 noisy / biased observation；
- `L_phys`：物理图边是否可信；
- `tr(Z^T LZ)`：chart 坐标是否应该在该图上低频/平滑；
- `lambda`：相信 chart 初值和相信物理图先验之间的证据权衡；
- closed-form solve：固定图一次性后处理是否足够。

## 1. 无线物理证据层

### 1.1 ADP / timestamp dissimilarity：物理图的来源，不是精修创新本身

核心论文：`05_ADP_Timestamp_Dissimilarity_CC`

这篇论文给当前方法提供的不是“新 metric”，而是 `L_phys` 的合法来源。它说明 channel charting 不能直接依赖 raw CSI 欧氏距离，而要构造更贴近空间邻域的 dissimilarity。ADP 利用角延迟结构，timestamp 利用运动连续性，fused metric 再通过 geodesic graph 形成更稳定的全局关系。

对 refinement 的机制学习：

- `L_phys` 可以被解释成 radio-geometry graph，而不是任意 kNN graph。
- ADP/time 图独立于 `Y0` 的输出坐标构造，但与 FCF 的 pair-distance 训练监督同源；它适合作为显式 source-manifold regularizer，不能直接被写成统计独立的外部真值。
- geodesic graph 不是只为了训练 FCF；它也可以成为 post-hoc graph operator。

不能推出：

- 不能把 ADP/time metric 当作当前工作的原创点。
- 不能默认所有 ADP/time 近邻都是可靠物理边；timestamp 会受速度变化、轨迹交叉影响，ADP 会受多径和局部不可辨性影响。

对当前公式的影响：

```text
`L_phys` 的价值来自“重用显式 ADP/time source manifold”，不是来自 Laplacian 形式本身；其优势是相对 `Y0` 自图减少循环依赖，而不是引入完全独立的新监督信息。
```

### 1.2 Uncertainty-aware geodesic loss：物理距离是随机证据，不是确定标签

核心论文：`10_Uncertainty_Aware_Geodesic_Loss`

这篇论文最重要的机制不是某个具体 loss，而是把 dissimilarity / geodesic distance 从确定值改成带分布、带不确定性的监督信号。它显式处理 `p(d | Delta)` 这类条件关系，说明同样的 ADP/time dissimilarity 对真实距离的约束强度可能不同。

对 refinement 的机制学习：

- 当前 `W_ij` 同时承担 similarity 和 confidence 两个角色。
- 如果每条边的不确定性不同，单纯使用一个确定 `L_phys` 会隐藏不确定性。
- `lambda` 不只是全局超参，它也可以被理解为图先验可信度。

不能推出：

- 不能简单说“给边加 uncertainty 权重”就足够成为优化 idea。
- 该论文主要服务训练 loss，不直接解决 post-hoc refinement。

对当前公式的影响：

```text
tr(Z^T LZ) 中每条边的惩罚强度应被理解为物理证据强度；
如果证据强度不均匀，全局同质图先验就是近似。
```

### 1.3 TDoA / NLoS：传播条件决定物理证据是否可信

核心论文：`12_TDoA_Self_Supervised_NLoS`

这篇论文说明 TDoA、TRP 位置和 CIR 统计可以支撑 global-scale self-supervised localization，但 NLoS 会给 timing geometry 引入系统性偏差，因此需要识别和 mask NLoS-induced noisy measurements。

对 refinement 的机制学习：

- NLoS 不是随机噪声，而是物理证据失效机制。
- 如果某些边来自 NLoS 或被 NLoS 污染，它们不应和 LoS 边具有同样正则强度。
- “局部 refinement”是否合理，应看传播条件是否局部异质，而不是为了复杂而分块。

不能推出：

- 当前没有 NLoS mask 时，不能声称已经处理 NLoS。
- 不能把所有 tail error 都归因于 NLoS；它只是一个强候选机制。

对当前公式的影响：

```text
L_phys 不是单一可信对象；它可能包含不同传播条件下的边。
```

### 1.4 UNILocPro：重型融合框架提供对照边界

核心论文：`13_UNILocPro_Unified_Localization`

这篇论文的价值在于展示前沿 CC 方向如何处理 mixed LoS/NLoS：model-based geometry、CC loss、LoS/NLoS identification、GOSPA/Wasserstein metric、OT/map loss、timestamp triplet 等多组件一起工作。

对 refinement 的机制学习：

- CC 前沿趋势是多源物理证据融合。
- 当前 diffusion refinement 的优势可能在轻量、post-training、model-agnostic，而不是比重型训练框架更全面。
- 如果未来引用 UNILocPro，应把它当作“复杂训练融合”的对照，而不是直接竞争同一技术层级。

不能推出：

- 当前后处理 refinement 不能声称拥有 UNILocPro 的 map/OT/model-based anchor 能力。
- 不能用 UNILocPro 的性能叙事包装当前闭式精修。

对当前公式的影响：

```text
当前方法是“把物理证据写进 graph operator”，不是“把所有物理证据写进 training loss”。
```

### 1.5 LoS Channel Charting：距离失效可以是结构性歧义

核心论文：`30_LoS_Channel_Charting`

这篇论文分析 LoS MIMO-OFDM 下 PI distance 的 periodic / oscillatory ambiguity，并提出 identifiability 概念。关键点是：一个 channel distance 若不能可靠识别空间邻域，那么后续 charting 算法无法凭空恢复正确邻域。

对 refinement 的机制学习：

- 错误边可能来自距离度量的结构性歧义，而不是训练噪声。
- `L_phys` 的错误不是简单小扰动；某些系统参数会导致远点看起来近。
- thresholding / identifiability 思想说明“建图前验证边”比“建图后盲目平滑”更基本。

不能推出：

- 该文是 LoS/free-space 强假设分析，不能直接覆盖 mixed NLoS。
- 不能把 PI distance 的结论无条件迁移到 ADP/time fused metric。

对当前公式的影响：

```text
如果物理距离本身存在 ambiguity，L_phys 可能把错误邻域强行写进正则项。
```

### 1.6 Physical Channel Fingerprints：物理图可以更深，不一定停留在 ADP/time

核心论文：`31_Physical_Fingerprints_CC`

这篇论文用 cluster-based GBSM / PCF 表示更物理的 channel fingerprints，并强调 PCF 能 compactly capture physical channel characteristics，支持 sCSI reconstruction 等通信任务。

对 refinement 的机制学习：

- ADP/time 只是物理证据的一层；更深层的物理参数、cluster/fingerprint 表示也可以成为图构造依据。
- 如果要从 `L_phys` 本身升级，方向不一定是改权重公式，也可能是改物理表示。
- 下游通信任务可以成为评价 chart 物理性的另一类证据。

不能推出：

- 当前没有 PCF 提取时，不能声称使用了 physical fingerprint graph。
- PCF 论文面向 sCSI acquisition / channel knowledge，不是 post-hoc chart refinement。

对当前公式的影响：

```text
L_phys 的“物理性”有层次：ADP/time 是浅物理图，PCF/GBSM 是更深物理表示。
```

## 2. 图信号恢复层

### 2.1 graphLa+：后估计图正则是正规数学框架

核心论文：`15_Data_Dependent_Regularization_GL`

graphLa+ 的结构是：先用已有 reconstructor 得到 preliminary approximation，再基于它构造 graph Laplacian，把图拉普拉斯嵌入 regularization term。它证明这种 data-dependent graph regularization 可作为 ill-posed inverse problem 的稳定正则方法。

对 refinement 的机制学习：

- `Y0 -> graph regularization -> Z` 不是随意后处理，有 inverse-problem 文献邻居。
- 当前方法与 graphLa+ 的区别在于：图不是只来自初始解，而来自 ADP/time 物理证据。
- graphLa+ 可以支撑“post-estimation refinement”的数学叙事。

不能推出：

- graphLa+ 已经成熟，不能把“初值 + 图拉普拉斯正则”本身当作创新。
- graphLa+ 没有无线物理图语义，不能替当前方法证明物理图独特性。

对当前公式的影响：

```text
当前 refinement 是 graphLa+ 类框架在 channel chart 坐标上的物理图版本。
```

### 2.2 GGLR：普通 GLR 的先验可能不适合连续几何坐标

核心论文：`16_Gradient_Graph_Laplacian_Regularizer`

GGLR 论文指出普通 GLR 倾向于 piecewise constant / low-frequency graph signal，会在缓慢变化信号中产生 staircase effect。它提出基于梯度图的 regularizer，使 planar / piecewise planar signal 成为更自然的低能量对象。

对 refinement 的机制学习：

- chart 坐标不是类别标签，而是连续几何坐标。
- 对坐标值做 GLR 等价于鼓励相邻点坐标接近；这可能压平真实空间变化。
- 如果真实 chart 局部更接近平面/仿射变化，正则项应关注局部梯度或几何变化，而不是只惩罚坐标差。

不能推出：

- 不能直接照搬 GGLR，除非能定义 chart 上的局部梯度、采样坐标或 embedding coordinates。
- 不能仅凭 GGLR 文献就断定当前 GLR 过平滑；需要检查当前图信号是否出现这类问题。

对当前公式的影响：

```text
tr(Z^T LZ) 是“坐标值平滑”先验；它不是唯一合理的图几何先验。
```

### 2.3 High-frequency / Framelet：高频不等于错误

核心论文：`18_GNN_High_Frequency_Booster`、`32_Graph_Denoising_Framelet`

这两类论文共同提醒：图平滑常被解释为低通滤波，但图信号中的高频既可能是噪声，也可能是有用结构。Framelet / spectral denoising 还区分 feature noise 和 structure noise，并通过多尺度/变换域稀疏性处理图表示。

对 refinement 的机制学习：

- 当前闭式解 `(I + lambda L)^-1 Y0` 是低通型隐式扩散。
- 如果 chart 的真实局部转弯、边界、路径分叉表现为高频，盲目低通会损伤几何。
- 图结构错误和坐标噪声是两类不同问题，不能只用一个平滑项概括。

不能推出：

- 不能把所有高频都保留；也不能把所有高频都压掉。
- 不能未实现多尺度分析时就把当前方法写成 multi-scale graph denoising。

对当前公式的影响：

```text
精修的目标不是降低所有 graph frequency，而是区分错误高频和真实几何高频。
```

### 2.4 Iterated graphLa：迭代的意义是更新图或估计，不是重复滤波

核心论文：`23_Impedance_Inversion_Iterated_GL`

iterated graphLa 的核心是用 initial guess 构造 data-adaptive graph operator，再通过迭代得到更稳定的 reconstruction。论文也承认 iterated version 的理论性质仍需更严格证明。

对 refinement 的机制学习：

- 迭代的合理性来自 graph / estimate 的相互更新。
- 如果 `L_phys` 完全固定，当前二次问题一次闭式解已经是最优解；重复应用同一扩散算子没有新信息。
- 若要让迭代有文献依据，必须有 graph update、reliability update、local estimate update 或 denoiser update。

不能推出：

- 不能因为 iterated GL 有效果，就给当前固定 `L` 加“多轮 diffusion”。
- 不能忽视理论边界：迭代图正则比一次正则更难证明。

对当前公式的影响：

```text
one-shot vs iterative 的分界：L 或可靠性是否随 Z 更新。
```

### 2.5 Robust graph filter identification：图本身可能是观测噪声

核心论文：`33_Robust_Graph_Filter_Identification`

这篇论文面对的核心问题是：GSP 方法常假设 supporting graph 已知，但实际图可能有 link perturbations。若忽略图错误，依赖图 topology 的方法可能 suboptimal；论文联合估计 graph filter 和 denoised graph。

对 refinement 的机制学习：

- `L_phys` 本身是观测对象，不是真值。
- 错误边通过 graph filter / diffusion 会传播到输出坐标。
- 如果使用高阶滤波、迭代扩散或强正则，图错误的影响会更严重。

不能推出：

- 当前方法没有联合估计真实 graph，不能声称已解决 graph perturbation。
- 不能把“物理图”这个名字当成图无误的证明。

对当前公式的影响：

```text
精修不只是在 noisy Y0 上去噪，也可能需要面对 noisy L_phys。
```

### 2.6 Verifying smoothness：GLR 前提应被验证

核心论文：`34_Verifying_Smoothness_Graph_Signals`

这篇论文的重要性在于把“graph signal smoothness”作为可检验假设，而不是默认前提。它从图滤波/频谱/统计检测角度讨论如何判断观测图信号是否符合平滑模型。

对 refinement 的机制学习：

- 在当前方法中，`Y0[:,0]` 和 `Y0[:,1]` 被假设为 `L_phys` 上的平滑图信号。
- 如果这个假设不成立，GLR refinement 的理论解释会变弱。
- 先验证 smoothness，再谈 diffusion，比直接调 `lambda` 更符合 GSP 文献逻辑。

不能推出：

- 不能只因为 refinement 后指标变好，就说 smoothness 假设已被证明。
- 该文提供的是检测视角，不是直接的 chart repair algorithm。

对当前公式的影响：

```text
tr(Z^T LZ) 前应先问：chart coordinates 在这个物理图上真的应该低频吗？
```

### 2.7 RED / Unrolling / Deep GLR：后处理可以学习化，但目标应明确

核心论文：`26_Unrolling_Plug_and_Play_GGLR`、`27_Interpretable_Deep_Denoise_GL`、`28_Graph_Signal_Denoising_RED`

这些论文共同说明，图优化可以被展开成可训练层，也可以把 denoiser 显式写入正则项。它们强调可解释性、少参数、可初始化、可学习的求解过程。

对 refinement 的机制学习：

- 当前闭式 refinement 是最简单、最可解释的 fixed denoiser。
- 如果要解决 out-of-sample 或 deployment，学习化 refinement 才有明确动机。
- 学习化不应只是“学习 lambda”，而应对应 teacher/student、unrolled correction layer、或 explicit denoiser prior。

不能推出：

- 当前没有实现 unrolling/RED 时，不能把它写进方法贡献。
- 不应为了显得复杂而把简单闭式解改成黑箱网络。

对当前公式的影响：

```text
post-hoc fixed solver 与 learnable correction layer 的区别在于是否服务新样本和部署。
```

## 3. 跨论文合成：精修优化的真实问题不是旋钮，而是假设

从这些核心论文看，当前 refinement 公式的每个“可改点”都对应一个假设：

| 公式对象 | 当前默认假设 | 触发怀疑的论文机制 | 学到的判断问题 |
|---|---|---|---|
| `Y0` | 初始 chart 是带小噪声的几何观测 | hard users、NLoS、metric ambiguity、global distortion | `Y0` 的错误是普通噪声，还是结构性偏差？ |
| `L_phys` | 物理图边可靠 | uncertainty、LoS/NLoS、PI ambiguity、graph perturbation | 哪些边是可信物理约束，哪些边可能是错误传播通道？ |
| `W_ij` | 权重只表示相似度 | uncertainty-aware loss、NLoS mask、robust graph | 权重是否还应表达置信度？ |
| `tr(Z^T LZ)` | 坐标在图上应低频 | GGLR、high-frequency、framelet、smoothness testing | 低频先验是否会压掉真实几何结构？ |
| `lambda` | 全局统一权衡成立 | bias-variance、uncertainty、区域性 propagation | 证据权衡是否应边/点/区域自适应？ |
| one-shot solve | 固定图足够 | graphLa+、iterated GL、robust graph filtering | 是否有必要更新图或可靠性？ |
| post-hoc output | 只修当前 batch 即可 | on-the-fly CC、RED、unrolling | 是否需要支持新样本？ |

因此，后续如果要合理想优化 idea，不能从“多试几个 `lambda`”开始，而要从下面的论文问题开始：

```text
当前实验中究竟是哪一个默认假设最可能被违反？
```

这仍然不是正式 idea，而是论文学习后的判断框架。

## 4. 我目前对主线2的理解状态

现在对主线2的理解可以压缩为一句话：

> 精修不是一个 Laplacian smoothing trick，而是一个由 radio-physical evidence 和 graph-signal prior 共同约束的 post-training chart repair 问题。

但这句话目前仍是学习结论，不是投稿 claim。要变成论文 idea，还需要之后独立完成：

- 明确当前实验中哪个假设真的被违反；
- 找到可计算的证据；
- 设计只针对该假设的最小方法升级；
- 验证它不只是更复杂，而是解决了文献指出的问题。

本轮到此为止，仍不正式提出优化 idea。
