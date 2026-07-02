# diffusion_refined idea 优化与论文叙事笔记

本笔记用于后续 idea 优化和论文写作，不安排新实验。这里的核心假设是：当前 idea 指的是“在 Siamese FCF 初始 channel chart 之后，用 ADP/time 物理图解一个 graph-harmonic / diffusion refinement 问题”。

## 1. 一句话判断

如果写成“我们在 channel chart 后面加了一个图拉普拉斯平滑”，这个 idea 很难单独成文，因为 graph Laplacian smoothing / graph signal denoising 已经是成熟方法。

更有希望的写法是：

> 将 Siamese FCF 看成 ADP/time geodesic source manifold 的带噪神经坐标实现，并通过同一源流形的显式图算子做隐式扩散精修，降低随机 pair regression 未能完全消除的坐标噪声与尾部失真。

这句话把贡献从“用了 Laplacian”转成了“用与 FCF 监督同源、但独立于输出坐标构造的显式流形算子校正神经坐标近似”。不能再把它写成两套统计独立证据相互纠错。

## 2. 为什么导师会说现在还不能发

从评审视角看，当前 idea 的主要风险不是“公式错”，而是“贡献显得太薄”。

| 风险 | 严重性 | 为什么会被质疑 | 防守方式 |
|---|---:|---|---|
| 贡献像简单组合 | Major | FCF、ADP/time dissimilarity、geodesic graph、Laplacian regularization 都已有文献基础；简单相加会被看成工程后处理 | 把贡献收窄为“physics-informed graph-signal refinement for channel chart coordinates”，强调图来源、目标信号、失真类型和证据链 |
| 机制证据不足 | Major | 只说性能提升不够，评审会问为什么是物理图起作用，而不是任意 smoothing | 使用物理图 vs 欧氏图的现有结果解释机制：欧氏图改善均值但不一定修复最坏点，物理图更可能修复 tail error |
| 叙事边界不清 | Minor/Major | 如果同时讲新 metric、新训练、新正则、新系统，贡献会散 | 只保留 2-3 个贡献：物理图精修公式、物理图与 chart 内部图的区分、tail-error 修复现象 |
| transductive 限制 | Minor | 当前闭式求解是样本级 refinement，新样本 out-of-sample 不是天然解决 | 在论文中主动声明边界：这是 post-hoc chart refinement，不是完整 online localization framework |

因此目前版本不是不能继续做，而是不能用“普通拉普拉斯精修”这个叙事去投。它需要被重新包装成一个更明确的研究问题。

## 3. idea 的五维评分

| 维度 | 当前分数 | 证据 | 如何抬高叙事 |
|---|---:|---|---|
| Higher | 7 | 物理图精修优于 Siamese FCF：mean `0.4093 -> 0.3187`，P95 `0.8558 -> 0.6346`，max `7.8936 -> 1.1414` | 不只报平均误差，突出 tail-error correction 和 topology metrics |
| Faster | 4 | 闭式线性系统相对重新训练轻量，但目前没有效率叙事 | 作为次要优点即可，不要把效率写成主贡献 |
| Stronger | 8 | 欧氏图精修 mean 有提升但 max 变差，源 ADP/time 图显著降低 max error，说明它在当前数据上对极端误差更稳 | 主打 robustness：尾部控制；错误邻域或局部畸变只能作为待验证机制 |
| Cheaper | 5 | 不需要重新训练 FCF，利用已有 ADP/time 信息，但还不是核心卖点 | 可作为“post-hoc, training-decoupled”优点，不要过度声称低成本 |
| Broader | 6 | 这是把 graph signal denoising 移植到 channel chart coordinate refinement | 可强调跨领域迁移：inverse problem / graph denoising 思想进入 CC 后处理 |

当前最适合主打的两个维度是 **Stronger** 和 **Higher**。不要把论文写成 Faster 或 Cheaper 论文。

## 4. 三种叙事版本

### 版本 A：弱叙事，不建议

> We improve channel charting by applying Laplacian smoothing to the learned chart.

问题：这会立刻引出“这不是早就有了吗”的审稿反应。Laplacian smoothing 本身不是新贡献。

### 版本 B：可用叙事

> We formulate post-hoc channel chart refinement as graph-signal denoising over an ADP/time physical graph.

优点：把方法放到 graph signal denoising 框架，公式清楚，相关工作好组织。

不足：如果只到这里，仍然可能被认为是“把现成 GSP 方法用于 CC”。

### 版本 C：更强叙事

> We show that the key to post-hoc channel chart refinement is not smoothing itself, but whether the graph is reconstructed from the FCF output or reused from the ADP/time geodesic source manifold: the former mainly performs self-smoothing, whereas the latter explicitly reimposes the manifold that supervised the FCF and yields stronger tail control.

优点：这句话把物理图 vs 欧氏图变成“输出自图平滑 vs 源流形算子重用”的机制消融，而不是错误地要求欧氏图必须失败。

建议后续论文以版本 C 为主线，版本 B 作为方法定义。

## 5. 本轮重新对齐：精修优化不是调参

当前不应该直接问：

```text
L 怎么改？
正则项怎么改？
lambda 怎么改？
全局还是局部？
一次还是迭代？
后处理还是可学习？
```

这些问题本身是对的，但不能当成调参菜单。正确顺序应该是：

```text
论文机制
-> 该机制解决了什么失败假设
-> 该失败假设是否存在于当前 chart refinement
-> 才能决定 L、正则项、lambda、迭代方式或学习方式是否该改
```

当前 baseline 可以写成：

```math
Z^\star = \arg\min_Z \|Z-Y_0\|_F^2 + \lambda \operatorname{tr}(Z^\top L_{\mathrm{phys}}Z)
       = (I+\lambda L_{\mathrm{phys}})^{-1}Y_0 .
```

这个公式隐含了几个强假设：

- `Y0` 是带噪观测，误差近似可由二次 fidelity 吸收；
- `L_phys` 是可信图；
- chart 坐标在 `L_phys` 上应该是低频/平滑图信号；
- 全局统一的 `lambda` 可以描述所有点、所有边的可信程度；
- 一次闭式扩散足够，不需要更新图或更新可靠性；
- refinement 只服务当前样本，不处理 out-of-sample。

后续优化如果要像论文贡献，必须针对这些假设中的某一个提出有文献依据的修正，而不是随机换一个 `lambda` 或随手改 `L`。

## 6. 论文机制到精修问题的映射

| 精修问题 | 相关论文机制 | 论文真正说明了什么 | 对当前 refinement 的学习 |
|---|---|---|---|
| `L` 是否可信 | 10 uncertainty-aware geodesic loss；12 TDoA/NLoS；13 UNILocPro；30 LoS CC；31 physical fingerprints CC；33 robust graph filter identification | 无线几何证据不是确定真值。ADP/time、PI distance、LoS path、NLoS path、physical fingerprints 都可能有条件性、歧义和观测扰动 | 不能把 `L_phys` 当作绝对正确图；更合理的问题是：哪些物理边可作为修复约束，哪些边应被降权、剔除或验证 |
| 正则项是否该是普通 GLR | 16 GGLR；18 high-frequency booster；20 Retinex gradient GL；24 time-varying graph signal；32 graph denoising framelet | 普通 GLR 偏向低频 / piecewise constant，可能产生 staircase、过平滑，并压掉真实高频结构 | 对 chart 坐标直接做 `tr(Z^T L Z)` 只是最简单先验；若 chart 几何包含真实梯度变化或局部方向结构，正则项可能应从“坐标平滑”变成“几何变化/梯度/多尺度结构约束” |
| `lambda` 是否只是调参 | 16 GGLR；28 RED；15 graphLa+；34 verifying smoothness | 正则强度应对应 bias-variance、噪声水平、先验可信度或 smoothness hypothesis，而不是纯经验按钮 | `lambda` 可以被理解为“相信初始 chart”与“相信物理图先验”的证据权衡；如果有边/点不确定性，单一全局 `lambda` 的假设会变弱 |
| 全局还是局部修复 | 10 geodesic loss；13 UNILocPro；30 LoS CC；34 smoothness verification | 不同区域、不同传播条件、不同图频率下，物理相似性不一定同质；LoS/NLoS 或非凸区域会改变几何证据可靠性 | 不应先验假设全局统一扩散一定合适；是否局部化要由“物理证据在不同区域是否同质”来决定 |
| 一次还是迭代 | 15 graphLa+；23 iterated graphLa；33 robust graph filter identification | 当图依赖初始解或图本身有扰动时，一次固定图正则不一定够；迭代的意义是更新图/估计量，而不是重复平滑 | 只有当 `L`、边可靠性或 chart-induced evidence 会随 `Z` 改变时，迭代才有论文依据；单纯多跑几次 diffusion 不是贡献 |
| 后处理还是可学习 | 26 unrolling PnP/GGLR；27 interpretable deep denoise GL；28 RED；32 framelet denoising | 优化过程可以展开成少参数、可解释、可训练的层；denoiser 可以显式进入正则项或算法迭代 | 如果要解决 transductive 或 out-of-sample 弱点，可以把 refinement 变成 teacher、layer 或轻量 correction module；但这已经是比当前后处理更复杂的方向 |
| 是否需要先验证平滑假设 | 34 verifying smoothness of graph signals；33 graph perturbation | GSP 方法常默认信号在图上平滑，但有论文专门研究如何检验 smoothness 假设、处理图扰动 | 当前 refinement 的第一性问题是：`Y0` 的两个坐标在 `G_phys` 上是否真的应该平滑；如果不检验，方法容易被批评为套用 GLR |

## 7. 主线2核心论文学习

### 7.1 无线物理证据论文给出的约束

ADP/time/fused metric 论文线说明，channel charting 的关键不是 raw CSI 欧氏距离，而是找到能保留局部空间邻域的 radio dissimilarity。它支持当前使用 ADP/time 图作为物理证据，但也提醒：metric 好坏通常只通过最终 chart 表现体现，本身仍可能产生错误邻域。

Uncertainty-aware geodesic loss 更进一步：dissimilarity 不应被当作确定距离，而应建模为条件分布或带方差的观测。它直接反驳了“物理图边一定可靠”的简单假设。对 refinement 来说，最重要的不是照搬它的训练 loss，而是学习它的思想：物理边是 noisy geometric evidence。

TDoA/NLoS 和 UNILocPro 说明 LoS/NLoS 会决定 model-based geometry 是否可信。UNILocPro 用 LoS/NLoS identification、model-based anchor、CC loss、OT/map 共同处理 mixed propagation。对当前 refinement 的启发是：物理约束不应在所有点上等价使用；传播条件会改变修复约束的可信度。

LoS Channel Charting 论文分析 PI distance 的 periodic / oscillatory ambiguity，并从系统参数角度讨论 identifiability。它对 refinement 的价值是：无线距离或相似度的失效可以是结构性的，不是普通噪声。错误边可能来自 metric ambiguity。

Physical Fingerprints CC 把 channel chart 推向更物理的 cluster / GBSM / PCF 表示。它提醒当前 ADP/time 图只是物理证据的一种浅层形式；如果未来要增强 `L_phys`，更深的方向可能是利用物理参数或 fingerprint 表示，而不是只在 ADP/time 权重上做小修。

### 7.2 图正则论文给出的约束

graphLa+ 的核心不是“用了图拉普拉斯”，而是“用初始解构造 data-dependent graph，再做正则化”。这与当前 `Y0 -> refinement` 很接近，但 graphLa+ 也说明这个框架已经成熟，所以它只能作为数学支架，不能当作创新本身。

iterated graphLa 的意义是：当图由初始估计诱导时，迭代可以让图和解互相改进。迁移到当前任务时，只有当物理图、chart 图或边可靠性会随 refined chart 更新时，迭代才有意义。若 `L_phys` 完全固定，迭代只是重复应用低通滤波，贡献很弱。

GGLR 明确指出普通 GLR 会偏向 piecewise constant，并造成 staircase；它提出用 gradient graph Laplacian 促进 piecewise planar。对 chart refinement 来说，这很关键：二维 chart 坐标不是标签常数，而是连续几何坐标。普通 GLR 可能压平真实空间变化，因此正则项可能需要描述局部几何梯度，而不只是让邻居坐标相等。

High-frequency booster 和 framelet denoising 共同说明：低频平滑不是总是正确。图信号中的高频可能是噪声，也可能是有用结构。当前 refinement 如果过度降低所有高频，会修掉真实转弯、边界或局部几何变化。

Robust graph filter identification 说明实际图可能是 perturbed graph，使用错误图做图滤波会放大误差，尤其高阶滤波或多步传播会传播图错误。对当前 refinement 的直接提醒是：若 `L_phys` 有错误边，扩散会把错误边变成错误坐标约束。

Verifying Smoothness of Graph Signals 说明 smoothness 本身是可检验假设，而不是默认成立。当前工作若要严谨，需要能说清楚 chart 坐标在 ADP/time 图上为什么应该平滑，或者至少能度量什么时候不平滑。

RED / unrolling / deep GLR 论文线说明，图正则求解可以变成可解释的 denoiser、unrolled layer 或 teacher。它不是当前立刻要做的实验，但它指出后处理版本的长期弱点：transductive，不能自然处理新样本。

## 8. 本轮学习得到的精修认识

第一，`L` 不是“换一个图”的问题，而是物理证据建模问题。CC 文献反复说明 metric、LoS/NLoS、PI ambiguity、physical fingerprint 都会影响图边是否可信。因此后续若改 `L`，必须说清楚是基于哪类物理证据、解决哪类错误边。

第二，普通 GLR 的含义是很强的：它认为相连点的 chart 坐标应相近，等价于低频/平滑先验。GGLR、framelet、高频相关论文说明这个假设可能过强。因此“改正则项”不是为了花样，而是为了避免把真实几何变化当成噪声。

第三，`lambda` 不应被理解成单纯实验超参。它本质上是 `Y0` fidelity 和 graph prior 的证据权衡。若不同点或边的可信度不同，全局 `lambda` 就是在假设所有区域噪声和先验可靠性相同。

第四，全局 refinement 是否合理，取决于物理图证据是否全局同质。LoS/NLoS、非凸区域、metric ambiguity 都说明这个条件不一定成立。局部化的依据应该来自物理证据分布，而不是为了做复杂方法。

第五，迭代 refinement 的论文依据来自“图或可靠性随解更新”。如果当前只固定 `L_phys`，一次闭式解已经是该二次问题的最优解；继续迭代没有新的数学意义。要让迭代成立，必须有 graph update、reliability update 或 denoiser update。

第六，可学习 refinement 的依据来自 RED/unrolling/deep GLR，但它对应的是另一个层级的问题：如何把后处理变成可部署、可泛化的 correction module。它不能被简单写成“把 lambda 学一下”，而应和 out-of-sample 或 teacher-student 问题绑定。

## 9. 当前不能草率声称的内容

不能说“欧氏图换了就不灵”。现在实验已经显示欧氏图比未精修好，所以更准确的理解是：导师原话的预期大概率是“欧氏图不能证明物理方法的独特性，或者不能像物理图那样修复关键 tail error”，不是“欧氏图一定比 baseline 差”。

不能把我之前写的 shortcut、fold、tear 等 failure mode 当成文献既有分类。它们最多是基于 geodesic mismatch、NLoS、metric ambiguity、over-smoothing 等论文现象总结出的工作假设。除非后续给出可计算定义和论文证据，否则不能当作正式理论。

不能把 reliability-weighted diffusion 当成已经足够的优化 idea。它确实有 uncertainty/NLoS 文献依据，但如果只是在边权上乘一个置信度，创新仍可能偏弱。它必须连接到更具体的物理证据模型或图扰动鲁棒性问题。

不能把 graphLa+ 当成新贡献。它最多证明“初始解 + 数据图正则”是成熟且可辩护的数学框架。当前工作的创新必须来自 channel charting 的物理图、图可信性、修复对象或部署方式。

## 10. 下一步只继续学习时应该重点读什么

暂时不推进实验。下一轮学习应继续围绕主线2，只问这些问题：

1. `L_phys` 的错误边如何从论文中被建模：重点复读 10、12、13、30、33。
2. 普通 GLR 对 chart 坐标是否过强：重点复读 16、18、32、34。
3. 固定一次闭式 refinement 的边界在哪里：重点复读 15、23、28。
4. 后处理如何变成可泛化模块：重点复读 26、27、28，但只作为长期方向。

补充学习矩阵见：`notes/refinement_repair_paper_grounded_mechanism_matrix.md`。该矩阵把 34 篇论文逐篇映射到 refinement 假设、可迁移机制和不能草率推出的结论。

核心论文机制卡片见：`notes/refinement_repair_core_mechanism_cards.md`。该卡片进一步把 ADP/time、uncertainty、NLoS、GGLR、graphLa、RED、robust graph filter、smoothness verification 等核心论文压缩成“机制 -> 当前公式影响 -> 不能推出什么”。

假设审计学习笔记见：`notes/refinement_repair_assumption_audit_learning.md`。该笔记把当前 refinement 公式拆成 `Y0`、`L_phys`、`W_ij`、GLR 正则、`lambda`、one-shot solve、post-hoc 七个假设，用于后续 idea 前判断“到底哪个假设被论文证据挑战”。

实验现象到文献假设的映射见：`notes/refinement_repair_experiment_observation_assumption_mapping.md`。该笔记把当前 MDS、Siamese FCF、物理图 diffusion、欧氏 kNN diffusion 的结果放回七个假设中，区分哪些判断已有实验支撑，哪些仍只是文献边界提醒。

`L_phys` 可靠性专题学习见：`notes/refinement_repair_lphys_reliability_learning.md`。该笔记从 ADP/time、uncertainty-aware geodesic loss、TDoA/NLoS、UNILocPro、LoS identifiability、physical fingerprints、robust graph filter、smoothness verification 这些论文中，专门整理物理图作为修复证据的来源、失效机制和不能越界声称的内容。

图正则算子先验专题学习见：`notes/refinement_repair_operator_prior_learning.md`。该笔记从 graphLa+、GGLR、framelet graph denoising、robust graph filter、iterated graphLa、RED/unrolling、smoothness verification 等论文中，整理当前 `tr(Z^T LZ)` 闭式扩散算子的默认假设和边界。

学习覆盖审计见：`notes/refinement_repair_literature_coverage_audit.md`。该笔记把 34 篇参考论文按当前学习深度分成机制层、叙事/边界层和外围支撑层，并明确后续继续学习应补强 CC failure diagnosis 与训练/后处理边界。

CC 失效证据与训练—后处理边界专题见：`notes/refinement_repair_cc_failure_and_boundary_learning.md`。该笔记精读 01、03、04、06、08、14 六篇论文，严格区分论文明确支持的 false-neighbor short circuit、局部 metric 有效范围、非均匀采样、非仿射失真、物理约束歧义、NLoS embedding collapse 等机制与自行归纳的术语，并逐篇划清 metric design、training-time supervision、affine alignment、environment engineering 和 post-hoc repair。

多源约束、下游任务、环境工程与图高频专题见：`notes/refinement_repair_multisource_downstream_environment_frequency_learning.md`。该笔记精读 07、09、11、18，说明多源物理信息分别约束 absolute range、relative displacement 和 global offset；记录 chart 到 channel-prediction/sum-rate 之间的完整评价链；确认 11 是 14 的早期版本，不能重复计数；并限定 GNN heterophily/high-frequency 结论不能直接迁移为二维 chart 坐标的高通精修。

本轮阶段性结论是：优化精修的入口不是“调哪个参数”，而是先判断当前 refinement 违反了哪一个文献中已经指出的假设：图可能不可信、图信号未必低频、物理证据不均匀、固定一次后处理不可泛化。只有对应到这些假设之后，才谈得上怎么改 `L`、正则项、`lambda`、局部性、迭代或学习化。
