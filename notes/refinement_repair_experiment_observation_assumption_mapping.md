# Refinement / Repair 实验现象到文献假设的映射

更新时间：2026-06-25

本笔记继续学习参考论文，只吸收知识，不正式提出优化 idea，不安排新实验。它的作用是把已经完成的欧氏图和物理图 refinement 结果放回文献学习框架中，判断哪些假设已经被当前结果支持，哪些仍然只是论文提醒。

当前参考结果来自 `results/method_summary.md`。

| 方法 | CT | TW | KS | mean | median | P90 | P95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADP/time geodesic + metric MDS | 0.9845 | 0.9899 | 0.1063 | 0.5275 | 0.4003 | 0.8837 | 1.1303 | 9.1438 |
| ADP/time geodesic + Siamese FCF | 0.9957 | 0.9960 | 0.0840 | 0.4093 | 0.3722 | 0.7142 | 0.8558 | 7.8936 |
| 物理图 diffusion refinement | 0.9982 | 0.9982 | 0.0651 | 0.3187 | 0.2929 | 0.5587 | 0.6346 | 1.1414 |
| 欧氏 kNN 图 diffusion refinement | 0.9955 | 0.9956 | 0.0804 | 0.3944 | 0.3672 | 0.6607 | 0.7645 | 8.4133 |

## 1. MDS 到 Siamese FCF：`Y0` 不是坏 chart，而是强 chart 里仍有 tail error

Siamese FCF 相比 metric MDS 明显改善：mean `0.5275 -> 0.4093`，P95 `1.1303 -> 0.8558`，max `9.1438 -> 7.8936`，CT/TW 也更高。这说明当前 refinement 的起点不是一个失效 chart，而是一个已经很强的 learned chart。

这对假设 A 有直接影响：

```text
Y0 不能被粗略理解成“很差的初值”，而应理解成“整体几何较好，但尾部仍有结构性不稳定”的观测。
```

对应的文献支撑是 ADP/time、global CC、uncertainty-aware geodesic loss、TDoA/NLoS、LoS channel charting 这些工作共同说明：channel chart 的错误不一定是均匀小噪声，可能来自 geodesic shortcut、传播条件差异、metric ambiguity 或 hard users。

但当前结果还不能证明 tail error 的具体来源。它只能说明 `Y0` 的误差分布不是只看 mean 就足够，因为 max 仍然很大。

## 2. 欧氏 kNN 图 refinement：chart 内部图能平滑局部噪声，但不等于物理修复

欧氏图 refinement 相比未精修 FCF 有提升：mean `0.4093 -> 0.3944`，median `0.3722 -> 0.3672`，P90 `0.7142 -> 0.6607`，P95 `0.8558 -> 0.7645`。因此不能再写“换成欧氏图就不灵”这种绝对表述。

但欧氏图的 max 从 `7.8936` 变成 `8.4133`，并且 CT/TW 从 `0.9957/0.9960` 轻微降到 `0.9955/0.9956`。这个现象更准确地说明：

```text
chart-internal graph smoothing 可以改善多数样本的局部波动，但它完全由 `Y0` 的当前坐标结构诱导，在本次实验中没有稳定控制最坏点。它可能强化局部错误邻域，但现有结果尚未直接证明这一具体机制。
```

这不是简单调参结论，而是和 graphLa+、robust graph filter、smoothness verification 的文献机制一致：

- graphLa+ 支持“初值 + data-dependent graph regularization”是正规框架，所以欧氏图改善并不意外；
- robust graph filter 论文提醒 supporting graph 若有 link perturbation，图滤波会把错误图结构传播到输出；
- smoothness verification 论文提醒“坐标在这个图上平滑”是需要检验的假设，不能因为 smoothing 后 mean 变好就认为图完全正确。

因此，欧氏图结果对假设 B 和 F 的学习价值是：如果 `L` 来自 `Y0` 自己，它主要是 chart-internal self-denoising；而 ADP/time 图重用 FCF 的源 geodesic manifold，避免完全由当前输出坐标闭环构图。现有消融不能直接定位哪条 `Y0` 邻域错误。

## 3. 物理图 refinement：当前结果最强支持的是“显式源流形重用”的价值

物理图 refinement 相比未精修 FCF 改善更明显：mean `0.4093 -> 0.3187`，median `0.3722 -> 0.2929`，P95 `0.8558 -> 0.6346`，max `7.8936 -> 1.1414`。相比欧氏图，物理图也显著更好，尤其是 max error 从 `8.4133` 降到 `1.1414`。

这个结果最直接支持的不是“Laplacian smoothing 有效”，而是：

```text
ADP/time 图不从 `Y0` 输出坐标构建，而是重用训练 FCF 时使用的 geodesic source manifold，因此比输出自图更能约束当前尾部误差。它与 `Y0` 的训练监督同源，不能写成统计独立证据。
```

对应的论文依据主要来自 ADP/time fused metric 和 channel charting 相关工作：ADP/time dissimilarity 不是从 learned chart 本身读出来的，而是由 CSI 物理结构和时间连续性提供的另一套几何证据。graphLa+ 类图正则只说明“图正则可行”，但不能解释为什么物理图比欧氏图强；真正的解释来自物理图的证据来源不同。

同时，这个结果也不能被过度解释。它不证明 `L_phys` 永远可信，也不证明所有 ADP/time 边都是正确的。uncertainty-aware geodesic loss、TDoA/NLoS、LoS ambiguity、robust graph filter 仍然提醒：物理图本身也是 noisy evidence，只是当前数据上它比 chart-internal graph 更有修复力。

## 4. CT/TW/KS 的变化：物理图更像结构修复，欧氏图更像局部平滑

从结构指标看，物理图 refinement 把 CT/TW 从 `0.9957/0.9960` 提升到 `0.9982/0.9982`，KS 从 `0.0840` 降到 `0.0651`。欧氏图 refinement 的 KS 也从 `0.0840` 降到 `0.0804`，但 CT/TW 轻微下降。

这个差异可以这样学习：

```text
欧氏图 refinement 主要降低定位误差分布的普通分位数；
物理图 refinement 同时改善普通分位数、tail error 和拓扑保持指标。
```

这让“物理图 vs 欧氏图”消融的意义更清楚：它不是为了证明欧氏图一定比 baseline 差，而是为了区分两种机制：

- chart-internal graph：依赖 learned chart 自身的邻域，适合解释局部 denoising；
- radio-physical graph：依赖与 FCF 监督同源的 ADP/time source manifold，更适合解释“显式源流形正则强于输出自图平滑”；是否修复 learned-neighborhood failure 仍需单独诊断。

这也解释了导师原话里“换了就不灵”的合理含义：更可能是指换成欧氏图后，方法失去“信道物理方法”的关键机制，而不是数值上必须全面低于未精修 baseline。

## 5. 当前结果对应七个假设的证据强弱

| 假设 | 当前结果给出的证据 | 学习结论 |
|---|---|---|
| A. `Y0` 是 ordinary noisy chart | FCF 整体强，但 max error 仍大 | `Y0` 更像“整体好、尾部不稳”的 chart；不能只按均匀噪声理解 |
| B. `L_phys` 是可信物理图 | 物理图远强于欧氏图，尤其 max error | 当前数据支持 ADP/time 图有强独立修复价值，但不证明每条边都可信 |
| C. `W_ij` 只需表示相似度 | 当前没有边可靠性诊断 | 文献提醒相似度和置信度应区分，但当前结果还没有直接证明必须改 |
| D. 普通 GLR 适合 chart 坐标 | 当前普通 GLR 效果很好 | 不能武断说 GLR 过平滑；GGLR/framelet 只是提醒潜在边界 |
| E. 全局单一 `lambda` 足够 | 只看到一个固定配置的结果 | 当前结果不支持也不反驳局部/自适应权衡；不能把它当成已证明问题 |
| F. 固定图一次闭式解足够 | 物理图一次解已经很强 | 在固定 `L_phys` 目标下，一次闭式解是合理的；迭代只有在图或可靠性更新时才有文献意义 |
| G. post-hoc transductive 足够 | 当前指标只评估已有样本 | 结果不能说明新样本部署；RED/unrolling/on-the-fly CC 只是长期边界提醒 |

## 6. 不能从当前结果直接推出的内容

不能推出 tail error 一定来自 NLoS。NLoS 文献说明传播条件可能导致系统性偏差，但当前实验没有 NLoS mask 或传播标签。

不能推出物理图没有错误边。物理图表现很好，只说明它在当前配置下比欧氏图更有效；robust graph filter 和 uncertainty 论文仍然要求把图看成 noisy evidence。

不能推出普通 GLR 已经是最优正则。当前普通 GLR 有强结果，所以不能急着引入 GGLR、framelet 或 high-frequency module；但文献提醒这些是普通低通先验的边界。

不能推出迭代或可学习化必然有必要。固定图二次问题的一次闭式解已经完成该目标的优化；只有当图、可靠性、denoiser 或 out-of-sample 目标进入问题时，迭代/学习化才有文献动机。

不能把欧氏图实验说成失败。它是一个部分成功但机制不同的对照：它证明普通 chart-internal smoothing 能改善平均表现，却不能解释物理图对 tail error 的强修复。

## 7. 本轮学习结论

当前实验结果和参考论文合在一起，最清楚地支持三点学习结论：

1. 当前 refinement 的强处不是“用了 Laplacian”，而是“重新显式施加了训练 FCF 所依赖的 ADP/time source manifold”，并且构图不依赖当前输出坐标。
2. 欧氏图结果说明普通平滑确实有用，因此论文叙事不能把对照写成完全失效；真正差异在 tail error 和结构指标。
3. 文献提醒的复杂方向，例如 uncertainty、graph perturbation、GGLR、framelet、RED/unrolling，目前都还只是边界知识；它们需要先和当前结果中的具体假设违背对上，才能变成后续方法构思。

所以，本轮学习后的更稳判断是：

```text
主线2不是“怎么把扩散调强”，而是理解何时、为什么、通过哪类图证据，post-hoc diffusion 能把 learned chart 的错误结构拉回到 radio-physical geometry。
```
