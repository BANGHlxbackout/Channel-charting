# Refinement / Repair 中图正则算子先验的论文学习

更新时间：2026-06-25

本笔记继续学习参考论文，只吸收知识，不正式提出优化 idea，不安排新实验。上一份笔记聚焦 `L_phys` 作为物理图证据是否可信；本笔记聚焦另一个核心：当前精修算子

```math
\operatorname{tr}(Z^\top LZ)=\frac12\sum_{i,j}W_{ij}\|Z_i-Z_j\|^2
```

到底默认了什么，以及图正则文献为什么提醒“普通 GLR 有用但不能被神化”。

当前 baseline：

```math
Z^\star=\arg\min_Z \|Z-Y_0\|_F^2+\lambda\operatorname{tr}(Z^\top L_{\mathrm{phys}}Z)
       =(I+\lambda L_{\mathrm{phys}})^{-1}Y_0
```

这里的学习问题不是“下一步换什么正则项”，而是：

```text
普通 GLR 在当前 chart refinement 中到底表达什么先验？
它在哪些条件下合理？
哪些论文说明它可能不够？
哪些复杂图正则目前只能作为边界知识，不能直接写成贡献？
```

## 1. graphLa+：当前公式有成熟数学邻居，但这也意味着它不是创新本身

对应论文：`15_Data_Dependent_Regularization_GL`

graphLa+ 的核心结构是：

```text
preliminary approximation -> construct graph -> graph Laplacian regularized reconstruction
```

PDF 摘要和正文都强调，它把 graph Laplacian operator 嵌入 regularization term，并基于初始近似构造 data-dependent graph。它还说明这种方法能形成 stable、convergent、mathematically robust 的 regularization operator。

对当前 refinement 的学习：

- 当前 `Y0 -> L_phys -> Z` 的结构不是随意后处理；它可以被放进 inverse problem / Tikhonov regularization 的成熟框架。
- 但正因为 graphLa+ 已经建立了“初值 + 图拉普拉斯正则”框架，普通 Laplacian refinement 本身不能作为主要创新。
- 当前工作若有独特性，应来自 channel charting 的 radio-physical graph 和 chart repair 语义，而不是来自二次 GLR 形式。

不能推出：

- 不能写“首次提出 graph Laplacian refinement”。
- 不能把 graphLa+ 的数学稳定性直接搬到当前 `L_phys`，因为当前图来自 ADP/time 物理证据，不是 graphLa+ 原始图构造。

## 2. 普通 GLR：它是低频/坐标接近先验，不是所有几何修复的同义词

对应论文：`16_Gradient_Graph_Laplacian_Regularizer`、`34_Verifying_Smoothness_Graph_Signals`

普通 GLR 的含义很具体：

```math
x^\top Lx=\sum_{(i,j)}W_{ij}(x_i-x_j)^2
```

在节点域，边权大的相邻点被鼓励取相似值；在频域，低 GLR 表示图信号能量更集中在低图频率。对当前二维 chart 来说，`Z[:,0]` 和 `Z[:,1]` 两个坐标分量都被当作 `L_phys` 上的平滑图信号。

这在物理近邻图正确时很合理：真实物理邻居在 chart 坐标里也应接近。但它同时带来强假设：

```text
所有连接边上的坐标差都更像噪声，而不是需要保留的真实几何变化。
```

对当前 refinement 的学习：

- 当前闭式解本质上是一个 low-pass implicit diffusion filter。
- 它适合修局部抖动、孤立坏点、普通噪声型坐标误差。
- 它不天然区分“错误高频”和“真实几何高频”。

不能推出：

- 不能把所有高频都当成 chart failure。
- 不能只因为指标变好，就说 chart 坐标在 `L_phys` 上的 smoothness 假设已被证明。

## 3. GGLR：普通 GLR 可能产生 staircase，连续几何坐标不一定适合 piecewise constant 先验

对应论文：`16_Gradient_Graph_Laplacian_Regularizer`

GGLR 论文明确指出，GLR 用于 signal restoration 时会促进 piecewise smooth / constant reconstruction；它还讨论了 GLR 对 gradually changing signals 的 staircase effect，即把本来连续变化的信号恢复成相邻常值块。

这对 chart refinement 很重要。二维 chart 坐标不是类别标签，也不是希望在局部保持常数的图标签；它更像连续几何场。若沿着真实轨迹坐标应逐渐变化，普通 GLR 可能把这种变化压低。

对当前 refinement 的学习：

- 普通 GLR 在数学上鼓励“邻居坐标相近”，但不懂局部仿射、局部方向、曲率或连续梯度。
- 若 chart 中存在真实转弯、边界、非均匀采样，过强 GLR 可能损伤几何解释。
- GGLR 的贡献是把先验从“值平滑”推向“梯度/局部平面结构平滑”。

不能推出：

- 当前结果中普通 GLR 效果很好，所以不能武断说必须换 GGLR。
- 不能直接照搬 GGLR，除非当前问题中能定义可靠的局部坐标、梯度估计或 chart manifold structure。
- GGLR 是边界提醒，不是当前已经完成的贡献。

## 4. Framelet / 多尺度图去噪：噪声不只有 feature noise，也可能有 structure noise

对应论文：`32_Graph_Denoising_Framelet`

framelet 图去噪论文强调，现实图数据不只是特征有噪声，图结构也可能有噪声。它的框架把 feature information 和 structure information 分开处理，并利用 framelet/multiscale regularizer 清理图表示。

迁移到当前问题，chart refinement 里的噪声至少有两类：

```text
坐标信号噪声：Y0 的二维坐标不准；
图结构噪声：L_phys 或欧氏图里的边不准。
```

当前二次 GLR 主要处理第一类：在给定图上平滑坐标。它没有显式处理第二类：图本身的边可能错。

对当前 refinement 的学习：

- 单一 GLR 把问题简化为“given graph, denoise signal”。
- 但参考论文提醒，图去噪问题也可能是“denoise signal + denoise structure”。
- 多尺度思想说明局部/全局结构可能不应由同一个低通尺度处理。

不能推出：

- 当前没有 framelet 变换，不能把方法写成多尺度图去噪。
- 不能因为 framelet 文献强，就认为当前必须改成 framelet；需要先证明单尺度低通损伤了 chart 结构。

## 5. Robust graph filter：高阶滤波和强扩散会放大图错误风险

对应论文：`33_Robust_Graph_Filter_Identification`

robust graph filter 论文指出，GSP 常默认 supporting graph 已知，但现实中图可能存在 link perturbation，包括错误建边、漏边、边权不确定。论文的图示还展示了图矩阵幂次在扰动下误差会变化，说明高阶图滤波与图错误之间关系很敏感。

对当前 refinement 的学习：

- 当前闭式解可以看作图滤波器 `(I+\lambda L)^{-1}`。
- 如果 `L` 有错误边，滤波器不是只消除 `Y0` 噪声，也会传播错误图结构。
- `lambda` 越大、扩散越强，越需要相信 supporting graph。

不能推出：

- 当前方法没有做 robust graph filter identification。
- 不能把“物理图”这个名字当成图无扰动的证据。
- 不能为了增强效果盲目加强扩散；图错时强扩散可能更危险。

## 6. Iterated graphLa：迭代的意义是更新图或估计，不是重复同一个滤波器

对应论文：`23_Impedance_Inversion_Iterated_GL`

iterated graphLa 的关键不是“迭代次数更多”，而是每一步利用更好的近似更新 graph Laplacian 权重，从而更好识别 ground-truth signal 中的连接关系。PDF 正文明确说：每一步有了更好的 approximation 后，可以更新 graph Laplacian 的 weights。

对当前 refinement 的学习：

- 如果 `L_phys` 固定，当前二次问题一次闭式解就是该目标下的最优解。
- 重复使用同一个 `L_phys` 和同一个目标，不会带来新的文献机制。
- 迭代只有在图、权重、可靠性、denoiser 或局部估计随 `Z` 更新时才有意义。

不能推出：

- 不能因为有 iterated graphLa 文献，就给当前固定图方法加多轮 diffusion 作为贡献。
- 不能忽略迭代方法本身更难证明、更难解释的边界。

## 7. RED / unrolling / interpretable deep GLR：学习化应对应明确的求解或部署目标

对应论文：`26_Unrolling_Plug_and_Play_GGLR`、`27_Interpretable_Deep_Denoise_GL`、`28_Graph_Signal_Denoising_RED`

这些论文说明图正则求解可以展开成可学习网络，也可以把 denoiser 显式写进 regularization by denoising 框架。关键点不是“深度化一定更好”，而是：

- unrolling 把迭代求解步骤变成可训练模块；
- GGLR/PnP 把图先验和 denoiser 结构结合；
- RED 需要明确 denoiser 和参数估计；
- interpretable deep GLR 试图保留图正则的数学解释，而不是完全黑箱化。

对当前 refinement 的学习：

- 当前闭式解是最透明的 fixed graph filter。
- 可学习化对应的是另一个任务层级：out-of-sample、deployment、teacher-student、learnable correction layer 或更复杂 denoising prior。
- 如果没有明确目标，学习化只会让原本清楚的闭式问题变复杂。

不能推出：

- 不能把“学习 `lambda`”等同于 deep graph denoising。
- 当前没有实现 unrolling/RED/PnP，不能写成方法贡献。
- 不能为了创新感而牺牲可解释性。

## 8. Smoothness verification：先验应被检验，而不是靠结果倒推

对应论文：`34_Verifying_Smoothness_Graph_Signals`

这篇论文提醒：许多 GSP 工作使用 smoothness assumption，但并不验证它。它把“观测图信号是否可被视作 smooth graph filter 输出”变成统计检测问题。

对当前 refinement 的学习：

```text
用 GLR 精修之前，理论上应先问：
Y0 的两个 chart 坐标分量，在 ADP/time 物理图上是否表现为足够平滑的图信号？
```

当前物理图 refinement 结果很好，说明这个假设在当前数据上有经验支持；但这不是严格证明。尤其是欧氏图能改善 mean/P95 但恶化 max，说明“平均平滑有效”和“图信号假设全局成立”不是同一件事。

不能推出：

- 不能把 performance improvement 当作 smoothness proof。
- 不能把 smoothness verification 论文直接写成 refinement 方法；它是诊断框架。

## 9. 当前算子层的五个假设

把上述论文合起来，当前闭式扩散算子至少有五个假设：

| 假设 | 当前公式如何默认它 | 哪些论文提醒边界 | 学习结论 |
|---|---|---|---|
| 信号平滑 | `Z[:,0]`、`Z[:,1]` 在 `L_phys` 上低频 | verifying smoothness、GGLR | smoothness 是前提，不是结果 |
| 值平滑足够 | 惩罚邻居坐标差即可 | GGLR、GTV | chart 坐标可能更接近连续几何场，不一定适合 piecewise constant |
| 单尺度低通足够 | 一个 `L` 和一个 `lambda` 处理所有结构 | framelet、多尺度图去噪 | 真实结构和噪声可能跨尺度混合 |
| 图结构固定可信 | 给定 `L` 后只修 `Z` | robust graph filter、framelet structure denoising | 错误边会变成错误扩散路径 |
| 一次闭式解足够 | 固定目标下直接求解 | graphLa+、iterated graphLa、unrolling | 固定图下一次解合理；迭代/学习化需有新信息更新 |

这张表给出的学习结论是：

```text
普通 GLR 是当前 refinement 的稳健起点，但不是完整的 chart repair 理论。
它提供的是“给定可信图，修平滑坐标信号”的先验；
而文献提醒我们，chart repair 还可能涉及连续几何、真实高频、图结构噪声、多尺度结构和部署泛化。
```

## 10. 当前学习状态

我现在对图正则论文的吸收可以压缩为：

```text
graphLa+ 支撑当前公式的数学合法性；
GGLR/framelet/robust graph/smoothness verification 暴露普通 GLR 的假设边界；
RED/unrolling 说明后处理可以学习化，但只有在目标升级时才合理。
```

这仍然不是正式优化 idea。它只是为后续构思建立边界：如果未来要讨论改正则项、改求解器、迭代或学习化，必须先说明当前五个算子假设中哪一个被文献和当前现象共同挑战。

