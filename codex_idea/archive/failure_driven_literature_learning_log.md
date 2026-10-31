# Failure-Driven Literature Learning Log

更新日期: 2026-07-09

## 0. 为什么需要重新学习

Smoothness-audited alpha-only 实验没有超过当前 graph-harmonic baseline：

```text
baseline mean / p95 / max = 0.3187 / 0.6346 / 1.1414
alpha-only mean / p95 / max = 0.3403 / 0.7166 / 1.2639
```

诊断显示：

```text
residual_hat vs initial error Spearman ~= 0.211
residual_hat vs baseline gain Spearman ~= 0.186
high-residual displacement > low-residual displacement
```

结论：

```text
L_phys Y0 residual 不是纯噪声，但它不是足够可靠的“强修开关”。
当前失败的是 residual -> alpha 的直接单调映射。
```

因此后续不能继续拍脑袋加模块，需要围绕失败原因学习文献。

## 1. 本轮联网检索问题

本轮检索不是泛泛找 channel charting，而是围绕三个问题：

1. Graph smoothing 为什么会过平滑，如何 edge-preserving？
2. 当 graph support 本身有噪声时，GSP 文献怎么联合处理 signal error 和 graph error？
3. Channel charting 文献中如何处理 dissimilarity / geodesic / NLoS / metric uncertainty，而不是把物理图当确定真值？

## 2. 第一批需要重点学习的论文

### 2.1 Robust Graph Filter Identification and Graph Denoising from Signal Observations

来源：
https://arxiv.org/abs/2210.08488

初步吸收：

- 该文明确指出，多数 GSP 方法默认 supporting graph 完全已知，但实际 graph 可能有观测误差和 perturbation。
- 它不是只在固定 graph 上滤波，而是联合估计 graph filter 和 graph denoising。
- 对当前工作的启发：不能只把 ADP/time graph 当成确定物理真值再调 alpha；更合理的新模块可能要显式建模 graph edge reliability 或 graph correction。

对当前失败的解释：

```text
我们的 alpha-only 默认 W_phys 完全正确。
但如果 W_phys 局部边有 shortcut / unreliable edge，
降低 alpha 只会让错误 graph 更强地支配 Z。
```

### 2.2 Graph Spatio-Spectral Total Variation Model for Hyperspectral Image Denoising

来源：
https://arxiv.org/abs/2207.11050

初步吸收：

- 该文批评普通 SSTV uniform local difference 难以同时去噪和保留复杂细节。
- 它使用 graph 反映 spatial structure，并把 weighted spatial difference operator 放进 TV-type regularization。
- 对当前工作的启发：新模块不应该继续用普通 L2 Laplacian smoothing。可以考虑 graph-TV / robust edge-preserving penalty：

```math
\sum_{ij} W_{ij} \rho(\|Z_i-Z_j\|)
```

其中 `rho` 不是二次惩罚，而是 Huber / Charbonnier / TV-like penalty。

对当前失败的解释：

```text
当前 baseline 是 L2 graph Laplacian。
alpha-only 本质上只是加强 L2 smoothing。
如果问题是局部结构或真实高频被压掉，继续加强 L2 只会更差。
```

### 2.3 Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss

来源：
https://arxiv.org/abs/2412.01715

初步吸收：

- 该文针对 dissimilarity metric-based CC，指出 Euclidean/geodesic discrepancy 和 dissimilarity uncertainty。
- 它把 dissimilarity 作为 probability distribution，而不是确定距离。
- 对当前工作的启发：ADP/time geodesic graph 应该带 uncertainty，不能被当作确定约束。

对当前失败的解释：

```text
L_phys Y0 residual 高，可能来自 Y0 错，也可能来自 graph/dissimilarity uncertainty。
alpha-only 把所有 residual 高的情况都解释成 Y0 不可信，这是错误归因。
```

### 2.4 TDoA-Based Self-Supervised Channel Charting with NLoS Mitigation

来源：
https://arxiv.org/abs/2510.08001

初步吸收：

- 该文面向真实 5G testbed，强调 NLoS distortion，并引入识别和 mask NLoS-induced noisy measurements 的机制。
- 对当前工作的启发：比“平滑审计”更像模块的是 measurement/edge corruption detection。

可能转化：

```text
ADP/time edge corruption detector
-> mask / attenuate suspicious physical edges
-> robust graph-harmonic or graph-TV refinement
```

### 2.5 UNILocPro: Unified Localization Integrating Model-Based Geometry and Channel Charting

来源：
https://arxiv.org/abs/2510.27394

初步吸收：

- 该文不是简单后处理，而是把 model-based geometry 和 CC 结合，并针对 LoS/NLoS 做 adaptive activation。
- 对当前工作的启发：真正的新模块可能不是“修完 chart 再平滑”，而是多证据 adaptive activation：

```text
which constraint is reliable here?
ADP graph / time graph / learned chart neighborhood / model-based cue
```

当前项目没有完整 model-based branch，但可以借鉴 adaptive activation 的思想。

## 3. 当前阶段形成的新判断

### 3.1 不能继续的方向

不应继续：

```text
residual high -> alpha low -> stronger smoothing
```

原因：

- residual 与真实错误只有弱相关；
- alpha 过强等价于把 lambda 局部放大；
- baseline 已经强，继续加 L2 smoothing 容易过平滑；
- 如果 graph 局部错误，强 smoothing 会放大错误边影响。

### 3.2 还值得继续学的方向

更值得继续学习：

1. Robust graph support / edge reliability
   - 目标不是给点加 alpha，而是先判断物理边是否可靠。

2. Edge-preserving graph regularization
   - 从 L2 Laplacian 改成 Huber / TV / Charbonnier / anisotropic penalty。

3. Uncertainty-aware physical graph
   - 把 ADP/time distance 当不确定观测，而不是确定真值。

4. NLoS / noisy measurement masking
   - 把无线物理测量中的异常边作为核心问题，而不是 chart smoothness residual。

## 4. 下一步学习清单

继续检索和精读这些关键词：

```text
robust graph signal processing topology uncertainty
graph total variation denoising edge preserving
anisotropic diffusion on graphs
Huber graph Laplacian regularization
uncertainty-aware channel charting
NLoS mitigation channel charting
robust geodesic graph construction
```

目标不是立刻提出模块，而是回答：

```text
当前项目中，新增模块到底应该处理 signal error，graph error，还是 penalty form error？
```

目前初步判断：

```text
更可能有效的新模块不是 node alpha。
更可能是 edge corruption / uncertainty-aware robust regularizer。
```

## 5. 第二轮联网学习整理

### 5.1 当前失败应重新归因成三个错误源

alpha-only 失败后，不能只问“alpha 怎么调”，而应区分三类错误：

| 错误源 | 在当前方法中的表现 | 文献对应问题 | 对模块设计的影响 |
|---|---|---|---|
| signal error | `Y0_i` 本身局部位置错 | graph signal denoising/restoration | 可以修，但不能直接全局加大 smoothing |
| graph error | `W_phys` 有 shortcut / unreliable edge / wrong neighborhood | robust GSP with topology perturbation | 需要先处理边可靠性，而不是先改点 fidelity |
| penalty error | L2 Laplacian 过分鼓励所有边两端接近 | TV / Huber / anisotropic diffusion | 需要换正则项形态，而不是加大 Laplacian |

这三个错误源在 alpha-only 里被混在一起了：

```text
r_i 高 -> 假设 signal error 高 -> 降低 alpha -> 更依赖 W_phys + L2
```

但实际可能是：

```text
r_i 高来自 graph error 或真实局部结构/曲率，
此时更依赖 W_phys + L2 会把错误放大。
```

### 5.2 Robust GSP: graph support 不是确定真值

#### Robust Graph Filter Identification and Graph Denoising from Signal Observations

来源：
https://arxiv.org/abs/2210.08488

关键事实：

- 论文明确指出 GSP 常用假设是 supporting graph 已知，但实际图可能有观测误差和扰动。
- 它把问题写成 graph filter identification + graph denoising 的联合优化，而不是在固定图上滤波。
- 它讨论的 perturbation 可以发生在 links 上。

对当前项目的更深启发：

```text
导师 baseline 的 W_phys 不能被当作完全正确。
ADP/time graph 是由 dissimilarity + geodesic + kNN 构造出来的估计图，
它可能存在 shortcut、局部密度偏差、时间边误连、非视距/多径造成的假近邻。
```

因此，真正的新模块可能应是：

```text
Physical Edge Reliability / Graph Denoising Module
```

而不是：

```text
Node Fidelity Alpha Module
```

更合理的数学方向：

```math
\min_{Z,\widetilde W}
\|Z-Y_0\|^2
+ \lambda \sum_{ij}\widetilde W_{ij}\|Z_i-Z_j\|^2
+ \mu R(\widetilde W, W_{\mathrm{phys}})
+ \nu \Phi(\widetilde W)
```

其中 `R` 约束新图不要偏离物理图太多，`\Phi` 控制图的稀疏性/连通性/局部一致性。

第一轮可实现的轻量版本不是联合优化，而是：

```text
先计算每条物理边的多证据一致性分数，
再做 edge attenuation，
最后用原 graph-harmonic solver。
```

### 5.3 Graph-SSTV / TV: L2 Laplacian 不是唯一正则

#### Graph Spatio-Spectral Total Variation Model for Hyperspectral Image Denoising

来源：
https://arxiv.org/abs/2207.11050

关键事实：

- 论文指出普通 SSTV uniform local difference 难以在去噪同时保留复杂细节。
- 它提出 Graph-SSTV，用 graph 反映 spatial structure，并用基于图的 weighted spatial difference operator。
- 它把问题放在 convex optimization 里求解。

对当前项目的更深启发：

导师 baseline 是：

```math
\min_Z \|Z-Y_0\|^2 + \lambda\sum_{ij}W_{ij}\|Z_i-Z_j\|^2
```

这是 L2 smoothing。L2 的问题是：边两端差异越大，惩罚增长越快，所以它强烈拉近大差异边。

如果大差异来自：

- 真实几何弯曲；
- local density 变化；
- graph shortcut；
- chart 局部 high-frequency 但有意义结构；

那么 L2 会过平滑。

更像模块的改法：

```text
Edge-Preserving Robust Graph Refinement Module
```

候选正则：

```math
\min_Z
\|Z-Y_0\|^2
+ \lambda \sum_{ij} W_{ij}\rho(\|Z_i-Z_j\|)
```

其中 `rho` 可以是：

```text
Huber: 小差异 L2，大差异 L1
Charbonnier: 平滑近似 L1
TV: 直接 L1 on graph gradients
```

意义：

```text
不是“高 residual 点多修”，
而是“正常边平滑，疑似边/大梯度边不强拉”。
```

这比 alpha-only 更像一个真正新模块，因为它改变的是 regularizer 的性质。

### 5.4 Uncertainty-aware CC: dissimilarity 不是确定距离

#### Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss

来源：
https://arxiv.org/abs/2412.01715

关键事实：

- 论文针对 metric/dissimilarity-based channel charting。
- 它指出 nonconvex low-dimensional structure 下 Euclidean distance 与 geodesic distance 有 discrepancy。
- 它把 dissimilarity 建模为 probability distribution，而不是 deterministic value。
- 它还强调多个 dissimilarity metrics 可以一致融合。

对当前项目的更深启发：

当前 ADP/time graph 构造流程是：

```text
ADP/time fused distance -> geodesic -> kNN graph -> W
```

但 alpha-only 隐含假设：

```text
W_phys 给出的近邻关系是确定可信的。
```

文献提示这个假设不稳。更合理的是每条边应有 uncertainty：

```math
W_{ij} = \mathbb{E}[w(d_{ij})],
\quad
\sigma_{ij} = uncertainty(d_{ij})
```

或者在 refinement 里：

```math
\sum_{ij} c_{ij} W_{ij}\|Z_i-Z_j\|^2,
\quad c_{ij}=f(\text{distance uncertainty})
```

这和普通 beta 不同：

```text
beta 不应来自 residual_hat 差异，
而应来自 ADP/time/geodesic/learned-chart 多证据不确定性。
```

### 5.5 TDoA/NLoS mitigation: 应显式识别 noisy measurement

#### TDoA-Based Self-Supervised Channel Charting with NLoS Mitigation

来源：
https://arxiv.org/abs/2510.08001

关键事实：

- 论文面向真实 5G testbed，认为 NLoS distortion 会影响 CC。
- 它使用 CIR、TDoA、TRP locations、short-interval UE displacement。
- 核心机制包含识别并 mask NLoS-induced noisy measurements。

对当前项目的更深启发：

无线定位/CC 里的错误不是普通噪声，而是结构化错误：

```text
NLoS / multipath / wrong delay / wrong time-neighbor / graph shortcut
```

因此新模块更应表达为：

```text
Physical Measurement Reliability Module
```

而不是：

```text
Smoothness Residual Module
```

如果当前数据有 ADP + timestamp，轻量可行的信号包括：

```text
ADP近但时间远：可疑
time近但ADP远：可疑或快速运动/非平稳
geodesic近但 direct fused distance 异常：可疑 shortcut
物理图近邻与 Y0/FCF chart 邻域完全不一致：可疑边
```

这些边级信号比 node residual 更接近 noisy measurement detection。

### 5.6 UNILocPro: adaptive activation 比固定融合更有论文感

#### UNILocPro: Unified Localization Integrating Model-Based Geometry and Channel Charting

来源：
https://arxiv.org/abs/2510.27394

关键事实：

- 它在 mixed LoS/NLoS 场景中结合 model-based localization 和 CC。
- 它基于 LoS/NLoS identification 做 adaptive activation。
- 它联合使用 pairwise distance loss、triplet loss、LoS-based loss、OT-based loss。

对当前项目的启发：

虽然我们没有完整 model-based branch，但 adaptive activation 的思路很重要：

```text
不是所有区域都用同一种几何证据。
不同边/点应根据可靠性选择不同约束。
```

对应到当前 baseline：

```text
当前：所有物理边进入同一个 Laplacian。
更强模块：每条边选择/激活 ADP evidence、time evidence、chart evidence 或 suppression。
```

潜在模块名：

```text
Multi-Evidence Edge Activation Module
```

输入：

```text
ADP distance
timestamp distance
geodesic distance
Y0 Euclidean distance
neighbor-rank consistency
```

输出：

```text
edge activation type / edge confidence
```

再进入 robust graph-harmonic solver。

### 5.7 graphLa+ / iterated graph Laplacian: 当前 baseline 的数学位置

#### A data-dependent regularization method based on the graph Laplacian

来源：
https://arxiv.org/abs/2312.16936

关键事实：

- graphLa+Psi 基于 preliminary approximation 构造 graph Laplacian。
- 正则项依赖 observed data/noise。
- 论文强调 graph Laplacian 可以作为已有方法 Psi 的后处理/增强支架。

#### Improved impedance inversion by the iterated graph Laplacian

来源：
https://arxiv.org/abs/2404.16324

关键事实：

- 从初始估计开始构造 graph Laplacian。
- 用 variational framework refinement。
- 迭代更新估计并重新校准 graph Laplacian。

对当前项目的启发：

导师 baseline 本身已经落在一个成熟范式里：

```text
initial estimate + graph Laplacian regularization
```

所以单纯说“加 Laplacian refinement”不够新。要形成模块，必须回答：

```text
无线物理图有什么特殊不确定性？
无线 graph edge 怎么被诊断/选择/鲁棒化？
为什么这个模块不是普通 graphLa+？
```

### 5.8 Graph signal restoration / unrolling: 固定参数是弱点

#### Graph Signal Restoration Using Nested Deep Algorithm Unrolling

来源：
https://arxiv.org/abs/2106.15910

关键事实：

- 论文指出传统 convex optimization restoration 的性能依赖手动固定的 hyper-parameters。
- 它用 unrolling 把 ADMM / PnP-ADMM 的参数变成可学习参数。
- 它保留解释性，同时参数量小。
- 它还指出 graph restoration 应对 graph perturbation 具有鲁棒性。

对当前项目的启发：

当前 solver 的弱点是：

```text
固定 lambda
固定 graph
固定 quadratic penalty
固定迭代公式
```

但直接做 deep unrolling 可能超出当前项目阶段。更保守的吸收方式：

```text
不要让 alpha/beta 直接依赖一个 residual；
而是先设计少量可解释的 edge features，
再学习或拟合一个低参数 edge activation 函数。
```

不过这会引入训练/验证协议问题，需要 held-out 防止泄露。

## 6. 第二轮学习后的候选模块排序

### 候选 A：Physical Edge Reliability Graph Denoising Module

核心问题：

```text
W_phys 不是确定真值，先做边可靠性诊断/去噪。
```

模块位置：

```text
ADP/time distance + geodesic neighbors + Y0
  -> edge reliability module
  -> W_reliable
  -> graph-harmonic / robust solver
```

为什么比 alpha-only 更好：

- alpha-only 改点 fidelity，可能放大错误边；
- edge reliability 先减少错误边传播；
- 更符合 robust GSP / NLoS masking 文献。

风险：

- 如果当前 W_phys 已经很干净，边去噪收益小；
- 如果 edge features 还是只依赖 Y0，可能又退化为 chart bias。

初步优先级：高。

### 候选 B：Edge-Preserving Robust Penalty Module

核心问题：

```text
L2 Laplacian 会过度惩罚大边差，导致过平滑。
```

模块位置：

```text
Y0 + W_phys
  -> robust graph-TV / Huber graph regularizer
  -> Z
```

为什么比 alpha-only 更好：

- 不再提高 diffusion 强度；
- 只改变大梯度边的惩罚形态；
- 能处理“真实高频/局部结构”与“错误高频”的混淆。

风险：

- 求解比线性 Jacobi 复杂；
- 若 baseline tail 修复依赖强 L2 拉回，大梯度截断可能削弱 tail 修复。

初步优先级：高，但实现复杂度中等。

### 候选 C：Multi-Evidence Edge Activation Module

核心问题：

```text
不同区域应激活不同证据：ADP、time、geodesic、chart neighborhood。
```

模块位置：

```text
ADP distance / timestamp distance / geodesic distance / Y0 distance
  -> edge feature vector
  -> activation/confidence
  -> W_active
  -> solver
```

为什么有潜力：

- 接近 UNILocPro 的 adaptive activation 思想；
- 更贴近无线物理语义；
- 可能形成清晰模块贡献。

风险：

- 需要谨慎避免用真实坐标训练 edge classifier；
- 无监督 activation 设计难度较高。

初步优先级：中高。

### 候选 D：Iterative Graph Recalibration Module

核心问题：

```text
一次性 fixed W refinement 可能不足。
```

模块位置：

```text
Y0 -> W_phys -> Z1 -> recalibrate graph -> Z2
```

为什么有启发：

- 来自 iterated graph Laplacian；
- 允许 refinement 后重新估计局部结构。

风险：

- 很容易变成“根据 chart 自己修自己”，导致 Euclidean graph ablation 的问题；
- 可能没有无线物理新意。

初步优先级：中低，暂不作为主线。

## 7. 目前最重要的学习结论

这轮学习之后，我对“是否继续”的判断更新为：

```text
继续是有必要的，但不该继续 smoothness residual alpha。
```

更合理的主线应从：

```text
node-wise adaptive smoothing
```

转为：

```text
edge-wise physical graph reliability / robust edge-preserving refinement
```

导师 baseline 的价值是提供强初始框架：

```text
Y0 + physical graph + graph-harmonic solver
```

真正需要新增的模块应该解决 baseline 未解决的问题：

```text
物理图边并非全可信；
L2 Laplacian 并非总合适；
无线 dissimilarity/geodesic 有不确定性；
NLoS/shortcut/noisy measurement 应显式处理。
```

下一步继续学习时，应优先围绕候选 A 和 B 深读，暂时不要设计新实验。

## 8. 第三轮学习：从“失败模块”反推真正需要的模块条件

本轮继续学习的目标不是马上提出新模块，而是把 alpha-only 失败后必须修正的认识写清楚：

```text
如果新模块只是把 baseline 的 W 或 lambda 做单调缩放，它大概率仍然是在修补现有方法；
如果新模块能改变“错误边如何进入扩散”“大梯度边如何被惩罚”“物理证据如何表达不确定性”，才可能真正优化结果。
```

### 8.1 失败模块暴露的不是“alpha 设计差”，而是错误对象选错

alpha-only 的逻辑是：

```text
Y0 在物理图上不平滑
-> 认为 Y0 该点不可信
-> 降低 fidelity
-> 让 graph-harmonic diffusion 更强
```

但实验诊断显示 residual 只和真实误差弱相关，并且 alpha-only 在所有 residual 分位上都变差。由此得到更严格的学习结论：

```text
L_phys Y0 residual 只能说明“Y0 与 W_phys 有分歧”，
不能说明“Y0 错、W_phys 对、应该更强扩散”。
```

这个分歧至少有三种来源：

| 分歧来源 | 如果误判会怎样 | 对新模块的要求 |
|---|---|---|
| Y0 坐标噪声 | 应该修 Y0 | 可以用物理图平滑 |
| W_phys 错边/不可靠边 | 强扩散会传播错误 | 必须先识别或抑制可疑边 |
| 真实几何高频/局部弯曲 | 强 L2 会过平滑 | 需要 edge-preserving 或高阶几何先验 |

因此，后续不能再做：

```text
node residual -> node alpha -> stronger L2 smoothing
```

至少要把问题转到 edge-level 或 penalty-level。

### 8.2 文献对“图错误”的支持：supporting graph 是证据，不是真值

`Robust Graph Filter Identification and Graph Denoising from Signal Observations`
本轮通过 arXiv 元数据核到题名、作者和摘要；是否已有最终 IEEE 期刊版本，后续需要再用 IEEE Xplore / Crossref 单独核实。因此这里先只把它作为机制学习来源，不把 venue 状态写成已完全确认的强引用。

这篇论文最关键的学习点是：

```text
GSP 常假设 supporting graph 已知；
但实际图可能存在 observation errors 和 perturbations；
忽略这些图缺陷时，依赖图拓扑的方法会得到次优结果。
```

映射到当前项目：

```text
W_phys 不是 ground-truth graph。
它是 ADP/time/geodesic/kNN 共同构造出的 physical evidence graph。
如果其中有 shortcut、错近邻、时间误连、局部密度偏差，
graph-harmonic solver 会把这些错边变成坐标约束。
```

这说明新模块更应该优先问：

```text
哪些 physical edges 不应该以原强度进入 Laplacian？
```

而不是先问：

```text
哪些 nodes 应该被更强平滑？
```

这也是 alpha-only 失败后最重要的方向修正。

### 8.3 文献对“L2 过平滑”的支持：不是所有大差异都该二次惩罚

`Graph Spatio-Spectral Total Variation Model for Hyperspectral Image Denoising`
本轮通过 arXiv 元数据核到题名、作者和摘要；是否已有最终 IEEE GRSL 期刊版本，后续需要再用 IEEE Xplore / Crossref 单独核实。该文指出传统 SSTV 对局部差分做 uniform 处理，在强噪声下难以同时去噪和保留细边缘/纹理；它用图结构构造 weighted spatial difference operator，并放入 TV-type regularization。

对当前项目的启发不是“照搬 hyperspectral denoising”，而是：

```text
普通 L2 Laplacian 对边两端差异做二次惩罚；
差异越大，拉近越强。
```

如果大差异来自错误边、局部真实几何变化或 chart 的边界结构，L2 会错误地强拉。alpha-only 实际上又放大了这个问题，因为它通过降低 alpha 增强了局部有效 lambda。

因此，真正可能改善结果的 penalty-level 模块应满足：

```math
\min_Z \|Z-Y_0\|_F^2
+ \lambda \sum_{ij} W_{ij}\rho(\|Z_i-Z_j\|)
```

其中 `rho` 不应总是二次型。候选包括：

```text
Huber / Charbonnier / graph-TV-like penalty
```

这类模块的实际意义是：

```text
小差异边继续平滑；
大差异边不再无限增强拉力；
从而减少错误边或真实几何高频被强行压平的风险。
```

### 8.4 文献对“值平滑不够”的支持：chart 坐标可能更接近局部几何场

`Fast & Robust Image Interpolation using Gradient Graph Laplacian Regularizer`
虽然是 2021 年论文，不在 2022-2026 主检索范围内，但它是理解 GGLR 的基础。该文明确指出 GLR 会促进 piecewise constant reconstruction，并可能产生 staircase effect；GGLR 转而促进 piecewise planar reconstruction。

这对 chart refinement 的启发很直接：

```text
二维 chart 坐标不是类别标签。
真实空间坐标沿轨迹应连续变化，局部更像平滑几何场或近似仿射场。
普通 GLR 只惩罚坐标值差，可能过度鼓励邻居坐标相同。
```

但这条线目前不能马上作为主模块，因为它需要定义可靠的局部梯度或局部仿射结构。当前项目还没有证明：

```text
baseline 的主要误差来自 staircase / 局部仿射破坏。
```

所以它是中期方向，不是下一轮最优先模块。

### 8.5 文献对“先验应被检验”的支持：smoothness 不是事后解释

`Verifying the Smoothness of Graph Signals: A Graph Signal Processing Approach`
把 graph signal smoothness 作为可验证假设，而不是默认事实。该文提出 smoothness detectors，用于检查给定数据和图是否满足 smooth graph signal 模型。

对当前项目的学习结论：

```text
Y0[:,0] 和 Y0[:,1] 在 W_phys 上是否 smooth，
不能只靠最终 localization error improvement 倒推。
```

alpha-only 失败说明：

```text
“不平滑”本身不是修复指令。
它只是一种异常信号，需要进一步判断异常来自 Y0、W_phys 还是正则假设。
```

因此 smoothness auditing 如果继续存在，也不能作为直接 actuator。更合理的位置是：

```text
diagnostic gate / evidence feature
```

而不是：

```text
direct alpha controller
```

### 8.6 CC 文献对“物理证据不确定性”的支持：ADP/time graph 不能等置信使用

`Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss`
明确把 dissimilarity 建模为 probability distribution，而不是确定距离。它还强调可以一致融合多个 dissimilarity metrics。

对当前 refinement 的含义：

```text
W_ij 当前同时表示 similarity 和 confidence。
但一条 ADP/time 近邻边可能“相似度高但可信度低”。
```

这解释了为什么简单 edge beta 也不能随便拍：

```text
beta_ij 必须来自 physical evidence uncertainty，
而不能只是来自 Y0 与 W_phys 的残差。
```

更合理的证据应来自：

```text
ADP rank consistency
timestamp rank consistency
direct fused distance vs geodesic distance discrepancy
mutual kNN consistency
local density / sigma stability
Y0-neighborhood agreement 只能作为辅助证据，不能作为唯一证据
```

### 8.7 CC 文献对“NLoS/noisy measurement masking”的支持：边可靠性要有无线语义

`TDoA-Based Self-Supervised Channel Charting with NLoS Mitigation`
本轮联网核到 arXiv 2025 版本；暂未核到稳定的期刊页面，所以不能把它当成已发表强引用。该文使用 CIR、TDoA、TRP location 和短时间 UE displacement，并显式识别和 mask NLoS-induced noisy measurements。

这篇论文给当前项目的启发是：

```text
无线物理测量错误不是普通噪声，而是有传播机制的结构性错误。
```

当前项目没有 NLoS 标签、TRP 几何或完整 model-based branch，所以不能声称处理 NLoS。但它提示一个更正确的模块方向：

```text
physical edge corruption detection / attenuation
```

而不是：

```text
generic smoothness residual smoothing
```

`UNILocPro` 进一步说明，前沿方法会根据 LoS/NLoS identification 做 model-based 与 CC-based 证据的 adaptive activation。它目前仍是 submitted/preprint 状态，不宜作为已发表强依据；但其机制启发是明确的：

```text
不同证据源不应该全局固定融合；
应该按边/区域选择激活或抑制。
```

### 8.8 当前最可信的模块设计前约束

经过这轮学习，我现在认为任何下一版优化模块必须满足下面四个约束，否则不值得实验：

| 约束 | 原因 | 对应文献启发 |
|---|---|---|
| 必须是 edge-level 或 penalty-level，不应再是单纯 node alpha | alpha-only 已证明 node residual 不可靠 | robust graph filter, uncertainty-aware CC |
| 必须能抑制可疑物理边的错误传播 | W_phys 可能有 link perturbation / shortcut | Robust GF + graph denoising, NLoS masking |
| 必须避免简单增强 L2 smoothing | L2 会强拉大差异边，可能过平滑 | Graph-SSTV, GGLR |
| 必须不使用 UE position 参与模块构造 | 否则数据泄露 | 当前 CC self-supervised protocol |

这意味着“结果更好”的合理路径不是调参，而是改变优化机制：

```text
baseline:
    fixed W_phys + quadratic Laplacian

更合理的新模块方向:
    reliability-filtered W_phys
    或 robust edge-preserving regularizer
    或二者的小规模组合
```

### 8.9 目前暂不应推进的方向

暂时不应推进：

```text
1. residual -> alpha 的任何变体
2. 固定 L_phys 的多轮 diffusion
3. 只学习 lambda 或只做局部 lambda map
4. 直接 deep unrolling / RED
5. 直接 GGLR / framelet
```

原因：

- 1 已被当前实验否定；
- 2 在固定图二次目标下没有新信息；
- 3 仍然没有解决错误边和 L2 过平滑；
- 4 会把问题复杂化，且当前没有 out-of-sample 目标；
- 5 需要先证明 chart 坐标存在对应的梯度/多尺度失效。

### 8.10 当前学习结论

本轮学习后，我对“新模块应该是什么类型”的判断进一步收窄：

```text
最优先不是 smoothness auditing。
最优先是 physical-edge reliability filtering。
第二优先是 robust edge-preserving graph regularization。
```

两者的区别是：

```text
physical-edge reliability filtering:
    先改进入 Laplacian 的边

robust edge-preserving regularization:
    不一定改边，但改变边差异被惩罚的方式
```

如果后续要真正设计模块，应该从这两条中选一个最小可验证版本，而不是再围绕 node confidence 做包装。

## 9. 本轮联网核查到的关键来源

| 论文 | 状态 | 链接 | 本轮使用目的 |
|---|---|---|---|
| Robust Graph Filter Identification and Graph Denoising from Signal Observations | arXiv 2022，期刊状态待 IEEE/Crossref 复核 | https://arxiv.org/abs/2210.08488 | 支撑 W_phys 可能有 link perturbation，图本身需被当作证据 |
| Graph Spatio-Spectral Total Variation Model for Hyperspectral Image Denoising | arXiv 2022，期刊状态待 IEEE/Crossref 复核 | https://arxiv.org/abs/2207.11050 | 支撑 edge-preserving / TV-type regularization，不应盲目增强 L2 |
| Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss | arXiv 2024 | https://arxiv.org/abs/2412.01715 | 支撑 dissimilarity uncertainty，W_ij 不应只表示 similarity |
| TDoA-Based Self-Supervised Channel Charting with NLoS Mitigation | arXiv 2025，期刊状态待复核 | https://arxiv.org/abs/2510.08001 | 支撑 noisy physical measurement mask / NLoS-induced corruption |
| UNILocPro: Unified Localization Integrating Model-Based Geometry and Channel Charting | submitted/preprint 2025 | https://arxiv.org/abs/2510.27394 | 支撑 adaptive activation 思想，但不能作为已发表强依据 |
| Verifying the Smoothness of Graph Signals | arXiv 2023 | https://arxiv.org/abs/2305.19618 | 支撑 smoothness 应作为诊断假设，而不是直接控制修复 |
| Fast & Robust Image Interpolation using GGLR | arXiv 2021 | https://arxiv.org/abs/2101.09951 | 支撑普通 GLR 的 staircase / piecewise constant 边界 |
