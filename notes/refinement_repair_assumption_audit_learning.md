# Refinement / Repair 假设审计学习笔记

更新时间：2026-06-25

本笔记继续学习参考论文，不正式提出优化 idea，不安排实验。它的目的不是回答“应该怎么改方法”，而是建立一个 **idea 前的假设审计框架**：后续任何精修优化都必须先说明它针对当前公式中的哪个假设、由哪些论文机制支持、不能越界声称什么。

当前 baseline：

```math
Z^\star=\arg\min_Z \|Z-Y_0\|_F^2+\lambda\operatorname{tr}(Z^\top L_{\mathrm{phys}}Z)
```

这个公式不是一个单纯的调参对象，而是七个假设的组合：

1. `Y0` 是可被二次 fidelity 修正的 noisy chart；
2. `L_phys` 是可信物理图；
3. `W_ij` 同时合理表达相似度和约束强度；
4. `tr(Z^T LZ)` 是适合 chart 坐标的先验；
5. 单一 `lambda` 能描述全局证据权衡；
6. 固定图一次闭式解足够；
7. post-hoc transductive refinement 满足任务需求。

## 1. 假设 A：`Y0` 是普通带噪 chart

### 当前公式如何使用这个假设

`||Z-Y0||_F^2` 把初始 FCF chart 看作一个可靠观测，只需要在其附近做修正。这个项默认 `Y0` 的错误更像连续的小噪声，而不是结构性错误。

### 相关论文证据

- ADP/time 和 uncertainty-aware geodesic loss 说明训练 chart 依赖 dissimilarity，而 dissimilarity 本身可能有不确定性。
- TDoA/NLoS 和 UNILocPro 说明 propagation condition 会带来系统性偏差。
- LoS Channel Charting 说明 distance ambiguity 可以造成结构性错误邻域。
- CC hard-user / Non-RIS 相关工作提示 tail error 可能来自环境不可辨性，而不是普通噪声。

### 如果该假设被违反，意味着什么

如果 `Y0` 的错误是 fold、shortcut、outlier island 或局部拓扑错接，那么二次 fidelity 会强迫 refined chart 仍然贴近错误结构。此时简单增大 `lambda` 可能只是更强地用图平滑压制错误，但不一定真正修复错误来源。

### 不能直接跳到什么

不能因为存在 tail error 就直接宣布需要 topology repair。必须先区分：

- 是 `Y0` 自身局部错误；
- 是 `L_phys` 错误边导致；
- 是 alignment / metric evaluation 造成；
- 还是少数样本本身物理不可辨。

## 2. 假设 B：`L_phys` 是可信物理图

### 当前公式如何使用这个假设

`L_phys` 决定哪些样本对被强制靠近。只要图中有错误边，正则项就会把错误边转成坐标约束。

### 相关论文证据

- ADP/time 论文支持物理图作为 radio-geometry graph。
- Uncertainty-aware geodesic loss 支持边距离/物理距离存在不确定性。
- TDoA/NLoS 支持不同传播条件下测量可靠性不同。
- LoS Channel Charting 支持 channel distance 可能有 periodic / oscillatory ambiguity。
- Robust graph filter identification 支持 supporting graph 本身可能有 link perturbation。

### 如果该假设被违反，意味着什么

精修可能不是在修 `Y0`，而是在把错误 `L_phys` 的结构传播到 `Z`。这尤其会影响强正则、多步扩散、迭代滤波或高阶图滤波。

### 不能直接跳到什么

不能简单说“物理图一定比欧氏图正确”。更准确的学习结论是：

```text
物理图提供独立于 Y0 的证据，但它本身仍是 noisy physical evidence。
```

## 3. 假设 C：`W_ij` 只需要表示相似度

### 当前公式如何使用这个假设

在 `tr(Z^T LZ)=1/2 sum_ij W_ij ||Z_i-Z_j||^2` 中，`W_ij` 越大，越强迫两个点靠近。它既是相似度，也是正则置信度。

### 相关论文证据

- Uncertainty-aware geodesic loss 明确区分 dissimilarity value 和 uncertainty。
- TDoA/NLoS 说明同样的几何测量在 LoS 与 NLoS 下置信度不同。
- Robust graph filter identification 说明 graph perturbation 需要被建模，而不是隐藏在固定图里。
- Verifying smoothness 说明 graph signal 是否平滑需要被检验。

### 如果该假设被违反，意味着什么

两个点“看起来相似”不等于“应被同等强度约束”。某些边可能相似度高但可靠性低，某些边相似度中等但物理证据稳定。把二者混在一个 `W_ij` 中，会让图正则缺乏可解释性。

### 不能直接跳到什么

不能把“边可靠性”写成泛泛创新。可靠性必须来自具体证据，例如 uncertainty model、NLoS mask、metric ambiguity、graph perturbation 或 smoothness test。

## 4. 假设 D：普通 GLR 适合 chart 坐标

### 当前公式如何使用这个假设

`tr(Z^T LZ)` 让相连节点坐标接近，本质是低频 / piecewise smooth 先验。对二维 chart，它等价于假设物理邻居在 chart 坐标中也应接近。

### 相关论文证据

- GGLR 指出普通 GLR 可能造成 staircase，并不适合缓慢变化或 planar signal。
- High-frequency booster 说明高频信息可能有用。
- Framelet denoising 说明 feature noise 和 structure noise 可分离，多尺度表示比单一低通更细。
- Verifying smoothness 说明 smoothness 是假设，不是默认真理。

### 如果该假设被违反，意味着什么

普通 GLR 可能把真实局部几何变化、转弯、边界、非均匀采样等当成噪声压掉。此时指标可能短期改善，但几何解释会变弱。

### 不能直接跳到什么

不能因为 GLR 有局限就立刻堆 GGLR / framelet / high-frequency module。必须先证明当前 chart 坐标在 `L_phys` 上出现了普通 GLR 难处理的信号形态。

## 5. 假设 E：全局单一 `lambda` 足够

### 当前公式如何使用这个假设

`lambda` 把 fidelity 和 graph prior 的权衡设为全局常数，等价于假设所有点、边、区域共享同一噪声水平和先验可信度。

### 相关论文证据

- Uncertainty-aware geodesic loss 显示不同 dissimilarity pair 的不确定性不同。
- LoS/NLoS 论文显示传播条件不同。
- LoS Channel Charting 显示 ambiguity 与系统参数、空间位置有关。
- graphLa+ / RED / GGLR 等文献都把 regularization strength 与 noise、bias-variance、prior validity 联系起来。

### 如果该假设被违反，意味着什么

一个 `lambda` 可能在可靠区域过弱，在不可靠区域过强；或在普通点有效，在 tail samples 上失效。此时调一个全局最优值只是折中，不一定揭示机制。

### 不能直接跳到什么

不能把 `lambda` 自适应当成纯工程增强。它必须回答：自适应依据是什么？边不确定性、点不确定性、区域传播条件、还是 smoothness 证据？

## 6. 假设 F：固定图一次闭式解足够

### 当前公式如何使用这个假设

`Z*=(I+lambda L)^-1Y0` 是固定 `L` 的二次问题最优解。一次求解已经完成该模型下的全部优化。

### 相关论文证据

- graphLa+ 支持 initial estimate + graph regularization。
- iterated graphLa 支持在图/解相互依赖时迭代更新。
- Robust graph filter identification 支持图本身可能需要估计或去噪。
- RED / unrolling 支持把求解过程做成更一般的 denoising / learnable inference。

### 如果该假设被违反，意味着什么

如果图可靠性依赖 `Z`，或修复后 chart 能反过来暴露错误边，那么固定图一次求解可能不够。但如果 `L` 完全固定，重复 diffusion 没有新的理论内容。

### 不能直接跳到什么

不能说“迭代一定更好”。迭代的文献依据必须是：

```text
每一轮有新的图、可靠性、denoiser 或约束被更新。
```

## 7. 假设 G：post-hoc transductive refinement 足够

### 当前公式如何使用这个假设

当前 refinement 修的是已有样本集合的 chart。新样本没有天然闭式映射，除非重新并入图或训练额外模型。

### 相关论文证据

- On-the-fly / triplet CC 关注 online 或快速可部署 charting。
- RED / unrolling / interpretable deep denoising 支持把求解过程变成可学习层或 teacher。
- UNILocPro 等训练式框架从一开始就设计可部署 mapping。

### 如果该假设被违反，意味着什么

如果任务目标是 online localization、real-time charting 或新样本泛化，纯 post-hoc refinement 的边界会变成主要问题。

### 不能直接跳到什么

不能为了“可学习”而可学习。只有当 out-of-sample、deployment、teacher-student 或 correction layer 是真正问题时，可学习 refinement 才有文献动机。

## 8. Idea-ready 判断边界

到目前为止，论文学习给出的不是一个具体优化方案，而是一个判断边界：

```text
任何后续优化 idea，都必须先选择一个被违反的假设。
```

如果一个想法说要改 `L`，它必须回答：

- 是因为 ADP/time 边有 uncertainty？
- 是因为 LoS/NLoS 可靠性不同？
- 是因为 PI / radio distance ambiguity？
- 是因为 graph perturbation？

如果一个想法说要改正则项，它必须回答：

- 是因为普通 GLR 压掉真实几何高频？
- 是因为 chart 坐标更像局部仿射/梯度场？
- 是因为 feature noise 和 structure noise 需要分开？
- 是因为 smoothness test 不支持普通低频假设？

如果一个想法说要迭代，它必须回答：

- 每一轮更新什么？
- 更新是否有论文机制支持？
- 是否只是重复同一个滤波器？

如果一个想法说要可学习化，它必须回答：

- 它解决 out-of-sample 还是部署？
- 它是否保留物理解释？
- 它是否只是把一个闭式问题黑箱化？

## 9. 当前学习状态

现在我对参考论文的吸收已经不只是“有哪些相关工作”，而是形成了一个判断框架：

```text
无线物理证据层决定 L_phys 和 W_ij 是否可信；
图信号恢复层决定 tr(Z^T LZ) 和求解方式是否合适；
当前实验现象只能在这两层共同解释下进入 idea 构思。
```

这仍然不是正式 idea。下一步如果继续学习，应继续把参考论文中的具体实验现象、指标和限制，映射到上述七个假设上，而不是提前给出方法。
