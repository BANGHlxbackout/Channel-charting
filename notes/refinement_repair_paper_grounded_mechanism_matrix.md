# Refinement / Repair 主线论文机制学习矩阵

更新时间：2026-06-25

本笔记继续学习 `C:\Users\Hype beast\Desktop\参考论文` 中 34 篇论文，目标不是正式提出 idea，也不安排实验，而是把每篇论文对“如何优化精修”能够提供的知识拆成可追溯机制。

当前讨论主线是：

```text
Y0 = trained channel chart
L_phys = ADP/time/radio-physical graph
Z* = argmin_Z ||Z - Y0||_F^2 + lambda Tr(Z^T L_phys Z)
```

本笔记采用的阅读问题是：

```text
这篇论文解决了什么机制问题？
它能支撑 refinement 的哪个假设修正？
它不能被用来证明什么？
它更接近 L / regularizer / lambda / locality / iteration / learnability 中哪一类问题？
```

## 1. 论文机制总表

| # | 文件 | 论文机制 | 对 refinement/repair 的可迁移认识 | 不能草率推出什么 |
|---:|---|---|---|---|
| 01 | `Indoor_Localization_Robust_Global_CC` | 用局部 CIR distance 和 neighborhood shortest path 学习 global geometry | local metric 只在近邻有效；global geodesic graph 依赖足够高且近似均匀的采样密度 | 不能把 geodesic graph 当作天然可靠，也不能据此声称已有 post-hoc failure detector |
| 02 | `Triplet_Loss_On_the_Fly_CC` | 用 triplet / lightweight on-the-fly CC 快速得到可更新 chart | 初始 chart 可以是快速但粗糙的，这给 post-hoc refinement 留出合理位置 | 不能把后处理精修说成在线 CC 的替代；它目前仍是 transductive refinement |
| 03 | `Model_Based_Approaches_CC` | 从 CSI 显式估计 AoA 与 range，直接得到单点坐标 | model-based coordinate evidence 可绕过一般 DR，但能力依赖阵列、多载波、传播模型、SNR 与参数可估计性 | 主要证据来自指定信道模型仿真，不能当作 NLoS 或真实部署中的无条件真值 |
| 04 | `Wireless_CC_Theory_Practice` | CC 理论、指标、部署形态与挑战综述 | 噪声可产生 false neighbors 并 short-circuit 流形；CT/TW 与 KS 分别诊断邻域和全局距离保持 | 不提供 CC failure 的统一 taxonomy，也不直接支持某个 refinement 算子 |
| 05 | `ADP_Timestamp_Dissimilarity_CC` | ADP dissimilarity + timestamp fusion + geodesic graph | `L_phys` 的核心来源；ADP/time 图是有文献基础的 radio-geometry graph | 不能把 ADP/time metric 当作当前工作的原创点 |
| 06 | `Augmenting_CC_Localization` | 比较仿射后对齐与训练期 AoA/ToA augmentation | affine transform 只能修坐标自由度，不能修 non-affine distortion；物理融合的主要收益发生在 FCF 训练期 | 不能把 training augmentation 当作现成 post-hoc repair，也不能把 alignment 与几何修复混为一谈 |
| 07 | `Global_Scale_Self_Supervised_CC` | 用 ToA/TRP absolute range 和 laser/ICP relative displacement 训练 global-scale FCF | 多源证据约束不同几何自由度，且 ICP、少 TRP、LoS 假设各有失效条件 | 仿真仅覆盖两个 LoS TRP；不能把 sensor fusion 结论直接迁移为当前图精修贡献 |
| 08 | `CC_Real_World_DMIMO` | 用 bilateration 与 LoS bounding-box loss 学习 real-world coordinates | 物理约束需要 LoS 激活，并可能因 AP pair 太少产生不可识别多解；仿射后处理是否有效取决于原 chart 质量 | 不能把高接收功率等同于可靠距离，也不能把训练式弱监督写成当前 post-hoc repair |
| 09 | `CC_Prediction_DMIMO_CSI` | 由 chart 轨迹外推、三角网覆盖和 CSI 插值完成 channel prediction | chart 可承载下游任务，但收益还依赖 dynamics、coverage、interpolation 和 task metric | 短时域可能不如 outdated CSI，且评估排除三角网外样本；不能由 CT/TW/KS 提升直接推出 sum-rate 提升 |
| 10 | `Uncertainty_Aware_Geodesic_Loss` | 把 dissimilarity/geodesic distance 建模成带不确定性的监督 | 最直接支撑“物理边不是等置信度”的认识；`W_ij` 可被理解为相似性与置信度的合成 | 不能简单写成“给边乘 uncertainty 就是新 idea”；该文主要作用在训练 loss |
| 11 | `CC_Smart_Radio_Environments` | 用静态 EMS codebook 改变传播环境并改善 CC tail performance | radio geometry 的可辨识性受传播环境与表面配置影响；该文是 14 的早期版本 | 不研究时间变化或 graph drift，不能与 14 重复计数，也不能声称当前方法处理动态环境 |
| 12 | `TDoA_Self_Supervised_NLoS` | TDoA / NLoS self-supervised localization | LoS/NLoS 是系统性可靠性差异，不是普通高斯噪声；局部图边应考虑传播条件 | 不能在没有 NLoS 识别或代理证据时硬声称能处理 NLoS |
| 13 | `UNILocPro_Unified_Localization` | model-based + CC + LoS/NLoS + OT/map 的统一框架 | 证明前沿趋势是多物理证据融合；也说明轻量 post-training correction 的边界和机会 | 当前 refinement 不能自称等价于 UNILocPro 这种完整训练框架 |
| 14 | `CC_Non_RIS` | 扩展 static EMS 工作，分析 SNR–dissimilarity 权衡与 NLoS tail | 高 SNR 或高 dissimilarity 单独最大化都不足；NLoS 区域可能发生 embedding collapse | 证据来自特定 30 GHz ray-tracing/EMS 场景，不能把 collapse 直接认定为当前 tail error 来源 |
| 15 | `Data_Dependent_Regularization_GL` | graphLa+：初始解 + data-dependent graph Laplacian 正则 | 当前 `Y0 -> L -> Z` 有正规的 inverse-problem 数学邻居；“后估计图正则”不是随意后处理 | graphLa+ 已是成熟框架，不能当作当前创新点 |
| 16 | `Gradient_Graph_Laplacian_Regularizer` | GGLR：普通 GLR 的 staircase/分段常数问题，改用 gradient graph | 支撑“正则项不能只盯坐标值平滑”；chart 坐标是连续几何场，可能更适合梯度/局部仿射先验 | 不能直接照搬 GGLR，除非定义 chart 上的局部坐标/梯度估计 |
| 17 | `Laplacian_Enhanced_Tensor_Kriging` | 用 Laplacian 增强 tensor / kriging 型恢复 | 图拉普拉斯可作为空间/图结构先验嵌入估计问题 | 不足以证明 channel chart refinement 的物理图特殊性 |
| 18 | `GNN_High_Frequency_Booster` | 用 complement Laplacian 为 GNN 节点分类补充高频 | 只支持“低通对某些 GNN/heterophily 任务不充分”这一一般边界 | 任务是离散标签和 hidden representation；不能据此认定 chart 坐标需要高通、complement graph 或 CLAR |
| 19 | `Hyperspectral_GL_Detection` | 图拉普拉斯用于高光谱检测/异常结构 | GLR 可用于突出或抑制结构异常，说明图先验和任务目标相关 | 不支持直接把异常检测目标等同于 chart 修复目标 |
| 20 | `Retinex_Gradient_GL` | 梯度域 / Retinex / GL 联合建模 | 支撑“在梯度或变化量上正则”这一类思路；不是所有任务都应平滑原始值 | 不能证明当前 chart 必须做 gradient GL，只说明普通值平滑有局限 |
| 21 | `Complex_GL_Grid_States` | complex-valued graph Laplacian / grid states | 图拉普拉斯能适配复值/物理状态估计 | 对二维 chart refinement 的直接启发较弱，主要提供物理图信号视角 |
| 22 | `Mixed_Graph_Signal_Denoising` | 混合图信号去噪 | 多种图信号成分可能对应不同噪声/平滑假设 | 不能把 chart 误差简单看成单一同方差噪声 |
| 23 | `Impedance_Inversion_Iterated_GL` | it-graphLa：迭代更新图拉普拉斯正则结果 | 支撑“迭代只有在图/估计随解更新时有意义”；固定 `L` 重复平滑不是新机制 | 不能用它证明固定 `L_phys` 多次 diffusion 有理论贡献 |
| 24 | `Time_Varying_Graph_Signal` | 时变图信号建模 | 如果 chart/物理图随时间或环境变化，静态图先验可能不够 | 当前静态 refinement 不应声称处理时变场景，除非显式建模 |
| 25 | `Interpretable_Lightweight_Transformer` | 轻量、可解释模型设计 | 提醒后续若学习化，应优先保留轻量和可解释，而不是堆复杂网络 | 不直接支撑 Laplacian refinement 公式 |
| 26 | `Unrolling_Plug_and_Play_GGLR` | 把 PnP / GGLR 优化展开成可学习过程 | 支撑“后处理求解器可以变成 layer/teacher”的长期学习方向 | 当前阶段不能把可学习 unrolling 当作已经完成的贡献 |
| 27 | `Interpretable_Deep_Denoise_GL` | GLR MAP 求解展开成少参数可解释 denoiser | 支撑从固定闭式解走向少参数可学习 refinement 的理论位置 | 不能只说“学习 lambda”就等同于深度图去噪 |
| 28 | `Graph_Signal_Denoising_RED` | RED：把 denoiser 显式写入 graph regularization | 支撑把 refinement 看成 denoiser/regularizer 的统一框架；也解释参数估计的重要性 | RED 比当前方法复杂，不应在未实现时写成方法贡献 |
| 29 | `Parameter_Efficient_CT_GL` | 参数高效 CT / graph regularization | 说明图正则可和深度模型结合并保持参数效率 | 对当前 CC 精修主要是风格启发，不是直接理论来源 |
| 30 | `LoS_Channel_Charting` | 分析 LoS MIMO-OFDM 下 PI distance 的周期/振荡歧义和 identifiability | 强力支撑“radio distance 可能有结构性歧义”；错误边不是随机噪声，可能由系统参数和传播模型导致 | 不能把 LoS 条件下的结论直接推广到 mixed NLoS 场景 |
| 31 | `Physical_Fingerprints_CC` | 用 cluster-based GBSM/PCF 构造物理 fingerprint chart | 提醒 `L_phys` 可以从更物理的 fingerprint/cluster parameter 来，而不必永远停在 ADP/time | 当前没有 PCF 时不能声称已使用这种深物理表示 |
| 32 | `Graph_Denoising_Framelet` | framelet / multi-scale feature + structure denoising | 支撑“图去噪可以分解 feature noise 和 structure noise，并用多尺度而非单低通” | 不支持把当前单一 Laplacian 直接称为多尺度结构去噪 |
| 33 | `Robust_Graph_Filter_Identification` | 图扰动下的 robust graph filter identification 和 graph denoising | 最直接支撑“图本身可能错，错误图滤波会放大误差”；可用于反思 `L_phys` 的鲁棒性 | 不能推出我们已经估计了真实图；当前只是在学习该问题 |
| 34 | `Verifying_Smoothness_Graph_Signals` | 检验 graph signal smoothness 假设 | 支撑“先验证 chart 坐标是否在物理图上平滑”；GLR 的前提不是默认真 | 不能只报告 refinement 提升就等于证明平滑假设成立 |

## 2. 精修公式的假设审计

当前二次精修公式看似简单，但每一项都有文献假设：

```math
\min_Z \|Z-Y_0\|_F^2 + \lambda \operatorname{tr}(Z^\top L_{\mathrm{phys}}Z)
```

### 2.1 Fidelity 项的假设

`||Z-Y0||_F^2` 假设 `Y0` 是围绕真实 latent chart 的同质噪声观测。CC 文献显示这不总成立：

- hard users / NLoS / metric ambiguity 可能造成局部系统性偏差；
- tail error 可能不是小高斯扰动；
- 初始 FCF 的错误邻域可能把整块区域折叠或拉远。

因此 fidelity 项的文献风险是：它把结构性 chart 错误当作普通坐标噪声。

### 2.2 Laplacian 项的假设

`tr(Z^T L Z)` 假设物理图相邻点的 chart 坐标应该接近。这个假设受两类论文共同约束：

- ADP/time/LoS/PCF 论文支持物理图作为 radio-geometry evidence；
- uncertainty/NLoS/robust graph filter/smoothness verification 论文提醒图边和图平滑假设需要验证。

因此 Laplacian 项不是“越强越好”的平滑，而是“在可信物理边上施加几何一致性”。

### 2.3 单一 `lambda` 的假设

全局 `lambda` 假设所有点、边、区域共享同一先验强度。文献中至少有四类反例：

- uncertainty-aware geodesic loss：不同 dissimilarity pair 的不确定性不同；
- LoS/NLoS：传播条件不同；
- PI ambiguity：距离歧义与系统参数和空间位置有关；
- robust graph filter：图扰动程度可能局部不同。

因此 `lambda` 的学习重点不是“数值调大调小”，而是理解它代表的证据权衡是否应随边、点或区域变化。

### 2.4 固定图一次求解的假设

闭式解

```math
Z^*=(I+\lambda L)^{-1}Y_0
```

假设图已经给定且足够可信。graphLa+ / iterated GL / robust graph filter 论文说明，如果图来自估计、观测或初值，那么图本身也是不确定对象。一次固定图求解很清楚、稳定、可解释，但它不能处理“图边本身需要修复”的情况。

### 2.5 后处理的假设

post-hoc refinement 假设只需要修当前 batch 的 chart。RED / unrolling / on-the-fly CC / lightweight methods 说明，如果目标是新样本可用或部署，后处理会遇到 out-of-sample 限制。当前学习阶段只记录这个边界，不把它当作已经解决的问题。

## 3. 从论文中学到的“改法必须有原因”

下面不是正式 idea，而是学习得到的判据：以后如果考虑某个精修优化，必须先能回答它对应哪篇论文的机制。

| 表面改法 | 只有在什么论文机制成立时才合理 | 否则会变成什么 |
|---|---|---|
| 改 `L` | 有证据表明当前物理图存在错误边、低置信边或更深物理 fingerprint 可用 | 随便换图 |
| 改 `W_ij` | 有 uncertainty、LoS/NLoS、PI ambiguity、graph perturbation 等边可靠性依据 | 在原权重上拍脑袋乘系数 |
| 改正则项 | 有 evidence 表明普通 GLR 会过平滑 chart 几何或压掉真实高频 | 为复杂而复杂 |
| 改 `lambda` | 有噪声/先验可信度/bias-variance/边不确定性解释 | 纯超参搜索 |
| 局部 refinement | 有区域性传播差异、局部 metric ambiguity 或局部 smoothness 失效证据 | 人为分块 |
| 迭代 refinement | 图、可靠性或 denoiser 会随 `Z` 更新 | 重复执行同一个低通滤波器 |
| 可学习 refinement | 需要 out-of-sample、teacher-student 或可部署 correction layer | 把简单闭式解复杂化 |
| 多尺度 refinement | 有 feature noise 和 structure noise 分离，或真实几何高频需要保留 | 套 framelet/wavelet 名词 |
| robust refinement | 有图扰动模型或错误边传播风险 | 泛泛说鲁棒 |
| smoothness verification | 需要证明 `Y0` 坐标在 `L_phys` 上符合 GLR 前提 | 只做事后解释 |

## 4. 当前学习后的主线2理解

主线2不是“在导师公式附近调几个参数”，而是把精修看成一个由两类证据共同约束的图信号恢复问题：

```text
无线物理证据层：
ADP / timestamp / LoS-NLoS / model geometry / physical fingerprints
-> 决定哪些边、哪些区域、哪些物理约束可信

图信号恢复层：
GLR / GGLR / framelet / RED / unrolling / robust graph filtering / smoothness testing
-> 决定可信图上的坐标信号应该怎样被恢复
```

当前导师方法处在这两个层之间的最简单交点：

```text
ADP/time physical graph + GLR closed-form refinement
```

这个交点可解释、轻量、能与当前实验结果对应。但如果要从论文中吸取优化知识，应该优先围绕四个假设继续学习：

1. 物理图是否可信；
2. chart 坐标是否真的适合普通 GLR；
3. 图与正则强度是否应该局部/边自适应；
4. 后处理是否需要向可部署 correction 过渡。

本轮仍不正式提出优化 idea。这里的价值是把后续所有可能想法都约束到论文机制上，避免把 refinement 优化降级成随便调参。
