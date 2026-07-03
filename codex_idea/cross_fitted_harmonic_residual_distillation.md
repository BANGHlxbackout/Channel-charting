# Cross-Fitted Harmonic Residual Distillation for Diffusion-Refined Channel Charting

更新时间：2026-07-02

## 0. 当前定位

本文档记录一个新的主候选优化 idea：

```text
Cross-Fitted Harmonic Residual Distillation, CF-HRD
交叉拟合图调和残差蒸馏
```

它不是导师原始 `diffusion_refined` 方法的替代物，而是对导师方法的升级：

```text
导师 baseline 解决的是：已训练 chart Y0 如何用 ADP/time 物理图做 transductive refinement。
CF-HRD 解决的是：这个 refinement 产生的物理图修复信息，如何在无标签、无自锚泄漏的条件下蒸馏回一个可泛化 correction module。
```

更短地说：

```text
把当前一次性物理图调和精修，从 post-hoc solver 变成可学习的、可 out-of-sample 使用的 residual repair teacher。
```

这条主线仍然围绕 `refinement / repair`，不是回到前端 metric training。

## 1. 与导师 baseline 的对齐

导师当前 baseline 是：

```math
Y_0 = f_\theta(H),
```

```math
Z^\star
=
\arg\min_Z
\|Z-Y_0\|_F^2
+
\lambda \operatorname{tr}(Z^\top L_{\mathrm{phys}} Z),
```

等价线性系统为：

```math
(I+\lambda L_{\mathrm{phys}})Z^\star=Y_0.
```

其中：

| 符号 | 含义 |
|---|---|
| `H_i` | 第 `i` 个样本的 CSI / CIR / 信道特征输入。 |
| `f_theta` | Siamese FCF，输入信道特征，输出二维 channel chart 坐标。 |
| `Y0 in R^{n x 2}` | 已训练 FCF 在 `n` 个样本上的初始 chart，`Y0_i=f_theta(H_i)`。 |
| `d_phys(i,j)` | ADP/time fused geodesic distance。它来自导师 baseline 的源流形监督。 |
| `W_phys` | 由 `d_phys` 的 kNN 和 local scaling 构造的物理图边权。 |
| `L_phys=D-W_phys` | 物理图组合 Laplacian。 |
| `Z in R^{n x 2}` | 精修后的 chart 坐标。 |
| `lambda` | 初始 chart fidelity 与物理图 Dirichlet energy 之间的全局权衡。 |

当前结果说明这个 baseline 是有效的：

| 方法 | mean | p95 | max | CT/TW |
|---|---:|---:|---:|---:|
| Siamese FCF `Y0` | 0.4093 | 0.8558 | 7.8936 | 0.9957 / 0.9960 |
| ADP/time 物理图 GLR | 0.3187 | 0.6346 | 1.1414 | 0.9982 / 0.9982 |
| `Y0` 欧氏图 GLR | 0.3944 | 0.7645 | 8.4133 | 0.9955 / 0.9956 |

因此，CF-HRD 不能推翻导师核心。它应继承两个事实：

1. `L_phys` 的价值来自 ADP/time source manifold，而不是普通 Laplacian 形式本身。
2. `L_phys` 独立于 `Y0` 输出坐标构图，但与 FCF 的训练监督同源，不能被写成统计独立真值。

## 2. 为什么不继续走 CF-PCGR

已有 `codex_idea/cross_fitted_physical_consistency_gated_refinement.md` 和 `codex_idea/cf_pcgr_validation_report.md` 表明：

```text
cross-fitted physical residual 有诊断价值，
但把 residual 直接映射成 node-wise alpha_i fidelity gate，
没有超过当前物理图 GLR baseline。
```

关键观察是：

```text
residual vs 初始误差有相关性；
但连续 alpha-gated Tikhonov solver 会削弱当前强 GLR 的整体平衡。
```

所以 CF-HRD 不再把 cross-fitted residual 当作最终 solver 的门控权重，而是保留它真正有价值的部分：

```text
用遮蔽自身锚点后的物理图调和解，产生 leakage-free repair residual target。
```

换句话说，CF-HRD 的核心不是：

```text
大 residual -> 改 alpha_i -> 再解一次图扩散。
```

而是：

```text
大 residual / 小 residual 都作为 teacher signal，
训练一个 residual correction module 去模仿物理图调和修复。
```

## 3. 核心 idea

CF-HRD 的一句话定义：

```text
在训练样本上随机遮蔽一部分节点的自身 chart 锚点，用 ADP/time 物理图从其余节点图调和预测这些遮蔽节点，得到无自锚泄漏的修复残差；再把这个残差蒸馏给一个由 CSI 和物理 landmark context 条件化的轻量 correction module，使物理图 refinement 的修复能力可用于新样本。
```

它要解决的不是“普通 GLR 还不够平滑”，而是当前导师 baseline 的两个论文级短板：

1. **Transductive limitation**：`Z=(I+\lambda L)^{-1}Y0` 只给当前样本集输出 refined coordinates，新 UE 到来时不能直接获得同等级修复。
2. **Self-anchor leakage**：如果直接把完整 GLR 结果 `Z^\star` 当 teacher，`Z_i^\star` 中包含 `Y0_i` 自己的 fidelity 锚点，student 容易学成 identity / small smoothing，而不是学习物理图邻域给出的修复信息。

CF-HRD 的改法是：

```text
用 cross-fitting 去掉 teacher 对被预测点自身 Y0_i 的直接依赖，
再用 residual distillation 把“物理图邻域如何修复这个点”学出来。
```

## 4. 方法公式

### 4.1 数据、图与初始 chart

给定训练样本集合：

```math
V=\{1,\ldots,n\}.
```

每个节点 `i` 有信道输入：

```math
H_i.
```

基础 FCF 输出：

```math
Y_{0,i}=f_\theta(H_i)\in\mathbb{R}^2.
```

ADP/time 物理图：

```math
G_{\mathrm{phys}}=(V,E_{\mathrm{phys}},W_{\mathrm{phys}}).
```

边权沿用导师 baseline 的 local-scaled 形式：

```math
W_{ij}^{\mathrm{phys}}
=
\exp\left(
-
\frac{d_{\mathrm{phys}}(i,j)^2}{\sigma_i\sigma_j}
\right),
```

其中 `sigma_i` 是节点 `i` 的局部尺度，例如第 `k` 个物理邻居距离。图 Laplacian 为：

```math
L_{\mathrm{phys}}=D_{\mathrm{phys}}-W_{\mathrm{phys}},
\qquad
D_{ii}=\sum_j W_{ij}^{\mathrm{phys}}.
```

### 4.2 Cross-fitted harmonic teacher

把节点划分为 `K` 个互不重叠的 fold：

```math
V=M_1\cup\cdots\cup M_K,
\qquad
M_a\cap M_b=\varnothing \quad (a\ne b).
```

对第 `k` 个 fold，定义对角遮蔽矩阵：

```math
S^{(-k)}_{ii}
=
\begin{cases}
0, & i\in M_k,\\
1, & i\notin M_k.
\end{cases}
```

这里 `S^{(-k)}_{ii}=0` 的意思是：节点 `i` 的 `Y0_i` 不作为自己的 fidelity anchor。

对每个 fold 解图调和 teacher：

```math
Z^{(-k)}
=
\arg\min_Z
\sum_{i\notin M_k}\|Z_i-Y_{0,i}\|_2^2
+
\lambda_T\operatorname{tr}(Z^\top L_{\mathrm{phys}}Z).
```

等价线性系统为：

```math
\left(S^{(-k)}+\lambda_T L_{\mathrm{phys}}\right)Z^{(-k)}
=
S^{(-k)}Y_0.
```

变量说明：

| 符号 | 含义 |
|---|---|
| `M_k` | 第 `k` 个被遮蔽节点集合。 |
| `S^{(-k)}` | 对角矩阵；未遮蔽点为 1，遮蔽点为 0。 |
| `lambda_T` | teacher 图调和强度，可先取导师 baseline 的 `lambda=0.15`。 |
| `Z^{(-k)}` | 第 `k` 个遮蔽问题的图调和解。 |

对 `i in M_k`，`Z_i^{(-k)}` 是一个关键对象：

```text
它不是完整 GLR 的输出；
它是在不使用 Y0_i 自身锚点的情况下，
由 ADP/time 物理图邻域和其他节点 chart 坐标推断出的 out-of-fold 坐标。
```

### 4.3 Leakage-free residual target

对每个节点 `i`，令 `k(i)` 表示它所在的 fold，定义 teacher residual：

```math
\Delta_i^\star
=
Z_i^{(-k(i))}-Y_{0,i}.
```

也可以定义 teacher coordinate：

```math
T_i^\star
=
Z_i^{(-k(i))}
=
Y_{0,i}+\Delta_i^\star.
```

变量说明：

| 符号 | 含义 |
|---|---|
| `Delta_i^star` | 物理图调和 teacher 对节点 `i` 的修复残差。 |
| `T_i^star` | 节点 `i` 的 out-of-fold teacher 坐标。 |
| `Y0_i` | 初始 FCF 坐标，作为 residual 的起点，而不是 teacher 的自身锚点。 |

这一项是 CF-HRD 的核心：它避免了直接蒸馏完整 GLR 时的自锚泄漏。

完整 GLR teacher：

```math
Z^\star=(I+\lambda L)^{-1}Y_0
```

中，`Z_i^\star` 直接依赖 `Y0_i`。CF-HRD teacher：

```math
Z_i^{(-k(i))}
```

在生成 `i` 的 target 时去掉了 `Y0_i` 的 fidelity 项，因此更像一个物理图邻域预测任务。

### 4.4 Landmark-conditioned student

如果只让 student 输入 raw CSI：

```math
g_\phi(H_i)
```

它可能很难学习图调和 residual，因为 residual 本质上依赖节点在 ADP/time source manifold 上的上下文位置。为此，CF-HRD 引入一个轻量物理 landmark context：

```math
q_i=\Gamma(i;\mathcal{A},d_{\mathrm{phys}}).
```

其中 `\mathcal{A}={a_1,\ldots,a_B}` 是 `B` 个物理 landmark，可由 ADP/time graph 上的 farthest-point sampling、k-medoids 或均匀轨迹采样得到。`q_i` 可以取：

```math
q_i
=
\left[
\exp\left(-\frac{d_{\mathrm{phys}}(i,a_b)^2}{\sigma_i\sigma_{a_b}}\right)
\right]_{b=1}^B,
```

或取归一化的 landmark distance：

```math
q_i
=
\left[
\frac{d_{\mathrm{phys}}(i,a_b)}{\sigma_i+\epsilon}
\right]_{b=1}^B.
```

student correction module 定义为：

```math
\widehat{\Delta}_i
=
g_\phi(H_i,q_i),
```

最终输出：

```math
\widehat{Y}_i
=
f_\theta(H_i)+g_\phi(H_i,q_i).
```

变量说明：

| 符号 | 含义 |
|---|---|
| `\mathcal{A}` | 物理 landmark 集合。 |
| `B` | landmark 数量。 |
| `q_i` | 节点 `i` 相对 landmark 的物理图上下文特征。 |
| `g_phi` | 轻量 residual correction module。 |
| `widehat{Delta}_i` | student 预测的修复残差。 |
| `widehat{Y}_i` | 可 out-of-sample 使用的最终 refined chart 输出。 |

### 4.5 Distillation loss

最小版本只训练 `g_phi`，冻结 `f_theta`：

```math
\mathcal{L}_{\mathrm{HRD}}(\phi)
=
\sum_{i=1}^n
\omega_i
\left\|
g_\phi(H_i,q_i)-\Delta_i^\star
\right\|_2^2
+
\beta
\sum_{i=1}^n
\left\|
g_\phi(H_i,q_i)
\right\|_2^2.
```

其中：

| 符号 | 含义 |
|---|---|
| `omega_i` | target 置信权重。第一版可取 1；后续可由物理邻域稳定性或 teacher 多次采样方差定义。 |
| `beta` | residual 幅度正则，防止 student 产生过大 correction。 |

若要联合微调 FCF，可写为：

```math
\mathcal{L}_{\mathrm{joint}}(\theta,\phi)
=
\mathcal{L}_{\mathrm{FCF}}(\theta)
+
\mu
\sum_i
\omega_i
\left\|
f_\theta(H_i)+g_\phi(H_i,q_i)-T_i^\star
\right\|_2^2
+
\beta\sum_i\|g_\phi(H_i,q_i)\|_2^2.
```

但第一版建议冻结 `f_theta`，原因是：

```text
导师 baseline 的 FCF 已经很强；
先验证 residual 是否可学习，比端到端重训更干净。
```

### 4.6 新样本 inference

对新样本 `H_new`：

1. 用原 FCF 得到：

```math
Y_{0,\mathrm{new}}=f_\theta(H_{\mathrm{new}}).
```

2. 用 ADP/time metric 计算它到 landmark 的物理上下文：

```math
q_{\mathrm{new}}=\Gamma(\mathrm{new};\mathcal{A},d_{\mathrm{phys}}).
```

3. 输出：

```math
\widehat{Y}_{\mathrm{new}}
=
Y_{0,\mathrm{new}}
+
g_\phi(H_{\mathrm{new}},q_{\mathrm{new}}).
```

这使当前 transductive refinement 获得一个可部署版本。它不要求每个新样本都参与一次全图线性系统求解；如果有 batch 新样本，也可以把 CF-HRD 输出作为 warm start，再做小规模局部图调和。

## 5. 最小可行性证据

已做过一个 sanity check：在当前 `n=20997` 数据上，对遮蔽节点去掉自身 anchor 后，用同一个 ADP/time 物理图求 harmonic teacher。结果显示，即使没有 `Y0_i` 自锚，teacher 对 masked nodes 仍然接近完整 GLR 的修复效果。

随机遮蔽比例下 masked nodes 的 mean / p95 / max：

| mask | FCF on masked | cross-fitted teacher | full GLR on same nodes |
|---|---|---|---|
| random 1% | 0.413 / 0.848 / 1.164 | 0.322 / 0.630 / 0.885 | 0.321 / 0.619 / 0.842 |
| random 5% | 0.413 / 0.852 / 3.978 | 0.330 / 0.655 / 1.113 | 0.327 / 0.641 / 1.074 |
| random 10% | 0.409 / 0.847 / 5.866 | 0.319 / 0.630 / 1.081 | 0.315 / 0.620 / 1.054 |
| random 20% | 0.412 / 0.881 / 5.674 | 0.326 / 0.661 / 1.127 | 0.321 / 0.635 / 1.098 |
| contiguous 10% | 0.386 / 0.794 / 1.229 | 0.324 / 0.629 / 1.069 | 0.334 / 0.597 / 0.887 |

这个结果只能支持一个有限结论：

```text
cross-fitted harmonic teacher 是有效的 repair target。
```

它还不能证明：

```text
student g_phi 一定能从 H_i 和 q_i 学会这个 residual；
也不能证明 CF-HRD 已经优于当前 transductive GLR。
```

因此，CF-HRD 的第一阶段验证重点应是 residual distillation 的泛化，而不是继续证明 graph solver 有效。

## 6. 文献灵感与启发来源

### 6.1 直接支撑导师 baseline 的论文

| 论文 | 标注 | 启发 |
|---|---|---|
| Stephan, Euchner, ten Brink, "Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting", IEEE TCOM 2024. DOI: `10.1109/TCOMM.2024.3386571` | 直接前置 | ADP/time fused metric 是 `L_phys` 的来源；CF-HRD 不能把物理图写成新发明，只能把它作为 source-manifold teacher。 |
| Ferrand et al., "Wireless Channel Charting: Theory, Practice, and Applications", IEEE Communications Magazine 2023. DOI: `10.1109/MCOM.001.2200344` | CC 背景与部署边界 | CC 最终需要 FCF 或 out-of-sample extension 服务新样本；这正是 transductive GLR 的短板。 |
| Yassine et al., "Leveraging triplet loss and nonlinear dimensionality reduction for on-the-fly channel charting", 2022. | on-the-fly / parametric CC 邻近背景 | 说明 channel charting 不只追求离线 embedding，也追求新样本快速投影；CF-HRD 的贡献应放在 refinement 的可部署化，而不是重新做前端 triplet learning。 |

### 6.2 把 refinement 变成 teacher 的图正则论文

| 论文 | 标注 | 启发 |
|---|---|---|
| Bianchi et al., "A Data-Dependent Regularization Method Based on the Graph Laplacian", arXiv:`2312.16936` | graphLa+ 数学邻居 | "初值 + 图 Laplacian 正则" 是成熟框架，所以普通 GLR 本身不是创新；CF-HRD 的新意应来自 cross-fitted teacher 和 residual distillation。 |
| Chen, Eldar, Zhao, "Graph Unrolling Networks: Interpretable Neural Networks for Graph Signal Denoising", IEEE TSP 2021 / arXiv:`2006.01301`. DOI: `10.1109/TSP.2021.3087905` | 图优化学习化 | 图 denoising solver 可以被学习化；但 CF-HRD 不直接 unroll 当前 solver，而是先用 solver 生成无泄漏 residual target。 |
| Cai, Cheung, Chen, "Unrolling Plug-and-Play Gradient Graph Laplacian Regularizer for Image Restoration", 2025/2026 manuscript in local corpus | 可解释 unrolling / GGLR | 学习化不等于黑箱化；如果未来升级 CF-HRD，可把 correction module 做成少参数、物理图可解释结构。 |
| Hosseini et al., "Constructing an Interpretable Deep Denoiser by Unrolling Graph Laplacian Regularizer" | deep GLR denoiser | 支持把 GLR 相关优化看作可解释 denoiser；但当前 CF-HRD 只吸收 teacher-student 思路，不声称实现 unrolled GLR。 |
| Kojima, Higashi, Tanaka, "Graph Signal Denoising Using Regularization by Denoising and Its Parameter Estimation", arXiv:`2512.14213` | RED / denoiser parameter estimation | 说明 graph denoiser 可以作为显式正则或可估计模块；CF-HRD 的 correction module 是更轻量的 channel-chart residual denoiser。 |

### 6.3 为什么必须 cross-fitting 的论文启发

| 论文 | 标注 | 启发 |
|---|---|---|
| Euchner, Stephan, ten Brink, "Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss", arXiv:`2412.01715` | dissimilarity uncertainty | 物理距离不是确定标签，而是 noisy geometric evidence。CF-HRD 不把完整 GLR 当真值，而用 out-of-fold teacher 降低 self-anchor leakage。 |
| Rey, Tenorio, Marques, "Robust Graph Filter Identification and Graph Denoising from Signal Observations", arXiv:`2210.08488` | graph perturbation 边界 | supporting graph 不是绝对真图。CF-HRD 的 teacher target 要保留 `omega_i` 或多次采样方差，避免把错图传播蒸馏成固定偏差。 |
| Ghorbani et al., "GKD: Semi-supervised Graph Knowledge Distillation for Graph-Independent Inference", arXiv:`2104.03597`, DOI:`10.48550/arXiv.2104.03597` | graph-to-student distillation 邻近工作 | graph distillation 本身已有。CF-HRD 不能声称首创 graph distillation；差异是把 ADP/time graph-harmonic repair residual 蒸馏给 channel chart correction module。 |

### 6.4 这些文献没有直接解决的问题

```text
现有 CC 文献：主要改 metric、训练 loss、parametric FCF 或 out-of-sample extension；
现有图正则文献：主要研究一般 graph signal denoising / inverse problem；
现有 graph distillation：主要解决图结构不可用于 inference 的分类/预测问题。
```

CF-HRD 试图填的更窄缺口是：

```text
在 channel charting 中，如何把一个有效但 transductive 的 ADP/time graph-harmonic refinement，
转化成无标签、无自锚泄漏、可用于新 CSI 样本的 residual repair module。
```

## 7. 与已有方法的差异

| 方法 | 机制 | 不足 | CF-HRD 的差异 |
|---|---|---|---|
| 导师原始 GLR baseline | `(I+lambda L_phys)Z=Y0` | transductive；新样本需重建图或插值 | 用 cross-fitted teacher 生成 residual target，再蒸馏成 correction module |
| CF-PCGR | cross-fitted residual -> node-wise fidelity `alpha_i` | 验证结果未超过 GLR baseline | residual 不再直接控 solver，而是作为 teacher target |
| 直接蒸馏完整 GLR | student 拟合 `Z^star-Y0` | teacher 含 `Y0_i` 自锚泄漏，容易学 identity/smoothing | 遮蔽节点自身 anchor，只在 out-of-fold 节点上取 target |
| graph unrolling / RED | 学习优化迭代或 denoiser | 泛用图信号方法，未结合 ADP/time CC teacher | 先保留导师 solver 作为 teacher，不急着改成复杂 unrolled network |
| GKD 类 graph distillation | 图 teacher -> graph-independent student | 非 channel charting；目标多为分类/预测 | 蒸馏对象是二维 chart repair residual，teacher 是 ADP/time graph harmonic solver |
| on-the-fly / out-of-sample CC | 新 CSI 快速投影 | 通常不处理已训练 chart 的物理图后修复 | 在 FCF 输出后增加物理图 residual correction |

## 8. 预期贡献写法

更稳的 claim：

```text
We propose a cross-fitted graph-harmonic residual distillation framework
that turns a transductive ADP/time physical-graph chart refinement solver
into an inductive residual correction module.
The cross-fitted teacher removes each target node's self-anchor during
harmonic prediction, reducing leakage when distilling graph-based repair
signals into a CSI- and landmark-conditioned student.
```

中文：

```text
本文提出交叉拟合图调和残差蒸馏框架，
将 transductive 的 ADP/time 物理图 chart refinement solver
转化为可用于新样本的 residual correction module。
通过在生成 teacher target 时遮蔽目标节点自身 chart anchor，
方法降低了直接蒸馏完整 GLR 解的自锚泄漏风险。
```

不要写成：

```text
首次提出图蒸馏；
首次提出图 Laplacian 精修；
证明物理图是真实几何图；
证明 learned chart 的所有错误结构都可被诊断。
```

## 9. 可证伪预测

如果 CF-HRD 成立，应看到：

1. `Delta_i^star` 可被 `H_i,q_i` 学到，student 在 held-out samples 上优于原始 FCF。
2. 相比直接蒸馏完整 GLR residual，cross-fitted residual teacher 的 held-out 泛化更好。
3. 加入 landmark context `q_i` 比只用 raw CSI 的 `g_phi(H_i)` 更好，说明 residual 依赖物理 source-manifold 上下文。
4. 在训练样本上，student 不一定超过 full GLR；但在新样本 / held-out 设置中，应明显缩小 FCF 与 transductive GLR 的差距。
5. 如果把 `L_phys` 换成 `Y0` 欧氏图，teacher residual 的质量和 student 泛化应下降，说明收益来自 ADP/time source manifold。

如果结果是：

```text
student 只能在训练点拟合 residual，held-out 无提升；
q_i 没有帮助；
直接蒸馏完整 GLR 与 cross-fitted teacher 无差异；
或 correction module 导致 max error 恶化；
```

则 CF-HRD 失败，应回退到导师原始 GLR baseline，而不是继续包装。

## 10. 主要风险

### 风险 1：student 学不到 graph-harmonic residual

`Delta_i^star` 依赖邻域结构，不一定由单点 CSI 决定。因此需要 `q_i`。如果 `q_i` 仍不足，说明 residual 是强 transductive 信号，不适合蒸馏。

### 风险 2：teacher graph 仍可能有错边

ADP/time 图有效但不是 ground truth。若错边进入 teacher，student 会把错边造成的偏差固化。防守方式是引入 `omega_i`：对 fold、`lambda_T`、neighbor subset 重复采样，teacher residual 方差高的点降权。

### 风险 3：贡献可能被认为是普通 teacher-student

防守重点不是“用了蒸馏”，而是：

```text
cross-fitted no-self-anchor harmonic teacher
+
ADP/time source-manifold residual
+
landmark-conditioned inductive chart repair.
```

缺任何一个，创新都会变弱。

### 风险 4：如果只在全数据上评估，论文叙事站不住

CF-HRD 的目标是解决 transductive limitation，因此必须评估 held-out / out-of-sample。只报告全数据 refined chart 指标，会把它退化成普通后处理。

## 11. 第一版实现建议

第一版保持最小：

1. 冻结现有 Siamese FCF，读取当前 `Y0`。
2. 用 `K=5` 或 `K=10` folds 生成 `Delta_i^star`。
3. 选 `B=128` 或 `B=256` 个 ADP/time landmarks，构造 `q_i`。
4. 训练一个小 MLP `g_phi(H_i,q_i)`，目标是 `Delta_i^star`。
5. 对比：
   - 原始 FCF；
   - transductive full GLR；
   - 直接蒸馏 full GLR residual；
   - CF-HRD 无 `q_i`；
   - CF-HRD 有 `q_i`；
   - 欧氏图 teacher 版本。

第一版不建议端到端重训 `f_theta`，也不建议加入复杂 graph neural network。先证明 residual target 可学，再考虑复杂化。

## 12. 当前结论

CF-HRD 比 CF-PCGR 更适合作为新的优化 idea，原因是：

1. 它保留导师 baseline 的最强部分：ADP/time physical graph harmonic refinement。
2. 它不再用 residual 直接调 `alpha_i`，避免重复 CF-PCGR 已经失败的路径。
3. 它针对一个真实论文短板：transductive refinement 不能直接服务新样本。
4. 它的核心技术点不是泛泛蒸馏，而是 cross-fitted no-self-anchor teacher。
5. 它有明确可证伪实验：held-out residual distillation 是否缩小 FCF 与 full GLR 的差距。

因此，当前最合理的定位是：

```text
导师 baseline = 强 transductive physical graph refinement；
CF-HRD = 把该 refinement 的物理图修复能力蒸馏成 inductive residual repair module。
```
