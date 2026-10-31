# Refinement / Repair：CC 失效证据与训练—后处理边界学习

更新时间：2026-06-25

本笔记继续学习参考论文，只吸收知识，不正式提出优化 idea，不安排实验。本轮精读：

- 01 `Indoor Localization with Robust Global Channel Charting`
- 03 `Model-Based Approaches to Channel Charting`
- 04 `Wireless Channel Charting: Theory, Practice, and Applications`
- 06 `Augmenting Channel Charting with Classical Wireless Source Localization Techniques`
- 08 `Channel Charting in Real-World Coordinates with Distributed MIMO`
- 14 `Towards Channel Charting Enhancement with Non-Reconfigurable Intelligent Surfaces`

本笔记不建立自创的 failure taxonomy。下面只整理论文明确展示或讨论的失效机制，并把“论文事实”和“对当前 refinement 的学习含义”分开。

## 1. 六篇论文共同回答了什么

这六篇论文把 chart 误差拆成了至少三个不同层次：

1. **输入证据层**：CSI 距离、时间、AoA、ToA、接收功率或信道特征是否真的携带可辨识的空间关系。
2. **测地图与训练层**：局部关系如何传播成全局几何，训练目标是否能消除尺度、旋转、弯曲和错误邻域。
3. **坐标校正层**：已经得到的 chart 能否通过仿射变换或其他后处理映射到目标几何。

因此，论文中的“chart 不好”不是一种单一噪声。它可能来自错误邻居、局部距离超出有效范围、非均匀采样、物理模型失配、约束不可识别、非仿射失真，或信道本身缺少位置可辨识性。

## 2. 论文 01：局部距离有效，不等于全局图可靠

### 论文明确给出的机制

- CIR distance 只在空间近邻范围内近似与欧氏距离线性相关；对远点既不可靠，也不具有唯一反演关系。
- 论文通过 neighborhood graph 上的最短路径，把局部距离累积成 global geodesic distance。
- 这一推导依赖高空间采样密度：局部边足够短，局部线性关系才能沿路径成立。
- 论文进一步明确指出，构造有效的全局 neighborhood graph 理想上需要**高且近似均匀的采样密度**；总样本多不代表空间分布均匀。
- triplet 与 constrained AE 难以恢复全局几何，论文分别归因于 triplet 构造假设过强和远距离 metric 的局限。

### 对当前 refinement 的学习含义

物理图不是因为使用 ADP/time 就天然正确。全局几何质量同时取决于：

```text
局部距离的有效范围 + 邻域选择 + 采样覆盖 + 最短路径传播
```

这篇论文支持“ADP/time 等物理度量可定义显式几何图”，但在当前 baseline 中该图同时也是 FCF 的训练监督来源，因此不能称为统计独立证据。它也不支持“所有物理边和所有 geodesic 路径都可信”，更没有给出现成的 post-hoc chart failure detector。

## 3. 论文 03：模型给出绝对几何，但能力来自强结构假设

### 论文明确给出的机制

- 该工作绕过一般 dimensionality reduction，直接从 CSI 估计 AoA `theta` 和距离 `rho`。
- 局部几何保持不是显式目标，而是准确估计单点位置后的结果。
- 不同方法依赖阵列、相位、功率、已知路径损耗关系、多载波以及 MUSIC 等结构。
- 论文包含 LOS、Quadriga LOS 和 Quadriga NLOS 仿真，并显示低 SNR 下普通 PCA/SM/AE、LR/ISQ 明显退化；MM/JM/RS 相对更稳健。
- 证据主要来自指定信道模型下的仿真。论文自己也指出 vanilla LOS 模型是理想化模型，理论偏差与方差分析仍未完成。

### 对当前 refinement 的学习含义

model-based evidence 的价值是提供 chart 之外的几何参照，但它不是免费真值。其可靠性依赖硬件结构、传播模型、SNR 和参数可估计性。

所以“物理信息”至少包含两类不同对象：

- 从数据相似性构造的 radio-geometry evidence；
- 从显式传播模型估计的 coordinate evidence。

两者都可能失败，但失败条件不同，不能在 refinement 中不加区分地视为同质约束。

## 4. 论文 04：错误邻居会直接破坏流形拓扑

### 论文明确给出的机制

- CSI 估计噪声会把本不相邻的样本变成 false neighbors。
- 对依赖局部 neighborhood 的 DR 方法，错误邻居可能“short circuit”流形的两个折叠部分，从而扭曲推断出的 topology。
- Trustworthiness 和 continuity 比较 reference space 与 latent space 的邻域是否一致：
  - latent space 中出现不该出现的邻居，会损害 trustworthiness；
  - reference space 中应保留的邻居在 latent space 丢失，会损害 continuity。
- Kruskal stress 衡量成对距离的全局匹配程度，和 CT/TW 不是同一种信息。
- 参数化 charting、非参数 charting和 out-of-sample extension 属于不同部署形态。
- 论文明确提出 continuous/lifelong learning、环境变化、分布式多视图和阵列/传播场景仍是开放问题。

### 对当前 refinement 的学习含义

这是六篇中对“结构性错误”最直接的理论证据：错误邻域并非普通坐标噪声，它会改变流形连接关系。

但论文只给出一般 DR 机制，没有建立 CC 专用的 fold/tear/bridge 等标准分类。因此后续只能使用“false-neighbor short circuit”“missing/extra neighbors”等有原文依据的表述，不能把自定义名称冒充既有 taxonomy。

## 5. 论文 06：仿射后处理只能修坐标自由度，不能修非仿射失真

### 论文明确给出的机制

- 基于 ADP/time fused geodesic dissimilarity 训练的 FCF 能保留局部邻域并恢复 L 形全局拓扑，但输出仍可任意缩放、旋转和翻转。
- 使用 classical localization estimates 学习仿射变换，可以处理这些全局坐标自由度。
- 论文明确指出：如果 chart 已有 non-affine distortion，仿射变换无法修正。
- 其示例中，仿射对齐后的 L 形夹角仍偏钝；把 AoA、ToA 与 dissimilarity 同时放进训练 loss 后，边更直且夹角更接近真实直角。
- augmented CC 的提升发生在 FCF 训练阶段，而不是对固定 chart 做图扩散。

### 对当前 refinement 的学习含义

这篇论文划清了两类后处理：

```text
坐标对齐：修 scale / rotation / reflection / translation
几何修复：改变局部或全局非仿射结构
```

当前 graph refinement 属于第二类，而非普通 alignment。论文 06 证明训练期融合物理信息可以减少非仿射失真，但没有解决“固定已训练 chart 如何做 post-hoc repair”。

## 6. 论文 08：弱物理约束存在可识别性和误分类边界

### 论文明确给出的机制

- self-supervised chart 通常保留局部几何，但全局会缩放、旋转和 warp，并处于任意坐标系。
- bilateration loss 只使用 AP 相对接收功率与已知 AP 位置；LoS bounding-box loss 使用 AP 的 LoS 区域。
- bilateration 只在至少有一对 LoS AP 时可用；AP pair 太少时，同一约束可能对应多个位置，产生定位歧义。
- 接收功率在 NLoS 下不能可靠表示距离，因此论文先用功率阈值判断 LoS，再激活相应约束。
- 阈值由人工观察选取，论文明确把自动化 LoS/NLoS 分类留作后续工作。
- 在复杂室内场景中，即使使用大量标签拟合仿射变换，也无法修正原 chart 的 non-affine distortion；但如果原 chart 全局几何本来就好，仿射变换可以表现很好。
- 论文特别区分：训练时使用位置标签才称 supervised/semi-supervised；训练后才用标签做变换不属于同一概念。

### 对当前 refinement 的学习含义

物理约束的关键不只是“有没有”，还包括：

- 约束是否在当前传播条件下成立；
- 约束是否足以唯一确定几何；
- 用于激活约束的 LoS/NLoS 判别是否可靠；
- 初始 chart 的失真是否属于后处理算子能够表达的变换族。

这比简单说“给边加可靠性”更深：某些证据即便置信度很高，也可能因约束本身不可识别而无法确定正确位置。

## 7. 论文 14：高 SNR 和高 dissimilarity 都不等于高可辨识性

### 论文明确给出的机制

- 在其 30 GHz 城市射线追踪场景中，只最大化信号增益会抑制 location fingerprints，从而降低 CC 质量。
- 随机相位增加信道差异性，但会损失 SNR。
- 论文的核心观察是 SNR 与 spatial dissimilarity 之间需要平衡，而不是单独最大化任一项。
- 无 EMS 时，NLoS 区域会在 embedding 中塌缩成紧密簇，空间次序被破坏。
- quantile-driven 目标针对 worst-case users，报告了 P90 localization error 和 severe trajectory dropouts 的改善。
- 证据范围是特定 EMS、码本、场景和仿真配置；论文也承认性能依赖 EMS 位置和 codebook，完整最优设计需要联合网络规划。

### 对当前 refinement 的学习含义

这篇论文最重要的知识不是 EMS 本身，而是：

```text
可测得更强的信号，不代表不同位置更容易区分；
更大的特征差异，也不代表这些差异稳定可靠。
```

因此，physical graph 的质量不能只由单一距离幅值或单一 SNR proxy 判断。它还取决于局部位置可辨识性和最坏区域是否发生特征塌缩。

## 8. 论文支持的失效机制，不是标准分类

目前可以严谨使用的证据单元如下：

| 论文支持的现象 | 直接来源 | 可以说明什么 | 不能说明什么 |
|---|---|---|---|
| 噪声产生 false neighbors，并 short-circuit 流形 | 04 | 错误邻域会扭曲 topology | 不能自动推出当前数据发生了同样错误 |
| local metric 对远点失去可靠性或唯一性 | 01 | 全局距离不能直接由任意 pairwise CSI distance 得到 | 不能证明所有 geodesic 都错误 |
| 非均匀采样削弱 global neighborhood graph | 01 | 图可靠性依赖覆盖密度 | 不能仅凭样本总量判断 |
| chart 存在任意坐标自由度 | 04, 06, 08 | scale/rotation/reflection 不等于结构错误 | 不能把所有对齐误差当作 repair 对象 |
| affine transform 无法修 non-affine distortion | 06, 08 | 对齐与几何修复必须区分 | 不能证明任意 nonlinear repair 都合理 |
| LoS 物理关系在 NLoS 下失效 | 06, 08 | 物理证据有传播条件 | 不能把 NLoS 当作当前 tail error 的已证实来源 |
| 少量物理约束可能不可识别 | 08 | 高置信约束仍可能有多解 | 不能只靠置信度解决所有问题 |
| NLoS 区域发生 embedding collapse | 14 | 环境可辨识性不足会形成区域性失败 | 不能泛化为所有数据集的固定 failure type |
| 高 SNR 与高 dissimilarity 需权衡 | 14 | graph evidence 质量不是单指标问题 | 不能直接迁移其 EMS 优化目标到当前方法 |

这张表是“证据索引”，不是一个已经被领域接受的 failure taxonomy。

## 9. Training 与 post-hoc refinement 的边界

| 工作 | 物理信息进入的位置 | 是否修改已训练 chart | 是否天然支持新样本 |
|---|---|---:|---:|
| 01 global geodesic Siamese | metric 构造与 FCF 训练 | 否 | 是，参数化 FCF |
| 03 model-based CC | 直接单点坐标估计 | 不适用 | 是 |
| 04 theory/practice | 给出 parametric / nonparametric / online 边界 | 不提出 repair | 取决于实现 |
| 06 affine alignment | 训练后坐标变换 | 只做仿射变换 | 是，变换可复用 |
| 06 augmented CC | FCF 训练 loss | 否 | 是 |
| 08 bilateration / box loss | FCF 训练 loss | 否 | 是 |
| 08 affine baseline | 训练后坐标变换 | 只做仿射变换 | 是 |
| 14 EMS enhancement | 先改变传播环境，再训练 chart | 否 | 依赖已训练映射 |
| 当前 physical graph diffusion | 固定 `Y0` 后的图优化 | 是 | 当前是 transductive |

由此可以确认：当前主线 2 在这组文献中不是重复工作。相邻论文主要改 metric、训练 supervision、传播环境或坐标对齐；当前方法研究的是已有 chart 的 post-hoc physical-graph correction。与此同时，它也必须单独回答图可靠性、非仿射修复能力和 out-of-sample 边界，不能借用训练式方法的结论替代。

## 10. 本轮吸收后的知识状态

本轮补读后，原来的两个学习缺口已有明显收敛：

1. **CC failure diagnosis**：已经有论文支持的具体机制语言，包括 false neighbors、short-circuit、local-metric range、sampling-density failure、non-affine distortion、constraint ambiguity 和 NLoS embedding collapse。
2. **训练与后处理边界**：已经能逐篇区分 metric design、training-time physical supervision、affine alignment、environment engineering 与 post-hoc repair。

仍然没有完成的部分是：

- 上述机制还没有变成针对当前数据的可计算诊断；
- 这些现象不是领域已有的统一分类体系；
- 目前不能判断哪一种机制实际主导当前 FCF 的 tail error；
- 还不能据此正式选择优化 idea。

当前更准确的学习结论是：

```text
参考论文已经把“为什么 chart 可能需要 repair”解释到了机制层；
下一阶段仍需继续吸收剩余论文，建立这些机制与图恢复算子之间更完整的对应关系，
但本轮不正式设计方法。
```
