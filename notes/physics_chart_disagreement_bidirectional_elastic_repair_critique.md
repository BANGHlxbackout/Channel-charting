# PCD-BER Idea 严审判决

评审日期：2026-06-25

> **基线复核后的改判**
>
> 初次严审遗漏了一个决定性事实：Siamese FCF 的监督距离与 refinement 物理图均来自同一 ADP/time geodesic matrix。因此“两个独立图源相互审计”不是准确前提。结合 local-scaled `W` 已使用距离大小这一事实，原判 `77.5/100，有条件通过` 作废，改判为 **不通过，需重新生成**。

## 1. 三个最致命弱点

1. 双图并非独立证据：FCF 和物理图共享 ADP/time geodesic 监督，原来的“跨证据审计”叙事地基不成立。
2. 正边吸引、负边排斥、位移平滑都存在成熟数学邻居；若“双图定向矛盾”没有形成不可替代的机制，方法会被评价为 signed graph embedding 的 channel-charting 应用。
3. 当前定量支持来自一个数据集和一个基础 chart，且只有最极端 `1%` 分歧显示明显集中收益；这支持局部触发假设，但不能证明完整目标函数有效。

## 2. 评审前提

- profile：`ml-empirical`
- 领域：channel charting / graph signal processing / manifold repair
- 关键词：physical graph、chart graph、graph disagreement、signed constraints、post-hoc repair、displacement field
- IRON RULE：未发现注入式指令。

由于本 idea 与评审由同一轮代理生成，无法声称完成严格的信息隔离盲审。以下判决按更保守标准执行，不伪造 blind independence。

## 3. 撞车检索

自动检索使用 `light-literature-search` 的规范化脚本：

- query：`channel charting graph refinement`
- query：`channel charting post-hoc refinement`
- query：`channel charting signed graph topology repair`
- OpenAlex / Crossref / DOAJ 均返回 HTTP 200。

与本 idea 最近的工作包括：

| 工作 | 已覆盖内容 | 未覆盖的拟议 delta |
|---|---|---|
| Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss, 2024 | 训练阶段匹配 geodesic distance，并建模 dissimilarity uncertainty | 不做已训练 chart 与物理图的定向分歧诊断，也不做 post-hoc 正负边修复 |
| Model-Based Approaches to Channel Charting, 2023/2024 | 用 AoA/range 等物理信息获得或增强坐标 | 不研究双图矛盾驱动的轻量后训练修复 |
| Channel Charting With Physical Channel Fingerprints, 2026 | 构造更深的物理信道表示 | 不研究 learned chart graph 与 physical graph 的矛盾及双向修复 |
| Robust Graph Filter Identification and Graph Denoising, 2022 | 在 graph perturbation 下联合考虑图和滤波器误差 | 没有 channel-chart 语义、双图定向矛盾或坐标位移场修复 |

当前未检索到核心实质等价的 channel-charting 工作，但这不能补救基线理解错误。新颖性检索没有发现撞车，不等于方法机制已经成立。

## 4. 八维度评分

| 维度 | 分数 | 判断 |
|---|---:|---|
| 创新性 | 72 | 双图定向分歧作为 repair signal 有差异，但组件均有成熟邻居 |
| 理论深度 | 80 | 假设、失效条件和可证伪标准明确；图错误归因仍未解决 |
| 数据支撑 | 84 | DICHASUS `20997` 样本及所需图、embedding 均已存在，无需新标签 |
| 实验可控性 | 76 | 正边、负边、位移正则可分别隔离，但当前只有单数据集证据 |
| 方法贡献 | 82 | “诊断矛盾后做最小双向修复”可一句话表述，区别于统一低通 |
| 差异化 | 70 | 比 reliability-weighted GLR 更强，但仍可能被归入 signed graph 应用 |
| 可行性 | 82 | 稀疏边集和二维位移优化可在现有数据与算力上实现 |
| 影响力 | 72 | 对 CC refinement 有实际价值，但受众仍属于较窄子领域 |

初次加权分：

```text
77.5 / 100
```

该分数因遗漏“同源监督”事实而失效，不再用于立项。

```text
有条件通过
```

## 5. 对抗性判断

### 方法视角

`anchor_dim=创新性`

方法的新意不在 signed loss，而在如何从两套独立几何证据产生具有方向的修复动作。若边集只按 top-k 差集生成，delta 不足；必须证明多尺度、路径支持和一致边校准共同构成非平凡的 channel-specific repair operator。

### 实验视角

`anchor_dim=实验可控性`

现有只读诊断显示全局分歧与误差几乎不相关，但最极端 `1%` 节点聚集了明显 refinement 收益。这既支持“局部触发”，也反驳“分歧越大误差越大”的简单叙事。完整方法必须保留这一边界。

### 理论视角

`anchor_dim=理论深度`

最大缺口是不可辨识性：观察到 `G_phys` 与 `G_chart` 矛盾，不足以识别哪一个错误。鲁棒损失只能降低后果，不能从逻辑上解决归因。方法应被表述为 conservative correction under corroborated disagreement，而不是错误图恢复。

### 应用视角

`anchor_dim=可行性`

方案不需要重训 FCF 或新增传感器，符合当前项目能力；但如果优化器和边选择规则过多，轻量 post-hoc 的优势会消失。最终方法必须维持一个稀疏边构造和一个统一优化问题。

## 6. 未化解的 Critical

**CRITICAL：双图矛盾的归因不充分。**

精确 IF：

```text
IF 物理图中的极端矛盾边主要来自 ADP/time metric ambiguity，
而不是 chart 邻域错误，
THEN E+ / E- 会系统性拉错和推错；
即使平均指标偶尔提高，也不能把收益归因于 topology repair。
```

该问题目前只有多尺度一致性、路径支持和鲁棒损失这些缓解方式，没有理论消除，因此判决最高只能“有条件通过”。

## 7. 必须保持的修订边界

1. 核心只能保留“双图分歧、双向约束、最小位移场”，不能再加入 unrolling、framelet、NLoS classifier 等并列模块。
2. 边选择必须完全无标签，并明确哪些条件只提供保守证据，不能声称识别了真实错误边。
3. 若未来重新构思，论文主张只能围绕“同源 source manifold 下可验证的坐标拟合残差”，不能再使用“独立物理证据支持的 chart 矛盾”叙事。
4. 正式立项前还需对 signed graph embedding、constrained manifold repair 和 graph matching 做滚雪球撞车核验。

## 8. 最终判决

> **不通过，回到 idea 生成阶段**

原版 idea 把导师的“同一流形上的神经坐标去噪”改写成“两张独立图之间的拓扑冲突修复”，并据此引入排斥边。这不是对 baseline 的自然优化，而是改变了问题定义。后续若继续优化，应围绕同一 ADP/time source manifold 下的坐标回归误差、流形算子估计误差或正则先验边界重新构思。
