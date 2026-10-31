# Topology-Aware Physical Graph Repair：问题驱动学习笔记

更新时间：2026-07-09

## 0. 本轮学习目标

这份笔记不是重新堆文献，也不是继续修补 `R_ij W_ij`。

当前导师 baseline 是：

```math
Z^\star=\arg\min_Z \|Z-Y_0\|_F^2+\lambda\sum_{ij}W_{ij}^{phys}\|Z_i-Z_j\|_2^2.
```

它的核心短板不是单个 `lambda`，也不是边权公式太简单，而是：

```text
它把 physical graph 直接当成 diffusion graph，
并把所有边都写成同一种 attraction / smoothing 约束。
```

因此本轮学习只围绕三个可转成模块的问题：

```text
1. 哪些物理边是可靠吸引边？
2. 哪些边可能是 shortcut / wrong edge，不能强吸引？
3. folding / tearing / false-neighbor 这类拓扑错误，需要什么不同于 Laplacian smoothing 的约束？
```

## 1. 本地论文与已有笔记已经给出的确定结论

本地资料来源：

```text
C:\Users\Hype beast\Desktop\参考论文
notes/refinement_repair_core_mechanism_cards.md
notes/refinement_repair_paper_grounded_mechanism_matrix.md
notes/refinement_repair_lphys_reliability_learning.md
notes/refinement_repair_cc_failure_and_boundary_learning.md
```

### 1.1 Channel charting 文献给出的 failure 不是“噪声”一种

本地相关论文：

```text
01_Indoor_Localization_Robust_Global_CC
04_Wireless_CC_Theory_Practice
06_Augmenting_CC_Localization
08_CC_Real_World_DMIMO
10_Uncertainty_Aware_Geodesic_Loss
12_TDoA_Self_Supervised_NLoS
13_UNILocPro_Unified_Localization
30_LoS_Channel_Charting
```

学到的机制：

| 机制 | 对当前 refinement 的意义 |
|---|---|
| false neighbor / short-circuit | 错误邻边会改变 manifold topology，不是普通坐标噪声 |
| local metric range | 局部物理距离有效，不代表任意 geodesic 扩散边都可作为一步吸引边 |
| non-affine distortion | 仿射对齐不能修，普通 Laplacian smoothing 也未必能修 |
| LoS/NLoS 条件差异 | 物理证据不是同质可靠；边可靠性可能有传播条件 |
| geodesic uncertainty | `W_ij` 不能同时无损表达 similarity 和 confidence |
| identifiability / ambiguity | radio distance 的错误可能是结构性歧义，不是小扰动 |

对模块的直接启发：

```text
不要再把 refinement 写成“所有物理边都吸引”。
应该先把边分成不同约束类型：
reliable attraction / uncertain metric-preserving / suspicious shortcut。
```

### 1.2 图信号文献说明普通 GLR 不是唯一合理正则

本地相关论文：

```text
15_Data_Dependent_Regularization_GL
16_Gradient_Graph_Laplacian_Regularizer
18_GNN_High_Frequency_Booster
32_Graph_Denoising_Framelet
33_Robust_Graph_Filter_Identification
34_Verifying_Smoothness_Graph_Signals
```

学到的机制：

| 机制 | 对当前 refinement 的意义 |
|---|---|
| graphLa+ | `Y0 -> graph regularization -> Z` 有数学邻居，但本身不新 |
| GGLR | 普通 GLR 鼓励坐标值接近，可能压平连续几何变化 |
| high-frequency / framelet | 图高频不全是噪声，可能包含真实几何弯折或边界 |
| robust graph filter | supporting graph 本身可能有 link perturbation |
| verifying smoothness | graph signal smoothness 是需要检验的假设 |

对模块的直接启发：

```text
不能只做低通 smoothing。
需要把“坐标接近”与“局部几何/距离结构保持”分开。
```

## 2. 联网补充检索到的机制线索

本轮联网检索关键词：

```text
robust manifold learning shortcut edge removal geodesic graph
topology-preserving dimensionality reduction
neighbor embedding attraction repulsion spectrum
graph signal denoising edge reliability topology graph Laplacian
channel charting uncertainty geodesic loss
```

重点补充线索如下。

### 2.1 Attraction-Repulsion Spectrum in Neighbor Embeddings

检索来源：

```text
Attraction-Repulsion Spectrum in Neighbor Embeddings
JMLR / arXiv: https://arxiv.org/abs/2007.08902
```

关键机制：

```text
Neighbor embedding 不只是 attraction。
不同方法可以看作 attraction term 和 repulsion term 的不同组合。
```

对当前方法的启发：

导师 baseline 只有 attraction：

```math
\sum_{ij}W_{ij}\|Z_i-Z_j\|^2.
```

如果存在 false neighbor / shortcut，继续 attraction 会加重 folding。  
因此 topology repair 模块至少应考虑：

```text
可靠边：attraction
可疑边：remove attraction 或转成 weak repulsion / margin constraint
```

可转成当前模块的形式：

```math
\lambda_a\sum_{(i,j)\in E_a}w_{ij}\|Z_i-Z_j\|^2
+
\lambda_r\sum_{(i,j)\in E_r}\eta_{ij}[m_{ij}-\|Z_i-Z_j\|]_+^2.
```

这里 `E_r` 不是所有非邻居，而是经诊断的 suspicious shortcut edge。  
这样比简单 `R_ij W_ij` 更有结构能力。

### 2.2 Topology-preserving dimensionality reduction / representation learning

检索来源：

```text
Learning Topology-Preserving Data Representations
arXiv: https://arxiv.org/abs/2302.00136

Topology-Preserving Dimensionality Reduction via Interleaving Optimization
arXiv: https://arxiv.org/abs/2201.13012
```

关键机制：

```text
Topology preservation 关注的不只是 pairwise distance，
而是邻域连通、洞、组件、局部/全局拓扑结构是否被破坏。
```

对当前方法的启发：

当前指标 CT/TW/KS 只在结果上评价邻域/距离；baseline solver 本身没有拓扑错误诊断。  
新模块如果想处理 folding/tearing，需要引入“局部邻域一致性诊断”，例如：

```text
物理邻域 N_phys(i) 与 chart 邻域 N_Y(i) 的交并差；
extra neighbors: chart 里靠近但物理不支持；
missing neighbors: 物理上近但 chart 里远；
ambiguous neighbors: geodesic 近但 direct/local support 弱。
```

这类诊断不使用 UE position，只用 `Y0`、`d_fused`、`d_geo` 和邻域 rank。

### 2.3 Robust graph / shortcut 思路

检索来源：

```text
robust manifold learning shortcut edge removal
short-circuit errors in manifold learning
robust graph filter identification
```

本地最相关论文：

```text
33_Robust_Graph_Filter_Identification
```

关键机制：

```text
supporting graph 可能有错误 link。
错误 link 会通过 graph filter / diffusion 传播到信号恢复结果。
```

对当前方法的启发：

原版 direct-support 过滤失败，是因为它只问：

```text
d_fused 是否支持 d_geo？
```

但真正应该问：

```text
这条边如果作为 attraction，会不会造成 chart topology shortcut？
```

因此边可靠性不能只由 physical metric 内部判断，还要看它和 `Y0` 的局部拓扑关系：

```text
phys says near, Y0 says far: 可能是需要 repair 的 missing edge，也可能是 wrong physical shortcut；
phys says far, Y0 says near: 可能是 chart false neighbor / folding；
phys and Y0 both near: reliable attraction；
phys and Y0 both far: irrelevant。
```

这提示新模块必须是“物理图 - chart 图分歧诊断”，不是单源 physical edge gate。

### 2.4 GGLR / 高阶局部几何先验

检索来源：

```text
Gradient Graph Laplacian Regularizer
arXiv: https://arxiv.org/abs/2206.04245
```

关键机制：

普通 GLR 容易把连续变化信号压成 piecewise constant / staircase。  
GGLR 的思想是让局部变化/梯度结构也成为正则对象。

对当前方法的启发：

二维 chart 坐标不是分类标签；真实空间沿路径应连续变化，而不是在每条边上都尽量相等。  
因此可以考虑加入 metric / scale preserving 约束：

```math
\lambda_s\sum_{(i,j)\in E_s}\gamma_{ij}
\left(\|Z_i-Z_j\|-\hat d_{ij}\right)^2.
```

其中 `E_s` 是中置信边，`hat d_ij` 来自归一化 geodesic/fused distance。  
这比 attraction 更温和：它不要求所有边塌到一起，而是保留局部尺度。

### 2.5 Verifying smoothness / smoothness assumption testing

检索来源：

```text
Verifying Smoothness of Graph Signals
arXiv: https://arxiv.org/abs/2305.19618
```

关键机制：

图信号平滑性不是默认成立，而是要检验。  
对当前方法，`Y0[:,0]` 和 `Y0[:,1]` 是否在 `L_phys` 上低频，是 refinement 合法性的前提。

对当前方法的启发：

新模块可以先做局部 smoothness violation diagnosis，但不能停在 alpha-only。  
诊断结果应该决定“错误类型”：

```text
low violation + graph/chart一致：保留；
high violation + phys支持但chart远：repair missing relation；
high violation + chart近但phys不支持：标记 chart false neighbor；
phys edge causing local rank conflict：疑似 shortcut。
```

## 3. 现在对 Physical Edge Reliability 的修正判断

失败的是：

```text
Direct-Support-Aware Edge Filtering
R_ij = f(d_fused / d_geo)
W_rel = R_ij W_phys
```

它失败的根本原因：

```text
它只改变边强度，不改变边的语义；
它仍把所有保留下来的边写成 attraction；
它不能区分 useful geodesic relation 和 harmful shortcut；
它没有利用 Y0 与物理图之间的拓扑分歧。
```

Physical Edge Reliability 若继续，必须升级为：

```text
Physics-Chart Disagreement Driven Topology Repair
```

也就是可靠性不再是物理距离内部的一个数，而是：

```text
physical graph evidence
+ current chart neighborhood evidence
+ local rank / smoothness conflict
=> edge type
```

## 4. 候选模块雏形：多类型物理图修复，而不是单一 Laplacian

输入：

```text
Y0: initial chart
d_fused: direct ADP/time fused distance
d_geo: geodesic distance
W_phys: baseline physical diffusion graph
N_Y: kNN graph in Y0 space
```

### 4.1 边类型诊断

对 candidate pair `(i,j)` 构造四类分数：

```text
phys_rank_ij: j 在 i 的物理/geodesic 邻域 rank
chart_rank_ij: j 在 i 的 Y0 欧氏邻域 rank
direct_support_ij: d_fused 是否支持 d_geo
local_consistency_ij: N_phys(i) 与 N_phys(j)、N_Y(i) 与 N_Y(j) 的邻域重叠
```

边类型：

```text
E_a reliable attraction:
    phys near + chart near + local neighborhoods consistent

E_m missing / repair attraction:
    phys near + chart far + direct/local support strong

E_s scale-preserving:
    phys near + chart moderately far 或 support 中等

E_r suspicious shortcut:
    phys near + chart far + direct/local support weak 或 neighborhood consistency low

E_f chart false-neighbor repulsion:
    chart near + phys far
```

### 4.2 优化目标

第一版不应一开始全上，最小但有实质变化的目标可以是：

```math
\min_Z
\|Z-Y_0\|_F^2
+
\lambda_a\sum_{(i,j)\in E_a\cup E_m}w_{ij}\|Z_i-Z_j\|^2
+
\lambda_s\sum_{(i,j)\in E_s}\gamma_{ij}
(\|Z_i-Z_j\|-\hat d_{ij})^2
+
\lambda_r\sum_{(i,j)\in E_r\cup E_f}\eta_{ij}
[m_{ij}-\|Z_i-Z_j\|]_+^2.
```

和导师 baseline 的本质区别：

```text
baseline: 所有物理边都是 attraction low-pass smoothing
new module: 不同拓扑诊断结果进入不同约束类型
```

### 4.3 为什么这个比 edge reliability 更像实质模块

```text
1. 它不只是调边权，而是改变约束族；
2. 它显式处理 false neighbor / shortcut；
3. 它保留中置信边的尺度信息，而不是简单吸引或删除；
4. 它利用 Y0 与物理图的分歧来诊断错误类型；
5. 它能做清晰消融：attraction only / +scale / +repulsion。
```

## 5. 当前不能立刻实验的点

现在还不能直接写代码跑完整模块，因为还有两个关键设计需要继续学习/定稿：

```text
1. E_r suspicious shortcut 的无监督判据不能拍脑袋；
2. scale-preserving target d_hat_ij 的尺度归一化要避免借用 UE position。
```

但已经可以排除：

```text
继续做 R_ij * W_ij 的 direct-support 静态过滤。
```

## 6. 下一步学习清单

继续重点读/查：

```text
1. robust manifold learning / shortcut edge removal:
   找可计算的 shortcut 判据。

2. attraction-repulsion neighbor embedding:
   明确 repulsion term 如何只作用于 selected negative pairs，避免全局 O(N^2)。

3. topology-preserving DR:
   学局部邻域一致性和拓扑保持指标，转成无监督边分类信号。

4. GGLR / local affine graph regularization:
   学 scale/gradient-preserving 项如何稳定优化。
```

本轮结论：

```text
Physical Edge Reliability 可以继续，
但必须从 edge reweighting 升级为 topology-aware, multi-constraint physical graph repair。
```
