# 2023–2026 多视角 Channel Charting 与边不确定性研究

> 日期：2026-10-09  
> 目标：为“多视角边不确定性 + Shortest Path 前 geodesic 修正”提供一手文献、公式边界和实验约束。  
> 研究边界：只保留论文原文、作者代码、官方数据/教程。普通 cell-free MIMO 资源分配、泛多模态
> 分类、普通 GNN 不作为方法证据。

## 0. 核心结论

1. 2023–2026 年文献分别研究了 ADP/time dissimilarity、带均值/方差的测地训练损失，及分布式
   AP/阵列的空间信息。这些工作提供动机，未证明本项目的边分数或加价规则有效。
2. 没有找到一手论文已经完成“每阵列 ADP → 跨阵列 disagreement → edge uncertainty → risk-aware
   shortest path”这一完整链条。该 uncertainty estimator 是本项目的新假设，不是已有论文公式。
3. 当前 DICHASUS 数据的四组不是同一阵列任意切片：`specv2.0.json` 记录四个不同位置、不同朝向的
   真实阵列 A–D。2026 年 multi-site CC 论文还在相同 `cf02/cf03/cf04` 轨迹上明确将四阵列当作四个
   不同基站视角。因此可把它们视为多视角；但它们同时观测同一 UE，共享同一环境中的部分系统误差，
   而遮挡/多径又可能是视角特有的；文献并不支持将其误差视为 IID。
4. 2024 uncertainty-aware CC 论文对不同 metric 及 ADP path edges 采用零协方差近似，却把同一路径的
   timestamp edges 按完全相关处理；它建模距离分布并修改训练 loss，本项目的确定性加价不是其推导。
   四阵列也不能当作 IID 观测、把 view variance 除以 `sqrt(4)`。
5. 本项目先在完全相同的 baseline fused-kNN 上检验跨阵列 disagreement，再以固定 `kappa=0.5`
   增加 ADP-source edge traversal cost。subset Dijkstra + MDS 只是低成本筛选；最终必须验证
   Siamese `Y0` 及导师精修后的 `Z`，并将训练目标与精修图的作用分开。
6. timestamp 是异质运动 cue，不是第五个同分布阵列 view。第一版应保持原 `min(ADP, gamma*time)`，
   只对 ADP-dominant edges 应用数组 disagreement；all-edge 版本只能作为次要对照。
7. 文献没有替我们证明 disagreement 是可靠性。必须先通过实现与原图回归 Gate 0，再在原候选边上用
   GT-only 离线诊断和 shuffled/mass-matched controls 检查条件增量与干预收益；只通过 MDS/`Y0`
   不等于最终 `Z` 改进。

## 1. 当前数据是否真的提供四个视角

### 1.1 数据结构

本地：

```text
H.npy: (20997, 32, 13), complex64
common_dichasus.py -> (20997, 4, 8, 13)
```

官方 `datasets/specv2.0.json` 记录四个阵列：

| official array | location | direction |
|---|---|---|
| A | `[2.6747, -13.8973, 1.560075]` | `[-0.45019035, 0.89293261, 0]` |
| B | `[-11.250275, -9.689025, 1.5336]` | `[0.79588938, 0.60544207, 0]` |
| C | `[-1.531375, -15.0595, 1.5772]` | `[0.21111578, 0.97746106, 0]` |
| D | `[-12.684425, -4.483325, 1.78985]` | `[0.99638112, -0.08499802, 0]` |

旧预处理的 stored view 顺序为 `C/B/A/D`。`AP_pos.npy` 是全零占位，不可作为真实 AP 坐标。

### 1.2 可行性预检查

固定 seed `20261009`，随机抽取 300,000 个非同点 pair，分别计算四阵列 ADP。GT 只在所有 pair 和
score 冻结后用于检查：

```text
array rank correlations: 0.096--0.298
rank corr(array ADP, physical distance): 0.624 / 0.476 / 0.383 / 0.348
```

以每个 view 随机 pair ADP 的最低 2% 表示该 view “支持接近”：

| support views | pair count | physical median | physical P90 |
|---:|---:|---:|---:|
| 1 | 18,032 | 1.883 m | 4.363 m |
| 2 | 2,280 | 0.852 m | 2.203 m |
| 3 | 288 | 0.282 m | 0.859 m |
| 4 | 136 | 0.045 m | 0.252 m |

这只是 feasibility evidence：它说明 view agreement 有潜在物理含义，不证明某个阈值或 uncertainty
公式正确。这里的随机 pair 也不能代表 aggregate ADP 选出的 20-NN 边：单个 view 误报很近，
其余 view 报远时，四份 ADP 之和仍可能把该 pair 排除在候选图外。正式实验必须在 baseline
candidate edges 上重新做冻结后的诊断，并量化实际可干预边数。

## 2. 直接相关的一手文献

### 2.1 Stephan, Euchner, ten Brink：ADP + timestamp dissimilarity

- 预印本：[Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting](https://arxiv.org/abs/2308.09539)
- 2023 arXiv，后发表于 IEEE Transactions on Communications 2024。
- 作者教程与实现：[DICHASUS Dissimilarity Metric-Based Channel Charting](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/)

论文/教程给出的主链：

$$
D^{\mathrm{ADP}}
\longrightarrow
D^{\mathrm{fused}}_{ij}
=\min\{D^{\mathrm{ADP}}_{ij},\gamma|t_i-t_j|\}
\longrightarrow
k\mathrm{NN}
\longrightarrow
D^{\mathrm{geo}}.
$$

作者明确讨论两种 cue 的互补：重访旧位置时 timestamp 失败而 CSI 可成功；CSI 突变时 timestamp 可
提供局部连续性。论文中的 `min` 基于 cf0x 机器人“近似匀速或静止”的运动假设，`gamma` 从
ADP/time 比值的双峰直方图选取，并明言其他数据集未必满足该假设。因此它是数据/运动假设依赖的
融合规则，不是经校准的置信度或不确定性。

**对本项目的约束：**

- 保留 paper-native fused graph 是合理强 baseline；
- timestamp 不应被当成与四阵列同分布的第五个 view；
- 第一次实验不要同时改 `gamma`、candidate kNN 和 uncertainty；
- per-array ADP 必须求和复现当前 aggregate ADP，保证只拆视角、不换 metric。

### 2.2 Euchner, Stephan, ten Brink：uncertainty-aware geodesic loss

- 论文：[Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss](https://arxiv.org/abs/2412.01715)
- Asilomar 2024。
- 官方代码：[Geodesic-Uncertainty-Loss-ChannelCharting](https://github.com/Jeija/Geodesic-Uncertainty-Loss-ChannelCharting)

论文把 dissimilarity 对应的真实距离建模为随机变量；路径由多个距离 realization 产生，
路径均值、方差进入训练 loss，而非直接用不确定性上调固定图的边代价。论文 Eq. (4) 是

$$
\mathcal L_{\mathrm{geo,unc}}
=\sum_{(i,j)\in\mathcal B}
\frac{|\rho_{\mathrm{geo}}^{(i,j)}-\mu_{\mathrm{geo}}^{(i,j)}|^2}
{2(\sigma_{\mathrm{geo}}^{(i,j)})^2}.
$$

官方代码的 `GaussianDissimilarityMetric` 约定每个 metric 返回 path mean/variance；具体 Keras 实现用的是
数值等价的 batch mean 版本（去掉常数 `1/2`，并加 `1e-6`）：

$$
\mathcal L_{\mathrm{geo}}
=\operatorname{mean}
\frac{(d_{\mathrm{chart,path}}-\mu_{\mathrm{path}})^2}
{\sigma^2_{\mathrm{path}}+\epsilon}.
$$

官方代码的 `GaussianDissimilarityModel`：

- 为多个 dissimilarity realizations 生成 shortest paths；
- 对每条 path 累加各 metric 的 mean/variance；
- 明确注释“不同 dissimilarity models 完全不相关”；
- `ADPDissimilarityMetric.mean_variance_along_path` 直接累加各 edge variance，并注释假设 ADP
  observations 完全不相关。

论文同时把同一路径的 timestamp edge 误差按完全相关近似处理；它还指出完整四阵列数据上，
uncertainty-aware loss 与不加该项的 geodesic+acceleration loss 表现接近。因此“有 uncertainty 模型”
本身不保证完整多阵列图谱改善。

代码仓库同时提供 all-arrays 和 one-array notebooks；论文在同一 DICHASUS cf0x 数据上展示了单阵列 chart。
这证明单阵列包含可用信息，但论文没有做“跨阵列 disagreement 能校准边错误”的实验，也不能由
all-array 与 single-array 结果直接推出视角误差独立或 disagreement 可作为方差。

**不能直接照搬的内容：**

- 四个阵列共享 UE 时刻和同一传播环境的部分系统因素，同时又有视角特有的遮挡/多径，文献不支持将它们视为四个 IID 样本；
- 官方代码是 partial source，ADP realization 处还用 `zeros_like(sqrt(variance*scale))` 作为抽样标准差，
  不能把仓库每个细节视为论文公式的完整复现；
- 原论文的 ADP-to-mean/variance mapping 由外部 callback 提供，不等于从四阵列 disagreement 自动得到
  校准物理方差。

**本项目采用的迁移：**借用“边关系可能不可靠、路径需要处理不可靠关系”的问题动机；
用 robust disagreement 与 additive traversal cost 作待验证的确定性规则，不称其为概率方差、
校准置信上界或上述论文 loss 的复现。

### 2.3 Taner, Palhares, Studer：Distributed-MIMO CC

- 论文：[Channel Charting in Real-World Coordinates with Distributed MIMO](https://arxiv.org/abs/2406.13722)
- 2024 arXiv；IEEE Transactions on Wireless Communications 2025，DOI `10.1109/TWC.2025.3559723`。

论文系统模型包含多个分布式多天线 AP。方法使用：

- AP-wise received power 比较构造 bilateration loss；
- receive-power threshold 估计 LoS AP；
- AP 位置和 LoS bounding boxes 提供弱监督坐标约束；
- 与 timestamp triplet loss 组合；
- 在模拟场景和 DICHASUS measured indoor scenario 上评价。

论文的 measured DICHASUS 特征同样是 `32 antennas × 13 taps`。它证明不同物理 AP/阵列能提供空间上不同的
receive-power/LoS 约束，且 AP 可见性随 UE 位置/LoS 状态变化；它未证明 per-array ADP 误差互相独立。

**对本项目的启发：**

- 四阵列不是等质量传感器，未来可用 per-array receive power/visibility 形成 availability weight；
- 当前第一版不使用 AP 位置或 LoS bounding boxes，以保持纯自监督并隔离 disagreement；
- 如果第一版通过，power-gated view availability 是下一层消融，不应现在混入。

### 2.4 Aghajari et al.：12 网关 LoRa multipoint CC

- [Multi-Point Channel Charting for Long Range Wireless Networks: an Experimental Study](https://doi.org/10.1109/MeditCom58224.2023.10266632)
- IEEE MeditCom 2023；作者机构存档提供 accepted manuscript 与完整摘要。

该工作在真实 LoRa 多网关网络中把 12 个 gateway 的接收功率组成联合特征，用 t-SNE 和 k-means 生成/分析
radio-geometry chart，并报告增加 gateway 数量能改善聚类。它是 2023--2026 范围内直接的 multipoint 实测证据，
但用的是联合 RSS 特征，没有 per-view ADP、edge uncertainty 或 risk-aware shortest path。

### 2.5 Vindas, Guillaud：Multi-site latent-space alignment

- [Multi-Site Wireless Channel Charting Through Latent Space Alignment](https://doi.org/10.1109/SPAWC60668.2024.10694402)
- IEEE SPAWC 2024，HAL `hal-04685466`。

该工作区分了两个问题：早期 multipoint CC 在同一环境中融合多 AP 特征；multi-site CC 则让覆盖区仅部分重叠的
多个基站分别学习 chart，再将潜空间对齐到共同坐标系。它是多视角融合的直接近邻工作，但处理的是 chart-level
alignment，不是样本对 edge-level disagreement。

### 2.6 Grimaldi et al.：Sheaf-theoretic distributed multi-site CC

- 论文：[A Sheaf-Theoretic Framework for Distributed Multi-Site Channel Charting](https://arxiv.org/abs/2607.03480)
- 作者代码：[SPAICOM/sheaf-theoretic-channel-charting](https://github.com/SPAICOM/sheaf-theoretic-channel-charting)
- arXiv 2026；作者页标注为 IEEE SPAWC 2026。

该文在 DICHASUS `cf02/cf03/cf04` 上把四个阵列明确视为四个不同基站，称其从不同空间视角观测同一传播环境。
方法为每站训练本地 encoder，在重叠覆盖上用正交 transport map/sheaf gluing 约束对齐 embedding，并与 2024
optimal-transport alignment 比较。这是“四阵列可作为四物理视角”的最直接文献支持；但它使用已知同时/同位置对应来
对齐潜表示，没有定义 ADP edge uncertainty，也没有修正 shortest-path cost。

### 2.7 Stephan et al.：真实 D-MIMO 上的 Channel Charting 应用

- [Channel Charting-Based Channel Prediction on Real-World Distributed Massive MIMO CSI](https://arxiv.org/abs/2410.11486)
- 2024。

该工作验证 Channel Charting 在真实 distributed massive MIMO 和用户运动上的 downstream utility。它不
提出多视角 edge uncertainty，但支持两个边界：chart 必须保持可信的空间/时间关系；只改善 transductive
affine error 不等于 downstream 或 OOS 性能一定改善。

### 2.8 2024 LoS channel-distance ambiguity 分析

- [Optimizing Multicarrier Multiantenna Systems for LoS Channel Charting](https://ieeexplore.ieee.org/document/10577595/)
- IEEE Transactions on Wireless Communications 2024，DOI `10.1109/TWC.2024.3417892`。

该论文分析 phase-insensitive channel distance 的周期/振荡歧义：物理远 UE 可能在 channel distance 上
显得接近，并提出消除歧义的新距离与系统设计原则。

**对本项目的启发：**单一阵列的 phase-insensitive distance 产生 false-near pair 有明确无线物理来源；
多个空间分离阵列的错误模式可能不同，因此跨阵列 corroboration 有类比动机。但该文分析的不是本项目的 ADP
metric，所以它不是“ADP disagreement 能检测 false edge”的直接证据。

### 2.9 2024 velocity/map cue 工作

- [Velocity-Based Channel Charting With Spatial Distribution Map Matching](https://ieeexplore.ieee.org/document/10591331/)
- IEEE JISPIN 2024，DOI `10.1109/JISPIN.2024.3424768`。

该工作使用 noisy velocity 和粗地图约束 Channel Charting，说明 CSI 之外的运动 cue 可以补充 chart
geometry。它不提供 per-array uncertainty；对本项目主要用于强调 timestamp/velocity cue 与 CSI views
的统计语义不同，必须分开建模。

## 3. 文献没有解决的空白

本次限定检索没有发现 2023–2026 一手工作同时满足：

1. 对同一 distributed-MIMO sample 分别计算各阵列 dissimilarity；
2. 用跨阵列一致/分歧估计每条候选边的 uncertainty；
3. 不使用 UE GT 校准该 uncertainty；
4. 将 uncertainty 直接用于 shortest-path 前的 edge cost；
5. 使用 shuffled/mass-matched controls 证明收益来自 view correspondence。

这是限定检索的结果，不等于排除所有相关先例；若形成方法结果，投稿前仍须核对更近的
edge reliability / risk-aware graph 工作。上述 multipoint/multi-site 工作覆盖了联合特征、
chart alignment 和跨站一致性，但未覆盖上述五步链条。
所以本项目的潜在贡献不是“首次多视角 Channel Charting”，而是：

> 从同一 D-MIMO CSI 的物理分离阵列中，构造无需位置标签的 edge disagreement，并验证它能否成为
> geodesic graph 的风险代理。

这也是风险所在：没有现有定理保证 disagreement 必然对应 false edge，必须先过 edge diagnostic gate。

## 4. 论文公式与本项目推导的边界

| 内容 | 来源 |
|---|---|
| ADP inner-product dissimilarity | Stephan et al. 2023/2024 |
| `min(ADP, gamma*time)` fusion | Stephan et al. 2023/2024 |
| dissimilarity 作为带 mean/variance 随机变量 | Euchner et al. 2024 |
| path mean/variance 进入 geodesic loss | Euchner et al. 2024 |
| 分布式 AP 提供互补 power/LoS 信息 | Taner et al. 2024/2025 |
| 单视角 channel distance 可有 false-near ambiguity | LoS distance paper 2024 |
| per-array ADP 精确分解与求和回归 | 当前代码的代数性质 |
| view-local scaling + log-MAD disagreement | 本项目提出 |
| `risk = fused_cost × disagreement` | 本项目提出 |
| `cost = mean + kappa × risk` | 本项目提出的 risk-sensitive surrogate |
| 相关视角下使用 `sum risk` 而非独立方差相加 | 本项目的保守设计 |

上述 `mean` 仅指原 fused edge cost，不能解释为已校准物理距离的条件期望。
`sum risk` 是确定性路径代价设计，不是相关误差下的概率上界。

## 5. 推荐的理论形式

对阵列 `a`：

$$
D^{(a)}_{ij}
=13-\sum_{\tau=1}^{13}
|\langle\hat h_i^{(a,\tau)},\hat h_j^{(a,\tau)}\rangle|^2,
\qquad
D^{\mathrm{ADP}}=\sum_aD^{(a)}.
$$

用 view-local 第 20 邻居尺度：

$$
x^{(a)}_{ij}
=\log\frac{D^{(a)}_{ij}+\epsilon}
{\sqrt{\sigma_i^{(a)}\sigma_j^{(a)}}+\epsilon}.
$$

robust disagreement：

$$
m_{ij}=\operatorname{median}_a x^{(a)}_{ij},
$$

$$
s_{ij}=1.4826\operatorname{median}_a|x^{(a)}_{ij}-m_{ij}|,
$$

$$
u_{ij}=\min\left(s_{ij}/Q_{0.95}(s),1\right).
$$

**四视角 MAD 的离散边界：**当三个 $x^{(a)}$ 完全相同、一个任意极端时，上式的 MAD 等于 0。因此
它实际检测的是“至少两个视角形成的分裂”，而不是“任意单视角与其他视角不同”。这个性质不是文献定理，
也不能用“robust”一词掩盖。所以实验必须同时报告 single-view dissent 的频率，并保留 support-count/range 类对照；
如果错误边主要是 `3+1` 模式，MAD 作为主分数应被证伪。

保留 baseline fused cost：

$$
\mu_{ij}=\min(D^{\mathrm{ADP}}_{ij},\gamma|t_i-t_j|).
$$

主方法只对 ADP-dominant edge：

$$
g_{ij}=\mathbf1[D^{\mathrm{ADP}}_{ij}\le\gamma|t_i-t_j|],
$$

$$
c^{(\kappa)}_{ij}
=\mu_{ij}\left(1+\kappa u_{ij}g_{ij}\right).
$$

路径代价：

$$
C_\kappa(P)
=\sum_{e\in P}\mu_e+\kappa\sum_{e\in P}\mu_eu_eg_e.
$$

这里 `u` 是 uncertainty proxy，不是 calibration 后的概率或标准差；`kappa` 是 risk aversion，不是
Gaussian confidence multiplier。

## 6. 为什么第一版不用 view weighting

四阵列的单 view 质量明显不同；理论上可以学习权重。但无 GT 的权重估计容易出现两个问题：

1. 与其他 views 不同的阵列可能是坏传感器，也可能是唯一看到有效路径的互补阵列；
2. 用全局 view agreement 给权重会压制局部有用但全局不同的视角。

第一版采用 equal-view median/MAD，具 permutation invariance，并通过忽略单一 outlier view 获得稳健性；这也正是前述
`3+1` 盲点。只有当 Gate 1
证明 disagreement 有用后，才研究基于 receive power、visibility 或 leave-one-view agreement 的权重。

## 7. 必需对照与停止规则

### 强对照

- original fused geodesic；
- stratified-shuffled disagreement；
- uniform added-risk mass matching；
- cross-view pair shuffle；
- simple support-count score；
- ADP-source-only 与 all-edge 分开；
- single-array diagnostic，不把最好单阵列隐藏在 aggregate 结果里。

所有干预对照必须在同一原始候选图上运行；同一无向端点对只生成一个分数，
映射回原 directed kNN 时互为邻居的两向使用相同乘因子，保留原始微小代价差；
不得额外对称化 baseline。shuffled 与 support-count 路径干预对照
须匹配主方法的 eligibility 与新增总代价。`kappa=0.5` 是预先固定的主方法，
`0.25/1.0` 仅作敏感性，不能看 GT 后换选。

`stratified shuffle` 和 `cross-view pair shuffle` 不能各只跑一个随机 seed。Gate 1 应用至少 200 次廉价
permutation 得到 null distribution；Gate 2 对固定主 `kappa=0.5` 追加 20 个预注册 control seeds，
否则“优于 shuffle”只是一次随机抽样的陈述。

### 先验失败条件

1. per-array ADP 求和不能复现 aggregate ADP；
2. disagreement 在控制 fused distance/rank/degree/source 后 AUC 不超过预注册阈值；
3. 只在单一阵列或高 degree 区成立；
4. 不能优于 view-shuffled score；
5. MDS 只改善 mean 而 P95 不改善，或可疑边使用率下降但良好路径损害抵消收益；
6. 固定主 `kappa=0.5` 只在单一 MDS seed 有效；
7. 与 uniform/shuffled risk 无差异；
8. geodesic stretch、断连或局部路径爆炸；
9. 最终完整方法的 `Z` 未稳定超过配对原始 `Z` 及最强匹配对照。

上述 `AUC >= 0.60`、`1.5x`、`7/10 deciles`、`P95 >= 2%` 等都是本项目的预注册工程门槛，
不是上述论文推导出的统计或物理界。它们只能决定“是否值得在当前数据上继续”，不能单独支撑泛化性主张。
边级 CI 还必须处理一条边共享两个端点的 dyadic dependence；仅按任意“source endpoint”做单向 cluster bootstrap
不足以给出有效覆盖率。

完整数值门槛、代码接口、单元测试与分阶段执行顺序见：

[02_multiview_edge_uncertainty_geodesic_plan.md](02_multiview_edge_uncertainty_geodesic_plan.md)

## 8. 参考文献与一手资源

1. Stephan, Euchner, ten Brink, *Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for
   Channel Charting*, 2023/2024. <https://arxiv.org/abs/2308.09539>
2. DICHASUS official tutorial, *Dissimilarity Metric-Based Channel Charting*.
   <https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/>
3. Euchner, Stephan, ten Brink, *Uncertainty-Aware Dimensionality Reduction for Channel Charting with
   Geodesic Loss*, Asilomar 2024. <https://arxiv.org/abs/2412.01715>
4. Official uncertainty-aware CC code.
   <https://github.com/Jeija/Geodesic-Uncertainty-Loss-ChannelCharting>
5. Taner, Palhares, Studer, *Channel Charting in Real-World Coordinates with Distributed MIMO*,
   2024/IEEE TWC 2025. <https://arxiv.org/abs/2406.13722>
6. Aghajari et al., *Multi-Point Channel Charting for Long Range Wireless Networks: an Experimental
   Study*, IEEE MeditCom 2023. <https://doi.org/10.1109/MeditCom58224.2023.10266632>
7. Vindas and Guillaud, *Multi-Site Wireless Channel Charting Through Latent Space Alignment*, IEEE
   SPAWC 2024. <https://doi.org/10.1109/SPAWC60668.2024.10694402>
8. Grimaldi et al., *A Sheaf-Theoretic Framework for Distributed Multi-Site Channel Charting*, 2026.
   <https://arxiv.org/abs/2607.03480>; official code:
   <https://github.com/SPAICOM/sheaf-theoretic-channel-charting>
9. Stephan, Euchner, ten Brink, *Channel Charting-Based Channel Prediction on Real-World Distributed
   Massive MIMO CSI*, 2024. <https://arxiv.org/abs/2410.11486>
10. *Optimizing Multicarrier Multiantenna Systems for LoS Channel Charting*, IEEE TWC 2024.
   <https://doi.org/10.1109/TWC.2024.3417892>
11. *Velocity-Based Channel Charting With Spatial Distribution Map Matching*, IEEE JISPIN 2024.
   <https://doi.org/10.1109/JISPIN.2024.3424768>
