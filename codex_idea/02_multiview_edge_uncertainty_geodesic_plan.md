# 方向一：多视角边不确定性 + 上游 Geodesic 修正

> 状态：待执行的分阶段实验设计；先过 Gate 0，再按停止条件进入后续 Gate。  
> 本方案只改变原 Stage-1 候选边的行走代价；最终以导师 Stage-2 精修后的 `Z` 判断是否改进。  
> 不同时实现距离尾部 kernel、robust Stage-2 objective、SC²-PPR 或 Siamese 新损失。

## 0. 结论先行

上一轮 Stage-2 共同支持定向降权未超过距离分层随机对照；这否定了当时的选边加降权规则，
不能据此断定失败的唯一根因，也不能推出上游多视角修正有效。本方案检验一个不同、尚未验证的问题：
四阵列分歧能否在原 Stage-1 候选边中识别值得加价的关系，并最终改善 `Z`。

本轮改为：

```text
四个物理分离、朝向不同的天线阵列
 -> 四份 per-array ADP 关系
 -> 同一候选边上的跨阵列 disagreement
 -> 边级风险代理
 -> Shortest Path 之前增加行走代价
 -> MDS 快速验证
 -> 通过后才生成完整 geodesic、重训 Siamese，并用固定精修比较最终 Z
```

核心原则：

1. 不使用 `Y0` 计算可靠性；
2. 不使用 UE 真实位置构图、归一化、选择 `kappa` 或停止迭代；
3. 不删除边、不新增边，保留 baseline fused kNN 的候选结构；
4. 保留现有 `D_fused` 作为中心代价，只增加多视角风险项；
5. 四阵列是相关视角，不假设独立，不把 view variance 除以 `sqrt(4)`；
6. Gate 0 先核验数据与原图复现；Gate 1 再证明 disagreement 在原候选边上有条件增量信息；
7. MDS 未稳定超过强对照时，不重训 Siamese；MDS 和 `Y0` 的收益都不代替最终 `Z` 的收益。

## 1. 研究问题与假设

### 1.1 主问题

对 baseline fused-kNN 中的一条候选边 `(i,j)`，如果四个分布式阵列对其 CSI 相似性判断不一致，
这种分歧能否在不使用真实位置的情况下，预测该边更可能是 geodesic shortcut，并通过增加其行走代价
改善 MDS、`Y0`，尤其是同配置精修后的最终 `Z`？

### 1.2 可证伪假设

**H1：视角互补。** 四份 per-array ADP 不能完全相同；它们应相关但保留明显的 view-specific 信息。

**H2：边级增量。** 在控制 baseline fused distance、邻居 rank、时间/ADP 来源和节点 degree 后，
跨阵列 disagreement 仍能富集物理跨度异常的候选边。

**H3：路径机制。** 风险修正后，原本参与短路的可疑边使用率下降，且测地关系和图谱误差改善，
同时报告真实近邻被误罚及原本良好路径被拉长的损害。单靠加价后的使用率下降不是机制证据。

**H4：特异性。** 真实 disagreement 必须优于同分布的分层 shuffled disagreement 和相同总新增代价
的 uniform-risk control；否则收益不能归因于多视角一致性。

**H5：稳定性。** 预先固定 `kappa=0.5` 为唯一主方法；其他 `kappa` 只作敏感性观察。
主方法须跨 MDS 和训练 seed 稳定，且最终 `Z` 超过原精修图及匹配对照。

## 2. 当前数据中“四视角”的准确含义

当前 `H.npy` 为：

```text
(N, 32, 13) = 20,997 samples × 32 antennas × 13 delay taps
```

`common_dichasus.py` 将其恢复为：

```text
(N, 4, 8, 13)
```

`datasets/specv2.0.json` 记录四个物理阵列 A/B/C/D 位于不同位置并具有不同朝向。当前预处理文件的
存储顺序不是 A/B/C/D，而是：

| stored view | spec array | antenna indices |
|---:|---|---|
| 0 | C | `[0,13,31,29,3,7,1,12]` |
| 1 | B | `[30,26,21,25,24,8,22,15]` |
| 2 | A | `[28,5,10,14,6,2,16,18]` |
| 3 | D | `[19,4,23,17,20,11,9,27]` |

注意：旧预处理生成的 `AP_pos.npy` 是四行全零占位。本方法不使用 AP 坐标，不能误把该文件当成真实
阵列位置。`specv2.0.json` 只用于核验 view 身份，不进入 edge score。

### 2.1 本地预检查证据

使用固定 seed `20261009` 抽取 300,000 个随机样本对，分别计算四个阵列 ADP，仅用 GT 做冻结后的
可行性评价：

- 四阵列 ADP 两两秩相关约为 `0.096--0.298`，不是重复视图；
- 每个阵列 ADP 与物理距离秩相关分别约为 `0.624/0.476/0.383/0.348`；
- 将每个阵列最接近的 2% 作为该 view 的支持：
  - 只有 1 个 view 支持时，物理距离中位数约 `1.883 m`；
  - 2 个 view 支持时约 `0.852 m`；
  - 3 个 view 支持时约 `0.282 m`；
  - 4 个 view 一致支持时约 `0.045 m`。

这不是正式结果，也不能用于选择阈值；它只证明“跨阵列一致性”在该数据上不是空概念。

## 3. 文献给出的约束

### 3.1 ADP/time fused metric

Stephan、Euchner、ten Brink 的 ADP/time 工作使用：

$$
D^{\mathrm{fused}}_{ij}
=\min\left(D^{\mathrm{ADP}}_{ij},\gamma |t_i-t_j|\right),
$$

再用 fused kNN 与 shortest path 构造 geodesic。原论文的 `min` 基于 cf0x 机器人“近似匀速或静止”的
运动假设，`gamma` 从 ADP/time 比值的双峰直方图选取，并明言其他数据集未必满足该假设。
因此它是数据/运动假设依赖的融合规则，不能自动解释为校准后的置信度。

### 3.2 Uncertainty-aware geodesic Channel Charting

Euchner、Stephan、ten Brink 2024 将 dissimilarity 对应的真实距离建模为随机变量，并将路径均值、
方差带入 geodesic loss；论文先从多个距离 realization 取得路径，再在训练 loss 中使用路径分布。
它对不同 metric 及 ADP path edges 使用零协方差近似，对同一路径的 timestamp edges 则按完全相关处理。
本方案的确定性边代价加成不是该论文的公式复现。

本项目不能直接照搬“独立方差相加”：四阵列同时观测同一发射样本，共享同一传播环境的部分系统因素，
同时又有视角特有的遮挡/多径，文献并不支持 IID 假设。本轮只把跨阵列分歧作为 uncertainty/risk proxy，
并采用保守的 additive risk，不声称获得了
校准概率或真实标准差。

### 3.3 Distributed-MIMO Channel Charting

2024--2025 的 D-MIMO Channel Charting 工作以及其 DICHASUS 实验表明，分布式 AP/阵列能提供空间上不同的
receive-power/LoS 约束；其主方法使用 AP receive power、AP 位置和 LoS bounding boxes 构造弱监督 loss。
这些结果支持“物理分离阵列是不同视角”，但没有直接提供“per-array ADP disagreement = edge
uncertainty”的公式。本文件的 disagreement estimator 是新假设，必须通过 Gate 1 独立验证。

2023 的真实 LoRa multipoint CC 将 12 个 gateway 的 RSS 组成联合特征；2024 的 multi-site CC 对齐部分重叠
站点的本地 chart。2026 的 sheaf-theoretic multi-site CC 更在相同 DICHASUS `cf02/cf03/cf04` 上明确将
四阵列当作四个不同基站视角。这些工作加强了“视角真实存在”的证据，但方法是特征融合或潜空间对齐，
仍然没有证明本计划的边级风险公式。

## 4. 数学定义

### 4.1 每阵列 ADP

令 `a in {0,1,2,3}` 表示 stored view。对每个 delay tap `tau`，把阵列内 8 天线向量归一化：

$$
\widehat{\mathbf h}^{(a,\tau)}_i
=\frac{\mathbf h^{(a,\tau)}_i}
{\|\mathbf h^{(a,\tau)}_i\|_2+\epsilon}.
$$

per-array ADP 定义为：

$$
D^{(a)}_{ij}
=13-
\sum_{\tau=1}^{13}
\left|
\left\langle
\widehat{\mathbf h}^{(a,\tau)}_i,
\widehat{\mathbf h}^{(a,\tau)}_j
\right\rangle
\right|^2.
$$

当前 `run_adp_fused.py` 先按 array/tap 分别归一化再求和，因此必须满足回归不变量：

$$
D^{\mathrm{ADP}}_{ij}
=\sum_{a=0}^{3}D^{(a)}_{ij}.
$$

这是实现正确性的硬检查，不是近似关系。

### 4.2 每个 view 的局部尺度

固定：

```text
k_view = 20
```

令 `sigma_i^(a)` 为样本 `i` 在 `D^(a)` 中第 20 个非自身邻居的距离。对 baseline candidate edge：

$$
q^{(a)}_{ij}
=\frac{D^{(a)}_{ij}+\epsilon}
{\sqrt{\sigma_i^{(a)}\sigma_j^{(a)}}+\epsilon},
$$

$$
x^{(a)}_{ij}=\log q^{(a)}_{ij}.
$$

局部尺度消除阵列灵敏度和局部采样密度的一阶影响；log 将乘性尺度差异转为加性差异。

### 4.3 Robust view disagreement

定义四视角中心：

$$
m_{ij}=\operatorname{median}_{a}x^{(a)}_{ij},
$$

以及 robust dispersion：

$$
s_{ij}
=1.4826\operatorname{median}_{a}
\left|x^{(a)}_{ij}-m_{ij}\right|.
$$

`1.4826` 是 Gaussian-consistent MAD 的常用尺度因子；这里只有四个相关视角，因此 `s_ij` 只称为
disagreement scale，不称为已校准标准差。

还必须显式承认四样本 MAD 的离散性：当三个 view 完全相同、一个任意极端时，`s_ij = 0`。
因此主分数检测的是“至少两个 view 形成的分裂”，不是任意 single-view dissent。Gate 1 要另外报告
`3+1` 模式频率，并用 G6 support-count 及 `max(x)-min(x)` 仅诊断对照检查这个盲点；不允许将 MAD 宣称为
一般的“所有跨视角分歧”。

在所有 baseline 无向候选边上冻结：

$$
s_{95}=Q_{0.95}(s_{ij}),
$$

$$
u_{ij}=\min\left(\frac{s_{ij}}{s_{95}+\epsilon},1\right)\in[0,1].
$$

所有归一化量都由 CSI、timestamp 和候选图得到，不读取 `UE_pos`。

### 4.4 Baseline edge source

现有 aggregate ADP 和 timestamp cost 为：

$$
d^{\mathrm{ADP}}_{ij}=\sum_aD^{(a)}_{ij},
\qquad
d^{\mathrm{time}}_{ij}=\gamma|t_i-t_j|,
$$

$$
\mu_{ij}=D^{\mathrm{fused}}_{ij}
=\min(d^{\mathrm{ADP}}_{ij},d^{\mathrm{time}}_{ij}).
$$

记录来源：

$$
g_{ij}=\mathbf 1[d^{\mathrm{ADP}}_{ij}\le d^{\mathrm{time}}_{ij}].
$$

`g=1` 表示该边的 baseline cost 由 ADP 决定。

### 4.5 主方法：ADP-source risk cost

定义风险尺度：

$$
r_{ij}=\mu_{ij}\,u_{ij}\,g_{ij}.
$$

风险修正边代价：

$$
c^{(\kappa)}_{ij}
=\mu_{ij}+\kappa r_{ij}
=\mu_{ij}\left(1+\kappa u_{ij}g_{ij}\right).
$$

正式主方法固定为 `kappa=0.5`；以下相邻值只作敏感性报告，不依据本数据的 GT/MDS 指标换选主值：

```text
kappa = 0（回归原图）, 0.5（主方法）, 0.25/1.0（敏感性）
```

性质：

- `kappa=0` 必须逐元素复现 baseline graph；
- 只增加行走代价，不删除或新增边；
- 四阵列一致时 `u` 低，代价接近 baseline；
- ADP-dominant 且跨阵列分歧高时，Shortest Path 更倾向绕行；
- timestamp-dominant edge 在主方法中保持不变，因为 `u` 不是 timestamp uncertainty。

### 4.6 探索性 all-edge variant

单独报告但不作为主结果：

$$
c^{(\kappa,\mathrm{all})}_{ij}
=\mu_{ij}(1+\kappa u_{ij}).
$$

它检验“数组一致性可否作为 timestamp edge 的第三方 corroboration”。不能在看到 Gate 1 后临时把它
改成主方法。

### 4.7 Path risk 的解释

对路径 `P`：

$$
C_\kappa(P)
=\sum_{e\in P}\mu_e
+\kappa\sum_{e\in P}r_e.
$$

这里使用 `sum(r_e)`，而不是在独立假设下使用 `sqrt(sum(r_e^2))`。它可解释为对相关误差更保守的
additive risk budget。当前实验不声称 `C_kappa` 是某个精确置信水平的 upper bound。

## 5. 固定候选图与避免混杂

候选图必须完全复用 baseline：

```text
数据：datasets/adp_old_shard4/{H.npy,timestamps.npy}
基线：results/adp_old_fused/stride1_n20997_seed10/{adp.npy,fused.npy,geodesic.npy,distance_data.npz,result.json}
保存配置：N=20997、stride=1、neighbors=20、gamma=12.983333587646484
```

实际运行前以本机 `result.json`、数组头和 `distance_data.npz` 的 `indices` 复核这些值与样本顺序；
另一电脑不得仅凭上述文本替代核验。

1. 读取已有 `fused.npy`；
2. 从 baseline `result.json` 读取实际 `neighbors`；
3. 用同一 `NearestNeighbors(metric="precomputed")` 与 `kneighbors_graph(mode="distance")` 重建
   directed kNN；
4. 每个无向端点对只计算一个 `u`、一个 source flag 和一份对照分数；对原图中存在的
   directed entries 施加相同乘因子，保留各 entry 的原始代价，单向存在时不得凭空添加反向边；
5. 继续调用 `shortest_path(..., directed=False)`。

第一轮禁止：

- 使用四阵列 union-kNN 替换 baseline 候选边；
- 删除 disagreement 高的边；
- 修改 `gamma`；
- 修改 baseline neighbor count；
- 同时加入共同邻居、SC²、AP 坐标、LoS 标签或 `Y0`；
- 修改 MDS 或 Siamese loss。

原始 `fused.npy` 在少量互为邻居的端点对上已有浮点级两向差异；Gate 0 实测为 2,759 对，
最大约 `1.14e-5`。不可为得到数值完全相同的双向边而额外对称化原图，
否则 `kappa=0` 无法逐项复现 baseline。干预只共享同一个乘因子。
这样结果只改变同一候选图上的行走风险。

## 6. 强对照

| ID | 方法 | 目的 |
|---|---|---|
| G0 | baseline fused graph | 原始对照 |
| G1 | 主方法 `ADP-source u` | 目标方法 |
| G2 | `all-edge u` | timestamp edge 探索性扩展 |
| G3 | stratified-shuffled `u` | 排除距离、rank、degree、来源混杂 |
| G4 | uniform-risk mass match | 排除仅仅增加总体路径代价 |
| G5 | cross-view pair shuffle 后重算 `u` | 验证 edge-wise view correspondence，而非各 view 边际分布 |
| G6 | support-count score | 检查简单“支持阵列个数”是否已足够 |

G1/G3/G4/G5/G6 的正式路径干预都只作用于同一 ADP-source eligibility。
G3/G5/G6 保留原始 `u_control` 供 Gate 1 诊断；写入图时将控制风险项
`mu*u_control*g` 乘固定常数，使唯一无向 eligible edges 上的风险项总和等于 G1。
不裁剪缩放后的风险项，也不据 GT 调整该常数；否则改善可能只来自新增总代价不同。

### 6.1 Stratified shuffle

对 `u` 在以下联合 strata 内打乱：

```text
source (ADP/time)
× fused-distance decile
× unordered edge 的 rank bin（双向都存在时取较小 rank；同时记录 mutual/one-way）
× 两端 union-degree pair bin
```

只在唯一无向边 ID 上打乱，之后把同一乘因子映射回原 directed structure。Gate 1 固定 200 个 permutation seeds
`10..209` 建立 null distribution。Gate 2 对主 `kappa=0.5` 先用 seed `10` 筛查；初筛通过后，
用 seeds `10..29` 跑 G3/G5 确认。不得把某个控制 seed 或敏感性 `kappa` 的较好结果换成主结果。

### 6.2 Uniform-risk mass match

在主方法 eligibility 中使用常数 `u_bar`，使唯一无向边上的新增总代价完全匹配 G1：

$$
\bar u
=\frac{\sum_e\mu_e u_e g_e}
{\sum_e\mu_e g_e}.
$$

### 6.3 Cross-view pair shuffle

固定 view 0 的唯一无向边 observations；对 view 1--3 分别在与 G3 相同的 strata 内打乱 edge 对应，
再重算 `m/s/u`。它保留每个 view 在分层内的边际分布，但破坏同一 edge 的四阵列对应；
两向映射仍须相同。该对照也会破坏部分节点/时间结构，不能单独证明全部收益仅来自 edge-wise correspondence。

### 6.4 Support-count control

定义：

$$
h^{(a)}_{ij}=\mathbf 1[q^{(a)}_{ij}\le1],
\qquad
n^{\mathrm{support}}_{ij}=\sum_a h^{(a)}_{ij},
$$

$$
u^{\mathrm{support}}_{ij}=1-\frac{n^{\mathrm{support}}_{ij}}4.
$$

G6 与 G1 使用相同 eligibility；将 G6 的 `u` 按新增总代价匹配 G1 后再比较。
如果 G6 等于或优于 G1，记录 MAD 主方案未显示独特优势；本轮不在看 GT 后切换主方法。

## 7. Gate 0：实现回归与无泄漏

必须先通过：

1. 四份 `D^(a)` 在分块检查中对称、对角为 0、有限，范围在理论数值误差允许区间 `[0,13]`；
2. 分块检查 `sum_a D^(a)` 与现有 `adp.npy`：`rtol <= 1e-5, atol <= 1e-5`；
3. stored view 与 A/B/C/D mapping 和 `specv2.0.json` 的 antenna set 一致；
   若原始 TFRecord 可用，还需仅用 CSI/时间字段核对首两条保留样本与 `H.npy`/`timestamps.npy` 的顺序；
4. `u` 对 view permutation 不变；
5. 四视角完全相同时 `u=0`；
6. 三个 view 完全相同、一个极端异常时，按公式明确回归 `MAD=0`，`range` 诊断必须检出该差异，
   support-count 只在异常跨过 `q<=1` 阈值时才要变化；
7. `kappa=0` 的 sparse graph data 与 baseline 完全一致；
8. `kappa=0` 对 `np.linspace(0, N-1, 32, dtype=int)` 的源点重算 Dijkstra，逐行与缓存 baseline
   `geodesic.npy` 在 `rtol=1e-5, atol=1e-4` 内一致，并记录最大绝对/相对差；
   全距离矩阵到 Gate 3 才重算；
9. 风险图边结构不变、连通、无 NaN/Inf；互为邻居的两向边乘因子一致，
   原始两向代价差须单独报告，不能偷偷对称化；
10. score/build 进程不得加载 `UE_pos.npy`，GT evaluation 必须是独立入口；
11. 不使用全零 `AP_pos.npy`；
12. 所有输出记录数组顺序、`k_view`、`s95`、source counts、32 个源点的选择和软件版本。

现有缓存缺失时先按原代码与参数复现原图及所需对照；文件缺失本身不是方法失败，
但未能复现原图就不得进入效果实验。Gate 0 任一适用回归项失败，停止效果实验。

## 8. Gate 1：先证明 edge uncertainty 有信息

### 8.1 冻结的 GT-only 标签

GT 只在所有 edge scores、strata 和 controls 保存后加载。
随机全样本对的四视角相关性预检查不代表 baseline 20-NN 候选边；Gate 1 的总体、分层和停止判断
只使用原候选边。先报告 ADP-source 边数、高 `u` 边数、`3+1` 模式及各组物理跨度，确认待修边并非极少数。

报告两类标签：

1. 固定物理距离阈值：`0.5 m / 1.0 m / 2.0 m`；
2. 在 `source × fused-distance decile × rank bin × endpoint-degree-pair bin` 内，物理端点距离 top 10% 的 edge。

第二类标签用于控制 baseline distance 和 neighbor rank；它仍只是离线 suspect proxy，不是单边因果真值。

### 8.2 必报指标

- 四 view 两两 Pearson/Spearman；
- 每个 view 与物理距离的 GT-only Spearman；
- `u` 与物理距离的总体和分层 Spearman；
- 三个固定阈值的 ROC-AUC、PR-AUC；
- conditional top-10% 标签的 AUC；
- `u` quintile 的 mean/median/P95 physical distance；
- top vs bottom `u` quintile 的 suspect enrichment；
- ADP-source 与 time-source 分开报告；
- mutual/one-way kNN、degree quartile、轨迹边界/内部、local-scale quartile 分组；
- node risk-mass concentration：top 10% nodes 承担比例和 P95/max incident risk；
- 固定源点的基线最短路径中，高 `u`/suspect 边的使用率；此项只衡量干预可达性，
  不把端点物理跨度或使用率当作单边因果真值；
- 200 次按节点重抽样的区间和连续时间块重抽样敏感性；边共享端点、样本又沿轨迹相关，
  这些区间只作稳定性描述，不宣称已获得独立样本意义下的 95% 覆盖率。

### 8.3 Gate 1 通过条件

主方法只在以下条件全部满足时进入 Gate 2：

1. ADP-source conditional AUC point estimate `>= 0.60`，并报告上述重抽样区间；
2. high-`u` quintile 的 suspect rate 至少为 low-`u` quintile 的 `1.5×`；
3. 上述方向在至少 7/10 fused-distance deciles 成立；
4. 真实 `u` 的主 AUC 与富集量均超过 200 次 G3/G5 null distribution 的第 95 百分位；
5. 不得只在单一阵列、单一 degree quartile 或极少数节点区域成立；
6. GT 检查前已冻结全部 score 与 controls。

若 Gate 1 失败：停止，不运行干预后的 shortest path，不重训 Siamese。结论仅限于
“当前四阵列 MAD 分数在原候选边上未满足预设识别门槛”；不临时搜索公式或阈值。

若 ADP-source 通过、time-source 失败：只保留 G1；若两者分别通过，G2 仍只能作为次要候选。

## 9. Gate 2：低成本 Geodesic + MDS 机制验证

### 9.1 只计算固定 target subset 的 shortest path

先固定：

```text
mds_indices = np.arange(0, 20997, 10)  # 2100 samples
MDS seeds = 10, 20, 30
MDS max_iter/n_init 与 baseline 相同
```

对每个 graph 用 `scipy.sparse.csgraph.dijkstra(indices=mds_indices, directed=False)`，只保存
  `2100 × 20997` 距离并提取 `2100 × 2100` 子矩阵。G0 必须在同一子集、同一 MDS seed 下重跑，
  不与历史全样本 MDS 数值直接排名。第一轮不要为每个 `kappa/control` 写出完整
`20997 × 20997` geodesic。

### 9.2 必跑矩阵

```text
G0: baseline
G1: main ADP-source risk, kappa=0.5
G3: stratified-shuffled u, kappa=0.5
G4: uniform-risk mass matched, kappa=0.5
G5: cross-view pair shuffle, kappa=0.5
G6: support-count, same eligibility and added-risk mass, kappa=0.5
G1 sensitivity: kappa=0.25/1.0（只报告，不换选主值）
G2: all-edge variant, kappa=0.5（探索性次要表，不参与主方法选择）
```

若 G1 `kappa=0.5` 通过初筛，再运行 G3/G5 的 20 个预注册 seeds。
G4 是确定性对照，无需重复。控制组的 MDS 使用与真实分数成对的相同 seed；必须保存全部 control 结果。

### 9.3 Path 机制指标

- candidate graph component count；
- geodesic ratio `D_kappa / D_baseline` 的 median/P90/P95/max；
- shortest path hop-count 分布；
- Gate-1 suspect edges 的 path usage / approximate edge betweenness；
- high-`u` 与 low-`u` edge 的 path usage change；
- 原本物理接近、低误差路径被拉长的比例和幅度；报告局部修复与 collateral harm；
- runtime、peak RAM、disk；
- MDS CT/TW/KS 和 affine mean/median/P90/P95/max error。

### 9.4 Gate 2 通过条件

固定主方法 G1 `kappa=0.5` 必须同时满足：

1. 至少 2/3 MDS seeds 的 mean 和 P95 同时优于配对 G0；`0.25/1.0` 仅作为稳定性敏感性报告；
2. 三 seed 的 median P95 相对配对 G0 改善至少 `2%`；
3. 三 seed 的 median mean error 相对配对 G0 改善至少 `1%`；
4. KS 绝对恶化不超过 `0.005`；CT/TW 各自绝对下降不超过 `0.001`；
5. 同一 `kappa=0.5` 下优于 G4；将每个控制 seed 的三次 MDS P95 配对改进率取中位数，
   G1 的对应中位数须超过 G3/G5 各自 20 个控制值中的至少 19 个；
6. P95 geodesic stretch 不超过 `1.35`，median stretch 不超过 `1.15`；
7. 图保持连通，无局部 geodesic 爆炸；
8. suspect-edge path usage 下降方向与 P95 改善一致，且良好路径损害没有抵消目标收益。

如果只改善 mean、不改善 P95/path usage，不支持“修复 shortcut”的机制；停止在 Gate 2。

如果 G6 等于或优于 G1，记录 MAD 主方案没有独特优势并停止本轮主线；不在看 GT 后改选 G6。

`AUC >= 0.60`、`1.5x`、`7/10 deciles`、`P95 >= 2%` 等都是本项目的预注册工程门槛，
不是参考文献推导的统计界或物理界。它们只支持当前 DICHASUS 数据上的 go/no-go；没有第二环境前不得宣称泛化。

## 10. Gate 3：完整 Geodesic、Siamese 与最终精修图 Z

只有 Gate 2 通过才执行。

### 10.1 固定规则

只执行预先固定的 G1 `kappa=0.5`。Gate 2 的 GT 指标仅决定继续或停止，
不能用于换选 `kappa`、MAD 公式或 G2/G6。当前数据上的 Gate 筛选仍是探索性证据，
独立环境验证前不宣称泛化。

### 10.2 完整输出

对 G0、G1、G4 及 Gate 2 中 G3/G5 各自最强的一个控制 seed 计算完整 geodesic。
“最强”按 Gate 2 的三 MDS seed 中位 P95 配对改进率预先排序，同值用 mean 配对改进率排序。
这些是对主方法最不利的对照，不成为替代主方法。输出必须含
`geodesic.npy` 和全样本顺序的 `distance_data.npz`，以兼容 `run_adp_siamese.py --distance-run`。
原 `geodesic_top512.npy` 是 Stage-2 邻居缓存，不是 Stage-1 候选图；风险图的 Stage-2 邻居缓存须另算。

### 10.3 Siamese 配对实验

```text
training seeds = 10, 20, 30
baseline geodesic 与 risk geodesic 使用相同网络、schedule、pair count 和 seed
CT/TW/KS 使用跨方法、跨 seed 固定的评价样本；mean/P95 使用全体 20,997 样本
```

先比较配对训练的原始 Siamese `Y0`，再固定 `graph_k=128`、`lambda=0.15`、200 次迭代，
做 `Y0` 与 `D_geo` 的交叉精修：

| Stage-2 初始图 | 原始 `D_geo` 建图 | 风险 `D_geo` 建图 |
|---|---|---|
| 原始 `Y0` | B00：导师基线 `Z` | B01：只改精修图 |
| 风险 `Y0` | B10：只改训练目标 | B11：完整方法 `Z` |

这四格使用每个 seed 的配对 `Y0`，并在同一评价协议下计算。B11 对 B00 是最终收益；
B10/B01 分开检验收益来自训练目标还是 Stage-2 图。Gate 2 最强 G3/G4/G5 对照也须按相同完整流程
得到最终 `Z`，不能只用它们的 MDS 结果支持 G1 特异性。

Gate 3 最终通过条件：

1. B11 相对配对 B00 在至少 2/3 training seeds 的全样本 mean、P95 同时改善；
2. 三 seed 的配对改进率中位数：P95 至少 `2%`，mean 至少 `1%`；
3. CT/TW/KS 不超过 Gate 2 的恶化边界，且局部修复与 collateral harm 分开报告；
4. B11 相对同 seed、同精修配置的 G3/G4/G5 完整流程对照，在至少 2/3 training seeds 的
   全样本 mean 和 P95 同时更好；
5. B10/B01 只作作用位置归因，不用其中较好的格子事后替换 B11 主结果；
6. 训练 loss 改善不能代替 chart 指标；报告全部 seed，不只报告最好 seed。

## 11. 结论边界

Gate 3 全部通过才称本数据、固定 transductive 协议上的最终 `Z` 有一定改进。
三个训练 seed 只检验训练随机性，不是三个独立环境；没有第二环境时不宣称跨环境或未见 CSI 泛化。
Gate 0/1/2 通过而 Gate 3 失败时，保留其实现或诊断证据，不把 MDS/`Y0` 的收益上升为最终改进。

## 12. 建议的代码结构

`run_adp_fused.py` 可复用 ADP 公式与原图重建，但它只输出 aggregate ADP 和完整全源测地矩阵，
不能直接给出 per-view score 或固定源点的分阶段 Dijkstra。只为这两部分在
`methods/multiview_edge_uncertainty/` 增加最小实现；训练和精修复用现有入口，不覆盖旧 baseline：
`common_dichasus.load_data1` 会同时加载 `UE_pos.npy`，因此 score/build 部分只直接读取
`H.npy` 和 `timestamps.npy`，不能通过该加载器间接读入 GT；评价入口再单独加载 GT。

```text
methods/multiview_edge_uncertainty/gate0.py（本轮实现）
methods/multiview_edge_uncertainty/evaluate.py（仅后续 Gate 需要时实现）
tests/test_multiview_adp_gate0.py
```

### 12.1 深模块接口

```python
def compute_view_scales_and_edge_adp(
    h_by_array: np.ndarray,
    baseline_graph: scipy.sparse.csr_matrix,
    *,
    block_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Blockwise top-20 scales and four ADP values per candidate edge, without GT."""

def build_multiview_edge_risk(
    edge_adp: np.ndarray,
    view_scales: np.ndarray,
    fused: np.ndarray,
    timestamps: np.ndarray,
    gamma: float,
    baseline_graph: scipy.sparse.csr_matrix,
    *,
    k_view: int = 20,
    eps: float = 1e-12,
) -> MultiViewEdgeRisk:
    """Return one frozen score per unordered candidate edge and its controls."""

def apply_risk_cost(
    baseline_graph: scipy.sparse.csr_matrix,
    edge_risk: np.ndarray,
    *,
    kappa: float,
) -> scipy.sparse.csr_matrix:
    """Preserve directed sparse structure; reciprocal entries receive one multiplier."""
```

`MultiViewEdgeRisk` 至少包含：

```text
directed row/col and canonical unordered edge id
per-view D/q/log-q
median log-q
raw MAD disagreement
s95
normalized u
ADP/time source flag
canonical rank bin, mutual/one-way flag
endpoint union-degree pair bin
support count
all frozen control scores
array mapping metadata
```

## 13. 最小单元测试

1. 单 view ADP 小矩阵与 dense reference 一致；
2. 四 view ADP 求和严格复现 aggregate ADP；
3. array permutation 不改变 `u`；
4. 全相同 views 得 `u=0`；
5. 三个完全相同、一个 outlier 时确认 `MAD=0` 且 range 非零；两个阵营分裂时 `u>0`；
6. local scale 排除 self，准确取第 20 邻居；
7. `u` 有限且位于 `[0,1]`；
8. source flag 与 `min(ADP, gamma*time)` 一致；
9. `kappa=0` graph 完全不变；
10. `kappa>0` 只能增加符合 eligibility 的边代价；
11. sparse structure、方向和 edge order 不变；同一无向边的两个 directed entries 若都存在，
    共享乘因子且保留原始代价差；
12. stratified shuffle 保持每个 stratum 的 score multiset；
13. uniform control 的新增总代价与主方法一致；
14. cross-view pair shuffle 保持各 view 的边际值；
15. 固定 32 源点 Dijkstra 与缓存 baseline 测地距离一致；subset Dijkstra 与小图 full shortest path 一致；
16. score/build 测试在 `UE_pos.npy` 不存在时仍通过；
17. metadata 中 stored-view 到 A/B/C/D mapping 正确。

## 14. 输出目录与磁盘预算

```text
results/multiview_edge_uncertainty/
  REPORT.md
  edge_risk/
    edge_observations.npz
    metadata.json
    gate0_regression.json
  gate1/
    edge_diagnostics.json
    edge_bins.csv
    edge_diagnostics.png
  gate2/
    <method>_kappa<value>_seed<seed>/
      subset_geodesic.npy
      embedding.npz
      result.json
      chart.png
  gate3/
    <method>_kappa<value>/
      geodesic.npy
      distance_data.npz
      result.json
    paired_seed<seed>/
      y0_and_z_metrics.json
      b00_b01_b10_b11_metrics.json
```

若保存四份完整 float32 per-array ADP，仅它们就约占 `7.05 GB`，且不是本方案必需产物。
优先逐块更新每 view 的非自身 top-20 尺度，同时只保存原候选边的四份 ADP；
对 aggregate `adp.npy` 作逐块回归。记录峰值 RAM、临时盘占用及实际运行时间。
Gate 2 只保存固定子集 geodesic；完整 `geodesic.npy` 只为 Gate 3 的固定主方法和必要对照生成。

## 15. 执行顺序与输入

```text
方案文件：codex_idea/02_multiview_edge_uncertainty_geodesic_plan.md

先阅读：
1. docs/research/Research_Context.md、docs/research/research_state.md 及其 active task
2. codex_idea/multiview_edge_uncertainty_2023_2026.md
3. 本文件
4. run_adp_fused.py、run_adp_siamese.py、run_diffusion_refined_charting.py
5. common_dichasus.py、preprocess_adp_old_shard.py、datasets/specv2.0.json

不要修改旧 baseline 和 SC2 实验；新增独立模块、入口、评价脚本与测试。
不要硬编码 D:\\diffusion_refined_charting，所有路径用 CLI 参数和 repo-relative default。

严格 Gate 0 -> Gate 1 -> Gate 2 -> Gate 3：
- Gate 0 未通过，停止；
- Gate 1 未证明真实跨阵列 disagreement 有条件增量，停止；
- Gate 2 未稳定改善 MDS mean/P95、控制 collateral harm 并胜过 shuffled/uniform controls，停止；
- 未通过 Gate 2，不生成完整 geodesic，不重训 Siamese；
- Gate 3 包含完整 Siamese 和原 diffusion 的交叉比较；未通过 Gate 3，不宣称最终 Z 改进。

每个 Gate 完成后先写 machine-readable JSON，再写结论。不得在看 GT 后改变公式、k_view、
s95、strata、固定主 kappa 或通过标准。若发现实现问题，只修复 bug 并完整重跑相关 Gate，记录变更。
```

## 16. 参考文献

1. Phillip Stephan, Florian Euchner, Stephan ten Brink, “Angle-Delay Profile-Based and Timestamp-Aided
   Dissimilarity Metrics for Channel Charting,” 2023/IEEE TCOM 2024.  
   <https://arxiv.org/abs/2308.09539>
2. Florian Euchner, Phillip Stephan, Stephan ten Brink, “Uncertainty-Aware Dimensionality Reduction for
   Channel Charting with Geodesic Loss,” Asilomar 2024.  
   <https://arxiv.org/abs/2412.01715>  
   Official code: <https://github.com/Jeija/Geodesic-Uncertainty-Loss-ChannelCharting>
3. Sueda Taner, Victoria Palhares, Christoph Studer, “Channel Charting in Real-World Coordinates with
   Distributed MIMO,” arXiv 2024 / IEEE TWC 2025.  
   <https://arxiv.org/abs/2406.13722>
4. Hossein Aghajari et al., “Multi-Point Channel Charting for Long Range Wireless Networks: an
   Experimental Study,” IEEE MeditCom 2023.  
   <https://doi.org/10.1109/MeditCom58224.2023.10266632>
5. Yamil Vindas, Maxime Guillaud, “Multi-Site Wireless Channel Charting Through Latent Space
   Alignment,” IEEE SPAWC 2024.  
   <https://doi.org/10.1109/SPAWC60668.2024.10694402>
6. Enrico Grimaldi et al., “A Sheaf-Theoretic Framework for Distributed Multi-Site Channel Charting,”
   2026.  
   <https://arxiv.org/abs/2607.03480>  
   Official code: <https://github.com/SPAICOM/sheaf-theoretic-channel-charting>

更完整的 2023--2026 多视角文献核查与“论文公式/本项目推导”的边界见：

```text
codex_idea/multiview_edge_uncertainty_2023_2026.md
```
