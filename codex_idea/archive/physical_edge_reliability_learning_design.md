# Physical Edge Reliability：可行性验证方案

更新时间：2026-07-09

## 0. 当前决策

这个文件只保留当前要验证的清晰版本，不再保留旧的三因子乘法设计。

当前优先验证的模块是：

```text
Direct-Support-Aware Physical Edge Filtering
直接物理支持感知的边可靠性过滤
```

它不是完整论文最终 idea，只是验证 Physical Edge Reliability 方向是否有实验价值的第一步。

暂时不验证：

```text
R_mut: mutual kNN 因子，只能后续消融
R_src: ADP/time source agreement，第一版排除
Uncertainty-propagated path variance: 更贴近 Uncertainty-Aware Geodesic Loss，但实现成本更高，作为第二阶段
```

当前第一版只验证一个问题：

```text
导师 baseline 中的 geodesic top-k diffusion edge，是否也得到 direct fused distance 支持？
```

## 1. 为什么先验证这个模块

导师 baseline 的流程是：

```text
ADP/time fused direct distance
-> fused kNN graph
-> all-pairs geodesic distance
-> geodesic top-k graph
-> local-scaled graph-harmonic diffusion
```

这里存在一个明确风险：

```text
d_geo(i,j) 小，但 d_fused(i,j) 不小。
```

这说明 `i,j` 可能只是沿 shortest path 近，而不是 direct fused metric 下的直接近邻。作为 geodesic 距离估计，这没有问题；但作为 graph-harmonic solver 的一步扩散边，可能会把多跳关系当成一步平滑约束。

所以第一版不泛泛做“可靠性”，只做一个可检验假设：

```text
如果 geodesic diffusion edge 缺少 direct fused support，就降低它的扩散强度。
```

## 2. 第一版公式

先按导师 baseline 构造物理图：

```math
W_{ij}^{phys}
=
\exp
\left(
-
\left(
\frac{d_{ij}^{geo}}
{\sqrt{\sigma_i\sigma_j}}
\right)^2
\right).
```

对每条 baseline 图边 `(i,j)`，计算：

```math
q_{ij}
=
\left[
\log
\frac{d_{ij}^{fused}+\epsilon}
{d_{ij}^{geo}+\epsilon}
\right]_+.
```

其中：

```text
q_ij 小：direct fused distance 支持这条 geodesic edge
q_ij 大：这条边更像 path-mediated geodesic relation
```

用边集合上的中位数归一化：

```math
\tau_q=\operatorname{median}_{(a,b)\in E}q_{ab}.
```

可靠性：

```math
R_{ij}^{path}
=
\rho
+
(1-\rho)
\exp
\left(
-
\frac{q_{ij}}
{\tau_q+\epsilon}
\right).
```

建议第一版固定：

```text
rho = 0.6
```

然后：

```math
W_{ij}^{rel}=R_{ij}^{path}W_{ij}^{phys}.
```

优化目标：

```math
\min_Z
\|Z-Y_0\|_F^2
+
\lambda
\sum_{(i,j)}
W_{ij}^{rel}
\|Z_i-Z_j\|_2^2.
```

## 3. 必须做的去混淆版本

因为 `R_ij <= 1`，直接使用 `W_rel=R W_phys` 会让总边权下降，容易和“等效减小 lambda”混淆。

所以必须同时构造 energy-preserving 版本：

```math
\widehat W_{ij}^{rel}
=
cR_{ij}^{path}W_{ij}^{phys},
```

```math
c=
\frac{\sum_{(i,j)}W_{ij}^{phys}}
{\sum_{(i,j)}R_{ij}^{path}W_{ij}^{phys}+\epsilon}.
```

解释标准：

```text
raw W_rel 变好，但 energy-preserving W_rel 不变好：
    不能说明可靠性过滤有效，可能只是平滑变弱。

energy-preserving W_rel 仍变好：
    才能说明边强度重分配本身有价值。
```

还必须跑 lower-lambda baseline：

```text
用原 W_phys，不加 R_path，只降低 lambda。
```

如果 lower-lambda baseline 复现了提升，说明不是模块贡献。

## 4. 可行性验证步骤

### Step 0：不跑 solver 的图诊断

先只计算图统计：

```text
num_edges
mean/median/p90/p95/max of q_ij
mean/median/min/p5/p95 of R_path
sum(W_rel)/sum(W_phys)
deg_rel/deg_phys 的 mean/median/p5/p95
```

判断：

```text
如果 R_path 几乎全是 1：
    模块没有实际作用，不值得跑。

如果大量边接近 rho：
    gate 过强，先调高 rho 或停止。

如果少量边被温和降权：
    可以进入 solver 验证。
```

### Step 1：跑 raw reliability refinement

使用：

```text
W_rel = R_path * W_phys
lambda = 0.15
graph_k = 128
iterations = 200
```

记录：

```text
mean / median / p90 / p95 / max
CT / TW / KS
graph statistics
```

### Step 2：跑 energy-preserving reliability refinement

使用：

```text
W_rel_hat = c * R_path * W_phys
lambda = 0.15
```

这是核心判据。它用于确认收益是否来自边重分配，而不是总平滑减弱。

### Step 3：跑 lower-lambda baseline

用原始 `W_phys`，不加可靠性：

```text
lambda_eff = 0.15 * sum(W_rel) / sum(W_phys)
```

如果 raw reliability 的提升和这个 lower-lambda baseline 差不多，则模块不成立。

### Step 4：只在 Step 1-3 有正信号时做消融

后续才考虑：

```text
mutual-only
path + mutual
```

但这不是第一轮主验证。

## 5. 成功和失败判据

当前强 baseline：

```text
mean = 0.3187
p95 = 0.6346
max = 1.1414
CT/TW = 0.9982 / 0.9982
KS = 0.0651
```

强成功：

```text
energy-preserving reliability 仍优于 baseline；
mean 或 p95 改善；
max 不恶化；
CT/TW/KS 不明显下降。
```

弱成功：

```text
raw reliability 有改善；
energy-preserving 基本持平；
lower-lambda baseline 不能完全复现。
```

失败：

```text
raw reliability、energy-preserving reliability 都不如 baseline；
或 lower-lambda baseline 能解释所有改善；
或 max 明显恶化，说明可靠性过滤破坏了 tail 修复路径。
```

## 6. 数据泄露边界

允许使用：

```text
fused.npy
geodesic.npy
geodesic_top512.npy
init embedding Y0 只进入 solver，不进入 R_path
```

不允许使用：

```text
UE position
localization error
post-hoc affine error
按真实结果选择边或调 R_path 参数
```

第一版 `R_path` 不使用：

```text
Y0 Euclidean distance
Y0 residual
ADP/source agreement
mutual rank
```

## 7. 和 Uncertainty-Aware Geodesic Loss 的关系

`Uncertainty-Aware Dimensionality Reduction for Channel Charting with Geodesic Loss` 没有提出 `R_path`。

这篇论文真正的方法机制是：

```text
记住 geodesic path q(i,j)
把 d | Delta 建模成随机变量
沿 path 传播均值和方差
用 1 / sigma_geo^2 加权 geodesic loss
```

因此它直接启发的模块应是第二阶段：

```text
Uncertainty-Propagated Geodesic Edge Reliability
```

该第二阶段需要重建 fused kNN graph 的 predecessor matrix：

```text
shortest_path(graph, directed=False, return_predecessors=True)
```

然后对每条 geodesic diffusion edge 恢复 path，并沿 path 传播不确定性。

这比第一版 `R_path` 更贴近论文，但成本更高，且需要设计无监督的 `sigma_m^2` 模型。因此不作为第一轮可行性验证。

## 8. 现在到底怎么做

当前最小可行验证只做：

```text
Direct-Support-Aware Physical Edge Filtering
```

实现范围：

```text
1. 复用 baseline 的 graph construction 和 solver；
2. 新增 R_path 计算；
3. 跑 raw W_rel；
4. 跑 energy-preserving W_rel；
5. 跑 lower-lambda baseline；
6. 输出图统计和结果表。
```

如果这一步失败：

```text
Physical Edge Reliability 方向至少在这个 direct-support 版本上不成立；
不要继续堆 R_mut/R_src。
```

如果这一步成立：

```text
再考虑第二阶段的 path uncertainty propagation，
或考虑 mutual rank 作为辅助消融。
```

## 9. 2026-07-09 实验结果记录

实验已经完成，结果文件：

```text
results/direct_support_edge_refined_charting/summary.md
```

结论：

```text
当前 Direct-Support-Aware Physical Edge Filtering 没有通过验收。
```

核心证据：

| method | lambda used | mean | p95 | max | KS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Graph-harmonic baseline | 0.1500 | 0.318748 | 0.634639 | 1.141431 | 0.065065 |
| raw W_rel | 0.1500 | 0.321490 | 0.637436 | 1.140360 | 0.065474 |
| energy-preserving W_rel | 0.1500 | 0.322308 | 0.636373 | 1.154181 | 0.065743 |
| lower-lambda W_phys | 0.1344 | 0.318396 | 0.637179 | 1.128972 | 0.064922 |

解释：

```text
raw W_rel 和 energy-preserving W_rel 都没有超过导师 baseline；
energy-preserving 变差，说明边重分配本身没有带来收益；
lower-lambda 原图控制组反而更好，说明当前收益更像 lambda sensitivity，而不是 reliability module。
```

后续约束：

```text
不要继续在这个 direct-support 版本上堆 R_mut/R_src。
如果继续 Physical Edge Reliability，需要先解决：
1. 如何区分有用的多跳 geodesic 边和错误 shortcut 边；
2. 如何避免可靠性退化成全局减小 lambda。
```
