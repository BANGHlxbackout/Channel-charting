# Direct-Support-Aware Physical Edge Filtering 实验总结

日期：2026-07-09

## 1. 验收标准

本轮只验证一个新增模块：

```text
geodesic physical graph W_phys
-> Direct-Support-Aware Physical Edge Filtering
-> W_rel
-> graph-harmonic solver
```

模块公式：

```math
q_{ij}=\left[\log \frac{d_{ij}^{fused}+\epsilon}{d_{ij}^{geo}+\epsilon}\right]_+,
```

```math
R_{ij}=\rho+(1-\rho)\exp\left(-\frac{q_{ij}}{\tau_q+\epsilon}\right),
```

```math
W_{ij}^{rel}=R_{ij}W_{ij}^{phys}.
```

强成功标准：

```text
energy-preserving reliability 仍优于 baseline；
mean 或 p95 改善；
max 不恶化；
CT/TW/KS 不明显下降。
```

失败标准：

```text
raw 和 energy-preserving 都不如 baseline；
或 lower-lambda baseline 能解释 raw 的变化；
或 max 明显恶化。
```

## 2. 图诊断

| item | value |
| --- | ---: |
| edges | 3,088,334 |
| q median | 0.0000 |
| q mean | 0.0404 |
| q p90 / p95 | 0.1363 / 0.2735 |
| R mean / median | 0.9164 / 1.0000 |
| R min / p5 | 0.6000 / 0.6000 |
| sum(W_rel) / sum(W_phys) | 0.8960 |
| degree ratio mean / median | 0.8998 / 0.9105 |
| degree ratio p5 / p95 | 0.7762 / 0.9752 |

关键现象：

```text
tau_q_raw_median = 0。
```

因此第一版实现实际接近“二值式 direct-support 过滤”：`q=0` 的边保持 1，`q>0` 的边大多被压到 `rho=0.6`。总边权下降约 10.4%，不是极端降权，但已经会显著改变平滑强度。

## 3. 结果

baseline：

| method | lambda | CT | TW | KS | mean | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph-harmonic baseline | 0.1500 | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292916 | 0.558674 | 0.634639 | 1.141431 |

本轮实验：

| method | lambda used | CT | TW | KS | mean | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw W_rel | 0.1500 | 0.998180 | 0.998178 | 0.065474 | 0.321490 | 0.295567 | 0.561890 | 0.637436 | 1.140360 |
| energy-preserving W_rel | 0.1500 | 0.998172 | 0.998160 | 0.065743 | 0.322308 | 0.297371 | 0.560944 | 0.636373 | 1.154181 |
| lower-lambda W_phys | 0.1344 | 0.998223 | 0.998232 | 0.064922 | 0.318396 | 0.292455 | 0.559024 | 0.637179 | 1.128972 |

对应结果文件：

```text
results/direct_support_edge_refined_charting/stride1_n20997_direct_support_diagnostic_k128_lambda0.15_rho0.6_seed10/result.json
results/direct_support_edge_refined_charting/stride1_n20997_direct_support_raw_k128_lambda0.15_rho0.6_seed10/result.json
results/direct_support_edge_refined_charting/stride1_n20997_direct_support_energy_preserving_k128_lambda0.15_rho0.6_seed10/result.json
results/direct_support_edge_refined_charting/stride1_n20997_direct_support_lower_lambda_k128_lambda0.15_rho0.6_seed10/result.json
```

## 4. 结论

当前 Direct-Support-Aware Physical Edge Filtering 没有通过验收。

原因：

```text
1. raw W_rel 的 mean、p95、CT/TW/KS 均比baseline 略差；
2. energy-preserving W_rel 继续变差，且 max 从 1.1414 变到 1.1542；
3. lower-lambda 原图控制组反而得到最好的 mean、max、KS。
```

因此本轮不能说明“direct fused support 边过滤”带来了有效结构优化。相反，它说明：

```text
当前 baseline 的 lambda=0.15 可能略偏强；
简单降低原物理图平滑强度已经比 direct-support reweighting 更有效；
本版 R_path 过滤可能误伤了有用的 geodesic 边。
```

## 5. 对后续 idea 的影响

这不是 Physical Edge Reliability 大方向的最终失败，但说明这个第一版模块太粗：

```text
只用 d_fused / d_geo 判断 geodesic diffusion edge 是否可靠是不够的。
```

如果继续做 reliability，下一版不应该继续堆 `R_mut/R_src` 这种静态因子，而应先解决两个问题：

```text
1. 如何区分“有用的多跳 geodesic 边”和“错误 shortcut 边”；
2. 如何让可靠性不退化成全局减小 lambda。
```

当前更直接的实验线索是：

```text
先把baseline 的 lambda sensitivity 跑清楚。
```

但这属于下一步实验，不在本轮模块验收内。
