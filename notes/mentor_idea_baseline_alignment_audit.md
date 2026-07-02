# 导师 Diffusion-Refined Channel Charting Idea 对齐审计

更新时间：2026-06-25

## 1. 导师 Idea 的准确技术链

导师当前 baseline 不是“拿另一张独立物理图纠正错误 chart”，而是：

```text
ADP/time fused dissimilarity
-> kNN shortest-path geodesic matrix
-> Siamese FCF 用该 geodesic matrix 做 pair-distance regression
-> 得到同一源流形的近似二维坐标 Y0
-> 再由该 geodesic matrix构造 local-scaled graph W
-> 用图 Laplacian / Laplace-Beltrami 近似去除 Y0 在源流形上的高频坐标噪声
```

数学目标为：

```math
\min_Z
\|Z-Y_0\|_F^2
+
\lambda\sum_{ij}W_{ij}\|Z_i-Z_j\|_2^2,
```

其中：

```math
W_{ij}
=
\exp\left(
-\frac{d_{ij}^2}{\sigma_i\sigma_j}
\right).
```

因此 `W` 同时使用：

- geodesic kNN 拓扑；
- geodesic distance 大小；
- 节点局部尺度 `sigma_i`。

## 2. `Y0` 与 `G_phys` 的真实关系

二者不是统计意义上的独立证据。

`run_adp_siamese.py` 中：

```python
distances = np.load(args.distance_run / "geodesic.npy")
loss = mean((||C(H_i)-C(H_j)|| - distances[i,j])^2 / ...)
```

`run_diffusion_refined_charting.py` 中又从同一个 `geodesic.npy` 取近邻并建图。

更准确的表述是：

```text
Y0：神经网络对 ADP/time geodesic source manifold 的参数化坐标近似；
G_phys：同一 source manifold 的非参数显式图近似。
```

refinement 的价值来自用全图结构正则化随机 pair regression 的坐标实现，而不是引入了完全独立的新物理信息。

## 3. 欧氏图消融到底说明什么

欧氏图 refinement 使用 `Y0` 自身的二维欧氏距离建图，保持相同：

- 求解器；
- `k=128`；
- `lambda=0.15`；
- local scaling；
- 迭代次数。

结果：

| 方法 | Mean | P95 | Max | CT/TW |
|---|---:|---:|---:|---:|
| 未精修 FCF | 0.4093 | 0.8558 | 7.8936 | 0.9957 / 0.9960 |
| 欧氏图精修 | 0.3944 | 0.7645 | 8.4133 | 0.9955 / 0.9956 |
| ADP/time 图精修 | 0.3187 | 0.6346 | 1.1414 | 0.9982 / 0.9982 |

可以支持：

```text
从 Y0 自身构图的普通平滑可以降低一般坐标波动；
但重用源 ADP/time 流形构造的算子，在当前数据上能更有效地约束尾部误差和整体结构。
```

不能直接支持：

- 欧氏图必然比未精修差；
- ADP/time 图与 FCF 是完全独立的两套证据；
- 欧氏图中的具体边已经被证明是错误邻居；
- max error 的改善已经证明存在某一种 topology failure；
- 所有物理图与 chart 图的分歧都应被修复。

## 4. 我此前理解正确的部分

- 当前方法是 post-hoc、transductive refinement；
- 普通 Laplacian 形式本身不是主要创新；
- 欧氏图提升并不否定物理图的作用；
- 物理图与欧氏图的真正差异主要体现在 tail error 和结构指标；
- 后续优化不能退化为调 `k`、`lambda` 或重复固定图扩散。

## 5. 我此前跑偏的部分

### 5.1 把同源监督写成独立证据

此前多次使用“独立 radio-geometry evidence”描述 `G_phys`。更严谨的说法应是：

```text
它独立于 Y0 的输出坐标构图，但不独立于 Y0 的训练监督来源。
```

### 5.2 把欧氏图结果解释成已证明错误邻域传播

欧氏图 max 变差与 CT/TW 略降和这一机制相容，但不是直接证明。还可能涉及图尺度、密度、谱性质或极端节点的自图结构。

### 5.3 把物理图与 chart 图分歧直接定义为错误

二者分别使用 source geodesic metric 和二维 embedding Euclidean metric。尤其在非凸流形上，两种距离不能直接等价；分歧可能来自合理降维、度量不匹配或源 metric 误差。

### 5.4 低估当前 `W` 中的距离信息

当前 GLR 不显式保持目标边长，但距离大小已通过 local-scaled heat kernel 进入权重，不能说“只保留邻接、距离基本丢失”。

## 6. 对后续优化 Idea 的约束

与导师 baseline 对齐的优化必须从以下对象之一出发：

1. FCF 对同一 ADP/time source manifold 的坐标近似误差；
2. local-scaled graph 对 Laplace-Beltrami operator 的估计误差；
3. Dirichlet coordinate smoothness 对连续 chart 几何是否合适；
4. 固定同源图 Tikhonov 去噪的统计或部署边界。

如果引入 chart graph，它只能作为：

```text
对当前坐标实现的辅助诊断或自适应信息，
```

不能未经证明就升级为与 source graph 对等的独立真值，更不能把邻域差集直接作为负边。

## 7. 最终结论

导师 idea 的核心应准确压缩为：

> 使用训练 FCF 时所依赖的 ADP/time geodesic source manifold，构造局部尺度 Laplace-Beltrami 图算子，对该流形的神经坐标近似进行自监督图 Tikhonov 去噪。

我此前没有弄错公式和实验主线，但把“同源流形的显式正则”过度解释成“独立物理图对 learned chart 的拓扑审计”。这一偏差已经影响到 PCD-BER 中排斥边的合理性，因此该候选 idea 已撤回，后续需要基于上述准确基线重新生成。
