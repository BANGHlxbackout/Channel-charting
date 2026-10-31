# Multi-View Reliable Edge & Huber Graph-TV Refinement

设计日期: 2026-07-09

---

## 0. 动机

当前最优 baseline 是 graph-harmonic diffusion refinement:

```
(I + λ L) Z = Y₀
```

其中 `W_phys` 来自 ADP/time geodesic kNN 图，`L` 是 L2 图 Laplacian。

已有两个变体未超越 baseline：

| 变体 | 机制 | 失败原因 |
|---|---|---|
| Smoothness-audited alpha-only | `r_i`高→`α_i`低→更强扩散 | `r_i` 与真实误差弱相关(Spearman≈0.21)；高 residual 可能来自错边/几何弯曲，强扩散反而放大错误 |
| Direct-support edge filtering | `d_fused/d_geo` 偏离→边降权 | 单证据边过滤 + 仍用 L2 Laplacian，两项问题各只解决一半 |

这两个失败暴露 baseline 两个未解决的问题：

1. **图边不可靠问题 (graph error)** — `W_phys` 不是确定真值，可能含有 shortcut、时间误连、局部密度导致的伪近邻
2. **罚项形式问题 (penalty error)** — L2 Laplacian 对边差异做二次惩罚，差异越大拉力越强，导致过平滑

---

## 1. 核心主张

1. **物理图边应由多证据交叉验证**（ADP 邻域一致性、时间一致性、直接融合距离一致性），而不是由 chart smoothness residual 判定——避免 alpha-only 的循环论证
2. **L2 Laplacian 应替换为 edge-preserving 罚项**（Huber/Charbonnier）——大梯度边（来自真实几何弯曲或残留 shortcut）不应被二次惩罚强拉
3. **这两项修改互补**——同时处理 graph error 和 penalty error，覆盖 alpha-only 和 direct-support 各自只部分解决的两种失效模式

---

## 2. 模块架构

```
ADP distance ─┐
timestamp dist─┤──> Multi-View Edge Reliability ──> W_reliable (sparse)
geodesic dist ─┘
                                               │
Y0 ─────────────────────> Huber Graph-TV Solver ──> Z (refined chart)
                     ↑         (IRLS outer loop)
                     └── W_reliable ──┘
```

---

## 3. Multi-View Edge Reliability（多证据边可靠性）

对物理图中每条 geodesic kNN 边 `(i,j)`，仅从原始物理测量（ADP、timestamp、fused distance）计算三个证据分数，**不使用 Y0 或 UE position**。

### 证据 A: ADP 邻域一致性

```
rank_adp(i,j) = j 在 i 的直接 ADP 距离邻域中的排名
r_adp = min(1, rank_adp(i,j) / k)        ∈ (0, 1]
```

如果 `(i,j)` 是 geodesic 近邻但在 direct ADP 下距离很远，说明 geodesic 路径可能经过 shortcut。这条边应被降权。

**物理含义：** 两个 UE 在 geodesic 图上近，但在直接信道距离上不近 → 它们的"近邻"关系是通过中间节点桥接出来的，不是直接物理相似。

**计算：** 从 ADP 距离矩阵取每行的 top-k 排序，建立 rank lookup table（O(nk) 预处理，O(1) 查询）。

### 证据 B: 时间一致性

```
Δt = |t_i - t_j|  (秒)
Δt_mid = median({Δt 对所有 W_phys 边})
r_time = exp(-Δt² / (2 · Δt_mid²))    ∈ (0, 1]
```

时间上远离的样本即使 geodesic 近，信道可能已发生变化。这种边应被怀疑。

**物理含义：** 信道在时间上连续变化，时间差大的样本间的 geodesic 近邻关系可能是时间平均效应造成的。

**计算：** 只需要 timestamp 数组和中位数计算，O(1) 每条边。

### 证据 C: 直接融合距离一致性（复用 direct-support 工作）

```
d_fused = min(ADP_ij, γ · Δt)
q = max(log(d_fused / d_geo_ij), 0)
r_fused = exp(-q / τ)                  ∈ (0, 1]
```

当 geodesic 距离远小于直接融合距离时，geodesic shortcut 假说成立。

**物理含义：** `q` 量化 geodesic 距离被"压缩"了多少——压缩越多，边越可能是 shortcut。

### 综合可靠性权重

```
β_ij = (r_adp)^α₁ · (r_time)^α₂ · (r_fused)^α₃
W_reliable_ij = β_ij · W_phys_ij
```

**性质：**
- 所有证据来自物理测量（ADP, timestamp, fused distance），不包含 Y0 → 无循环论证
- β ∈ [0, 1]，乘法形式：任何一条证据认为边可疑，综合可靠性即降低
- 每条被抑制的边有可追溯的物理原因
- `α₁=α₂=α₃=0` → 退化为 baseline W_phys（消融可控）

### 与 direct-support 的关键区别

| | Direct-support | Multi-view (本设计) |
|---|---|---|
| 证据数量 | 1 个 (d_fused/d_geo) | 3 个 (ADP rank, time, fused) |
| 计算依赖 | 仅 fused + geodesic | ADP + timestamp + fused + geodesic |
| 可追溯性 | 单因素 | 可逐证据消融 |
| Y0 使用 | 否 | 否 |

---

## 4. Huber Graph-TV Regularization

将 L2 Laplacian 替换为 edge-preserving Huber 罚项：

```
min_Z  ||Z - Y₀||²_F  +  λ Σ_{ij} W_reliable_ij · ρ_δ(||Z_i - Z_j||)
```

其中：

```
ρ_δ(x) = { x²/2,                |x| ≤ δ
         { δ·(|x| - δ/2),       |x| > δ
```

### 与 L2 的对比

| 性质 | L2 Laplacian | Huber Graph-TV |
|---|---|---|
| 小差异惩罚 | 二次（温和） | 同 L2 |
| 大差异惩罚 | 二次（很强） | 线性（有界） |
| 过平滑风险 | 高 | 低 |
| 求解复杂度 | 线性系统 | IRLS 迭代 |
| δ→∞ | — | 退化为 L2 |

### δ 的物理含义

`δ` 控制"多大差异算大"。在 chart 坐标空间中：
- 如果 chart 坐标尺度归一化 ≈ 10m，`δ=0.1` 表示 0.1m 以上的差异不再二次惩罚
- `δ` 应适配 chart 坐标的典型局部梯度分布

---

## 5. IRLS 求解器

使用迭代重加权最小二乘 (IRLS) 外循环包裹内层 Jacobi 迭代：

```
Algorithm MultiViewHuberRefine(Y0, W_reliable, λ, δ, outer_iters, inner_iters):
    Z = Y0
    for t = 1 to outer_iters:
        // IRLS step: 从当前 Z 计算每条边的 Huber 权重
        for each edge (i,j):
            d = ||Z_i - Z_j||
            w_huber_ij = 1              若 d ≤ δ
            w_huber_ij = δ / d         若 d > δ

        // 构建本次外循环的加权图
        W_current = W_reliable ⊙ w_huber

        // 内层 Jacobi: 求解 (I + λ L_current) Z_new = Y0
        Z_new = Z
        for inner = 1 to inner_iters:
            每个节点 i:
                acc = Σ_{j∈N(i)} W_current_ij · Z_new_j
                Z_new_i = (Y0_i + λ · acc_i) / (1 + λ · deg_i)

        Z = Z_new

    return Z
```

### 收敛特性

- 外层 IRLS 通常 5-10 次足够（Huber 是凸的，IRLS 保证收敛到全局最优）
- 内层 Jacobi 每次外层 50-100 步，不比 baseline 的 200 步多
- 总步数 ~10×50=500，约为 baseline 200 步的 2.5 倍
- 可提前终止：当 ||Z^(t) - Z^(t-1)|| < ε

### 与 baseline Jacobi 的关系

| 组件 | Baseline Jacobi | Huber IRLS Jacobi |
|---|---|---|
| 图权重 | 固定 W_phys | 每外层更新 W_reliable ⊙ w_huber^(t) |
| 更新公式 | 同 | 同（只是权重不同） |
| 收敛保证 | 线性系统 | Huber 凸优化, IRLS 全局收敛 |

---

## 6. 参数系统与消融

### 参数一览

| 参数 | 语义 | 合理范围 | 默认建议 |
|---|---|---|---|
| λ | 整体正则强度 | 0.05-0.4 | 0.15（同 baseline） |
| δ | Huber 截断阈值 | 0.01-0.5 | 0.1 |
| α₁ | ADP 证据指数 | 0-2 | 1.0 |
| α₂ | 时间证据指数 | 0-2 | 0.5 |
| α₃ | 融合距离证据指数 | 0-2 | 1.0 |
| outer_iters | IRLS 外层迭代 | 1-20 | 10 |
| inner_iters | 内层 Jacobi | 20-100 | 50 |

### 退化路径

| 参数设定 | 退化结果 |
|---|---|
| δ→∞, α₁=α₂=α₃=0, outer_iters=1 | baseline (W_phys + L2) |
| δ→∞, 任何 α | multi-view edge filtering + L2 |
| δ∈(0,∞), α=0, outer_iters≥1 | W_phys + Huber（纯正则替换） |
| δ∈(0,∞), α>0, outer_iters≥1 | 完整模块 |

### 实验计划

阶段 1 — Huber diagnostic（独立验证正则替换效果）：
```
outer_iters=1, δ∈{0.01, 0.05, 0.1, 0.2, 0.5}, α=0
```

阶段 2 — Multi-view diagnostic（独立验证边过滤效果）：
```
outer_iters=5, δ→∞, α∈{0.5,1.0,2.0}
```

阶段 3 — Full module（组合验证）：
```
outer_iters=10, δ=0.1, α=[1.0, 0.5, 1.0]
```

---

## 7. 为什么这个模块可能超越 baseline

### 减少尾部误差（P95 0.635m → 预期改善）

Baseline 尾部误差的可能来源：geodesic shortcut 边将物理断开区域的点坐标拉近。

- Multi-view 证据在此截断：如果 ADP 和 timestamp 都认为边可疑，边权重被抑制
- 即使边权重未被完全抑制，Huber 的 δ 限制了该边对坐标的最大拉力

### 防止过平滑

- Baseline L2: ||Z_i - Z_j|| 越大，惩罚越强 → 边缘被强行拉平
- Huber δ: 大差异边接受"不超过 δ 的拉力" → 保护局部几何弯曲

### 无循环论证

- Alpha-only 用 r_i = ||L_phys Y0_i|| 决定 alpha，但 r_i 本身依赖 Y0 和 W_phys 两者
- Multi-view 只依赖物理测量，证据可预先计算一次

### 可解释性

- 每条被降权的边可以回答"为什么"：ADP 排名高? 时间远? 融合距离不匹配?
- 这为论文实验分析提供了结构化讨论空间

---

## 8. 实现方案

### 文件结构

新入口点：
```
run_multi_view_huber_refined_charting.py
```

与现有脚本复用关系：
- `common_metrics.py`, `common_dichasus.py`, `run_adp_fused.py`, `run_adp_siamese.py` — 复用
- `compute_neighbor_indices()` — 复用
- `solve_graph_harmonic_refinement()` — 不直接复用，但 inner Jacobi 逻辑几乎相同

### 核心函数签名

```python
def compute_multi_view_reliability(
    geodesic_neighbors: np.ndarray,  # (n, top_k) neighbor indices
    adp_distances: np.ndarray,        # (n, n) direct ADP distance
    timestamps: np.ndarray,           # (n,) float seconds
    fused_distances: np.ndarray,      # (n, n) fused distances
    graph_k: int,
    gamma: float,
    alpha_adp: float = 1.0,
    alpha_time: float = 0.5,
    alpha_fused: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Returns per-edge beta weights and diagnostic statistics."""
```

```python
def solve_huber_graph_tv_refinement(
    init_embedding: np.ndarray,
    graph: sp.coo_matrix,
    lambda_smooth: float,
    delta_huber: float,
    outer_iterations: int,
    inner_iterations: int,
    device: torch.device,
) -> np.ndarray:
    """Solve Huber graph-TV by IRLS outer loop + Jacobi inner loop."""
```

### 测试结构

```
tests/test_multi_view_huber_refinement.py
```

测试点：
- r_adp: 已知 ADP 一致/不一致时的正确计算
- r_time: 时间一致/不一致时的衰减
- r_fused: q 计算与 beta 映射
- IRLS 单步: 退化到 L2（δ→∞ 时）
- IRLS 完整: 凸性保证下收敛到闭式解（小问题上的数值验证）

---

## 9. 风险评估

| 风险 | 概率 | 缓解 |
|---|---|---|
| Multi-view 过滤了过多边，移除有用的近邻 | 中 | 乘积形式允许逐证据消融；指数 α 可调至 0 观察退化 |
| Huber outer loop 震荡不收敛 | 低 | 内层 Jacobi 有阻尼效果；Z 只在完整内层求解后更新 |
| 对 mean/median 误差无改进 | 中 | 主要目标为 P95/max 尾部；如果 mean 持平也是可接受结果 |
| 计算时间增加 2.5 倍 | 高 | 可设置提前收敛条件；outer_iters=1 的 diagnostic mode 先验证趋势 |
| 三个证据之一贡献为负（单独消融反而变差） | 低 | 乘积形式天然允许 α=0 关闭坏证据；可以报告并选择子集 |

---

## 10. 与文献学习结论的关系

| 文献启发 | 对应设计 |
|---|---|
| W_phys 不能被当作确定真值 (Robust GF) | Multi-view edge reliability 为每条边输出可信度 |
| L2 过平滑问题 (Graph-SSTV) | Huber ρ_δ 替换 L2, IRLS 求解 |
| dissimilarity uncertainty (Uncertainty-Aware CC) | r_adp/r_fused 直接建模距离的不确定性 |
| NLoS/noisy measurement masking (TDoA NLoS) | 三个物理证据从源头判断测量可靠性 |
| adaptive activation (UNILocPro) | 每条边的 β 是"激活度"，非二值开关 |
| iterated graph Laplacian (graphLa+) | IRLS 外循环等价于 iterative graph recalibration |

---

## 11. 与 direct-support 的根本区别

Direct-support 的基本假设是：
```
一条定量的边级可靠性分数 + 仍然用 L2 Laplacian → 期望改善
```

本模块的基本假设是：
```
图边可靠性和正则罚项是两个独立问题，必须同时处理。
```

实验上，这是两个完全独立验证的维度：
- 如果 multi-view + L2 改善 → 图边可靠性是当前主要瓶颈
- 如果 W_phys + Huber 改善 → 正则罚项形式是当前主要瓶颈
- 如果两者都改善且叠加更大 → 两个问题都存在