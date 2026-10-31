# G1-ACF：复用原始 CSI 自相关特征的 Graph-FCF 实验执行规格

> 版本：v4 / 2026-10-16  
> 状态：**实现、Sanity、资源 Pilot、两组 full-N 训练与统一评价已完成**。结果见 `results/g1_acf_graph_fcf/REPORT.md`。  
> 优先级：**本文件取代此前 G1 v1/v2/v3 的模型及特征实现决策**。旧规格仅用于追溯。  
> 目标：优先验证是否能在当前真实 DICHASUS 数据的同一实验协议下获得更好的 Channel Chart 指标。这里的 G1 是参考图式 Channel Charting 研究的项目适配方案，不声明复现他人方法或直接构成论文创新。  
> Codex 执行原则：**直接实现、测试、运行、汇报；遇到关键数据/环境缺失时明确报告，不擅自改图、改输入特征、缩小 full-N 或把真实位置用于训练。**

---

## 0. 给 Codex 的直接任务

在包含以下实际 baseline 文件的工作目录执行：

```text
run_adp_fused.py
run_adp_siamese.py
run_diffusion_refined_charting.py
common_dichasus.py
common_metrics.py
preprocess_adp_old_shard.py
```

**G1 的冻结路线：**

```text
现有预处理的 H.npy（N,32,13）
   -> common_dichasus.load_data1 -> H: (N,4,8,13), complex64
   -> 原封不动复用 run_adp_siamese.AdpFeatureLayer
   -> 26,624 维真实 CSI 自相关特征（包含跨阵列相关、功率）
   -> Linear(26624,512) + ReLU + BatchNorm1d(512)
   -> 2 层 Residual Sparse GCN（使用现有 Stage-1 fused kNN 图）
   -> Linear(512,2) -> Y_G1
   -> 使用原有 geodesic.npy 的 stress loss 训练
   -> 与保存的 B0（Siamese FCF）结果比较
   -> 若 G1 生成成功，可选将 G1 输出送入现有 Stage-2 做 G1+S2
```

**不要再执行旧 G1 的三种特征方案：**
- v1：832 维 raw CSI real/imag；
- v2：3328 维归一化阵列内相关 + 52 维功率；
- v3：三分支 3328 + 9984 + 52、额外相位门控；
- 当前 G1 **直接复用现有 `AdpFeatureLayer`，不开发新特征、不做 Beam-ADP166、不引入相位门控**。

2026-10-16 用户补充：**G1-Identity 为必要消融**。它与 G1-Graph 共用特征、模型、参数、Loss、优化器、训练/验证点对和评价协议；唯一改动是通过 `--mode identity` 将传播邻接替换为单位矩阵。**不得为了获得收益而暗改监督信息来源。**

---

## 1. 必须理解的真实代码事实

| 项目 | 本地源代码与事实 | G1 要求 |
|---|---|---|
| TFRecord 处理 | `preprocess_adp_old_shard.py::process_csi` 完成 FFT shift、STO/CPO 校正、IFFT、`507:520` 时延截取、32 天线重新排序 | **不修改、不重新设计** |
| 模型数据 | `common_dichasus.load_data1`：`H.npy` 为 `(N,32,13)`，`h_by_array` 为 `(N,4,8,13)`、`complex64` | 直接读取 |
| 原 B0 特征 | `run_adp_siamese.AdpFeatureLayer.forward` 的 `einsum("bamt,bcnt->btacmn")` | 直接 import，严禁重新手写不一致版本 |
| 特征物理意义 | 每个 tap 生成全部 32 根天线之间的复数 `H_t H_t^H`（含阵列内部、跨阵列项及幅度） | 保留所有 26,624 个实维度 |
| Stage-1 图源 | `run_adp_fused.fused_geodesic` 基于 `fused.npy` 用 sklearn kNN 构图并计算最短路径 | 重建**同一张**图作为 GCN 传播图 |
| 训练 target | 同目录 `geodesic.npy` | 原样复用 |
| B0 loss | `run_adp_siamese.siamese_loss` | 同一定义 |
| Evaluation | `common_metrics.chart_metrics`、`localization_error_summary`、`run_adp_siamese.plot_chart` | 直接复用 |
| GT / UE 坐标 | `UE_pos.npy` | 只用于画图和训练结束后评价；**不能用于训练、学习率/早停或 checkpoint 选择** |
| AP 坐标 | 当前 `preprocess_adp_old_shard.py` 将 `AP_pos.npy` 写为全零占位 | **禁止作为实际阵列几何信息使用** |

### 1.1 关键特征定义：严格等于旧 B0

实际代码：

```python
from run_adp_siamese import AdpFeatureLayer

# h_batch: torch.complex64, shape (B, 4, 8, 13)
with torch.no_grad():
    x_batch = AdpFeatureLayer()(h_batch)  # (B, 26624), float32
```

内部特征为：

\[
C_{i,t}[p,q] = H_{i,p,t}\overline{H_{i,q,t}},\quad p,q\in\{1,\ldots,32\},
\]

输出维度：

\[
13\times 32\times32\times2=26624.
\]

本轮**不要**提前将其改为 Hermitian 半矩阵压缩、归一化协方差、ADP projector、阵列间单独归一化或 beamspace。现有自相关的绝对幅值包含功率变化；进行逐样本 L2 normalization / LayerNorm(26624) 可能改变这类信息。**本轮原始 26624 维输入不做额外的逐样本归一化。**

边界说明：该自相关特征对同一个 tap 上全部 32 根天线的公共相位旋转不变，对跨阵列**相对**相位仍敏感；它不保证抵抗所有跨阵列校准漂移。

### 1.2 数据及目录

工作目录从已存在的 baseline `.py` 所在目录起算。预期数据目录包含：

```text
datasets/adp_old_shard4/
    H.npy
    UE_pos.npy
    AP_pos.npy
    timestamps.npy
```

完整测地距离运行目录的**典型**形式：

```text
results/adp_old_fused/stride1_n20997_seed10/
    adp.npy
    fused.npy
    geodesic.npy
    distance_data.npz
    result.json
```

**这只是示例路径；使用实际 Windows 仓库中的现有路径。** 当前交付 ZIP 没有附带这些大型 `.npy`，Codex 必须先检查存在性。如果没有，按下文命令生成 distance run 或停止并说明缺少什么，不能假装完成真实实验。

---

## 2. 新增文件和代码边界

新增：

```text
methods/g1_acf_graph_fcf/run.py           # CLI、模型、训练、指标及结果保存
methods/g1_acf_graph_fcf/graph_utils.py            # fused 图重建、affinity、缓存、数据核验
methods/g1_acf_graph_fcf/compare.py        # 从已保存 embedding 进行统一指标对比
tests/test_g1_acf_graph_fcf.py    # 单元测试及小图反向传播测试
```

禁止修改下列现有基线的行为：

```text
preprocess_adp_old_shard.py
common_dichasus.py
common_metrics.py
run_adp_fused.py
run_adp_siamese.py
run_diffusion_refined_charting.py
```

如需公共功能，直接 import 现有函数。不要改写原始 B0 结果、覆盖已有 `model.pt` / `embedding.npz` / `result.json`，不要修改旧 SC² 分支。所有 G1 缓存、模型和结果保存在新的 `results/g1_acf_graph_fcf/` 下。

---

## 3. 图构建：必须对齐已保存的 Stage-1 距离

### 3.1 输入一致性检查

G1 CLI 至少提供：

```text
--data-path
--distance-run
--out-dir
--stride
--max-samples
--seed
--device
--mode                  # graph / identity
--phase                 # sanity / pilot / full
--hidden-dim
--num-layers
--dropout
--max-steps
--pairs-per-step
--eval-every
--patience-evals
--feature-chunk-size
--force
```

检查下列条件，不满足直接停止：

1. `data = load_data1(data_path, seed=seed)`、`idx = run_adp_fused.select_indices(len(data.h_by_array), stride, max_samples)`；
2. `distance_data.npz['indices']` 与 `idx` **逐元素一致**；
3. `fused.npy` 和 `geodesic.npy` 都是 `(N,N)`，对称/有限，图构建使用原始 `float32` 距离值；
4. `result.json` 中 `num_samples`、`stride`、`seed` 和数据路径/数据身份能与本次运行对应，冲突则停；
5. **实际 k 只从 `distance-run/result.json['neighbors']` 获取**；
6. 不得从 `run_adp_fused.py` 默认的 `--neighbors 20` 猜测 k。它在加载旧 `geodesic.npy` 缓存时可能把 `used_neighbors` 还原成命令行默认值，所以一定要做下面的 shortest-path 验证。

为了节省内存：`np.load(path, mmap_mode='r')` 读取大的 `fused.npy` 和 `geodesic.npy`，禁止把完整 geodesic 矩阵复制到 GPU。`result.json` 的数据路径允许相对路径与规范化绝对路径等价，其他元数据异常必须记录。

### 3.2 复用 sklearn 调用语义

```python
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

nbrs = NearestNeighbors(
    n_neighbors=used_neighbors,
    metric='precomputed',
    n_jobs=-1,
).fit(fused)

g_dir = kneighbors_graph(
    nbrs,
    used_neighbors,
    metric='precomputed',
    mode='distance',
    n_jobs=-1,
).tocsr()
```

- 默认期望每个节点有 `used_neighbors` 个**非自身**有向邻居，以实际重建结果为准并断言；
- 边 support 是有向 kNN 的**无向并集**：`i->j` 或 `j->i` 存在，就保留无向边 `(i,j)`；
- 通过 COO 的 `row/col` 维护 support，**不能因 `fused[i,j]==0` 就丢掉合法零权距离边**；
- 原始边距离一律取 `fused[i,j]`，不要对两个方向求和，也不要用 `geodesic.npy` 替代邻接图。

令 `rho_i` 为节点 i 的第 k 个**有向**近邻的 fused 距离，构建：

\[
w_{ij}=\exp\left[-\frac{D_{\mathrm{fused}}(i,j)^2}
{\max(\rho_i,10^{-6})\max(\rho_j,10^{-6})}\right].
\]

仅在保留的无向边上赋此 affinity，然后加入单位自环，执行对称归一化：

\[
\widehat A=\widetilde D^{-1/2}(W+I)\widetilde D^{-1/2},
\quad \widetilde D_{ii}=\sum_j(W+I)_{ij}.
\]

输出 `torch.float32` sparse COO/CSR 邻接矩阵。需要检查：对称、非负、有限、每节点自环存在、无额外非自环边。`torch.sparse.mm` 不需要对 adjacency 求梯度，只需正确对模型参数求梯度。

### 3.3 防止图与 `D_geo` 不一致

从确定性的 32 个源节点重新执行：

```python
scipy.sparse.csgraph.shortest_path(g_dir, directed=False, indices=source_ids)
```

与保存的 `geodesic.npy[source_ids]` 全行对比，记录 `max_abs_diff`、`mismatch_count` 和容差。预期在现有已验证数据配置中可完全一致；若不一致必须**停止**，并排查 k、tie、zero-edge、数据顺序和缓存来源。禁止悄悄换成另一张图继续训练。

缓存仅位于：

```text
results/g1_acf_graph_fcf/_graph_cache/
```

缓存 metadata 至少记录 `distance-run` 规范化路径、fused 文件大小/mtime（必要时 hash）、`N`、`used_neighbors`、kernel、schema 版本；不匹配即重建。**不要写回 source distance run。**

---

## 4. 26,624 维特征的实现与内存约束

### 4.1 唯一特征来源

```python
feature_layer = AdpFeatureLayer().eval()
with torch.no_grad():
    for start in range(0, N, feature_chunk_size):
        stop = min(N, start + feature_chunk_size)
        hb = torch.from_numpy(h[start:stop]).to(feature_device)  # complex64
        fb = feature_layer(hb).float().cpu().numpy()             # (B, 26624)
        feature_memmap[start:stop] = fb
```

- `h = data.h_by_array[idx]`；绝对不能改变阵列和 tap 顺序；
- `feature_chunk_size=128` 默认，可按显存减小到 64；
- `float32` 保存，使用 NumPy `.npy` memmap 或等价可随机读取缓存；
- **不额外做** `normalize_csi_vectors`、CSI 样本级功率归一化、跨阵列块门控、beam 变换；
- 特征没有可训练参数，缓存可避免 2,000 个 step 重复计算 CSI 相关矩阵；
- full N≈20,997 时仅原始特征缓存约 **2.08 GiB**（十进制约 2.24 GB），必须在启动前检查磁盘剩余空间。

缓存索引、文件路径与 hash/size 元信息不一致就拒绝复用，所有特征与原始 `AdpFeatureLayer` 在随机选取的样本上作数值逐元素比较。

### 4.2 GPU 资源处理

首选路径：将缓存的 `float32` node features 复制到 GPU，模型 full-graph 训练；但这本身占用约 2.08 GiB VRAM，加上投影/梯度/optimizer 以及其他张量需要更多显存。完整 `D_geo` 仅存 CPU mmap，每 step 按 pair indices 读取 target。

若显存不足，**先保持同一数学模型和全图 adjacency**，实现 `CPU memmap + 分块 Linear 投影 + activation checkpoint/recomputation` 的内存安全模式：

1. 26,624 维 feature matrix 驻留 CPU 磁盘映射，不常驻 GPU；
2. 按连续节点块复制到 GPU 执行同一 `Linear(26624,512)`；
3. 拼接投影后的 `(N,512)`，后续 BatchNorm/GCN 依然在**完整**图上传播；
4. 通过 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 等有梯度校验的重计算机制保证 projector 权重获得**正确梯度**，而非 `no_grad` 冻结 projector；
5. 对固定随机小图比较普通实现与内存安全模式的 forward / projector gradients（数值容差），通过后才跑 full N；
6. 如果当前 Windows/PyTorch 环境无法可靠实现，记录真实显存/OOM 情况后暂停，不要悄悄改模型或降采样。

图邻接在 GPU 用 sparse，禁止把 `(N,N)` 图转换成 dense。优先保证 `float32` 的 sparse mm 稳定；只有验证正确后才考虑混合精度。训练耗时可能显著高于 832 维方案，这是预期成本，不等于实现失败。

---

## 5. Graph-FCF 的冻结模型（替换旧 G1 输入及网络骨架）

默认：

```text
input_dim       = 26624
hidden_dim      = 512
num_gcn_layers  = 2
dropout         = 0.1
output_dim      = 2
```

结构（保持原输入的幅度，不在原始特征上先做 per-sample LayerNorm）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSparseGCN(nn.Module):
    def __init__(self, hidden=512, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj):
        message = torch.sparse.mm(adj, h)
        delta = F.relu(self.norm(self.lin(message)))
        return h + self.drop(delta)

class GraphFCFWithExistingACF(nn.Module):
    def __init__(self, input_dim=26624, hidden=512, layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        # 采用与 B0 已用特征投影相近的 BN，而不是在投影前丢掉绝对幅度
        self.input_bn = nn.BatchNorm1d(hidden, eps=1e-3, momentum=0.01)
        self.convs = nn.ModuleList(
            [ResidualSparseGCN(hidden, dropout) for _ in range(layers)]
        )
        self.head_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 2)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features, adj):
        h = self.input_proj(features)
        h = self.input_bn(F.relu(h))
        for conv in self.convs:
            h = conv(h, adj)
        return self.out(self.head_norm(h))
```

CPU-memmap 分块路径应仅更换 `self.input_proj(features)` 的**计算实现**，余下的 BatchNorm、两层 GCN、输出头应完全一致。禁止把每个块分别做 BatchNorm（会改变原 full-graph 计算）。

两层 Residual GCN 是本轮唯一主模型；不要再引入额外 attention、多关系四图、GAT、功率门控、SC² 权重或新的自监督 loss。如果 G1 指标不足，先把真实结果交付，再由后续研究决定是否修改。

---

## 6. Loss、采样与训练

### 6.1 原有 geodesic stress

对抽样点对 `(i,j)`，

\[
\mathcal L = \frac1{|\mathcal P|}\sum_{(i,j)\in\mathcal P}
\frac{\left(\|Y_i-Y_j\|_2-D_{geo}(i,j)\right)^2}
{D_{geo}(i,j)+m},
\qquad m=Q_{0.01}(D_{geo}).
\]

代码可直接复用：

```python
from run_adp_siamese import siamese_loss
loss = siamese_loss(y[left], y[right], target, margin)
```

`margin` 与原 B0 一样来自**完整** `geodesic.npy` 的 1% quantile；大矩阵量化若需内存优化，必须仍与原定义数值一致，不允许只采样猜测量化值。

### 6.2 采样和无 GT 早停

- `pairs_per_step = 65536`：每次从所有非对角节点对中均匀随机抽样；
- 固定 `validation_pairs = 200000`、`validation_seed = seed + 100000`：无序 pair 去重，排除 `i==j`；
- 每次训练 pair 排除 validation key `min(i,j)*N+max(i,j)`；
- 训练 target 和 validation target 都仅来自 `geodesic.npy`，不可访问 `UE_pos` 进行学习；
- 每 20 step 计算一次 validation stress，保存最优 `model_best.pt`；
- 连续 12 次评价无改善早停，输出必须来自**最优 checkpoint**。

优化超参数：

```text
optimizer               = AdamW
learning_rate           = 1e-3
weight_decay            = 1e-5
max_steps               = 2000
warmup_steps            = 100
cosine_min_lr           = 1e-5
gradient_clip_norm      = 5.0
pairs_per_step          = 65536
validation_pairs        = 200000
eval_every              = 20
patience_evals          = 12
seed                    = 10
```

每次更新：

```python
model.train()
optimizer.zero_grad(set_to_none=True)
y = model(x_all, a_hat)                    # 一次完整图传播
left, right = sample_train_pairs(...)
target = d_geo[left, right]               # CPU mmap gather -> GPU
loss = siamese_loss(y[left], y[right], target, margin)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
scheduler.step()
```

**不要**把普通随机节点 mini-batch 当作 full-graph 消息传播，它会切掉跨 batch 的图边。CPU feature streaming 仅是投影的数值等价计算，后续 GCN 仍保留全部节点与邻接关系。

训练期间记录所有 loss、学习率、最优 step、耗时和显存；出现 NaN/Inf、图不一致、输出坍缩或 backward 错误必须停止并汇报。

---

## 7. 执行顺序和 Windows PowerShell 命令

请先切换到 `run_adp_siamese.py` 所在目录。示例使用脚本当前目录下的 `datasets/`、`results/`。如仓库路径不同，**先打印实际路径并只调整路径**。

### Step A. 环境和数据确认

```powershell
python -c "import numpy, scipy, sklearn, torch, matplotlib; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
Test-Path datasets/adp_old_shard4/H.npy
Test-Path datasets/adp_old_shard4/UE_pos.npy
Test-Path results/adp_old_fused/stride1_n20997_seed10/geodesic.npy
Test-Path results/adp_old_fused/stride1_n20997_seed10/fused.npy
```

若已有正确的 full-N distance run，**不要重算 ADP/geodesic**。如果只有小样本 distance run，也不要拿它配 full-N 的 H 使用。

### Step B. 单元测试

```powershell
python -m pytest tests/test_g1_acf_graph_fcf.py -q
```

若 Windows Python 环境没有 pytest，可安装到当前项目使用的环境中；不引入 PyTorch Geometric 等非必要依赖。

### Step C. 小规模真实数据冒烟

**必须先找到或创建与 stride=10 完全匹配的距离缓存**。若尚不存在：

```powershell
python run_adp_fused.py `
  --data-path datasets/adp_old_shard4 `
  --stride 10 --seed 10 `
  --out-dir results/adp_old_fused
```

在 N=20,997 条数据上 `stride=10` 的样本数是 2,100；先通过 `distance_data.npz['indices']` 验证。

```powershell
python -m methods.g1_acf_graph_fcf.run `
  --mode graph --phase sanity `
  --data-path datasets/adp_old_shard4 `
  --distance-run results/adp_old_fused/stride10_n2100_seed10 `
  --stride 10 --seed 10 `
  --device cuda:0 `
  --feature-chunk-size 128 `
  --max-steps 100 --eval-every 10 --patience-evals 6 `
  --out-dir results/g1_acf_graph_fcf
```

没有 CUDA 时，可改 `--device cpu` 完成小图或 stride-10 的功能测试；不要将 CPU 冒烟运行当作 full-N 训练成绩。

**冒烟通过条件**：数据/索引一致；重建图最短路复验成功；特征与原 `AdpFeatureLayer` 一致；稀疏 GCN forward/backward 成功；训练 loss 下降趋势合理；生成全部产物且指标有限。

### Step D. 完整 G1 主实验（优先）

```powershell
python -m methods.g1_acf_graph_fcf.run `
  --mode graph --phase full `
  --data-path datasets/adp_old_shard4 `
  --distance-run results/adp_old_fused/stride1_n20997_seed10 `
  --stride 1 --seed 10 `
  --device cuda:0 `
  --hidden-dim 512 --num-layers 2 --dropout 0.1 `
  --feature-chunk-size 128 `
  --pairs-per-step 65536 `
  --max-steps 2000 --eval-every 20 --patience-evals 12 `
  --out-dir results/g1_acf_graph_fcf
```

默认不覆盖完整结果。同一目录再次运行应使用新 run name 或显式 `--force`，并在日志标注。

**重要：这是一轮 full-N、seed=10 的探索性实验，不是重复种子后的论文最终结论。** 如算力预算不足，允许先执行 full-N 的 200–400 step **预跑**来判断可行性，但必须在结果中标记 `pilot`、不能当作完整结果或虚构最终改善；实际 full-N 训练要求按上述设置或明确记录早停。

### Step E. 对比 B0 / G1

B0 结果应从现有 `run_adp_siamese.py` 的真实运行目录读取；绝对不能将空白的 ZIP `results/` 视为已有模型结果。

```powershell
python -m methods.g1_acf_graph_fcf.compare `
  --baseline-embedding results/adp_old_siamese/stride1_n20997_seed10/embedding.npz `
  --identity-embedding results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/embedding.npz `
  --graph-embedding results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/embedding.npz `
  --seed 10 `
  --out-dir results/g1_acf_graph_fcf/comparison
```

上例为已完成 full-N 两组的实际目录。运行 Identity 训练时，使用相同训练命令，仅将 `--mode graph` 切换为 `--mode identity`。

### Step F. 可选：G1 + 原导师 Stage 2

只在 G1 原始图谱、结果和 B0 对比已经保存后运行。优先复用导师当前已有 `graph_k=128, lambda=0.15, iterations=200` 配置，必要时读取当前实验记录核实：

```powershell
python run_diffusion_refined_charting.py `
  --data-path datasets/adp_old_shard4 `
  --distance-run results/adp_old_fused/stride1_n20997_seed10 `
  --init-embedding results/g1_acf_graph_fcf/<G1实际运行目录>/embedding.npz `
  --stride 1 --seed 10 `
  --graph-k 128 --lambda-smooth 0.15 --iterations 200 `
  --out-dir results/g1_acf_graph_fcf_stage2
```

注意：原有 `run_diffusion_refined_charting.py` 会把 `geodesic_top512.npy` 邻居缓存写入 `distance-run`。**本轮不能改原脚本行为**，运行前要确认这一旧脚本副作用是否被现有仓库允许；若不允许，先报告并暂缓可选 Stage-2，不要为此修改原 baseline。

---

## 8. 评价协议（只改模型，统一重算指标）

评价时使用**同一数据、同一 `indices`、同一抽样子集**读取原 B0 / G1 的 embedding：

```python
from common_metrics import chart_metrics, localization_error_summary
from run_adp_siamese import plot_chart

# 一次性生成共同 eval indices，保存供 B0/G1/S2 复用
subset = np.array(
    random.Random(seed).sample(range(N), max(1, int(0.2 * N))),
    dtype=np.int64,
)
metrics = chart_metrics(pos[subset], embedding[subset])
errors = localization_error_summary(pos, embedding)
```

- `CT`、`TW`、`KS` 计算在同一固定 20% 节点子集；
- mean/median/P90/P95/max affine 后误差使用全部 N 节点；
- 直接复用 `plot_chart(out_dir/'chart.png', pos, embedding, title)`，左真实轨迹、右 affine 后图谱；
- 保存 `metric_subset_seed10.npy` 和 `eval_protocol.json`；
- 不从各自旧 `result.json` 直接拼接可能采用不同随机子集的 CT/TW/KS；要读取 embedding 后共同**重新计算**；
- 所有汇报都注明属于 **same-dataset/transductive chart quality**；旧 `affine_transform` 用同一批真实坐标拟合并评价，所以这里的定位误差是 **in-sample affine-aligned error**，不是 held-out 定位能力。

需要输出表：

| 方法 | CT ↑ | TW ↑ | KS ↓ | Mean (m) ↓ | Median (m) ↓ | P95 (m) ↓ | Max (m) ↓ | Params | Train time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Siamese | | | | | | | | | | |
| G1-ACF Graph-FCF | | | | | | | | | | |
| B0 + Stage 2（如已有） | | | | | | | | | | |
| G1-ACF + Stage 2（可选） | | | | | | | | | | |

**主问题：G1 是否在 KS 和 affine-aligned mean/P95 上获得收益，且 CT/TW 不出现明显退化。** 所有数据都必须真实记录；若没有提升，同样报告结果和训练诊断，不为追求提升而抹除负结果。

---

## 9. 必须测试的内容

1. **原始特征一致性**：随机复数样本 `(B,4,8,13)`，新缓存中对应行与直接调用 `AdpFeatureLayer` 的输出逐元素 `allclose`，形状精确 `(B,26624)`；另测与全 32 天线相关块一致；
2. **相位性质**：同 tap 的公共相位旋转不改变自相关结果；独立阵列相位旋转一般会改变跨阵列相关块；
3. **图复验**：kNN directed / undirected union / 32 源最短路与 `D_geo` 比较；支持零距离非自环；
4. **图归一化**：稀疏张量 finite / symmetric / nonnegative / self-loop；
5. **全图 backward**：随机 32–128 节点输入 `(N,26624)`，损失 `backward()`，`input_proj`、GCN、`out` 参数梯度都存在且有限；
6. **内存安全路径（仅当启用）**：同一小图上 chunked/projector-checkpoint 模式的预测和梯度与直接模式在浮点容差内相符，确认 BN 在拼接后进行；
7. **pair/validation**：无自配对、无训练/验证 pair 交叉，pair-key 逻辑正确；
8. **小图优化**：在固定小图与少量 pair 上可下降，不出现全部节点输出重合的 collapse；
9. **缓存与索引防错**：`distance-run` 或 `H.npy` 改变时拒绝加载旧 cache；
10. **端到端**：stride-10 与 full-N 的结果目录正确、`embedding.npz` 与 `indices` 对齐、`chart.png` 使用基线绘图函数。

---

## 10. 输出契约与交付验收

每个 G1 run 目录包含：

```text
result.json
embedding.npz               # arrays: indices / positions / embedding
model_best.pt
model_last.pt
training_history.json
graph_metadata.json
feature_metadata.json
chart.png
run_command.txt
```

`result.json` 必须保存：数据路径/身份、`distance-run`、index hash、实际 N、`used_neighbors`、边数、input_dim=26624、hidden_dim、层数、dropout、随机种子、所有优化参数、最优 validation stress / best step、训练耗时、峰值显存、版本号、训练/评估指标和任何偏离冻结规格的行为。具体日志要区分“完成”与“未运行/失败”。

`comparison/` 目录至少输出：

```text
comparison.csv
comparison.md
eval_protocol.json
metric_subset_seed10.npy
```

若可选 Stage-2 完成，另保存 G1+S2 的 embedding、result、chart，并把它加入统一评价表。

### 完成判定

- **Implementation PASS**：测试通过，原 B0 特征/模型/训练未变化；绘图复用接口的默认输出保持一致；
- **Smoke PASS**：真实 stride-10 run 成功并保存完整结果；
- **Full-N PASS**：真实 full-N、seed10 训练完成（或按协议早停），最佳模型可复现输出；
- **Result PASS**：B0 与 G1 的共同评价子集指标及图像保存；
- **Improvement**：必须从实际 `comparison.csv` 的数字判断，可以是 PASS/NO GAIN，不能预填。

若 full-N 因资源或数据不可用失败，请明确报告失败步骤、日志、真实内存/运行环境信息及最小修复建议，**不要声称实验完成**。

### Codex 最后回复必须包含

1. 实现了哪些新文件、改了哪些文件（应当**没有**改变 baseline）；
2. 当前是在 Implementation / Smoke / Full-N 哪一步，哪些确实跑完；
3. 训练样本数、实际 k、图源复验结果；
4. B0 / G1 的真实指标表（若未跑完则明确留空）；
5. G1 是否比 B0 获得指标收益，哪些指标改善/退化；
6. 结果文件的真实相对路径；
7. 未解决的问题、下一步最短操作。

**先执行 G1-ACF，不要因为曾讨论过 v1/v2/v3/Beam-ADP166 又在本轮自发添加多个输入方案。**

## 恢复检查点（2026-10-17）

- 目标与授权：实现 G1-Graph 和必要的 G1-Identity；完成索引、特征、图、梯度与 Loss 检查，真实 stride-10 Sanity 和 full-N 短资源 Pilot。**用户于 2026-10-16 确认两组完整 full-N 训练，各最多 2,000 step、合计约 6 GPU 小时上限。**先 Graph 后 Identity，训练后统一评价 B0/Identity/Graph，输出统一颜色的原始与仿射四列图。
- 代码：`methods/g1_acf_graph_fcf/graph_utils.py`、`methods/g1_acf_graph_fcf/run.py`、`methods/g1_acf_graph_fcf/compare.py`、`tests/test_g1_acf_graph_fcf.py`。2026-10-17 按用户要求从根目录归入 methods；入口说明见该目录 README。原 B0 特征/模型/训练未改；只给 `run_adp_siamese.plot_chart` 增加外部 axes/原始图选项，默认两列图经像素对比保持完全一致。当前工作树原有 `common_metrics.py` 的 GBK 注释导致 Python 导入失败，仅加编码声明以恢复导入，函数未改。
- 已完成与证据：5 项 G1 单元检查通过；`results/g1_acf_graph_fcf/REPORT.md` 记录 stride-10 两模式各 100 step Sanity、full-N Graph 5 step Pilot、图复验、两模式各 2,000 step 正式训练和统一评价。两组 pair hash、模型参数和 best step 一致，仅传播邻接不同。full-N 图 32 源最短路与缓存最大差 0；原始两向微差保留。比较 CSV 和原始/仿射四列图在 `results/g1_acf_graph_fcf/comparison/`。
- 结论：G1-Graph 小幅优于 B0，但 Mean/Median/P95、CT/TW 均不及 G1-Identity，仅 KS 微优；这一次固定设置不支持邻域传播具有可辨认的正向作用。完整数值和边界以主报告为准。
- 维护完成：两版四列图已直接用 baseline `plot_chart` 重画，指标不变；B0 原 `result.json`/`chart.png` 从 master 提交 `61178b5` 原样恢复，六项指标与比较表一致。B0 明确对应 `adp_old_siamese/stride1_n20997_seed10`，不与 Stage-2 Z 混用。
- 指标完整性：比较代码已导出原函数的全部误差 Mean/Median/P90/P95/Max 以及 CT/TW/KS，每行附 `source_json`。Graph/Identity 的 Max 为 1.463179/1.429263 m，B0 为 7.893630 m；三组 P90 基本相同。来源为已有 full-N embedding/JSON，无新增训练。
- 未完成：未提交或上传；没有执行可选 G1+Stage-2 或新节点泛化实验。
- 下一动作：用户已确认 G1 方向可继续探索；本次负向消融只约束当前冻结设置，方向未终止。后续具体实验尚未设计或运行。
