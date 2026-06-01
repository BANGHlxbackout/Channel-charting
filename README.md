# Graph-Harmonic Diffusion Refined Channel Charting（图扩散精修）

这是我们在 channel charting 上的改进工作的代码。核心思路：先用 ADP+时间戳测地相异性
训练出一张自监督的 channel chart（Siamese FCF），再用**图拉普拉斯的闭式平滑**
`(I + λL) Z = Y0` 对这张图做去噪精修——图 `L` 只用 CSI 推出来的 ADP/时间测地邻居
构建，**不用任何真实坐标**，保持自监督。

在同一份 DICHASUS 实测数据（20997 样本）上，它比所有基线都好：

| 方法 | CT | TW | KS | 均值误差 | P95 | 最大误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADP/time fused MDS | 0.9917 | 0.9934 | 0.0907 | 0.458 m | 0.961 m | 8.90 m |
| ADP/time Siamese FCF（基线） | 0.9957 | 0.9960 | 0.0841 | 0.409 m | 0.854 m | 7.99 m |
| **图扩散精修（本工作）** | **0.9982** | **0.9982** | **0.0651** | **0.319 m** | **0.635 m** | **1.14 m** |

> 最能打的点不是 CT/TW 的第三位小数，而是**定位误差 / 尾部**：最大误差从 7.99 m → 1.14 m。

---

## 目录说明

```
run_diffusion_refined_charting.py   ← 本工作（idea 1）：图扩散精修
run_multiscale_stress_charting.py   ← idea 2（多尺度采样），目前没超过基线，留作对照/消融
run_adp_fused.py                    ← 产出测地距离（精修的输入之一）
run_adp_siamese.py                  ← 产出初始 chart embedding（精修的输入之一）
preprocess_adp_old_shard.py         ← 从原始 tfrecords 生成 datasets/adp_old_shard4
preprocess_dichasus_numpy.py        ← 预处理的解码辅助
common_dichasus.py / common_metrics.py  ← 数据加载 + CT/TW/KS、仿射定位误差等指标
notes/                              ← 想法说明 + 全部复现记录
datasets/adp_old_shard4/            ← 【空】数据放这里（见下）
results/                            ← 【空】运行输出放这里
```

`notes/new_idea_diffusion_refined_channel_charting.md` 是这个想法的完整说明（数学推导、
目标函数、为什么可发表、待办），**先读它**。

---

## 数据（目录留空，需自行准备）

`datasets/` 和 `results/` 是空的（大文件没打包）。需要的数据是公开的
**DICHASUS dataset v2.0，`dichasus-cf02 / cf03 / cf04`**。

准备方式二选一：
1. 下载原始 tfrecords，然后跑 `preprocess_adp_old_shard.py` 生成
   `datasets/adp_old_shard4/`（里面应有 `H.npy / UE_pos.npy / AP_pos.npy / timestamps.npy`）；
2. 或直接向我们要已预处理好的 `adp_old_shard4/` 四个 .npy 放进该目录。

> 协议：拼接 cf02/cf03/cf04 后取 `full_dataset.shard(4, 0)`，共 20997 样本；
> STO/CPO 校准、IFFT 到时域、taps `507:520`、固定四阵列天线排序。

---

## 环境

Python 3，依赖：`numpy scipy scikit-learn torch matplotlib`；预处理另需 `tfrecord`。
有 GPU 更好（精修用 GPU Jacobi 迭代解线性系统，几秒钟）。

---

## 运行流程（4 步）

```bash
# 1. 预处理：生成 datasets/adp_old_shard4（已有数据可跳过）
python3 preprocess_adp_old_shard.py

# 2. ADP+时间融合测地距离 + MDS（产出 results/adp_old_fused/...，含测地距离）
python3 run_adp_fused.py --data-path datasets/adp_old_shard4 \
  --stride 1 --out-dir results/adp_old_fused

# 3. Siamese FCF 学初始 chart（产出 results/adp_old_siamese/.../embedding.npz）
python3 run_adp_siamese.py --data-path datasets/adp_old_shard4 \
  --stride 1 --out-dir results/adp_old_siamese

# 4. ★ 图扩散精修（本工作）
python3 run_diffusion_refined_charting.py \
  --data-path datasets/adp_old_shard4 \
  --distance-run results/adp_old_fused/stride1_n20997_seed10 \
  --init-embedding results/adp_old_siamese/stride1_n20997_seed10/embedding.npz \
  --out-dir results/diffusion_refined_charting \
  --stride 1 --graph-k 128 --lambda-smooth 0.15 --iterations 200
```

输出在 `results/diffusion_refined_charting/.../`，含 `result.json`（指标）、
`embedding.npz`、`chart.png`。`graph_k=128, lambda=0.15` 是目前搜到的最佳设置。

---

## 接下来要做的（把"内部最好"做成"能投的"）

目前这版是 transductive（没留测试集、单种子、20% 子集算指标），而且数学本身是
标准的图拉普拉斯/流形正则化平滑——直接当"新方法"投会被审稿人质疑。要补：

1. **方法学硬伤**：留 held-out 测试集；跑 5 个种子报均值±方差；主打位置误差/尾部。
2. **决定性消融：物理图 vs 欧氏图** —— 把 ADP/物理测地图换成普通欧氏 kNN 图，
   证明换了就不灵。这是区分"通用平滑"和"信道物理方法"的命根子实验。
3. **λ / 迭代次数的过平滑分析** + 一个有原则的停止准则（能量平台）。
4. **跨数据集 / 跨基础 chart**：在 Fraunhofer 5G/UWB 或 P2 上、在 triplet/AE chart 上也试。
5. **inductive / 样本外扩展**（Nyström 或蒸馏回 FCF），堵"只能离线"的批评。
6. 写作时**主动引用并承认**算子是标准的（Belkin–Niyogi 流形正则化、GFHF、Diffusion Maps、
   GSP 图 Tikhonov 去噪），把贡献定位成系统性、物理驱动、免训练的 CC 精修研究。

`notes/` 里有更详细的复现记录和待办。
