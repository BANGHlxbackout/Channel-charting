# Refinement / Repair：多源约束、下游任务、环境工程与图高频学习

更新时间：2026-06-25

本笔记继续学习参考论文，不正式提出优化 idea，不安排实验。本轮精读：

- 07 `Global-Scale Self-Supervised Channel Charting with Sensor Fusion`
- 09 `Channel Charting-Based Channel Prediction on Real-World Distributed Massive MIMO CSI`
- 11 `Channel Charting in Smart Radio Environments`
- 18 `Improving Your Graph Neural Networks: A High-Frequency Booster`

## 1. 论文 07：多源信息不是简单拼接，而是不同几何约束的互补

### 论文机制

论文在训练 FCF 时同时使用三类信息：

1. CIR/CSI 作为模型输入；
2. 已知 TRP 坐标与 ToA，形成 UE 到各 TRP 的绝对距离约束；
3. laser scanner 加 ICP，估计两个时刻之间的相对位移。

ToA 约束把 chart 拉到 global coordinate frame；laser displacement 约束补充沿轨迹的相对几何。论文明确指出，单纯 differential bilateration 可能产生位置歧义，因此把它改成每个样本到 TRP 的绝对 range loss。

laser/ICP 证据也不是恒定可靠：当两次扫描差异过大，ICP 可能失败，所以论文对时间距离较远的 pair 减小或关闭相应权重。少于三个 TRP 时还可能存在全局 offset ambiguity，论文在训练后用 ToA 再估计一个偏移向量。

### 证据范围

- 实验是重建轨迹上的 ray-tracing 仿真，不是实测 CSI。
- 场景使用两个 LoS TRP、已知 UE 高度和训练期 laser。
- 论文把 NLoS 扩展留作未来工作。
- 表中 `Triplets+Bilat.` 在该设置下 CE90 很差，说明“加入物理 loss”本身不保证有效；约束形式与可识别性很关键。

### 对当前 refinement 的学习含义

多源证据的价值不是“源越多越好”，而是不同源分别约束：

```text
absolute range / relative displacement / global offset
```

每类证据有自己的失效条件和作用域。论文 07 的核心方法仍是 training-time supervision；其 post-training offset correction 只修平移偏差，不等于一般 chart repair。

## 2. 论文 09：chart 质量与下游收益之间还有完整中间链条

### 论文机制

该工作把 ADP/timestamp fused metric 训练出的 FCF 用于未来 CSI 预测：

1. FCF 把新 CSI 映射到 chart；
2. 根据最近 `K` 个 chart 位置线性外推未来 latent position；
3. 在训练 chart 上做 Delaunay triangulation；
4. 对包含预测位置的三角形顶点所对应 CSI 做重心插值；
5. 用预测 CSI 计算 achievable sum rate。

因此，下游表现不只取决于 chart 的 CT/TW/KS，还取决于：

- latent dynamics 是否能被线性外推；
- 预测点是否落在训练 chart 的三角网覆盖内；
- 局部 CSI 是否适合做重心插值；
- array selection 与随机相位处理是否正确。

### 关键结果与边界

- train/test chart 的 CT/TW/KS 和仿射后 MAE 很接近，说明 FCF 在同一环境的新样本上有良好映射能力。
- CC predictor 在短预测时域不如 outdated CSI；优势出现在较长时域。
- 以 0.192 秒采样间隔计，CC-interp 在约 1 秒以上的预测时域表现最好。
- 评估排除了预测 chart 位置不在任何训练三角形内的样本，因此结果不是无条件覆盖全部预测点。
- 数据局限于单用户、低且近似恒定速度，只选一个阵列下行；曲线轨迹使线性外推失效。

### 对当前 refinement 的学习含义

这篇论文支持“几何一致的 chart 可以承载下游无线任务”，但不支持：

```text
CT/TW/KS 提升 -> 下游性能必然提升
```

要得到下游 claim，必须连通 chart、运动模型、覆盖范围、插值器和任务指标的完整评价链。当前 refinement 结果只有 chart/localization 指标，不能借论文 09 直接宣称 sum-rate 或 channel-prediction 改善。

## 3. 论文 11 与论文 14：同一工作线，不能重复计数

### 文献关系

论文 11 是 2025 年的较早版本，主要使用 t-SNE 和 static EMS codebook，强调通过改变传播环境来提高 CSI spatial dissimilarity，并以 upper-quantile CT/TW/localization error 选配置。

论文 14 是扩展版本，加入：

- semi-supervised autoencoder，以验证非参数与参数 mapping 的一致性；
- SNR 与 dissimilarity 的显式权衡分析；
- idealized RIS、随机静态相位等更多对照；
- trajectory dropout 等更完整的 tail evaluation。

因此，11 和 14 应视为同一研究线的早期版与扩展版。后续文献论证时不能把它们当作两个独立团队、两个独立发现来增强证据数量。

### 对当前 refinement 的学习含义

这一工作线证明的是**上游环境工程会改变 radio geometry 的可辨识性**。固定图并不是脱离环境的抽象对象；传播环境、EMS 位置和 codebook 都会改变 CSI covariance 与 pairwise dissimilarity。

但它并不研究环境随时间变化后的 graph drift，也没有提出 post-hoc correction。它使用离线、场景特定的 ray tracing 和 codebook search，性能强依赖部署位置与配置。因此可吸收的知识是“图证据受传播环境控制”，不能扩张成“当前方法已经处理动态环境”。

## 4. 论文 18：高频有用，但其任务与当前 chart refinement 不同

### 论文机制

论文研究半监督节点分类。其问题设定是：

- GNN message passing 往往表现为低通滤波；
- 多层堆叠会 over-smooth；
- heterophilic graph 中，相连节点标签可能不同，低通同质化假设与任务冲突。

论文从原图 complement 中采样非边，构造 Complement Laplacian Regularization（CLAR），把高通与原图低通正则组合。实验指标是节点分类 accuracy、深层 GCN 的 over-smoothing 和删边下的 topology robustness。

### 为什么不能直接迁移

当前 chart refinement 与论文 18 有三处本质差异：

1. 当前信号是二维连续坐标，不是分类 hidden representation。
2. 当前物理图的期望关系本来就是“物理近邻在 chart 中应靠近”，更接近 homophilic smoothness；没有论文证据证明它是 heterophilic graph。
3. complement edge 在节点分类中用于补高频，不代表“物理图中不相连的任意点应该被推远”。

因此论文 18 只能支持一个弱边界：

```text
低通并非对所有图任务都充分，高频可能携带任务信息。
```

它不能直接支持：

```text
当前 chart 坐标必须加入 complement graph、high-pass loss 或 CLAR。
```

若以后要迁移，必须先证明当前 chart 的哪一部分真实几何信息位于 `L_phys` 的高频谱段，并与错误高频区分。当前论文学习和实验尚未提供这项证据。

## 5. 四篇论文放在同一框架中的位置

| 论文 | 主要对象 | 信息进入阶段 | 解决的核心问题 | 对 post-hoc repair 的直接性 |
|---|---|---|---|---|
| 07 | ToA + TRP + laser | FCF 训练；另有平移 offset 校正 | global scale 与多源几何约束 | 间接 |
| 09 | chart + trajectory + CSI interpolation | 下游推理 | channel aging / future CSI | 评价边界 |
| 11/14 | 传播环境 + EMS | 数据生成之前 | 提高信道位置可辨识性 | 上游边界 |
| 18 | GNN hidden representation | GNN 训练 | heterophily / over-smoothing | 弱迁移 |

这四篇论文分别位于当前 pipeline 的不同位置，不能被压成同一种“精修相关工作”：

```text
环境 -> 物理观测 -> chart 训练 -> post-hoc repair -> 下游任务
11/14      07                           当前方法        09
                         18 属于另一类 GNN 分类问题
```

## 6. 本轮吸收后的知识

1. 多源物理信息需要按约束类型、可识别性和可靠范围拆开，不能只做无差别 fusion。
2. 下游收益需要完整任务链验证，chart 指标改善不能替代下游指标。
3. 环境工程改变的是 graph evidence 的生成机制，不等于 graph refinement。
4. 同一论文线的早期版与扩展版必须去重，避免夸大文献支持。
5. GNN 高频论文只能说明普通低通过强的可能性，不能直接证明当前坐标 refinement 需要高通。

本轮仍然只完成知识吸收，没有据此正式选择或提出优化 idea。
