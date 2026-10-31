# Channel Charting 相关论文轻量检索

更新时间：2026-06-25

本笔记按 `light-literature-search` 流程重新检索 Channel Charting 相关论文。目标不是硬凑 15 篇，而是区分：

- 正式期刊、SCI2+ 高置信核心文献；
- Channel Charting 奠基/经典文献；
- 重要但属于会议、新刊、Letters、Open Journal 或分区待核验的候选；
- 与当前 `diffusion_refined` / physical graph repair 主线真正相关的论文。

## 1. 检索留痕

检索源：

- Crossref：核 DOI、出版类型、venue。
- OpenAlex：补充引用数和漏检项。

时间范围：

```text
2018-2026
```

选择这个范围是为了同时覆盖 Channel Charting 的奠基论文和 2022-2026 的最新期刊工作。

检索式：

```text
channel charting
wireless channel charting
triplet channel charting
model based channel charting
angle delay profile channel charting
timestamp channel charting
distributed MIMO channel charting
LoS channel charting
physical channel fingerprints channel charting
channel charting beam prediction
channel charting localization
TDoA channel charting NLoS
self-supervised channel charting
multipoint channel charting
```

命中统计：

```text
Crossref raw hits: 412
OpenAlex raw hits: 420
Deduplicated Channel Charting records: 72
```

过滤规则：

- 标题必须显式包含 `Channel Charting` 或 `channel charting-based`。
- 普通 wireless localization、radio map、CSI fingerprint、NLoS localization 不进入本表核心，除非论文明确围绕 Channel Charting。
- arXiv-only 不作为核心引用。

## 2. SCI2+ 高置信期刊核心文献（2022-2026）

这些是当前最适合作为正式论文核心相关工作的 Channel Charting 期刊文献。中科院分区仍需学校数据库最终核验，但从 venue 层级看属于高置信 SCI2+ 支撑。

| # | 论文 | 年份 | 期刊 | DOI | Crossref / OpenAlex 引用 | 与当前工作的关系 |
|---:|---|---:|---|---|---|---|
| 1 | Wireless Channel Charting: Theory, Practice, and Applications | 2023 | IEEE Communications Magazine | https://doi.org/10.1109/MCOM.001.2200344 | 33 / 27 | CC 综述和问题定义，适合作背景与评价指标来源。 |
| 2 | Beam SNR Prediction Using Channel Charting | 2023 | IEEE Transactions on Vehicular Technology | https://doi.org/10.1109/TVT.2023.3275280 | 16 / 13 | 说明 chart 能服务下游无线任务，可支撑 refined chart 的应用价值。 |
| 3 | Model-Based Approaches to Channel Charting | 2024 | IEEE Transactions on Communications | https://doi.org/10.1109/TCOMM.2023.3329522 | 2 / 2 | model-based geometry 与 learned CC 的直接对照；支撑物理证据参与 chart 修复。 |
| 4 | Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting | 2024 | IEEE Transactions on Communications | https://doi.org/10.1109/TCOMM.2024.3386571 | 29 / 29 | 与导师 ADP/time fused metric 最直接相关，是物理图构建的核心前置文献。 |
| 5 | Optimizing Multicarrier Multiantenna Systems for LoS Channel Charting | 2024 | IEEE Transactions on Wireless Communications | https://doi.org/10.1109/TWC.2024.3417892 | 4 / 4 | 关注 LoS 条件和物理可分辨性，说明 chart 质量受无线物理观测设计影响。 |
| 6 | Channel Charting in Real-World Coordinates With Distributed MIMO | 2025 | IEEE Transactions on Wireless Communications | https://doi.org/10.1109/TWC.2025.3559723 | 10 / 7 | D-MIMO、AP 几何和真实坐标约束；与 physical consistency correction 高度相关。 |
| 7 | A Channel Charting-Based Semi-Supervised Positioning Algorithm With Co-Training | 2026 | IEEE Transactions on Communications | https://doi.org/10.1109/TCOMM.2026.3693171 | 0 / - | CC + semi-supervised positioning，代表前端训练/定位框架路线。 |
| 8 | Channel Charting With Physical Channel Fingerprints For Massive MIMO-OFDM Channel Acquisition | 2026 | IEEE Transactions on Wireless Communications | https://doi.org/10.1109/TWC.2026.3700686 | 0 / - | physical channel fingerprints 与 CC 结合，可支撑“物理 fingerprint 是 chart repair 证据”。 |

## 3. 奠基和经典文献

这些文献不都在 2022-2026 范围内，但对 Channel Charting 方向的脉络很重要。写 introduction / related work 时应适当引用。

| 论文 | 年份 | 类型 / 期刊 | DOI | 引用 | 作用 |
|---|---:|---|---|---|---|
| Channel Charting: Locating Users Within the Radio Environment Using Channel State Information | 2018 | IEEE Access | https://doi.org/10.1109/ACCESS.2018.2866979 | 190 / 202 | Channel Charting 早期奠基论文，定义用 CSI 在 radio geometry 中定位用户。 |
| Multipoint Channel Charting for Wireless Networks | 2018 | Asilomar | https://doi.org/10.1109/ACSSC.2018.8645281 | 54 / 54 | multipoint CC 早期代表，适合写多点/多基站 CC 背景。 |
| Triplet-Based Wireless Channel Charting: Architecture and Experiments | 2021 | IEEE Journal on Selected Areas in Communications | https://doi.org/10.1109/JSAC.2021.3087251 | 52 / - | triplet-based neural CC 代表性期刊论文，是 FCF/Siamese/triplet 路线的重要背景。 |
| Efficient Channel Charting via Phase-Insensitive Distance Computation | 2021 | IEEE Wireless Communications Letters | https://doi.org/10.1109/LWC.2021.3109295 | 18 / 17 | phase-insensitive distance，和 dissimilarity / metric 设计相关。 |

## 4. 高相关但暂不计入 SCI2+核心的期刊候选

| 论文 | 年份 | 期刊 | DOI | 原因 |
|---|---:|---|---|---|
| Indoor Localization With Robust Global Channel Charting: A Time-Distance-Based Approach | 2023 | IEEE Transactions on Machine Learning in Communications and Networking | https://doi.org/10.1109/TMLCN.2023.3256964 | 与 time-distance / robust global CC 强相关，但 TMLCN 新刊分区需核验。 |
| Channel Charting Aided Pilot Reuse for Massive MIMO Systems With Spatially Correlated Channels | 2022 | IEEE Open Journal of the Communications Society | https://doi.org/10.1109/OJCOMS.2022.3225054 | 正式期刊但分区需核验；偏 CC 应用。 |
| Velocity-Based Channel Charting With Spatial Distribution Map Matching | 2024 | IEEE Journal of Indoor and Seamless Positioning and Navigation | https://doi.org/10.1109/JISPIN.2024.3424768 | 主题相关，但 JISPIN 新刊分区需核验。 |
| TDoA-Based Self-Supervised Channel Charting With NLoS Mitigation | 2026 | IEEE Transactions on Machine Learning in Communications and Networking | https://doi.org/10.1109/TMLCN.2026.3688619 | 与 NLoS-aware CC 高度相关，但 TMLCN 分区需核验。 |
| Model-assisted channel charting for 5G-Advanced localization with sparse data | 2026 | Digital Communications and Networks | https://doi.org/10.1016/j.dcan.2026.05.006 | 主题相关，期刊分区需核验。 |
| Efficient Base Station Handover via Multipoint Channel Charting-Based SNR Prediction | 2026 | IEEE Communications Letters | https://doi.org/10.1109/LCOMM.2025.3627797 | Letters 论文，可作应用补充，不作核心方法文献。 |
| Mobile User Trajectory Anomaly Detection via Unsupervised Channel Charting | 2025 | IEEE Wireless Communications Letters | https://doi.org/10.1109/LWC.2025.3620296 | Letters 论文，可作应用补充。 |

## 5. 重要会议论文 / arXiv 思想来源

这些论文很可能对 idea 有启发，但不要写成 SCI2+ 核心文献。

| 论文 | 年份 | Venue | DOI | 对当前工作的启发 |
|---|---:|---|---|---|
| Leveraging triplet loss and nonlinear dimensionality reduction for on-the-fly channel charting | 2022 | SPAWC | https://doi.org/10.1109/SPAWC51304.2022.9834035 | on-the-fly / fast CC，可作为弱初值 refinement 的动机。 |
| Improving Triplet-Based Channel Charting on Distributed Massive MIMO Measurements | 2022 | SPAWC | https://doi.org/10.1109/SPAWC51304.2022.9833925 | D-MIMO measurements 上的 triplet CC 改进。 |
| Channel Charting with Angle-Delay-Power-Profile Features and Earth-Mover Distance | 2022 | Asilomar | https://doi.org/10.1109/IEEECONF56349.2022.10052099 | ADP/ADPP features 与距离设计，和导师 ADP/time metric 前置相关。 |
| Augmenting Channel Charting with Classical Wireless Source Localization Techniques | 2023 | Asilomar | https://doi.org/10.1109/IEEECONF59524.2023.10477049 | classical localization evidence 参与 CC，启发物理证据修复 learned chart。 |
| Channel Charting in Real-World Coordinates | 2023 | Globecom | https://doi.org/10.1109/GLOBECOM54140.2023.10437714 | 后续 TWC 期刊版的会议前身。 |
| Global Scale Self-Supervised Channel Charting with Sensor Fusion | 2024 | Globecom Workshops | https://doi.org/10.1109/GCWKSHP64532.2024.11101632 | sensor fusion / global-scale CC，启发多源物理证据。 |
| Multi-Site Wireless Channel Charting Through Latent Space Alignment | 2024 | SPAWC | https://doi.org/10.1109/SPAWC60668.2024.10694402 | 多站点 latent alignment，与 chart consistency 相关。 |
| Channel Charting in Smart Radio Environments | 2026 | WCNC | https://doi.org/10.1109/WCNC65185.2026.11555380 | smart radio environment 下的 CC，适合做未来相关工作跟踪。 |

## 6. 文献脉络

### 6.1 从 CSI embedding 到 neural / triplet CC

代表：

- 2018 IEEE Access: Channel Charting using CSI.
- 2021 JSAC: Triplet-Based Wireless Channel Charting.
- 2022 SPAWC: On-the-fly channel charting.

这一线主要解决：

```text
CSI / channel features -> low-dimensional chart
```

对当前工作的启发：

```text
Y0 = FCF(CSI)
```

这个前端 chart 学习并不是我们的主要创新，应该作为 baseline / initial chart。

### 6.2 从 metric 设计到物理 dissimilarity

代表：

- ADP/timestamp dissimilarity metrics.
- ADPP + EMD.
- phase-insensitive distance.

这一线主要解决：

```text
什么距离更能反映 radio geometry?
```

对当前工作的启发：

```text
ADP/time graph 不只是 kNN graph，而是物理证据图。
```

### 6.3 从 self-supervised CC 到 model-based / real-coordinate CC

代表：

- Model-Based Approaches to Channel Charting.
- Channel Charting in Real-World Coordinates With Distributed MIMO.
- LoS Channel Charting.

这一线主要解决：

```text
如何把 learned chart 与真实物理坐标、AP geometry、LoS/model constraints 对齐?
```

对当前工作的启发：

```text
refinement 可以被写成 post-training physical consistency correction，
而不是重新训练一个更重的 model-based CC framework。
```

### 6.4 CC 的下游任务

代表：

- Beam SNR prediction using CC.
- Channel acquisition with physical channel fingerprints.
- Base station handover via multipoint CC.

这一线主要解决：

```text
chart 是否对无线系统任务有用?
```

对当前工作的启发：

```text
修复 chart 的价值可以从定位误差扩展到无线任务稳定性，
但当前阶段不要把下游任务写成主贡献。
```

## 7. 对 diffusion_refined 的直接启发

当前主线：

```text
Y0 -> physical consistency audit -> graph-based refinement / repair -> Z
```

最相关的 Channel Charting 文献排序：

1. **Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting**  
   直接提供 ADP/time physical graph 的前置依据。

2. **Model-Based Approaches to Channel Charting**  
   提供 model-based physical evidence 与 learned chart 的对照。

3. **Channel Charting in Real-World Coordinates With Distributed MIMO**  
   支撑“把 chart 拉回物理坐标/物理一致性”的叙事。

4. **Indoor Localization With Robust Global Channel Charting**  
   虽分区需核验，但 time-distance / robust global geometry 与当前修复思想高度相关。

5. **Triplet-Based Wireless Channel Charting: Architecture and Experiments**  
   neural/triplet chart 学习的代表性基础。

6. **Wireless Channel Charting: Theory, Practice, and Applications**  
   综述性背景与评价指标来源。

7. **TDoA-Based Self-Supervised Channel Charting With NLoS Mitigation**  
   NLoS-aware CC 与 topology defect / wrong-edge repair 关系密切，但分区需核验。

8. **Channel Charting With Physical Channel Fingerprints For Massive MIMO-OFDM Channel Acquisition**  
   支撑物理 fingerprint 作为修复证据。

## 8. 写作边界

可以写：

> Existing Channel Charting studies mainly improve the front-end metric, representation learning objective, model-based coordinate alignment, or downstream wireless tasks. In contrast, our work focuses on post-training physical-consistency repair of an already learned chart using ADP/time/radio-geometry evidence.

不要写：

- “已有 15 篇 SCI2+ Channel Charting 论文都支持我们的 refinement。”
- 把会议 / arXiv / 新刊分区未核验论文写成 SCI2+ 核心引用。
- 把泛无线定位、radio map、普通 NLoS localization 混成 Channel Charting 方法文献。
- 把导师的 Laplacian refinement 写成新的 Channel Charting metric。

更稳的论文叙事：

```text
Channel Charting literature tells us where the physical evidence comes from:
ADP/time, model-based geometry, LoS/NLoS, D-MIMO, real-coordinate constraints.

Graph regularization literature tells us how to repair a graph-supported signal.

The publishable gap is the missing bridge:
physical-consistency auditing and topology-aware repair of learned channel charts.
```

