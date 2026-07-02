# diffusion_refined 参考论文学习总报告

更新时间：2026-06-25

## 1. 学习任务与边界

本轮学习对象是：

- `C:\Users\Hype beast\Desktop\参考论文` 中 34 篇 PDF；
- channel charting、物理距离/全局几何、LoS/NLoS、真实坐标、多源融合、图 Laplacian 正则、图去噪、图扰动、可学习展开等相关知识；
- 当前主线 2：对已训练 channel chart 做 refinement / repair。

本轮只学习和形成知识结构，没有正式提出优化 idea，也没有安排下一步实验。

学习标准不是“能复述摘要”，而是：

```text
能说明每条相关文献机制挑战当前 refinement 的哪个假设；
能区分可迁移知识与弱类比；
能识别重复文献、证据范围和不能越界的 claim；
后续能够据此合理构思优化 idea，而不是随机改 L、lambda 或堆模块。
```

## 2. 完整性核对

- 参考 PDF：34 篇。
- 深读摘录：`notes/paper_deep_extracts/` 中 34 份 `.deep.txt`。
- 论文机制矩阵：01–34 全部存在，无缺号。
- CC 核心论文与边界论文：01–14、30、31 已按原文机制复核。
- 图恢复核心论文：15、16、23、26、27、28、32、33、34 已进入算子/图可靠性层。
- 外围论文：17–22、24、25、29 已明确其可用范围和不能直接迁移之处。

34 篇的学习深度不是机械平均分配。直接决定主线 2 的论文读到机制层；外围论文读到“能否迁移、为什么不能滥用”的边界层。对当前任务而言，这比把每篇都强行解释成核心创新依据更可靠。

## 3. 当前已经建立的知识结构

### 3.1 当前 refinement 不是一个公式，而是七个假设

当前 baseline：

```math
Z^\star=\arg\min_Z\|Z-Y_0\|_F^2+
\lambda\operatorname{tr}(Z^\top L_{\mathrm{phys}}Z)
```

背后包含：

1. `Y0` 是适合用二次 fidelity 修正的 noisy chart；
2. `L_phys` 是可信物理图；
3. `W_ij` 能同时表示 similarity 与约束可信度；
4. 普通 GLR 是适合二维 chart 坐标的先验；
5. 单一全局 `lambda` 足够；
6. 固定图的一次闭式解足够；
7. transductive post-hoc refinement 满足部署需求。

后续任何优化都必须先指出哪条假设被论文或数据挑战，而不是从模块名出发。

### 3.2 物理图与图恢复算子必须分层讨论

文献已经把问题分成两个独立层次：

```text
证据层：ADP/time/ToA/AoA/LoS/PCF 等能否构成可信 radio geometry
算子层：给定图后，普通 GLR、GGLR、framelet、RED、unrolling 等怎样恢复信号
```

物理图表现好，不代表普通 GLR 是唯一最优算子；普通 GLR 有数学依据，也不代表输入物理图没有错边。导师方法的真正特殊性目前主要来自独立 `L_phys`，不是 Laplacian 形式本身。

### 3.3 CC 误差已有论文支持的具体机制

目前有原文依据的机制包括：

- CSI 噪声产生 false neighbors，并 short-circuit 流形；
- local metric 超出近邻有效范围后失去可靠性或唯一性；
- 非均匀采样削弱 global neighborhood graph；
- chart 的 scale/rotation/reflection 与结构性失真不同；
- affine transform 无法修 non-affine distortion；
- LoS 物理关系在 NLoS 下可能失效；
- 少量物理约束可能具有多解，不满足可识别性；
- NLoS 区域可能因位置 fingerprint 不足发生 embedding collapse；
- 高 SNR 与高 dissimilarity 都不能单独保证位置可辨识性。

这些是机制证据，不是领域已有的统一 failure taxonomy。当前仍不能把自定义的 fold/tear/bridge 等词写成文献标准分类。

### 3.4 训练增强、坐标对齐和 post-hoc repair 已经分清

- metric design：在生成监督距离时改变 chart 学习依据；
- training-time physical supervision：把 ToA/AoA/TRP/laser/LoS 等写进 FCF loss；
- affine alignment：训练后修 scale、rotation、reflection、translation；
- post-hoc graph repair：直接改变固定已训练 chart 的非仿射几何；
- deployable correction layer：学习一个可应用于新样本的修复映射。

当前方法属于第四类，且当前仍是 transductive。相邻 CC 论文多数属于前三类，因此既能提供机制启发，也不能被说成已经解决了当前问题。

### 3.5 普通 GLR 的能力与边界已经明确

普通 GLR 的合理性：

- `initial estimate + graph regularization` 是成熟 inverse-problem 框架；
- 当前物理图实验对 mean、P95、max、CT/TW/KS 均有强改善；
- 固定 `L` 的二次目标一次闭式解已经完成该模型下的优化。

普通 GLR 的边界：

- 它默认 graph smoothness 和低频先验；
- supporting graph 错误会被滤波传播；
- 值平滑可能产生 staircase 或压制真实变化；
- feature noise 与 structure noise 可能需要分开；
- smoothness 应验证，而不是默认；
- 迭代只有在 graph、reliability、denoiser 或约束更新时才有新意义；
- 可学习化只有在 out-of-sample、部署或更一般先验成为真实问题时才有必要。

### 3.6 当前实验与文献的关系已经能够严谨解释

欧氏 kNN refinement 比未精修好，说明普通 chart-internal smoothing 有效；它不是失败对照。

但欧氏图的 max error 变差，CT/TW 略降，而物理图显著改善 tail 和结构指标。这支持：

```text
欧氏图主要利用 Y0 内部邻域做 denoising；
物理图重用与 FCF 训练监督同源的 ADP/time radio-geometric manifold；它独立于 `Y0` 输出坐标构图，但不是统计独立的新证据。当前结果说明源流形算子比 `Y0` 自图更能控制尾部误差，尚不能据此认定具体 learned-neighborhood failure。
```

这比“换了就不灵”更准确，也符合 graphLa、robust graph filter、smoothness verification 和 CC metric 文献的共同机制。

### 3.7 下游价值不能由 chart 指标直接代替

论文 09 表明 chart 可以服务 future CSI prediction 和 sum rate，但中间还需要：

- latent dynamics；
- training-chart coverage；
- Delaunay/nearest-neighbor interpolation；
- array selection；
- prediction horizon；
- 任务专用指标。

因此，当前 CT/TW/KS 或定位误差改善不能直接推出下游通信性能提升。

### 3.8 已识别并处理弱证据与重复证据

- 论文 11 是论文 14 的早期版本，不能作为两个独立发现重复计数。
- 论文 18 的 high-frequency 结论针对 GNN node classification、heterophily 和 hidden representation，不能直接推出二维 chart 坐标需要 CLAR 或 complement graph。
- 图像、高光谱、CT、Retinex、tensor kriging 等外围论文只能提供 graph restoration 的一般边界，不能证明 channel-chart repair 的独特性。

## 4. 是否达到后续合理构思 idea 的标准

当前已经达到“可以基于相关知识合理构思优化 idea”的知识标准，原因是：

1. 已能从论文机制出发定位当前公式的被违反假设；
2. 已能判断一个候选方向是在解决真实问题，还是只做参数/模块变化；
3. 已能区分 channel-specific contribution 与通用 graph regularization；
4. 已能识别训练方法、后处理方法和部署方法的不同贡献层级；
5. 已能对候选方向做证据范围、创新重复和 fatal assumption 审查。

这不等于已经选定最终 idea。本轮按要求没有进入正式 idea 生成。

## 5. 仍然不知道的内容

深读文献后，剩余未知已经从“论文没读懂”转为“需要当前数据证据才能回答”：

- 当前 FCF 的少数大误差具体由哪种失效机制主导；
- 当前 `L_phys` 中是否存在错边、错路径或区域性低可辨识；
- chart 坐标在 `L_phys` 上的频谱是否真的暴露普通 GLR 的不足；
- 更复杂正则是否会优于当前已经很强的普通 GLR；
- post-hoc refinement 如何对未见样本部署；
- refined chart 是否会改善具体下游无线任务。

这些问题不能继续靠多读一篇相似论文直接证明，但文献已经给出了后续判断它们所需的概念和证据标准。

## 6. 当前笔记文件情况

### 原始学习材料

- `notes/paper_deep_extracts/`：34 篇深读文本，作为原文复核入口。
- `notes/paper_learning_extracts/`：早期批量学习摘录，保留作辅助检索。

### 总索引与全覆盖

- `notes/diffusion_refined_idea_optimization_writing_notes.md`：当前主线总索引；旧叙事之后已改为文献驱动的假设与机制学习。
- `notes/refinement_repair_paper_grounded_mechanism_matrix.md`：34 篇逐篇机制、可迁移知识和禁止越界 claim。
- `notes/refinement_repair_literature_coverage_audit.md`：学习深度、剩余缺口与覆盖状态审计。
- `notes/diffusion_refined_reference_papers_learning_report.md`：本总报告。

### 核心机制专题

- `notes/refinement_repair_core_mechanism_cards.md`：核心论文的压缩机制卡。
- `notes/refinement_repair_assumption_audit_learning.md`：当前公式的七假设审计。
- `notes/refinement_repair_lphys_reliability_learning.md`：物理图来源、可靠性和错边风险。
- `notes/refinement_repair_operator_prior_learning.md`：普通 GLR、GGLR、framelet、RED、unrolling 等算子边界。

### CC 与当前结果专题

- `notes/refinement_repair_cc_failure_and_boundary_learning.md`：01/03/04/06/08/14 的 CC failure evidence 与训练—后处理边界。
- `notes/refinement_repair_multisource_downstream_environment_frequency_learning.md`：07/09/11/18 的多源、下游、环境工程与高频迁移边界。
- `notes/refinement_repair_experiment_observation_assumption_mapping.md`：当前 MDS、FCF、物理图和欧氏图结果到七假设的映射。

### 背景检索笔记

- `notes/channel_charting_light_literature_search.md`
- `notes/graph_laplacian_regularization_light_literature_search.md`

这两份用于追溯文献来源和领域背景，不是当前机制判断的最高优先级。

### 历史 idea 文件

- `notes/diffusion_refined_stronger_idea_candidates.md`
- `notes/new_idea_diffusion_refined_channel_charting.md`

这两份属于较早阶段的构思产物。后续正式想 idea 时不能直接沿用，应以本报告、机制矩阵、七假设审计和两个新专题为当前知识依据重新判断。

## 7. 最终学习结论

现在对参考论文的理解已经从“哪些工作相关”推进到：

```text
为什么当前物理图 refinement 有效；
它依赖哪些尚未验证的假设；
相邻论文具体解决 pipeline 的哪个阶段；
哪些机制可迁移，哪些只是弱类比；
什么证据才足以支撑后续优化与论文 claim。
```

当前已经具备后续正式构思和严格筛选优化 idea 的知识基础。本次任务停在学习成果，不进入 idea 输出。
