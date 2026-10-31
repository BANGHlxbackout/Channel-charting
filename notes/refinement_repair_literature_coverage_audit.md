# Refinement / Repair 参考论文学习覆盖审计

更新时间：2026-06-25

本笔记继续学习参考论文，只做覆盖审计和学习状态判断，不正式提出优化 idea，不安排新实验。它回答一个更实际的问题：

```text
34 篇论文里，哪些已经被读到“能支撑后续构思”的机制层？
哪些只是作为边界/背景被吸收？
哪些如果后续要正式想 idea，还需要再补读？
```

当前可用材料：

- `C:\Users\Hype beast\Desktop\参考论文`：34 篇 PDF。
- `notes/paper_deep_extracts/`：34 篇深读摘录。
- `notes/refinement_repair_paper_grounded_mechanism_matrix.md`：34 篇总表。
- `notes/refinement_repair_core_mechanism_cards.md`：核心论文机制卡。
- `notes/refinement_repair_lphys_reliability_learning.md`：`L_phys` 可靠性专题。
- `notes/refinement_repair_operator_prior_learning.md`：图正则算子先验专题。
- `notes/refinement_repair_experiment_observation_assumption_mapping.md`：当前结果到假设映射。
- `notes/refinement_repair_cc_failure_and_boundary_learning.md`：CC 失效证据与训练—后处理边界专题。
- `notes/refinement_repair_multisource_downstream_environment_frequency_learning.md`：多源约束、下游任务、环境工程与图高频专题。

## 1. 覆盖状态总览

34 篇论文已经全部至少进入总表，说明没有完全漏掉的论文。但学习深度不均匀：

| 层级 | 含义 | 论文编号 |
|---|---|---|
| A. 机制层 | 已经能明确说明“它挑战当前 refinement 哪个假设、能迁移什么、不能推出什么” | 01, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 23, 26, 27, 28, 30, 31, 32, 33, 34 |
| B. 迁移边界层 | 已深入理解原任务机制，但对当前主线只能作谨慎边界提醒 | 18 |
| C. 外围支撑层 | 主要作为图正则、应用或风格补充；直接决定精修优化的作用较弱 | 02, 17, 19, 20, 21, 22, 24, 25, 29 |

这个分层不是论文价值高低，而是相对于当前主线2的学习深度。

## 2. A 层：已经进入机制层的论文

这些论文已经可以直接进入后续 idea 优化思考，但当前仍不正式提出 idea。

| 编号 | 论文 | 当前已经学到的机制 | 对当前公式的直接影响 |
|---:|---|---|---|
| 01 | robust global CC / time-distance | local CIR metric 只在近邻有效；global geodesic graph 依赖高且近似均匀的采样密度 | 挑战“物理图只要样本多就全局可靠”的假设 |
| 03 | model-based CC | 显式 AoA/range 模型可直接给坐标，但依赖阵列、多载波、传播模型与 SNR | 说明物理坐标证据有独立价值，也有明确模型失配边界 |
| 04 | CC theory/practice | CSI 噪声可产生 false neighbors，short-circuit 流形并扭曲 topology；CT/TW/KS 诊断对象不同 | 为“结构性邻域错误”提供直接论文依据 |
| 05 | ADP/time dissimilarity | ADP/time 提供显式 radio-geometric source manifold；在当前 pipeline 中它也用于 FCF 监督，因此不独立于训练目标 | 支撑 `L_phys` 来源，但也提醒 fused metric 不是 ground truth |
| 06 | augmenting CC with localization | 仿射对齐只能修坐标自由度，不能修 non-affine distortion；AoA/ToA 融合发生在训练阶段 | 划清 alignment、training augmentation 与 post-hoc repair |
| 07 | global-scale CC with sensor fusion | ToA/TRP 提供 absolute range，laser/ICP 提供 relative displacement；证据有独立失效范围 | 说明多源约束必须按几何作用和可靠范围区分，而非无差别 fusion |
| 08 | real-world D-MIMO coordinates | LoS 物理约束依赖传播条件；少量 AP pair 会产生不可识别歧义；仿射变换可能完全失败 | 挑战“高置信物理约束必然唯一正确”和“对齐等于修复” |
| 09 | CC-based channel prediction | chart 到 sum-rate 还经过 latent extrapolation、coverage、CSI interpolation 和 array selection | 限定 chart 指标不能直接替代下游任务证据 |
| 10 | Uncertainty-aware geodesic loss | dissimilarity/geodesic distance 是带不确定性的监督证据 | 挑战 `W_ij` 同时表达 similarity/confidence 的近似 |
| 11 | CC in smart radio environments | 静态 EMS 改变 CSI 可辨识性与 tail performance；是 14 的早期版本 | 说明 radio graph 受环境生成机制影响，并要求同源论文去重 |
| 12 | TDoA/NLoS | NLoS-corrupted measurements 需要 mask 或降权 | 挑战全局同质物理图和单一 `lambda` |
| 13 | UNILocPro | LoS/NLoS identification 决定 model-based/CC 证据如何激活 | 提醒当前后处理是轻量修复，不是完整融合框架 |
| 14 | CC non-RIS / EMS | 高 SNR 与高 dissimilarity 单独最大化都不足；NLoS 区域可能发生 embedding collapse | 说明物理图质量涉及位置可辨识性与 worst-case 区域，不是单一距离幅值 |
| 15 | graphLa+ | 初值 + graph Laplacian regularization 是成熟 inverse-problem 框架 | 支撑公式合法性，同时削弱“普通 GLR 本身创新” |
| 16 | GGLR | 普通 GLR 可能 piecewise constant / staircase；梯度先验适合连续变化信号 | 挑战 `tr(Z^T LZ)` 的值平滑先验 |
| 23 | iterated graphLa | 迭代合理性来自图/权重随估计更新 | 约束 one-shot vs iterative 的边界 |
| 26 | unrolling PnP GGLR | 优化求解可展开为可学习模块，且可学习图相似结构 | 支撑长期可学习化边界，但不是当前已完成贡献 |
| 27 | interpretable deep GLR | GLR MAP 解可展开成少参数可解释 denoiser | 说明学习化应保留解释性，不是只学 `lambda` |
| 28 | graph RED | denoiser 可显式进入 graph signal regularization | 支撑 denoiser/regularizer 统一视角 |
| 30 | LoS channel charting | channel distance 可能有 identifiability / ambiguity 问题 | 支撑“物理图可能有结构性错边” |
| 31 | physical fingerprints CC | 物理证据可以来自更深的 PCF/GBSM 表示 | 提醒 `L_phys` 的物理层次可更深，但当前未使用 |
| 32 | graph denoising framelet | feature noise 与 structure noise 可分离，多尺度处理图表示 | 挑战单一低通 GLR 足够的假设 |
| 33 | robust graph filter | supporting graph 可能有 link perturbation，图滤波会传播图错误 | 挑战固定可信图假设 |
| 34 | verifying smoothness | graph signal smoothness 应被检验 | 挑战“坐标在物理图上平滑”作为默认真理 |

目前真正支撑主线2的硬核知识主要来自这 15 篇。后续若进入 idea 构思，优先从这些论文里找机制，而不是从外围论文里拼概念。

## 3. B 层：已经理解叙事作用，但仍需补读机制细节的论文

该层论文已经读懂原任务机制，但与当前二维 chart refinement 的对象或目标不同，不能直接变成方法依据。

| 编号 | 论文 | 当前吸收程度 | 如果后续正式想 idea，需补读的问题 |
|---:|---|---|---|
| 18 | high-frequency booster | 已确认其高频机制服务 GNN 节点分类、heterophily 与 hidden representation | 若要迁移，必须先证明 chart 的真实几何信息位于 `L_phys` 高频段；当前没有该证据 |

B 层的价值主要是防止论文叙事太窄。它们告诉我们：当前问题不只是“图正则”，也是 CC 里的物理证据、真实坐标、全局几何、环境状态和下游任务问题。

## 4. C 层：外围支撑，当前不应强行拔高

这些论文目前只能做补充，不宜直接拿来支撑核心方法创新。

| 编号 | 论文 | 当前作用 | 不能强行怎么用 |
|---:|---|---|---|
| 02 | on-the-fly triplet CC | 支撑初始 chart 可快速/在线产生，后处理有位置 | 不能说当前 refinement 解决 online CC |
| 17 | Laplacian-enhanced tensor kriging | 说明 GLR 可嵌入高维估计 | 不能证明物理图 chart repair 的独特性 |
| 19 | hyperspectral GL detection | 说明 GLR 可用于异常/检测任务 | 不能把目标检测机制等同于 chart tail repair |
| 20 | Retinex gradient GL | 支撑梯度域正则的一般思想 | 不能直接说 chart 必须做 Retinex/GGLR |
| 21 | complex GL grid states | 提供物理图信号视角 | 对二维 chart 坐标直接启发弱 |
| 22 | mixed graph signal denoising | 支撑混合图信号/噪声不是单一成分 | 不能直接落成当前方法贡献 |
| 24 | time-varying graph signal | 提醒静态图可能不够 | 当前数据/实验不是时变图建模，不能声称处理时变 |
| 25 | interpretable lightweight transformer | 提醒学习化要轻量可解释 | 不直接支撑 Laplacian refinement |
| 29 | parameter-efficient CT GL | 支撑图正则与深度模型可结合且参数高效 | 对 CC refinement 主要是风格启发 |

C 层论文不是没用，而是目前证据距离主线2更远。后续写作时可以少量引用作为 broader graph restoration context，不能把它们堆成核心贡献依据。

## 5. 当前学习缺口

目前不是“没学完”，而是还有三类缺口会影响后续 idea 质量：

### 5.1 CC failure diagnosis 已有机制证据，但还没有当前数据诊断器

补读 01、03、04、06、08、14 后，已经能用论文原文支持以下机制：false neighbors 会 short-circuit 流形、local metric 有有效范围、非均匀采样削弱 global graph、仿射变换不能修 non-affine distortion、LoS 约束在 NLoS 下失效、少量物理约束可能不可识别、NLoS 区域可能发生 embedding collapse。

现在能说：

```text
chart 可能因上述具体机制产生结构性邻域或几何错误，不能只当普通坐标噪声。
```

还不能说：

```text
当前数据的 tail error 已被定位为其中某一种机制，或领域已有统一 failure taxonomy。
```

### 5.2 前端训练与后端 refinement 的边界还需更清楚

用户已经明确当前对话围绕主线2，但论文中很多工作是在训练阶段改 metric/loss/architecture。后续如果正式构思，需要非常清楚哪些机制属于前端训练，哪些才属于后处理 repair。

这个缺口对应 02、05、06、07、08、10、12、13、26、27、28。

现在能说：

```text
很多文献机制可以启发 refinement，但不能直接说它们已经是 refinement 方法。
```

经过 06、07、08 的补读，现在已经能把：

```text
training-time supervision / metric design / post-hoc graph repair / deployable correction layer
```

四者边界分清。剩余问题不是概念边界，而是未来正式构思时如何证明某个 post-hoc 机制确实解决当前数据中的失效。

### 5.3 “复杂图正则是否真的必要”还不能提前下结论

当前物理图普通 GLR 效果很强，尤其 max error 明显下降。因此 GGLR、framelet、RED、unrolling 不能被当作必需升级。它们现在只说明普通 GLR 的理论边界。

这个缺口对应 16、18、20、22、26、27、28、32、34。

现在能说：

```text
普通 GLR 有明确边界。
```

还不能说：

```text
当前实验已经证明需要 GGLR/framelet/RED。
```

## 6. 当前学习是否足够支撑后续想优化 idea

按“能否合理想优化 idea”的标准，目前已经具备三条基础：

1. 能把导师原公式拆成多个假设，而不是只看成 Laplacian smoothing。
2. 能把物理图证据和图信号恢复算子分开讨论。
3. 能判断某个优化方向是否有论文机制依据，避免随便调参。

但还没有达到“可以直接定最终 idea”的程度，因为还缺少：

- 将论文机制变成针对当前 chart 的可计算诊断；
- 哪个假设在当前结果中最值得被优化的证据排序；
- 将外围图恢复论文与 CC 特有几何问题进一步交叉核对，防止弱迁移。

因此当前学习状态应表述为：

```text
已经具备构思优化 idea 的机制底座，并已划清主要训练/后处理边界；仍需把论文机制与当前 chart 的可计算诊断对应起来，才能进入正式 idea 选择。
```

这仍然不等于提出 idea。

## 7. 后续继续学习时的优先顺序

01、03、04、06、07、08、09、11、14、18 的 CC/边界论文补读已完成。继续学习而不正式想 idea 时，不再按“还没读到”推进，而应做跨论文综合核对：

1. 检查 34 篇中是否存在同源版本、重复证据或任务对象不一致。
2. 把 CC 侧失效机制与 graph restoration 侧算子假设逐项对应。
3. 对外围论文只保留能被当前问题严格迁移的机制，删除概念性类比。
4. 最后形成“已学到什么、仍不知道什么、哪些笔记可作为后续构思依据”的总学习报告。
