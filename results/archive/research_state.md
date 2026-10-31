# Research State

## 版本信息

- Version：v2.0
- Updated：2026-09-02
- 当前阶段：**Phase 2C — Empirical Failure Diagnosis**
- Phase 1 — Baseline Audit：已完成
- Phase 2A — Channel Charting-specific Literature Audit：已完成
- Phase 2B — Cross-Domain Prior Art / Failure-Mode Audit：已完成
- 当前暂定目标期刊：**IEEE Transactions on Communications (TCOM)**
- 最终论文方向：**尚未确定**

> 本文件是当前项目的 research consensus / single source of truth。  
> 只保存已经确认的 pipeline facts、阶段性 literature conclusions、仍未验证的 hypotheses、当前候选 research problems、优先级与下一步任务。  
> 不作为 raw experiment log，也不记录未经保留配置与数值支撑的历史实验结果。

---

# 1. 当前研究状态

当前项目已经完成 baseline audit，以及两轮 literature audit。

目前已经明确：

1. 当前导师 baseline 的 **Stage 1 / Stage 2 ownership**；
2. 当前 Stage 2 的数学本质是 generic graph-Laplacian / Tikhonov smoothing；
3. Channel Charting literature 中，graph / geodesic / Laplacian 在 chart construction 和 FCF training 阶段已有充分 prior art；
4. 在本轮检索范围内，`existing Y_0 → independent intrinsic-geometry refinement → Z` 的 **post-hoc refinement** literature 明显比 construction / training / alignment / out-of-sample 更稀疏；
5. 但仅仅把成熟 graph smoothing 方法迁移到 Channel Charting，并不足以自动形成 TCOM-level contribution；
6. H3 / H4 在 generic literature 中已经高度成熟，直接迁移 novelty risk 很高；
7. 当前最值得优先验证的是 **H1 → H2**；
8. **Candidate B — out-of-sample / inductive post-hoc refinement** 继续保留，但当前 Phase 2C 暂不处理。

当前下一步不是设计最终 refinement algorithm，而是先验证：

> 当前 `D_geo` 中是否真实存在 harmful structural errors，以及当同一个 `D_geo` 同时驱动 Stage 1 和 Stage 2 时，Stage 2 是否具备纠正这些 errors 的能力。

---

# 2. 已确认的 Baseline Pipeline

## 2.1 Stage 1 — Published / inherited Channel Chart generation

Stage 1 主要继承 Stephan et al. 2024 的已有方法，不应被视为导师 baseline 的新增 contribution。

Pipeline：

```text
CSI
→ ADP dissimilarity
→ timestamp-aided fusion
→ kNN graph
→ geodesic distance D_geo
→ Siamese forward charting function
→ initial Channel Chart Y_0
```

关键映射：

```text
Y_0 = g_theta(H)
```

Siamese FCF 在形式上是 parametric 的，可以将 CSI 映射到 2-D initial chart。

需要保持严格表述：

> Parametric FCF 在形式上具备 out-of-sample mapping 所需的映射形式，但当前已有实验 protocol 本身尚未通过严格 held-out test 证明 inductive generalization。

---

## 2.2 Stage 2 — Advisor-provided post-processing baseline

导师 baseline 从 initial Channel Chart `Y_0` 已经生成之后开始。

输入：

```text
Y_0 + D_geo
```

处理流程：

```text
D_geo
→ geodesic-neighbor graph
→ locally scaled edge weights W
→ graph Laplacian L
→ Laplacian / Tikhonov smoothing
→ refined chart Z
```

当前 refinement 近似求解：

```text
(I + λL) Z = Y_0
```

当前记录配置：

```text
graph_k = 128
λ = 0.15
iterations = 200
```

因此，导师当前 baseline 应理解为：

> 一个 **embedding-space post-hoc graph refinement baseline**，而不是一套完整的 CSI-to-chart 新算法。

---

## 2.3 Stage 1–Stage 2 coupling

Stage 1 和 Stage 2 并不独立。

同一份 geodesic information 被使用了两次：

```text
D_geo → Siamese target → Y_0
D_geo → refinement graph → Z
```

**已经确认：**

> `D_geo` 同时参与 Stage 1 chart learning 和 Stage 2 refinement graph construction。

**尚未确认：**

> 这种 same-source coupling 是否会在当前 DICHASUS experiment 中导致 structural error preservation、amplification 或其他形式的 error reinforcement。

---

# 3. 已确认的实验事实

在当前 transductive experimental protocol 下，已有记录结果为：

| Method | CT ↑ | TW ↑ | KS ↓ | Mean error ↓ | Median ↓ | P95 ↓ | Max ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADP/time MDS | 0.9917 | 0.9934 | 0.0907 | 0.458 m | 0.385 m | 0.961 m | 8.897 m |
| Siamese `Y_0` | 0.9957 | 0.9960 | 0.0841 | 0.409 m | 0.371 m | 0.854 m | 7.986 m |
| Refined `Z` | 0.9982 | 0.9982 | 0.0651 | 0.319 m | 0.293 m | 0.635 m | 1.142 m |

当前数据能够支持的结论仅限于：

> 在同一 transductive dataset 和当前 evaluation protocol 下，Stage 2 refinement 相比未经 refinement 的 Siamese chart `Y_0`，其记录的 CT、TW、KS 和 affine-aligned error 均有所改善。

当前数据**不能**据此证明：

- held-out localization generalization；
- unseen-CSI refinement improvement；
- inductive deployment capability。

---

# 4. 已确认的架构 / 实验限制

## 4.1 Stage 2 当前是 transductive 的

Stage 1 提供：

```text
H → Y_0
```

但 Stage 2 需要 dataset-level geodesic matrix 和 refinement graph。

当前完整 pipeline 没有定义单个 unseen CSI sample 的：

```text
H_new → Y_new → Z_new
```

路径，除非重新构建或扩展 graph。

这是当前 architecture 中已确认的 limitation。

---

## 4.2 当前 evaluation protocol 还不足以支持严格 generalization claim

已确认的问题包括：

- geodesic construction 使用完整 selected dataset；
- Siamese pair sampling 使用同一 dataset；
- refinement 使用同一 dataset-level graph；
- 当前没有严格 held-out unseen test protocol；
- 当前最佳 `graph_k` 和 `λ` 是根据 ground-truth-based evaluation metrics 选择的；
- affine alignment 在同一批 samples 上完成 fitting 和 evaluation；
- CT/TW/KS 使用 random evaluation subset；
- MDS 与 neural/refinement results 没有使用完全一致的 controlled sample protocol；
- 当前已有 refinement 结果来自一个记录的 seed。

这些问题属于 **evaluation validity / experimental rigor**，后续必须修复，但它们本身不自动构成算法创新。

---

# 5. Phase 2A — Channel Charting-specific Literature Conclusions

Phase 2A 已完成。

## 5.1 已支持的 literature position

本轮检索范围内，Channel Charting literature 已经形成较成熟的研究簇：

- dissimilarity / metric design；
- graph / geodesic-based chart construction；
- autoencoder / Siamese / triplet FCF learning；
- global geometry / weak supervision；
- real-world coordinate alignment；
- multipoint / multi-site alignment；
- out-of-sample / parametric FCF；
- streaming / continual Channel Charting。

相比之下：

```text
existing Channel Chart Y_0
→ independent intrinsic-geometry refinement
→ refined chart Z
```

这一 **Category 4 — post-hoc geometry refinement** 的 literature density 明显较低。

---

## 5.2 与 advisor baseline 高度同构的 prior work

当前检索范围内暂未发现明确实现：

```text
existing Y_0
+
CSI-derived / geodesic D_geo
→ independent graph construction
→ graph Laplacian / Tikhonov / diffusion
→ refined embedding Z
```

的 Channel Charting prior work。

准确表述只能是：

> **“在本轮检索范围内暂未发现高度同构的方法。”**

不能升级为：

- no prior work；
- unexplored；
- first；
- first-ever。

---

## 5.3 Phase 2A 的核心判断

当前可以接受：

> **Post-hoc intrinsic Channel Chart refinement 是一个具有清晰 pipeline boundary、值得继续调查的 candidate problem formulation。**

但不能接受：

> **标准 Laplacian/Tikhonov post-processing 本身已经构成 TCOM-level contribution。**

原因是 graph Laplacian、Tikhonov regularization、graph smoothing 和 manifold denoising 都属于成熟的通用方法。

---

## 5.4 Out-of-sample literature position

Out-of-sample Channel Charting 本身已有明确 prior work，包括：

- non-parametric chart 的 out-of-sample extension；
- parametric FCF；
- streaming / continual Channel Charting。

但当前检索范围内暂未发现明确统一：

```text
Training data:
Y_0 → independent intrinsic refinement → Z

Unseen data:
H_new → Y_new → Z_new
```

的成熟 Channel Charting formulation。

因此：

> **Candidate B — out-of-sample / inductive post-hoc refinement 继续保留。**

这仍然只是 candidate research problem，不是 novelty claim。

---

# 6. Phase 2B — Cross-Domain Prior Art Conclusions

Phase 2B 已完成。

核心结论：

> Cross-domain literature 并没有否定 post-hoc Channel Chart refinement 作为 research problem 的价值，但明显削弱了任何仅依赖 generic graph smoothing、adaptive neighborhood、edge-preserving filtering 或 standard graph repair 的 novelty claim。

---

## 6.1 H1 — False edges / geodesic shortcuts

### Hypothesis

错误 local graph edge 可能形成 shortest-path shortcut，从而导致大量 `D_geo` 被系统性低估或扭曲。

### Generic literature status

- manifold shortcut；
- noisy neighborhood graph；
- shortcut detection；
- adaptive neighborhood；
- robust manifold distance；

都已有成熟或活跃 prior art。

### 当前状态

**Generic failure：已高度确认。**

**Current DICHASUS failure：未确认。**

**Direct-transfer novelty risk：High。**

### 当前作用

H1 目前不适合作为“我们发现的新 generic failure”。

它更适合成为：

> **一个 Channel-Chart-specific empirical diagnostic：当前 wireless-derived `D_geo` 中到底有没有 harmful structural shortcuts？**

---

## 6.2 H2 — Same-source geometric evidence / correlated information

当前结构：

```text
D_geo → Siamese target → Y_0
D_geo → Stage-2 graph → Z
```

### 已确认事实

same-source coupling 确实存在。

### 尚未确认

目前没有证据证明：

```text
same D_geo used twice
⇒ error reinforcement
```

也不能直接称其为 confirmation bias。

### 更准确的当前 formulation

Stage 2：

```text
Z = (I + λL)^(-1) Y_0
```

只依据由 `D_geo` 构造的同一 graph `L` 判断哪些 coordinate variation 是 high / low graph frequency。

因此当前可以提出的有限结构性判断是：

> 如果某种 structural distortion 本身来自 `D_geo`，且在该 `D_geo` graph 上表现为 low graph-frequency structure，那么 current Tikhonov refinement 缺少独立于 `D_geo` 的 evidence 去识别并纠正它。

这仍然是 **mechanistic hypothesis / structural limitation**，不是已经证实的 error reinforcement。

### Generic literature status

- confirmation bias；
- multi-view learning；
- consistency / inconsistency modeling；
- unreliable graph regularization；

均存在成熟邻近研究。

但当前未找到完全同构于：

```text
same metric
→ train representation
→ construct regularizer
→ refine same representation
```

的成熟统一 formulation。

### 当前状态

**Exact same-source formulation maturity：较低。**

**Current structural coupling：Confirmed。**

**Current error reinforcement：Unverified。**

**Direct-transfer novelty risk：Medium。**

### 当前优先级

**High。**

但 H2 应在 H1 diagnostic 获得 evidence 后重点验证。

---

## 6.3 H3 — Fixed-k / non-uniform sampling density

### Hypothesis

fixed-k adjacency 在 sampling density 不均匀时可能产生 inappropriate neighborhood topology。

### Generic literature status

- local scaling；
- adaptive-k；
- mutual-kNN；
- CkNN；
- variable neighborhood；
- adaptive manifold graph；

均已经高度成熟。

### Baseline coverage

当前 Stage 2 已使用 local-scale weighting：

```text
W_ij = exp(-d_ij^2 / (σ_i σ_j))
```

因此 baseline 已经部分考虑 local scale heterogeneity。

但：

```text
weight adaptation ≠ topology adaptation
```

fixed-k adjacency degree 仍然存在。

### 当前状态

**Generic literature：Highly mature。**

**Baseline coverage：Partially covered。**

**Current DICHASUS density failure：Unverified。**

**Direct-transfer novelty risk：Very High。**

### 当前优先级

**Low。**

除非后续出现明显的 Channel-Chart-specific density phenomenon，否则不作为主论文问题。

---

## 6.4 H4 — Laplacian over-smoothing / geometry destruction

### Hypothesis

quadratic Laplacian/Tikhonov smoothing 可能在去除噪声时同时削弱真实 local geometry。

### Generic literature status

该问题已经高度成熟：

- manifold denoising；
- graph total variation；
- graph trend filtering；
- vector-valued graph signal recovery；
- anisotropic diffusion；
- bilateral graph filtering；
- node-adaptive regularization；
- nonconvex edge-preserving regularization。

当前 advisor operator：

```text
Z = (I + λL)^(-1) Y_0
```

与经典 manifold denoising / graph Tikhonov operator 在数学结构上高度接近。

### 当前状态

**Generic problem：Highly mature。**

**Current DICHASUS over-smoothing：Unverified。**

**Straightforward-transfer novelty risk：Very High。**

### 当前优先级

**Low。**

以下直接替换目前均不应作为主 contribution：

```text
Laplacian → graph TV
Laplacian → graph trend filtering
Laplacian → anisotropic diffusion
Laplacian → bilateral graph filtering
uniform λ → node-adaptive λ
```

这些方法以后可以作为 baseline / ablation / mechanism toolbox。

---

# 7. 当前 Hypothesis Priority

当前优先级调整为：

| Hypothesis | Generic literature maturity | 当前 pipeline fact | 当前 failure 已验证？ | Direct-transfer novelty risk | Priority |
|---|---|---|---|---|---|
| **H1 — false edge / shortcut** | Mature / Active | shortest-path `D_geo` 存在 | No | High | **High diagnostic** |
| **H2 — same-source evidence** | exact formulation 较少；analogies mature | `D_geo` used twice 已确认 | No | Medium | **Highest research interest** |
| **H3 — fixed-k / density** | Highly mature | fixed-k + local scaling 已确认 | No | Very High | Low |
| **H4 — over-smoothing** | Highly mature | quadratic Laplacian confirmed | No | Very High | Low |

当前建议的因果顺序：

```text
H1:
Is the geometric evidence wrong?

        ↓ if yes

H2:
Can a refinement driven by the same evidence
actually correct that error?
```

---

# 8. 当前 Candidate Research Problems

以下仍然只是 candidate research problems。

## Candidate A — Post-hoc Channel Chart refinement as a research setting

Research question：

> 当完整的 Channel Chart `Y_0` 已经学习完成后，能否在不重新训练原始 FCF 的前提下，通过独立 post-hoc stage 改善其 intrinsic geometry？

### 当前状态

- Phase 2A 支持其作为一个相对稀疏、边界清晰的 candidate problem formulation；
- Phase 2B 证明 generic graph smoothing 本身不足以形成 method novelty；
- 是否具有最终 TCOM-level value，取决于 Phase 2C 能否发现真实 Channel-Chart-specific failure / need。

**Status：保留，尚未选定。**

---

## Candidate B — Out-of-sample / inductive post-hoc refinement

Research question：

> Post-hoc refined Channel Chart 能否同时保留或恢复对 unseen CSI 的 mapping capability？

目标形式：

```text
Training data:
Y_0 → Z

Unseen sample:
H_new → Y_new → Z_new
```

### 当前状态

- 当前 Stage 2 architecture 确实缺少该 path；
- Phase 2A 暂未发现成熟的 Category 4 + Category 5 unified formulation；
- Phase 2B 没有否定这一 candidate；
- 当前 Phase 2C 暂不验证它。

**Status：保留，具有较高潜力，但尚未选定。**

---

## Candidate C — Reliability-aware / topology-aware refinement

Research question：

> Refinement 能否区分可靠与不可靠的 wireless-derived manifold relations，而不是默认 `D_geo` graph 完全可信？

### 当前状态

该 candidate 现在与 H1 / H2 强相关。

只有当 Phase 2C 证明：

- `D_geo` 中确实存在 harmful structural errors；
- same-source refinement 对这些 errors 的 correction 能力有限；

它才可能升级为强 candidate。

**Status：Conditional / waiting for H1-H2 evidence。**

---

## Candidate D — Geometry-preserving refinement

Research question：

> Refinement 能否减少 distortion，同时避免破坏真实 local geometry？

### 当前状态

generic graph literature 已高度成熟，直接采用 TV / trend filtering / anisotropic smoothing 等方法 novelty risk 极高。

**Status：明显降级。**

---

# 9. 当前禁止提前宣称的结论（Do Not Claim Yet）

不要宣称：

- “We are the first to propose graph refinement for Channel Charting.”
- “Post-hoc Channel Chart refinement has not been studied before.”
- “Existing Channel Chart refinement methods are all transductive.”
- “No prior work exists for Category 4.”
- “The current DICHASUS experiment contains harmful geodesic shortcuts.”
- “Stage 2 reinforces Stage 1 errors.”
- “Using the same `D_geo` twice causes confirmation bias.”
- “The current best refinement causes over-smoothing.”
- “Fixed-k currently produces incorrect topology.”
- “0.319 m is held-out localization accuracy.”
- “The current refinement improves unseen-CSI localization.”
- “The current method has demonstrated inductive generalization.”
- “Laplacian / graph diffusion itself is the method novelty.”
- “Replacing Laplacian with graph TV / adaptive-k / anisotropic diffusion is sufficient novelty.”

在证据充分之前，应使用：

```text
current pipeline limitation
candidate failure mode
working hypothesis
literature-supported analogy
to be empirically verified
in the current search scope
```

等谨慎表述。

---

# 10. 当前研究决策

当前仍然**不放弃 Track A**。

但 Track A 已经从：

```text
find a stronger graph refinement algorithm
```

收缩为：

```text
diagnose whether a Channel-Chart-specific
post-hoc refinement problem actually exists
```

当前不进入：

- GNN refinement；
- Transformer / attention refinement；
- graph rewiring model；
- graph TV / trend filtering direct replacement；
- adaptive-k direct replacement；
- new diffusion design；
- refinement distillation iteration；
- large-scale parameter search。

当前 priority：

```text
Phase 2C:
H1 diagnostic
→ if supported
H2 diagnostic
→ then decide whether a genuinely
Channel-Chart-specific refinement problem exists
```

Candidate B 同时保留，等待后续 Research Direction Decision。

---

# 11. 当前阶段 — Phase 2C Empirical Failure Diagnosis

当前正在设计 diagnostic experiments。

## 11.1 H1 核心问题

> 当前 `D_geo` 中是否真实存在少量 harmful local edges / shortcuts，对大量 global geodesic relations 产生 disproportionate distortion？

需要围绕：

- local edge physical-distance mismatch；
- shortest-path usage / edge betweenness；
- global `D_geo` vs physical-distance distortion；
- suspicious-edge impact；
- targeted removal vs random removal；

进行诊断。

Ground-truth UE positions 可以用于 **offline diagnosis**，但不能成为未来 deployable refinement algorithm 的必要输入。

---

## 11.2 H2 核心问题

仅当 H1 获得一定 evidence 后重点执行：

> 当 `D_geo` 明显错误时，`Y_0` 是否继承同类 distortion；而使用同一 `D_geo` 的 Stage 2 是 correcting、preserving，还是 amplifying 这些 errors？

需要研究：

```text
e_geo(i,j)
→ e_Y0(i,j)
→ e_Z(i,j)
```

并进行 pair-level、region-level 和 high-error-quantile analysis。

不能预设：

```text
same D_geo used twice ⇒ confirmation bias
```

只有当实验显示 error 进一步恶化时，才可以讨论 amplification / reinforcement。

---

# 12. Phase 2C 的 Go / No-Go 逻辑

当前预设决策逻辑：

```text
H1 unsupported
→ H1/H2 reliability route 降级
→ 优先重新评估 Candidate B / Track B

H1 supported, H2 unsupported
→ structural D_geo error 存在
→ 但 current Stage 2 仍能有效纠正
→ same-source limitation 不足以成为核心 story
→ 重新评估 Candidate B 或其他 CC-specific problem

H1 supported, H2 supported
→ 建立 Channel-Chart-specific structural failure evidence
→ 进入 03 Research Direction Decision
→ 比较 reliability-aware refinement / Candidate B / possible combined formulation
```

这里的 Go / No-Go criteria 仍需根据 Phase 2C 最终 diagnostic design 和实际实验结果进一步细化。

---

# 13. Target Journal Positioning

当前暂定目标：

> **IEEE Transactions on Communications (TCOM)**

TCOM 当前作为以下方面的研究标尺：

- communication relevance；
- technical contribution；
- novelty strength；
- experimental rigor；
- clear problem formulation。

当前不要求“颠覆性科研成果”，但不能将：

```text
generic mature method
+ Channel Chart application
+ metric improvement
```

直接视为足够 contribution。

更理想的论文结构应接近：

```text
Channel-Chart-specific problem
→ evidence that the problem actually exists
→ limitation of current method
→ non-trivial mechanism
→ controlled experiments / ablations / robustness
```

最终投稿期刊可根据最终 research direction 调整。

---

# 14. Research State 更新规则

本文件只在：

- 一个 research phase 完成；
- 当前 research consensus 发生实质变化；
- candidate priority / direction decision 发生变化；

之后更新。

本文件**不是**：

- raw experiment log；
- brainstorming dump；
- 所有 idea 的收集文件；
- 长对话逐字 summary；
- 失败实验的完整历史。

实验历史应单独记录在未来的：

```text
experiment_log.md
```

中。

开始新的研究阶段或新对话时，应优先阅读：

1. `Research_Context.md`
2. `research_state.md`

如果问题涉及已经执行过的实验，再读取：

3. `experiment_log.md`
