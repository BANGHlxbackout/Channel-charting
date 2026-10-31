# Research State

## 版本信息

- Version：v2.1
- Updated：2026-09-03
- Current phase：**Phase 03 — Research Direction Decision**
- Current substage：**Entering Round 1 — Candidate Reconstruction & Decision Framework**
- Final research direction：**Not selected**

### Phase status

- Phase 1 — Baseline Audit：**Complete**
- Phase 2A — Channel-Charting-specific Literature Audit：**Complete**
- Phase 2B — Cross-Domain Prior-Art / Failure-Mode Audit：**Complete**
- Phase 2C — Empirical Failure Diagnosis：**Complete enough for research-direction decision**
- Phase 03 — Research Direction Decision：**Entering Round 1**

> 本版本是进入 Phase 03 前的 **transitional state**。  
> 它同步 Phase 2C 已批准的稳定研究共识，但**不选择最终论文方向、不设计新模型、不启动新实验**。

---

## 1. 当前研究状态

当前项目已经完成：

1. baseline ownership / pipeline audit；
2. Channel Charting-specific literature positioning；
3. cross-domain graph / manifold prior-art audit；
4. 一轮 falsification-oriented empirical failure diagnosis。

当前研究任务已经从：

```text
Research Gap Exploration
```

转入：

```text
03 Research Direction Decision
```

当前不再继续 open-ended Phase 2C diagnostics。

当前 Round 1 的任务是：

```text
Candidate Reconstruction
→ Unified Decision Framework
→ Weak-Direction Pruning
→ retain at most 2–3 serious directions
```

当前仍然**没有确定最终论文方向**。

---

## 2. 已确认的 Baseline Ownership / Pipeline Facts

### 2.1 Stage 1 — Published / inherited Channel Chart generation

Stage 1 主要继承已有 Channel Charting 方法，尤其是 ADP / timestamp-aided dissimilarity、geodesic distance 与 Siamese FCF 这一类 published pipeline。

当前实际 Stage-1 pipeline：

```text
CSI
→ ADP + timestamp fused dissimilarity
→ 20-NN graph
→ shortest-path geodesic D_geo
→ Siamese FCF
→ initial Channel Chart Y_0
```

关键映射：

```text
Y_0 = g_theta(H)
```

Siamese FCF 在形式上是 parametric 的，可定义 CSI 到 initial chart coordinate 的映射。

Stage 1 不应被描述成 advisor post-processing baseline 的新增 contribution。

---

### 2.2 Stage 2 — Advisor-provided post-hoc baseline

Advisor baseline 从已经生成的 `Y_0` 开始：

```text
Y_0 + D_geo
→ geodesic-neighbor graph
→ locally scaled weights
→ graph Laplacian
→ Laplacian / Tikhonov refinement
→ Z
```

当前记录配置：

```text
Stage-2 graph_k = 128
lambda = 0.15
iterations = 200
```

近似求解：

```text
(I + lambda L) Z = Y_0
```

因此，advisor baseline 应被理解为：

> **embedding-space post-hoc graph refinement baseline**

而不是一套完整的新 `CSI → Z` Channel Charting algorithm。

---

### 2.3 Stage 1–Stage 2 same-source coupling

同一份 `D_geo` 被使用两次：

```text
D_geo → Siamese target → Y_0
D_geo → Stage-2 refinement graph → Z
```

这是 **confirmed pipeline fact**。

当前没有 empirical evidence 证明该 coupling 会导致：

- confirmation bias；
- error reinforcement；
- Stage-1 error preservation；
- Stage-1 error amplification。

---

## 3. 已确认的 Baseline Experimental Facts

在当前已有的 **transductive experimental protocol** 下，记录结果为：

| Method | CT ↑ | TW ↑ | KS ↓ | Mean error ↓ | Median ↓ | P95 ↓ | Max ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADP/time MDS | 0.9917 | 0.9934 | 0.0907 | 0.458 m | 0.385 m | 0.961 m | 8.897 m |
| Siamese `Y_0` | 0.9957 | 0.9960 | 0.0841 | 0.409 m | 0.371 m | 0.854 m | 7.986 m |
| Refined `Z` | 0.9982 | 0.9982 | 0.0651 | 0.319 m | 0.293 m | 0.635 m | 1.142 m |

当前这些结果只支持：

> 在同一 transductive dataset 和当前 evaluation protocol 下，Stage 2 refinement 相比未经 refinement 的 Siamese chart `Y_0`，记录的 CT、TW、KS 与 affine-aligned error 均有所改善。

这些结果**不能**证明 held-out / unseen-CSI localization generalization。

---

## 4. 已确认的 Architecture / Evaluation Limitations

### 4.1 Stage 2 当前是 transductive 的

Stage 1 形式上提供：

```text
H_new → Y_new
```

但当前 Stage 2 依赖 dataset-level `D_geo` 与 refinement graph。

当前完整 pipeline 没有定义单个 unseen CSI sample 的：

```text
H_new → Y_new → Z_new
```

deployable path，除非重新构建 / 扩展 graph 或另行定义 out-of-sample mechanism。

这是当前 architecture-level confirmed limitation。

需要保持区分：

> Parametric FCF 具备 out-of-sample mapping 的函数形式；当前实验尚未严格证明其 held-out inductive generalization，更没有证明 post-hoc refined chart 的 unseen-CSI generalization。

---

### 4.2 当前 evaluation protocol 不足以支持严格 generalization claim

已确认 caveats 包括：

- geodesic construction 使用完整 selected dataset；
- Siamese pair sampling 使用同一 dataset；
- refinement 使用同一 dataset-level graph；
- 没有严格 held-out unseen test protocol；
- 当前最佳 `graph_k` 和 `lambda` 使用 ground-truth-based evaluation metrics 选择；
- affine alignment 在同一批 samples 上 fitting 与 evaluation；
- CT / TW / KS 使用 random evaluation subset；
- MDS 与 neural / refinement result 尚未使用完全一致的 controlled sample protocol；
- 当前主要 refinement result 来自一个记录 seed。

因此当前 baseline result 不能用于强 generalization / deployment claim。

---

## 5. Phase 2A — Channel Charting Literature Conclusions

Phase 2A 已完成 Channel Charting-specific literature audit。

### 5.1 已确认的 literature position

已有 Channel Charting 文献中：

- graph / geodesic / manifold methods 已大量用于 **chart construction**；
- graph / physical / topology constraints 已用于 **FCF training / regularization**；
- chart 生成后的 affine / Procrustes / map matching / real-world coordinate mapping 已形成 **alignment / calibration** 类工作；
- parametric FCF 与 non-parametric extension 已形成 **out-of-sample** 类工作。

因此：

> graph、geodesic、Laplacian 等技术本身并不是 Channel Charting 中的新元素。

---

### 5.2 Post-hoc intrinsic refinement 的 literature density 较低

Phase 2A 重点区分了：

```text
Category 1 — Chart construction
Category 2 — Training regularization
Category 3 — Alignment / coordinate transformation
Category 4 — Post-hoc intrinsic geometry refinement
Category 5 — Out-of-sample extension
```

本轮检索范围内：

> `existing Y_0 → independent intrinsic-geometry refinement → Z` 的 Category 4 literature density 明显较低。

同时：

> 在 Phase 2A 检索范围内，暂未发现与  
> `Y_0 + D_geo → graph → Laplacian/Tikhonov → Z`  
> 高度同构的 Channel Charting method。

这只是 **problem-level gap signal**，不是 “no prior work” 或 “first” claim。

---

### 5.3 Post-hoc refinement + out-of-sample 的交集仍是 gap signal

已有工作分别研究：

- parametric FCF 的 unseen CSI mapping；
- non-parametric chart 的 out-of-sample extension；
- post-training alignment / coordinate transformation。

但 Phase 2A 检索范围内：

> 暂未发现成熟的 `post-hoc intrinsic refinement + out-of-sample extension` 统一 setting。

这使 Candidate B 值得继续比较，但仍不构成 publication novelty proof。

---

## 6. Phase 2B — Cross-Domain Prior-Art Conclusions

Phase 2B 已完成 graph signal processing、graph learning、manifold learning、embedding refinement 等邻近领域 audit。

### 6.1 H1 generic problem

False edges / shortcut edges / noisy graph topology / robust neighborhood construction 在 generic manifold / graph literature 中已经有成熟 prior art。

因此：

> “检测坏边并删除 / rewiring” 本身具有较高 straightforward-transfer novelty risk。

---

### 6.2 H2 exact setting

Pseudo-label confirmation bias、unreliable graph regularization、multi-view / co-regularization 等邻近理论成熟。

但是：

> “同一个 `D_geo` 同时塑造 `Y_0` 并再次作为 Stage-2 refinement authority，从而造成 error preservation / reinforcement”

这一 exact formulation 并未被 generic literature 直接证明。

因此 H2 只能被视为：

```text
confirmed same-source coupling
+
unresolved empirical consequence
```

---

### 6.3 H3 generic literature maturity

Fixed-k / non-uniform sampling density、local scaling、adaptive neighborhood、continuous kNN、mutual kNN 等 generic literature 高度成熟。

当前 Stage 2 已使用 locally scaled weights，因此已部分覆盖 local-scale heterogeneity。

直接把：

```text
fixed-k → adaptive-k / mutual-kNN / CkNN
```

作为主要 contribution，novelty risk 很高。

---

### 6.4 H4 generic literature maturity

Laplacian / Tikhonov denoising、manifold diffusion、graph TV、graph trend filtering、anisotropic / edge-preserving smoothing、adaptive regularization 等已经形成成熟方法体系。

当前 Stage-2 operator：

```text
(I + lambda L)^(-1) Y_0
```

在数学结构上属于成熟的 generic graph / manifold smoothing family。

因此：

> 将 Laplacian 简单替换成 TV、anisotropic diffusion、bilateral filtering、trend filtering 等，不足以自动形成 TCOM-level novelty。

---

### 6.5 Phase 2B 总体 implication

Generic graph / manifold mechanism transfer 本身不足以支撑论文 contribution。

一个值得保留的方向必须形成：

```text
Channel-Chart-specific Problem
→ Evidence
→ Closest Work
→ Gap
→ non-trivial Mechanism
→ Minimum Experiment / Falsification
→ Controlled Evidence
```

---

## 7. Phase 2C — Frozen Empirical Status

Phase 2C core empirical diagnosis 已完成，并足以进入 research-direction decision。

### 7.1 Stage-1 graph reconstruction validity

Phase 2C 重建了实际 Stage-1 20-NN graph，并对 sampled shortest-path distances 与保存的 `D_geo` 进行验证。

稳定事实：

```text
N = 20,997
Stage-1 k = 20
directed nnz = 419,940
connected components = 1
```

对 32 sampled sources：

```text
validated source-target distances = 671,904
exact float32 match fraction vs saved D_geo = 1.0
max absolute difference = 0.0
mismatch count = 0
```

因此该 reconstructed graph 对 Phase 2C diagnosis 被视为有效。

---

### 7.2 H1 frozen decision

正式 frozen status：

```text
H1_counterfactual_supported = false
H1_supported = null
```

准确含义：

> **The pre-specified clean / low-collateral-damage harmful-edge isolation criterion was not satisfied.**

不得解释为：

- no edge effect；
- H1 fully falsified；
- H1 fully proven。

---

### 7.3 Phase 2C key empirical observation

Phase 2C 观察到：

> selected-edge deletion 对部分 severe geodesic underestimation 具有 localized causal intervention effects，同时也造成 collateral degradation。

因此，当前 evidence 支持的是：

```text
selected-edge deletion
→ repairs some severe relations
→ degrades some other relations
```

而不是：

```text
selected edges = clean binary false edges
```

---

### 7.4 Reviewer boundary on collateral degradation

Collateral degradation **不能**证明：

- selected edges intrinsically “useful”；
- selected edges physically correct；
- 每条 selected edge 本身同时具有 individual good / bad semantics。

仍然可能的解释包括：

- shortest-path rerouting；
- graph redundancy；
- alternative path structure；
- distributed dependence among multiple edges；
- 其他未被当前 intervention isolated 的 graph-level effects。

因此：

```text
“these edges carry useful geometry”
```

不是 confirmed fact，只能作为 possible interpretation。

---

### 7.5 Connectivity is not the explanation for the observed collateral damage

Phase 2C 所有 modified graphs 保持：

```text
connected components = 1
giant component ratio = 1.0
unreachable fresh-test pairs = 0
```

因此 collateral degradation 不能归因于 graph disconnection。

---

### 7.6 No wireless causal attribution was established

Selected-edge population 只有 modest timestamp-branch enrichment。

当前没有建立：

- timestamp fusion 是 harmful mechanism；
- ADP branch 是 harmful mechanism；
- NLoS 是 causal mechanism；
- 任何明确的 Channel-Chart-specific wireless causal source。

---

## 8. Current Hypothesis Status

### H1 — False local edges / geodesic shortcuts

原始 strong route：

```text
small set of clean binary harmful edges
→ remove / repair
→ clean geodesic improvement
```

在 frozen criterion 下未获支持。

当前状态：

- `H1_counterfactual_supported = false`
- `H1_supported = null`
- clean binary harmful-edge route：**Downgraded**
- Priority as primary research direction：**Low**
- Current role：**diagnostic / ablation / control**

仍可记录的 empirical observation 是：

> 少量 selected edges 对部分 severe geodesic underestimation 具有 localized causal intervention effect，但 binary deletion 同时引入 collateral degradation。

不能把 selected edges 定义为已确认的 “false edges”。

---

### H2 — Stage 1–Stage 2 same-source error preservation / reinforcement

Current status：

- **Unresolved**
- **Not executed**
- same-source `D_geo` coupling：**Confirmed pipeline fact only**
- error preservation：**Unsupported**
- error amplification：**Unsupported**
- confirmation bias：**Unsupported**

Current priority：

```text
Conditional diagnostic
```

执行条件：

> 只有当 Phase 03 最终保留一个 serious same-source / reliability-oriented direction，并且 H2 diagnostic 能够区分该方向时，才执行 H2。

因此：

> **Do NOT automatically execute H2.**

---

### H3 — Sampling-density / fixed-k sensitivity

Current priority：

```text
Low
```

原因：

- generic literature highly mature；
- current baseline 已部分使用 local scaling；
- 直接 adaptive-neighborhood transfer novelty risk 很高；
- 当前没有需要把 H3 提升为主 research direction 的 stable evidence。

---

### H4 — Laplacian over-smoothing / geometry destruction

Current priority：

```text
Low
```

原因：

- generic graph/manifold denoising literature highly mature；
- edge-preserving / TV / anisotropic / trend-filtering mechanisms 已有大量 prior art；
- straightforward replacement 很可能只是 generic method transfer；
- 当前没有理由将 H4 作为 primary paper direction。

---

## 9. Current Candidate Research Directions

当前 Candidate 均未被选为最终论文方向。

### Candidate A — Broader Post-hoc Channel Chart Refinement

Research setting：

```text
existing Channel Chart Y_0
→ independent post-hoc processing
→ refined chart Z
```

Current status：

- **Retained as a research setting**
- **Not selected**
- advisor baseline provides positive transductive evidence that post-hoc processing can change / improve chart metrics
- Category-4 literature density较低是 problem-level gap signal

但 Candidate A 若要成为论文方向，仍必须证明：

1. 存在明确 Channel-Chart-specific problem；
2. generic graph / manifold formulation 不足以解决；
3. 有 non-generic / substantially adapted mechanism；
4. 可以形成 clear falsification / minimum experiment；
5. communications relevance 足够明确。

因此：

> “post-hoc refinement literature sparse” 本身不等于足够 novelty。

---

### Candidate B — Out-of-Sample / Inductive Post-hoc Refinement

Research question：

> Post-hoc refinement 能否保留或恢复 parametric FCF 的 out-of-sample capability，使 refined chart 对 unseen CSI 具有定义明确的 mapping path？

Confirmed architecture limitation：

```text
Stage 1:
H_new → Y_new

Current Stage 2:
Y_0 + dataset-level D_geo / graph → Z

Missing:
H_new → Y_new → Z_new
```

Current status：

- Priority：**Raised / High for Phase 03 comparison**
- Evidence basis：
  - confirmed architecture limitation；
  - Phase 2A literature gap signal at Category-4 + Category-5 intersection；
  - 不依赖 H1 / H2 failure mechanism 成立；
- Empirically validated：**No**
- Final direction selected：**No**

Critical boundary：

> **architecture limitation ≠ publication novelty**

Candidate B 在 Phase 03 必须继续判断：

- closest work boundary；
- problem significance；
- communication relevance；
- non-trivial mechanism requirement；
- minimum held-out experiment；
- feasibility / workload；
- publication potential；
- falsification criterion。

---

### Candidate C — Reliability / Selective Refinement

Research question：

> 是否能够在 refinement 中区分 graph / manifold relations 的 reliability，而不是默认所有 relations 等价可信？

Current status：

- Phase 2C 后 empirical motivation：**Increased**
- Candidate type：**Still generic candidate**
- GT-free reliability signal：**Not established**
- Channel-Chart-specific wireless mechanism：**Not established**
- Novelty evidence：**Insufficient**
- Final direction selected：**No**

Phase 2C mixed intervention effects 只说明：

> binary deletion 太粗糙 / intervention effect heterogeneous

不能推出：

> reliability weighting 一定是正确 solution。

Candidate C 若要继续，必须先证明：

- 可部署、GT-free 的 reliability evidence；
- Channel-Chart-specific information source；
- generic adaptive weighting / graph learning 为什么不够；
- non-obvious method gap。

---

### Candidate D — Geometry-Preserving Generic Refinement

Research question：

> refinement 能否降低 chart distortion，同时减少 generic Laplacian over-smoothing？

Current status：

```text
Downgraded / Low priority
```

原因：

- generic literature highly mature；
- direct TV / anisotropic / bilateral / trend-filtering transfer novelty risk 很高；
- 当前缺乏 Channel-Chart-specific mechanism。

Candidate D 当前更适合作为：

```text
baseline / ablation / comparison family
```

而不是 primary paper direction。

---

## 10. TCOM Quality-Bar Positioning

Tentative target journal：

> **IEEE Transactions on Communications (TCOM)**

TCOM 在当前项目中用于约束研究质量，而不是录用保证。

一个进入最终 shortlist 的方向至少需要形成：

```text
Research Problem
→ Evidence
→ Closest Work
→ Gap
→ Research Hypothesis
→ Channel-Chart / Communications-specific Mechanism
→ Minimum Experiment / Falsification
→ Controlled Evidence
```

重点要求：

- technical contribution 不能只是 generic module transfer；
- communications relevance 必须明确；
- closest work boundary 必须清晰；
- experimental protocol 必须能够支撑所做 claim；
- held-out / deployment claim 必须有对应 held-out evidence；
- ablation 和 falsification 必须能够验证 mechanism，而不是只展示指标提高。

TCOM 是 tentative quality benchmark，不代表最终一定投稿或录用。

---

## 11. Current Do-Not-Claim Boundaries

当前禁止提前宣称：

### Literature / novelty

- “We are the first to propose graph refinement for Channel Charting.”
- “Post-hoc Channel Chart refinement has not been studied before.”
- “No prior work exists.”
- “Existing Channel Chart refinement methods are all transductive.”
- “Candidate B is novel because the architecture is currently transductive.”
- “Architecture limitation itself proves publication novelty.”

### H1 / Phase 2C

- “H1 is fully proven.”
- “H1 is fully falsified.”
- “There is no edge effect.”
- “Stage 1 contains a clean set of binary bad / false edges.”
- “The selected edges are physically incorrect.”
- “The selected edges are intrinsically useful.”
- “Each selected edge individually has both good and bad semantics.”
- “Timestamp fusion is the identified harmful mechanism.”
- “Phase 2C proved geodesic shortcut correction is the paper direction.”

Allowed formulation：

> `H1_counterfactual_supported = false` under the frozen clean / low-collateral-damage criterion, while selected-edge deletion showed localized repair effects together with collateral degradation.

---

### H2 / same-source coupling

- “Same-source coupling causes confirmation bias.”
- “Stage 2 reinforces Stage-1 errors.”
- “Stage 2 preserves Stage-1 structural errors.”
- “Stage 2 amplifies Stage-1 errors.”
- “H2 is supported.”

Allowed formulation：

> Same-source `D_geo` coupling is a confirmed pipeline fact; its empirical consequence remains unresolved.

---

### H3 / H4

- “Fixed-k currently causes topology failure on the current run.”
- “The current refinement has been proven to over-smooth true geometry.”
- “Replacing Laplacian with TV / anisotropic diffusion solves a Channel-Chart-specific problem.”

---

### Generalization / localization

- “0.319 m is held-out localization accuracy.”
- “The current refinement improves unseen-CSI localization.”
- “The current method demonstrates inductive generalization.”
- “Candidate B has been empirically validated.”
- “The current Phase 2C results demonstrate held-out / unseen-CSI performance.”

---

### Candidate selection

- “Reliability-aware refinement is the final paper direction.”
- “Candidate B is selected.”
- “Track A must remain the final research track.”

当前最终方向仍然：

```text
Not selected
```

---

## 12. Current Research Decision

### 12.1 Stop open-ended Phase 2C diagnosis

当前决定：

> **Do NOT continue open-ended Phase 2C diagnostics.**

Phase 2C 已经完成 enough empirical diagnosis for research-direction decision。

---

### 12.2 Do not automatically execute H2

H2 不再是 current next action。

只有当 Phase 03 保留 serious same-source / reliability direction，并且 H2 可以作为 discriminating diagnostic 时，才执行。

---

### 12.3 Do not begin method development

当前不进入：

- new graph model；
- GNN；
- adaptive weighting；
- new diffusion；
- inductive architecture implementation；
- hyperparameter optimization；
- Codex method-development task。

当前还没有完成 research-direction selection。

---

### 12.4 Do not select Candidate B yet

Candidate B 当前 priority raised / High，只表示：

> 它应在 Phase 03 comparison 中得到更高关注。

不表示：

- 已选择；
- 已证明 novelty；
- 已证明可发表；
- 已证明 empirically effective。

---

### 12.5 Proceed to Phase 03

当前正式进入：

```text
03 Research Direction Decision
```

Current substage：

```text
Entering Round 1 — Candidate Reconstruction & Decision Framework
```

Round 1 应比较：

1. **Candidate B — Out-of-sample / inductive post-hoc refinement**
2. **Candidate C — Reliability / selective refinement**
3. **Candidate A — Broader post-hoc refinement**
4. **only genuinely competitive Track-B alternatives**

Candidate D 和 binary harmful-edge route 当前不作为 primary Round-1 leading directions，但可作为 comparison / reference。

---

## 13. Phase 03 Round 1 Decision Framework

Round 1 不做方法开发。

目标是对 serious candidates 使用统一框架重建并比较。

每个方向至少需要明确：

### Research Problem
具体解决什么问题？

### Evidence
这个问题是否由：

- confirmed pipeline fact；
- existing empirical observation；
- literature limitation；

提供支持？

### Closest Work
最接近的已有工作是什么？

### Gap
已有方法真正缺少什么？

### Mechanism Requirement
需要什么类型的 mechanism 才可能不是 straightforward transfer？

### Communications Relevance
为什么这是 Channel Charting / wireless communications problem，而不只是 generic graph / ML problem？

### Minimum Experiment
最小验证实验是什么？

### Falsification
什么结果出现时应该放弃该方向？

### Feasibility
现有数据、代码、计算资源能否支持？

### Reuse
能否合理复用当前 DICHASUS / ADP / FCF / evaluation pipeline？

### Publication Potential
在 TCOM quality bar 下是否有可能形成：

```text
problem
→ evidence
→ non-trivial mechanism
→ controlled evaluation
```

完整闭环？

---

## 14. Current Next Task

```text
03 Research Direction Decision — Round 1
```

### Goal

> Reconstruct the serious candidate set, establish unified decision criteria, prune weak directions, and retain **at most 2–3 directions** for focused Round 2 verification.

### Round 1 scope

允许：

- candidate reconstruction；
- evidence audit；
- prior-work boundary comparison；
- unified scoring / decision framework；
- pruning；
- identification of missing evidence needed for Round 2。

当前不允许：

```text
No new experiment.
No Codex task.
No final method design.
No final research-direction selection in Round 1.

Even if only one candidate appears clearly dominant,
stop after Round 1 and wait for upper-level research review
before deciding whether Round 2 can be shortened or skipped.
```

默认目标仍然是：

> 保留最多 2–3 个 serious candidates 进入 Round 2 focused verification。

---

## 15. Historical Phase-2 Decision — Superseded as Current Action

以下内容具有历史价值，但已不再是 current state：

```text
H1 → H2 diagnostic sequence
H2 = highest research interest
Candidate C waits for H1/H2 evidence
continue designing H1/H2 diagnostics
Phase-2 H1/H2 Go/No-Go tree as next action
```

这些属于 prior Phase-2 reasoning。

它们已经被最终 Phase 2C synthesis supersede。

当前 action 是：

```text
03 Research Direction Decision — Round 1
```

而不是继续 Phase 2C diagnostic chain。

---

## 16. Research State Update Rule

`research_state.md` 是当前阶段已经确认 research consensus 的 single source of truth。

本文件不用于保存：

- raw experiment logs；
- brainstorming dump；
- 全部 literature notes；
- chat transcript；
- 尚未经过 review 的临时 interpretation。

Phase 2C 详细 evidence 继续保存在：

```text
phase2c_research_synthesis.md
phase2c_empirical_handoff.md
H1_COUNTERFACTUAL_REPORT.md
h1_counterfactual_summary.json
```

Phase 03 Round 1 / Round 2 完成后，
由上层 research review 决定是否进入 Round 3。

只有 Phase 03 最终 research-direction decision
经过上层 research review 批准后，
才更新 `research_state.md` 到下一 major version
（预计 v3.0）。

Workflow：

```text
Round 1
→ upper-level review
→ Round 2
→ upper-level review
→ Round 3 / final direction decision
→ upper-level approval
→ research_state.md v3.0
```

开始 Phase 03 或新的研究对话时，应优先读取：

1. `Research_Context.md`
2. `research_state.md v2.1`
3. `phase2c_research_synthesis.md`

---

## 17. Current State Summary

```text
Current phase:
Phase 03 — Research Direction Decision

Current substage:
Entering Round 1 — Candidate Reconstruction & Decision Framework

Final research direction:
Not selected

Phase 1:
Complete

Phase 2A:
Complete

Phase 2B:
Complete

Phase 2C:
Empirical diagnosis complete enough for direction decision

H1_counterfactual_supported:
false

H1_supported:
null

Binary harmful-edge route:
Downgraded / Low / diagnostic-control role

H2:
Unresolved / not executed / conditional diagnostic only

H3:
Low

H4:
Low

Candidate A:
Retained research setting / not selected

Candidate B:
Priority raised / High for Phase 03 comparison / not selected

Candidate C:
Empirical motivation increased / still generic / not selected

Candidate D:
Downgraded / Low

Current action:
03 Research Direction Decision — Round 1

New experiment:
No

Codex task:
No

Method development:
No
```
