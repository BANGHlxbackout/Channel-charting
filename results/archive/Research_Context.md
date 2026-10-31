# Research Context

## 当前研究目标

目标是完成 Channel Charting / 无线定位相关硕士研究，并形成至少一个具有明确创新性、可以完成实验验证和论文撰写的研究方向。

当前尚未确定最终论文方法。

研究的首要目标不是坚持某一个已有方法，而是找到：

- 研究问题明确；
- 具有真实 novelty；
- 能够完成实验验证；
- 能够合理复用现有数据、代码和研究基础；
- 具有论文发表潜力；
- 最终能够支撑硕士毕业论文

的研究方向。

---

## 当前已有研究基础

目前已有：

- 导师提供的 Channel Charting 相关论文；
- Channel Charting baseline 代码；
- diffusion / manifold refinement 后处理实验；
- 与当前 baseline 有关的方法说明和研究笔记。

这些内容构成当前研究基础，但不是最终研究范围。

### 当前 baseline 的研究归属

当前导师 baseline 不是一套从 CSI 开始完全新设计的 Channel Charting 方法，必须明确区分两个阶段。

#### Stage 1：Published / inherited Channel Chart generation

前半部分主要继承 Project Sources 中 Stephan et al. 2024 的已有方法：

```text
CSI
→ ADP / timestamp-aided dissimilarity
→ kNN / geodesic distance
→ Siamese forward charting function
→ initial Channel Chart Y_0
```

这一阶段应作为已发表方法的复现 / 继承基础，不应被描述成导师 post-processing baseline 的创新。

#### Stage 2：Advisor-provided post-processing baseline

导师新增的 baseline 从初始 Channel Chart `Y_0` 之后开始：

```text
Y_0 + existing geodesic information
→ graph construction
→ Laplacian / Tikhonov smoothing
→ refined Channel Chart Z
```

因此，后续分析创新点时必须明确区分：

- inherited Channel Chart generation；
- advisor-provided post-processing；
- 两阶段之间的信息 coupling；
- evaluation protocol。

不能把整条 CSI→Z pipeline 统一视为导师的新方法。

---

## 当前候选方向

### Track A：Channel Charting 后处理

当前最成熟的候选路线是研究已经生成的 Channel Chart 的后处理 / refinement，例如：

- graph construction；
- graph refinement；
- Laplacian / Tikhonov smoothing；
- diffusion；
- manifold refinement；
- topology-preserving post-processing；
- out-of-sample refinement。

当前导师 baseline 是 Track A 的研究起点，但尚未证明该方向具有足够的论文创新空间。

不要默认必须在 Laplacian、diffusion 或 kNN 图上继续迭代。

### Track B：Channel Charting 其他问题

如果后处理方向创新空间有限，应主动研究其他 Channel Charting 问题，包括但不限于：

- CSI dissimilarity metric；
- ADP / CIR / angular-delay representation；
- graph construction；
- geodesic distance；
- metric learning；
- triplet / contrastive learning；
- self-supervised learning；
- weak supervision；
- global geometry；
- real-world coordinates；
- NLoS robustness；
- multipoint / distributed MIMO；
- transductive / inductive Channel Charting；
- unseen trajectory / unseen UE generalization；
- domain robustness。

优先考虑可以复用当前 CSI 数据、代码和实验体系的问题。

### Track C：邻近无线定位方向

如果 Track A / B 缺乏足够论文价值，可以研究与当前基础高度相关的：

- CSI-based localization；
- weakly supervised localization；
- wireless representation learning；
- CSI metric learning；
- graph-based localization；
- manifold learning for wireless localization。

不要轻易跳到与当前研究基础完全无关的领域。

---

## 关于当前 diffusion idea

`new_idea_diffusion_refined_channel_charting.md` 中的方法属于历史研究过程中形成的一个 Candidate Idea。

它不是已经确定的论文方法。

阅读该文件时，应：

1. 理解它提出的问题和机制；
2. 检查是否已有高度相似工作；
3. 分析其 novelty；
4. 判断是否值得继续实验；
5. 必要时可以否定该方案。

不要因为该文件已经存在，就默认必须沿该方向继续。

同样，不要因为当前 Laplacian/Tikhonov baseline 已经取得较好的实验结果，就自动认为论文必须继续优化该 baseline 的指标。

---

## 研究原则

### 1. Problem-first

优先回答：

**什么问题值得研究？**

然后再回答：

**应该使用什么方法？**

不要反过来因为已有 diffusion、GNN、attention、Transformer 或 graph 方法，而人为寻找一个能够使用它的问题。

### 2. Baseline is evidence, not the final answer

导师 baseline 的作用是：

- 提供可运行的研究起点；
- 验证 post-processing 可能改善 Channel Chart；
- 暴露潜在 failure modes；
- 为后续 research question 提供实验依据。

它不等于已经确定的最终论文方法。

### 3. Literature-first novelty judgment

在判断一个想法是否有创新性之前，应优先检索相关论文。

尤其需要区分：

- graph 用于 chart construction；
- graph / topology constraints 用于 training；
- chart alignment / coordinate transformation；
- 已生成 chart 的 post-hoc geometry refinement；
- out-of-sample extension。

不要因为某个技术没有在当前 Project 中使用过，就自动认为它具有 novelty。

### 4. 区分事实、结论、推断和假设

后续讨论必须明确区分：

- 当前代码事实；
- 当前实验事实；
- 已发表论文结论；
- 根据文献或机制得到的推断；
- 尚未验证的研究假设。

不能把 hypothesis 在多轮讨论后逐渐表述成已确认事实。

### 5. 不以单纯指标优化驱动研究

不要把研究目标设定为：

```text
不断修改代码 → 指标继续提高 → 再寻找论文解释
```

优先建立：

```text
Research Problem
→ Literature Gap
→ Research Hypothesis
→ Proposed Mechanism
→ Controlled Experiment
→ Evidence
```

然后再进行完整实现和指标优化。

### 6. 允许否定当前方向

如果文献检索、理论分析或实验表明：

- novelty 不足；
- 与已有方法高度重复；
- 只能形成模块替换；
- 无法提出清晰 research hypothesis；
- 无法构造有说服力的实验；
- 工作量显著高于论文价值；

允许最终结论是：

> 当前 diffusion / graph refinement 方向不值得继续投入，需要转向其他 Channel Charting 或邻近无线定位问题。

---

## Research State 文件

`research_state.md` 用于记录当前已经确认的研究进度和研究共识。

开始新的研究阶段或新对话时，应优先读取：

1. `Research_Context.md`
2. `research_state.md`

其中：

- `Research_Context.md` 保存长期研究目标、边界和原则；
- `research_state.md` 保存当前阶段、已确认事实、未验证假设、禁止提前宣称的结论和下一步任务。

新的研究结论只有经过检查和确认后，才写入 `research_state.md`。
