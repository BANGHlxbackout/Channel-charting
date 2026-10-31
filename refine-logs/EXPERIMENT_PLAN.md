# G1 Stage-1 时间连续性 loss 实验计划

**日期：**2026-10-25  
**状态：**计划已形成；新 loss 尚未实现、训练或评价。  
**使用流程：**ARIS `experiment-plan` 的 claim → evidence → run order。用户本轮明确要求形成 plan、不做新一轮 review；不启动 `research-refine` 的 reviewer 循环。

## 问题与选择

当前 G1-Graph 复用 B0 的 geodesic endpoint stress，只换图传播。完整同构实验中 Graph Mean/P95 为 0.399771/0.846337 m，Identity 为 0.398343/0.843197 m：传播没有带来可辨认的净收益。[原 G1 报告](../results/g1_acf_graph_fcf/REPORT.md)。

此前路径折线 loss 的离线 Gate 说明当前 `D_geo` 的物理路径长度语义对尺度选择敏感；不能据此断言路径 loss 一定无效，也不足以直接投入训练。另取 59,995 个均匀随机非自身点对时，`D_geo` 与真实端点距离的 Spearman 为 0.982649，现有 G1-Graph 对真实端点距离为 0.991562；继续强制全局距离排序贴近 `D_geo` 缺少改善依据。数据与复跑脚本见[离线 JSON](../results/g1_acf_graph_fcf/loss_semantic_gate/result.json)及[诊断入口](../diagnostics/g1_loss_semantics/run.py)。这些 GT 数值仅用于离线判断，不进入下述训练或 checkpoint。

更直接的未利用信号是**时间连续性**。原数组未按时间排序，须先按 timestamp 排序。相邻时间差的 99.9 分位为 0.961 s；用 `0<Δt≤1 s` 可保留 20,955 个连续三点组。已经训练的图谱沿这些三点组的速度变化范数中位数：GT 0.056382、G1-Graph 0.328982、G1-Identity 0.338135；P95 分别为 0.190822、1.053172、1.166058。数值在各图谱分别仿射对齐后计算，只是离线现象。图谱抖动是否造成 Mean/P95 误差仍待训练验证，不能仅凭此诊断声称可改善。

**方法假设：**在原 geodesic stress 维持全局尺度和非坍缩约束时，用无标签时间顺序惩罚过快的相邻速度变化，可能减少 G1 图谱轨迹抖动而保留已有的空间邻域关系。此类加速度先验已见于 [Euchner 等人的实测数据论文](https://arxiv.org/html/2412.01715)，因此单纯加入它不是新颖性结论；[2609 图传播论文](https://arxiv.org/html/2609.06204)只提供本轮 G1 邻域传播的背景，不能把不同数据集的结果搬过来。

## 冻结候选 loss

设 `o` 是按现有 timestamp 升序的原样本索引；只取相邻两个时间间隔均满足 `0<Δt≤1 s` 的三点 `(o[t−1],o[t],o[t+1])`。令

`v[t−1]=(Y[o[t]]−Y[o[t−1]])/Δt[t−1]`，`v[t]=(Y[o[t+1]]−Y[o[t]])/Δt[t]`。

使用鲁棒速度变化项

`L_time = mean_t (sqrt(||v[t]−v[t−1]||² + ε²)−ε)`，其中 `ε=1e−6`，不使用 GT、物理速度或物理加速度。总 loss 为 `L = L_old_geodesic_stress + λ·L_time`。原 fused 图、`D_geo`、GCN 邻接、特征、模型参数、随机 pair、AdamW 和原学习率日程不变。

为避免任意手调 λ，训练前在**同一 seed=10 初始化、Identity 邻接**的一次 full-N 前向上，分别计算原 loss 与时间项的初始参数梯度范数 `g_old`、`g_time`，冻结 `λ=0.1·g_old/g_time`，并将这个**同一个数值**用于 Graph 与 Identity 新训练。若 `g_time` 非有限或近零，Sanity 失败，停止；不借助 GT 调 λ。0.1 只是预先固定的弱正则强度，不保证收益。时间项使用全部有效三点组，避免另一套采样 RNG；原训练/验证点对流及 hash 应与旧 run 一致。checkpoint 仍由**原验证点对上的 geodesic endpoint stress**选择，GT 不参与。

## Claim map 与决定性对照

| ID | 待检验的有限主张 | 最小证据 | 结果块 |
|---|---|---|---|
| C1 | 时间速度连续性项可以改善 full-N G1-Graph 图谱 | G1-Graph+Time 的 Mean、P95 均低于原 Graph，CT/TW/KS 不出现实质退化；同样数据和评价样本 | B1 |
| C2 | 收益可归因于图传播与时间项的交互 | Graph+Time 优于 Identity+Time；Graph 相对旧 Graph 的收益大于 Identity 相对旧 Identity 的收益 | B2 |
| 反解释 | 只是所有 FCF 都受益于时间正则，或只是图变平滑 | 若 Identity 获得同等/更大收益，只报告通用 Stage-1 loss 效果；同时检查过平滑、转角损伤和 Max/P95 | B2、B3 |

### B1：主结果，MUST-RUN（在 Sanity 通过且完整预算获确认后）

- 数据：原 `datasets/adp_old_shard4`，stride=1、N=20,997、seed=10；同一 frozen fused、`D_geo`、ACF 缓存与样本索引；同节点传导式评价。
- 方法：原 B0、原 G1-Identity、原 G1-Graph，以及仅新增时间 loss 的 G1-Identity+Time、G1-Graph+Time。旧 checkpoint 和 B0 原实现不改。
- 固定训练：两组新 run 按旧协议各最多 2,000 step、每 step 65,536 对、每 20 step 用同一 200,000 验证点对、原早停和 best/last 保存规则；同一新 loss λ、时间三点组和随机种子，仅传播邻接为 fused Graph/I。
- 主要指标：仿射 Mean、P95；同时报告 CT、TW、KS、Median、P90、Max、训练时间、峰值显存。CT/TW/KS 沿用 seed=10 固定 20% 样本；全部 20,997 节点分别仿射对齐计算误差。不把传导式结果写成未见节点能力。
- 成功判据：Graph+Time 的 Mean 与 P95 **同时低于**旧 Graph 和 Identity+Time，且 CT/TW/KS 无明显退化。单 seed 只算候选有效，不算稳定创新结论。
- 失败解释：若只降低时间 loss 或图谱轨迹抖动，却不改善 Mean/P95，则停止这个 loss 方向；若 Identity 同等受益，则可保留通用 loss 观察，但不声称图传播正效应。

### B2：交互与损害，MUST-RUN

定义误差改善 `ΔG=Error(Graph+Time)−Error(Graph)`，`ΔI=Error(Identity+Time)−Error(Identity)`；对 Mean 与 P95 分别检验 `ΔG<ΔI`，并与新两臂直接比较。报告时间速度变化分布、沿时间排序图谱、急转弯区的局部误差和 Max。GT 只做事后评价，不参与训练、选模或 λ。此块复用 B1 的保存 embedding，不新增训练。

### B3：稳定性，NICE-TO-HAVE

仅在 seed=10 同时通过 B1/B2 时，另定预算重复匹配种子；旧 seed=10 结果不能冒充多种子证据。首轮不加 path loss、不换图、不做 Stage-2、不叠加额外平滑项，也不做 GT 指导调参。此方法属于已发表时间先验在 G1 上的受控适配，不以单次结果宣称论文新颖性。

## 执行顺序与成本

| 阶段 | 动作 | 通过/停止条件 | 预计 GPU 成本 |
|---|---|---|---:|
| M0 Sanity | 只在 `methods/g1_acf_graph_fcf/` 增加 CLI loss 选项；核对 timestamp 排序和索引、有效三点组、旧 pair hash；小步检查两项梯度、有限值、best/last | 任一索引/梯度/NaN/坍缩异常即停止，不改数据规模掩盖错误 | 约数分钟 |
| M1 Pilot | full-N Graph 与 Identity 各 5–20 step，记录前向耗时、GPU 总占用及 PyTorch 峰值；不排名指标 | 两组同 λ、pair hash、时间组且资源可行才进入完整训练 | 预计 <0.2 GPU 小时 |
| M2 Full | 按 Graph+Time → Identity+Time 顺序，各最多 2,000 step | 达到预先确认的资源上限则保留 checkpoint 并标未完成；异常先定位 | 旧两臂实测合计 3.199 GPU 小时，新项预计约 3.5–4.5 小时，须以 Pilot 修正并确认上限 |
| M3 Compare | 统一重算五方法指标、CSV、raw/affine 多列图和主报告 | 比对索引、抽样索引、JSON/CSV/图标题及主报告一致 | 无训练 |

代码优先扩展现有 G1 入口的单一 loss CLI，旧默认行为需保持逐项一致。比较复用 `methods/g1_acf_graph_fcf/compare.py` 和 `run_adp_siamese.plot_chart`；结果放同一 G1 实验家族，完整指标 CSV、原始与仿射 Chart、`eval_protocol.json`、best/last checkpoint、日志和资源数据按 `AGENTS.md` 交付。完整训练预算是**拟议**，本文件不启动训练，也不沿用上次 G1 专属的 6 GPU 小时授权。

## 关键风险和边界

1. 轨迹确有真实转角，过强速度正则可能抹掉结构；使用线性尾部惩罚，并在结果中单独报告转角区损害。
2. 旧验证只检查 geodesic stress，可能不选择时间 loss 最小的 checkpoint；这是保持旧/新 checkpoint 协议可比的代价，保存 last 供诊断，但不依 GT 选第二个 checkpoint。
3. Graph 旧时间抖动已略低于 Identity；如果时间项仅让两者同样变平滑，不足以支持图传播正向作用。
4. 单 seed、同节点评估不支持 OOS 或跨数据集泛化；发表层面的机制和新颖性证据需要另外研究决定。

## 本轮检查表

- [x] 问题、候选 loss、已发表最近工作及离线事实有来源。
- [x] 冻结只改 loss 的两个新训练臂与复用的旧对照。
- [x] 明确主指标、交互判据、Sanity/Pilot/Full 与停止条件。
- [ ] 实现、Sanity、Pilot 和预算确认；当前均未执行。
- [ ] full-N 训练、图谱与完整指标；当前均未执行。
