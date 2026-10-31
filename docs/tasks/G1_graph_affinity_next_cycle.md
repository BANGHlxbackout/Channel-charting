# G1-Graph Stage-1 改进：当前任务与恢复检查点

日期：2026-10-26。用户本轮明确要在实测天线数据上继续改善已有 G1-Graph；[2609 图网络论文](https://arxiv.org/abs/2609.06204)仅作启发。本任务只研究 Stage 1，不执行 Stage 2，不采用旧的路径 loss 或时间速度变化项。

**最新范围修正（2026-10-26）**：用户要求优先思考如何改进现有 loss，并质疑单独改 ADP 传播权重的优先级。以下 ADP-affinity Sanity/Pilot 事实保留，但其 full-N 正式训练计划暂停，不能沿用下文预算请求自动启动。现有 ACF 预处理、fused 图和 `D_geo` 保留；loss 候选的保存图谱梯度诊断已完成，模型尚未按新 loss 训练。

## 当前 loss 方向：候选设计，未训练

当前实现为 `L_geo = mean[(||z_i-z_j||-D_geo(i,j))²/(D_geo(i,j)+m)]`，训练点对在全节点对上均匀抽样。原图仅占全部无序点对约 0.104%；40 万随机点对的只读样本中，最短 1% 的点对只贡献约 1.51% 的总 loss。已训练 G1-Graph 在 228773 条原无向边上的 stress 略低于 G1-Identity（0.472659 vs 0.480077），但全局 chart Mean/P95 略差。由此不能简单再加邻边 stress 或只追求局部相似。

拟借鉴 position-diffusion 的**局部一致性思想**，按本任务改为锚定式损失，而不是直接替换原 loss：从原 fused support 和 G1 已用的 Gaussian 权重构造对称 affinity（含自环），然后按行归一化得固定 `P`；令 `s_i = sqrt(sum_j P_ij D_geo(i,j)²)`，`L_diff = mean_i[||z_i-(Pz)_i||²/(s_i+m)]`，`L_new = L_geo + λ L_diff`。不使用 GT、raw CSI、ADP-only 邻接、额外训练参数或 Stage-2 输出。原 stress 维持全局尺度与防塌缩；扩散项约束局部坐标粗糙度。可先以 `λ=1` 作单一固定候选：在保存的收敛 Graph embedding 上 `L_diff≈0.197`，约为原验证 stress `1.873` 的 10%；这仅是数值量级校准，不是 GT 调参。

只读探索：原 fused 邻域的 GT 坐标平均邻域残差约 `0.0352 m`，已训练 Graph/Identity 仿射图为 `0.0546/0.0565 m`。对保存 embedding 作**一次离线传播诊断**（非训练结果、非本轮 Stage-2 方法）后，Graph Mean/P95 从 `0.399771/0.846337 m` 到 `0.390746/0.822420 m`；Identity 同时从 `0.398343/0.843197 m` 到 `0.388984/0.820068 m`。这提示局部粗糙度可能可降，但 Identity 也受益，不能归功于 GCN，也不能推断训练上述 loss 一定会复制该改善。GT 只在此离线评价使用，不进入候选损失或 checkpoint。

若决定继续验证：只改变 loss，从**随机初始化**训练 `G1-Graph+L_new`，条件达标后再训练 `G1-Identity+L_new`；两者都用同一固定 `P` 计算辅助项，Identity 的网络传播仍是 `I`，以隔离 GCN 传播贡献。保留原特征、fused 邻接、`D_geo`、模型、点对、seed、优化器和评价；checkpoint 仍依据原 `L_geo` 验证损失，不依 GT 选模。原 full-N 曲线在 100 step 验证 loss 约144，400 step约4.30，800 step约2.03，最终约1.87，故 5–100 step 只可查实现/资源，不能判断图谱质量；800 step 也只是趋势，最终比较需依原协议最多2000 step。先核验代码/梯度/显存，再按旧训练耗时估计新 Graph 约2 GPU小时、若通过门槛的 Identity 约2 GPU小时；完整训练前依用户要求确认新预算。主要看同一 full-N 的 Mean/P95/CT/TW/KS、Max 与逐点损害。Graph 新 loss 必须先在主要指标上超过已保存的最好同构 Identity，而不只是超过旧 Graph；若失败，停止这条候选，不自动调 `λ`。若 Graph 通过，再跑 Identity 新 loss 判断收益是否来自图传播。`λ=1` 和完整判据在训练前固定。纯平滑 loss 单独训练有常数解，因此不考虑该方案。

### 已完成：保存图谱的梯度方向核验（2026-10-26）

[完整报告](../../results/g1_acf_graph_fcf/loss_direction_diagnostic/REPORT.md)、[代码](../../diagnostics/g1_loss_direction/run.py)、[比较 CSV](../../results/g1_acf_graph_fcf/loss_direction_diagnostic/comparison/comparison.csv)。固定随机点对、`λ=1`、同等 RMS 坐标步长只由 `D_geo` 决定；GT 只在步后评价。Graph 的组合方向使 54.43% 节点误差降低、45.57% 增大，Mean/P95 从 `0.399771/0.846337` 到 `0.398884/0.843249 m`；单独原 loss 梯度步为 `0.399704/0.845223 m`。高误差五分位平均改善 `0.001907 m`，逐点损害变化的 P95 为 `+0.011682 m`。Identity 同样从 `0.398343/0.843197` 到 `0.397428/0.840959 m`，仍优于 Graph。新增项在当前图谱附近提供小幅有利且与原 stress 不同的方向，但它不是 Graph 独有；直接移动 embedding 不等于神经网络重训。另作无 GT 随机初始化前向检查：Graph 的 `L_geo≈250.321`、`L_diff≈0.0165`，Identity 为 `250.076/0.0333`；新增项在训练起点的数值比原 loss 小得多，但这不等于已验证训练过程或参数梯度。当前证据不足以称 loss 改进已验证。若继续，下一步是从头训练的受控实验；不得用少量步数的图谱质量或保存图谱上的自由坐标移动冒充最终收益。

### 理论与问题定位复核（2026-10-26；覆盖上文训练建议）

`L_diff=||Qz||_S²/N`，其中 `Q=I-P`。若在同一物理尺度下把当前图谱写为 `z=p+e`，负梯度降低平方位置误差需要 `⟨Qe,Qz⟩_S>0`，即 `||Qe||_S²+⟨Qe,Qp⟩_S>0`；真实位置在图上有边界变化、错误边或误差并非局部高频时，这个条件没有保证。实际从随机初始化训练沿参数梯度 `J_theta^T ∇_z L` 运动；旧 checkpoint 的自由坐标方向不能推出参数更新或最终指标。随机初始化时辅助 loss 数值小也不能单独证明梯度小。

额外只读核对：原 fused-k20 邻域中，已训练 Graph 的逐点仿射误差与局部坐标残差 `||z-Pz||` 的 Spearman 相关仅 `0.0892`，Identity 为 `0.1029`。Graph 初始误差最低/最高五分位的平均局部残差分别 `0.04749/0.06060 m`，GT 对应 `0.03196/0.04122 m`；两组中图谱比 GT 更粗糙的比例约 `73.19%/71.81%`。因此粗糙度普遍偏高，却不能有效定位高误差节点。旧 Graph 的邻边 stress 略优于 Identity 而 Mean/P95 略差，也不支持“再加强局部图一致性就能反超”。

**当前判定**：现有理论与证据不足以将 `λ=1` 组合 loss 推入从头完整训练；暂停本候选的实施和预算申请，不把离线单步好转称为方法收益。下一步先做不改训练的误差来源定位：在相同节点和点对上，分别核对 `D_geo` 相对真实距离的离线偏差、Graph/Identity 相对 `D_geo` 的残差、最终仿射误差，并按测地距离、边来源和空间区域分组。GT 只作离线归因。若主要问题是 `D_geo` 目标有系统偏差，才研究输入可判定的目标或点对可靠性；若目标合理而模型没拟合，则研究表示与优化；只有高误差明确由可信边上的局部噪声造成，才考虑有保护条件的局部正则。此时不预设新 loss 公式或调 `λ`。

### 用户确认后的误差来源检查（2026-10-27）

本节完成上述只读检查，详见 [报告](../../results/g1_acf_graph_fcf/error_source_diagnostic/REPORT.md)、[代码](../../diagnostics/g1_error_source/run.py)及 [pair_groups.csv](../../results/g1_acf_graph_fcf/error_source_diagnostic/pair_groups.csv)。固定 399983 个随机点对、全部 228773 条原图无向边、同一保存的 Graph/Identity embedding；GT 只作离线比例拟合、归因和评价。`D_geo` 与物理距离的 Spearman 为 `0.982725`，但用全局比例换算后短距离 0–10% 点对平均高估 `0.4536 m`，长距离 90–100% 平均低估 `0.3904 m`。Graph 对 `D_geo` 的原生距离残差在两端反向（`-19.02/+21.60` 原单位），最终仿射点对偏差小得多；不能把更紧贴全部目标直接等同于更接近 GT。时间来源边的目标绝对偏差均值 `0.0415 m`，ADP 来源边 `0.1768 m`，但按节点汇总的邻边目标偏差与 Graph 节点误差 Spearman 仅 `-0.0257`；随机点对目标偏差关联 `0.1705`，Graph 对目标拟合残差关联 `0.0597`。Identity 关联相近。空间区域也未出现稳定对应。**结论**：可确认目标有距离相关尺度失配，不能确认它是高误差节点的主要原因，也不能用来源类别直接定新训练权重；旧平滑 loss 仍暂停。下一具体问题是训练时可观测的 ADP/时间差/路径结构，能否在相同距离与空间范围内识别目标失配；若不能，暂不提出可靠性加权 loss。

## 已暂停的 ADP-affinity 分支：原问题与实验事实

现有 G1-Graph 的 Mean/P95 0.399771/0.846337 m，分别略差于同构 G1-Identity 的 0.398343/0.843197 m；冻结 Graph checkpoint 换 I 会明显变差，说明网络依赖邻接，却没有独立的几何净收益。要判断的是：**保留原 fused-k20 的边集合，只用 ADP 而非 ADP/时间最小值决定 GCN 传播强度，能否让邻居消息带来可辨认的 Stage-1 收益？**

具体例子：若两节点时间很近但 ADP 不相似，原 fused cost 很小，使 GCN 强烈混合两者特征；新规则仍保留这条边，但以 ADP cost 和各节点的 ADP-k20 局部尺度降低其权重。训练目标仍是保存的 fused 图最短路 `D_geo` 的原 `siamese_loss`；改动实际发生在 **GCN 的传播邻接**，而不是 loss 或 Stage-1 最短路图。

## 已有证据与最强反解释

- 228,773 条原无向候选边中，179,147 条由时间项主导。只读检查显示：若固定这些边，仅换 ADP 亲和度，时间来源边的权重占比由 83.93% 降到 71.51%，节点邻居消息占比的中位数由 0.934 降到 0.895，未出现广泛自环主导。GT 离线加权边长均值由 0.184 降到 0.164 m，但 P90 由 0.390 略升到 0.396 m；这只是探索性图质量线索，不是 chart 收益。
- 4096 节点的临时离线检查中，ADP-only k20 的物理邻居距离中位数比 fused k20 短，但节点 P90 没有稳定改善，平均仅重合 10.27/20 个邻居。因此首轮保留原边集合，仅改变权重；不直接替换整张图。GT 仅用于该离线判断，不进入拟议传播权重或训练。
- 用户已明确保留现有 ACF 预处理；raw CSI 输入不是本轮或后续候选，不据原论文的输入方式改动特征。上述图质量检查只是临时探索，未另立实验或方法结论。
- 最强反解释：时间邻居即使 ADP 不近，也可能沿实际运动轨迹提供有用上下文；ADP 降权或许改善平均误差，却损害转角/尾部节点。即使新方法无收益，也只能否定本次权重规则，不能否定所有图结构。

## 已暂停分支的原冻结候选与对照

- 数据/模型/特征：沿用 full-N 20,997 节点、seed10、26,624 维原 ACF、两层 Residual Sparse GCN、相同模型参数和 feature cache。
- 原有有向 fused-k20 图及其无向 support、原 `D_geo`、loss、AdamW、point pair 流、验证 pair、学习率、早停、checkpoint、评价子集均不变。新权重只在原 support 上取 `D_ADP(i,j)` 与双方第20个**非自身** ADP 邻居距离的乘积作 Gaussian 分母；自环1和对称归一化不变。没有新的可训练参数或可调阈值。
- 必要对照：保存的 B0、G1-Identity、原 fused-affinity G1-Graph，以及唯一新训练臂 **G1-Graph/ADP-affinity**。前三者直接读保存 embedding/checkpoint，不重训。Graph/Identity 已同构；新旧 Graph 仅传播亲和度不同。
- 完整实验的候选支持条件（须在正式运行前再次核验）：新 Graph 的 Mean、P95 均至少比当前最强 Stage-1 同构 Identity 低 0.005 m；CT/TW 相对 Identity 各下降不超过 0.0005，KS 上升不超过 0.0005。Max 和逐点 collateral harm 单独报告；仅 KS 或短程图质量改善不算成功。单 seed 只能算当前数据上的探索性支持，不算稳定创新结论。
- GT 仅用于离线诊断及训练后 `common_metrics` 仿射评价/绘图，不能进入新邻接、loss、lambda、checkpoint 或模型选择。保留原 200000 验证点对的索引偏向，并如实标记，不在本轮暗改 sampler。

## 用户提出的其他两个方向（不并入当前单变量实验）

- 三层传播有可检验性，但应先在现有 Residual Sparse GCN 中仅从两层增至三层，且对 Graph 和 Identity 同时做三层对照；直接改用论文的 `GCNConv` 会同时改变算子、归一化、残差路径和深度。三层是否改善实测数据尚无证据；本轮 ADP-affinity 实验不改层数。
- 论文的 position-diffusion 平滑目标单独最小化时，常数 embedding 即可取零损失；用户现要求根据本任务设计**组合式** loss。上面的 `L_geo + λ L_diff` 是待审候选，尚未实施或训练。

## 已暂停的 ADP-affinity 原执行顺序与资源边界

1. **Implementation/Sanity**：在既有 `methods/g1_acf_graph_fcf/` 的 Graph 入口加一个 `--affinity adp` 选择；默认 `fused` 行为保持不变。检查原 fused 最短路复验、同一无向 support/样本顺序、新旧邻接确有差别、对称/有限、特征/梯度/loss 与原实现一致。真实 stride10 仅做功能 Sanity，其图需 k35，不可当 full-N 排名。
2. **Full-N 资源 Pilot**：新 Graph 在 N20997 上 5–20 step，测量时间与峰值显存、检查 finite/backward/checkpoint；标记 Pilot，不作效果结论。
3. **Full-N 决定性运行**：仅在用户核对 Pilot 后确认新预算，再按原最多2000 step冻结协议跑一个新 Graph。旧 Graph 全训练耗时 6224.64 s，拟议单臂成本约2–3 GPU小时；旧6 GPU小时授权不自动转用。
4. **统一评价**：如完整运行完成，四方法在相同4199节点计算CT/TW/KS，在所有20997节点分别仿射拟合计算Mean/Median/P90/P95/Max。遵循 `AGENTS.md`，保存完整 CSV、`source_json`、GT/各方法的raw与affine Chart、best/last、配置、日志及一份主报告。

若输入/索引/图或梯度检查失败，停止；若仅 Pilot 可行但未确认预算，不声称完整效果。正式负结果保留，不自动改变 k、loss、强度或叠加新模块。

## 恢复检查点（唯一；2026-10-27）

- 目标/授权：用户确认只读检查 `D_geo` 目标偏差、Graph/Identity 拟合残差及最终误差，以决定是否需要改 loss；保留 ACF、fused 图与原训练结果，不做 Stage-2，也不采用旧时间/path loss 方案。ADP-affinity 与旧平滑 loss 的 full-N 计划暂停；任何新 loss 完整训练都需要先固定方案并确认新预算。
- 代码/输入：`methods/g1_acf_graph_fcf/{run.py,graph_utils.py}`；数据 `datasets/adp_old_shard4/`，距离缓存 `results/adp_old_fused/stride1_n20997_seed10/`，已有 G1 Graph/Identity 结果在 `results/g1_acf_graph_fcf/`。当前分支和未提交工作树须在继续前核验，不覆盖已有运行。
- 已完成：ARIS 相关流程与 [2609 原论文](https://arxiv.org/html/2609.06204v1) 的关键方法已核对；原结果与临时离线图质量检查已审阅；单一新亲和度分支已实现，5 项 G1 `unittest` 通过，新增测试证明 support 不变、权重变化且对称。真实 stride10 Sanity 已跑完：N=2100、k=35、100 step，训练 8.819892 s，峰值 531593216 bytes；full-N Pilot 已跑完：N=20997、k=20、5 step，训练 27.510492 s，峰值 707292672 bytes。两者均梯度、loss 有限，真实 full-N 图复验 32 源最短路 0 差、无向边 228773；与原 full-N Graph 的索引、位置、4199 指标抽样节点、首个训练点对 hash、验证点对 hash 和参数量相同。阶段完整指标 CSV 与各 `result.json` 逐项一致，已目视检查 Chart。产物分别在 `results/g1_acf_graph_fcf/stride10_n2100_seed10_graph_adp_sanity/`、`results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_adp_pilot/`；阶段报告见 `results/g1_acf_graph_fcf/adp_affinity/REPORT.md`。此外，已完成组合式 loss 的保存图谱梯度核验，代码和报告见上节链接；有限差分、逐点/完整指标、raw/affine 图及 CSV 核对通过。两项正式 full-N 均未运行。
- 未解问题：已确认目标距离相关失配，但其与节点高误差关联弱；不能从本轮判断修正它能改善训练图谱。误差来源检查代码、399983 随机点对索引、228773 边来源、完整分组 CSV 和报告已保存于 `diagnostics/g1_error_source/run.py` 与 `results/g1_acf_graph_fcf/error_source_diagnostic/`；当前分支 `codex/new-stage1`，本轮未提交或上传。下一动作：如继续，先对训练时可得的 ADP/时间差/路径结构做距离和空间匹配后的**只读**目标偏差判别检查，再决定是否有可部署的 loss 调整依据。不自动启动新 loss Pilot 或 full-N。三层深度仍不并入本轮。
