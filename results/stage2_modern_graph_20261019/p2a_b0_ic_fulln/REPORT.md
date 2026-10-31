# P2A：B0 Robust-IC full-N（2026-10-20）

## 结论与停止

实际阶段FULL-N，N=20997；27个冻结配置全部执行，数值有效27个。P2A已停止；没有执行P2B/P3/P4/P5、Stage-1重训或Edge调参。Edge仍NUMERICAL_BLOCKER，IC/FN既有G1结论保留。

| 方法 | 数值有效/总数 | SURVIVOR | TARGET_HIT | 家族判定 |
|---|---:|---:|---:|---|
| IC8-P | 9/9 | 2 | 0 | SURVIVOR |
| IC9-P | 9/9 | 0 | 0 | NO_GO_EXPLORATORY |
| IC9-G | 9/9 | 0 | 0 | NO_GO_EXPLORATORY |
| Robust-IC-all | 27/27 | 2 | 0 | SURVIVOR |

## 实际结果与解释

- **只有有限尾部改善**：IC8-P的alpha=0.5/0.8、eta=delta0通过SURVIVOR；以alpha=0.5为代表，相对S2-J，KS改善0.565402%、Mean改善0.260341%、P95改善1.898925%。对CG的对应改善为0.565396%/0.260332%/1.898810%。只有P95超过1%门槛，因此ADVISOR_HIT和TARGET_HIT均为false。两网格点输出几乎相同，不是两次独立复验。
- **IC9是有效负结果**：IC9-P和IC9-G各9个配置全部数值有效，但均无SURVIVOR；按冻结门槛均NO_GO_EXPLORATORY。中点IC9-P相对J的KS/Mean/P95恶化9.758851%/6.639979%/3.240918%；IC9-G恶化9.287778%/7.392743%/5.106757%。不能将其归咎于未收敛。
- **二维group没有一致增益**：同alpha=0.5、eta=delta0且实际delta相同，IC9-G相对IC9-P的KS改善0.429190%，Mean退化0.705893%，P95退化1.807267%。1.5delta0下KS与Mean略改善而P95仍退化；0.5delta0两者近乎相同。全部配对和参数均保留，没有换参数比较。
- **辅助变量的实际作用边界**：IC9-P/G共18个最终状态的P均精确为0。此时固定观测球下目标简化为beta*||L_IC U||（标量或Fro范数），alpha对该已实现解主要只改变正比例系数，因此alpha扫描几乎重复。这里未把P固定为0；完整prox及激活回放始终保留。IC8-P有配置出现非零P（最多173563条边），但两个SURVIVOR配置P也为0；这两点的收益不能作为稀疏辅助项或边权不确定性鲁棒性的正面机制证据。
- **图谱观察**：已实际查看中点J、IC8-P、IC9-P/G的两列图及raw/affine多列图。IC8-P与J整体轮廓几乎重合，符合小幅指标变化；IC9保持整体连续形状但边界/转角更圆滑，在大修正预算下红色分支更窄，KS和尾部误差反而升高。没有整体坍缩；视觉平滑不等于几何质量改善，也没有凭肉眼宣称某局部区域已修复。
- **结论边界与停止**：当前支持一个接近二次平滑、尾部略有改善的IC8数值点；不支持当前冻结IC9-P/G在此B0/G128上的优越性。G3是单数据探索性筛选，不构成多种子显著性、跨数据泛化、论文创新或扰动鲁棒性证明。P2A在此停止；不扩G1 Charts，也不执行P2B/P3/P4/P5。

## 固定输入、目标、数值与协议

- 输入为已核验P0的B0、G128、delta0=0.09517167367990964、S2-J/CG；原图1544167无向边，只读复用。IC权重仅除以无向正边均值0.5592081927929078。没有变更图/相对权重或使用UE GT训练。
- 固定alpha={0.2,0.5,0.8} × eta={0.5,1,1.5}delta0 × IC8-P/IC9-P/IC9-G，共27。标量模型x/y独立球与box；group二维共球无分轴box。原始IC8平方与IC9非平方D.T(WDU-P)保持不变。
- 原P1合成入口不适合full-N及独立阶段；新增ic_cuda.py和run_ic_fulln.py/evaluate_ic_fulln.py。仅执行相同目标PDHG，float64稀疏CUDA。步长固定按观测球半径/beta平衡比例，乘积tau*sigma*norm_bound=.9025不变；每配置记录实际tau/sigma。没有基于GT/性能改动数值参数。
- CUDA graph批处理50步；只有数学充分证书保证辅助P仍为0时才跳过其冗余计算。证书/box失效则恢复并按完整原迭代重放该chunk，不限制P=0。GPU与CPU同一步长逐迭代误差约1e-16；含active aux及独立凸reference测试均通过。
- 初次30000步；仅同一目标一次冷启动重试120000步。prox<2e-6、feas停止<1e-9；G2硬门槛ball<=1e-4、finite/节点一致、无持续目标恶化、中心化RMS>=输入0.05、最大同点比例<0.1%。没有因达到上限标收敛。
- CT/TW/KS使用P0同一固定4199节点；误差指标使用相同全部20997点，各方法独立仿射拟合/评价。直接复用common_metrics和run_adp_siamese.plot_chart。GT仅评价/绘图；全数据传导式结果，不代表held-out部署或边权扰动鲁棒性。
- 只有数值有效配置进入性能评价；未收敛配置保留状态/日志/诊断图，指标留空。网格数值不完整时标G3_INCOMPLETE_NUMERICAL，不能解释成方法性能NO_GO。

数值与全参数扫描：[parameter_scan.csv](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/parameter_scan.csv)；G2检查[G2.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/G2.json)；实际数值资源/每个attempt状态与完整轨迹见各配置目录。Sanity100/1000步检查单独标SANITY，不是正式完成结果。

## 全部有效结果指标

# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/references/b0/result.json) | unavailable | unavailable |
| B0+S2-J | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558675 | 0.634640 | 1.141431 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/references/s2_j/result.json) | unavailable | unavailable |
| B0+S2-CG | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558674 | 0.634639 | 1.141432 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/references/s2_cg/result.json) | unavailable | unavailable |
| IC8-P a=0.2 eta=0.5d0 | 0.997668 | 0.997719 | 0.070720 | 0.345193 | 0.317932 | 0.606627 | 0.720304 | 5.676128 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a20_eta050/result.json) | unavailable | 468262912 |
| IC8-P a=0.2 eta=1d0 | 0.998154 | 0.998150 | 0.065260 | 0.320782 | 0.299721 | 0.553076 | 0.628414 | 1.136871 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a20_eta100/result.json) | unavailable | 468262912 |
| IC8-P a=0.2 eta=1.5d0 | 0.997908 | 0.997835 | 0.072454 | 0.346093 | 0.317125 | 0.610252 | 0.733579 | 1.272069 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a20_eta150/result.json) | unavailable | 468262912 |
| IC8-P a=0.5 eta=0.5d0 | 0.997927 | 0.997955 | 0.070276 | 0.347487 | 0.319950 | 0.615430 | 0.734243 | 2.816398 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a50_eta050/result.json) | unavailable | 468262912 |
| IC8-P a=0.5 eta=1d0 | 0.998255 | 0.998238 | 0.064697 | 0.317918 | 0.294713 | 0.549514 | 0.622589 | 1.129928 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a50_eta100/result.json) | unavailable | 468262912 |
| IC8-P a=0.5 eta=1.5d0 | 0.998062 | 0.997960 | 0.071424 | 0.343199 | 0.315542 | 0.596762 | 0.725855 | 1.266139 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a50_eta150/result.json) | unavailable | 468262912 |
| IC8-P a=0.8 eta=0.5d0 | 0.997926 | 0.997954 | 0.070288 | 0.347535 | 0.319992 | 0.615654 | 0.734236 | 2.766965 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a80_eta050/result.json) | unavailable | 468262912 |
| IC8-P a=0.8 eta=1d0 | 0.998255 | 0.998238 | 0.064697 | 0.317918 | 0.294713 | 0.549514 | 0.622589 | 1.129928 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a80_eta100/result.json) | unavailable | 468262912 |
| IC8-P a=0.8 eta=1.5d0 | 0.998062 | 0.997960 | 0.071424 | 0.343199 | 0.315542 | 0.596762 | 0.725855 | 1.266139 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a80_eta150/result.json) | unavailable | 468262912 |
| IC9-P a=0.2 eta=0.5d0 | 0.997849 | 0.997864 | 0.073810 | 0.367350 | 0.338916 | 0.658345 | 0.794416 | 2.174598 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a20_eta050/result.json) | unavailable | 418604032 |
| IC9-P a=0.2 eta=1d0 | 0.997858 | 0.997688 | 0.071414 | 0.339913 | 0.314538 | 0.582498 | 0.655208 | 1.187498 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a20_eta100/result.json) | unavailable | 418604032 |
| IC9-P a=0.2 eta=1.5d0 | 0.997258 | 0.997009 | 0.085175 | 0.384945 | 0.347787 | 0.715115 | 0.859139 | 1.406826 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a20_eta150/result.json) | unavailable | 418604032 |
| IC9-P a=0.5 eta=0.5d0 | 0.997849 | 0.997864 | 0.073810 | 0.367350 | 0.338916 | 0.658345 | 0.794416 | 2.174598 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a50_eta050/result.json) | unavailable | 418604032 |
| IC9-P a=0.5 eta=1d0 | 0.997858 | 0.997688 | 0.071414 | 0.339913 | 0.314538 | 0.582498 | 0.655208 | 1.187498 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a50_eta100/result.json) | unavailable | 418604032 |
| IC9-P a=0.5 eta=1.5d0 | 0.997258 | 0.997009 | 0.085175 | 0.384945 | 0.347787 | 0.715115 | 0.859139 | 1.406826 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a50_eta150/result.json) | unavailable | 418604032 |
| IC9-P a=0.8 eta=0.5d0 | 0.997849 | 0.997864 | 0.073810 | 0.367350 | 0.338916 | 0.658345 | 0.794416 | 2.174598 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a80_eta050/result.json) | unavailable | 418604032 |
| IC9-P a=0.8 eta=1d0 | 0.997858 | 0.997688 | 0.071414 | 0.339913 | 0.314538 | 0.582498 | 0.655208 | 1.187498 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a80_eta100/result.json) | unavailable | 418604032 |
| IC9-P a=0.8 eta=1.5d0 | 0.997258 | 0.997009 | 0.085175 | 0.384945 | 0.347787 | 0.715115 | 0.859139 | 1.406826 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a80_eta150/result.json) | unavailable | 418604032 |
| IC9-G a=0.2 eta=0.5d0 | 0.997849 | 0.997863 | 0.073812 | 0.367363 | 0.338964 | 0.658190 | 0.794415 | 2.170456 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a20_eta050/result.json) | unavailable | 393438208 |
| IC9-G a=0.2 eta=1d0 | 0.997826 | 0.997685 | 0.071108 | 0.342312 | 0.312169 | 0.589706 | 0.667049 | 1.186563 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a20_eta100/result.json) | unavailable | 393438208 |
| IC9-G a=0.2 eta=1.5d0 | 0.997235 | 0.997044 | 0.084455 | 0.384019 | 0.340214 | 0.713257 | 0.870748 | 1.527629 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a20_eta150/result.json) | unavailable | 393438208 |
| IC9-G a=0.5 eta=0.5d0 | 0.997849 | 0.997863 | 0.073812 | 0.367363 | 0.338964 | 0.658190 | 0.794415 | 2.170456 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a50_eta050/result.json) | unavailable | 393438208 |
| IC9-G a=0.5 eta=1d0 | 0.997826 | 0.997685 | 0.071108 | 0.342312 | 0.312169 | 0.589706 | 0.667049 | 1.186563 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a50_eta100/result.json) | unavailable | 393438208 |
| IC9-G a=0.5 eta=1.5d0 | 0.997235 | 0.997044 | 0.084455 | 0.384019 | 0.340214 | 0.713257 | 0.870748 | 1.527629 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a50_eta150/result.json) | unavailable | 393438208 |
| IC9-G a=0.8 eta=0.5d0 | 0.997849 | 0.997863 | 0.073812 | 0.367363 | 0.338964 | 0.658190 | 0.794415 | 2.170456 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a80_eta050/result.json) | unavailable | 393438208 |
| IC9-G a=0.8 eta=1d0 | 0.997826 | 0.997685 | 0.071108 | 0.342312 | 0.312169 | 0.589706 | 0.667049 | 1.186563 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a80_eta100/result.json) | unavailable | 393438208 |
| IC9-G a=0.8 eta=1.5d0 | 0.997235 | 0.997044 | 0.084455 | 0.384019 | 0.340214 | 0.713257 | 0.870748 | 1.527629 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a80_eta150/result.json) | unavailable | 393438208 |


## G3 与配对解释

KS/Mean/P95均越小越好；改善率=(baseline-method)/baseline。ADVISOR_HIT要求对J至少两项>=1%、第三项不劣>1%；SURVIVOR要求分别对J和CG至少一项>=1%、其余不劣>2%；TARGET_HIT分别对两者至少两项>=1%、第三项不劣>1%。阈值用全精度，不按显示舍入判断。全部有效且没有SURVIVOR才NO_GO_EXPLORATORY；仅超过J的结果标SOLVER_LEVEL_GAIN_ONLY。
每配置布尔判据和对J/CG的实际改善率：[G3.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/G3.json)；IC9-G/P同alpha/eta、实际delta及全指标：[ic9_group_paired_comparison.csv](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9_group_paired_comparison.csv)。没有选择不同最优参数冒充几何适配消融。

| alpha | eta/delta0 | 两者有效 | delta_P | delta_G | KS_P | KS_G | Mean_P | Mean_G | P95_P | P95_G |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.5 | True | 0.047586 | 0.047586 | 0.073810 | 0.073812 | 0.367350 | 0.367363 | 0.794416 | 0.794415 |
| 0.2 | 1.0 | True | 0.095172 | 0.095172 | 0.071414 | 0.071108 | 0.339913 | 0.342312 | 0.655208 | 0.667049 |
| 0.2 | 1.5 | True | 0.142758 | 0.142758 | 0.085175 | 0.084455 | 0.384945 | 0.384019 | 0.859139 | 0.870748 |
| 0.5 | 0.5 | True | 0.047586 | 0.047586 | 0.073810 | 0.073812 | 0.367350 | 0.367363 | 0.794416 | 0.794415 |
| 0.5 | 1.0 | True | 0.095172 | 0.095172 | 0.071414 | 0.071108 | 0.339913 | 0.342312 | 0.655208 | 0.667049 |
| 0.5 | 1.5 | True | 0.142758 | 0.142758 | 0.085175 | 0.084455 | 0.384945 | 0.384019 | 0.859139 | 0.870748 |
| 0.8 | 0.5 | True | 0.047586 | 0.047586 | 0.073810 | 0.073812 | 0.367350 | 0.367363 | 0.794416 | 0.794415 |
| 0.8 | 1.0 | True | 0.095172 | 0.095172 | 0.071414 | 0.071108 | 0.339913 | 0.342312 | 0.655208 | 0.667049 |
| 0.8 | 1.5 | True | 0.142758 | 0.142758 | 0.085175 | 0.084455 | 0.384945 | 0.384019 | 0.859139 | 0.870748 |

## 图谱、资源与复现

comparison/根目录保留全部有效配置大表/原始及仿射图；每组固定alpha/eta的子目录另有清晰的GT / B0 / S2-J / S2-CG / IC8-P / IC9-P / IC9-G对比（数值无效者不混入有效比较）。各配置保留标准chart.png、embedding.npz、normalized_output.npz、result.json及每次attempt状态/轨迹。

- alpha=0.2, eta=0.5delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta050/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta050/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta050/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta050/7_column_affine_channel_chart.png)。
- alpha=0.2, eta=1.0delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta100/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta100/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta100/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta100/7_column_affine_channel_chart.png)。
- alpha=0.2, eta=1.5delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta150/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta150/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta150/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a20_eta150/7_column_affine_channel_chart.png)。
- alpha=0.5, eta=0.5delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta050/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta050/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta050/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta050/7_column_affine_channel_chart.png)。
- alpha=0.5, eta=1.0delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta100/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta100/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta100/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta100/7_column_affine_channel_chart.png)。
- alpha=0.5, eta=1.5delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta150/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta150/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta150/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a50_eta150/7_column_affine_channel_chart.png)。
- alpha=0.8, eta=0.5delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta050/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta050/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta050/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta050/7_column_affine_channel_chart.png)。
- alpha=0.8, eta=1.0delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta100/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta100/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta100/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta100/7_column_affine_channel_chart.png)。
- alpha=0.8, eta=1.5delta0：[表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta150/comparison.md) / [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta150/comparison.csv) / [raw](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta150/7_column_raw_channel_chart.png) / [affine](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/comparison/a80_eta150/7_column_affine_channel_chart.png)。

| 配置 | 状态 | attempt | iterations | prox max | delta | 求解loop秒 | attempt墙钟秒 | RAM峰值bytes | VRAM峰值bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IC8-P a=0.2 eta=0.5d0 | VALID | 1 | 4150 | 1.745328e-06 | 0.047586 | 77.374 | 77.473 | 1339936768 | 468262912 |
| IC8-P a=0.2 eta=1d0 | VALID | 1 | 6600 | 1.913163e-06 | 0.095172 | 123.317 | 123.412 | 1341231104 | 468262912 |
| IC8-P a=0.2 eta=1.5d0 | VALID | 1 | 9300 | 1.917296e-06 | 0.142758 | 174.048 | 174.148 | 1341009920 | 468262912 |
| IC8-P a=0.5 eta=0.5d0 | VALID | 1 | 3300 | 1.997941e-06 | 0.047586 | 61.765 | 61.882 | 1341370368 | 468262912 |
| IC8-P a=0.5 eta=1d0 | VALID | 1 | 6400 | 1.820038e-06 | 0.095172 | 118.561 | 118.973 | 1308606464 | 468262912 |
| IC8-P a=0.5 eta=1.5d0 | VALID | 1 | 9550 | 1.933209e-06 | 0.142758 | 178.617 | 178.713 | 1341038592 | 468262912 |
| IC8-P a=0.8 eta=0.5d0 | VALID | 1 | 3300 | 1.775494e-06 | 0.047586 | 61.735 | 61.832 | 1262960640 | 468262912 |
| IC8-P a=0.8 eta=1d0 | VALID | 1 | 6550 | 1.933763e-06 | 0.095172 | 122.632 | 122.730 | 1262665728 | 468262912 |
| IC8-P a=0.8 eta=1.5d0 | VALID | 1 | 9800 | 1.987341e-06 | 0.142758 | 183.302 | 183.409 | 1263042560 | 468262912 |
| IC9-P a=0.2 eta=0.5d0 | VALID | 1 | 200 | 1.320353e-07 | 0.047586 | 5.349 | 5.516 | 1273253888 | 418604032 |
| IC9-P a=0.2 eta=1d0 | VALID | 1 | 1350 | 1.992984e-06 | 0.095172 | 11.830 | 11.976 | 1273262080 | 418604032 |
| IC9-P a=0.2 eta=1.5d0 | VALID | 1 | 2800 | 1.814888e-06 | 0.142758 | 24.559 | 24.708 | 1273782272 | 418604032 |
| IC9-P a=0.5 eta=0.5d0 | VALID | 1 | 200 | 1.321970e-07 | 0.047586 | 1.761 | 1.907 | 1273827328 | 418604032 |
| IC9-P a=0.5 eta=1d0 | VALID | 1 | 1400 | 1.450317e-06 | 0.095172 | 12.183 | 12.343 | 1340780544 | 418604032 |
| IC9-P a=0.5 eta=1.5d0 | VALID | 1 | 2800 | 1.826004e-06 | 0.142758 | 24.532 | 24.674 | 1273839616 | 418604032 |
| IC9-P a=0.8 eta=0.5d0 | VALID | 1 | 200 | 1.323591e-07 | 0.047586 | 1.763 | 1.905 | 1273860096 | 418604032 |
| IC9-P a=0.8 eta=1d0 | VALID | 1 | 1400 | 1.456620e-06 | 0.095172 | 12.275 | 12.421 | 1273860096 | 418604032 |
| IC9-P a=0.8 eta=1.5d0 | VALID | 1 | 2800 | 1.837257e-06 | 0.142758 | 24.546 | 24.686 | 1273860096 | 418604032 |
| IC9-G a=0.2 eta=0.5d0 | VALID | 1 | 200 | 5.820184e-08 | 0.047586 | 5.368 | 5.502 | 1273602048 | 393438208 |
| IC9-G a=0.2 eta=1d0 | VALID | 1 | 1200 | 1.448522e-06 | 0.095172 | 10.508 | 10.653 | 1273602048 | 393438208 |
| IC9-G a=0.2 eta=1.5d0 | VALID | 1 | 2350 | 1.904366e-06 | 0.142758 | 20.555 | 20.695 | 1273606144 | 393438208 |
| IC9-G a=0.5 eta=0.5d0 | VALID | 1 | 200 | 5.824880e-08 | 0.047586 | 1.759 | 1.898 | 1273606144 | 393438208 |
| IC9-G a=0.5 eta=1d0 | VALID | 1 | 1200 | 1.452084e-06 | 0.095172 | 10.429 | 10.568 | 1341235200 | 393438208 |
| IC9-G a=0.5 eta=1.5d0 | VALID | 1 | 2350 | 1.910638e-06 | 0.142758 | 20.580 | 20.734 | 1273610240 | 393438208 |
| IC9-G a=0.8 eta=0.5d0 | VALID | 1 | 200 | 5.829583e-08 | 0.047586 | 1.754 | 1.894 | 1273610240 | 393438208 |
| IC9-G a=0.8 eta=1d0 | VALID | 1 | 1200 | 1.455663e-06 | 0.095172 | 10.497 | 10.646 | 1273610240 | 393438208 |
| IC9-G a=0.8 eta=1.5d0 | VALID | 1 | 2350 | 1.916951e-06 | 0.142758 | 20.561 | 20.701 | 1273610240 | 393438208 |

以上是后处理求解耗时，非训练耗时。solver loop包含数值日志/状态传回；attempt墙钟包含CUDA graph准备，不含最终压缩写盘；RAM为求解期间100ms采样working set，VRAM为PyTorch allocator峰值（含常驻图），非整机其它进程显存。完整网格墙钟包含写盘，见manifest.run_wall_time_s。baseline历史资源未知，不填造值。
代码与复现：[run_ic_fulln.py](D:/diffusion_refined_charting/methods/stage2_modern_graph/run_ic_fulln.py) / [ic_cuda.py](D:/diffusion_refined_charting/methods/stage2_modern_graph/ic_cuda.py) / [evaluate_ic_fulln.py](D:/diffusion_refined_charting/methods/stage2_modern_graph/evaluate_ic_fulln.py)；环境、输入hash、冻结配置与命令见[manifest.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/manifest.json)。未修改AGENTS、原baseline、P0输入/缓存、P1/FN/Edge；当前分支codex/new-stage1、HEAD f63652fc3bec725f4fddd1105d13e288cbf233dc，未提交/上传。

```powershell
D:/anaconda3/envs/diffusion_charting/python.exe -m unittest tests.test_stage2_ic_cuda -v
D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.run_ic_fulln --phase prepare
D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.run_ic_fulln --phase run
D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.evaluate_ic_fulln
```

已完成配置按config与source记录恢复，不重跑；独立复现应使用空输出目录/副本。若存在未完成attempt目录，入口主动报错，避免擅自追加数值重试。

**STOP：仅交付P2A真实B0结果，等待用户审核，不自动进入其它阶段。**

数值补充：27配置均首轮收敛，无重试；最大prox=1.9979413021448008e-6，最大feas=3.510502175218163e-17，CPU/CUDA objective最大相对差6.344453722320055e-16。全扫描墙钟1344.5738482000306秒（22.409564分钟），solver loop合计1322.1568219998153秒；峰值PyTorch显存468262912bytes，求解RSS采样峰值1341370368bytes。

两个IC8存活点normalized输出相对差=2.210902e-11；解释证据见 [interpretation_evidence.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/interpretation_evidence.json)。三份冻结baseline的重算指标差全部精确为0，见 [baseline_evaluation_checks.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/baseline_evaluation_checks.json)。
