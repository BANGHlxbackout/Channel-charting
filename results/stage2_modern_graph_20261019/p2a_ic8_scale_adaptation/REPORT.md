# P2A-R：B0 Robust-IC 天线数据尺度适配

## 状态与问题

**正式 full-N 已完成并STOP，等待用户审核**：N=20997，20/20配置执行，G2数值有效20/20。小图26项PASS。数值执行日期2026-10-20；中断恢复后的评价与交付日期2026-10-25。
本轮区分异常边分解相对同eta Ball-GLR的增量，与相对S2-J/S2-CG的最终任务收益。全部配置预先冻结；本报告中的最小Mean仅用于完整网格的描述，不构成独立验证集选优。

**结论**：P-active 10/15、IC-distinct 8/15；3个同eta机制正向信号均在eta=.03。IC8相对S2-J/CG的ADVISOR_HIT/SURVIVOR/TARGET_HIT全部为0，任务层面为NO_GO_EXPLORATORY。eta=.03的Mean/P95改善伴随Max显著恶化，不能概括为全面质量提升。

## 冻结目标与实现

`IC8-P: alpha*|P|_1 + (1-alpha)*|sqrt(W_IC)DU-P|²`；x/y独立观测球与原宽box。Ball-GLR使用同入口`aux_mode=disabled`，P固定为0、beta=1，目标严格为`|sqrt(W_IC)DU|²`。没有新增fidelity，也未用Huber近似替换求解目标。
B0、P0缓存D/G128、节点与相对权重不变；W_IC=W/mean(W_e)。eta固定为0.03/0.08/0.16/0.32/0.50，完全不由delta0生成。delta0只保留为历史对照。
运行中用户补充允许灵活调整eta，原方案为初步设计；此前禁止修改eta的限制不再作为后续硬边界。本批已在多个eta找到P-active，故没有为追求激活追加点，完成原20配置的匹配Ball比较。若未来调整必须记录时点和依据，不能把后验GT选优冒充预注册验证。
alpha由池化两坐标输入边梯度的q95/q99/q995计算；5%/1%/0.5%指坐标条目率，边union率另存。参数在本轮GT指标前冻结。

| alpha_source | quantile | kappa | alpha |
|---|---|---|---|
| q95 | 0.950000 | 0.161491 | 0.244131 |
| q99 | 0.990000 | 0.225798 | 0.311103 |
| q995 | 0.995000 | 0.262417 | 0.344191 |

使用原PDHG的IC8更新及投影/图存储；新入口隔离旧27网格。最初使用30000步/至多一次同目标120000步冷启动。用户随后授权调整步数，诊断后仅剩余3个eta=.50 IC8改为单次60000步，无额外重试；已完成17配置不重跑，原核函数/步长/容差不变。当前未完成的q95日志归档为人工预算调整中断，无有效embedding；其已记录耗时计入资源。float64，prox<2e-6、feas<1e-9才停止；不根据GT或迭代上限宣布收敛。
relative raw RMS的主分母是中心化B0 RMS；另存绝对raw RMS、Ball中心化RMS分母、全局与分轴移动余弦。绝对raw差异是原图谱坐标单位，不能当作米。P-active要求非零边union比例>=1e-4，IC-distinct再要求relative raw RMS>=1e-4。

## 数学与数值验收

小图16/32/64节点覆盖两端eta及3个alpha和Ball；最大凸参考相对目标差4.437895e-10，与旧IC8的200步U/P/dual误差均为0。一次CLARABEL optimal_inaccurate改用同显式目标的固定SCS参考通过；原失败日志保留。
20配置最大prox=1.998588e-06，最大feas=7.021004e-17；最小中心化spread/input=0.546307，最大同点比例=4.762585e-05。详见[G2.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/G2.json)。

| config | status | iterations | attempts | prox | feas | time_s | G3 |
|---|---|---|---|---|---|---|---|
| ball_glr_eta003 | VALID | 1950 | 1 | 1.890256e-06 | 0.000000e+00 | 34.754037 | NO_GO |
| ic8_q95_eta003 | VALID | 2800 | 1 | 1.795379e-06 | 0.000000e+00 | 54.252554 | NO_GO |
| ic8_q99_eta003 | VALID | 2700 | 1 | 1.742161e-06 | 4.388128e-18 | 52.504493 | NO_GO |
| ic8_q995_eta003 | VALID | 2700 | 1 | 1.695768e-06 | 0.000000e+00 | 51.677906 | NO_GO |
| ball_glr_eta008 | VALID | 5200 | 1 | 1.851285e-06 | 1.755251e-17 | 89.709098 | NO_GO |
| ic8_q95_eta008 | VALID | 5650 | 1 | 1.836886e-06 | 0.000000e+00 | 105.580514 | NO_GO |
| ic8_q99_eta008 | VALID | 5350 | 1 | 1.830896e-06 | 1.712867e-17 | 100.165340 | NO_GO |
| ic8_q995_eta008 | VALID | 5300 | 1 | 1.936728e-06 | 0.000000e+00 | 99.162271 | NO_GO |
| ball_glr_eta016 | VALID | 10350 | 1 | 1.946794e-06 | 3.510502e-17 | 178.608241 | NO_GO |
| ic8_q95_eta016 | VALID | 10500 | 1 | 1.896171e-06 | 0.000000e+00 | 196.099396 | NO_GO |
| ic8_q99_eta016 | VALID | 10550 | 1 | 1.940734e-06 | 0.000000e+00 | 198.874563 | NO_GO |
| ic8_q995_eta016 | VALID | 10550 | 1 | 1.997623e-06 | 0.000000e+00 | 204.490700 | NO_GO |
| ball_glr_eta032 | VALID | 20650 | 1 | 1.990097e-06 | 7.021004e-17 | 371.058156 | NO_GO |
| ic8_q95_eta032 | VALID | 20950 | 1 | 1.969198e-06 | 0.000000e+00 | 392.149224 | NO_GO |
| ic8_q99_eta032 | VALID | 21050 | 1 | 1.983821e-06 | 0.000000e+00 | 408.477376 | NO_GO |
| ic8_q995_eta032 | VALID | 21100 | 1 | 1.983731e-06 | 0.000000e+00 | 406.103776 | NO_GO |
| ball_glr_eta050 | VALID | 32250 | 2 | 1.985810e-06 | 0.000000e+00 | 1086.859145 | NO_GO |
| ic8_q95_eta050 | VALID | 32700 | 1 | 1.997698e-06 | 6.851468e-17 | 942.035409 | NO_GO |
| ic8_q99_eta050 | VALID | 32850 | 1 | 1.998588e-06 | 6.851468e-17 | 626.258699 | NO_GO |
| ic8_q995_eta050 | VALID | 32950 | 1 | 1.979944e-06 | 0.000000e+00 | 624.152727 | NO_GO |

## 为什么30000步未达标：预算与对偶暂态诊断

Ball-GLR eta=.50首轮prox=4.590913e-06；分解后primal最大3.003541e-16，dual最大4.590913e-06。残差来自对偶暂态，不是约束违例/NaN或持续发散。
15000–30000步的实测指数衰减率约0.000372579/step，理论对偶relaxation log(1+sigma/(2beta))=0.000372465/step，预测达到2e-6约32219–32229步，实际32250步。5000到30000步目标相对变化仅8.064849e-11。
首轮30000步与接受解的U相对差2.383315e-13；直接用精确二次梯度的primal KKT最大1.401236e-13。为保持原完整停止协议，本轮仍不把未收敛的PDHG状态算作VALID。
方案判断：输入尺度alpha、同eta Ball对照和原凸目标合理；旧较小eta网格继承的统一30000步上限不足以覆盖eta=.50。step scale随半径增加导致sigma变小，完整对偶证书变慢；机械冷启动会重复已完成的迭代。这是执行预算适配问题，尚无证据说明目标本身难解。60000约为观测所需步数的1.86倍安全余量，调整在本轮GT评价前完成。未修改步长规则或停止阈值。
[诊断数值](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/solver_budget_diagnosis.json) · [收敛图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/solver_budget_diagnosis.png) · [预算变更](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/solver_budget_amendment.json)

## 全部评价指标

CT/TW/KS复用P0 seed10的4199节点（20%，CT/TW邻居209）；误差在相同20997节点上分别做全数据仿射拟合/评价，单位米。GT仅用于此处评价与绘图；不是held-out或部署泛化。三基线与P0所有8指标复算差均<=1e-10。

| method | alpha_source | eta | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | unavailable | unavailable | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/references/b0/result.json) |
| B0+S2-J | unavailable | unavailable | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558675 | 0.634640 | 1.141431 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/references/s2_j/result.json) |
| B0+S2-CG | unavailable | unavailable | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558674 | 0.634639 | 1.141432 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/references/s2_cg/result.json) |
| Ball-GLR | disabled | 0.030000 | 0.997359 | 0.997334 | 0.074997 | 0.370084 | 0.339753 | 0.654522 | 0.782422 | 4.506750 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ball_glr_eta003/result.json) |
| IC8-P q95 | q95 | 0.030000 | 0.997148 | 0.997305 | 0.075226 | 0.365163 | 0.333298 | 0.643729 | 0.770576 | 6.765776 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q95_eta003/result.json) |
| IC8-P q99 | q99 | 0.030000 | 0.997215 | 0.997297 | 0.075016 | 0.365358 | 0.334014 | 0.644511 | 0.771273 | 6.437548 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q99_eta003/result.json) |
| IC8-P q995 | q995 | 0.030000 | 0.997242 | 0.997291 | 0.074960 | 0.365740 | 0.334435 | 0.645466 | 0.772845 | 6.258804 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q995_eta003/result.json) |
| Ball-GLR | disabled | 0.080000 | 0.998244 | 0.998258 | 0.064846 | 0.319206 | 0.293489 | 0.560399 | 0.640418 | 1.285580 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ball_glr_eta008/result.json) |
| IC8-P q95 | q95 | 0.080000 | 0.998220 | 0.998240 | 0.064964 | 0.320186 | 0.294814 | 0.560647 | 0.640019 | 1.494427 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q95_eta008/result.json) |
| IC8-P q99 | q99 | 0.080000 | 0.998244 | 0.998259 | 0.064845 | 0.319216 | 0.293508 | 0.560334 | 0.640424 | 1.290306 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q99_eta008/result.json) |
| IC8-P q995 | q995 | 0.080000 | 0.998244 | 0.998259 | 0.064845 | 0.319195 | 0.293494 | 0.560340 | 0.640426 | 1.285444 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q995_eta008/result.json) |
| Ball-GLR | disabled | 0.160000 | 0.997935 | 0.997820 | 0.075495 | 0.359047 | 0.326789 | 0.643941 | 0.777024 | 1.316641 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ball_glr_eta016/result.json) |
| IC8-P q95 | q95 | 0.160000 | 0.997898 | 0.997793 | 0.075837 | 0.359881 | 0.327246 | 0.649988 | 0.781436 | 1.319037 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q95_eta016/result.json) |
| IC8-P q99 | q99 | 0.160000 | 0.997935 | 0.997821 | 0.075501 | 0.359018 | 0.326891 | 0.644248 | 0.776613 | 1.316415 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q99_eta016/result.json) |
| IC8-P q995 | q995 | 0.160000 | 0.997935 | 0.997820 | 0.075493 | 0.359025 | 0.326842 | 0.644018 | 0.777034 | 1.316550 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q995_eta016/result.json) |
| Ball-GLR | disabled | 0.320000 | 0.996075 | 0.996149 | 0.119090 | 0.534624 | 0.464433 | 0.998674 | 1.193753 | 2.067657 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ball_glr_eta032/result.json) |
| IC8-P q95 | q95 | 0.320000 | 0.996042 | 0.996131 | 0.119414 | 0.536088 | 0.465351 | 1.000654 | 1.195502 | 2.070630 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q95_eta032/result.json) |
| IC8-P q99 | q99 | 0.320000 | 0.996074 | 0.996148 | 0.119095 | 0.534630 | 0.464300 | 0.998585 | 1.193544 | 2.067736 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q99_eta032/result.json) |
| IC8-P q995 | q995 | 0.320000 | 0.996075 | 0.996149 | 0.119091 | 0.534624 | 0.464433 | 0.998670 | 1.193766 | 2.067660 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q995_eta032/result.json) |
| Ball-GLR | disabled | 0.500000 | 0.993650 | 0.994062 | 0.155054 | 0.653838 | 0.568953 | 1.213486 | 1.449991 | 2.471895 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ball_glr_eta050/result.json) |
| IC8-P q95 | q95 | 0.500000 | 0.993645 | 0.994058 | 0.155087 | 0.653968 | 0.568989 | 1.213693 | 1.450309 | 2.472201 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q95_eta050/result.json) |
| IC8-P q99 | q99 | 0.500000 | 0.993650 | 0.994062 | 0.155054 | 0.653838 | 0.568953 | 1.213486 | 1.449991 | 2.471895 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q99_eta050/result.json) |
| IC8-P q995 | q995 | 0.500000 | 0.993650 | 0.994062 | 0.155054 | 0.653838 | 0.568953 | 1.213486 | 1.449991 | 2.471895 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/ic8_q995_eta050/result.json) |

## 方法机制收益：与匹配eta的Ball-GLR

15个IC配置中P-active=10，IC-distinct=8，精确P=0的QUADRATIC_REGIME=2；预注册正向信号=3。正向信号需数值有效、P-active、IC-distinct，并且KS/Mean/P95至少一项改善>=1%，其它不恶化>2%。表中收益为比例，正值表示改善。

| config | P_nonzero_fraction | relative_raw_RMS_difference_B0 | KS_improvement_vs_Ball | Mean_improvement_vs_Ball | P95_improvement_vs_Ball | MECHANISM_POSITIVE_SIGNAL |
|---|---|---|---|---|---|---|
| ic8_q95_eta003 | 0.049575 | 0.011631 | -0.003051 | 0.013296 | 0.015140 | True |
| ic8_q99_eta003 | 0.007393 | 0.009415 | -0.000254 | 0.012771 | 0.014250 | True |
| ic8_q995_eta003 | 0.003545 | 0.008274 | 0.000499 | 0.011739 | 0.012240 | True |
| ic8_q95_eta008 | 0.026302 | 0.001918 | -0.001831 | -0.003070 | 0.000623 | False |
| ic8_q99_eta008 | 0.001352 | 0.000242 | 0.000014 | -0.000032 | -0.000009 | False |
| ic8_q995_eta008 | 0.000174 | 0.000064 | 0.000016 | 0.000032 | -0.000012 | False |
| ic8_q95_eta016 | 0.008902 | 0.001591 | -0.004540 | -0.002323 | -0.005679 | False |
| ic8_q99_eta016 | 0.000352 | 0.000298 | -0.000080 | 0.000081 | 0.000529 | False |
| ic8_q995_eta016 | 0.000094 | 0.000104 | 0.000021 | 0.000059 | -0.000013 | False |
| ic8_q95_eta032 | 0.001870 | 0.001073 | -0.002722 | -0.002738 | -0.001465 | False |
| ic8_q99_eta032 | 0.000096 | 0.000074 | -0.000041 | -0.000010 | 0.000176 | False |
| ic8_q995_eta032 | 0.000005 | 0.000006 | -0.000004 | -0.000000 | -0.000010 | False |
| ic8_q95_eta050 | 0.000153 | 0.000096 | -0.000215 | -0.000199 | -0.000220 | False |
| ic8_q99_eta050 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False |
| ic8_q995_eta050 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False |

## 最终任务收益：与S2-J/S2-CG

IC8的ADVISOR_HIT=0/15，SURVIVOR=0/15，TARGET_HIT=0/15。ADVISOR_HIT/SURVIVOR/TARGET_HIT继续使用原G3，不把数值成功当性能成功。
描述性最小Mean配置：IC8-P q995 eta=0.08，Mean=0.319195、P95=0.640426、KS=0.064845；匹配Ball的Mean=0.319206、P95=0.640418。该配置的G3=NO_GO，P-active=True，IC-distinct=False。

## 必须回答的研究问题

1. **是否覆盖遗漏激活区间？** 输入尺度上已覆盖：alpha约0.244131/0.311103/0.344191，处于旧0.2与0.5之间；不等于最终P必然激活。最终P-active共10/15，精确零P为2/15。
2. **哪些配置激活、比例是否合理？** P-active：q95的全部5个eta；q99的.03/.08/.16；q995的.03/.08。有效激活边比例范围0.015348%–4.957495%。13/15配置精确P非零，但其中3个低于有效激活门槛。完整P幅度、L1、分位数、条目/边比例均在各mechanism.json。激活稀疏，没有大面积吞噬全部边差异；稀疏本身不能证明这些边就是无线异常边。
3. **是否区别于匹配eta Ball？** 8/15通过IC-distinct；其余不作为异常边机制实际参与的证据。raw差异计算未仿射对齐、未用GT。
4. **同eta是否优于Ball？** 3/15通过预注册机制正向信号；保留全部配对改善/退化。不能只因超越S2就宣称机制收益。
5. **是否超过S2-J/CG？** 依据原G3，IC8 SURVIVOR 0、TARGET_HIT 0；每个配置分别相对J/CG的全精度变化在G3.json。P=0或不可区分Ball的任务收益单列BALL_CONSTRAINT_GAIN。
6. **平均还是尾部？** eta=.03三个IC配置相对匹配Ball：Mean改善1.173870%–1.329649%，Median改善1.565333%–1.899983%，P90改善1.383613%–1.648915%，P95改善1.223998%–1.514022%；Max却恶化38.876209%–50.125364%。收益覆盖典型误差和P90/P95，同时有极端误差损害，不能笼统说“尾部都改善”。
7. **大eta结构退化或坍缩？** 数值硬门槛最小spread=0.546307，最大同点比例=4.762585e-05；全部eta原始/仿射图保留。未把仿射缩放恢复误写为原始尺度保持；具体图谱观察见人工复核段。
8. **旧IC8接近S2的问题解决了吗？** 已解决“无法证明机制参与”的覆盖缺口：P-active=10、IC-distinct=8。没有获得超越S2的任务收益。机制正向信号仅在eta=.03的小修正区；质量接近S2的eta=.08处，q95实际激活但无增益，q99/q995输出接近Ball且无>=1%增益。高质量区域的增量价值仍未建立。eta=.50的q99/q995为精确二次区域。旧P2A的数值与结论不回写。
9. **值得继续研究IC边权鲁棒性？** 仅足够支持一次另行审核的小范围机制核验，尚不足以支持继续大规模性能优化。3个正向信号是探索依据，Max损害和0个S2 SURVIVOR是明确反证边界。当前没有边权扰动证据，不能声称权重鲁棒性；本轮按任务层面NO_GO_EXPLORATORY停止，不自动追加实验。

## 资源、复现与产物

GPU：NVIDIA GeForce RTX 3050 Laptop GPU；float64。累计已记录求解循环6222.974s（含未达标尝试及人工中断前最后记录的耗时）；数值运行段累计墙钟6274.219s（含新产物保存/审计，未计人工诊断暂停间隔）；评价146.522s。峰值Torch allocated VRAM=468262400bytes，采样进程RAM=1367719936bytes；采样100ms，不能视为系统精确RAM峰值。没有训练，train_time_s为空；实际GPU总占用另见gpu_snapshot_during_fulln.csv，包含桌面/其它进程，不能冒充方法独占显存。
所有配置保留显式P/dual/U、solver.json/history.csv、运行配置与资源；不涉及神经网络checkpoint。旧1122个保护文件及冻结参数hash核验保持不变。

```powershell
D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_experiment --phase prepare
D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_sanity
D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_experiment --phase run
D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_evaluate
D:/anaconda3/envs/diffusion_charting/python.exe -m methods.stage2_modern_graph.ic8_scale_report
```
上述代码在新输出目录重放；现目录运行入口只复用已完成配置，不覆盖历史结果。全新重放需在隔离checkout保留相同相对输入路径与固定常量。

[parameter_scan.csv](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/parameter_scan.csv) · [mechanism_summary.csv](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/mechanism_summary.csv) · [G3.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/G3.json) · [manifest.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/manifest.json)

### 各eta标准comparison

- eta=0.03：[完整表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta003/comparison.md) · [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta003/comparison.csv) · [原始图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta003/raw_channel_chart.png) · [仿射图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta003/affine_channel_chart.png)
- eta=0.08：[完整表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta008/comparison.md) · [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta008/comparison.csv) · [原始图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta008/raw_channel_chart.png) · [仿射图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta008/affine_channel_chart.png)
- eta=0.16：[完整表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta016/comparison.md) · [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta016/comparison.csv) · [原始图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta016/raw_channel_chart.png) · [仿射图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta016/affine_channel_chart.png)
- eta=0.32：[完整表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta032/comparison.md) · [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta032/comparison.csv) · [原始图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta032/raw_channel_chart.png) · [仿射图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta032/affine_channel_chart.png)
- eta=0.5：[完整表](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta050/comparison.md) · [CSV](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta050/comparison.csv) · [原始图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta050/raw_channel_chart.png) · [仿射图](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/comparison/eta050/affine_channel_chart.png)

## 人工复核与结论边界

已实际查看5个eta的raw/affine共10张多列图、4张代表性原生图与求解诊断图。使用原plot_chart、GT颜色、散点5/无边线、equal/grid、180dpi；每组固定GT/B0/S2-J/S2-CG/Ball/q95/q99/q995顺序。三参考方法的aligned_charts数组跨eta逐元素一致；图像独立自动轴限不用于判断跨eta缩放。

图谱观察：eta=.03保留较多B0的散点粗糙和个别偏离轨迹的点；三个IC与Ball宏观形状接近，局部点位置确有不同。eta=.08呈现接近S2的轨迹带；eta=.16/.32/.50的绿色横向区域逐渐压窄、红色分支弯曲，仿射对齐后仍呈扇形失真。raw RMS的最小保留比例0.546307、所有节点有限且无重复点坍缩；这属于结构退化，未达到全点/线性坍缩的观察。

机制收益与极端损害：eta=.03 q95对Ball的Mean由0.370084到0.365163m，P95由0.782422到0.770576m；Max却由4.506750到6.765776m。另两种alpha也有Mean/P95的小幅收益及Max恶化。没有独立边错误标签，不能把P解释为已识别无线异常边。

任务收益：本批最小Mean为q995 eta=.08，Mean0.319195m、P950.640426m，相对S2-J分别恶化0.140338%和0.911677%；KS0.064845相对S2-J改善0.338081%。CT/TW略有改善，但没有达到原G3端点，15/15均NO_GO。此点P-active=true但IC-distinct=false，不能用它作为异常边分解成功的证据。

研究判断：alpha适配已覆盖原遗漏的参与区间；单纯激活不足以建立更好的最终Chart。3个同eta正向信号可作为有限机制线索，不能忽略极端误差代价，也不能等同对边权扰动鲁棒。粗eta网格的负结果不是连续参数全局最优证明；收到eta灵活授权后因已找到激活点而未增加点，没有按GT反向补扫。

方案与执行策略：凸目标及同eta对照可执行；原30000步上限沿用旧小eta范围而未适配慢对偶暂态，机械冷重试造成额外计算。用户授权后只改上限，余下3点均在32700/32850/32950步通过原完整停止条件；未放宽门槛。首轮未达标与人工预算调整中断分别保留，不混成有效性能结果。

实际分支`codex/new-stage1`，HEAD`f63652fc3bec725f4fddd1105d13e288cbf233dc`；本轮尚未提交或上传。旧有工作区改动未清理。
**STOP**：不执行P2B/P3/P4/P5、Edge、Stage-1训练、额外网格或边权扰动，等待用户审核。
