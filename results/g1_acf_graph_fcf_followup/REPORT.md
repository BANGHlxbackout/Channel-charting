# G1 后续实验：E1 固定 Stage-2 组合比较与 E2 冻结传播诊断

日期：2026-10-17。**E1→E2 已按顺序完成，均为 full-N 后处理/推理，没有 G1 重训。**

结论：当前两种 G1 初始 chart 未改善固定导师 Stage-2 完整流程；训练好的 Graph 确实依赖邻接，但这种依赖未形成优于重新训练 Identity 对照的几何收益。

## 1. 对两份 GPT 文档的独立判断

实际文件为 `docs/tasks/G1_ACF_2026-10-17.md` 与 `docs/tasks/G1_S2_2026-10-17.md`，标题对应用户所指阶段结论和下一轮执行指令。第一份保留原样，既有数值及单种子、传导式、整体 G1 与图传播贡献分开的边界成立。第二份的 E1/E2 可执行，未发现要求修改训练目标或使用 GT 学习的阻塞性设计问题，但须补充下列边界：

1. E1 问的是相同冻结后处理下的最终质量，不是 GCN 新算法是否普遍有效。相同 S2 是线性、保持常数的迭代，原始 chart 的整体旋转/平移/尺度不会要求额外 GT 调参；各自仿射评价维持原协议。
2. E2 只检验固定权重对邻接替换的敏感性。换 I 同时改变训练时输入分布，不能替代已训练 Graph vs 已训练 Identity，也不能单独判定信息冗余、残差遮蔽或错误邻居哪一个是原因。Identity 中也有 Linear/LayerNorm/ReLU 更新，delta 大不等于邻居贡献大。
3. 文档中的“明显/实际幅度”没有预设数值阈值。本轮只报告完整绝对/相对差与复现数值误差，不事后选择阈值或宣称统计显著性。
4. 代码审查新发现：原 `validation_pairs` 对随机无序 key 排序后取前 200,000 个，会偏向低索引；本次实际所有 pair 的较小端点都不超过 **6,155**。这不是 GT 泄漏，两模式共享同一规则和 hash，但它不代表全体点对均匀验证。当前训练结果及 checkpoint 不改，E2 stress 仅对这一冻结集合解释。
5. 第二份指令提及历史 S2 的“既有 GT 选参 provenance”。本地 registry 仍将 lambda sweep 的选择协议标为待核验，因此不能把历史选参过程写成已经独立核实。本轮只沿用 k=128、lambda=0.15、200 次，不进行新选参。

## 2. E1：输入、算子与复现

三组 `indices`、`positions` 逐元素一致，N=20,997；D_geo 全矩阵有限且没有非对角零值。GT 只进入最后评价/绘图，不传给 refinement。

直接 import `run_diffusion_refined_charting.py` 的三个原函数：top_k=512、graph_k=128、lambda_smooth=0.15、200 次 Jacobi。新 neighbor 缓存只写本轮 `e1_stage2/cache/`，两次重算逐项一致，且与旧源缓存无任何不同 entry。同一个 graph 对象有 **3,088,334** 个 entry，三组仅改变初始 Y。图签名 `a800ec4a21980923f11ccc800b2c040df77be0eb306fa1b89143f7bb36b1a77a`。

历史 Stage-2 JSON 从 Git `61178b5` 只读取得，原目录和历史 embedding 未覆盖。B0+S2 Mean 复现到 **0.31874796748161316 m**；KS 与历史值差约 4.4e-9。CUDA index_add 的重复求解 raw 最大差 0.000274658，满足冻结 atol=1e-4、rtol=1e-5 的 allclose；不宣称逐位一致。保存的历史 B0+S2 raw embedding 最大差 0.000244141。

全部六组从 embedding 共同重算：CT/TW/KS 读取原来的同一 4,199 节点子集；全部 20,997 节点分别全数据仿射拟合并计算米制误差。source_json、评价样本与对齐协议保存在 comparison 中。表中原三组训练耗时来自既有记录，本轮训练耗时为零；新增 S2/推理耗时另列于资源段落。

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [result.json](D:/diffusion_refined_charting/results/adp_old_siamese/stride1_n20997_seed10/result.json) | unavailable | unavailable |
| G1-Identity | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/result.json) | 5292.062650499996 | 698310656 |
| G1-Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/result.json) | 6224.643790000002 | 707461120 |
| B0+S2 | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558675 | 0.634640 | 1.141432 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e1_stage2/b0/result.json) | unavailable | 111771648 |
| G1-Identity+S2 | 0.998164 | 0.998159 | 0.066559 | 0.323004 | 0.296902 | 0.565467 | 0.648439 | 1.146562 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e1_stage2/g1_identity/result.json) | unavailable | 111771648 |
| G1-Graph+S2 | 0.998152 | 0.998142 | 0.066692 | 0.324695 | 0.299163 | 0.565541 | 0.649174 | 1.132579 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e1_stage2/g1_graph/result.json) | unavailable | 111771648 |

**E1 判断**：

- KS/Mean/P95 三项均为 B0+S2 最好，其次 Identity+S2，最后 Graph+S2。Identity+S2 与 Graph+S2 的 Mean 分别比 B0+S2 高 0.004256/0.005947 m；P95 高 0.013800/0.014534 m。Stage-1 的小幅收益未转化为更好的完整 S2 pipeline。
- Graph+S2 比 Identity+S2 的 KS 高 0.000133、Mean 高 0.001691 m、P95 高 0.000734 m。Graph+S2 的 Max 低 0.013983 m，但这一极端端点优势不能改写主要端点均较差的事实。
- 图上三种 +S2 均比各自 Stage-1 更紧凑，三份 refined 轮廓接近；没有 Graph+S2 独有的明显结构修复。图的整齐程度不能替代共同指标排序。

原始/仿射七列图和完整 CSV：`e1_stage2/comparison/`。这是 GT 加六组的七列图，不覆盖旧 G1 三组 comparison。

## 3. E2：冻结权重复现与敏感性

使用原 best checkpoint，固定 `eval()`、关闭梯度及 dropout；参数和 BN buffer 前后签名相同。两个原生配置先复现后才执行 Graph checkpoint @ I。Graph native 全节点 raw 最大差 **0.0000610352**，Identity native 最大差 **0**，满足预设 atol=rtol=1e-4；原验证 stress 均精确复现。特征与邻接复用原定义和缓存，32 源测地距离复验最大差 0。

验证点对 hash 与原训练完全一致：`86664796b714af93838f11d8c37aebc3b895913671b9347d231072aed4772f80`。在该固定有偏点对集合上：

| 冻结权重 / 邻接 | validation stress |
|---|---:|
| Graph checkpoint @ Graph | 1.872658 |
| Graph checkpoint @ I | 3.208952 |
| Identity checkpoint @ I | 1.862524 |

Graph 改为 I 后 stress 增加 **1.336294（71.36%）**。同一 Graph checkpoint 的节点 raw 位移 Mean=15.747149、Median=12.967083、P95=37.699429，单位为原始 chart 坐标而非米；相对原输出中心化 RMS 的变化为 **9.26%**，远大于 native 浮点复现误差。

两层逐节点比率（L2 norm；h 范数分母下限 eps=1e-12；完整样本数组及分布另存）：

| checkpoint / 邻接 | 层 | delta/h P10 | 中位数 | P90 | (Ah-h)/h P10 | 中位数 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Graph ckpt @ Graph | 1 | 0.262313 | 0.859600 | 1.276910 | 0.152682 | 0.357220 | 0.565046 |
| Graph ckpt @ Graph | 2 | 0.236212 | 0.597705 | 0.842199 | 0.121671 | 0.239588 | 0.390652 |
| Graph ckpt @ I | 1 | 0.276209 | 0.823934 | 1.197916 | 0.000000 | 0.000000 | 0.000000 |
| Graph ckpt @ I | 2 | 0.244022 | 0.594373 | 0.871347 | 0.000000 | 0.000000 | 0.000000 |
| Identity ckpt @ I | 1 | 0.226609 | 0.624670 | 1.111791 | 0.000000 | 0.000000 | 0.000000 |
| Identity ckpt @ I | 2 | 0.192164 | 0.322559 | 0.419198 | 0.000000 | 0.000000 | 0.000000 |

Graph native 的 delta/h 中位数为 0.860、0.598，(Ah-h)/h 中位数为 0.357、0.240，不能解释为传播项近乎消失。反过来，两个 Identity 邻接配置的 delta 也不为零，而 Ah-h 严格为零，验证了“delta 大不能等价于邻居作用大”。

完整离线图谱评价如下，仅用于展示冻结替换的后果，不把 probe 当成新训练方法主排名：

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Graph ckpt @ Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e2_propagation/graph_at_graph/result.json) | unavailable | 400580096 |
| Graph ckpt @ I | 0.991064 | 0.990633 | 0.099622 | 0.491343 | 0.438520 | 0.872250 | 1.025123 | 8.185935 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e2_propagation/graph_at_identity/result.json) | unavailable | 400580096 |
| Identity ckpt @ I | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e2_propagation/identity_at_identity/result.json) | unavailable | 400580096 |

换 I 后的米制 Mean 从 0.399771 上升到 0.491343，P95 从 0.846337 上升到 1.025123，Max 升到 8.185935 m；图上条纹变散、局部偏离变大。**结果不支持“Graph 没用到邻接”或“残差项非常弱”的简单解释；它依赖训练时邻接，但这仍不等于比重新训练 Identity 更好。**这些观测没有在剩余解释之间作唯一归因。

E2 的 layer_statistics.csv、逐节点 layer_values.npz、embedding_changes.npz、validation_keys.npy、三组 result.json、comparison 与图片均在 `e2_propagation/`。

## 4. 资源、故障与可复现性

E1 完整计算/评价耗时 27.17 s；下列为各次稀疏求解测量，**不含 top-k 构造阶段的显存峰值**，该构造峰值本次没有单独留存：

- B0+S2：求解 0.418 s，稀疏求解阶段峰值分配 0.104 GiB，线性系统相对残差 2.745e-06。
- G1-Identity+S2：求解 0.344 s，稀疏求解阶段峰值分配 0.104 GiB，线性系统相对残差 2.815e-06。
- G1-Graph+S2：求解 0.334 s，稀疏求解阶段峰值分配 0.104 GiB，线性系统相对残差 2.757e-06。

E2 含图复验、冻结推理与评价共 38.80 s；三次前向分别为 4.120/1.538/1.434 s，程序峰值分配 0.373 GiB（不代表系统全部显存）。没有训练预算消耗或 checkpoint 选择。

第一次 E1 启动在 CUDA 尚未 lazy-init 时重置显存统计报错，图尚未计算；只修正新脚本的统计初始化顺序，原算法未改。失败日志保留为 `e1_startup_failed.log`，正式运行日志是 `e1_console.log`/`e2_console.log`。之后无 OOM、NaN、复现失败或中断。

代码：`methods/g1_acf_graph_fcf/run_stage2_comparison.py`、`diagnose_graph_propagation.py`；`compare.py` 仅扩展任意方法数与读取原评价子集，保持旧三组 CLI。原 S2/GCN 数学实现、历史 embedding/模型/原始 JSON/源缓存未改。测试：原 5 项，加 E1 2 项、E2 3 项，共 10 项通过；检查旧算法小图等价、共享 graph、相同 subset、hooks 不改变输出或 BN 状态、eval/direct 与 streaming 一致。

正式命令（仓库根目录，Python 为 diffusion_charting 环境）：

```powershell
python -m methods.g1_acf_graph_fcf.run_stage2_comparison --device cuda:0 --out-root results/g1_acf_graph_fcf_followup
python -m methods.g1_acf_graph_fcf.diagnose_graph_propagation --device cuda:0 --out-root results/g1_acf_graph_fcf_followup
python -m unittest tests.test_g1_acf_graph_fcf tests.test_g1_stage2_comparison tests.test_g1_graph_propagation_diagnostic -q
```

完整输入/初始分支和工作树状态见 `preflight.json`；E1/E2 的 graph、checkpoint、source_json、运行配置与检查结果见各自 manifest。当前分支 `codex/new-stage1`，本轮改动未提交或上传。

## 5. 研究决定与停止边界

我建议**不将当前 fused-Graph 配置作为下一阶段主候选继续重复训练**：同构 Identity 在 Stage-1 主要误差更好，固定 S2 后它们又都不及 B0+S2。Identity 保留为有用的 Stage-1 架构/训练对照，不能宣称它已替代导师完整 baseline。

用户保留的 G1 探索方向不因此被整体否定。若继续投入，需要先形成更具体、能被区分的图关系问题；当前证据尚不能在“监督冗余”和“聚合关系不合适”等解释间选定一个。此处是基于已完成实验的投入建议，没有设计或启动新模型。

未执行：可选 Identity checkpoint @ Graph、可选 GT 空间影响分布图、任何新种子/重训/新 S2 参数或 G2。E1+E2 已按冻结停止点收尾，第一份阶段结论文档的既有数值与结论边界保持原样。
