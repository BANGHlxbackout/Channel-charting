# G1-ACF Graph-FCF：实现与实验结果

日期：2026-10-16。**状态：Sanity、Full-N 五步资源 Pilot、G1-Graph 与 G1-Identity 完整训练及统一评价均已完成。**Sanity/Pilot 指标仍只用于功能和资源检查，不混入正式排名。

## 问题与固定对照

目标是在当前实测 DICHASUS 数据及原有 Stage-1 图/距离上，比较完整 G1-Graph 与仅把传播邻接替换为单位矩阵的 G1-Identity。两者共用 26,624 维 B0 自相关特征、14,162,434 个参数、geodesic stress、AdamW、训练/验证点对生成器、checkpoint 规则和评价协议。原 B0 的特征、模型和训练实现未改；后续为直接复用原绘图代码，仅给 `plot_chart` 增加外部 axes 和原始/仿射选项，默认两列图与修改前逐像素相同。GT 不参与训练或 checkpoint 选择。

代码：`methods/g1_acf_graph_fcf/graph_utils.py`、`methods/g1_acf_graph_fcf/run.py`、`methods/g1_acf_graph_fcf/compare.py`、`tests/test_g1_acf_graph_fcf.py`。现有工作树的 `common_metrics.py` 有 GBK 注释导致 Python 无法导入；仅增加编码声明，指标函数未改。

## 实施检查

- `D:\anaconda3\envs\diffusion_charting\python.exe -m unittest tests.test_g1_acf_graph_fcf -v`：5/5 通过，覆盖特征相位性质、图及单位邻接、训练/验证点对隔离、直接与分块投影的输出/梯度一致，以及四列比较产物。
- full-N `N=20,997`、`k=20`：重建 419,940 条有向边；固定 32 源到全部节点的最短路与已保存 `geodesic.npy` 最大差 0、错配 0。GCN 的对称 affinity 只用原有无向边 support；原始 directed cost 和最短路未修改。
- 原图互为邻居的 2,759 对边有浮点级两向差，最大 `1.144409e-5`。传播 affinity 取端点排序后 `fused[min(i,j),max(i,j)]` 作为该无向边的成本，赋同一对称权重；这不是原图最短路的边权，不能与其混称。
- 特征缓存抽样行与 B0 `AdpFeatureLayer` 输出一致。分块投影小图的 projector、GCN 与输出头梯度和直接模式一致；真实运行未出现 NaN/Inf、缺失梯度或坍缩。

## Sanity：真实 stride-10，非完整结果

用原 `run_adp_fused.py` 为完全相同的 `indices` 建立 `results/adp_old_fused/stride10_n2100_seed10/`；该子样本图为连通需 `k=35`，与 full-N 的 `k=20` 不同。MDS 仅用于生成距离 run 附带输出，设置 `mds-reduction=10, mds-iter=10, mds-n-init=1`；G1 只读取 fused/geodesic/indices。

两组均为 `N=2,100`、seed 10、100 step、每 step 65,536 训练点对、200,000 验证点对、每 10 step 验证。loss 有限并下降，模型与图谱保存。

| 模式 | 最佳 step | CT | TW | KS | Mean (m) | Median (m) | P95 (m) | 训练耗时 | PyTorch 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1-Graph Sanity | 100 | 0.5740 | 0.7395 | 0.6071 | 3.5470 | 3.4054 | 6.6317 | 8.21 s | 0.495 GiB |
| G1-Identity Sanity | 100 | 0.6160 | 0.7318 | 0.5771 | 3.1737 | 3.0073 | 6.1681 | 7.21 s | 0.494 GiB |

Sanity 数据与图稀疏度不同于 full-N，训练也只有 100 step；不据此判断最终优劣。源产物分别在 `stride10_n2100_seed10_graph_sanity/` 和 `stride10_n2100_seed10_identity_sanity/`。

## Pilot：full-N 五步资源检查，非完整结果

G1-Graph 使用 `N=20,997`、`k=20`、CPU 特征 memmap + GPU 分块投影，5 step 后按 Pilot 协议保存 checkpoint。训练 18.80 s，整体启动到评价输出 40.37 s，PyTorch 峰值分配 707,292,672 bytes（0.659 GiB）。结果在 `stride1_n20997_seed10_graph_pilot/`；5 step 指标只证明执行可行，不用于方法比较。

## 完整训练资源预算与停止条件（已确认并执行）

- **拟运行范围**：full-N seed 10 的 Graph 与 Identity 各 1 次；每次最多 2,000 step、每 step 65,536 点对、每 20 step 评价 200,000 固定验证点对；连续 12 次评价无改善按冻结规则早停。不调参、不扩展方法。
- **时间**：Pilot 按 18.80 s / 5 step 外推约 2.1 小时/组；两组训练约 4.2 小时。考虑长时运行、评价与环境波动，为两组预留 **最多约 6 小时 GPU 占用**；早停则更短。这是估算而非保证。
- **显存/磁盘**：RTX 3050 Laptop GPU 4 GB；Pilot 的 PyTorch 峰值分配 0.659 GiB，不等于进程总显存或长跑上限。full-N 特征缓存约 2.24 GB 已生成；当前 D: 可用约 18.26 GB。两组最佳/末尾权重合计约 227 MB，另有小型图谱与日志。主存 17.0 GB；full-N 矩阵继续 CPU mmap，不传 GPU。
- **停止**：图或索引复验失败、梯度缺失/非有限、NaN/Inf、输出坍缩、显存不足时停止并保留失败事实；不改图、输入或训练规模来掩盖失败。

用户确认后，两组均按上述冻结设置运行至 2,000 step；实际训练总计 3.199 小时，未触及约 6 GPU 小时预算上限。没有修改算法、输入、样本规模或 checkpoint 规则。

## 正式训练进度（2026-10-16）

用户已确认两组各最多 2,000 step、合计约 6 GPU 小时预算。实际按 Graph 后 Identity 顺序执行，控制台日志保存为 `graph_full_console.log` / `identity_full_console.log`。Graph 前 100 step 用 **257.25 s**（平均 2.57 s/step），Identity 前 100 step 用 **267.63 s**（平均 2.68 s/step）；两者训练日志均有 2,000 条逐 step loss/LR/耗时记录。Graph 的 step 100 train/validation loss 为 145.9134/144.2779，均有限。Graph 运行中 `nvidia-smi` 显示 GPU 总占用约 **1,921 MiB**、温度 66°C；此系统数值包含其他 GPU 占用。两组均无 OOM、NaN、梯度异常或中断。

## 正式 full-N 结果：统一重算

三份 embedding 的 `indices` 逐元素相同，真实位置数值一致；CT/TW/KS 均用 seed 10 的同一 20% 节点子集（4,199 个节点）重算，Mean/Median/P95 均在相同 20,997 节点上分别拟合仿射变换后计算。GT 只用于训练结束后的评价、仿射对齐及绘图。B0 就是 `results/adp_old_siamese/stride1_n20997_seed10/embedding.npz`，不是 Stage-2 Z。原 B0 `result.json` 和 `chart.png` 已从 master 提交 `61178b5` 原样恢复，原始六项指标与本次重算完全一致。原记录含总流程 `elapsed_s=1633.554753`，但未单独记录训练耗时或峰值显存，所以表中的训练专用资源仍标为未单独留存。

| 方法 | CT ↑ | TW ↑ | KS ↓ | Mean (m) ↓ | Median (m) ↓ | P90 (m) ↓ | P95 (m) ↓ | Max (m) ↓ | 训练时间 | PyTorch 峰值分配 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Siamese FCF | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | 未单独留存（总流程 1,633.55 s） | 未留存 |
| G1-Identity | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 | 5,292.06 s | 0.650 GiB |
| G1-Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 | 6,224.64 s | 0.659 GiB |

Graph 与 Identity 的模型参数均为 14,162,434，训练/验证点对生成规则、seed、最大步数及早停规则一致。两份 `result.json` 的 `validation_pair_hash` 和 `first_train_pair_hash` 完全相同；两组实际均跑满 2,000 step，验证 loss 选出的 best step 均为 1,920。Graph/Identity 最佳 validation loss 分别为 1.872658/1.862524。运行产物分别位于 `stride1_n20997_seed10_graph_full/` 与 `stride1_n20997_seed10_identity_full/`，含 `result.json`、`run_config.json`、`model_best.pt`、`model_last.pt`、`training_history.json`、`embedding.npz`、图/特征 metadata、`chart.png`、`run_command.txt`。`completion_checks.json` 核验两组产物、2,000 条日志、有限损失、best 与验证最小值一致、相同索引/模型形状/点对哈希及预算通过。当前分支 `codex/new-stage1`；代码、记录和结果尚未提交或上传。

**问题 1：Graph 是否优于 B0？**在原先比较的 CT/TW/KS/Mean/Median/P95 六项上均有小幅改善。Graph 的 CT/TW 分别高 0.001235/0.000862，KS 低 0.004687，Mean/Median/P95 分别低 0.009577/0.007650/0.009456 m。这只说明整套 G1-Graph 在同数据传导式图谱评价中优于旧 B0，不能单独归因于邻域传播；补全 P90 和 Max 的结果如下。

补全误差指标后，Graph 的 P90 比 B0 高 0.000256 m，基本相同；最大误差则从 B0 的 7.893630 m 降至 1.463179 m。Identity 的最大误差也降到 1.429263 m，且 P90 比 Graph 略低，因此极端误差改善同样不能归因于传播。这里的 Max 是全 N 样本仿射误差的最大值，不能解释成大部分节点都改善了相同幅度。

**问题 2 与 4：Graph 是否优于 Identity、能否支持传播的正向作用？**在主要误差端点上不能。Graph 的 KS 比 Identity 低 0.000062，但 CT/TW 分别低 0.000019/0.000020，Mean/Median/P95 分别高 0.001428/0.002862/0.003140 m。离线逐点仿射误差中 Graph 优于 Identity 的节点占 47.17%，平均误差反而高 1.43 mm。因此这次固定图、模型、seed 与预算的结果**不支持邻居传播提供可辨认的正向收益**；不能扩大为所有图方法无效。

**问题 3：变化体现在哪里？**仿射图上两种 G1 都比 B0 更清晰地保留左上弧区与右下分支的扫描轨迹，但 Graph 和 Identity 的整体轮廓、颜色梯度和局部纹理非常接近，未见 Graph 独有的明显结构修复。两者差异主要是小幅数值变化；原始图的坐标方向与尺度任意，须结合仿射图阅读。四列图在 `comparison/four_column_raw_channel_chart.png` 和 `comparison/four_column_affine_channel_chart.png`，共享真实位置颜色映射；对齐坐标在 `comparison/aligned_charts.npz`，指标和协议在 `comparison/comparison.csv`、`comparison/comparison.md`、`comparison/eval_protocol.json`。

本结果仅为 **single-seed、same-node transductive** 比较，不能宣称未见节点泛化。Graph mean 0.399771 m 也尚未超过已有 Stage-2 主基线的 0.3187 m；本轮没有做可选的 G1+Stage-2。完成后依冻结范围收尾，不自动改图或调参。

## 2026-10-17：目录、绘图与 B0 历史产物纠正

- 本轮 G1 三个代码入口归入 `methods/g1_acf_graph_fcf/`；根目录旧入口已移走。使用 `python -m methods.g1_acf_graph_fcf.run` 和 `python -m methods.g1_acf_graph_fcf.compare`。历史 `run_command.txt` 保留原执行路径，不改写历史命令；测试仍放在 `tests/`。
- 旧四列图另写了散点和布局设置，未直接复用 baseline 绘图函数；已改为由 `run_adp_siamese.plot_chart` 直接画四列。沿用原点大小 5、grid alpha 0.25、equal axes、每两列 9×4 英寸及 180 dpi；只把原两列扩为四列，并为 raw 使用 `align=False`。默认原两列调用经修改前后 PNG 像素对比完全一致。原始/仿射四列图已重画，指标数值不变。
- 起初只检查工作目录就将 B0 原记录判为缺失，漏查 Git 历史。保存原记录的 `61178b5` 在 master/origin/master 上，当前 `codex/new-stage1` 不包含该提交。已将原 `result.json` 和 `chart.png` 恢复到原目录，并核验与 Git blob 完全一致；未覆盖 B0 模型或 embedding，未重训。
- 用户确认方向可以继续探索。这一次固定设置的 Graph/Identity 结论保留，不能据此把整个方向判为结束；后续具体实验尚未制定或运行。
- 按用户要求补全误差指标导出：直接复用 `common_metrics.localization_error_summary` 的全部 Mean/Median/P90/P95/Max。`comparison.csv`/`comparison.md` 使用 `ct,tw,ks,mean_m,median_m,p90_m,p95_m,max_m,source_json` 字段，并保留 method、训练耗时和显存；每行 `source_json` 指向对应原始结果。指标从保存 embedding 按既定共同评价子集重算，不涉及重训。

## 2026-10-17：无重训后续 E1/E2（原结果保留）

按用户授权完成固定 S2 的六组比较和冻结 checkpoint 邻接敏感性诊断，完整新产物及报告在 `results/g1_acf_graph_fcf_followup/REPORT.md`。两种 G1+S2 的 KS/Mean/P95 均未超过 B0+S2；Graph 模型推理换 I 后 stress 上升 71.36%，说明模型依赖邻接，但这种依赖没有形成相对重新训练 Identity 的净收益。代码审查另发现原固定验证点对存在排序截断造成的索引偏向，本轮原样复用并限制 stress 解释；旧 checkpoint、embedding、训练数值与比较文件未改。
