# G1-ACF Graph-FCF

当前实现与 Identity 对照共用 26,624 维 B0 自相关特征、模型、Loss 和训练/验证点对；`--mode graph` 与 `--mode identity` 只改变传播邻接。

- `run.py`：模型与训练入口。
- `graph_utils.py`：原始图/测地距离复验、传播 affinity、特征缓存和点对采样。
- `compare.py`：共同评价及四列原始/仿射图，直接调用 `run_adp_siamese.plot_chart`。
- 测试：`tests/test_g1_acf_graph_fcf.py`。
- 任务：`docs/tasks/G1_GraphFCF.md`；结果：`results/g1_acf_graph_fcf/REPORT.md`。

从仓库根目录调用：

```powershell
python -m methods.g1_acf_graph_fcf.run --help
python -m methods.g1_acf_graph_fcf.compare `
  --baseline-embedding results/adp_old_siamese/stride1_n20997_seed10/embedding.npz `
  --identity-embedding results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/embedding.npz `
  --graph-embedding results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/embedding.npz `
  --seed 10 --out-dir results/g1_acf_graph_fcf/comparison
```

历史 `run_command.txt` 保留当时的根目录入口，不改写已执行命令；今后使用上面的模块路径。原有训练结果、缓存、best/last 权重和日志保留原位置。

B0 明确指 `run_adp_siamese.py` 生成的 `results/adp_old_siamese/stride1_n20997_seed10/embedding.npz`，不是 Stage-2 的 Z。

已完成单种子 full-N Graph/Identity 对照；本次结果未显示传播相对 Identity 的净收益。用户确认此方向仍可继续探索，首轮训练结果保留，已授权的后续结果见下。

2026-10-17 无重训后续已完成：`run_stage2_comparison.py` 调用原 S2 函数作六组统一比较；`diagnose_graph_propagation.py` 作冻结权重邻接敏感性与层统计。结果见 `results/g1_acf_graph_fcf_followup/REPORT.md`，执行规范及检查点见 `docs/tasks/G1_S2_2026-10-17.md`。当前 G1+S2 未超过 B0+S2；Graph 对邻接敏感，但未带来额外净收益。后续训练未启动。
