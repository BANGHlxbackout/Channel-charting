# G1 Stage-1 时间 loss 运行跟踪

日期：2026-10-25。所有新训练均为 **TODO**；此表不是完成记录。

| ID | 阶段 | 内容 | 优先级 | 状态 | 产物/判据 |
|---|---|---|---|---|---|
| T0 | Sanity | 时间索引、loss、梯度与原 pair hash 核验 | MUST | TODO | 检查日志；异常即停 |
| T1 | Pilot | full-N Graph/Identity 各 5–20 step 资源检查 | MUST | TODO | 实际耗时、GPU 占用、峰值显存 |
| T2 | Full | G1-Graph+Time ≤2,000 step | MUST，待预算确认 | TODO | best/last、日志、embedding、指标 |
| T3 | Full | G1-Identity+Time ≤2,000 step | MUST，待预算确认 | TODO | 同 T2，仅邻接不同 |
| T4 | Compare | B0 / Identity / Identity+Time / Graph / Graph+Time 统一评价 | MUST，依赖 T2–T3 | TODO | 完整 CSV、raw/affine 图、报告 |
| T5 | Stability | 追加匹配种子 | OPTIONAL，依赖主判据通过及新预算 | TODO | 多 seed 波动与交互 |

计划依据：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。旧 G1 已完成结果保留在[原报告](../results/g1_acf_graph_fcf/REPORT.md)；没有重训或覆盖。
