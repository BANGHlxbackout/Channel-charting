# 内容审查：连接权重调整

**总体判断：这是跨实验生成能力的两页试运行，内容尚不足以直接给导师。** Word 共 2 页、2 张嵌入图、无可编辑比较表；PDF 与两页 PNG 已实际渲染。原始试运行和本审阅副本 SHA-256 一致。

| 项目 | 本次审查 | 需重点审阅的具体位置 |
|---|---|---|
| 研究问题 | 第 1 页说明“精修图中的连接可能有误”，要比较定向选边与随机选边。 | “共同支持”只出现为方法名称，未说明为什么共同邻居可能是可靠性线索；导师可能不知实验动机。 |
| 方法与对照 | 写出减半权重，且随机对照匹配选边数量与距离范围。 | 未交代本次调整的是既有图谱的后处理连接，也未说明初始图谱和其他条件保持不变。 |
| 核心图表 | 第 1 页显示跨度偏大连接比例，第 2 页显示空间上的修复与损害。 | 两张源图的图例/坐标轴仍用 SC2、D_geo、英文，图注尚未完整解释分层、蓝红颜色和两种选边对象；不看技术报告难以独立解读。 |
| 结果与结论 | 平均误差下降但 P95 变差，随机对照均值更好，结论方向与技术报告一致。 | 只有一段数值，没有正式对比表；未说明随机行是 20 个掩码的逐指标中位数，容易误解为一张图谱或独立重复数据。自动 12/12 未逐项核对正文数值。 |
| 下一步 | 缺失。 | 第 2 页未写后续要验证哪一项不确定性，尚不构成完整组会汇报。 |
| 表达与版式 | 文字简短，2 页没有溢出。 | 两页留白较大，第二页研究解释不足；这不需要靠添加日志填满，应精选真正能解释结论的表与短文。 |

## 来源与发布状态

- **Gitee 已核实**：`codex/sc2-stage2-edge-consensus` 的提交 `f927cd53669f1eca1230e982233d1c04bb4b3f24`。[技术报告](https://gitee.com/cactus-jacque/diffusion-charting/blob/f927cd53669f1eca1230e982233d1c04bb4b3f24/results/sc2_stage2_edge_consensus/REPORT.md)、[选边诊断图](https://gitee.com/cactus-jacque/diffusion-charting/blob/f927cd53669f1eca1230e982233d1c04bb4b3f24/results/sc2_stage2_edge_consensus/posthoc_selected_edge_diagnostic.png)、[逐点损害图](https://gitee.com/cactus-jacque/diffusion-charting/blob/f927cd53669f1eca1230e982233d1c04bb4b3f24/results/sc2_stage2_edge_consensus/posthoc_repair_harm_map.png)、[主结果 JSON](https://gitee.com/cactus-jacque/diffusion-charting/blob/f927cd53669f1eca1230e982233d1c04bb4b3f24/results/sc2_stage2_edge_consensus/result.json) 和 [20 次随机补全指标](https://gitee.com/cactus-jacque/diffusion-charting/blob/f927cd53669f1eca1230e982233d1c04bb4b3f24/results/sc2_stage2_edge_consensus/complete_metrics_followup.json) 均在该提交中；Word 中只链接了技术报告和两张图。
- **本地数值来源**：`results/sc2_stage2_edge_consensus/REPORT.md`、`result.json` 和 `complete_metrics_followup.json`。Word 中 0.318748/0.318021 m、P95 0.634640/0.636267 m、随机均值中位数 0.317755 m 与报告陈述相符，但尚未建立该 Word 正文到结果 JSON 的逐值自动校验。
- **评价边界**：同一 20,997 节点的传导式实验；随机 20 次仅变选边掩码，不是 20 份独立数据。真实位置只在离线评价和图中标注，不用于选边。
