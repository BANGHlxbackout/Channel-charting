# 内容审查：图传播与无传播对照

**总体判断：适合进入实际读者审阅，尚不能仅凭自动 PASS 判为导师可直接使用。** Word 共 3 页、1 张可编辑表、5 张嵌入图；PDF 与三页 PNG 已实际渲染。原始样例和本审阅副本 SHA-256 一致。

| 项目 | 本次审查 | 需重点审阅的具体位置 |
|---|---|---|
| 研究问题 | 基本清楚：改用新的节点表示与图谱映射网络后，邻域传播是否带来独立收益。 | 第 1 页开头的“自相关节点特征”“测地距离监督”对不熟悉项目的导师仍较密集，需确认是否保留这一级技术细节。 |
| 方法与对照 | 说明了无传播网络只替换邻接，模型参数、点对、损失和训练规则相同。 | 第 1 页图 1/2 内字号较小；正文“共轭外积”“单位邻接”仍需确认导师是否无需口头补充就能理解。 |
| 核心图表 | 第 2 页原始与仿射图分开，给出统一颜色映射和图谱轮廓观察；第 3 页有指标表。 | 原图自带 B0、G1-Identity、G1-Graph 英文代号，正文已给映射，但图独立传播时仍不够直观。第 3 页 CDF 三条曲线接近，读者未必能从图上看出 1–3 mm 差异；应审查是否需要此图。 |
| 结果与结论 | 表格支持“整套网络优于原初始图谱、图传播略弱于无传播对照”；明确没有证明传播的独立收益。 | 第 1 页“新网络整体有所改善”应结合第 3 页“尚未超过后处理精修主基线”阅读，防止误解成当前最佳方法。 |
| 下一步 | 提出先区分传播收益与特征收益。 | 第 3 页下一步仍偏宽泛，没有一项具体的可判别问题；是否作为本次组会提问需要读者审查。 |
| 表达与版式 | 正文使用中文方法描述，未直接贴日志；3 页中图表占主体。 | 证据区的 Gitee 技术文件标签和图片标题仍保留内部名；需确认导师接收材料时是否需要隐藏在附录/链接。 |

## 来源与发布状态

- **Gitee 已核实**：`codex/new-stage1` 的提交 `3f3ee72f551546ff0657259365b30266502174d9`。[技术报告](https://gitee.com/cactus-jacque/diffusion-charting/blob/3f3ee72f551546ff0657259365b30266502174d9/results/g1_acf_graph_fcf/REPORT.md)、[比较 CSV](https://gitee.com/cactus-jacque/diffusion-charting/blob/3f3ee72f551546ff0657259365b30266502174d9/results/g1_acf_graph_fcf/comparison/comparison.csv)、[原始图](https://gitee.com/cactus-jacque/diffusion-charting/blob/3f3ee72f551546ff0657259365b30266502174d9/results/g1_acf_graph_fcf/comparison/four_column_raw_channel_chart.png)、[仿射图](https://gitee.com/cactus-jacque/diffusion-charting/blob/3f3ee72f551546ff0657259365b30266502174d9/results/g1_acf_graph_fcf/comparison/four_column_affine_channel_chart.png)、[对齐坐标](https://gitee.com/cactus-jacque/diffusion-charting/blob/3f3ee72f551546ff0657259365b30266502174d9/results/g1_acf_graph_fcf/comparison/aligned_charts.npz)及方法/评价代码在该提交中；Word 中的蓝色标签是对应的可点击 Gitee 链接。
- **本地数值来源**：`results/g1_acf_graph_fcf/comparison/comparison.csv`；三行分别核对 `results/adp_old_siamese/stride1_n20997_seed10/result.json`、`results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/result.json`、`results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/result.json`。三份逐行 JSON 目前未在上述 Gitee 提交中，Word 明确标“待上传”。
- **图片来源**：方法图 1/2 从 `reporting/references/word/刘梓健-2026-10-21.docx` 提取；图谱图 3/4 来自上述 Gitee 提交；CDF 图 5 基于保存的 `results/g1_acf_graph_fcf/comparison/aligned_charts.npz`。
- **评价口径**：同一 `N=20997` 节点顺序；CT/TW/KS 固定 4199 节点，仿射误差在全 N 节点分别拟合与评价；单种子、传导式，不代表留出能力。
