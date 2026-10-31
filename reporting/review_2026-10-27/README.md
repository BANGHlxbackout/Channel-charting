# 组会 Word 内容审阅包（2026-10-27）

本目录是已生成样例的**逐字节副本**，供用户与“工作流自动化升级”对话审阅。此轮没有修改两份 Word 的正文或图片，没有提交 Git，也没有向 Gitee 上传。G1 为 3 页，SC² 为 2 页；页数依内容而定，均在 2–4 页范围内。

`PACKAGE_VALIDATION.json` 保存副本与原件的 SHA-256 一致性、页数、图片/表格数量及自动验收摘要。

| 样例 | 可编辑 Word | 实际 Word 导出 PDF | 逐页预览 | 内容清单 | 自动验收 | 内容审查与来源 |
|---|---|---|---|---|---|---|
| 图传播与无传播对照 | `g1/report.docx` | `g1/preview.pdf` | `g1/page-1.png`—`page-3.png` | `g1/report.json` | `g1/validation.json`，15/15 PASS | `g1/CONTENT_REVIEW.md`、`comparison.csv`、`eval_protocol.json` |
| 连接权重调整 | `sc2/report.docx` | `sc2/preview.pdf` | `sc2/page-1.png`—`page-2.png` | `sc2/report.json` | `sc2/validation.json`，12/12 PASS | `sc2/CONTENT_REVIEW.md`、`result.json`、`complete_metrics_followup.json` |

自动 PASS 只覆盖文件、样式、来源和页数等已实现规则。两份内容审查都列出仍需导师视角判断的问题；**不要将本目录标为最终可直接汇报版本**。G1 的主要数值和表格已与源 JSON 核对；SC² 的试运行未建立统一比较 CSV，12/12 也未包含正文数值对源 JSON 的逐项自动核对，仍需按技术报告及结果 JSON 复核。

审阅问题：不看代码和日志，仅看 Word，是否能理解为什么做、如何公平比较、图表实际说明什么、结论及下一步？请指出具体页和段落或图片。收到反馈后再决定修改内容规则、局部重绘或 PPT 开发。
