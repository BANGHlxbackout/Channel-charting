# 每周组会汇报

本目录保存参考 Word/PPT、母版、生成脚本和样例；**项目 Skill 定义在 [`.agents/skills/channel-chart-meeting-report/SKILL.md`](../.agents/skills/channel-chart-meeting-report/SKILL.md)**，不放在 `reporting/` 下。

以后可直接提出：“将 `<实验名或结果目录>` 整理成组会 Word 汇报。”执行者先读仓库 `AGENTS.md`、当前研究状态、对应任务与技术报告，确认结果完成及对照可比，再从本目录选择参考风格与证据图表。`style-guide.md` 规定读者表达、字体及来源边界。Word 按内容控制在 2–4 页；G1 三页仅是本次样例。

- `references/`：用户历史 Word/PPT 原件，只作只读参照。
- `templates/weekly-report.docx`：去掉具体实验文字的可编辑 Word 母版。
- `examples/g1-pilot/`：第一版 G1 历史样式样例，保留不覆盖。
- `examples/style_test_v2/`：本轮按三份参考 Word 完成样式与视觉回归的 G1、SC² 新版历史样例；对照与逐页验收见 `VISUAL_QA.md`。下次只借用 JSON 结构，按新实验重写内容，不复制 G1/SC² 文案。
- `scripts/build_report.py`：从清单、真实 CSV/JSON 与图片生成 Word；核对 Gitee 分支、提交和已发布文件。
- `scripts/render_report.ps1`：用本机 Word 导出 PDF 并逐页生成 PNG。
- `scripts/validate_report.py`：检查来源、可比性、图片、可选的可编辑表格、样式、页数与 Gitee 链接。

下面是新实验使用的命令模板，先为该实验创建新的内容清单和输出目录。脚本不从 G1/SC² 样例自动继承实验文字；输出文件已存在时拒绝覆盖。

```powershell
$py = 'C:\Users\Hype beast\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$spec = 'reporting/本次实验/report.json'
$docx = 'reporting/本次实验/刘梓健-YYYY-MM-DD.docx'
$preview = 'reporting/本次实验/preview'
& $py reporting/scripts/build_report.py $spec $docx
& .\reporting\scripts\render_report.ps1 -Docx $docx -OutputDir $preview -PythonPath $py
& $py reporting/scripts/validate_report.py $spec $docx $preview
```

`reference_style_audit.json` 保存三份 Word 原件的真实样式；三者不完全同版，当前母版选用 2026-10-21 风格。PPT 原稿已保留，PPT 母版与自动生成尚未验证。新版 Word 的 `validation.json` 和 `VISUAL_QA.md` 分别记录机器检查与实际页面目视检查。当前仓库的 `results/` 默认被 Git 忽略；正式上传时需逐文件审核并显式加入必要结果，上传后再以远端提交复核 Word 中的 Gitee 证据链接。
