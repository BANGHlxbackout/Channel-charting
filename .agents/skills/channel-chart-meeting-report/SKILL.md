---
name: channel-chart-meeting-report
description: Create or quality-check Chinese Channel Charting group-meeting Word reports from completed experiments in this repository. Use for readable, source-checked reports that follow the retained Word references and actual page rendering; do not use for experiment execution or technical logs.
---

# Channel Charting 组会 Word

Use this skill for the mentor-facing report. The technical experiment report and research records remain the evidence source; a historical G1 or SC² report is only a style-test example, never this week's progress by default.

## Route the request

- For a **style or visual audit**, read the three retained Word files in `reporting/references/word/`, `reporting/reference_style_audit.json`, and `reporting/style-guide.md`. Their styles differ; the current editable template follows the 2026-10-21 reference. Check the actual Word render. Do not change experimental conclusions or metrics to solve a formatting problem.
- For a **new experiment report**, also read `AGENTS.md`, the live research state, and that experiment's technical report, protocol and saved results. Establish completion status and comparability before selecting figures or writing conclusions. Do not run new experiments just to fill a report.

## Produce the Word

Create a new experiment-specific JSON content list using the structure described in `reporting/README.md`; reuse example **structure**, never example wording or fixed section positions. Write concise academic Chinese that explains the question, essential method change, fair control, figure observation and supported conclusion. Put internal IDs and execution details in the traceability record. Let the content determine 2–4 pages; figures, tables and named sections are optional. Keep GT in the repository's permitted offline evaluation/plotting role.

Use the existing `reporting/templates/weekly-report.docx` and `reporting/scripts/build_report.py`. Keep original experiments, reference documents and earlier report versions intact. A Gitee link is valid only for a file verified in the stated remote commit; identify unpublished local evidence as such. Never infer permission to commit, upload or publish.

## Accept the result

Run `reporting/scripts/render_report.ps1` so **Microsoft Word** exports the PDF and page PNGs, then run `reporting/scripts/validate_report.py`. Inspect **every rendered page** at normal reading size for tiny figure text, blurred legends, cropped material, broken tables, awkward spacing and page breaks. Record the human visual/editorial findings beside the machine validation result; a passing script alone does not prove mentor readability. Revise and rerender until the authorized report is usable or identify the specific evidence gap.

Deliver the editable Word, PDF/page previews, source list, and machine/manual QA status. The maintained commands, style rules and examples live under `reporting/`; do not duplicate their scripts or assets inside this skill. PPT generation is not part of this Word workflow and has no validated template yet.
