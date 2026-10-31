from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


SOURCE = Path(
    r"D:\电脑管家迁移文件\xwechat_files\wxid_m7ijmvso692i22_9fcf\msg\file\2026-09\2026-9-25汇报(1).docx"
)
OUTPUT = Path(__file__).with_name("2026-9-25汇报_实验进展_待插表.docx")

REPLACEMENTS = [
    "三、本周实验进展",
    "本周只改 Stage-2 精修图中部分边的权重，不改 Stage-1、D_geo 或 Y0，也不重训 Siamese。比较了 SC² 选边、一阶选边和同距离层随机选边；原图 k=128、λ=0.15，迭代 200 次。",
    "SC² 选边使平均误差从 0.318748 m 降至 0.318021 m，但 P95 从 0.634640 m 升至 0.636267 m。兼容边随机降权的 20 次运行中，平均误差中位数为 0.317755 m，八项指标每次都优于原 baseline，且均值优于 SC² 定向选边。",
    "实验结果表格截图粘贴于下方。随机方法展示 20 次运行的中位数；CT、TW、KS 使用相同的评价子集。",
    "四、当前卡点",
    "- 共同支持低的边更常对应物理跨度异常的样本对，但优先减弱这些边不如随机减弱；目前没有可靠的定向选边规则。",
    "- 随机降权的平均误差改善约 1 毫米，仍只在同一张 transductive chart 上验证。改善原因和换数据后的效果尚不清楚，不能据此宣称形成新方法。",
    "五、下周计划",
    "1. 固定选边数量、距离分层和降权幅度，检验“高共同支持边更冗余”的假设，对照现有 20 次随机结果，重点看平均误差和 P95；若仍不优于随机，就停止用 SC² 分数定向选边。",
    "2. 若找到稳定优于随机的规则，再在独立数据或协议上验证；否则先查明哪些连接被减弱时修复更多、损害更少，再决定下一步方法。",
]


def add_placeholder(document, after_paragraph) -> None:
    table = document.add_table(rows=1, cols=1)
    table.autofit = False
    table.columns[0].width = Inches(6.2)
    row = table.rows[0]
    row.height = Inches(2.15)
    row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    cell = row.cells[0]
    cell.width = Inches(6.2)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    cell.text = "在此粘贴实验结果表格截图"
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:color"), "BFBFBF")
        borders.append(edge)
    tc_pr.append(borders)
    after_paragraph._p.addnext(table._tbl)


def main() -> None:
    document = Document(SOURCE)
    if len(document.paragraphs) != 17:
        raise ValueError(f"Unexpected paragraph count: {len(document.paragraphs)}")
    original_first_two = [p.text for p in document.paragraphs[:7]]
    for paragraph, replacement in zip(document.paragraphs[7:], REPLACEMENTS):
        paragraph.text = replacement
    if [p.text for p in document.paragraphs[:7]] != original_first_two:
        raise AssertionError("First two sections changed")
    document.paragraphs[10].paragraph_format.keep_with_next = True
    document.paragraphs[11].paragraph_format.space_before = Pt(8)
    add_placeholder(document, document.paragraphs[10])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
