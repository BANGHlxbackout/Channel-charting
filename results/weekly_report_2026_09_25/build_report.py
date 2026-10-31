from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches


SOURCE = Path(
    r"D:\电脑管家迁移文件\xwechat_files\wxid_m7ijmvso692i22_9fcf\msg\file\2026-09\2026-9-25汇报(1).docx"
)
OUTPUT = Path(__file__).with_name("2026-9-25汇报_实验更新.docx")

TEXT = [
    "一、本周调研",
    "本周复核 SC²-PCR（CVPR 2022）及作者代码。论文用匹配间的距离一致性和第三条匹配的共同支持，寻找可靠的匹配集合。",
    "- 可借鉴的是“关系需要周围证据支持”；论文没有给单个点打可信度，也没有直接修改图边权。",
    "- D_geo 与 Y0 不满足论文的刚体距离不变条件，本周仅借鉴这一思路，并非复现 SC²-PCR。",
    "二、与当前方法的联系",
    "导师 baseline 由 Siamese 得到 Y0，Stage-2 根据 D_geo 建图精修。本周不改 Stage-1、Y0 和 D_geo，只测试把少量 Stage-2 图边的权重减半；原图 k=128、λ=0.15、迭代 200 次。",
    "对照包括共同支持选边和随机选边；随机方法在每个 D_geo 距离层选择相同数量的边。真实 UE 位置只用于评价。",
    "三、本周实验结果",
    "下表与参考文档一样只列关键对比。随机行为固定 20 个选边种子（10–29）的中位数；其余行为单次结果。Mean、P95 越低越好，CT、TW 越高越好，KS 越低越好。",
    "在原 λ=0.15 下，兼容边随机降权的 20 次运行，CT、TW、KS 和全部五项误差都优于 baseline；SC² 选边虽改善均值，P95 却变差，也没有优于随机对照。",
    "用已有原图 λ=0.1344 作较强对照：原图均值 0.318396 m，兼容边随机降权后 20 次中位数 0.317687 m，但 P90 变差，说明不能只看平均误差。",
    "四、当前卡点",
    "- 共同支持低的边更常连接物理距离异常远的样本（24.6% 对 2.8%），但优先削弱它们不如随机削弱；异常不等于应该降权。",
    "- 随机降权的平均误差改善约 1 毫米，仍只在同一张 transductive chart 上验证。改善原因和换数据后的表现尚不清楚，不能据此宣称新方法。",
    "五、下周计划",
    "1. 检验“共同邻居多的边可能是冗余连接”：选边数、距离分层和降权幅度保持不变，比较高支持边降权与现有 20 次随机对照，重点看平均误差和 P95；若不优于随机，就停止用 SC² 分数定向选边。",
    "2. 若找到稳定优于随机的规则，再用独立数据或协议验证；否则保留随机降权作对照，转而查明哪些 Stage-2 连接在帮助修错。",
]

ROWS = [
    ("原图 λ0.15", "0.318748", "0.292917", "0.558674", "0.634640", "1.141431", "0.998220", "0.998223", "0.065065"),
    ("SC²选边 λ0.15", "0.318021", "0.293037", "0.558251", "0.636267", "1.128056", "0.998215", "0.998226", "0.064982"),
    ("兼容边随机 λ0.15", "0.317755", "0.291713", "0.557865", "0.633537", "1.135059", "0.998224", "0.998232", "0.064876"),
    ("全部边随机 λ0.15", "0.317847", "0.291831", "0.557988", "0.633682", "1.135528", "0.998226", "0.998233", "0.064887"),
    ("不兼容边随机 λ0.15", "0.318247", "0.292145", "0.558305", "0.633871", "1.138161", "0.998230", "0.998232", "0.064929"),
    ("原图 λ0.1344", "0.318396", "0.292453", "0.559025", "0.637179", "1.128971", "0.998223", "0.998232", "0.064922"),
    ("兼容边随机 λ0.1344", "0.317687", "0.291537", "0.559515", "0.637157", "1.123354", "0.998226", "0.998241", "0.064804"),
]


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{name}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:color"), "D9D9D9")
        borders.append(edge)
    tbl_pr.append(borders)


def add_table(document, after_paragraph, headers, rows, widths) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = False
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
    for values in (headers, *rows):
        row = table.rows[0] if values is headers else table.add_row()
        for index, value in enumerate(values):
            cell = row.cells[index]
            cell.width = Inches(widths[index])
            cell.text = value
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
    set_table_borders(table)
    after_paragraph._p.addnext(table._tbl)


def main() -> None:
    document = Document(SOURCE)
    if len(document.paragraphs) != len(TEXT):
        raise ValueError(f"Unexpected paragraph count: {len(document.paragraphs)}")
    for paragraph, replacement in zip(document.paragraphs, TEXT):
        paragraph.text = replacement
    add_table(
        document,
        document.paragraphs[8],
        ("方法", "Mean m", "P95 m", "CT", "TW", "KS"),
        [(row[0], row[1], row[4], *row[6:]) for row in ROWS[:5]],
        (1.85, 0.88, 0.88, 0.88, 0.88, 0.88),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
