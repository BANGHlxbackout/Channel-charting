"""Build an editable meeting DOCX from a reusable style template and report spec."""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "reporting/templates/weekly-report.docx"
METRIC_LABELS = {
    "ct": "CT↑", "tw": "TW↑", "ks": "KS↓", "mean_m": "Mean m↓",
    "median_m": "Median m↓", "p90_m": "P90 m↓", "p95_m": "P95 m↓",
    "max_m": "Max m↓",
}


def project_path(value):
    path = (ROOT / value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path outside repository: {value}")
    return path


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_gitee(repository):
    commit = repository["commit"]
    branch = repository["branch"]
    remote = repository["remote"]
    remote_url = git("remote", "get-url", remote).removesuffix(".git").rstrip("/")
    if remote_url != repository["url"].rstrip("/"):
        raise ValueError("Configured Gitee URL differs from the Git remote")
    lines = git("ls-remote", "--heads", remote, branch).splitlines()
    if len(lines) != 1 or lines[0].split()[0] != commit:
        raise ValueError("Gitee branch does not point to the stated commit")
    for item in repository["published"]:
        path = item["path"]
        if project_path(path) == ROOT or not git("ls-tree", "-r", "--name-only", commit, "--", path):
            raise ValueError(f"Not present in the verified Gitee commit: {path}")
    for item in repository.get("pending", []):
        path = item["path"]
        project_path(path)
        if git("ls-tree", "-r", "--name-only", commit, "--", path):
            raise ValueError(f"Already published; update the evidence status: {path}")
    return commit


def gitee_url(repository, path):
    return repository["url"].rstrip("/") + "/blob/" + repository["commit"] + "/" + quote(path, safe="/")


def result_metric(source, key):
    return float(source["metrics"][key] if key in ("ct", "tw", "ks")
                 else source["localization_errors_after_affine"][key])


def checked_comparison(spec):
    comparison = spec["comparison"]
    csv_path = project_path(comparison["csv"])
    protocol_path = project_path(comparison["protocol"])
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["method"]: row for row in csv.DictReader(stream)}
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not ("transductive" in protocol["chart_quality"]
            and "separate affine" in protocol["alignment"]):
        raise ValueError("Comparison protocol does not match this report")
    for method in comparison["methods"]:
        row = rows[method["id"]]
        source = Path(row["source_json"])
        if not source.is_absolute():
            source = project_path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        result = json.loads(source.read_text(encoding="utf-8"))
        for key in METRIC_LABELS:
            if abs(float(row[key]) - result_metric(result, key)) > 1e-9:
                raise ValueError(f"CSV/JSON mismatch: {method['id']} {key}")
    return rows, protocol


def set_cell_fill(cell, color):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(shd)


def add_table(document, block, rows):
    keys = block["metrics"]
    if any(key not in METRIC_LABELS for key in keys):
        raise ValueError("Unknown metric column")
    headers = ["方法"] + [METRIC_LABELS[key] for key in keys]
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = False
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement("w:" + side)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:color"), "D9D9D9")
        borders.append(edge)
    table._tbl.tblPr.append(borders)
    usable_cm = (document.sections[0].page_width - document.sections[0].left_margin
                 - document.sections[0].right_margin) / Cm(1)
    widths = [3.2] + [(usable_cm - 3.2) / len(keys)] * len(keys)
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.width = Cm(widths[i])
        cell.text = header
        set_cell_fill(cell, "21415A")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(10.5)
    for index, method in enumerate(block["methods"], 1):
        row = rows[method["id"]]
        values = [method["name"]] + [f"{float(row[key]):.4f}" for key in keys]
        cells = table.add_row().cells
        for col, value in enumerate(values):
            cell = cells[col]
            cell.width = Cm(widths[col])
            cell.text = value
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.paragraphs[0].alignment = (WD_ALIGN_PARAGRAPH.LEFT if col == 0
                                            else WD_ALIGN_PARAGRAPH.CENTER)
            if index % 2 == 0:
                set_cell_fill(cell, "EEF3F7")
            for run in cell.paragraphs[0].runs:
                run.bold = method.get("emphasis", False)
                run.font.size = Pt(10.5)
    return table


def add_hyperlink(paragraph, label, url):
    rid = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "21415A")
    props.append(color)
    run.append(props)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def add_sources(document, repository):
    paragraph = document.add_paragraph(style="Source")
    paragraph.add_run("Gitee 证据位置  ").bold = True
    add_hyperlink(paragraph, repository["url"], repository["url"])
    paragraph.add_run("  ·  " + repository["branch"] + " @ " + repository["commit"][:8])
    paragraph = document.add_paragraph(style="Source")
    paragraph.add_run("已核验：").bold = True
    for i, item in enumerate(repository["published"]):
        if i:
            paragraph.add_run("  ·  ")
        add_hyperlink(paragraph, item["label"], gitee_url(repository, item["path"]))
    if repository.get("pending"):
        paragraph = document.add_paragraph(style="Source")
        paragraph.add_run("待上传至 Gitee：").bold = True
        paragraph.add_run("、".join(item["label"] for item in repository["pending"]))


def add_block(document, block, rows):
    kind = block["type"]
    if kind == "paragraph":
        paragraph = document.add_paragraph(style="Normal")
        if block.get("lead"):
            paragraph.add_run(block["lead"] + "  ").bold = True
        paragraph.add_run(block["text"])
    elif kind == "figure":
        path = project_path(block["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        usable_cm = (document.sections[0].page_width - document.sections[0].left_margin
                     - document.sections[0].right_margin) / Cm(1)
        if block["width_cm"] > usable_cm:
            raise ValueError(f"Figure wider than the reference page content: {path}")
        paragraph = document.add_paragraph(style="Normal")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(path), width=Cm(block["width_cm"]))
        caption = document.add_paragraph(style="Caption")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.add_run(block["caption"])
    elif kind == "table":
        if not rows:
            raise ValueError("A table block requires a checked comparison CSV")
        add_table(document, block, rows)
    else:
        raise ValueError(f"Unknown report block: {kind}")


def build(spec, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not 2 <= len(spec["pages"]) <= 4:
        raise ValueError("Meeting reports must contain 2–4 content pages")
    repository = spec["repository"]
    verify_gitee(repository)
    rows, _ = checked_comparison(spec) if "comparison" in spec else ({}, None)
    document = Document(TEMPLATE)
    body = document.element.body
    for element in list(body):
        if not element.tag.endswith("}sectPr"):
            body.remove(element)
    report = spec["report"]
    document.add_paragraph(report["title"], style="Title")
    document.add_paragraph(report["date"] + "  ·  " + report["scope"], style="Meta")
    summary = document.add_paragraph(style="Normal")
    summary.add_run("结论  " + report["summary"]).bold = True
    for index, page in enumerate(spec["pages"]):
        heading = document.add_paragraph(page["heading"], style="Heading 1")
        if index:
            heading.paragraph_format.page_break_before = True
        for block in page["blocks"]:
            add_block(document, block, rows)
    add_sources(document, repository)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    print(build(spec, args.output.resolve()))


if __name__ == "__main__":
    main()
