"""Validate a generated DOCX, Gitee evidence and rendered page preview."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pypdfium2 as pdfium
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from PIL import Image
from docx.oxml.ns import qn

from build_report import (ROOT, TEMPLATE, checked_comparison,
                          gitee_url, project_path, verify_gitee)

STYLE_REFERENCE = ROOT / "reporting/references/word/刘梓健-2026-10-21.docx"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def embedded_images(document):
    images = []
    for shape in document.inline_shapes:
        rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        images.append(document.part.related_parts[rid].blob)
    return images


def blocks(spec, kind):
    return [block for page in spec["pages"] for block in page["blocks"]
            if block["type"] == kind]


def cell_fill(cell):
    props = cell._tc.tcPr
    shading = props.find(qn("w:shd")) if props is not None else None
    return shading.get(qn("w:fill")) if shading is not None else None


def validate(spec, docx_path, preview_dir):
    checks = {}
    with ZipFile(docx_path) as package:
        checks["docx_package"] = package.testzip() is None
    document = Document(docx_path)
    template = Document(TEMPLATE)
    reference = Document(STYLE_REFERENCE)
    fields = ("page_width", "page_height", "top_margin", "bottom_margin",
              "left_margin", "right_margin", "header_distance", "footer_distance")
    checks["page_geometry_matches_template"] = all(
        getattr(document.sections[0], key) == getattr(template.sections[0], key)
        for key in fields
    )
    checks["font_and_paragraph_styles_match_template"] = all(
        document.styles[name].element.xml == template.styles[name].element.xml
        for name in ("Normal", "Title", "Heading 1", "Caption", "Meta", "Source")
    )
    checks["template_styles_match_reference"] = all(
        template.styles[name].element.xml == reference.styles[name].element.xml
        for name in ("Normal", "Title", "Heading 1", "Caption", "Meta", "Source")
    ) and all(
        getattr(template.sections[0], key) == getattr(reference.sections[0], key)
        for key in fields
    )
    checks["paragraph_roles_from_reference"] = all(
        paragraph.style.name in ("Normal", "Title", "Heading 1", "Caption", "Meta", "Source")
        for paragraph in document.paragraphs
    )
    body = "\n".join(p.text for p in document.paragraphs)
    checks["title_and_date"] = (spec["report"]["title"] in body
                                and spec["report"]["date"] in body)
    checks["no_template_placeholders"] = "【" not in body and "】" not in body

    expected_figures = [project_path(block["path"]).read_bytes()
                        for block in blocks(spec, "figure")]
    images = embedded_images(document)
    checks["all_result_images_embedded"] = (
        len(images) == len(expected_figures)
        and all(digest(a) == digest(b) for a, b in zip(images, expected_figures))
    )
    content_width = (document.sections[0].page_width - document.sections[0].left_margin
                     - document.sections[0].right_margin)
    checks["figures_fit_page_width"] = all(shape.width <= content_width
                                            for shape in document.inline_shapes)
    checks["embedded_figures_at_least_180_dpi"] = all(
        min(Image.open(io.BytesIO(blob)).width / shape.width.inches,
            Image.open(io.BytesIO(blob)).height / shape.height.inches) >= 180
        for shape, blob in zip(document.inline_shapes, images)
    )
    figure_paragraphs = [i for i, p in enumerate(document.paragraphs)
                         if p._p.xpath(".//w:drawing")]
    checks["each_figure_has_centered_caption"] = (
        len(figure_paragraphs) == len(images)
        and all(i + 1 < len(document.paragraphs)
                and document.paragraphs[i + 1].style.name == "Caption"
                and document.paragraphs[i + 1].alignment == 1
                for i in figure_paragraphs)
    )
    published_paths = {item["path"] for item in spec["repository"]["published"]}
    figure_sources_ok = True
    for block in blocks(spec, "figure"):
        source = block.get("source")
        if not source:
            figure_sources_ok = False
        elif source["kind"] == "gitee":
            figure_sources_ok &= source["path"] in published_paths
        elif source["kind"] == "reference_docx":
            reference = project_path(source["path"])
            if reference.is_file() and source.get("media"):
                with ZipFile(reference) as package:
                    figure_sources_ok &= (digest(package.read(source["media"]))
                                          == digest(project_path(block["path"]).read_bytes()))
            else:
                figure_sources_ok = False
        elif source["kind"] == "derived":
            figure_sources_ok &= (project_path(source["input"]).is_file()
                                  and source["input"] in published_paths)
        else:
            figure_sources_ok = False
    checks["figures_have_named_sources"] = bool(figure_sources_ok)
    tables = blocks(spec, "table")
    checks["editable_table_count"] = len(document.tables) == len(tables)
    reference_header = cell_fill(reference.tables[0].cell(0, 0))
    checks["table_color_and_font_match_reference"] = all(
        all(cell_fill(cell) == reference_header for cell in table.rows[0].cells)
        and all(
            (cell_fill(cell) == "EEF3F7" if row_index % 2 == 0 else cell_fill(cell) is None)
            for row_index, row in enumerate(table.rows[1:], 1) for cell in row.cells
        )
        and all(
            paragraph.style.name == "Normal"
            and all(run.font.size is not None and abs(run.font.size.pt - 10.5) < 1e-6
                    for run in paragraph.runs)
            for row in table.rows for cell in row.cells for paragraph in cell.paragraphs
        )
        for table in document.tables
    )
    if "comparison" in spec:
        rows, protocol = checked_comparison(spec)
        table_ok = True
        for table, block in zip(document.tables, tables):
            if len(table.rows) != len(block["methods"]) + 1:
                table_ok = False
                continue
            for index, method in enumerate(block["methods"], 1):
                expected = [method["name"]] + [
                    f"{float(rows[method['id']][key]):.4f}" for key in block["metrics"]
                ]
                if [cell.text for cell in table.rows[index].cells] != expected:
                    table_ok = False
        checks["table_values_match_csv_and_json"] = table_ok
        method_ids = [method["id"] for method in spec["comparison"]["methods"]]
        if "embeddings" in protocol:
            data = [np.load(Path(protocol["embeddings"][name])) for name in method_ids]
            checks["same_node_indices_and_positions"] = bool(all(
                np.array_equal(data[0]["indices"], other["indices"])
                and np.array_equal(data[0]["positions"], other["positions"])
                for other in data[1:]
            ))
        if "metric_subset_path" in protocol:
            subset = np.load(Path(protocol["metric_subset_path"]))
            checks["metric_subset_valid"] = bool(
                subset.size == protocol["metric_subset_size"]
                and np.unique(subset).size == subset.size
                and subset.min() >= 0 and subset.max() < protocol["num_samples"]
            )

    repository = spec["repository"]
    verify_gitee(repository)
    links = {rel.target_ref for rel in document.part.rels.values()
             if rel.reltype == RT.HYPERLINK}
    required = {repository["url"]} | {
        gitee_url(repository, item["path"]) for item in repository["published"]
    }
    checks["gitee_links_point_to_verified_commit"] = required.issubset(links)
    checks["pending_files_not_linked_as_published"] = all(
        item["path"] not in link for item in repository.get("pending", [])
        for link in links
    )

    narrative = "\n".join(
        [spec["report"]["title"], spec["report"]["summary"]]
        + [block.get("text", block.get("caption", ""))
           for page in spec["pages"] for block in page["blocks"]]
    )
    checks["internal_ids_absent_from_reader_text"] = all(
        term not in narrative for term in spec["report"].get("internal_terms", [])
    )
    checks["execution_log_phrases_absent"] = re.search(
        r"运行成功|执行了.{0,12}脚本|\b(?:Sanity|Pilot|checkpoint|console\.log)\b",
        narrative, flags=re.IGNORECASE
    ) is None

    pdf_path = preview_dir / (docx_path.stem + ".pdf")
    pdf = pdfium.PdfDocument(pdf_path)
    checks["rendered_page_count"] = 2 <= len(pdf) == len(spec["pages"]) <= 4
    checks["rendered_pages_are_a4"] = all(
        abs(pdf[i].get_size()[0] - 595.3) < 1
        and abs(pdf[i].get_size()[1] - 841.9) < 1
        for i in range(len(pdf))
    )
    checks["all_page_previews"] = all(
        (preview_dir / f"page-{i}.png").is_file()
        and (preview_dir / f"page-{i}.png").stat().st_size > 10000
        for i in range(1, len(spec["pages"]) + 1)
    )
    manual_review = [
        "逐页检查图内文字、图例、裁切、留白和分页",
        "检查导师能否不读技术日志就理解正文表达",
        "在目标电脑的 Word 中确认字体替换和分页",
    ]
    if repository.get("pending"):
        manual_review.append("待上传证据发布后重新核对 Gitee 链接")
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "docx_sha256": digest(docx_path.read_bytes()),
        "gitee_commit": repository["commit"],
        "published_gitee_paths": [item["path"] for item in repository["published"]],
        "pending_gitee_paths": [item["path"] for item in repository.get("pending", [])],
        "manual_review_required": manual_review,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("docx", type=Path)
    parser.add_argument("preview_dir", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = validate(spec, args.docx, args.preview_dir)
    output = args.docx.parent / "validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["status"], sum(result["checks"].values()), "/",
          len(result["checks"]), output)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
