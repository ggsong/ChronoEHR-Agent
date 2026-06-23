#!/usr/bin/env python3
"""Audit the English brief DOCX draft and rendered review pages."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def extract_docx_text_and_table_count(docx_path: Path) -> tuple[str, int]:
    if not docx_path.exists():
        return "", 0
    with zipfile.ZipFile(docx_path) as archive:
        if "word/document.xml" not in archive.namelist():
            return "", 0
        root = ET.fromstring(archive.read("word/document.xml"))
    text = "\n".join(node.text or "" for node in root.findall(".//w:t", WORD_NS))
    table_count = len(root.findall(".//w:tbl", WORD_NS))
    return text, table_count


def audit(project_root: Path) -> pd.DataFrame:
    docx_path = project_root / "outputs" / "reports" / "ChronoEHR_Methods_Results_English_Brief.docx"
    pdf_path = project_root / "outputs" / "reports" / "rendered_english_brief" / "ChronoEHR_Methods_Results_English_Brief.pdf"
    rendered_dir = project_root / "outputs" / "reports" / "rendered_english_brief"
    text, table_count = extract_docx_text_and_table_count(docx_path)
    rows: list[dict[str, str]] = []

    rows.append(
        row(
            "docx_exists",
            "PASS" if docx_path.exists() and docx_path.stat().st_size > 0 else "FAIL",
            str(docx_path),
            "English brief DOCX draft must exist and be non-empty.",
        )
    )
    rows.append(
        row(
            "pdf_render_exists",
            "PASS" if pdf_path.exists() and pdf_path.stat().st_size > 0 else "FAIL",
            str(pdf_path),
            "Rendered PDF review artifact must exist.",
        )
    )

    required_phrases = [
        "ChronoEHR-Agent",
        "Research-use boundary",
        "not a medical QA system",
        "does not provide diagnosis or treatment recommendations",
        "Structured Abstract",
        "Methods Brief",
        "Results Brief",
        "Traditional Baseline Models",
        "Leakage And Sensitivity Interpretation",
        "Limitations",
    ]
    for phrase in required_phrases:
        rows.append(row(f"docx_phrase:{phrase[:35]}", "PASS" if phrase in text else "FAIL", str(docx_path), phrase))

    rows.append(
        row(
            "docx_table_count",
            "PASS" if table_count >= 4 else "FAIL",
            str(docx_path),
            f"tables={table_count}; expected at least 4 tables in the brief.",
        )
    )

    pages = sorted(rendered_dir.glob("page-*.png"))
    rows.append(
        row(
            "rendered_page_count",
            "PASS" if len(pages) == 3 else "FAIL",
            str(rendered_dir),
            f"pages={len(pages)}; expected 3 rendered review pages.",
        )
    )
    for page in pages:
        rows.append(
            row(
                f"rendered_page_nonempty:{page.name}",
                "PASS" if page.stat().st_size > 0 else "FAIL",
                str(page),
                f"size={page.stat().st_size} bytes",
            )
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, checks: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "english_brief_docx_audit.md"
    failures = checks[checks["status"].ne("PASS")]
    text = f"""# English Brief DOCX Audit

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This audit checks whether the English brief DOCX review draft exists, contains the required research-use boundary and core sections, includes tables, and has rendered review pages. It is a document-quality check for a local research tool, not a clinical deployment review.

## Check Table

{markdown_table(checks)}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "english_brief_docx_audit.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report = write_report(args.project_root, checks)
    failures = int((checks["status"] != "PASS").sum())
    print(f"Wrote {report}")
    print(f"English brief DOCX audit checks: {len(checks)}")
    print(f"Failures: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
