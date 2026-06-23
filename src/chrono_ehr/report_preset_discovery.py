#!/usr/bin/env python3
"""Discover manuscript/report export presets for ChronoEHR-Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


DEFAULT_EXPORT_CONFIG: dict[str, Any] = {
    "language": "zh",
    "outputs": {
        "main_docx": "outputs/reports/ChronoEHR_Methods_Results_Draft.docx",
        "supplement_docx": "outputs/reports/ChronoEHR_Supplementary_Appendix.docx",
    },
    "main_document": {
        "include_sections": {
            "title": True,
            "methods": True,
            "cohort_summary": True,
            "prediction_time_delta": True,
            "feature_group_ablation": True,
            "selected_feature_sets": True,
            "ed_los_sensitivity": True,
            "threshold_analysis": True,
            "decision_curve": True,
            "subgroup_performance": True,
            "next_writing_suggestions": True,
        }
    },
    "supplement": {
        "include_tables": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
        "max_rows": {"S4": 30, "S10": 90},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def preset_label(path: Path) -> str:
    stem = path.stem
    if stem == "manuscript_export_template":
        return "full_zh"
    if stem == "manuscript_export_brief":
        return "brief_zh"
    if stem == "manuscript_export_english_brief":
        return "english_brief"
    return stem.replace("manuscript_export_", "")


def preset_purpose(label: str) -> str:
    if label == "full_zh":
        return "完整中文 Methods/Results draft and supplementary appendix."
    if label == "brief_zh":
        return "Shorter Chinese draft with a reduced supplementary appendix."
    if label == "english_brief":
        return "English brief scaffold for future language-aware DOCX export."
    return "Custom manuscript/report export preset."


def discover_presets(project_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((project_root / "configs").glob("manuscript_export*.json")):
        config = read_json(path)
        merged = {**DEFAULT_EXPORT_CONFIG, **config}
        status = str(merged.get("status", "ready"))
        executable = status != "scaffold_only"
        outputs = merged.get("outputs", {})
        sections = merged.get("main_document", {}).get("include_sections", {})
        enabled_sections = [key for key, value in sections.items() if value]
        disabled_sections = [key for key, value in sections.items() if not value]
        tables = merged.get("supplement", {}).get("include_tables", [])
        main_docx = outputs.get("main_docx", DEFAULT_EXPORT_CONFIG["outputs"]["main_docx"])
        supplement_docx = outputs.get("supplement_docx", DEFAULT_EXPORT_CONFIG["outputs"]["supplement_docx"])
        markdown_draft = "outputs/reports/chronic_disease_methods_results_english_brief.md" if preset_label(path) == "english_brief" else ""
        rows.append(
            {
                "preset": preset_label(path),
                "config_path": str(path.relative_to(project_root)),
                "status": status,
                "executable": executable,
                "language": merged.get("language", "zh"),
                "purpose": preset_purpose(preset_label(path)),
                "main_docx": main_docx,
                "main_docx_exists": (project_root / main_docx).exists(),
                "supplement_docx": supplement_docx,
                "supplement_docx_exists": (project_root / supplement_docx).exists(),
                "markdown_draft": markdown_draft,
                "markdown_draft_exists": bool(markdown_draft and (project_root / markdown_draft).exists()),
                "enabled_main_sections": ", ".join(enabled_sections),
                "disabled_main_sections": ", ".join(disabled_sections),
                "supplement_tables": ", ".join(map(str, tables)),
                "command": f"python3 src/chrono_ehr/run_study.py --manuscript-docx --manuscript-export-config {path.relative_to(project_root)}"
                if executable
                else "Scaffold only; implement English body-text templates before DOCX export.",
            }
        )
    yaml_path = project_root / "configs" / "manuscript_export_template.yaml"
    if yaml_path.exists():
        rows.append(
            {
                "preset": "yaml_reference_template",
                "config_path": str(yaml_path.relative_to(project_root)),
                "status": "reference_only",
                "executable": False,
                "language": "zh",
                "purpose": "Human-readable reference template; current executable exporter uses JSON configs.",
                "main_docx": "",
                "main_docx_exists": False,
                "supplement_docx": "",
                "supplement_docx_exists": False,
                "markdown_draft": "",
                "markdown_draft_exists": False,
                "enabled_main_sections": "",
                "disabled_main_sections": "",
                "supplement_tables": "",
                "command": "Use the matching JSON config instead.",
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.astype(object).where(pd.notna(df), "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, presets: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "report_preset_discovery.md"
    executable = presets[presets["executable"].eq(True)]
    text = f"""# Report Preset Discovery

这个报告列出 ChronoEHR-Agent 当前可用的 manuscript/report 导出 preset。它只读取配置和已生成文件，不重新运行模型，也不重新导出 Word。

## Summary

- Total presets/configs found: {len(presets)}
- Executable JSON presets: {len(executable)}
- Boundary: these are research-report export modes, not clinical note templates and not medical advice.

## Preset Table

{markdown_table(presets)}

## Practical Use

- `full_zh` 适合完整材料包：主 Methods/Results draft + 完整 S1-S10 补充附录。
- `brief_zh` 适合快速发给项目评阅者或自己复盘：保留核心结果，减少部分细长补充表。
- `english_brief` 目前的 DOCX preset 仍是 scaffold；但英文 Markdown brief 可以通过 `python3 src/chrono_ehr/run_study.py --english-brief-draft` 生成，供人工审阅和后续英文 DOCX 模板开发。
- YAML 文件目前是人类可读模板，不直接执行；真正导出请使用 JSON preset。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    presets = discover_presets(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "report_preset_discovery.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    presets.to_csv(table_path, index=False)
    report = write_report(args.project_root, presets)
    print(f"Wrote {report}")
    print(presets[["preset", "executable", "language", "main_docx_exists", "supplement_docx_exists"]].to_string(index=False))


if __name__ == "__main__":
    main()
