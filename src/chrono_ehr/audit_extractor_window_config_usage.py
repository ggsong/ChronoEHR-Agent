#!/usr/bin/env python3
"""Audit which extractors consume the shared feature-window specs."""

from __future__ import annotations

import argparse
from pathlib import Path

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXTRACTORS = [
    {
        "name": "diabetes_lab_24h",
        "script": "src/chrono_ehr/mimic_diabetes_lab_features_24h.py",
        "expected_window": "first_24h",
        "source": "labs",
    },
    {
        "name": "diabetes_med_24h",
        "script": "src/chrono_ehr/mimic_diabetes_med_features_24h.py",
        "expected_window": "first_24h",
        "source": "diabetes_medications",
    },
    {
        "name": "chronic_vitals_24h_discharge",
        "script": "src/chrono_ehr/mimic_chronic_vital_features.py",
        "expected_window": "first_24h, admission_to_discharge",
        "source": "vital_signs",
    },
    {
        "name": "chronic_procedures_24h_discharge",
        "script": "src/chrono_ehr/mimic_chronic_procedure_features.py",
        "expected_window": "first_24h, admission_to_discharge",
        "source": "procedure_events",
    },
    {
        "name": "chronic_general_medications_24h_discharge",
        "script": "src/chrono_ehr/mimic_chronic_med_features.py",
        "expected_window": "first_24h, admission_to_discharge",
        "source": "general_medications",
    },
    {
        "name": "ckd_labs_24h_discharge",
        "script": "src/chrono_ehr/mimic_ckd_lab_features.py",
        "expected_window": "first_24h, admission_to_discharge",
        "source": "labs",
    },
    {
        "name": "heart_failure_labs_24h_discharge",
        "script": "src/chrono_ehr/mimic_heart_failure_lab_features.py",
        "expected_window": "first_24h, admission_to_discharge",
        "source": "labs",
    },
    {
        "name": "hypertension_labs_24h_discharge",
        "script": "src/chrono_ehr/mimic_hypertension_lab_features.py",
        "expected_window": "first_24h, admission_to_discharge",
        "source": "labs",
    },
]


def audit(project_root: Path) -> list[dict[str, str]]:
    rows = []
    for item in EXTRACTORS:
        path = project_root / item["script"]
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        uses_loader = "feature_window_spec_loader" in text
        uses_add_window = "add_window_end" in text
        uses_source_spec = "source_spec" in text
        has_window_arg = "--window-spec" in text
        status = "configured" if uses_loader and has_window_arg else "partial" if uses_loader else "manual"
        rows.append(
            {
                "extractor": item["name"],
                "script": item["script"],
                "source": item["source"],
                "expected_window": item["expected_window"],
                "status": status,
                "uses_loader": str(uses_loader).lower(),
                "uses_add_window": str(uses_add_window).lower(),
                "uses_source_spec": str(uses_source_spec).lower(),
                "has_window_arg": str(has_window_arg).lower(),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    columns = [
        "extractor",
        "source",
        "expected_window",
        "status",
        "uses_loader",
        "uses_add_window",
        "uses_source_spec",
        "has_window_arg",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, rows: list[dict[str, str]]) -> None:
    tables = project_root / "outputs" / "tables"
    reports = project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    pd.DataFrame(rows).to_csv(tables / "extractor_window_config_usage.csv", index=False)
    configured = sum(row["status"] == "configured" for row in rows)
    manual = sum(row["status"] == "manual" for row in rows)
    text = f"""# Extractor Window Config Usage Audit

这个报告检查 feature extractor 是否已经消费 `configs/feature_window_specs.json`。目标不是一次性改完所有脚本，而是把迁移状态变成可审计对象。

- Extractors checked: {len(rows)}
- Configured: {configured}
- Manual / not yet migrated: {manual}

{markdown_table(rows)}

## Interpretation

- `configured`：脚本已读取 feature-window specs，并提供 `--window-spec` 参数。
- `manual`：脚本仍在本地手写时间窗逻辑，后续应迁移。
- `partial`：脚本开始引用 loader，但还缺少命令行参数或完整 source/window 调用。
"""
    (reports / "extractor_window_config_usage_report.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit(args.project_root)
    write_outputs(args.project_root, rows)
    configured = sum(row["status"] == "configured" for row in rows)
    print(f"Extractor window config usage: configured={configured}/{len(rows)}")


if __name__ == "__main__":
    main()
