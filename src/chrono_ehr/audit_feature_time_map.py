#!/usr/bin/env python3
"""Audit feature availability against prediction time and leakage-risk labels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from validate_study_config import PROJECT, load_yaml_with_ruby


USABILITY_COLUMNS = {
    "admission": "usable_for_admission_prediction",
    "inhospital": "usable_for_inhospital_prediction",
    "discharge": "usable_for_discharge_prediction",
}

RISK_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def read_feature_time_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["variable_name"]: row for row in reader}


def infer_prediction_stage(config: dict[str, Any]) -> str:
    outcome_prediction_time = str(config.get("outcome", {}).get("prediction_time", "")).lower()
    primary = config.get("prediction_times", {}).get("primary", {})
    primary_name = str(primary.get("name", "")).lower() if isinstance(primary, dict) else ""
    primary_time = str(primary.get("time_column", "")).lower() if isinstance(primary, dict) else ""
    combined = " ".join([outcome_prediction_time, primary_name, primary_time])
    if "admit" in combined:
        return "admission"
    if "24h" in combined or "inhospital" in combined:
        return "inhospital"
    return "discharge"


def enabled_feature_sets(config: dict[str, Any]) -> dict[str, list[str]]:
    feature_sets = config.get("feature_sets", {})
    enabled: dict[str, list[str]] = {}
    if not isinstance(feature_sets, dict):
        return enabled
    for name, spec in feature_sets.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("enabled") is True:
            enabled[name] = [str(variable) for variable in spec.get("variables", [])]
    return enabled


def forbidden_features(config: dict[str, Any]) -> list[str]:
    feature_sets = config.get("feature_sets", {})
    forbidden: list[str] = []
    if not isinstance(feature_sets, dict):
        return forbidden
    for name, spec in feature_sets.items():
        if not isinstance(spec, dict):
            continue
        if "forbidden" in name or "high_risk" in name:
            forbidden.extend(str(variable) for variable in spec.get("variables", []))
    return forbidden


def classify_row(variable: str, row: dict[str, str] | None, usability_column: str, feature_role: str) -> dict[str, str]:
    if row is None:
        status = "blocked_or_forbidden" if feature_role == "forbidden_or_high_risk" else "needs_manual_review"
        reason = (
            "变量没有出现在 feature_time_map 中；因为它已经列入 forbidden/high-risk，默认不进入模型。"
            if feature_role == "forbidden_or_high_risk"
            else "变量没有出现在 feature_time_map 中，需要人工补充时间点标注。"
        )
        return {
            "variable_name": variable,
            "feature_role": feature_role,
            "source_table": "unknown",
            "time_available": "unknown",
            "usable_at_prediction_time": "unknown",
            "leakage_risk": "unknown",
            "audit_status": status,
            "reason": reason,
        }

    usable = row.get(usability_column, "unknown").strip().lower()
    risk = row.get("leakage_risk", "unknown").strip().lower()
    if feature_role == "forbidden_or_high_risk":
        status = "blocked_or_forbidden"
    elif usable == "yes":
        if risk == "critical":
            status = "fail_leakage_risk"
        elif risk == "high":
            status = "conditional_review"
        else:
            status = "pass"
    elif usable == "conditional" or risk == "medium":
        status = "conditional_review"
    else:
        status = "fail_leakage_risk"

    return {
        "variable_name": variable,
        "feature_role": feature_role,
        "source_table": row.get("source_table", ""),
        "time_available": row.get("time_available", ""),
        "usable_at_prediction_time": row.get(usability_column, ""),
        "leakage_risk": row.get("leakage_risk", ""),
        "audit_status": status,
        "reason": row.get("reason", ""),
    }


def audit(config: dict[str, Any], feature_map: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stage = infer_prediction_stage(config)
    usability_column = USABILITY_COLUMNS[stage]
    rows: list[dict[str, str]] = []

    for set_name, variables in enabled_feature_sets(config).items():
        for variable in variables:
            item = classify_row(variable, feature_map.get(variable), usability_column, f"enabled:{set_name}")
            rows.append(item)

    for variable in forbidden_features(config):
        rows.append(classify_row(variable, feature_map.get(variable), usability_column, "forbidden_or_high_risk"))

    status_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["audit_status"]] = status_counts.get(row["audit_status"], 0) + 1
        risk_counts[row["leakage_risk"]] = risk_counts.get(row["leakage_risk"], 0) + 1

    summary = {
        "prediction_stage": stage,
        "usability_column": usability_column,
        "variables_checked": len(rows),
        "status_counts": status_counts,
        "risk_counts": risk_counts,
    }
    return rows, summary


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variable_name",
        "feature_role",
        "source_table",
        "time_available",
        "usable_at_prediction_time",
        "leakage_risk",
        "audit_status",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, str]], summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    problem_rows = [
        row
        for row in rows
        if row["audit_status"] in {"conditional_review", "fail_leakage_risk", "needs_manual_review"}
        and not row["feature_role"].startswith("forbidden")
    ]
    forbidden_rows = [row for row in rows if row["feature_role"] == "forbidden_or_high_risk"]
    lines = [
        "# Feature Time Map Audit",
        "",
        "这个报告回答一个很具体的问题：在当前预测时间点，这些变量能不能作为模型特征使用？",
        "",
        "## Summary",
        "",
        f"- Prediction stage: `{summary['prediction_stage']}`",
        f"- Usability column checked: `{summary['usability_column']}`",
        f"- Variables checked: {summary['variables_checked']}",
        f"- Audit status counts: {summary['status_counts']}",
        f"- Leakage risk counts: {summary['risk_counts']}",
        "",
        "## Enabled Feature Issues",
        "",
    ]
    if not problem_rows:
        lines.append("- No enabled feature failed the time-availability audit.")
    else:
        for row in problem_rows:
            lines.append(
                f"- `{row['variable_name']}`: {row['audit_status']}; "
                f"usable={row['usable_at_prediction_time']}; risk={row['leakage_risk']}; {row['reason']}"
            )

    lines.extend(["", "## Forbidden Or High-Risk Variables", ""])
    for row in forbidden_rows:
        lines.append(
            f"- `{row['variable_name']}`: risk={row['leakage_risk']}; usable={row['usable_at_prediction_time']}; {row['reason']}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `pass`：按当前标注，这个变量可以在预测时间点使用。",
            "- `conditional_review`：不是绝对不能用，但需要说明取值窗口，例如只取出院前已出的化验结果。",
            "- `fail_leakage_risk`：当前预测时间点不应使用，容易把未来信息带入模型。",
            "- `blocked_or_forbidden`：这是 outcome、未来事件或明显高风险变量，默认不进入模型。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "diabetes_mimic_readmission.yaml")
    parser.add_argument("--feature-map", type=Path, default=PROJECT / "docs" / "mimic_diabetes_feature_time_map.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_with_ruby(args.config)
    feature_map = read_feature_time_map(args.feature_map)
    rows, summary = audit(config, feature_map)

    table_path = args.project_root / "outputs" / "tables" / "mimic_diabetes_feature_time_audit.csv"
    report_path = args.project_root / "outputs" / "reports" / "mimic_diabetes_feature_time_audit_report.md"
    write_csv(rows, table_path)
    write_report(rows, summary, report_path)

    failing_enabled = [
        row
        for row in rows
        if row["audit_status"] in {"fail_leakage_risk", "needs_manual_review"}
        and not row["feature_role"].startswith("forbidden")
    ]
    print(f"Feature time audit checked {summary['variables_checked']} variables")
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")
    if failing_enabled:
        print(f"Found {len(failing_enabled)} enabled feature problems")
        sys.exit(1)


if __name__ == "__main__":
    main()
