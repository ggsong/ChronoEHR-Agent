#!/usr/bin/env python3
"""Compare feature availability across admission, inhospital, and discharge prediction times."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from audit_feature_time_map import USABILITY_COLUMNS, classify_row, read_feature_time_map
from validate_study_config import PROJECT, load_yaml_with_ruby


STAGES = ["admission", "inhospital", "discharge"]


def candidate_features(config: dict[str, Any]) -> dict[str, list[str]]:
    feature_sets = config.get("feature_sets", {})
    mapping: dict[str, list[str]] = {}
    if not isinstance(feature_sets, dict):
        return mapping
    for set_name, spec in feature_sets.items():
        if not isinstance(spec, dict):
            continue
        if "forbidden" in set_name or "high_risk" in set_name:
            continue
        for variable in spec.get("variables", []):
            mapping.setdefault(str(variable), []).append(set_name)
    return mapping


def compare(config: dict[str, Any], feature_map: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidates = candidate_features(config)
    rows: list[dict[str, str]] = []
    for variable, feature_sets in sorted(candidates.items()):
        for stage in STAGES:
            audited = classify_row(
                variable=variable,
                row=feature_map.get(variable),
                usability_column=USABILITY_COLUMNS[stage],
                feature_role="candidate:" + ",".join(feature_sets),
            )
            audited["prediction_stage"] = stage
            audited["feature_sets"] = ",".join(feature_sets)
            rows.append(audited)

    summary: dict[str, Any] = {
        "candidate_variables": len(candidates),
        "stages": STAGES,
        "status_by_stage": {},
        "usable_pass_by_stage": {},
        "blocked_by_stage": {},
    }
    for stage in STAGES:
        stage_rows = [row for row in rows if row["prediction_stage"] == stage]
        status_counts: dict[str, int] = {}
        for row in stage_rows:
            status_counts[row["audit_status"]] = status_counts.get(row["audit_status"], 0) + 1
        summary["status_by_stage"][stage] = status_counts
        summary["usable_pass_by_stage"][stage] = [
            row["variable_name"] for row in stage_rows if row["audit_status"] == "pass"
        ]
        summary["blocked_by_stage"][stage] = [
            row["variable_name"]
            for row in stage_rows
            if row["audit_status"] in {"fail_leakage_risk", "needs_manual_review"}
        ]
    return rows, summary


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prediction_stage",
        "variable_name",
        "feature_sets",
        "source_table",
        "time_available",
        "usable_at_prediction_time",
        "leakage_risk",
        "audit_status",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_symbol(status: str) -> str:
    return {
        "pass": "PASS",
        "conditional_review": "REVIEW",
        "fail_leakage_risk": "BLOCK",
        "needs_manual_review": "REVIEW",
        "blocked_or_forbidden": "BLOCK",
    }.get(status, status.upper())


def write_report(rows: list[dict[str, str]], summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variables = sorted({row["variable_name"] for row in rows})
    by_var_stage = {(row["variable_name"], row["prediction_stage"]): row for row in rows}

    lines = [
        "# Prediction Time Comparison",
        "",
        "这个报告比较同一组候选变量在不同预测时间点下的可用性。它不重新训练模型，只检查变量是否会引入未来信息。",
        "",
        "## Summary",
        "",
        f"- Candidate variables: {summary['candidate_variables']}",
        "",
        "| Prediction stage | PASS | REVIEW | BLOCK |",
        "|---|---:|---:|---:|",
    ]
    for stage in STAGES:
        counts = summary["status_by_stage"][stage]
        review = counts.get("conditional_review", 0) + counts.get("needs_manual_review", 0)
        block = counts.get("fail_leakage_risk", 0) + counts.get("blocked_or_forbidden", 0)
        lines.append(f"| {stage} | {counts.get('pass', 0)} | {review} | {block} |")

    lines.extend(["", "## Variable Matrix", ""])
    lines.extend(
        [
            "| Variable | Admission | Inhospital | Discharge | Why it matters |",
            "|---|---|---|---|---|",
        ]
    )
    for variable in variables:
        admission = by_var_stage[(variable, "admission")]
        inhospital = by_var_stage[(variable, "inhospital")]
        discharge = by_var_stage[(variable, "discharge")]
        reason = discharge["reason"] or admission["reason"] or ""
        lines.append(
            f"| `{variable}` | {status_symbol(admission['audit_status'])} | "
            f"{status_symbol(inhospital['audit_status'])} | {status_symbol(discharge['audit_status'])} | {reason} |"
        )

    lines.extend(["", "## Recommended First-Pass Feature Sets", ""])
    for stage in STAGES:
        pass_vars = summary["usable_pass_by_stage"][stage]
        review_vars = [
            row["variable_name"]
            for row in rows
            if row["prediction_stage"] == stage and row["audit_status"] == "conditional_review"
        ]
        blocked = summary["blocked_by_stage"][stage]
        lines.append(f"### {stage}")
        lines.append("")
        lines.append(f"- Safe first-pass variables: {', '.join(f'`{var}`' for var in pass_vars) or 'None'}")
        lines.append(f"- Needs protocol explanation: {', '.join(f'`{var}`' for var in review_vars) or 'None'}")
        lines.append(f"- Do not use at this stage: {', '.join(f'`{var}`' for var in blocked) or 'None'}")
        lines.append("")

    lines.extend(
        [
            "## Plain-Language Interpretation",
            "",
            "- 入院时预测最严格，不能使用完整住院时长、出院去向、出院后随访事件等未来信息。",
            "- 住院中预测可以使用已经发生、已经有结果的变量，但必须写清楚时间窗，例如前 24 小时。",
            "- 出院时预测可以使用出院前已经知道的信息，但仍要警惕 outcome proxy，例如转归、随访窗口内事件、下一次入院时间。",
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
    rows, summary = compare(config, feature_map)

    table_path = args.project_root / "outputs" / "tables" / "mimic_diabetes_prediction_time_comparison.csv"
    report_path = args.project_root / "outputs" / "reports" / "mimic_diabetes_prediction_time_comparison_report.md"
    write_csv(rows, table_path)
    write_report(rows, summary, report_path)

    print(f"Compared {summary['candidate_variables']} candidate variables across {len(STAGES)} prediction stages")
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
