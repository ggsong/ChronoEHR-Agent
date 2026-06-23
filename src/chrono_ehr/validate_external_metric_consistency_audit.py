#!/usr/bin/env python3
"""Validate the external metric consistency audit output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_GROUPS = {
    "summary_vs_hard_metrics",
    "summary_vs_technical",
    "summary_vs_selection_rationale",
    "summary_vs_bootstrap_ci",
    "technical_vs_calibration_decision",
    "boundary_declarations",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "external_metric_consistency_audit.csv"
    report_path = project_root / "outputs" / "reports" / "external_metric_consistency_audit.md"
    table = read_csv(table_path)
    rows = [
        row("audit_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("audit_report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)

    required_columns = {
        "check_group",
        "benchmark_row",
        "compared_artifacts",
        "field",
        "source_value",
        "target_value",
        "abs_delta",
        "tolerance",
        "status",
        "detail",
    }
    missing = sorted(required_columns - set(table.columns))
    rows.append(row("required_columns_present", "PASS" if not missing else "FAIL", str(table_path), "missing=" + ",".join(missing)))
    groups = set(table["check_group"].astype(str))
    rows.append(row("expected_check_groups", "PASS" if EXPECTED_GROUPS <= groups else "FAIL", str(table_path), "groups=" + ",".join(sorted(groups))))
    rows.append(row("enough_consistency_checks", "PASS" if len(table) >= 200 else "FAIL", str(table_path), f"checks={len(table)}"))
    failures = table[table["status"].astype(str).ne("PASS")]
    rows.append(row("all_consistency_checks_pass", "PASS" if failures.empty else "FAIL", str(table_path), f"failures={len(failures)}"))

    boundary = table[table["check_group"].astype(str).eq("boundary_declarations")]
    rows.append(row("boundary_checks_present", "PASS" if len(boundary) >= 4 else "FAIL", str(table_path), f"boundary_checks={len(boundary)}"))
    rows.append(row("boundary_checks_pass", "PASS" if not boundary.empty and boundary["status"].astype(str).eq("PASS").all() else "FAIL", str(table_path), "CDSL/eICU/CHARLS/decision-curve boundaries"))
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rows.append(row("report_declares_research_boundary", "PASS" if "research model evaluation consistency audit only" in report_text else "FAIL", str(report_path), "research-only boundary"))
    rows.append(row("report_has_pass_status", "PASS" if "Overall status: `PASS`" in report_text else "FAIL", str(report_path), "overall PASS"))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = validate(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "external_metric_consistency_audit_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_metric_consistency_audit_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Metric Consistency Audit Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research consistency-audit outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External metric consistency validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
