#!/usr/bin/env python3
"""Validate external calibration-method rationale outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_ROWS = {
    "CDSL early-window best",
    "CDSL full-stay naive reference",
    "eICU calibrated logistic reference",
    "eICU best calibrated RF/HGB",
    "CHARLS calibrated logistic reference",
    "CHARLS best calibrated RF/HGB",
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
    table_path = project_root / "outputs" / "tables" / "external_calibration_method_rationale.csv"
    supp_path = project_root / "outputs" / "tables" / "supplementary_appendix" / "table_s20_external_calibration_method_rationale.csv"
    report_path = project_root / "outputs" / "reports" / "external_calibration_method_rationale.md"
    table = read_csv(table_path)
    rows = [
        row("table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("supplement_exists", "PASS" if exists(supp_path) else "FAIL", str(supp_path), f"size={supp_path.stat().st_size if supp_path.exists() else 0}"),
        row("report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)
    required = {
        "benchmark_row",
        "dataset",
        "feature_set",
        "model",
        "selected_calibration_method",
        "candidate_methods",
        "candidate_method_count",
        "selected_calibration_rank_within_model",
        "best_calibration_method_by_mae",
        "selected_is_best_calibration_mae",
        "selected_mean_absolute_calibration_error",
        "best_mean_absolute_calibration_error",
        "raw_mean_absolute_calibration_error",
        "selected_mae_delta_vs_raw",
        "selected_decision_positive_advantage_thresholds",
        "selected_decision_best_threshold",
        "selected_decision_best_net_benefit_advantage",
        "rationale_status",
        "rationale_note",
        "boundary_note",
    }
    missing = sorted(required - set(table.columns))
    rows.append(row("required_columns_present", "PASS" if not missing else "FAIL", str(table_path), "missing=" + ",".join(missing)))
    labels = set(table["benchmark_row"].astype(str))
    rows.append(row("expected_rows_present", "PASS" if EXPECTED_ROWS <= labels else "FAIL", str(table_path), "rows=" + ",".join(sorted(labels))))
    rows.append(row("exactly_six_rows", "PASS" if len(table) == 6 else "FAIL", str(table_path), f"rows={len(table)}"))
    rows.append(row("rationale_status_pass", "PASS" if table["rationale_status"].astype(str).eq("PASS").all() else "FAIL", str(table_path), "all PASS"))
    rows.append(row("candidate_counts_positive", "PASS" if table["candidate_method_count"].astype(int).gt(0).all() else "FAIL", str(table_path), "candidate_method_count > 0"))
    rows.append(row("calibration_ranks_positive", "PASS" if table["selected_calibration_rank_within_model"].astype(int).gt(0).all() else "FAIL", str(table_path), "rank > 0"))
    metric_ok = True
    for col in ["selected_mean_absolute_calibration_error", "best_mean_absolute_calibration_error", "raw_mean_absolute_calibration_error", "selected_decision_best_threshold"]:
        metric_ok = metric_ok and table[col].dropna().between(0, 1).all()
    rows.append(row("metrics_in_unit_interval", "PASS" if metric_ok else "FAIL", str(table_path), "metrics in [0,1]"))
    methods = set(table["selected_calibration_method"].astype(str))
    rows.append(row("expected_selected_methods", "PASS" if {"raw_traditional", "platt_validation"} <= methods else "FAIL", str(table_path), "methods=" + ",".join(sorted(methods))))
    notes = " ".join(table["boundary_note"].astype(str) + " " + table["rationale_note"].astype(str))
    rows.append(row("boundary_notes_present", "PASS" if "naive upper-reference" in notes and "not chronic readmission" in notes and "longitudinal cohort extension" in notes else "FAIL", str(table_path), "expects dataset boundaries"))
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rows.append(row("report_declares_research_boundary", "PASS" if "research calibration-method rationale only" in report_text else "FAIL", str(report_path), "research boundary"))
    forbidden = ["recommended treatment", "ready for clinical deployment"]
    rows.append(row("report_avoids_clinical_claims", "PASS" if not any(token in report_text.lower() for token in forbidden) else "FAIL", str(report_path), "forbidden wording absent"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_calibration_method_rationale_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_calibration_method_rationale_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Calibration-Method Rationale Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research calibration-method rationale outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External calibration-method rationale validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
