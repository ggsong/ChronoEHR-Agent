#!/usr/bin/env python3
"""Validate CHARLS calibration and decision-curve outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    deciles_path = project_root / "outputs" / "tables" / "charls_calibration_deciles.csv"
    summary_path = project_root / "outputs" / "tables" / "charls_calibration_summary.csv"
    decision_path = project_root / "outputs" / "tables" / "charls_decision_curve.csv"
    predictions_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_logistic_baseline_predictions.csv"
    report_path = project_root / "outputs" / "reports" / "charls_calibration_decision_curve_report.md"
    deciles = read_csv(deciles_path)
    summary = read_csv(summary_path)
    decision = read_csv(decision_path)
    predictions = read_csv(predictions_path)
    rows = [
        row("deciles_exist", "PASS" if not deciles.empty else "FAIL", str(deciles_path), f"rows={len(deciles)}"),
        row("summary_exist", "PASS" if not summary.empty else "FAIL", str(summary_path), f"rows={len(summary)}"),
        row("decision_curve_exist", "PASS" if not decision.empty else "FAIL", str(decision_path), f"rows={len(decision)}"),
        row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
    ]
    if deciles.empty or summary.empty or decision.empty:
        return pd.DataFrame(rows)

    split_values = set(summary["split"].astype(str))
    rows.append(row("expected_splits", "PASS" if split_values == {"train", "validation", "test"} else "FAIL", str(summary_path), "splits=" + ",".join(sorted(split_values))))
    decile_counts = deciles.groupby("split")["decile"].nunique()
    bad_deciles = sorted(decile_counts[decile_counts.ne(10)].index.astype(str).tolist())
    rows.append(row("ten_deciles_per_split", "PASS" if not bad_deciles else "FAIL", str(deciles_path), "bad=" + ",".join(bad_deciles)))
    calibration_range = deciles["mean_predicted_risk"].between(0, 1).all() and deciles["observed_event_rate"].between(0, 1).all()
    rows.append(row("calibration_rates_in_range", "PASS" if calibration_range else "FAIL", str(deciles_path), "predicted/observed rates within [0,1]"))
    summary_range = summary["mean_absolute_calibration_error"].between(0, 1).all() and summary["max_absolute_calibration_error"].between(0, 1).all()
    rows.append(row("summary_errors_in_range", "PASS" if summary_range else "FAIL", str(summary_path), "calibration errors within [0,1]"))
    threshold_counts = decision.groupby("split")["threshold_probability"].nunique()
    bad_thresholds = sorted(threshold_counts[threshold_counts.ne(8)].index.astype(str).tolist())
    rows.append(row("expected_thresholds_per_split", "PASS" if not bad_thresholds else "FAIL", str(decision_path), "bad=" + ",".join(bad_thresholds)))
    curve_range = decision["threshold_probability"].between(0, 1).all() and decision["alert_rate"].between(0, 1).all()
    rows.append(row("decision_curve_rates_in_range", "PASS" if curve_range else "FAIL", str(decision_path), "threshold/alert rates within [0,1]"))
    if not predictions.empty:
        test_pred = predictions[predictions["split"].astype(str).eq("test")]
        test_deciles = deciles[deciles["split"].astype(str).eq("test")]
        rows.append(row("test_decile_rows_sum_to_predictions", "PASS" if int(test_deciles["n"].sum()) == len(test_pred) else "FAIL", str(deciles_path), f"decile_n={int(test_deciles['n'].sum())}, predictions={len(test_pred)}"))
    preferred = set(decision["preferred_strategy"].astype(str))
    rows.append(row("preferred_strategy_values", "PASS" if preferred <= {"model", "treat_all", "treat_none"} else "FAIL", str(decision_path), "values=" + ",".join(sorted(preferred))))
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
    table_path = args.project_root / "outputs" / "tables" / "charls_calibration_decision_curve_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "charls_calibration_decision_curve_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CHARLS Calibration and Decision-Curve Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research evaluation outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CHARLS calibration/decision validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
