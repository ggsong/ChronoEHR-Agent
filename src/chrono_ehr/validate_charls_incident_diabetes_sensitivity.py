#!/usr/bin/env python3
"""Validate CHARLS incident diabetes sensitivity outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_SCENARIOS = {"primary", "no_bmi", "outcome_2013_only", "outcome_2015_only", "age_ge_50", "age_ge_60"}


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
    metrics_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_sensitivity_metrics.csv"
    predictions_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_sensitivity_predictions.csv"
    coefficients_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_sensitivity_coefficients.csv"
    leakage_path = project_root / "outputs" / "tables" / "charls_leakage_gate.csv"
    report_path = project_root / "outputs" / "reports" / "charls_incident_diabetes_sensitivity_report.md"
    metrics = read_csv(metrics_path)
    predictions = read_csv(predictions_path)
    coefficients = read_csv(coefficients_path)
    leakage = read_csv(leakage_path)
    rows = [
        row("metrics_exist", "PASS" if not metrics.empty else "FAIL", str(metrics_path), f"rows={len(metrics)}"),
        row("predictions_exist", "PASS" if not predictions.empty else "FAIL", str(predictions_path), f"rows={len(predictions)}"),
        row("coefficients_exist", "PASS" if not coefficients.empty else "FAIL", str(coefficients_path), f"rows={len(coefficients)}"),
        row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
        row("leakage_gate_passed", "PASS" if not leakage.empty and not leakage["status"].eq("blocked").any() else "FAIL", str(leakage_path), f"blocked={int(leakage['status'].eq('blocked').sum()) if not leakage.empty else 'missing'}"),
    ]
    if metrics.empty:
        return pd.DataFrame(rows)
    scenarios = set(metrics["scenario"].astype(str))
    rows.append(row("expected_scenarios", "PASS" if EXPECTED_SCENARIOS <= scenarios else "FAIL", str(metrics_path), "missing=" + ",".join(sorted(EXPECTED_SCENARIOS - scenarios))))
    split_counts = metrics.groupby("scenario")["split"].nunique()
    bad_splits = sorted(split_counts[split_counts.ne(3)].index.astype(str).tolist())
    rows.append(row("each_scenario_has_three_splits", "PASS" if not bad_splits else "FAIL", str(metrics_path), "bad=" + ",".join(bad_splits)))
    test = metrics[metrics["split"].eq("test")]
    rows.append(row("one_test_row_per_scenario", "PASS" if len(test) == len(EXPECTED_SCENARIOS) else "FAIL", str(metrics_path), f"test_rows={len(test)}"))
    metric_cols = ["AUROC", "AUPRC", "Brier_score"]
    in_range = all(test[col].between(0, 1).all() for col in metric_cols) if not test.empty else False
    rows.append(row("test_metrics_in_range", "PASS" if in_range else "FAIL", str(metrics_path), "AUROC/AUPRC/Brier within [0,1]"))
    events_ok = test["events"].astype(int).gt(0).all() if "events" in test else False
    rows.append(row("test_events_nonzero", "PASS" if events_ok else "FAIL", str(metrics_path), "each test scenario should have events"))
    if not predictions.empty:
        pred_scenarios = set(predictions["scenario"].astype(str))
        risks_ok = predictions["predicted_risk"].between(0, 1).all() if "predicted_risk" in predictions else False
        rows.append(row("prediction_scenarios_present", "PASS" if EXPECTED_SCENARIOS <= pred_scenarios else "FAIL", str(predictions_path), "missing=" + ",".join(sorted(EXPECTED_SCENARIOS - pred_scenarios))))
        rows.append(row("predicted_risk_in_range", "PASS" if risks_ok else "FAIL", str(predictions_path), "risk should be within [0,1]"))
    if not coefficients.empty:
        bad_features = coefficients[~coefficients["feature"].astype(str).str.startswith("charls_baseline_")]
        rows.append(row("coefficient_features_baseline_only", "PASS" if bad_features.empty else "FAIL", str(coefficients_path), f"bad_rows={len(bad_features)}"))
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
    table_path = args.project_root / "outputs" / "tables" / "charls_incident_diabetes_sensitivity_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "charls_incident_diabetes_sensitivity_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CHARLS Incident Diabetes Sensitivity Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research sensitivity outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CHARLS sensitivity validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
