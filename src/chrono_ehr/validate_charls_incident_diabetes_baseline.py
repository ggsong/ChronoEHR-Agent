#!/usr/bin/env python3
"""Validate CHARLS incident diabetes logistic baseline outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_METRIC_COLUMNS = {
    "study",
    "feature_set",
    "prediction_time",
    "model",
    "split",
    "n",
    "events",
    "event_rate",
    "AUROC",
    "AUPRC",
    "Brier_score",
    "feature_count",
}
LABEL_COLUMN = "incident_diabetes_2013_or_2015"
FORBIDDEN_FEATURE_TOKENS = ["followup", "2013", "2015", "r2", "r3", "outcome", "diabetes_2013", "diabetes_2015"]


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
    metrics_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_logistic_baseline_metrics.csv"
    predictions_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_logistic_baseline_predictions.csv"
    coefficients_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_logistic_baseline_coefficients.csv"
    leakage_path = project_root / "outputs" / "tables" / "charls_leakage_gate.csv"
    matrix_path = project_root / "data" / "processed" / "charls_incident_diabetes_baseline_features.csv"
    report_path = project_root / "outputs" / "reports" / "charls_incident_diabetes_logistic_baseline_report.md"

    metrics = read_csv(metrics_path)
    predictions = read_csv(predictions_path)
    coefficients = read_csv(coefficients_path)
    leakage = read_csv(leakage_path)
    matrix = read_csv(matrix_path)
    rows = [
        row("metrics_exist", "PASS" if not metrics.empty else "FAIL", str(metrics_path), f"rows={len(metrics)}"),
        row("predictions_exist", "PASS" if not predictions.empty else "FAIL", str(predictions_path), f"rows={len(predictions)}"),
        row("coefficients_exist", "PASS" if not coefficients.empty else "FAIL", str(coefficients_path), f"rows={len(coefficients)}"),
        row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
        row("leakage_gate_passed", "PASS" if not leakage.empty and not leakage["status"].eq("blocked").any() else "FAIL", str(leakage_path), f"blocked={int(leakage['status'].eq('blocked').sum()) if not leakage.empty else 'missing'}"),
    ]
    if metrics.empty:
        return pd.DataFrame(rows)

    missing = sorted(REQUIRED_METRIC_COLUMNS - set(metrics.columns))
    rows.append(row("required_metric_columns", "PASS" if not missing else "FAIL", str(metrics_path), "missing=" + ",".join(missing)))
    split_values = set(metrics["split"].astype(str)) if "split" in metrics else set()
    rows.append(row("expected_splits", "PASS" if split_values == {"train", "validation", "test"} else "FAIL", str(metrics_path), "splits=" + ",".join(sorted(split_values))))
    test = metrics[metrics["split"].eq("test")]
    rows.append(row("single_test_row", "PASS" if len(test) == 1 else "FAIL", str(metrics_path), f"test_rows={len(test)}"))
    if not test.empty:
        item = test.iloc[0]
        metric_ok = pd.notna(item["AUROC"]) and pd.notna(item["AUPRC"]) and pd.notna(item["Brier_score"])
        range_ok = 0 <= float(item["AUROC"]) <= 1 and 0 <= float(item["AUPRC"]) <= 1 and 0 <= float(item["Brier_score"]) <= 1
        rows.append(row("test_metrics_nonmissing", "PASS" if metric_ok else "FAIL", str(metrics_path), f"AUROC={item['AUROC']}, AUPRC={item['AUPRC']}, Brier={item['Brier_score']}"))
        rows.append(row("test_metrics_in_range", "PASS" if range_ok else "FAIL", str(metrics_path), f"AUROC={item['AUROC']}, AUPRC={item['AUPRC']}, Brier={item['Brier_score']}"))
        rows.append(row("reports_auroc_and_auprc", "PASS" if {"AUROC", "AUPRC"}.issubset(metrics.columns) else "FAIL", str(metrics_path), "both AUROC and AUPRC required"))
    if not predictions.empty and not test.empty:
        test_n = int(test.iloc[0]["n"])
        pred_test_n = int(predictions[predictions["split"].eq("test")].shape[0])
        rows.append(row("prediction_rows_match_test_n", "PASS" if pred_test_n == test_n else "FAIL", str(predictions_path), f"pred_test={pred_test_n}, metric_test={test_n}"))
        in_range = predictions["predicted_risk"].between(0, 1).all() if "predicted_risk" in predictions else False
        rows.append(row("predicted_risk_in_range", "PASS" if in_range else "FAIL", str(predictions_path), "risk should be within [0,1]"))
        label_values = sorted(predictions[LABEL_COLUMN].dropna().unique().tolist()) if LABEL_COLUMN in predictions else []
        rows.append(row("prediction_label_binary", "PASS" if set(label_values).issubset({0, 1}) and len(label_values) == 2 else "FAIL", str(predictions_path), f"values={label_values}"))
    if not coefficients.empty and "feature" in coefficients:
        bad_features = [
            feature
            for feature in coefficients["feature"].astype(str)
            if not feature.startswith("charls_baseline_") or any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
        ]
        rows.append(row("coefficient_features_baseline_only", "PASS" if not bad_features else "FAIL", str(coefficients_path), "bad=" + ",".join(bad_features[:20])))
    if not matrix.empty and not metrics.empty:
        feature_count = len([column for column in matrix.columns if column.startswith("charls_baseline_")])
        metric_counts_ok = metrics["feature_count"].astype(int).eq(feature_count).all()
        rows.append(row("metric_feature_count_matches_matrix", "PASS" if metric_counts_ok else "FAIL", str(matrix_path), f"matrix_features={feature_count}"))
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
    table_path = args.project_root / "outputs" / "tables" / "charls_incident_diabetes_logistic_baseline_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "charls_incident_diabetes_logistic_baseline_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CHARLS Incident Diabetes Logistic Baseline Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research baseline outputs only; no clinical recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CHARLS logistic baseline validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
