#!/usr/bin/env python3
"""Validate eICU first-24h logistic baseline outputs."""

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
    metrics_path = project_root / "outputs" / "tables" / "eicu_first24h_logistic_baseline_metrics.csv"
    predictions_path = project_root / "outputs" / "tables" / "eicu_first24h_logistic_baseline_predictions.csv"
    coefficients_path = project_root / "outputs" / "tables" / "eicu_first24h_logistic_baseline_coefficients.csv"
    leakage_path = project_root / "outputs" / "tables" / "eicu_leakage_gate.csv"

    metrics = read_csv(metrics_path)
    predictions = read_csv(predictions_path)
    coefficients = read_csv(coefficients_path)
    leakage = read_csv(leakage_path)
    rows = [
        row("metrics_exist", "PASS" if not metrics.empty else "FAIL", str(metrics_path), f"rows={len(metrics)}"),
        row("predictions_exist", "PASS" if not predictions.empty else "FAIL", str(predictions_path), f"rows={len(predictions)}"),
        row("coefficients_exist", "PASS" if not coefficients.empty else "FAIL", str(coefficients_path), f"rows={len(coefficients)}"),
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
    table_path = args.project_root / "outputs" / "tables" / "eicu_first24h_logistic_baseline_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "eicu_first24h_logistic_baseline_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# eICU First-24h Logistic Baseline Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research baseline outputs only; no clinical recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"eICU baseline validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
