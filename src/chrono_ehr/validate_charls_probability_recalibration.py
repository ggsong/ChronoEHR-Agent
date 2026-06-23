#!/usr/bin/env python3
"""Validate CHARLS probability recalibration outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_METHODS = {"raw", "intercept_validation", "platt_validation", "isotonic_validation"}
EXPECTED_SPLITS = {"train", "validation", "test"}


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
    table_dir = project_root / "outputs" / "tables"
    report_path = project_root / "outputs" / "reports" / "charls_probability_recalibration_report.md"
    paths = {
        "predictions": table_dir / "charls_probability_recalibration_predictions.csv",
        "metrics": table_dir / "charls_probability_recalibration_metrics.csv",
        "deciles": table_dir / "charls_probability_recalibration_deciles.csv",
        "summary": table_dir / "charls_probability_recalibration_summary.csv",
        "decision": table_dir / "charls_probability_recalibration_decision_curve.csv",
        "baseline_predictions": table_dir / "charls_incident_diabetes_logistic_baseline_predictions.csv",
    }
    data = {key: read_csv(path) for key, path in paths.items()}
    rows = []
    for key in ["predictions", "metrics", "deciles", "summary", "decision"]:
        rows.append(row(f"{key}_exist", "PASS" if not data[key].empty else "FAIL", str(paths[key]), f"rows={len(data[key])}"))
    rows.append(row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"))
    if any(data[key].empty for key in ["predictions", "metrics", "deciles", "summary", "decision"]):
        return pd.DataFrame(rows)

    predictions = data["predictions"]
    metrics = data["metrics"]
    deciles = data["deciles"]
    summary = data["summary"]
    decision = data["decision"]
    baseline = data["baseline_predictions"]

    methods = set(predictions["calibration_method"].astype(str))
    rows.append(row("expected_methods", "PASS" if methods == EXPECTED_METHODS else "FAIL", str(paths["predictions"]), "methods=" + ",".join(sorted(methods))))
    splits = set(predictions["split"].astype(str))
    rows.append(row("expected_splits", "PASS" if splits == EXPECTED_SPLITS else "FAIL", str(paths["predictions"]), "splits=" + ",".join(sorted(splits))))
    probability_range = predictions["calibrated_risk"].between(0, 1).all() and predictions["raw_predicted_risk"].between(0, 1).all()
    rows.append(row("probabilities_in_range", "PASS" if probability_range else "FAIL", str(paths["predictions"]), "raw/calibrated risks within [0,1]"))

    if not baseline.empty:
        expected_rows = len(baseline) * len(EXPECTED_METHODS)
        rows.append(row("prediction_rows_match_baseline", "PASS" if len(predictions) == expected_rows else "FAIL", str(paths["predictions"]), f"observed={len(predictions)}, expected={expected_rows}"))

    metric_pairs = set(zip(metrics["calibration_method"].astype(str), metrics["split"].astype(str)))
    expected_pairs = {(method, split) for method in EXPECTED_METHODS for split in EXPECTED_SPLITS}
    rows.append(row("metric_method_split_complete", "PASS" if metric_pairs == expected_pairs else "FAIL", str(paths["metrics"]), f"pairs={len(metric_pairs)}"))
    metric_range = metrics["Brier_score"].between(0, 1).all() and metrics["mean_predicted_risk"].between(0, 1).all()
    rows.append(row("metric_values_in_range", "PASS" if metric_range else "FAIL", str(paths["metrics"]), "Brier and mean prediction within [0,1]"))

    decile_counts = deciles.groupby(["calibration_method", "split"])["decile"].nunique()
    bad_deciles = ["/".join(key) for key, value in decile_counts.items() if int(value) != 10]
    rows.append(row("ten_deciles_per_method_split", "PASS" if not bad_deciles else "FAIL", str(paths["deciles"]), "bad=" + ",".join(bad_deciles)))
    summary_pairs = set(zip(summary["calibration_method"].astype(str), summary["split"].astype(str)))
    rows.append(row("summary_method_split_complete", "PASS" if summary_pairs == expected_pairs else "FAIL", str(paths["summary"]), f"pairs={len(summary_pairs)}"))
    threshold_counts = decision.groupby(["calibration_method", "split"])["threshold_probability"].nunique()
    bad_thresholds = ["/".join(key) for key, value in threshold_counts.items() if int(value) != 8]
    rows.append(row("eight_thresholds_per_method_split", "PASS" if not bad_thresholds else "FAIL", str(paths["decision"]), "bad=" + ",".join(bad_thresholds)))

    preferred = set(decision["preferred_strategy"].astype(str))
    rows.append(row("preferred_strategy_values", "PASS" if preferred <= {"model", "treat_all", "treat_none"} else "FAIL", str(paths["decision"]), "values=" + ",".join(sorted(preferred))))

    test_metrics = metrics[metrics["split"].astype(str).eq("test")].set_index("calibration_method")
    if "raw" in test_metrics.index:
        raw_brier = float(test_metrics.loc["raw", "Brier_score"])
        best_calibrated = float(test_metrics.loc[[m for m in EXPECTED_METHODS if m != "raw"], "Brier_score"].min())
        rows.append(row("test_brier_improves_after_recalibration", "PASS" if best_calibrated < raw_brier else "FAIL", str(paths["metrics"]), f"raw={raw_brier:.6f}, best_calibrated={best_calibrated:.6f}"))
        raw_error = float(test_metrics.loc["raw", "absolute_mean_error"])
        best_error = float(test_metrics.loc[[m for m in EXPECTED_METHODS if m != "raw"], "absolute_mean_error"].min())
        rows.append(row("test_mean_prediction_error_improves", "PASS" if best_error < raw_error else "FAIL", str(paths["metrics"]), f"raw={raw_error:.6f}, best_calibrated={best_error:.6f}"))

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
    table_path = args.project_root / "outputs" / "tables" / "charls_probability_recalibration_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "charls_probability_recalibration_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CHARLS Probability Recalibration Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research recalibration outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CHARLS probability recalibration validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
