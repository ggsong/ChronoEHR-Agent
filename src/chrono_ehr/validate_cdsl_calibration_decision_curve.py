#!/usr/bin/env python3
"""Validate CDSL calibration and decision-curve outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_SPLITS = {"train", "val", "test"}


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
    deciles_path = table_dir / "cdsl_calibration_deciles.csv"
    summary_path = table_dir / "cdsl_calibration_summary.csv"
    decision_path = table_dir / "cdsl_decision_curve.csv"
    predictions_path = table_dir / "cdsl_traditional_baselines_predictions.csv"
    report_path = project_root / "outputs" / "reports" / "cdsl_calibration_decision_curve_report.md"
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
    if deciles.empty or summary.empty or decision.empty or predictions.empty:
        return pd.DataFrame(rows)

    expected_pairs = set(zip(predictions["feature_set"].astype(str), predictions["model"].astype(str)))
    expected_group_count = len(expected_pairs) * len(EXPECTED_SPLITS)
    summary_groups = set(zip(summary["feature_set"].astype(str), summary["model"].astype(str), summary["split"].astype(str)))
    rows.append(row("summary_group_complete", "PASS" if len(summary_groups) == expected_group_count else "FAIL", str(summary_path), f"groups={len(summary_groups)}, expected={expected_group_count}"))
    split_values = set(summary["split"].astype(str))
    rows.append(row("expected_splits", "PASS" if split_values == EXPECTED_SPLITS else "FAIL", str(summary_path), "splits=" + ",".join(sorted(split_values))))

    decile_counts = deciles.groupby(["feature_set", "model", "split"])["decile"].nunique()
    bad_deciles = ["/".join(key) for key, value in decile_counts.items() if int(value) != 10]
    rows.append(row("ten_deciles_per_group", "PASS" if not bad_deciles else "FAIL", str(deciles_path), "bad=" + ",".join(bad_deciles[:10])))
    calibration_range = deciles["mean_predicted_risk"].between(0, 1).all() and deciles["observed_event_rate"].between(0, 1).all()
    rows.append(row("calibration_rates_in_range", "PASS" if calibration_range else "FAIL", str(deciles_path), "predicted/observed rates within [0,1]"))
    summary_range = summary["mean_absolute_calibration_error"].between(0, 1).all() and summary["max_absolute_calibration_error"].between(0, 1).all()
    rows.append(row("summary_errors_in_range", "PASS" if summary_range else "FAIL", str(summary_path), "calibration errors within [0,1]"))

    threshold_counts = decision.groupby(["feature_set", "model", "split"])["threshold_probability"].nunique()
    bad_thresholds = ["/".join(key) for key, value in threshold_counts.items() if int(value) != 8]
    rows.append(row("eight_thresholds_per_group", "PASS" if not bad_thresholds else "FAIL", str(decision_path), "bad=" + ",".join(bad_thresholds[:10])))
    curve_range = decision["threshold_probability"].between(0, 1).all() and decision["alert_rate"].between(0, 1).all()
    rows.append(row("decision_curve_rates_in_range", "PASS" if curve_range else "FAIL", str(decision_path), "threshold/alert rates within [0,1]"))
    preferred = set(decision["preferred_strategy"].astype(str))
    rows.append(row("preferred_strategy_values", "PASS" if preferred <= {"model", "treat_all", "treat_none"} else "FAIL", str(decision_path), "values=" + ",".join(sorted(preferred))))

    test_deciles = deciles[deciles["split"].astype(str).eq("test")]
    test_pred = predictions[predictions["split"].astype(str).eq("test")]
    decile_total = int(test_deciles["n"].sum())
    expected_total = len(test_pred)
    rows.append(row("test_decile_rows_sum_to_predictions", "PASS" if decile_total == expected_total else "FAIL", str(deciles_path), f"decile_n={decile_total}, predictions={expected_total}"))
    positive_advantage = int((decision[decision["split"].astype(str).eq("test")]["net_benefit_advantage"] > 0).sum())
    rows.append(row("test_has_positive_model_advantage_rows", "PASS" if positive_advantage > 0 else "FAIL", str(decision_path), f"positive_rows={positive_advantage}"))
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
    table_path = args.project_root / "outputs" / "tables" / "cdsl_calibration_decision_curve_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "cdsl_calibration_decision_curve_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CDSL Calibration and Decision-Curve Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research evaluation outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CDSL calibration/decision validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
