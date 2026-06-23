#!/usr/bin/env python3
"""Validate external RF/HGB model-comparison recalibration outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_DATASETS = {"eICU", "CHARLS"}
EXPECTED_MODELS = {"random_forest_balanced", "hist_gradient_boosting_weighted"}
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
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    table_dir = project_root / "outputs" / "tables"
    report_path = project_root / "outputs" / "reports" / "external_model_comparison_recalibration.md"
    paths = {
        "predictions": table_dir / "external_model_comparison_recalibration_predictions.csv",
        "calibrators": table_dir / "external_model_comparison_recalibration_calibrators.csv",
        "metrics": table_dir / "external_model_comparison_recalibration_metrics.csv",
        "deciles": table_dir / "external_model_comparison_recalibration_deciles.csv",
        "summary": table_dir / "external_model_comparison_recalibration_summary.csv",
        "decision": table_dir / "external_model_comparison_recalibration_decision_curve.csv",
        "eicu_source": table_dir / "eicu_first24h_model_comparison_predictions.csv",
        "charls_source": table_dir / "charls_incident_diabetes_model_comparison_predictions.csv",
    }
    data = {key: read_csv(path) for key, path in paths.items()}
    rows = []
    for key in ["predictions", "calibrators", "metrics", "deciles", "summary", "decision"]:
        rows.append(row(f"{key}_exist", "PASS" if not data[key].empty else "FAIL", str(paths[key]), f"rows={len(data[key])}"))
    rows.append(row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"))
    if any(data[key].empty for key in ["predictions", "calibrators", "metrics", "deciles", "summary", "decision"]):
        return pd.DataFrame(rows)

    predictions = data["predictions"]
    metrics = data["metrics"]
    deciles = data["deciles"]
    summary = data["summary"]
    decision = data["decision"]
    datasets = set(predictions["dataset"].astype(str))
    models = set(predictions["model"].astype(str))
    methods = set(predictions["calibration_method"].astype(str))
    splits = set(predictions["split"].astype(str))
    rows.append(row("expected_datasets", "PASS" if datasets == EXPECTED_DATASETS else "FAIL", str(paths["predictions"]), "datasets=" + ",".join(sorted(datasets))))
    rows.append(row("expected_models", "PASS" if models == EXPECTED_MODELS else "FAIL", str(paths["predictions"]), "models=" + ",".join(sorted(models))))
    rows.append(row("expected_methods", "PASS" if methods == EXPECTED_METHODS else "FAIL", str(paths["predictions"]), "methods=" + ",".join(sorted(methods))))
    rows.append(row("expected_splits", "PASS" if splits == EXPECTED_SPLITS else "FAIL", str(paths["predictions"]), "splits=" + ",".join(sorted(splits))))

    probability_range = predictions["calibrated_risk"].between(0, 1).all() and predictions["raw_predicted_risk"].between(0, 1).all()
    rows.append(row("probabilities_in_range", "PASS" if probability_range else "FAIL", str(paths["predictions"]), "raw/calibrated risks within [0,1]"))
    expected_source_rows = 0
    for key in ["eicu_source", "charls_source"]:
        source = data[key]
        if not source.empty:
            expected_source_rows += int(source[source["model"].astype(str).isin(EXPECTED_MODELS)].shape[0])
    rows.append(row("prediction_rows_match_sources", "PASS" if len(predictions) == expected_source_rows * len(EXPECTED_METHODS) else "FAIL", str(paths["predictions"]), f"observed={len(predictions)}, expected={expected_source_rows * len(EXPECTED_METHODS)}"))

    expected_grid = {
        (dataset, model, method, split)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
        for method in EXPECTED_METHODS
        for split in EXPECTED_SPLITS
    }
    metric_grid = set(zip(metrics["dataset"].astype(str), metrics["model"].astype(str), metrics["calibration_method"].astype(str), metrics["split"].astype(str)))
    rows.append(row("metric_grid_complete", "PASS" if metric_grid == expected_grid else "FAIL", str(paths["metrics"]), f"rows={len(metrics)}"))
    metric_range = metrics["Brier_score"].between(0, 1).all() and metrics["mean_predicted_risk"].between(0, 1).all()
    rows.append(row("metric_values_in_range", "PASS" if metric_range else "FAIL", str(paths["metrics"]), "Brier and mean prediction within [0,1]"))

    decile_counts = deciles.groupby(["dataset", "model", "calibration_method", "split"])["decile"].nunique()
    bad_deciles = ["/".join(key) for key, value in decile_counts.items() if int(value) != 10]
    rows.append(row("ten_deciles_per_grid_cell", "PASS" if not bad_deciles and len(decile_counts) == len(expected_grid) else "FAIL", str(paths["deciles"]), "bad=" + ",".join(bad_deciles[:20])))
    summary_grid = set(zip(summary["dataset"].astype(str), summary["model"].astype(str), summary["calibration_method"].astype(str), summary["split"].astype(str)))
    rows.append(row("summary_grid_complete", "PASS" if summary_grid == expected_grid else "FAIL", str(paths["summary"]), f"rows={len(summary)}"))
    threshold_counts = decision.groupby(["dataset", "model", "calibration_method", "split"])["threshold_probability"].nunique()
    bad_thresholds = ["/".join(key) for key, value in threshold_counts.items() if int(value) != 8]
    rows.append(row("eight_thresholds_per_grid_cell", "PASS" if not bad_thresholds and len(threshold_counts) == len(expected_grid) else "FAIL", str(paths["decision"]), "bad=" + ",".join(bad_thresholds[:20])))
    preferred = set(decision["preferred_strategy"].astype(str))
    rows.append(row("preferred_strategy_values", "PASS" if preferred <= {"model", "treat_all", "treat_none"} else "FAIL", str(paths["decision"]), "values=" + ",".join(sorted(preferred))))

    failures = []
    for (dataset, model), group in metrics[metrics["split"].astype(str).eq("test")].groupby(["dataset", "model"], sort=True):
        raw = group[group["calibration_method"].astype(str).eq("raw")]
        calibrated = group[~group["calibration_method"].astype(str).eq("raw")]
        if raw.empty or calibrated.empty:
            failures.append(f"{dataset}/{model}:missing")
            continue
        if float(calibrated["Brier_score"].min()) >= float(raw.iloc[0]["Brier_score"]):
            failures.append(f"{dataset}/{model}:brier")
        if float(calibrated["absolute_mean_error"].min()) >= float(raw.iloc[0]["absolute_mean_error"]):
            failures.append(f"{dataset}/{model}:mean_error")
    rows.append(row("test_brier_and_mean_error_improve", "PASS" if not failures else "FAIL", str(paths["metrics"]), "bad=" + ",".join(failures)))
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
    table_path = args.project_root / "outputs" / "tables" / "external_model_comparison_recalibration_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_model_comparison_recalibration_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Model-Comparison Recalibration Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research recalibration outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External model-comparison recalibration validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
