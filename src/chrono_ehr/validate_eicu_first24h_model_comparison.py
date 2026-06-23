#!/usr/bin/env python3
"""Validate eICU first-24h model-comparison outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


LABEL_COLUMN = "hospital_mortality"
EXPECTED_MODELS = {
    "logistic_regression_balanced",
    "random_forest_balanced",
    "hist_gradient_boosting_weighted",
}
EXPECTED_SPLITS = {"train", "validation", "test"}
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
FORBIDDEN_FEATURE_TOKENS = ["mortality", "discharge", "death", "expired", "future", "offset"]


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
    metrics_path = project_root / "outputs" / "tables" / "eicu_first24h_model_comparison_metrics.csv"
    predictions_path = project_root / "outputs" / "tables" / "eicu_first24h_model_comparison_predictions.csv"
    importances_path = project_root / "outputs" / "tables" / "eicu_first24h_model_comparison_importances.csv"
    report_path = project_root / "outputs" / "reports" / "eicu_first24h_model_comparison_report.md"
    leakage_path = project_root / "outputs" / "tables" / "eicu_leakage_gate.csv"
    matrix_path = project_root / "data" / "processed" / "eicu_first24h_feature_matrix_skeleton.csv"

    metrics = read_csv(metrics_path)
    predictions = read_csv(predictions_path)
    importances = read_csv(importances_path)
    leakage = read_csv(leakage_path)
    matrix = read_csv(matrix_path)
    rows = [
        row("metrics_exist", "PASS" if not metrics.empty else "FAIL", str(metrics_path), f"rows={len(metrics)}"),
        row("predictions_exist", "PASS" if not predictions.empty else "FAIL", str(predictions_path), f"rows={len(predictions)}"),
        row("importances_exist", "PASS" if not importances.empty else "FAIL", str(importances_path), f"rows={len(importances)}"),
        row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
        row("leakage_gate_passed", "PASS" if not leakage.empty and not leakage["status"].eq("blocked").any() else "FAIL", str(leakage_path), f"blocked={int(leakage['status'].eq('blocked').sum()) if not leakage.empty else 'missing'}"),
    ]
    if metrics.empty:
        return pd.DataFrame(rows)

    missing = sorted(REQUIRED_METRIC_COLUMNS - set(metrics.columns))
    rows.append(row("required_metric_columns", "PASS" if not missing else "FAIL", str(metrics_path), "missing=" + ",".join(missing)))
    models = set(metrics["model"].astype(str)) if "model" in metrics else set()
    splits = set(metrics["split"].astype(str)) if "split" in metrics else set()
    rows.append(row("expected_models", "PASS" if models == EXPECTED_MODELS else "FAIL", str(metrics_path), "models=" + ",".join(sorted(models))))
    rows.append(row("expected_splits", "PASS" if splits == EXPECTED_SPLITS else "FAIL", str(metrics_path), "splits=" + ",".join(sorted(splits))))
    rows.append(row("model_split_grid_complete", "PASS" if len(metrics) == len(EXPECTED_MODELS) * len(EXPECTED_SPLITS) else "FAIL", str(metrics_path), f"rows={len(metrics)}"))
    test = metrics[metrics["split"].eq("test")]
    rows.append(row("test_rows_per_model", "PASS" if len(test) == len(EXPECTED_MODELS) else "FAIL", str(metrics_path), f"test_rows={len(test)}"))
    if not test.empty:
        metric_ok = test[["AUROC", "AUPRC", "Brier_score"]].notna().all().all()
        range_ok = (
            test["AUROC"].between(0, 1).all()
            and test["AUPRC"].between(0, 1).all()
            and test["Brier_score"].between(0, 1).all()
        )
        rows.append(row("test_metrics_nonmissing", "PASS" if metric_ok else "FAIL", str(metrics_path), "AUROC/AUPRC/Brier required"))
        rows.append(row("test_metrics_in_range", "PASS" if range_ok else "FAIL", str(metrics_path), "metrics should be within [0,1]"))
    if not predictions.empty:
        in_range = predictions["predicted_risk"].between(0, 1).all() if "predicted_risk" in predictions else False
        rows.append(row("predicted_risk_in_range", "PASS" if in_range else "FAIL", str(predictions_path), "risk should be within [0,1]"))
        label_values = sorted(predictions[LABEL_COLUMN].dropna().unique().tolist()) if LABEL_COLUMN in predictions else []
        rows.append(row("prediction_label_binary", "PASS" if set(label_values).issubset({0, 1}) and len(label_values) == 2 else "FAIL", str(predictions_path), f"values={label_values}"))
        if not matrix.empty:
            expected_rows = len(matrix) * len(EXPECTED_MODELS)
            rows.append(row("prediction_rows_match_matrix_models", "PASS" if len(predictions) == expected_rows else "FAIL", str(predictions_path), f"predictions={len(predictions)}, expected={expected_rows}"))
    if not importances.empty and "feature" in importances:
        bad_features = [
            feature
            for feature in importances["feature"].astype(str)
            if feature != "not_available"
            and (
                not feature.startswith(("eicu_lab24h_", "eicu_vital24h_"))
                or any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
            )
        ]
        rows.append(row("importance_features_first24h_only", "PASS" if not bad_features else "FAIL", str(importances_path), "bad=" + ",".join(bad_features[:20])))
        importance_models = set(importances["model"].astype(str)) if "model" in importances else set()
        rows.append(row("importance_models_present", "PASS" if EXPECTED_MODELS.issubset(importance_models) else "FAIL", str(importances_path), "models=" + ",".join(sorted(importance_models))))
    if not matrix.empty and not metrics.empty:
        feature_count = len([column for column in matrix.columns if column.startswith(("eicu_lab24h_", "eicu_vital24h_"))])
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
    table_path = args.project_root / "outputs" / "tables" / "eicu_first24h_model_comparison_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "eicu_first24h_model_comparison_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# eICU First-24h Model Comparison Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research model-comparison outputs only; no clinical recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"eICU model comparison validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
