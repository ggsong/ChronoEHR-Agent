#!/usr/bin/env python3
"""Summarize logistic regression, random forest, and gradient boosting baselines across cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


LOGISTIC_TABLE = "outputs/tables/chronic_disease_prediction_time_benchmark.csv"
RF_TABLE = "outputs/tables/random_forest_baseline_performance.csv"
CALIBRATED_RF_TABLE = "outputs/tables/calibrated_random_forest_performance.csv"
GRADIENT_BOOSTING_TABLE = "outputs/tables/gradient_boosting_baseline_performance.csv"
CALIBRATED_GB_TABLE = "outputs/tables/calibrated_gradient_boosting_performance.csv"


def load_logistic(project_root: Path) -> pd.DataFrame:
    path = project_root / LOGISTIC_TABLE
    if not path.exists():
        raise FileNotFoundError(f"Missing logistic benchmark table: {path}")
    df = pd.read_csv(path)
    df = df[df["prediction_time"].eq("discharge")].copy()
    primary_feature_sets = {
        "diabetes": "discharge_safe_minimal",
        "ckd": "discharge_lab_minimal",
        "heart_failure": "discharge_lab_minimal",
        "hypertension": "discharge_lab_minimal",
    }
    df = df[df.apply(lambda row: row["feature_set"] == primary_feature_sets.get(row["cohort"]), axis=1)].copy()
    df["model"] = "logistic_regression"
    df["study"] = df["cohort"]
    return df[
        [
            "study",
            "cohort",
            "feature_set",
            "model",
            "n",
            "events",
            "event_rate",
            "AUROC",
            "AUPRC",
            "Brier_score",
        ]
    ]


def load_random_forest(project_root: Path) -> pd.DataFrame:
    path = project_root / RF_TABLE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "status" in df.columns and "skipped" in set(df["status"].astype(str)):
        return pd.DataFrame()
    df = df[df["split"].eq("test")].copy()
    if df.empty:
        return df
    df["cohort"] = df["study"]
    return df[
        [
            "study",
            "cohort",
            "feature_set",
            "model",
            "n",
            "events",
            "event_rate",
            "AUROC",
            "AUPRC",
            "Brier_score",
        ]
    ]


def load_calibrated_random_forest(project_root: Path) -> pd.DataFrame:
    path = project_root / CALIBRATED_RF_TABLE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["split"].eq("test")].copy()
    if df.empty:
        return df
    df["cohort"] = df["study"]
    return df[
        [
            "study",
            "cohort",
            "feature_set",
            "model",
            "n",
            "events",
            "event_rate",
            "AUROC",
            "AUPRC",
            "Brier_score",
        ]
    ]


def load_gradient_boosting(project_root: Path) -> pd.DataFrame:
    path = project_root / GRADIENT_BOOSTING_TABLE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df[df["split"].eq("test")].copy()
    if df.empty:
        return df
    if "cohort" not in df.columns:
        df["cohort"] = df["study"]
    return df[
        [
            "study",
            "cohort",
            "feature_set",
            "model",
            "n",
            "events",
            "event_rate",
            "AUROC",
            "AUPRC",
            "Brier_score",
        ]
    ]


def load_calibrated_gradient_boosting(project_root: Path) -> pd.DataFrame:
    path = project_root / CALIBRATED_GB_TABLE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df[df["split"].eq("test")].copy()
    if df.empty:
        return df
    if "cohort" not in df.columns:
        df["cohort"] = df["study"]
    return df[
        [
            "study",
            "cohort",
            "feature_set",
            "model",
            "n",
            "events",
            "event_rate",
            "AUROC",
            "AUPRC",
            "Brier_score",
        ]
    ]


def compute_delta(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in comparison.groupby("cohort", sort=False):
        logistic = group[group["model"].eq("logistic_regression")]
        if logistic.empty:
            continue
        log_row = logistic.iloc[0]
        for _, model_row in group[~group["model"].eq("logistic_regression")].iterrows():
            rows.append(
                {
                    "cohort": cohort,
                    "model": model_row["model"],
                    "logistic_feature_set": log_row["feature_set"],
                    "model_feature_set": model_row["feature_set"],
                    "delta_AUROC_minus_logistic": float(model_row["AUROC"] - log_row["AUROC"]),
                    "delta_AUPRC_minus_logistic": float(model_row["AUPRC"] - log_row["AUPRC"]),
                    "delta_Brier_minus_logistic": float(model_row["Brier_score"] - log_row["Brier_score"]),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No data found."
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in df[columns].itertuples(index=False):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(comparison: pd.DataFrame, delta: pd.DataFrame, output: Path) -> None:
    comparison_cols = ["cohort", "model", "feature_set", "n", "events", "AUROC", "AUPRC", "Brier_score"]
    delta_cols = [
        "cohort",
        "model",
        "delta_AUROC_minus_logistic",
        "delta_AUPRC_minus_logistic",
        "delta_Brier_minus_logistic",
    ]

    if delta.empty:
        interpretation = "- Random Forest results are not available yet."
    else:
        lines = []
        for model, part in delta.groupby("model", sort=False):
            positive_auroc = int((part["delta_AUROC_minus_logistic"] > 0).sum())
            positive_auprc = int((part["delta_AUPRC_minus_logistic"] > 0).sum())
            worse_brier = int((part["delta_Brier_minus_logistic"] > 0).sum())
            lines.append(f"- `{model}` AUROC improved in {positive_auroc}/{len(part)} cohorts.")
            lines.append(f"- `{model}` AUPRC improved in {positive_auprc}/{len(part)} cohorts.")
            lines.append(f"- `{model}` Brier score was higher, meaning worse calibration, in {worse_brier}/{len(part)} cohorts.")
        lines.append("- This is a useful baseline result: stronger ranking performance does not automatically mean better calibrated risk estimates.")
        interpretation = "\n".join(lines)

    text = f"""# Chronic Disease Model Baseline Comparison

这个报告比较每个慢病队列出院时安全特征集上的 logistic regression、Random Forest 和 gradient boosting baseline。

## Test Set Metrics

{markdown_table(comparison, [col for col in comparison_cols if col in comparison.columns])}

## Random Forest Minus Logistic

{markdown_table(delta, [col for col in delta_cols if col in delta.columns])}

## Interpretation

{interpretation}

## Notes

- Logistic regression remains the primary dependency-light baseline.
- Random Forest and gradient boosting are included as traditional ML comparators, not as clinical decision systems.
- Brier score should be read carefully: lower is better, and tree ensembles may need calibration before risk-probability use.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    logistic = load_logistic(args.project_root)
    rf = load_random_forest(args.project_root)
    calibrated_rf = load_calibrated_random_forest(args.project_root)
    gradient_boosting = load_gradient_boosting(args.project_root)
    calibrated_gradient_boosting = load_calibrated_gradient_boosting(args.project_root)
    parts = [logistic]
    if not rf.empty:
        parts.append(rf)
    if not calibrated_rf.empty:
        parts.append(calibrated_rf)
    if not gradient_boosting.empty:
        parts.append(gradient_boosting)
    if not calibrated_gradient_boosting.empty:
        parts.append(calibrated_gradient_boosting)
    comparison = pd.concat(parts, ignore_index=True)
    comparison = comparison.sort_values(["cohort", "model"]).reset_index(drop=True)
    delta = compute_delta(comparison)

    comparison.to_csv(tables / "chronic_disease_model_baseline_comparison.csv", index=False)
    delta.to_csv(tables / "chronic_disease_model_baseline_delta.csv", index=False)
    write_report(comparison, delta, reports / "chronic_disease_model_baseline_comparison.md")

    print("Model baseline summary complete")
    print(f"comparison_rows={len(comparison)} delta_rows={len(delta)}")


if __name__ == "__main__":
    main()
