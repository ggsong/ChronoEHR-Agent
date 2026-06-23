#!/usr/bin/env python3
"""Create calibration decile summaries for logistic and tree/boosting baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


LOGISTIC_PREDICTIONS = {
    "diabetes": ("outputs/tables/mimic_diabetes_prediction_time_test_predictions.csv", "discharge_safe_minimal"),
    "ckd": ("outputs/tables/mimic_ckd_test_predictions.csv", "discharge_lab_minimal"),
    "heart_failure": ("outputs/tables/mimic_heart_failure_test_predictions.csv", "discharge_lab_minimal"),
    "hypertension": ("outputs/tables/mimic_hypertension_test_predictions.csv", "discharge_lab_minimal"),
}
RF_PREDICTIONS = "outputs/tables/random_forest_baseline_predictions.csv"
CALIBRATED_RF_PREDICTIONS = "outputs/tables/calibrated_random_forest_predictions.csv"
GRADIENT_BOOSTING_PREDICTIONS = "outputs/tables/gradient_boosting_baseline_predictions.csv"
CALIBRATED_GB_PREDICTIONS = "outputs/tables/calibrated_gradient_boosting_predictions.csv"


def load_logistic_predictions(project_root: Path) -> pd.DataFrame:
    parts = []
    for cohort, (relative_path, feature_set) in LOGISTIC_PREDICTIONS.items():
        path = project_root / relative_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["feature_set"].eq(feature_set)].copy()
        df["cohort"] = cohort
        df["model"] = "logistic_regression"
        df["split"] = "test"
        parts.append(df[["cohort", "model", "feature_set", "split", "subject_id", "hadm_id", "readmission_30d", "predicted_risk"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_rf_predictions(project_root: Path) -> pd.DataFrame:
    path = project_root / RF_PREDICTIONS
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["split"].eq("test")].copy()
    df["cohort"] = df["study"]
    return df[["cohort", "model", "feature_set", "split", "subject_id", "hadm_id", "readmission_30d", "predicted_risk"]]


def load_calibrated_rf_predictions(project_root: Path) -> pd.DataFrame:
    path = project_root / CALIBRATED_RF_PREDICTIONS
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["split"].eq("test")].copy()
    df["cohort"] = df["study"]
    return df[["cohort", "model", "feature_set", "split", "subject_id", "hadm_id", "readmission_30d", "predicted_risk"]]


def load_gradient_boosting_predictions(project_root: Path) -> pd.DataFrame:
    path = project_root / GRADIENT_BOOSTING_PREDICTIONS
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["split"].eq("test")].copy()
    if df.empty:
        return df
    df["cohort"] = df["study"]
    return df[["cohort", "model", "feature_set", "split", "subject_id", "hadm_id", "readmission_30d", "predicted_risk"]]


def load_calibrated_gradient_boosting_predictions(project_root: Path) -> pd.DataFrame:
    path = project_root / CALIBRATED_GB_PREDICTIONS
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["split"].eq("test")].copy()
    if df.empty:
        return df
    df["cohort"] = df["study"]
    return df[["cohort", "model", "feature_set", "split", "subject_id", "hadm_id", "readmission_30d", "predicted_risk"]]


def calibration_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, model), group in predictions.groupby(["cohort", "model"], sort=False):
        ranked = group.sort_values("predicted_risk").copy()
        ranked["decile"] = pd.qcut(ranked["predicted_risk"].rank(method="first"), 10, labels=False) + 1
        for decile, part in ranked.groupby("decile", sort=True):
            mean_pred = float(part["predicted_risk"].mean())
            obs_rate = float(part["readmission_30d"].mean())
            rows.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "decile": int(decile),
                    "n": int(len(part)),
                    "mean_predicted_risk": mean_pred,
                    "observed_event_rate": obs_rate,
                    "absolute_calibration_error": abs(mean_pred - obs_rate),
                }
            )
    return pd.DataFrame(rows)


def calibration_summary(deciles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, model), group in deciles.groupby(["cohort", "model"], sort=False):
        weighted_error = (group["absolute_calibration_error"] * group["n"]).sum() / group["n"].sum()
        rows.append(
            {
                "cohort": cohort,
                "model": model,
                "mean_absolute_calibration_error": float(weighted_error),
                "max_absolute_calibration_error": float(group["absolute_calibration_error"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration(deciles: pd.DataFrame, path: Path) -> None:
    if deciles.empty:
        return
    cohorts = list(deciles["cohort"].drop_duplicates())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.2 * len(cohorts), 4.6), sharex=True, sharey=True)
    if len(cohorts) == 1:
        axes = [axes]
    colors = {
        "logistic_regression": "#4C78A8",
        "random_forest_sklearn": "#E15759",
        "calibrated_random_forest_platt": "#59A14F",
        "calibrated_random_forest_isotonic": "#F28E2B",
        "gradient_boosting_sklearn_hist": "#B07AA1",
        "calibrated_gradient_boosting_platt": "#9C755F",
        "calibrated_gradient_boosting_isotonic": "#76B7B2",
    }
    labels = {
        "logistic_regression": "Logistic",
        "random_forest_sklearn": "Random Forest",
        "calibrated_random_forest_platt": "RF Platt",
        "calibrated_random_forest_isotonic": "RF Isotonic",
        "gradient_boosting_sklearn_hist": "HistGB",
        "calibrated_gradient_boosting_platt": "HistGB Platt",
        "calibrated_gradient_boosting_isotonic": "HistGB Isotonic",
    }
    for ax, cohort in zip(axes, cohorts):
        part = deciles[deciles["cohort"].eq(cohort)]
        for model, group in part.groupby("model", sort=False):
            ax.plot(
                group["mean_predicted_risk"],
                group["observed_event_rate"],
                marker="o",
                label=labels.get(model, model),
                color=colors.get(model),
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        ax.set_title(cohort)
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        ax.grid(alpha=0.25)
        ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data found."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, report_path: Path) -> None:
    text = f"""# Chronic Disease Model Calibration Report

这个报告按 test set 风险十分位比较 logistic regression、Random Forest、calibrated Random Forest 和 gradient boosting 的预测风险校准情况。

## Calibration Error Summary

{markdown_table(summary)}

## Interpretation

- `mean_absolute_calibration_error` 越小越好，表示十分位平均预测风险更接近实际事件率。
- 这份报告和 Brier score 一起使用，避免只看 AUROC/AUPRC。
- Random Forest 和 gradient boosting 可以提升排序能力，但如果校准更差，后续需要考虑 calibrated classifier、isotonic regression 或 Platt scaling。

## Outputs

- `outputs/tables/chronic_disease_model_calibration_deciles.csv`
- `outputs/tables/chronic_disease_model_calibration_summary.csv`
- `outputs/figures/chronic_disease_model_calibration_deciles.png`
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.project_root / "outputs" / "tables"
    figures = args.project_root / "outputs" / "figures"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    predictions = pd.concat(
        [
            load_logistic_predictions(args.project_root),
            load_rf_predictions(args.project_root),
            load_calibrated_rf_predictions(args.project_root),
            load_gradient_boosting_predictions(args.project_root),
            load_calibrated_gradient_boosting_predictions(args.project_root),
        ],
        ignore_index=True,
    )
    deciles = calibration_deciles(predictions)
    summary = calibration_summary(deciles)
    deciles.to_csv(tables / "chronic_disease_model_calibration_deciles.csv", index=False)
    summary.to_csv(tables / "chronic_disease_model_calibration_summary.csv", index=False)
    plot_calibration(deciles, figures / "chronic_disease_model_calibration_deciles.png")
    write_report(summary, reports / "chronic_disease_model_calibration_report.md")
    print("Model calibration summary complete")
    print(f"prediction_rows={len(predictions)} decile_rows={len(deciles)}")


if __name__ == "__main__":
    main()
