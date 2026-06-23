#!/usr/bin/env python3
"""Create calibration summaries for selected feature set logistic models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from summarize_model_calibration import markdown_table


PREDICTIONS = "outputs/tables/chronic_disease_selected_feature_set_predictions.csv"
COMPARISON = "outputs/tables/chronic_disease_selected_feature_set_comparison.csv"


def load_predictions(project_root: Path) -> pd.DataFrame:
    path = project_root / PREDICTIONS
    if not path.exists():
        raise FileNotFoundError(f"Missing selected feature predictions: {path}. Run --selected-feature-sets first.")
    df = pd.read_csv(path)
    df["model"] = "selected_logistic_regression"
    df["split"] = "test"
    return df[["cohort", "model", "feature_set", "split", "subject_id", "hadm_id", "readmission_30d", "predicted_risk"]]


def calibration_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["cohort", "model", "feature_set"]
    for (cohort, model, feature_set), group in predictions.groupby(keys, sort=False):
        ranked = group.sort_values("predicted_risk").copy()
        ranked["decile"] = pd.qcut(ranked["predicted_risk"].rank(method="first"), 10, labels=False) + 1
        for decile, part in ranked.groupby("decile", sort=True):
            mean_pred = float(part["predicted_risk"].mean())
            obs_rate = float(part["readmission_30d"].mean())
            rows.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "feature_set": feature_set,
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
    keys = ["cohort", "model", "feature_set"]
    for (cohort, model, feature_set), group in deciles.groupby(keys, sort=False):
        weighted_error = (group["absolute_calibration_error"] * group["n"]).sum() / group["n"].sum()
        rows.append(
            {
                "cohort": cohort,
                "model": model,
                "feature_set": feature_set,
                "mean_absolute_calibration_error": float(weighted_error),
                "max_absolute_calibration_error": float(group["absolute_calibration_error"].max()),
            }
        )
    return pd.DataFrame(rows)


def make_supplementary_table(project_root: Path, calibration: pd.DataFrame) -> pd.DataFrame:
    comparison_path = project_root / COMPARISON
    if not comparison_path.exists():
        return calibration
    comparison = pd.read_csv(comparison_path)
    merged = comparison.merge(
        calibration,
        left_on=["cohort", "selected_feature_set"],
        right_on=["cohort", "feature_set"],
        how="left",
    )
    merged = merged[merged["model"].eq("selected_logistic_regression")].copy()
    return merged[
        [
            "cohort",
            "cohort_label",
            "prediction_time",
            "selected_features",
            "full_AUROC",
            "selected_AUROC",
            "delta_AUROC",
            "full_AUPRC",
            "selected_AUPRC",
            "delta_AUPRC",
            "full_Brier",
            "selected_Brier",
            "delta_Brier",
            "mean_absolute_calibration_error",
            "max_absolute_calibration_error",
        ]
    ]


def write_report(summary: pd.DataFrame, supplementary: pd.DataFrame, output: Path) -> None:
    text = f"""# Selected Feature Set Calibration Summary

这个报告只针对 selected feature set logistic models。它回答的问题是：精简后的 selected models 不只 AUROC/AUPRC 接近 full models，概率校准是否也还能接受？

## Calibration Summary

{markdown_table(summary)}

## Supplementary Table

{markdown_table(supplementary)}

## Interpretation

- `mean_absolute_calibration_error` 越小，说明十分位平均预测风险越接近实际再入院率。
- selected models 的作用是作为更小、更可解释的敏感性分析；论文主结果仍建议保留 full model 和 selected model 两套结果。
- 这仍然是 EHR 研究建模结果，不是临床诊疗建议。
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

    predictions = load_predictions(args.project_root)
    deciles = calibration_deciles(predictions)
    summary = calibration_summary(deciles)
    supplementary = make_supplementary_table(args.project_root, summary)

    deciles.to_csv(tables / "chronic_disease_selected_feature_set_calibration_deciles.csv", index=False)
    summary.to_csv(tables / "chronic_disease_selected_feature_set_calibration_summary.csv", index=False)
    supplementary.to_csv(tables / "chronic_disease_selected_feature_set_supplementary_table.csv", index=False)
    write_report(summary, supplementary, reports / "chronic_disease_selected_feature_set_calibration_report.md")
    print("Selected feature set calibration summary complete")
    print(f"prediction_rows={len(predictions)} decile_rows={len(deciles)}")


if __name__ == "__main__":
    main()
