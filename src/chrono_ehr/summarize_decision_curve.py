#!/usr/bin/env python3
"""Summarize decision-curve net benefit for final ChronoEHR models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from summarize_feature_selection import COHORT_LABELS, FINAL_FEATURE_SETS


PREDICTION_FILES = {
    "diabetes": "outputs/tables/mimic_diabetes_prediction_time_test_predictions.csv",
    "ckd": "outputs/tables/mimic_ckd_test_predictions.csv",
    "heart_failure": "outputs/tables/mimic_heart_failure_test_predictions.csv",
    "hypertension": "outputs/tables/mimic_hypertension_test_predictions.csv",
}

THRESHOLD_PROBABILITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def read_predictions(project_root: Path, cohort: str) -> pd.DataFrame:
    path = project_root / PREDICTION_FILES[cohort]
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    df["cohort"] = cohort
    df["cohort_label"] = COHORT_LABELS.get(cohort, cohort)
    return df


def decision_curve_metrics(df: pd.DataFrame, threshold_probability: float) -> dict[str, float | int | str]:
    if df.empty:
        raise ValueError("Cannot compute decision-curve metrics for an empty dataframe.")
    if not 0 < threshold_probability < 1:
        raise ValueError("threshold_probability must be between 0 and 1.")

    n = len(df)
    events = int(df["readmission_30d"].sum())
    non_events = n - events
    event_rate = events / n if n else 0.0
    weight = threshold_probability / (1 - threshold_probability)
    flagged = df[df["predicted_risk"].ge(threshold_probability)]
    alerts = len(flagged)
    tp = int(flagged["readmission_30d"].sum())
    fp = alerts - tp
    model_net_benefit = (tp / n) - (fp / n) * weight
    treat_all_net_benefit = event_rate - (non_events / n) * weight
    treat_none_net_benefit = 0.0
    best_reference = max(treat_all_net_benefit, treat_none_net_benefit)
    net_benefit_advantage = model_net_benefit - best_reference
    standardized_net_benefit = model_net_benefit / event_rate if event_rate else 0.0
    ppv = tp / alerts if alerts else 0.0
    recall = tp / events if events else 0.0
    preferred_strategy = "model" if net_benefit_advantage > 0 else ("treat_all" if treat_all_net_benefit > 0 else "treat_none")
    return {
        "n": n,
        "events": events,
        "event_rate": event_rate,
        "threshold_probability": threshold_probability,
        "alerts": alerts,
        "alert_rate": alerts / n if n else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "ppv": ppv,
        "recall": recall,
        "model_net_benefit": model_net_benefit,
        "treat_all_net_benefit": treat_all_net_benefit,
        "treat_none_net_benefit": treat_none_net_benefit,
        "net_benefit_advantage": net_benefit_advantage,
        "standardized_net_benefit": standardized_net_benefit,
        "preferred_strategy": preferred_strategy,
    }


def summarize(project_root: Path) -> pd.DataFrame:
    rows = []
    for cohort, stage_map in FINAL_FEATURE_SETS.items():
        predictions = read_predictions(project_root, cohort)
        for prediction_time, feature_set in stage_map.items():
            subset = predictions[predictions["feature_set"].eq(feature_set)].copy()
            if subset.empty:
                continue
            for threshold_probability in THRESHOLD_PROBABILITIES:
                metrics = decision_curve_metrics(subset, threshold_probability)
                rows.append(
                    {
                        "cohort": cohort,
                        "cohort_label": COHORT_LABELS.get(cohort, cohort),
                        "prediction_time": prediction_time,
                        "feature_set": feature_set,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No data available."
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                if -1 <= value <= 1:
                    values.append(f"{value:.3f}")
                else:
                    values.append(f"{value:.2f}")
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, output: Path) -> None:
    if summary.empty:
        text = "# Decision-Curve Net Benefit Analysis\n\nNo decision-curve rows were generated.\n"
        output.write_text(text, encoding="utf-8")
        return

    threshold20 = summary[summary["threshold_probability"].eq(0.20)].copy()
    threshold20_brief = threshold20[
        [
            "cohort_label",
            "prediction_time",
            "n",
            "events",
            "event_rate",
            "alerts",
            "alert_rate",
            "ppv",
            "recall",
            "model_net_benefit",
            "treat_all_net_benefit",
            "net_benefit_advantage",
            "preferred_strategy",
        ]
    ]
    by_model = (
        summary.groupby(["cohort_label", "prediction_time"], dropna=False)
        .agg(
            thresholds=("threshold_probability", "nunique"),
            model_preferred_thresholds=("preferred_strategy", lambda values: int((pd.Series(values) == "model").sum())),
            mean_net_benefit_advantage=("net_benefit_advantage", "mean"),
            max_net_benefit_advantage=("net_benefit_advantage", "max"),
            mean_standardized_net_benefit=("standardized_net_benefit", "mean"),
        )
        .reset_index()
    )
    mean_advantage_20 = threshold20["net_benefit_advantage"].mean()
    model_preferred_20 = int((threshold20["preferred_strategy"] == "model").sum())
    total_20 = len(threshold20)
    text = f"""# Decision-Curve Net Benefit Analysis

这个报告把最终 24h 和 discharge logistic models 转换成 decision-curve net benefit 指标，用于补充 AUROC/AUPRC、校准和固定 alert burden。它回答的是：在给定风险阈值下，模型标记高风险患者相比“全部标记”或“全部不标记”是否带来更高净收益。

本报告只用于 EHR 数据研究和模型评估，不提供医学诊疗建议。这里的 threshold probability 是研究分析阈值，不是临床处置阈值。

## Threshold Probability 0.20 Summary

在 threshold probability = 0.20 时，{model_preferred_20}/{total_20} 个最终模型的 net benefit 高于 treat-all 和 treat-none 两个参照策略；平均 net benefit advantage 为 {mean_advantage_20:.4f}。

{markdown_table(threshold20_brief, list(threshold20_brief.columns))}

## Across-Threshold Summary

{markdown_table(by_model, list(by_model.columns))}

## Interpretation Notes

- Model net benefit = true positives / n - false positives / n * threshold odds。
- Treat-all 表示把所有住院都标记为高风险；treat-none 表示不标记任何住院。
- Net benefit advantage 表示模型相对两个参照策略中较好的一个，多出来的净收益。
- 如果 preferred_strategy 不是 model，说明在该阈值下模型没有优于简单参照策略；这对论文很重要，因为它避免只报告 AUROC。
- Decision-curve analysis 是研究评估工具，不是临床决策建议。
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
    summary = summarize(args.project_root)
    summary.to_csv(tables / "chronic_disease_decision_curve.csv", index=False)
    write_report(summary, reports / "chronic_disease_decision_curve_report.md")
    print(f"Decision-curve rows: {len(summary)}")
    print(f"Wrote {reports / 'chronic_disease_decision_curve_report.md'}")


if __name__ == "__main__":
    main()
