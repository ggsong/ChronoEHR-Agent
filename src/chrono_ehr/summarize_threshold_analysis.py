#!/usr/bin/env python3
"""Summarize alert-burden threshold performance for final ChronoEHR models."""

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

ALERT_RATES = [0.05, 0.10, 0.20]


def read_predictions(project_root: Path, cohort: str) -> pd.DataFrame:
    path = project_root / PREDICTION_FILES[cohort]
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    df["cohort"] = cohort
    df["cohort_label"] = COHORT_LABELS.get(cohort, cohort)
    return df


def alert_metrics(df: pd.DataFrame, alert_rate: float) -> dict[str, float | int]:
    if df.empty:
        raise ValueError("Cannot compute threshold metrics for an empty dataframe.")
    ranked = df.sort_values("predicted_risk", ascending=False).reset_index(drop=True)
    n = len(ranked)
    events = int(ranked["readmission_30d"].sum())
    alerts = max(1, int(round(n * alert_rate)))
    flagged = ranked.iloc[:alerts]
    threshold = float(flagged["predicted_risk"].min())
    tp = int(flagged["readmission_30d"].sum())
    fp = alerts - tp
    fn = events - tp
    tn = n - events - fp
    event_rate = events / n if n else 0.0
    ppv = tp / alerts if alerts else 0.0
    recall = tp / events if events else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    lift = ppv / event_rate if event_rate else 0.0
    return {
        "n": n,
        "events": events,
        "event_rate": event_rate,
        "alert_rate": alert_rate,
        "alerts": alerts,
        "risk_threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "ppv": ppv,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "lift_vs_event_rate": lift,
    }


def summarize(project_root: Path) -> pd.DataFrame:
    rows = []
    for cohort, stage_map in FINAL_FEATURE_SETS.items():
        predictions = read_predictions(project_root, cohort)
        for prediction_time, feature_set in stage_map.items():
            subset = predictions[predictions["feature_set"].eq(feature_set)].copy()
            if subset.empty:
                continue
            for alert_rate in ALERT_RATES:
                metrics = alert_metrics(subset, alert_rate)
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
                if 0 <= value <= 1:
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
        text = "# Threshold And Alert-Burden Analysis\n\nNo threshold rows were generated.\n"
        output.write_text(text, encoding="utf-8")
        return

    top10 = summary[summary["alert_rate"].eq(0.10)].copy()
    top10_brief = top10[
        [
            "cohort_label",
            "prediction_time",
            "n",
            "events",
            "event_rate",
            "alerts",
            "ppv",
            "recall",
            "specificity",
            "lift_vs_event_rate",
        ]
    ]
    full = summary[
        [
            "cohort_label",
            "prediction_time",
            "alert_rate",
            "alerts",
            "risk_threshold",
            "ppv",
            "recall",
            "specificity",
            "npv",
            "lift_vs_event_rate",
        ]
    ]
    mean_ppv_top10 = top10["ppv"].mean()
    mean_recall_top10 = top10["recall"].mean()
    text = f"""# Threshold And Alert-Burden Analysis

这个报告把最终 24h 和 discharge logistic models 转换成固定 alert burden 场景，用于补充 AUROC/AUPRC。它回答的是：如果只标记风险最高的 5%、10% 或 20% 住院记录，能捕获多少 30 天再入院事件，PPV 大约是多少。

本报告只用于 EHR 数据研究和模型评估，不提供医学诊疗建议。

## Top 10% Alert Burden Summary

Top 10% alert burden 的平均 PPV 为 {mean_ppv_top10:.3f}，平均 recall 为 {mean_recall_top10:.3f}。这类结果适合放在补充材料中，帮助读者理解模型在固定工作量下的表现。

{markdown_table(top10_brief, list(top10_brief.columns))}

## Full Alert-Burden Table

{markdown_table(full, list(full.columns))}

## Interpretation Notes

- PPV 表示被模型标记为高风险的人中，实际发生 30 天再入院的比例。
- Recall/sensitivity 表示所有实际再入院事件中，有多少被 top-risk alert 捕获。
- Lift 表示 PPV 相对于该队列基础事件率提升了多少倍。
- 固定 alert burden 是研究报告中的模型行为描述，不是临床处置建议。
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
    summary.to_csv(tables / "chronic_disease_threshold_analysis.csv", index=False)
    write_report(summary, reports / "chronic_disease_threshold_analysis_report.md")
    print(f"Threshold rows: {len(summary)}")
    print(f"Wrote {reports / 'chronic_disease_threshold_analysis_report.md'}")


if __name__ == "__main__":
    main()
