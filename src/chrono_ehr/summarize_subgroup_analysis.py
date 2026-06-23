#!/usr/bin/env python3
"""Summarize subgroup performance for final ChronoEHR models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from summarize_feature_selection import COHORT_LABELS, FINAL_FEATURE_SETS


PREDICTION_FILES = {
    "diabetes": "outputs/tables/mimic_diabetes_prediction_time_test_predictions.csv",
    "ckd": "outputs/tables/mimic_ckd_test_predictions.csv",
    "heart_failure": "outputs/tables/mimic_heart_failure_test_predictions.csv",
    "hypertension": "outputs/tables/mimic_hypertension_test_predictions.csv",
}

COHORT_FILES = {
    "diabetes": "data/processed/mimic_diabetes_readmission_cohort.csv",
    "ckd": "data/processed/mimic_ckd_readmission_cohort.csv",
    "heart_failure": "data/processed/mimic_heart_failure_readmission_cohort.csv",
    "hypertension": "data/processed/mimic_hypertension_readmission_cohort.csv",
}

MIN_SUBGROUP_N = 500


def load_metrics():
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except ImportError as exc:  # pragma: no cover - dependency issue is reported to user.
        raise SystemExit("scikit-learn is required for subgroup AUROC/AUPRC summaries.") from exc
    return roc_auc_score, average_precision_score, brier_score_loss


def add_subgroup_columns(cohort: pd.DataFrame) -> pd.DataFrame:
    out = cohort.copy()
    out["age_group"] = pd.cut(
        out["anchor_age"],
        bins=[0, 49, 64, 79, np.inf],
        labels=["<50", "50-64", "65-79", "80+"],
        right=True,
    ).astype(str)
    out["gender_group"] = out["gender"].fillna("Unknown").astype(str)
    admission = out["admission_type"].fillna("Unknown").astype(str).str.upper()
    out["admission_type_group"] = np.select(
        [
            admission.str.contains("ELECTIVE", regex=False),
            admission.str.contains("EMERGENCY", regex=False),
            admission.str.contains("URGENT", regex=False),
        ],
        ["elective", "emergency", "urgent"],
        default="other",
    )
    prior = out["prior_admissions_count"].fillna(0)
    out["prior_admission_group"] = np.select(
        [prior.eq(0), prior.eq(1), prior.ge(2)],
        ["0", "1", "2+"],
        default="unknown",
    )
    return out


def read_cohort(project_root: Path, cohort: str) -> pd.DataFrame:
    path = project_root / COHORT_FILES[cohort]
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort file: {path}")
    cols = [
        "subject_id",
        "hadm_id",
        "anchor_age",
        "gender",
        "admission_type",
        "prior_admissions_count",
    ]
    return add_subgroup_columns(pd.read_csv(path, usecols=cols))


def read_predictions(project_root: Path, cohort: str) -> pd.DataFrame:
    path = project_root / PREDICTION_FILES[cohort]
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    df["cohort"] = cohort
    df["cohort_label"] = COHORT_LABELS.get(cohort, cohort)
    return df


def safe_auc(y_true: pd.Series, score: pd.Series, roc_auc_score) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_auprc(y_true: pd.Series, score: pd.Series, average_precision_score) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def top_alert_metrics(group: pd.DataFrame, alert_rate: float = 0.10) -> dict[str, float | int]:
    ranked = group.sort_values("predicted_risk", ascending=False).reset_index(drop=True)
    n = len(ranked)
    events = int(ranked["readmission_30d"].sum())
    alerts = max(1, int(round(n * alert_rate)))
    flagged = ranked.iloc[:alerts]
    tp = int(flagged["readmission_30d"].sum())
    return {
        "top10_alerts": alerts,
        "top10_ppv": tp / alerts if alerts else 0.0,
        "top10_recall": tp / events if events else 0.0,
    }


def summarize(project_root: Path) -> pd.DataFrame:
    roc_auc_score, average_precision_score, brier_score_loss = load_metrics()
    subgroup_columns = ["age_group", "gender_group", "admission_type_group", "prior_admission_group"]
    rows = []
    for cohort, stage_map in FINAL_FEATURE_SETS.items():
        predictions = read_predictions(project_root, cohort)
        cohort_df = read_cohort(project_root, cohort)
        merged = predictions.merge(cohort_df, on=["subject_id", "hadm_id"], how="left")
        for prediction_time, feature_set in stage_map.items():
            subset = merged[merged["feature_set"].eq(feature_set)].copy()
            if subset.empty:
                continue
            for subgroup_variable in subgroup_columns:
                for subgroup_value, group in subset.groupby(subgroup_variable, dropna=False):
                    n = len(group)
                    if n < MIN_SUBGROUP_N:
                        continue
                    y = group["readmission_30d"]
                    score = group["predicted_risk"]
                    alerts = top_alert_metrics(group)
                    rows.append(
                        {
                            "cohort": cohort,
                            "cohort_label": COHORT_LABELS.get(cohort, cohort),
                            "prediction_time": prediction_time,
                            "feature_set": feature_set,
                            "subgroup_variable": subgroup_variable,
                            "subgroup_value": str(subgroup_value),
                            "n": n,
                            "events": int(y.sum()),
                            "event_rate": float(y.mean()),
                            "AUROC": safe_auc(y, score, roc_auc_score),
                            "AUPRC": safe_auprc(y, score, average_precision_score),
                            "Brier_score": float(brier_score_loss(y, score)),
                            "mean_predicted_risk": float(score.mean()),
                            **alerts,
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
        text = "# Subgroup Performance Summary\n\nNo subgroup rows were generated.\n"
        output.write_text(text, encoding="utf-8")
        return

    by_variable = (
        summary.groupby("subgroup_variable", dropna=False)
        .agg(
            rows=("subgroup_value", "count"),
            min_n=("n", "min"),
            mean_event_rate=("event_rate", "mean"),
            mean_AUROC=("AUROC", "mean"),
            mean_AUPRC=("AUPRC", "mean"),
            mean_top10_ppv=("top10_ppv", "mean"),
        )
        .reset_index()
    )
    compact = summary[
        [
            "cohort_label",
            "prediction_time",
            "subgroup_variable",
            "subgroup_value",
            "n",
            "events",
            "event_rate",
            "AUROC",
            "AUPRC",
            "Brier_score",
            "top10_ppv",
            "top10_recall",
        ]
    ].copy()
    text = f"""# Subgroup Performance Summary

这个报告按年龄组、性别、入院类型和既往住院次数，对最终 24h 和 discharge logistic models 做分层性能汇总。它用于发现模型在不同基础人群中的表现差异，不能解释为因果关系，也不是临床诊疗建议。

当前只保留样本量不少于 {MIN_SUBGROUP_N} 的 subgroup，避免很小分层导致 AUROC/AUPRC 不稳定。

## Summary By Subgroup Variable

{markdown_table(by_variable, list(by_variable.columns))}

## Detailed Subgroup Table

{markdown_table(compact, list(compact.columns))}

## Interpretation Notes

- Subgroup AUROC/AUPRC 只描述模型在该分层内的排序表现，不代表某个变量导致再入院。
- 如果 subgroup 的事件率不同，AUPRC 和 PPV 会自然受到基础事件率影响，因此需要和 event_rate 一起解释。
- 这一步适合作为补充材料或下一步公平性/异质性分析的入口。
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
    summary.to_csv(tables / "chronic_disease_subgroup_performance.csv", index=False)
    write_report(summary, reports / "chronic_disease_subgroup_performance_report.md")
    print(f"Subgroup rows: {len(summary)}")
    print(f"Wrote {reports / 'chronic_disease_subgroup_performance_report.md'}")


if __name__ == "__main__":
    main()
