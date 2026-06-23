#!/usr/bin/env python3
"""Summarize feature-group ablations from the existing prediction-time benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


BENCHMARK_TABLE = "outputs/tables/chronic_disease_prediction_time_benchmark.csv"

COHORT_LABELS = {
    "diabetes": "糖尿病",
    "ckd": "CKD",
    "heart_failure": "心衰",
    "hypertension": "高血压",
}

COMPARISONS = {
    "diabetes": [
        ("24h", "24h labs", "admission_safe_minimal", "inhospital_24h_lab_minimal"),
        ("24h", "diabetes meds", "inhospital_24h_lab_minimal", "inhospital_24h_lab_med_minimal"),
        ("24h", "ICU vitals", "inhospital_24h_lab_med_minimal", "inhospital_24h_lab_med_vital_minimal"),
        ("24h", "ICU procedures", "inhospital_24h_lab_med_vital_minimal", "inhospital_24h_lab_med_vital_proc_minimal"),
        ("24h", "broad medications", "inhospital_24h_lab_med_vital_proc_minimal", "inhospital_24h_lab_med_vital_proc_genmed_minimal"),
        ("discharge", "discharge-safe process variables", "admission_safe_minimal", "discharge_safe_minimal"),
        ("discharge", "ICU vitals", "discharge_safe_minimal", "discharge_safe_vital_minimal"),
        ("discharge", "ICU procedures", "discharge_safe_vital_minimal", "discharge_safe_vital_proc_minimal"),
        ("discharge", "broad medications", "discharge_safe_vital_proc_minimal", "discharge_safe_vital_proc_genmed_minimal"),
    ],
    "ckd": [
        ("24h", "24h labs", "admission_safe_minimal", "inhospital_24h_lab_minimal"),
        ("24h", "ICU vitals", "inhospital_24h_lab_minimal", "inhospital_24h_lab_vital_minimal"),
        ("24h", "ICU procedures", "inhospital_24h_lab_vital_minimal", "inhospital_24h_lab_vital_proc_minimal"),
        ("24h", "broad medications", "inhospital_24h_lab_vital_proc_minimal", "inhospital_24h_lab_vital_proc_genmed_minimal"),
        ("discharge", "discharge labs", "admission_safe_minimal", "discharge_lab_minimal"),
        ("discharge", "ICU vitals", "discharge_lab_minimal", "discharge_lab_vital_minimal"),
        ("discharge", "ICU procedures", "discharge_lab_vital_minimal", "discharge_lab_vital_proc_minimal"),
        ("discharge", "broad medications", "discharge_lab_vital_proc_minimal", "discharge_lab_vital_proc_genmed_minimal"),
    ],
    "heart_failure": [
        ("24h", "24h labs", "admission_safe_minimal", "inhospital_24h_lab_minimal"),
        ("24h", "ICU vitals", "inhospital_24h_lab_minimal", "inhospital_24h_lab_vital_minimal"),
        ("24h", "ICU procedures", "inhospital_24h_lab_vital_minimal", "inhospital_24h_lab_vital_proc_minimal"),
        ("24h", "broad medications", "inhospital_24h_lab_vital_proc_minimal", "inhospital_24h_lab_vital_proc_genmed_minimal"),
        ("discharge", "discharge labs", "admission_safe_minimal", "discharge_lab_minimal"),
        ("discharge", "ICU vitals", "discharge_lab_minimal", "discharge_lab_vital_minimal"),
        ("discharge", "ICU procedures", "discharge_lab_vital_minimal", "discharge_lab_vital_proc_minimal"),
        ("discharge", "broad medications", "discharge_lab_vital_proc_minimal", "discharge_lab_vital_proc_genmed_minimal"),
    ],
    "hypertension": [
        ("24h", "24h labs", "admission_safe_minimal", "inhospital_24h_lab_minimal"),
        ("24h", "ICU vitals", "inhospital_24h_lab_minimal", "inhospital_24h_lab_vital_minimal"),
        ("24h", "ICU procedures", "inhospital_24h_lab_vital_minimal", "inhospital_24h_lab_vital_proc_minimal"),
        ("24h", "broad medications", "inhospital_24h_lab_vital_proc_minimal", "inhospital_24h_lab_vital_proc_genmed_minimal"),
        ("discharge", "discharge labs", "admission_safe_minimal", "discharge_lab_minimal"),
        ("discharge", "ICU vitals", "discharge_lab_minimal", "discharge_lab_vital_minimal"),
        ("discharge", "ICU procedures", "discharge_lab_vital_minimal", "discharge_lab_vital_proc_minimal"),
        ("discharge", "broad medications", "discharge_lab_vital_proc_minimal", "discharge_lab_vital_proc_genmed_minimal"),
    ],
}


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def read_benchmark(project_root: Path) -> pd.DataFrame:
    path = project_root / BENCHMARK_TABLE
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark table: {path}")
    return pd.read_csv(path)


def make_ablation_table(benchmark: pd.DataFrame) -> pd.DataFrame:
    tests = benchmark.set_index(["cohort", "feature_set"])
    rows = []
    for cohort, comparisons in COMPARISONS.items():
        for stage, group_added, baseline, augmented in comparisons:
            if (cohort, baseline) not in tests.index or (cohort, augmented) not in tests.index:
                continue
            base = tests.loc[(cohort, baseline)]
            aug = tests.loc[(cohort, augmented)]
            rows.append(
                {
                    "cohort": cohort,
                    "cohort_label": COHORT_LABELS.get(cohort, cohort),
                    "stage": stage,
                    "group_added": group_added,
                    "baseline_feature_set": baseline,
                    "augmented_feature_set": augmented,
                    "baseline_AUROC": float(base["AUROC"]),
                    "augmented_AUROC": float(aug["AUROC"]),
                    "delta_AUROC": float(aug["AUROC"] - base["AUROC"]),
                    "baseline_AUPRC": float(base["AUPRC"]),
                    "augmented_AUPRC": float(aug["AUPRC"]),
                    "delta_AUPRC": float(aug["AUPRC"] - base["AUPRC"]),
                    "baseline_Brier": float(base["Brier_score"]),
                    "augmented_Brier": float(aug["Brier_score"]),
                    "delta_Brier": float(aug["Brier_score"] - base["Brier_score"]),
                }
            )
    return pd.DataFrame(rows)


def make_summary(ablation: pd.DataFrame) -> pd.DataFrame:
    if ablation.empty:
        return ablation
    summary = (
        ablation.groupby(["stage", "group_added"], sort=False)
        .agg(
            comparisons=("cohort", "count"),
            cohorts_improved_AUROC=("delta_AUROC", lambda s: int((s > 0).sum())),
            cohorts_improved_AUPRC=("delta_AUPRC", lambda s: int((s > 0).sum())),
            mean_delta_AUROC=("delta_AUROC", "mean"),
            median_delta_AUROC=("delta_AUROC", "median"),
            mean_delta_AUPRC=("delta_AUPRC", "mean"),
            median_delta_AUPRC=("delta_AUPRC", "median"),
            mean_delta_Brier=("delta_Brier", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["stage", "mean_delta_AUPRC", "mean_delta_AUROC"], ascending=[True, False, False])


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No data available."
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(ablation: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    top_auprc = ablation.sort_values("delta_AUPRC", ascending=False).head(8)
    negative = ablation[(ablation["delta_AUROC"] < 0) | (ablation["delta_AUPRC"] < 0)].sort_values("delta_AUPRC").head(8)
    summary_cols = [
        "stage",
        "group_added",
        "comparisons",
        "cohorts_improved_AUROC",
        "cohorts_improved_AUPRC",
        "mean_delta_AUROC",
        "mean_delta_AUPRC",
        "mean_delta_Brier",
    ]
    detail_cols = [
        "cohort_label",
        "stage",
        "group_added",
        "baseline_feature_set",
        "augmented_feature_set",
        "delta_AUROC",
        "delta_AUPRC",
        "delta_Brier",
    ]
    text = f"""# Feature Group Ablation Summary

这个报告不是重新训练模型，而是把已经完成的 prediction-time benchmark 组织成特征组增量比较。它回答一个很实用的问题：在当前慢病再入院预测任务中，labs、ICU vitals、ICU procedures 和 broad medications 分别带来多少增量。

## Summary By Feature Group

{markdown_table(summary, summary_cols)}

## Largest AUPRC Gains

{markdown_table(top_auprc, detail_cols)}

## Groups With Negative Or Mixed Delta

{markdown_table(negative, detail_cols)}

## Interpretation

- `delta_AUROC` 和 `delta_AUPRC` 大于 0 表示加入该特征组后排序性能提高。
- `delta_Brier` 小于 0 通常表示概率误差变小；如果 AUROC 上升但 Brier 变差，说明模型排序变强但概率可能更不稳。
- Broad medications 在多个队列中带来较稳定 AUPRC 增益，特别是糖尿病和高血压。
- ICU vitals/procedures 的增益更不稳定，提示这些变量可能受 ICU 覆盖率和缺失模式影响，不能简单认为“变量越多越好”。
- 这张表可以作为后续 feature selection 或 grouped ablation 的路线图：优先深挖 broad medications，再检查 procedures/vitals 的覆盖率和缺失机制。
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

    benchmark = read_benchmark(args.project_root)
    ablation = make_ablation_table(benchmark)
    summary = make_summary(ablation)
    ablation.to_csv(tables / "chronic_disease_feature_group_ablation.csv", index=False)
    summary.to_csv(tables / "chronic_disease_feature_group_ablation_summary.csv", index=False)
    write_report(ablation, summary, reports / "chronic_disease_feature_group_ablation_report.md")
    print("Feature group ablation summary complete")
    print(f"comparisons={len(ablation)} groups={len(summary)}")


if __name__ == "__main__":
    main()
