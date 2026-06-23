#!/usr/bin/env python3
"""Build a MIMIC-IV hypertension 30-day readmission demo cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diagnosis_cohort_builder import HYPERTENSION_SPEC, build_diagnosis_readmission_cohort
from mimic_diabetes_cohort import (
    DEFAULT_PROJECT,
    DEFAULT_ROOT,
    check_split_overlap,
    metric_count_pct,
    metric_mean_sd,
    metric_median_iqr,
    write_summary_csv,
)


HTN_FEATURES = HYPERTENSION_SPEC.model_features


def summarize_missingness(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in HTN_FEATURES:
        missing = int(cohort[col].isna().sum())
        rows.append(
            {
                "variable": col,
                "missing_count": missing,
                "missing_percent": missing / len(cohort) if len(cohort) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_splits(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in cohort.groupby("split", sort=True):
        rows.append(
            {
                "split": split,
                "admissions": int(len(part)),
                "subjects": int(part["subject_id"].nunique()),
                "readmission_30d_count": int(part["readmission_30d"].sum()),
                "readmission_30d_rate": float(part["readmission_30d"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_table1(cohort: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "overall": cohort,
        "no_readmission_30d": cohort[cohort["readmission_30d"] == 0],
        "readmission_30d": cohort[cohort["readmission_30d"] == 1],
    }
    definitions = [
        ("admissions", lambda df: str(len(df))),
        ("subjects", lambda df: str(df["subject_id"].nunique())),
        ("age_mean_sd", lambda df: metric_mean_sd(df["anchor_age"])),
        ("female", lambda df: metric_count_pct(df["gender"].eq("F"))),
        ("current_hypertension_admission", lambda df: metric_count_pct(df["current_hypertension_admission"].eq(1))),
        ("prior_hypertension_diagnosis", lambda df: metric_count_pct(df["prior_hypertension_diagnosis"].eq(1))),
        ("length_of_stay_days_median_iqr", lambda df: metric_median_iqr(df["length_of_stay_days"])),
        ("prior_admissions_count_median_iqr", lambda df: metric_median_iqr(df["prior_admissions_count"])),
    ]
    rows = []
    for variable, func in definitions:
        row = {"variable": variable}
        for group_name, df in groups.items():
            row[group_name] = func(df)
        rows.append(row)
    return pd.DataFrame(rows)


def write_reports(summary: dict[str, int | float | str], split_overlap: dict[str, int], reports_dir: Path) -> None:
    audit = f"""# MIMIC Hypertension Cohort Audit Report

## Status

第一版 hypertension cohort 已生成。当前阶段是 `COHORT_READY`，可进入最小 prediction-time baseline。

## Key Counts

- Raw hypertension admissions: {summary["raw_hypertension_admissions"]}
- Raw hypertension subjects: {summary["raw_hypertension_subjects"]}
- Final index admissions: {summary["final_index_admissions"]}
- Final subjects: {summary["final_subjects"]}
- 30-day readmission count: {summary["readmission_30d_count"]}
- 30-day readmission rate: {summary["readmission_30d_rate"]:.2%}
- Prior-known hypertension rate: {summary["prior_hypertension_diagnosis_rate"]:.2%}
- Current-admission hypertension diagnosis rate: {summary["current_hypertension_admission_rate"]:.2%}

## Leakage Notes

- `current_hypertension_admission` 可用于回顾性队列构建和出院时分析。
- 入院时预测应优先使用 `prior_hypertension_diagnosis`，避免把当前住院出院诊断倒拿到入院时。
- `next_admittime`, `next_hadm_id`, `days_to_next_admission` 必须禁止进入模型。

## Patient Split Check

- train/validation overlap: {split_overlap["train_validation_overlap"]}
- train/test overlap: {split_overlap["train_test_overlap"]}
- validation/test overlap: {split_overlap["validation_test_overlap"]}
"""
    (reports_dir / "mimic_hypertension_leakage_audit_report.md").write_text(audit, encoding="utf-8")

    run_summary = f"""# MIMIC Hypertension Cohort Run Summary

- Final index admissions: {summary["final_index_admissions"]}
- Final subjects: {summary["final_subjects"]}
- 30-day readmission: {summary["readmission_30d_count"]} ({summary["readmission_30d_rate"]:.2%})
- Prior-only hypertension index admissions before exclusions: {summary["prior_only_hypertension_index_admissions"]}
- Current hypertension index admissions before exclusions: {summary["current_hypertension_index_admissions"]}
"""
    (reports_dir / "mimic_hypertension_run_summary.md").write_text(run_summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.project_root / "data" / "processed"
    tables_dir = args.project_root / "outputs" / "tables"
    reports_dir = args.project_root / "outputs" / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cohort, summary = build_diagnosis_readmission_cohort(args.mimic_root, HYPERTENSION_SPEC)
    split_overlap = check_split_overlap(cohort)
    cohort.to_csv(processed_dir / "mimic_hypertension_readmission_cohort.csv", index=False)
    write_summary_csv(summary, tables_dir / "mimic_hypertension_cohort_summary.csv")
    summarize_splits(cohort).to_csv(tables_dir / "mimic_hypertension_split_summary.csv", index=False)
    summarize_missingness(cohort).to_csv(tables_dir / "mimic_hypertension_feature_missingness.csv", index=False)
    summarize_table1(cohort).to_csv(tables_dir / "mimic_hypertension_table1_basic.csv", index=False)
    write_reports(summary, split_overlap, reports_dir)

    print("MIMIC hypertension cohort built")
    print(f"final_index_admissions={summary['final_index_admissions']}")
    print(f"final_subjects={summary['final_subjects']}")
    print(f"readmission_30d_rate={summary['readmission_30d_rate']:.4f}")


if __name__ == "__main__":
    main()
