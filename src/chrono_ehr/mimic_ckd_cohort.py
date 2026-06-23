#!/usr/bin/env python3
"""Build a MIMIC-IV CKD 30-day readmission demo cohort."""

from __future__ import annotations

import os
import argparse
import csv
import gzip
from pathlib import Path

import pandas as pd

from mimic_diagnosis_cohort_builder import CKD_SPEC, build_diagnosis_readmission_cohort
from mimic_diabetes_cohort import (
    DISCHARGE_SAFE_FEATURES,
    add_timeline_columns,
    assign_patient_split,
    check_split_overlap,
    metric_count_pct,
    metric_mean_sd,
    metric_median_iqr,
    read_admissions,
    read_patients,
    write_summary_csv,
)


DEFAULT_ROOT = Path(os.environ.get("MIMIC_IV_ROOT", "~/mimic-iv-3.1")).expanduser()
DEFAULT_PROJECT = Path(__file__).resolve().parents[2]

CKD_ICD9_PREFIXES = ("585",)
CKD_ICD10_PREFIXES = ("N18",)

CKD_FEATURES = [
    *DISCHARGE_SAFE_FEATURES,
    "current_ckd_admission",
    "prior_ckd_diagnosis",
    "known_ckd_before_or_current_admission",
]

FORBIDDEN_FEATURES = [
    "readmission_30d",
    "next_admittime",
    "next_hadm_id",
    "days_to_next_admission",
    "deathtime",
    "dod",
    "hospital_expire_flag",
    "postdischarge_death_within_30d",
]


def norm_code(code: str) -> str:
    return str(code).strip().upper().replace(".", "")


def is_ckd_code(code: str, version: str) -> bool:
    code = norm_code(code)
    version = str(version).strip()
    if version == "9":
        return code.startswith(CKD_ICD9_PREFIXES)
    if version == "10":
        return code.startswith(CKD_ICD10_PREFIXES)
    return False


def collect_ckd_ids(diagnoses_path: Path) -> tuple[set[int], set[int]]:
    ckd_hadm_ids: set[int] = set()
    ckd_subject_ids: set[int] = set()
    with gzip.open(diagnoses_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if is_ckd_code(row["icd_code"], row["icd_version"]):
                ckd_hadm_ids.add(int(row["hadm_id"]))
                ckd_subject_ids.add(int(row["subject_id"]))
    return ckd_hadm_ids, ckd_subject_ids


def add_ckd_history(timeline: pd.DataFrame) -> pd.DataFrame:
    df = timeline.sort_values(["subject_id", "admittime", "hadm_id"]).copy()
    previous_ckd_count = df.groupby("subject_id", sort=False)["current_ckd_admission"].cumsum() - df["current_ckd_admission"].astype(int)
    df["prior_ckd_diagnosis"] = previous_ckd_count.gt(0)
    df["known_ckd_before_or_current_admission"] = df["prior_ckd_diagnosis"] | df["current_ckd_admission"]
    return df


def build_cohort(mimic_root: Path) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    return build_diagnosis_readmission_cohort(mimic_root, CKD_SPEC)

    hosp = mimic_root / "hosp"
    admissions = read_admissions(hosp / "admissions.csv.gz")
    patients = read_patients(hosp / "patients.csv.gz")
    ckd_hadm_ids, ckd_subject_ids = collect_ckd_ids(hosp / "diagnoses_icd.csv.gz")

    timeline = add_timeline_columns(admissions)
    timeline = timeline.merge(patients, on="subject_id", how="left")
    timeline["current_ckd_admission"] = timeline["hadm_id"].isin(ckd_hadm_ids)
    timeline = add_ckd_history(timeline)

    timeline["adult"] = timeline["anchor_age"] >= 18
    timeline["valid_times"] = timeline["admittime"].notna() & timeline["dischtime"].notna()
    timeline["valid_los"] = timeline["length_of_stay_days"] >= 0
    timeline["in_hospital_death"] = (
        timeline["hospital_expire_flag"].fillna(0).astype(int).eq(1)
        | timeline["deathtime"].notna()
    )
    timeline["postdischarge_death_within_30d"] = (
        timeline["dod"].notna()
        & timeline["dischtime"].notna()
        & (timeline["dod"] >= timeline["dischtime"])
        & (timeline["dod"] <= timeline["dischtime"] + pd.Timedelta(days=30))
    )
    timeline["exclude_early_death_no_readmit"] = (
        timeline["postdischarge_death_within_30d"] & ~timeline["readmission_30d"]
    )

    base_mask = (
        timeline["known_ckd_before_or_current_admission"]
        & timeline["adult"]
        & timeline["valid_times"]
        & timeline["valid_los"]
    )
    cohort = timeline[
        base_mask
        & ~timeline["in_hospital_death"]
        & ~timeline["exclude_early_death_no_readmit"]
    ].copy()

    cohort["readmission_30d"] = cohort["readmission_30d"].astype(int)
    cohort["split"] = cohort["subject_id"].apply(assign_patient_split)
    for col in ["current_ckd_admission", "prior_ckd_diagnosis", "known_ckd_before_or_current_admission"]:
        cohort[col] = cohort[col].astype(int)

    output_columns = [
        "subject_id",
        "hadm_id",
        "split",
        "admittime",
        "dischtime",
        "readmission_30d",
        "days_to_next_admission",
        "next_hadm_id",
        "next_admittime",
        "next_admission_type",
        "postdischarge_death_within_30d",
        *CKD_FEATURES,
    ]
    cohort = cohort[output_columns].sort_values(["subject_id", "admittime", "hadm_id"])

    summary = {
        "mimic_root": str(mimic_root),
        "total_admissions": int(len(admissions)),
        "total_admission_subjects": int(admissions["subject_id"].nunique()),
        "total_patients_table": int(len(patients)),
        "raw_ckd_admissions": int(len(ckd_hadm_ids)),
        "raw_ckd_subjects": int(len(ckd_subject_ids)),
        "adult_ckd_valid_time_admissions": int(base_mask.sum()),
        "current_ckd_index_admissions": int((base_mask & timeline["current_ckd_admission"]).sum()),
        "prior_only_ckd_index_admissions": int((base_mask & ~timeline["current_ckd_admission"] & timeline["prior_ckd_diagnosis"]).sum()),
        "excluded_in_hospital_death": int((base_mask & timeline["in_hospital_death"]).sum()),
        "excluded_postdischarge_death_30d_no_readmission": int(
            (base_mask & ~timeline["in_hospital_death"] & timeline["exclude_early_death_no_readmit"]).sum()
        ),
        "final_index_admissions": int(len(cohort)),
        "final_subjects": int(cohort["subject_id"].nunique()),
        "readmission_30d_count": int(cohort["readmission_30d"].sum()),
        "readmission_30d_rate": float(cohort["readmission_30d"].mean()),
        "prior_ckd_diagnosis_rate": float(cohort["prior_ckd_diagnosis"].mean()),
        "current_ckd_admission_rate": float(cohort["current_ckd_admission"].mean()),
    }
    return cohort, summary


def summarize_missingness(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CKD_FEATURES:
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
        ("current_ckd_admission", lambda df: metric_count_pct(df["current_ckd_admission"].eq(1))),
        ("prior_ckd_diagnosis", lambda df: metric_count_pct(df["prior_ckd_diagnosis"].eq(1))),
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


def write_reports(cohort: pd.DataFrame, summary: dict[str, int | float | str], split_overlap: dict[str, int], reports_dir: Path) -> None:
    audit = f"""# MIMIC CKD Cohort Audit Report

## Status

第一版 CKD cohort 已生成。当前阶段是 `COHORT_READY`，可进入 CKD 24h/discharge lab feature extraction。

## Key Counts

- Raw CKD admissions: {summary["raw_ckd_admissions"]}
- Raw CKD subjects: {summary["raw_ckd_subjects"]}
- Final index admissions: {summary["final_index_admissions"]}
- Final subjects: {summary["final_subjects"]}
- 30-day readmission count: {summary["readmission_30d_count"]}
- 30-day readmission rate: {summary["readmission_30d_rate"]:.2%}
- Prior-known CKD rate: {summary["prior_ckd_diagnosis_rate"]:.2%}
- Current-admission CKD diagnosis rate: {summary["current_ckd_admission_rate"]:.2%}

## Leakage Notes

- `current_ckd_admission` 可用于出院时队列定义和出院时分析。
- 入院时预测应优先使用 `prior_ckd_diagnosis`，避免把当前住院出院诊断倒拿到入院时。
- `next_admittime`, `next_hadm_id`, `days_to_next_admission` 必须禁止进入模型。

## Patient Split Check

- train/validation overlap: {split_overlap["train_validation_overlap"]}
- train/test overlap: {split_overlap["train_test_overlap"]}
- validation/test overlap: {split_overlap["validation_test_overlap"]}
"""
    (reports_dir / "mimic_ckd_leakage_audit_report.md").write_text(audit, encoding="utf-8")

    run_summary = f"""# MIMIC CKD Cohort Run Summary

- Final index admissions: {summary["final_index_admissions"]}
- Final subjects: {summary["final_subjects"]}
- 30-day readmission: {summary["readmission_30d_count"]} ({summary["readmission_30d_rate"]:.2%})
- Prior-only CKD index admissions before exclusions: {summary["prior_only_ckd_index_admissions"]}
- Current CKD index admissions before exclusions: {summary["current_ckd_index_admissions"]}

## Outputs

- `data/processed/mimic_ckd_readmission_cohort.csv`
- `outputs/tables/mimic_ckd_cohort_summary.csv`
- `outputs/tables/mimic_ckd_split_summary.csv`
- `outputs/tables/mimic_ckd_feature_missingness.csv`
- `outputs/tables/mimic_ckd_table1_basic.csv`
- `outputs/reports/mimic_ckd_leakage_audit_report.md`
"""
    (reports_dir / "mimic_ckd_run_summary.md").write_text(run_summary, encoding="utf-8")


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

    cohort, summary = build_cohort(args.mimic_root)
    split_overlap = check_split_overlap(cohort)
    cohort.to_csv(processed_dir / "mimic_ckd_readmission_cohort.csv", index=False)
    write_summary_csv(summary, tables_dir / "mimic_ckd_cohort_summary.csv")
    summarize_splits(cohort).to_csv(tables_dir / "mimic_ckd_split_summary.csv", index=False)
    summarize_missingness(cohort).to_csv(tables_dir / "mimic_ckd_feature_missingness.csv", index=False)
    summarize_table1(cohort).to_csv(tables_dir / "mimic_ckd_table1_basic.csv", index=False)
    write_reports(cohort, summary, split_overlap, reports_dir)

    print("MIMIC CKD cohort built")
    print(f"final_index_admissions={summary['final_index_admissions']}")
    print(f"final_subjects={summary['final_subjects']}")
    print(f"readmission_30d_rate={summary['readmission_30d_rate']:.4f}")


if __name__ == "__main__":
    main()
