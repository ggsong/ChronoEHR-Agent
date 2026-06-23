#!/usr/bin/env python3
"""Build a MIMIC-IV diabetes 30-day readmission demo cohort."""

from __future__ import annotations

import os
import argparse
import csv
import gzip
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from study_config_loader import load_cohort_code_rules


DEFAULT_ROOT = Path(os.environ.get("MIMIC_IV_ROOT", "~/mimic-iv-3.1")).expanduser()
DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = DEFAULT_PROJECT / "configs" / "diabetes_mimic_readmission.yaml"

DIABETES_ICD9_PREFIXES = ("250",)
DIABETES_ICD10_PREFIXES = ("E08", "E09", "E10", "E11", "E12", "E13")

DISCHARGE_SAFE_FEATURES = [
    "anchor_age",
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "ed_los_hours",
    "length_of_stay_days",
    "prior_admissions_count",
    "days_since_prior_discharge",
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


def diabetes_code_prefixes(config_path: Path = DEFAULT_CONFIG) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return load_cohort_code_rules(
        config_path=config_path,
        rule_key="diabetes_code_rules",
        fallback_icd9=DIABETES_ICD9_PREFIXES,
        fallback_icd10=DIABETES_ICD10_PREFIXES,
    )


def is_diabetes_code(
    code: str,
    version: str,
    icd9_prefixes: tuple[str, ...] = DIABETES_ICD9_PREFIXES,
    icd10_prefixes: tuple[str, ...] = DIABETES_ICD10_PREFIXES,
) -> bool:
    code = norm_code(code)
    version = str(version).strip()
    if version == "9":
        return code.startswith(icd9_prefixes)
    if version == "10":
        return code.startswith(icd10_prefixes)
    return False


def collect_diabetes_ids(
    diagnoses_path: Path,
    icd9_prefixes: tuple[str, ...] = DIABETES_ICD9_PREFIXES,
    icd10_prefixes: tuple[str, ...] = DIABETES_ICD10_PREFIXES,
) -> tuple[set[int], set[int]]:
    diabetes_hadm_ids: set[int] = set()
    diabetes_subject_ids: set[int] = set()

    with gzip.open(diagnoses_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if is_diabetes_code(row["icd_code"], row["icd_version"], icd9_prefixes, icd10_prefixes):
                diabetes_hadm_ids.add(int(row["hadm_id"]))
                diabetes_subject_ids.add(int(row["subject_id"]))

    return diabetes_hadm_ids, diabetes_subject_ids


def read_admissions(admissions_path: Path) -> pd.DataFrame:
    usecols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "edregtime",
        "edouttime",
        "hospital_expire_flag",
    ]
    df = pd.read_csv(
        admissions_path,
        compression="gzip",
        usecols=usecols,
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
        low_memory=False,
    )
    return df


def read_patients(patients_path: Path) -> pd.DataFrame:
    usecols = ["subject_id", "gender", "anchor_age", "dod"]
    df = pd.read_csv(
        patients_path,
        compression="gzip",
        usecols=usecols,
        parse_dates=["dod"],
        low_memory=False,
    )
    return df


def assign_patient_split(subject_id: int) -> str:
    digest = hashlib.md5(str(int(subject_id)).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    if bucket < 700:
        return "train"
    if bucket < 850:
        return "validation"
    return "test"


def add_timeline_columns(admissions: pd.DataFrame) -> pd.DataFrame:
    df = admissions.sort_values(["subject_id", "admittime", "hadm_id"]).copy()
    grouped = df.groupby("subject_id", sort=False)

    df["prior_admissions_count"] = grouped.cumcount()
    df["prior_dischtime"] = grouped["dischtime"].shift(1)
    df["days_since_prior_discharge"] = (
        (df["admittime"] - df["prior_dischtime"]).dt.total_seconds() / 86400.0
    )

    df["next_hadm_id"] = grouped["hadm_id"].shift(-1)
    df["next_admittime"] = grouped["admittime"].shift(-1)
    df["next_admission_type"] = grouped["admission_type"].shift(-1)
    df["days_to_next_admission"] = (
        (df["next_admittime"] - df["dischtime"]).dt.total_seconds() / 86400.0
    )
    df["readmission_30d"] = df["days_to_next_admission"].between(0, 30, inclusive="both")

    df["length_of_stay_days"] = (
        (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
    )
    df["ed_los_hours"] = (
        (df["edouttime"] - df["edregtime"]).dt.total_seconds() / 3600.0
    )
    return df


def build_cohort(mimic_root: Path) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    hosp = mimic_root / "hosp"
    admissions_path = hosp / "admissions.csv.gz"
    patients_path = hosp / "patients.csv.gz"
    diagnoses_path = hosp / "diagnoses_icd.csv.gz"

    icd9_prefixes, icd10_prefixes = diabetes_code_prefixes()
    diabetes_hadm_ids, diabetes_subject_ids = collect_diabetes_ids(diagnoses_path, icd9_prefixes, icd10_prefixes)
    admissions = read_admissions(admissions_path)
    patients = read_patients(patients_path)

    timeline = add_timeline_columns(admissions)
    timeline = timeline.merge(patients, on="subject_id", how="left")

    timeline["is_diabetes_admission"] = timeline["hadm_id"].isin(diabetes_hadm_ids)
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
        timeline["is_diabetes_admission"]
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
        *DISCHARGE_SAFE_FEATURES,
    ]
    cohort = cohort[output_columns].sort_values(["subject_id", "admittime", "hadm_id"])

    summary = {
        "mimic_root": str(mimic_root),
        "total_admissions": int(len(admissions)),
        "total_admission_subjects": int(admissions["subject_id"].nunique()),
        "total_patients_table": int(len(patients)),
        "raw_diabetes_admissions": int(len(diabetes_hadm_ids)),
        "raw_diabetes_subjects": int(len(diabetes_subject_ids)),
        "adult_diabetes_valid_time_admissions": int(base_mask.sum()),
        "excluded_in_hospital_death": int((base_mask & timeline["in_hospital_death"]).sum()),
        "excluded_postdischarge_death_30d_no_readmission": int(
            (base_mask & ~timeline["in_hospital_death"] & timeline["exclude_early_death_no_readmit"]).sum()
        ),
        "final_index_admissions": int(len(cohort)),
        "final_subjects": int(cohort["subject_id"].nunique()),
        "readmission_30d_count": int(cohort["readmission_30d"].sum()),
        "readmission_30d_rate": float(cohort["readmission_30d"].mean()),
    }
    return cohort, summary


def summarize_missingness(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in DISCHARGE_SAFE_FEATURES:
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


def metric_mean_sd(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "NA"
    return f"{values.mean():.2f} ({values.std():.2f})"


def metric_median_iqr(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "NA"
    q1 = values.quantile(0.25)
    q2 = values.quantile(0.50)
    q3 = values.quantile(0.75)
    return f"{q2:.2f} ({q1:.2f}-{q3:.2f})"


def metric_count_pct(mask: pd.Series) -> str:
    count = int(mask.fillna(False).sum())
    denom = int(mask.shape[0])
    pct = count / denom if denom else 0.0
    return f"{count} ({pct:.1%})"


def summarize_table1(cohort: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "overall": cohort,
        "no_readmission_30d": cohort[cohort["readmission_30d"] == 0],
        "readmission_30d": cohort[cohort["readmission_30d"] == 1],
    }
    rows = []
    definitions = [
        ("admissions", lambda df: str(len(df))),
        ("subjects", lambda df: str(df["subject_id"].nunique())),
        ("age_mean_sd", lambda df: metric_mean_sd(df["anchor_age"])),
        ("female", lambda df: metric_count_pct(df["gender"].eq("F"))),
        ("length_of_stay_days_median_iqr", lambda df: metric_median_iqr(df["length_of_stay_days"])),
        ("ed_los_hours_median_iqr", lambda df: metric_median_iqr(df["ed_los_hours"])),
        ("prior_admissions_count_median_iqr", lambda df: metric_median_iqr(df["prior_admissions_count"])),
        (
            "emergency_or_urgent_admission",
            lambda df: metric_count_pct(df["admission_type"].astype(str).str.contains("EMER|URGENT|EW", case=False, regex=True)),
        ),
    ]
    for variable, func in definitions:
        row = {"variable": variable}
        for group_name, df in groups.items():
            row[group_name] = func(df)
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_csv(summary: dict[str, int | float | str], path: Path) -> None:
    rows = [{"metric": key, "value": value} for key, value in summary.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def check_split_overlap(cohort: pd.DataFrame) -> dict[str, int]:
    split_subjects = {
        split: set(part["subject_id"].unique())
        for split, part in cohort.groupby("split", sort=True)
    }
    return {
        "train_validation_overlap": len(split_subjects.get("train", set()) & split_subjects.get("validation", set())),
        "train_test_overlap": len(split_subjects.get("train", set()) & split_subjects.get("test", set())),
        "validation_test_overlap": len(split_subjects.get("validation", set()) & split_subjects.get("test", set())),
    }


def write_reports(
    cohort: pd.DataFrame,
    summary: dict[str, int | float | str],
    split_overlap: dict[str, int],
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    event_rate = summary["readmission_30d_rate"]
    audit = f"""# MIMIC 糖尿病再入院 Leakage Audit Report

## 审计结论

第一版 cohort 和 30 天再入院标签已生成。当前阶段结论为 `PASS_WITH_WARNINGS`：可以进入 baseline 建模，但建模时必须继续排除 outcome、未来住院信息和死亡相关 proxy。

## 研究设定

- 数据：`{summary["mimic_root"]}`
- 任务：糖尿病住院患者 30 天再入院预测
- 预测时间点：出院时 `dischtime`
- 随访窗口：出院后 30 天
- 最终住院样本数：{summary["final_index_admissions"]}
- 最终患者数：{summary["final_subjects"]}
- 30 天再入院事件率：{event_rate:.2%}

## 已检查项目

| 检查项 | 结果 | 解释 |
|---|---|---|
| 当前住院死亡排除 | PASS | 已排除院内死亡 index admission。 |
| 出院后 30 天内死亡且无再入院 | WARNING | 第一版已排除这些样本，避免没有完整再入院观察机会。 |
| 患者级切分 | PASS | 使用 `subject_id` 哈希切分 train/validation/test。 |
| train/validation 患者重叠 | {split_overlap["train_validation_overlap"]} | 应为 0。 |
| train/test 患者重叠 | {split_overlap["train_test_overlap"]} | 应为 0。 |
| validation/test 患者重叠 | {split_overlap["validation_test_overlap"]} | 应为 0。 |
| outcome 变量作为特征 | MUST_EXCLUDE | `readmission_30d` 只能作为标签。 |
| 未来住院信息作为特征 | MUST_EXCLUDE | `next_admittime`, `next_hadm_id`, `days_to_next_admission` 不能进入模型。 |
| 死亡相关变量作为特征 | MUST_EXCLUDE | `deathtime`, `dod`, `hospital_expire_flag` 不进入第一版特征集。 |
| 当前住院 ICD 诊断作为入院预测特征 | WARNING | 当前 demo 是出院时预测；若改成入院时预测，不能使用当前住院最终 ICD。 |

## 第一版允许的特征

{chr(10).join(f"- `{col}`" for col in DISCHARGE_SAFE_FEATURES)}

## 明确禁止进入模型的列

{chr(10).join(f"- `{col}`" for col in FORBIDDEN_FEATURES)}

## 下一步建模提醒

- 必须报告 AUROC 和 AUPRC，不能只报告 AUROC。
- 必须报告 30 天再入院事件率。
- Logistic regression 和 random forest 要先作为 baseline。
- 如果后续加入化验和用药，必须按 `dischtime` 截断，不能读取出院后的记录。
"""
    (reports_dir / "mimic_diabetes_leakage_audit_report.md").write_text(audit, encoding="utf-8")

    run_summary = f"""# MIMIC 糖尿病 cohort 运行摘要

## 关键结果

- 原始住院数：{summary["total_admissions"]}
- 原始住院患者数：{summary["total_admission_subjects"]}
- 糖尿病相关住院数：{summary["raw_diabetes_admissions"]}
- 糖尿病相关患者数：{summary["raw_diabetes_subjects"]}
- 成人且时间有效的糖尿病住院数：{summary["adult_diabetes_valid_time_admissions"]}
- 排除院内死亡：{summary["excluded_in_hospital_death"]}
- 排除出院后 30 天内死亡且无再入院：{summary["excluded_postdischarge_death_30d_no_readmission"]}
- 最终 index admissions：{summary["final_index_admissions"]}
- 最终 subjects：{summary["final_subjects"]}
- 30 天再入院：{summary["readmission_30d_count"]} ({event_rate:.2%})

## 输出文件

- `data/processed/mimic_diabetes_readmission_cohort.csv`
- `outputs/tables/mimic_diabetes_cohort_summary.csv`
- `outputs/tables/mimic_diabetes_split_summary.csv`
- `outputs/tables/mimic_diabetes_feature_missingness.csv`
- `outputs/tables/mimic_diabetes_table1_basic.csv`
- `outputs/reports/mimic_diabetes_leakage_audit_report.md`

## 解释

这个结果说明糖尿病 30 天再入院 demo 可以继续推进到 baseline 建模。第一版标签是 all-cause 30-day readmission；后续如要更接近临床再入院研究，可以进一步排除 elective readmission 或转院相关入院。
"""
    (reports_dir / "mimic_diabetes_run_summary.md").write_text(run_summary, encoding="utf-8")


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project_root
    processed_dir = project / "data" / "processed"
    tables_dir = project / "outputs" / "tables"
    reports_dir = project / "outputs" / "reports"
    ensure_dirs([processed_dir, tables_dir, reports_dir])

    cohort, summary = build_cohort(args.mimic_root)
    split_overlap = check_split_overlap(cohort)

    cohort.to_csv(processed_dir / "mimic_diabetes_readmission_cohort.csv", index=False)
    write_summary_csv(summary, tables_dir / "mimic_diabetes_cohort_summary.csv")
    summarize_splits(cohort).to_csv(tables_dir / "mimic_diabetes_split_summary.csv", index=False)
    summarize_missingness(cohort).to_csv(tables_dir / "mimic_diabetes_feature_missingness.csv", index=False)
    summarize_table1(cohort).to_csv(tables_dir / "mimic_diabetes_table1_basic.csv", index=False)
    write_reports(cohort, summary, split_overlap, reports_dir)

    print("MIMIC diabetes cohort built")
    print(f"final_index_admissions={summary['final_index_admissions']}")
    print(f"final_subjects={summary['final_subjects']}")
    print(f"readmission_30d_rate={summary['readmission_30d_rate']:.4f}")


if __name__ == "__main__":
    main()
