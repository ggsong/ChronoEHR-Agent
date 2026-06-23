"""Reusable MIMIC-IV diagnosis-code cohort builder.

This module captures the shared pattern used by CKD, heart failure, and future
ICD-code chronic disease cohorts: identify admissions by ICD prefixes, construct
current/prior/known diagnosis flags, apply common readmission exclusions, and
create a patient-level split.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mimic_diabetes_cohort import (
    DISCHARGE_SAFE_FEATURES,
    DEFAULT_PROJECT,
    add_timeline_columns,
    assign_patient_split,
    read_admissions,
    read_patients,
)
from study_config_loader import load_cohort_code_rules


@dataclass(frozen=True)
class DiagnosisCohortSpec:
    cohort_key: str
    display_name: str
    icd9_prefixes: tuple[str, ...]
    icd10_prefixes: tuple[str, ...]
    current_col: str
    prior_col: str
    known_col: str
    include_prior_history: bool = True
    config_path: str | None = None
    code_rule_key: str | None = None

    @property
    def diagnosis_features(self) -> list[str]:
        return [self.current_col, self.prior_col, self.known_col]

    @property
    def model_features(self) -> list[str]:
        return [*DISCHARGE_SAFE_FEATURES, *self.diagnosis_features]


CKD_SPEC = DiagnosisCohortSpec(
    cohort_key="ckd",
    display_name="CKD",
    icd9_prefixes=("585",),
    icd10_prefixes=("N18",),
    current_col="current_ckd_admission",
    prior_col="prior_ckd_diagnosis",
    known_col="known_ckd_before_or_current_admission",
    config_path="configs/ckd_mimic_readmission.yaml",
    code_rule_key="ckd_code_rules",
)

HEART_FAILURE_SPEC = DiagnosisCohortSpec(
    cohort_key="heart_failure",
    display_name="Heart failure",
    icd9_prefixes=("428",),
    icd10_prefixes=("I50",),
    current_col="current_hf_admission",
    prior_col="prior_hf_diagnosis",
    known_col="known_hf_before_or_current_admission",
    config_path="configs/heart_failure_mimic_readmission.yaml",
    code_rule_key="heart_failure_code_rules",
)

HYPERTENSION_SPEC = DiagnosisCohortSpec(
    cohort_key="hypertension",
    display_name="Hypertension",
    icd9_prefixes=("401", "402", "403", "404", "405"),
    icd10_prefixes=("I10", "I11", "I12", "I13", "I15"),
    current_col="current_hypertension_admission",
    prior_col="prior_hypertension_diagnosis",
    known_col="known_hypertension_before_or_current_admission",
    config_path="configs/hypertension_mimic_readmission.yaml",
    code_rule_key="hypertension_code_rules",
)

PRESETS = {
    "ckd": CKD_SPEC,
    "heart_failure": HEART_FAILURE_SPEC,
    "hypertension": HYPERTENSION_SPEC,
}


def norm_code(code: str) -> str:
    return str(code).strip().upper().replace(".", "")


def spec_code_prefixes(
    spec: DiagnosisCohortSpec,
    project_root: Path = DEFAULT_PROJECT,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not spec.config_path or not spec.code_rule_key:
        return spec.icd9_prefixes, spec.icd10_prefixes
    return load_cohort_code_rules(
        config_path=project_root / spec.config_path,
        rule_key=spec.code_rule_key,
        fallback_icd9=spec.icd9_prefixes,
        fallback_icd10=spec.icd10_prefixes,
    )


def matches_prefix(
    code: str,
    version: str,
    spec: DiagnosisCohortSpec,
    icd9_prefixes: tuple[str, ...] | None = None,
    icd10_prefixes: tuple[str, ...] | None = None,
) -> bool:
    if icd9_prefixes is None or icd10_prefixes is None:
        icd9_prefixes, icd10_prefixes = spec_code_prefixes(spec)
    code = norm_code(code)
    version = str(version).strip()
    if version == "9":
        return code.startswith(icd9_prefixes)
    if version == "10":
        return code.startswith(icd10_prefixes)
    return False


def collect_diagnosis_ids(diagnoses_path: Path, spec: DiagnosisCohortSpec) -> tuple[set[int], set[int]]:
    hadm_ids: set[int] = set()
    subject_ids: set[int] = set()
    icd9_prefixes, icd10_prefixes = spec_code_prefixes(spec)
    with gzip.open(diagnoses_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if matches_prefix(row["icd_code"], row["icd_version"], spec, icd9_prefixes, icd10_prefixes):
                hadm_ids.add(int(row["hadm_id"]))
                subject_ids.add(int(row["subject_id"]))
    return hadm_ids, subject_ids


def add_diagnosis_history(timeline: pd.DataFrame, spec: DiagnosisCohortSpec) -> pd.DataFrame:
    df = timeline.sort_values(["subject_id", "admittime", "hadm_id"]).copy()
    previous_count = df.groupby("subject_id", sort=False)[spec.current_col].cumsum() - df[spec.current_col].astype(int)
    df[spec.prior_col] = previous_count.gt(0)
    df[spec.known_col] = df[spec.prior_col] | df[spec.current_col]
    return df


def add_common_eligibility_columns(timeline: pd.DataFrame) -> pd.DataFrame:
    df = timeline.copy()
    df["adult"] = df["anchor_age"] >= 18
    df["valid_times"] = df["admittime"].notna() & df["dischtime"].notna()
    df["valid_los"] = df["length_of_stay_days"] >= 0
    df["in_hospital_death"] = df["hospital_expire_flag"].fillna(0).astype(int).eq(1) | df["deathtime"].notna()
    df["postdischarge_death_within_30d"] = (
        df["dod"].notna()
        & df["dischtime"].notna()
        & (df["dod"] >= df["dischtime"])
        & (df["dod"] <= df["dischtime"] + pd.Timedelta(days=30))
    )
    df["exclude_early_death_no_readmit"] = df["postdischarge_death_within_30d"] & ~df["readmission_30d"]
    return df


def build_diagnosis_readmission_cohort(
    mimic_root: Path,
    spec: DiagnosisCohortSpec,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    hosp = mimic_root / "hosp"
    admissions = read_admissions(hosp / "admissions.csv.gz")
    patients = read_patients(hosp / "patients.csv.gz")
    diagnosis_hadm_ids, diagnosis_subject_ids = collect_diagnosis_ids(hosp / "diagnoses_icd.csv.gz", spec)

    timeline = add_timeline_columns(admissions)
    timeline = timeline.merge(patients, on="subject_id", how="left")
    timeline[spec.current_col] = timeline["hadm_id"].isin(diagnosis_hadm_ids)
    timeline = add_diagnosis_history(timeline, spec)
    timeline = add_common_eligibility_columns(timeline)

    if spec.include_prior_history:
        diagnosis_mask = timeline[spec.known_col]
    else:
        diagnosis_mask = timeline[spec.current_col]
    base_mask = diagnosis_mask & timeline["adult"] & timeline["valid_times"] & timeline["valid_los"]

    cohort = timeline[base_mask & ~timeline["in_hospital_death"] & ~timeline["exclude_early_death_no_readmit"]].copy()
    cohort["readmission_30d"] = cohort["readmission_30d"].astype(int)
    cohort["split"] = cohort["subject_id"].apply(assign_patient_split)
    for col in spec.diagnosis_features:
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
        *spec.model_features,
    ]
    cohort = cohort[output_columns].sort_values(["subject_id", "admittime", "hadm_id"])

    key = spec.cohort_key
    summary = {
        "mimic_root": str(mimic_root),
        "total_admissions": int(len(admissions)),
        "total_admission_subjects": int(admissions["subject_id"].nunique()),
        "total_patients_table": int(len(patients)),
        f"raw_{key}_admissions": int(len(diagnosis_hadm_ids)),
        f"raw_{key}_subjects": int(len(diagnosis_subject_ids)),
        f"adult_{key}_valid_time_admissions": int(base_mask.sum()),
        f"current_{key}_index_admissions": int((base_mask & timeline[spec.current_col]).sum()),
        f"prior_only_{key}_index_admissions": int(
            (base_mask & ~timeline[spec.current_col] & timeline[spec.prior_col]).sum()
        ),
        "excluded_in_hospital_death": int((base_mask & timeline["in_hospital_death"]).sum()),
        "excluded_postdischarge_death_30d_no_readmission": int(
            (base_mask & ~timeline["in_hospital_death"] & timeline["exclude_early_death_no_readmit"]).sum()
        ),
        "final_index_admissions": int(len(cohort)),
        "final_subjects": int(cohort["subject_id"].nunique()),
        "readmission_30d_count": int(cohort["readmission_30d"].sum()),
        "readmission_30d_rate": float(cohort["readmission_30d"].mean()),
        f"{spec.prior_col}_rate": float(cohort[spec.prior_col].mean()),
        f"{spec.current_col}_rate": float(cohort[spec.current_col].mean()),
    }
    return cohort, summary
