#!/usr/bin/env python3
"""Build an eICU first-24h hospital mortality cohort skeleton.

This script prepares the stay-level cohort only. It intentionally does not
train models and does not extract high-volume lab/vital features.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from eicu_data_readiness import choose_root, find_first
from mimic_diabetes_baseline import DEFAULT_PROJECT


PATIENT_FILES = ["patient.csv", "patient.csv.gz"]
REQUIRED_COLUMNS = [
    "patientunitstayid",
    "uniquepid",
    "gender",
    "age",
    "ethnicity",
    "hospitalid",
    "wardid",
    "apacheadmissiondx",
    "admissionheight",
    "hospitaladmitoffset",
    "hospitaladmitsource",
    "hospitaldischargeoffset",
    "hospitaldischargestatus",
    "unittype",
    "unitadmitsource",
    "admissionweight",
    "unitdischargeoffset",
    "unitdischargestatus",
]
OUTCOME_FORBIDDEN_FEATURES = [
    "hospital_mortality",
    "icu_mortality",
    "hospitaldischargestatus",
    "unitdischargestatus",
    "hospitaldischargeoffset",
    "unitdischargeoffset",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--eicu-root", type=Path, help="Optional explicit eICU root.")
    return parser.parse_args()


def clean_age(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "unknown"}:
        return None
    if text.startswith(">"):
        return 90.0
    try:
        return float(text)
    except ValueError:
        return None


def mortality_label(value: object) -> int | None:
    text = str(value).strip().lower()
    if text in {"expired", "death", "dead", "deceased"}:
        return 1
    if text in {"alive", "home", "skilled nursing facility", "rehab"}:
        return 0
    return None


def split_for_patient(patient_id: object) -> str:
    digest = hashlib.md5(str(patient_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def read_patient_table(patient_path: Path) -> pd.DataFrame:
    usecols = pd.read_csv(patient_path, nrows=0, low_memory=False).columns.tolist()
    missing = sorted(set(REQUIRED_COLUMNS) - set(usecols))
    if missing:
        raise ValueError(f"patient table is missing required columns: {missing}")
    return pd.read_csv(patient_path, usecols=REQUIRED_COLUMNS, low_memory=False)


def build_cohort(patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_n = len(patient)
    cohort = patient.copy()
    cohort["age_years"] = cohort["age"].map(clean_age)
    cohort["hospital_mortality"] = cohort["hospitaldischargestatus"].map(mortality_label)
    cohort["icu_mortality"] = cohort["unitdischargestatus"].map(mortality_label)
    cohort["unit_los_minutes"] = pd.to_numeric(cohort["unitdischargeoffset"], errors="coerce")
    cohort["hospital_los_from_icu_admit_minutes"] = pd.to_numeric(cohort["hospitaldischargeoffset"], errors="coerce")
    cohort["hospitaladmitoffset"] = pd.to_numeric(cohort["hospitaladmitoffset"], errors="coerce")
    cohort["icu_admission_prediction_offset"] = 0
    cohort["first_24h_prediction_offset"] = 1440

    exclusions = []
    exclusions.append({"step": "raw_patient_rows", "remaining": start_n, "excluded": 0})

    before = len(cohort)
    cohort = cohort.dropna(subset=["patientunitstayid", "uniquepid"])
    exclusions.append({"step": "valid_stay_and_patient_ids", "remaining": len(cohort), "excluded": before - len(cohort)})

    before = len(cohort)
    cohort = cohort[cohort["age_years"].ge(18)]
    exclusions.append({"step": "adult_age_at_least_18", "remaining": len(cohort), "excluded": before - len(cohort)})

    before = len(cohort)
    cohort = cohort[cohort["hospital_mortality"].isin([0, 1])]
    exclusions.append({"step": "valid_hospital_mortality_label", "remaining": len(cohort), "excluded": before - len(cohort)})

    before = len(cohort)
    cohort = cohort[cohort["unit_los_minutes"].gt(0)]
    exclusions.append({"step": "positive_icu_los_offset", "remaining": len(cohort), "excluded": before - len(cohort)})

    cohort["eligible_admission_prediction"] = True
    cohort["eligible_first_24h_prediction"] = cohort["unit_los_minutes"].ge(1440)
    cohort["split"] = cohort["uniquepid"].map(split_for_patient)

    rename = {
        "patientunitstayid": "stay_id",
        "uniquepid": "patient_id",
        "apacheadmissiondx": "admission_diagnosis_text",
    }
    cohort = cohort.rename(columns=rename)
    ordered = [
        "stay_id",
        "patient_id",
        "split",
        "age_years",
        "gender",
        "ethnicity",
        "hospitalid",
        "wardid",
        "unittype",
        "unitadmitsource",
        "hospitaladmitsource",
        "admission_diagnosis_text",
        "admissionheight",
        "admissionweight",
        "hospitaladmitoffset",
        "icu_admission_prediction_offset",
        "first_24h_prediction_offset",
        "unit_los_minutes",
        "hospital_los_from_icu_admit_minutes",
        "eligible_admission_prediction",
        "eligible_first_24h_prediction",
        "hospital_mortality",
        "icu_mortality",
    ]
    return cohort[ordered].sort_values(["patient_id", "stay_id"]), pd.DataFrame(exclusions)


def summarize(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "stays", "value": len(cohort)},
        {"metric": "patients", "value": cohort["patient_id"].nunique()},
        {"metric": "hospital_mortality_rate", "value": round(float(cohort["hospital_mortality"].mean()), 4)},
        {"metric": "icu_mortality_rate", "value": round(float(cohort["icu_mortality"].mean()), 4)},
        {"metric": "first_24h_eligible_stays", "value": int(cohort["eligible_first_24h_prediction"].sum())},
        {"metric": "first_24h_eligible_rate", "value": round(float(cohort["eligible_first_24h_prediction"].mean()), 4)},
    ]
    for split, group in cohort.groupby("split"):
        rows.append({"metric": f"{split}_stays", "value": len(group)})
        rows.append({"metric": f"{split}_patients", "value": group["patient_id"].nunique()})
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, patient_path: Path, cohort: pd.DataFrame, summary: pd.DataFrame, exclusions: pd.DataFrame) -> Path:
    report_path = project_root / "outputs" / "reports" / "eicu_temporal_mortality_cohort_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    split_overlap = cohort.groupby("patient_id")["split"].nunique().gt(1).sum()
    text = f"""# eICU Temporal Mortality Cohort Skeleton

- Source table: `{patient_path}`
- Output cohort: `data/processed/eicu_temporal_mortality_cohort.csv`
- Boundary: cohort construction only; no medical QA, diagnosis, treatment recommendation, or model training.

## What This Cohort Means

This is the first eICU vertical slice for ChronoEHR-Agent. It defines adult ICU stays, a hospital mortality outcome, ICU-admission and first-24h prediction anchors, and a patient-level train/validation/test split.

The outcome and discharge offsets are included only as labels/anchors. They must not be used as early prediction features.

## Summary

{markdown_table(summary)}

## Exclusions

{markdown_table(exclusions)}

## Leakage Boundary

- Forbidden as early features: `{", ".join(OUTCOME_FORBIDDEN_FEATURES)}`
- First-24h feature extraction must keep only rows with event offset `<= 1440`.
- Patient split overlap count: `{split_overlap}`.
- Stays discharged or dead before 24h are flagged by `eligible_first_24h_prediction = False`.
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    selected_root, _ = choose_root(args.eicu_root)
    if selected_root is None:
        raise SystemExit("No eICU root with required tables was found. Run --eicu-readiness first.")
    patient_path = find_first(selected_root, PATIENT_FILES)
    if patient_path is None:
        raise SystemExit(f"No patient table found under {selected_root}")

    patient = read_patient_table(patient_path)
    cohort, exclusions = build_cohort(patient)
    summary = summarize(cohort)

    processed = args.project_root / "data" / "processed"
    tables = args.project_root / "outputs" / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(processed / "eicu_temporal_mortality_cohort.csv", index=False)
    summary.to_csv(tables / "eicu_temporal_mortality_cohort_summary.csv", index=False)
    exclusions.to_csv(tables / "eicu_temporal_mortality_cohort_exclusions.csv", index=False)
    split_summary = cohort.groupby("split").agg(stays=("stay_id", "size"), patients=("patient_id", "nunique"), mortality_rate=("hospital_mortality", "mean")).reset_index()
    split_summary.to_csv(tables / "eicu_temporal_mortality_split_summary.csv", index=False)
    report = write_report(args.project_root, patient_path, cohort, summary, exclusions)
    print(f"Wrote {report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
