#!/usr/bin/env python3
"""Validate synthetic ChronoEHR-Agent demo artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
REQUIRED_COHORT_COLUMNS = {
    "hadm_id",
    "subject_id",
    "prediction_time",
    "age",
    "gender",
    "admission_type",
    "prior_admissions_count",
    "length_of_stay_days",
    "lab_measurement_count",
    "abnormal_lab_count",
    "medication_change_count",
    "readmission_30d",
    "days_to_next_admission",
    "split",
    "synthetic_risk_score",
}
REQUIRED_RAW_TABLES = {
    "patients.csv": {"subject_id", "gender", "anchor_age"},
    "admissions.csv": {"hadm_id", "subject_id", "admittime", "dischtime", "admission_type"},
    "diagnoses.csv": {"hadm_id", "seq_num", "icd_code", "long_title"},
    "labs.csv": {"hadm_id", "charttime", "itemid", "label", "valuenum", "flag"},
    "medications.csv": {"hadm_id", "charttime", "drug", "route"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_dir = args.project_root / "outputs" / "demo"
    cohort_path = demo_dir / "synthetic_cohort.csv"
    metrics_path = demo_dir / "synthetic_demo_metrics.csv"
    report_path = demo_dir / "synthetic_demo_report.md"
    contract_path = demo_dir / "synthetic_ehr_contract.csv"
    raw_dir = demo_dir / "raw"
    raw_paths = [raw_dir / filename for filename in REQUIRED_RAW_TABLES]
    missing_files = [path for path in [cohort_path, metrics_path, report_path, contract_path, *raw_paths] if not path.exists()]
    if missing_files:
        raise SystemExit("Missing synthetic demo artifacts: " + ", ".join(str(path) for path in missing_files))

    for filename, required_columns in REQUIRED_RAW_TABLES.items():
        raw = pd.read_csv(raw_dir / filename)
        missing_raw_columns = sorted(required_columns - set(raw.columns))
        if missing_raw_columns:
            raise SystemExit(f"Raw table {filename} missing columns: " + ", ".join(missing_raw_columns))
        if raw.empty:
            raise SystemExit(f"Raw table {filename} is unexpectedly empty.")

    admissions = pd.read_csv(raw_dir / "admissions.csv")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    if not (admissions["admittime"] < admissions["dischtime"]).all():
        raise SystemExit("Synthetic admissions violate admittime < dischtime.")

    for filename in ["labs.csv", "medications.csv"]:
        table = pd.read_csv(raw_dir / filename)
        merged = table.merge(admissions[["hadm_id", "admittime", "dischtime"]], on="hadm_id", how="left")
        charttime = pd.to_datetime(merged["charttime"])
        inside_window = (charttime >= merged["admittime"]) & (charttime <= merged["dischtime"])
        if not inside_window.all():
            raise SystemExit(f"{filename} has post-prediction or pre-admission events.")

    cohort = pd.read_csv(cohort_path)
    metrics = pd.read_csv(metrics_path)
    contract = pd.read_csv(contract_path)
    if not contract["status"].eq("PASS").all():
        failures = contract.loc[~contract["status"].eq("PASS"), "check"].astype(str).tolist()
        raise SystemExit("Synthetic EHR contract failures: " + ", ".join(failures))
    missing_columns = sorted(REQUIRED_COHORT_COLUMNS - set(cohort.columns))
    if missing_columns:
        raise SystemExit("Synthetic cohort missing columns: " + ", ".join(missing_columns))
    if len(cohort) < 50:
        raise SystemExit("Synthetic cohort is unexpectedly small.")
    if not set(cohort["split"]).issuperset({"train", "test"}):
        raise SystemExit("Synthetic cohort must contain train and test splits.")
    if not cohort["synthetic_risk_score"].between(0, 1).all():
        raise SystemExit("Synthetic risk scores must be in [0, 1].")
    if not cohort["readmission_30d"].isin([0, 1]).all():
        raise SystemExit("Synthetic readmission labels must be binary.")
    if cohort["days_to_next_admission"].dropna().lt(0).any():
        raise SystemExit("Synthetic next admissions must occur after discharge.")
    if not set(metrics["split"]).issuperset({"train", "test"}):
        raise SystemExit("Synthetic metrics must contain train and test rows.")

    print(
        f"Synthetic demo validation PASS: {len(cohort)} rows, {len(metrics)} metric rows, "
        f"{len(contract)} contract checks."
    )


if __name__ == "__main__":
    main()
