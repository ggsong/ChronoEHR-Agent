#!/usr/bin/env python3
"""Generate and score a tiny synthetic EHR-style ChronoEHR-Agent demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
RAW_TABLE_COLUMNS = {
    "patients": ["subject_id", "gender", "anchor_age"],
    "admissions": ["hadm_id", "subject_id", "admittime", "dischtime", "admission_type"],
    "diagnoses": ["hadm_id", "seq_num", "icd_code", "long_title"],
    "labs": ["hadm_id", "charttime", "itemid", "label", "valuenum", "flag"],
    "medications": ["hadm_id", "charttime", "drug", "route"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--n-patients", type=int, default=120, help="Number of synthetic patients to generate.")
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


def make_raw_tables(n_patients: int, seed: int) -> dict[str, pd.DataFrame]:
    """Create small synthetic raw tables with EHR-like IDs and timestamps."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01 08:00:00")
    patients = pd.DataFrame(
        {
            "subject_id": np.arange(1, n_patients + 1),
            "gender": rng.choice(["F", "M"], size=n_patients),
            "anchor_age": rng.integers(35, 91, size=n_patients),
        }
    )

    admissions_rows: list[dict[str, object]] = []
    diagnoses_rows: list[dict[str, object]] = []
    labs_rows: list[dict[str, object]] = []
    medication_rows: list[dict[str, object]] = []
    hadm_id = 100000
    lab_itemids = {
        "creatinine": 50912,
        "glucose": 50931,
        "hemoglobin": 51222,
        "sodium": 50983,
    }
    chronic_codes = ["E11", "I10", "I50", "N18"]
    admission_types = ["URGENT", "EMERGENCY", "ELECTIVE"]
    drugs = ["insulin", "metformin", "furosemide", "lisinopril", "atorvastatin"]

    for patient in patients.itertuples(index=False):
        n_admissions = int(np.clip(rng.poisson(1.2) + 1, 1, 5))
        current_time = start + pd.Timedelta(days=int(rng.integers(0, 180)))
        for admission_number in range(n_admissions):
            hadm_id += 1
            length_of_stay = float(np.clip(rng.gamma(2.2, 1.8), 0.6, 18.0))
            admittime = current_time
            dischtime = admittime + pd.Timedelta(days=length_of_stay)
            admissions_rows.append(
                {
                    "hadm_id": hadm_id,
                    "subject_id": int(patient.subject_id),
                    "admittime": admittime,
                    "dischtime": dischtime,
                    "admission_type": str(rng.choice(admission_types, p=[0.35, 0.5, 0.15])),
                }
            )

            primary_code = str(rng.choice(chronic_codes, p=[0.35, 0.3, 0.2, 0.15]))
            diagnoses_rows.append(
                {
                    "hadm_id": hadm_id,
                    "seq_num": 1,
                    "icd_code": primary_code,
                    "long_title": f"synthetic diagnosis {primary_code}",
                }
            )
            if rng.random() < 0.45:
                secondary_code = str(rng.choice([code for code in chronic_codes if code != primary_code]))
                diagnoses_rows.append(
                    {
                        "hadm_id": hadm_id,
                        "seq_num": 2,
                        "icd_code": secondary_code,
                        "long_title": f"synthetic diagnosis {secondary_code}",
                    }
                )

            for label, itemid in lab_itemids.items():
                n_labs = int(rng.integers(1, 5))
                for _ in range(n_labs):
                    charttime = admittime + pd.Timedelta(hours=float(rng.uniform(1, max(length_of_stay * 24 - 1, 2))))
                    age_risk = max((float(patient.anchor_age) - 45) / 80, 0)
                    baseline = {
                        "creatinine": 0.9 + age_risk,
                        "glucose": 100 + 35 * age_risk,
                        "hemoglobin": 13.5 - 1.5 * age_risk,
                        "sodium": 139,
                    }[label]
                    value = float(rng.normal(baseline, {"creatinine": 0.25, "glucose": 25, "hemoglobin": 1.2, "sodium": 3}[label]))
                    abnormal = {
                        "creatinine": value > 1.3,
                        "glucose": value > 180,
                        "hemoglobin": value < 11,
                        "sodium": value < 135 or value > 145,
                    }[label]
                    labs_rows.append(
                        {
                            "hadm_id": hadm_id,
                            "charttime": charttime,
                            "itemid": itemid,
                            "label": label,
                            "valuenum": round(value, 3),
                            "flag": "abnormal" if abnormal else "normal",
                        }
                    )

            n_meds = int(rng.integers(1, 5))
            for _ in range(n_meds):
                medication_rows.append(
                    {
                        "hadm_id": hadm_id,
                        "charttime": admittime + pd.Timedelta(hours=float(rng.uniform(2, max(length_of_stay * 24 - 2, 3)))),
                        "drug": str(rng.choice(drugs)),
                        "route": str(rng.choice(["PO", "IV", "SC"], p=[0.55, 0.3, 0.15])),
                    }
                )

            readmission_pressure = 0.15 + 0.005 * max(int(patient.anchor_age) - 50, 0) + 0.08 * admission_number
            if rng.random() < min(readmission_pressure, 0.75):
                gap_days = int(rng.integers(3, 28))
            else:
                gap_days = int(rng.integers(45, 160))
            current_time = dischtime + pd.Timedelta(days=gap_days, hours=int(rng.integers(1, 18)))

    return {
        "patients": patients,
        "admissions": pd.DataFrame(admissions_rows),
        "diagnoses": pd.DataFrame(diagnoses_rows),
        "labs": pd.DataFrame(labs_rows),
        "medications": pd.DataFrame(medication_rows),
    }


def validate_raw_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Validate schema, key integrity, and prediction-time boundaries."""
    rows: list[dict[str, object]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    for table_name, required_columns in RAW_TABLE_COLUMNS.items():
        table = tables.get(table_name)
        add(f"{table_name}_exists", table is not None, f"required columns: {', '.join(required_columns)}")
        if table is None:
            continue
        missing = sorted(set(required_columns) - set(table.columns))
        add(f"{table_name}_columns", not missing, "missing: " + ", ".join(missing) if missing else "all required columns present")
        add(f"{table_name}_nonempty", len(table) > 0, f"rows={len(table)}")

    if "patients" not in tables or "admissions" not in tables:
        return pd.DataFrame(rows)

    patients = tables["patients"]
    admissions = tables["admissions"].copy()
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    patient_ids = set(patients["subject_id"])
    admission_ids = set(admissions["hadm_id"])

    add("admission_subject_foreign_key", set(admissions["subject_id"]).issubset(patient_ids), "all admissions should map to patients")
    add("admission_time_order", (admissions["admittime"] < admissions["dischtime"]).all(), "admittime must precede dischtime")
    add("hadm_id_unique", admissions["hadm_id"].is_unique, "one row per synthetic admission")

    for table_name in ["diagnoses", "labs", "medications"]:
        table = tables.get(table_name)
        if table is not None and "hadm_id" in table:
            add(f"{table_name}_hadm_foreign_key", set(table["hadm_id"]).issubset(admission_ids), "all rows should map to admissions")

    for table_name in ["labs", "medications"]:
        table = tables.get(table_name)
        if table is None or "charttime" not in table:
            continue
        merged = table.merge(admissions[["hadm_id", "admittime", "dischtime"]], on="hadm_id", how="left")
        charttime = pd.to_datetime(merged["charttime"])
        inside_window = (charttime >= merged["admittime"]) & (charttime <= merged["dischtime"])
        add(f"{table_name}_within_admission_window", bool(inside_window.all()), f"violations={(~inside_window).sum()}")

    labels = build_labels(admissions)
    valid_labels = labels["readmission_30d"].isin([0, 1]).all() and labels["days_to_next_admission"].dropna().ge(0).all()
    add("readmission_label_contract", bool(valid_labels), "labels are binary and next admissions occur after discharge")
    return pd.DataFrame(rows)


def build_labels(admissions: pd.DataFrame) -> pd.DataFrame:
    ordered = admissions.sort_values(["subject_id", "admittime", "hadm_id"]).copy()
    ordered["next_admittime"] = ordered.groupby("subject_id")["admittime"].shift(-1)
    ordered["days_to_next_admission"] = (ordered["next_admittime"] - ordered["dischtime"]).dt.total_seconds() / 86400
    ordered["readmission_30d"] = ordered["days_to_next_admission"].between(0, 30, inclusive="both").fillna(False).astype(int)
    return ordered[["hadm_id", "next_admittime", "days_to_next_admission", "readmission_30d"]]


def build_cohort_from_raw(tables: dict[str, pd.DataFrame], seed: int) -> pd.DataFrame:
    patients = tables["patients"]
    admissions = tables["admissions"].copy()
    labs = tables["labs"].copy()
    medications = tables["medications"].copy()
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    medications["charttime"] = pd.to_datetime(medications["charttime"])

    cohort = admissions.merge(patients, on="subject_id", how="left")
    cohort["length_of_stay_days"] = (cohort["dischtime"] - cohort["admittime"]).dt.total_seconds() / 86400
    cohort = cohort.sort_values(["subject_id", "admittime", "hadm_id"])
    cohort["prior_admissions_count"] = cohort.groupby("subject_id").cumcount()
    cohort = cohort.merge(build_labels(admissions), on="hadm_id", how="left")

    lab_features = (
        labs.assign(abnormal_lab=lambda df: df["flag"].eq("abnormal").astype(int))
        .groupby("hadm_id")
        .agg(abnormal_lab_count=("abnormal_lab", "sum"), lab_measurement_count=("itemid", "count"))
        .reset_index()
    )
    medication_features = medications.groupby("hadm_id").agg(medication_change_count=("drug", "nunique")).reset_index()
    cohort = cohort.merge(lab_features, on="hadm_id", how="left").merge(medication_features, on="hadm_id", how="left")
    for col in ["abnormal_lab_count", "lab_measurement_count", "medication_change_count"]:
        cohort[col] = cohort[col].fillna(0).astype(int)

    rng = np.random.default_rng(seed + 17)
    cohort["split"] = np.where(rng.random(len(cohort)) < 0.75, "train", "test")
    if cohort["split"].nunique() < 2 and len(cohort) > 1:
        cohort.loc[cohort.index[-1], "split"] = "test"

    cohort["prediction_time"] = "discharge"
    return cohort[
        [
            "hadm_id",
            "subject_id",
            "prediction_time",
            "gender",
            "anchor_age",
            "admission_type",
            "prior_admissions_count",
            "length_of_stay_days",
            "lab_measurement_count",
            "abnormal_lab_count",
            "medication_change_count",
            "readmission_30d",
            "days_to_next_admission",
            "split",
        ]
    ]


def score_rows(cohort: pd.DataFrame) -> pd.DataFrame:
    score = (
        -2.9
        + 0.022 * cohort["anchor_age"]
        + 0.3 * cohort["prior_admissions_count"]
        + 0.07 * cohort["length_of_stay_days"]
        + 0.15 * cohort["abnormal_lab_count"]
        + 0.12 * cohort["medication_change_count"]
    )
    out = cohort.copy()
    out["synthetic_risk_score"] = 1 / (1 + np.exp(-score))
    out = out.rename(columns={"anchor_age": "age"})
    return out


def metric_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", sort=True):
        predicted = (group["synthetic_risk_score"] >= 0.5).astype(int)
        actual = group["readmission_30d"].astype(int)
        accuracy = float((predicted == actual).mean())
        event_rate = float(actual.mean())
        rows.append(
            {
                "split": split,
                "n": int(len(group)),
                "event_rate": round(event_rate, 4),
                "mean_risk_score": round(float(group["synthetic_risk_score"].mean()), 4),
                "threshold_0_50_accuracy": round(accuracy, 4),
            }
        )
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, contract: pd.DataFrame, report_path: Path) -> None:
    table_lines = ["| split | n | event_rate | mean_risk_score | threshold_0_50_accuracy |", "|---|---:|---:|---:|---:|"]
    for row in metrics.to_dict(orient="records"):
        table_lines.append(
            f"| {row['split']} | {row['n']} | {row['event_rate']} | {row['mean_risk_score']} | {row['threshold_0_50_accuracy']} |"
        )
    contract_status = "PASS" if contract["status"].eq("PASS").all() else "FAIL"
    lines = [
        "# Synthetic ChronoEHR Demo",
        "",
        "This report was generated from artificial EHR-style raw tables only.",
        "",
        f"- Data contract status: `{contract_status}`",
        f"- Data contract checks: {len(contract)}",
        "",
        *table_lines,
        "",
        "The demo verifies that the public repository can generate raw EHR-like tables, validate schema and temporal boundaries, build a cohort, score rows, and validate artifacts without controlled clinical data.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_tables(tables: dict[str, pd.DataFrame], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(raw_dir / f"{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.project_root / "outputs" / "demo"
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = make_raw_tables(args.n_patients, args.seed)
    write_raw_tables(tables, raw_dir)
    contract = validate_raw_tables(tables)
    if not contract["status"].eq("PASS").all():
        contract.to_csv(out_dir / "synthetic_ehr_contract.csv", index=False)
        raise SystemExit("Synthetic raw EHR data contract failed.")

    cohort = build_cohort_from_raw(tables, args.seed)
    scored = score_rows(cohort)
    metrics = metric_rows(scored)
    scored.to_csv(out_dir / "synthetic_cohort.csv", index=False)
    metrics.to_csv(out_dir / "synthetic_demo_metrics.csv", index=False)
    contract.to_csv(out_dir / "synthetic_ehr_contract.csv", index=False)
    write_report(metrics, contract, out_dir / "synthetic_demo_report.md")
    print(f"Wrote synthetic EHR demo artifacts to {out_dir}")


if __name__ == "__main__":
    main()
