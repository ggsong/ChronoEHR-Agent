#!/usr/bin/env python3
"""Extract eICU first-24h lab and vital feature skeletons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from eicu_data_readiness import choose_root, find_first
from mimic_diabetes_baseline import DEFAULT_PROJECT


LAB_FILES = ["lab.csv", "lab.csv.gz"]
VITAL_FILES = ["vitalPeriodic.csv", "vitalPeriodic.csv.gz", "vitalperiodic.csv", "vitalperiodic.csv.gz"]
WINDOW_START = 0
WINDOW_END = 1440

LAB_NAME_MAP = {
    "bedside glucose": "glucose",
    "glucose": "glucose",
    "creatinine": "creatinine",
    "bun": "bun",
    "sodium": "sodium",
    "potassium": "potassium",
    "chloride": "chloride",
    "bicarbonate": "bicarbonate",
    "hco3": "bicarbonate",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "wbc x 1000": "wbc",
    "platelets x 1000": "platelets",
    "albumin": "albumin",
    "calcium": "calcium",
    "magnesium": "magnesium",
    "anion gap": "anion_gap",
    "lactate": "lactate",
    "ph": "ph",
    "pao2": "pao2",
    "paco2": "paco2",
}
LABS = sorted(set(LAB_NAME_MAP.values()))

VITAL_COLUMNS = {
    "heartrate": "heart_rate",
    "respiration": "respiratory_rate",
    "sao2": "spo2",
    "temperature": "temperature",
    "systemicsystolic": "systolic_bp",
    "systemicdiastolic": "diastolic_bp",
    "systemicmean": "mean_bp",
    "cvp": "cvp",
}
VITALS = list(VITAL_COLUMNS.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--eicu-root", type=Path, help="Optional explicit eICU root.")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def empty_slot() -> dict[str, Any]:
    return {"count": 0, "sum": 0.0, "min": None, "max": None, "last_offset": None, "last": None}


def update_state(state: dict[int, dict[str, dict[str, Any]]], grouped: pd.DataFrame, latest: pd.DataFrame, variable_col: str) -> None:
    latest_map = {
        (int(row.patientunitstayid), str(getattr(row, variable_col))): (float(row.event_offset), float(row.value))
        for row in latest.itertuples(index=False)
    }
    for row in grouped.itertuples(index=False):
        stay_id = int(row.patientunitstayid)
        variable = str(getattr(row, variable_col))
        by_var = state.setdefault(stay_id, {})
        slot = by_var.setdefault(variable, empty_slot())
        slot["count"] += int(row.count)
        slot["sum"] += float(row.sum)
        slot["min"] = float(row.min) if slot["min"] is None else min(float(slot["min"]), float(row.min))
        slot["max"] = float(row.max) if slot["max"] is None else max(float(slot["max"]), float(row.max))
        latest_item = latest_map.get((stay_id, variable))
        if latest_item is None:
            continue
        latest_offset, latest_value = latest_item
        if slot["last_offset"] is None or latest_offset >= float(slot["last_offset"]):
            slot["last_offset"] = latest_offset
            slot["last"] = latest_value


def load_eligible_stays(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "eicu_temporal_mortality_cohort.csv"
    usecols = ["stay_id", "patient_id", "split", "eligible_first_24h_prediction", "hospital_mortality"]
    cohort = pd.read_csv(path, usecols=usecols)
    return cohort[cohort["eligible_first_24h_prediction"].astype(bool)].copy()


def build_feature_frame(stays: list[int], variables: list[str], prefix: str, state: dict[int, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for stay_id in stays:
        row: dict[str, Any] = {"stay_id": int(stay_id)}
        by_var = state.get(int(stay_id), {})
        for variable in variables:
            slot = by_var.get(variable)
            base = f"{prefix}_{variable}"
            if slot is None or int(slot["count"]) == 0:
                row[f"{base}_count"] = 0
                row[f"{base}_mean"] = None
                row[f"{base}_min"] = None
                row[f"{base}_max"] = None
                row[f"{base}_last"] = None
                row[f"{base}_has"] = 0
            else:
                count = int(slot["count"])
                row[f"{base}_count"] = count
                row[f"{base}_mean"] = float(slot["sum"]) / count
                row[f"{base}_min"] = slot["min"]
                row[f"{base}_max"] = slot["max"]
                row[f"{base}_last"] = slot["last"]
                row[f"{base}_has"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def availability(features: pd.DataFrame, variables: list[str], prefix: str, total: int) -> pd.DataFrame:
    rows = []
    for variable in variables:
        count_col = f"{prefix}_{variable}_count"
        has_col = f"{prefix}_{variable}_has"
        with_feature = int(features[has_col].sum()) if has_col in features else 0
        rows.append(
            {
                "feature_group": prefix,
                "variable": variable,
                "stays_with_feature": with_feature,
                "eligible_stays": total,
                "coverage": round(with_feature / total, 4) if total else 0,
                "total_measurements": int(features[count_col].sum()) if count_col in features else 0,
            }
        )
    return pd.DataFrame(rows)


def extract_labs(lab_path: Path, eligible_stays: set[int], chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    state: dict[int, dict[str, dict[str, Any]]] = {}
    stats: dict[str, Any] = {
        "source": "lab",
        "chunks": 0,
        "raw_rows_scanned": 0,
        "eligible_stay_rows": 0,
        "window_rows": 0,
        "target_rows": 0,
        "numeric_rows": 0,
        "min_included_offset": None,
        "max_included_offset": None,
    }
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    for chunk in pd.read_csv(lab_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        stats["chunks"] += 1
        stats["raw_rows_scanned"] += int(len(chunk))
        chunk = chunk[chunk["patientunitstayid"].isin(eligible_stays)]
        stats["eligible_stay_rows"] += int(len(chunk))
        if chunk.empty:
            continue
        chunk["event_offset"] = pd.to_numeric(chunk["labresultoffset"], errors="coerce")
        chunk = chunk[chunk["event_offset"].between(WINDOW_START, WINDOW_END, inclusive="both")]
        stats["window_rows"] += int(len(chunk))
        if chunk.empty:
            continue
        chunk["lab_concept"] = chunk["labname"].astype(str).str.lower().str.strip().map(LAB_NAME_MAP)
        chunk = chunk[chunk["lab_concept"].notna()]
        stats["target_rows"] += int(len(chunk))
        if chunk.empty:
            continue
        chunk["value"] = pd.to_numeric(chunk["labresult"], errors="coerce")
        chunk = chunk[chunk["value"].notna()]
        stats["numeric_rows"] += int(len(chunk))
        if chunk.empty:
            continue
        stats["min_included_offset"] = float(chunk["event_offset"].min()) if stats["min_included_offset"] is None else min(float(stats["min_included_offset"]), float(chunk["event_offset"].min()))
        stats["max_included_offset"] = float(chunk["event_offset"].max()) if stats["max_included_offset"] is None else max(float(stats["max_included_offset"]), float(chunk["event_offset"].max()))
        grouped = (
            chunk.groupby(["patientunitstayid", "lab_concept"], sort=False)
            .agg(count=("value", "size"), sum=("value", "sum"), min=("value", "min"), max=("value", "max"))
            .reset_index()
        )
        latest = chunk.sort_values(["patientunitstayid", "lab_concept", "event_offset"]).groupby(["patientunitstayid", "lab_concept"], sort=False).tail(1)
        update_state(state, grouped, latest[["patientunitstayid", "lab_concept", "event_offset", "value"]], "lab_concept")
    stays = sorted(eligible_stays)
    features = build_feature_frame(stays, LABS, "eicu_lab24h", state)
    return features, availability(features, LABS, "eicu_lab24h", len(stays)), stats


def extract_vitals(vital_path: Path, eligible_stays: set[int], chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    state: dict[int, dict[str, dict[str, Any]]] = {}
    stats: dict[str, Any] = {
        "source": "vital",
        "chunks": 0,
        "raw_rows_scanned": 0,
        "eligible_stay_rows": 0,
        "window_rows": 0,
        "numeric_rows": 0,
        "min_included_offset": None,
        "max_included_offset": None,
    }
    usecols = ["patientunitstayid", "observationoffset", *VITAL_COLUMNS.keys()]
    for chunk in pd.read_csv(vital_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        stats["chunks"] += 1
        stats["raw_rows_scanned"] += int(len(chunk))
        chunk = chunk[chunk["patientunitstayid"].isin(eligible_stays)]
        stats["eligible_stay_rows"] += int(len(chunk))
        if chunk.empty:
            continue
        chunk["event_offset"] = pd.to_numeric(chunk["observationoffset"], errors="coerce")
        chunk = chunk[chunk["event_offset"].between(WINDOW_START, WINDOW_END, inclusive="both")]
        stats["window_rows"] += int(len(chunk))
        if chunk.empty:
            continue
        stats["min_included_offset"] = float(chunk["event_offset"].min()) if stats["min_included_offset"] is None else min(float(stats["min_included_offset"]), float(chunk["event_offset"].min()))
        stats["max_included_offset"] = float(chunk["event_offset"].max()) if stats["max_included_offset"] is None else max(float(stats["max_included_offset"]), float(chunk["event_offset"].max()))
        for raw_column, vital in VITAL_COLUMNS.items():
            sub = chunk[["patientunitstayid", "event_offset", raw_column]].copy()
            sub["value"] = pd.to_numeric(sub[raw_column], errors="coerce")
            sub = sub[sub["value"].notna()]
            stats["numeric_rows"] += int(len(sub))
            if sub.empty:
                continue
            sub["vital_concept"] = vital
            grouped = (
                sub.groupby(["patientunitstayid", "vital_concept"], sort=False)
                .agg(count=("value", "size"), sum=("value", "sum"), min=("value", "min"), max=("value", "max"))
                .reset_index()
            )
            latest = sub.sort_values(["patientunitstayid", "vital_concept", "event_offset"]).groupby(["patientunitstayid", "vital_concept"], sort=False).tail(1)
            update_state(state, grouped, latest[["patientunitstayid", "vital_concept", "event_offset", "value"]], "vital_concept")
    stays = sorted(eligible_stays)
    features = build_feature_frame(stays, VITALS, "eicu_vital24h", state)
    return features, availability(features, VITALS, "eicu_vital24h", len(stays)), stats


def markdown_table(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, summary: pd.DataFrame, stats: pd.DataFrame) -> Path:
    report_path = project_root / "outputs" / "reports" / "eicu_temporal_features_24h_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# eICU First-24h Temporal Feature Skeleton

- Boundary: feature extraction only; no model training and no clinical recommendation.
- Cohort restriction: only stays with `eligible_first_24h_prediction = True`.
- Time window: event offset `0 <= offset <= 1440` minutes.
- Lab source: `labresultoffset`.
- Vital source: `observationoffset`.

## Feature Availability

{markdown_table(summary)}

## Extraction Stats

{markdown_table(stats)}

## Interpretation

These files are safe for a first-24h ICU prediction task. They are not admission-time features, because they use data observed after ICU admission.
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    selected_root, _ = choose_root(args.eicu_root)
    if selected_root is None:
        raise SystemExit("No eICU root with required tables was found. Run --eicu-readiness first.")
    lab_path = find_first(selected_root, LAB_FILES)
    vital_path = find_first(selected_root, VITAL_FILES)
    if lab_path is None or vital_path is None:
        raise SystemExit(f"Missing lab or vital table under {selected_root}")

    cohort = load_eligible_stays(args.project_root)
    eligible_stays = set(cohort["stay_id"].astype(int))
    lab_features, lab_availability, lab_stats = extract_labs(lab_path, eligible_stays, args.chunksize)
    vital_features, vital_availability, vital_stats = extract_vitals(vital_path, eligible_stays, args.chunksize)
    combined = cohort[["stay_id", "patient_id", "split", "hospital_mortality"]].merge(lab_features, on="stay_id", how="left").merge(vital_features, on="stay_id", how="left")
    summary = pd.concat([lab_availability, vital_availability], ignore_index=True)
    stats = pd.DataFrame([lab_stats, vital_stats])

    processed = args.project_root / "data" / "processed"
    tables = args.project_root / "outputs" / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    lab_features.to_csv(processed / "eicu_lab_features_24h.csv", index=False)
    vital_features.to_csv(processed / "eicu_vital_features_24h.csv", index=False)
    combined.to_csv(processed / "eicu_first24h_feature_matrix_skeleton.csv", index=False)
    summary.to_csv(tables / "eicu_temporal_features_24h_availability.csv", index=False)
    stats.to_csv(tables / "eicu_temporal_features_24h_extraction_stats.csv", index=False)
    report = write_report(args.project_root, summary, stats)
    print(f"Wrote {report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
