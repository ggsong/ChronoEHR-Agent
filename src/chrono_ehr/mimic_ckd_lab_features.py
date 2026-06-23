#!/usr/bin/env python3
"""Extract CKD first-24h and discharge-time lab features in one labevents pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_window_spec_loader import (
    DEFAULT_FEATURE_WINDOW_SPEC,
    add_window_end,
    available_time_from_source,
    load_feature_window_spec,
    source_spec,
)
from mimic_diabetes_lab_features import empty_state, update_state
from mimic_ckd_lab_itemids import DEFAULT_MIMIC_ROOT, DEFAULT_PROJECT, FIRST_PASS_ITEMIDS


LABS = ["creatinine", "bun", "potassium", "sodium", "bicarbonate", "hemoglobin"]


def target_itemids() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for lab in LABS:
        for itemid in FIRST_PASS_ITEMIDS[lab]:
            mapping[int(itemid)] = lab
    return mapping


def load_cohort(project_root: Path, window_spec: dict | None = None) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "mimic_ckd_readmission_cohort.csv"
    usecols = ["hadm_id", "admittime", "dischtime"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["admittime", "dischtime"])
    df = add_window_end(df, "first_24h", output_col="window_24h_end", spec=window_spec)
    df = add_window_end(df, "admission_to_discharge", output_col="window_discharge_end", spec=window_spec)
    return df


def state_to_features(state: dict[int, dict[str, dict]], cohort_hadm: set[int], prefix: str) -> pd.DataFrame:
    rows = []
    for hadm_id in sorted(cohort_hadm):
        row = {"hadm_id": int(hadm_id)}
        by_lab = state.get(int(hadm_id), {})
        for lab in LABS:
            slot = by_lab.get(lab)
            col = f"{prefix}_{lab}"
            if slot is None:
                row[f"{col}_count"] = 0
                row[f"{col}_mean"] = None
                row[f"{col}_min"] = None
                row[f"{col}_max"] = None
                row[f"{col}_last"] = None
                row[f"{col}_abnormal_count"] = 0
                row[f"{col}_has"] = 0
            else:
                count = int(slot["count"])
                row[f"{col}_count"] = count
                row[f"{col}_mean"] = float(slot["sum"]) / count if count else None
                row[f"{col}_min"] = slot["min"]
                row[f"{col}_max"] = slot["max"]
                row[f"{col}_last"] = slot["last"]
                row[f"{col}_abnormal_count"] = int(slot["abnormal_count"])
                row[f"{col}_has"] = int(count > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_availability(features: pd.DataFrame, prefix: str, window: str) -> pd.DataFrame:
    rows = []
    total = len(features)
    for lab in LABS:
        has_col = f"{prefix}_{lab}_has"
        count_col = f"{prefix}_{lab}_count"
        rows.append(
            {
                "window": window,
                "lab": lab,
                "hadm_with_lab": int(features[has_col].sum()),
                "hadm_with_lab_percent": float(features[has_col].mean()) if total else 0.0,
                "total_measurements": int(features[count_col].sum()),
                "median_measurements_among_all": float(features[count_col].median()),
            }
        )
    return pd.DataFrame(rows)


def process_labevents(
    mimic_root: Path,
    cohort: pd.DataFrame,
    window_spec: dict | None = None,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    window_spec = window_spec or load_feature_window_spec()
    lab_source = source_spec(window_spec, "labs")
    labevents_path = mimic_root / "hosp" / "labevents.csv.gz"
    cohort_times = cohort.set_index("hadm_id")[["admittime", "window_24h_end", "window_discharge_end"]]
    cohort_hadm = set(cohort_times.index.astype(int))
    itemid_map = target_itemids()

    state_24h = empty_state()
    state_discharge = empty_state()
    stats = {
        "chunks": 0,
        "raw_rows_scanned": 0,
        "target_item_rows": 0,
        "cohort_hadm_rows": 0,
        "numeric_rows": 0,
        "time_window_24h_rows": 0,
        "time_window_discharge_rows": 0,
    }

    usecols = ["hadm_id", "itemid", "charttime", "storetime", "valuenum", "flag"]
    for chunk in pd.read_csv(
        labevents_path,
        compression="gzip",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        stats["chunks"] += 1
        stats["raw_rows_scanned"] += int(len(chunk))

        chunk = chunk[chunk["itemid"].isin(itemid_map)]
        stats["target_item_rows"] += int(len(chunk))
        if chunk.empty:
            continue

        chunk = chunk[chunk["hadm_id"].notna()]
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(cohort_hadm)]
        stats["cohort_hadm_rows"] += int(len(chunk))
        if chunk.empty:
            continue

        chunk = chunk.merge(cohort_times, left_on="hadm_id", right_index=True, how="left")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="coerce")
        chunk["available_time"] = available_time_from_source(chunk, lab_source)
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk["lab_name"] = chunk["itemid"].map(itemid_map)

        base = chunk["available_time"].notna() & chunk["valuenum"].notna() & (chunk["available_time"] >= chunk["admittime"])
        chunk = chunk[base].copy()
        stats["numeric_rows"] += int(len(chunk))
        if chunk.empty:
            continue

        flag = chunk["flag"].astype("string").fillna("").str.lower()
        chunk["is_abnormal"] = flag.ne("").astype(int)

        for state, end_col, stat_key in [
            (state_24h, "window_24h_end", "time_window_24h_rows"),
            (state_discharge, "window_discharge_end", "time_window_discharge_rows"),
        ]:
            part = chunk[chunk["available_time"] <= chunk[end_col]].copy()
            stats[stat_key] += int(len(part))
            if part.empty:
                continue
            grouped = (
                part.groupby(["hadm_id", "lab_name"], sort=False)
                .agg(
                    count=("valuenum", "size"),
                    sum=("valuenum", "sum"),
                    min=("valuenum", "min"),
                    max=("valuenum", "max"),
                    abnormal_count=("is_abnormal", "sum"),
                )
                .reset_index()
            )
            latest = (
                part.sort_values(["hadm_id", "lab_name", "available_time"])
                .groupby(["hadm_id", "lab_name"], sort=False)
                .tail(1)[["hadm_id", "lab_name", "available_time", "valuenum"]]
            )
            update_state(state, grouped, latest)

    features_24h = state_to_features(state_24h, cohort_hadm, "ckdlab24h")
    features_discharge = state_to_features(state_discharge, cohort_hadm, "ckdlabdischarge")
    availability = pd.concat(
        [
            summarize_availability(features_24h, "ckdlab24h", "first_24h"),
            summarize_availability(features_discharge, "ckdlabdischarge", "admission_to_discharge"),
        ],
        ignore_index=True,
    )
    return features_24h, features_discharge, availability, stats


def write_report(stats: dict[str, int], availability: pd.DataFrame, report_path: Path) -> None:
    lines = [
        "| Window | Lab | HADM with lab | Percent | Total measurements | Median measurements |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in availability.itertuples(index=False):
        lines.append(
            f"| {row.window} | {row.lab} | {int(row.hadm_with_lab)} | {row.hadm_with_lab_percent:.2%} | "
            f"{int(row.total_measurements)} | {row.median_measurements_among_all:.1f} |"
        )
    text = f"""# MIMIC CKD Lab Feature Extraction Report

## Time Windows

- First 24h: `admittime <= available_time <= min(admittime + 24h, dischtime)`
- Discharge-safe: `admittime <= available_time <= dischtime`
- Available time: `storetime` when present, otherwise `charttime`
- Window config: `configs/feature_window_specs.json`

## Scan Stats

- chunks: {stats["chunks"]}
- raw labevents rows scanned: {stats["raw_rows_scanned"]}
- target item rows: {stats["target_item_rows"]}
- CKD cohort HADM rows: {stats["cohort_hadm_rows"]}
- numeric rows after admission: {stats["numeric_rows"]}
- first-24h rows: {stats["time_window_24h_rows"]}
- admission-to-discharge rows: {stats["time_window_discharge_rows"]}

## Availability

{chr(10).join(lines)}
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--window-spec", type=Path, default=DEFAULT_FEATURE_WINDOW_SPEC)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = args.project_root / "data" / "processed"
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    window_spec = load_feature_window_spec(args.window_spec)
    cohort = load_cohort(args.project_root, window_spec)
    features_24h, features_discharge, availability, stats = process_labevents(args.mimic_root, cohort, window_spec, args.chunksize)
    features_24h.to_csv(processed / "mimic_ckd_lab_features_24h.csv", index=False)
    features_discharge.to_csv(processed / "mimic_ckd_lab_features_discharge.csv", index=False)
    availability.to_csv(tables / "mimic_ckd_lab_feature_availability.csv", index=False)
    pd.DataFrame([stats]).to_csv(tables / "mimic_ckd_lab_extraction_stats.csv", index=False)
    write_report(stats, availability, reports / "mimic_ckd_lab_feature_report.md")

    print("MIMIC CKD lab features extracted")
    print(f"first_24h_rows={stats['time_window_24h_rows']}")
    print(f"discharge_rows={stats['time_window_discharge_rows']}")


if __name__ == "__main__":
    main()
