#!/usr/bin/env python3
"""Extract first-24h and discharge-time ICU procedure-event features for chronic cohorts."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from feature_window_spec_loader import (
    DEFAULT_FEATURE_WINDOW_SPEC,
    add_window_end,
    available_time_from_source,
    load_feature_window_spec,
    source_spec,
)
from mimic_ckd_lab_itemids import DEFAULT_MIMIC_ROOT, DEFAULT_PROJECT


STUDIES = {
    "diabetes": "data/processed/mimic_diabetes_readmission_cohort.csv",
    "ckd": "data/processed/mimic_ckd_readmission_cohort.csv",
    "heart_failure": "data/processed/mimic_heart_failure_readmission_cohort.csv",
    "hypertension": "data/processed/mimic_hypertension_readmission_cohort.csv",
}

PROCEDURE_GROUPS = [
    "total",
    "peripheral_lines",
    "imaging",
    "invasive_lines",
    "procedures",
    "cultures",
    "ventilation",
    "intubation_extubation",
    "communication",
    "significant_events",
    "gi_gu",
    "dialysis",
    "medications",
    "other",
]


def normalize_group(category: str | float | None) -> str:
    text = "" if category is None or pd.isna(category) else str(category).strip().lower()
    mapping = {
        "access lines - peripheral": "peripheral_lines",
        "5-imaging": "imaging",
        "access lines - invasive": "invasive_lines",
        "4-procedures": "procedures",
        "6-cultures": "cultures",
        "2-ventilation": "ventilation",
        "1-intubation/extubation": "intubation_extubation",
        "7-communication": "communication",
        "3-significant events": "significant_events",
        "gi/gu": "gi_gu",
        "dialysis": "dialysis",
        "medications": "medications",
    }
    return mapping.get(text, "other")


def load_item_categories(mimic_root: Path) -> dict[int, str]:
    path = mimic_root / "icu" / "d_items.csv.gz"
    df = pd.read_csv(path, usecols=["itemid", "category"])
    return {int(row.itemid): normalize_group(row.category) for row in df.itertuples(index=False)}


def load_cohort(project_root: Path, cohort: str, relative_path: str, window_spec: dict) -> pd.DataFrame:
    path = project_root / relative_path
    usecols = ["hadm_id", "admittime", "dischtime"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["admittime", "dischtime"]).dropna()
    df["hadm_id"] = df["hadm_id"].astype(int)
    df["cohort"] = cohort
    df = add_window_end(df, "first_24h", output_col="window_24h_end", spec=window_spec)
    df = add_window_end(df, "admission_to_discharge", output_col="window_discharge_end", spec=window_spec)
    return df[["cohort", "hadm_id", "admittime", "window_24h_end", "window_discharge_end"]]


def load_all_cohorts(project_root: Path, window_spec: dict) -> pd.DataFrame:
    parts = [load_cohort(project_root, cohort, path, window_spec) for cohort, path in STUDIES.items()]
    return pd.concat(parts, ignore_index=True)


def empty_state() -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "minutes": 0.0})))


def update_state(state: dict[str, dict[int, dict[str, dict[str, float]]]], part: pd.DataFrame) -> None:
    grouped = (
        part.groupby(["cohort", "hadm_id", "procedure_group"], sort=False)
        .agg(count=("itemid", "size"), minutes=("duration_minutes", "sum"))
        .reset_index()
    )
    total = (
        part.groupby(["cohort", "hadm_id"], sort=False)
        .agg(count=("itemid", "size"), minutes=("duration_minutes", "sum"))
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        slot = state[str(row.cohort)][int(row.hadm_id)][str(row.procedure_group)]
        slot["count"] += int(row.count)
        slot["minutes"] += float(row.minutes)
    for row in total.itertuples(index=False):
        slot = state[str(row.cohort)][int(row.hadm_id)]["total"]
        slot["count"] += int(row.count)
        slot["minutes"] += float(row.minutes)


def state_to_features(state: dict[str, dict[int, dict[str, dict[str, float]]]], cohort_hadm: dict[str, set[int]], prefix: str) -> dict[str, pd.DataFrame]:
    outputs = {}
    for cohort, hadm_ids in cohort_hadm.items():
        rows = []
        cohort_state = state.get(cohort, {})
        for hadm_id in sorted(hadm_ids):
            row = {"hadm_id": int(hadm_id)}
            by_group = cohort_state.get(int(hadm_id), {})
            for group in PROCEDURE_GROUPS:
                slot = by_group.get(group, {"count": 0, "minutes": 0.0})
                col = f"{prefix}_{group}"
                count = int(slot["count"])
                row[f"{col}_count"] = count
                row[f"{col}_has"] = int(count > 0)
                row[f"{col}_minutes"] = float(slot["minutes"])
            rows.append(row)
        outputs[cohort] = pd.DataFrame(rows)
    return outputs


def summarize_availability(features: pd.DataFrame, cohort: str, prefix: str, window: str) -> pd.DataFrame:
    rows = []
    total = len(features)
    for group in PROCEDURE_GROUPS:
        count_col = f"{prefix}_{group}_count"
        has_col = f"{prefix}_{group}_has"
        minutes_col = f"{prefix}_{group}_minutes"
        rows.append(
            {
                "cohort": cohort,
                "window": window,
                "procedure_group": group,
                "hadm_with_group": int(features[has_col].sum()),
                "hadm_with_group_percent": float(features[has_col].mean()) if total else 0.0,
                "total_events": int(features[count_col].sum()),
                "total_minutes": float(features[minutes_col].sum()),
                "median_events_among_all": float(features[count_col].median()) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def process_procedureevents(
    mimic_root: Path,
    cohort_windows: pd.DataFrame,
    window_spec: dict,
    chunksize: int,
    max_chunks: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    procedure_path = mimic_root / "icu" / "procedureevents.csv.gz"
    proc_source = source_spec(window_spec, "procedure_events")
    category_map = load_item_categories(mimic_root)
    cohort_hadm = {cohort: set(group["hadm_id"].astype(int)) for cohort, group in cohort_windows.groupby("cohort", sort=False)}
    union_hadm = set(cohort_windows["hadm_id"].astype(int))

    state_24h = empty_state()
    state_discharge = empty_state()
    stats = {
        "chunks": 0,
        "raw_rows_scanned": 0,
        "cohort_hadm_rows": 0,
        "time_window_24h_rows": 0,
        "time_window_discharge_rows": 0,
    }

    usecols = ["hadm_id", "starttime", "endtime", "storetime", "itemid", "value", "valueuom"]
    for chunk in pd.read_csv(procedure_path, compression="gzip", usecols=usecols, chunksize=chunksize, low_memory=False):
        stats["chunks"] += 1
        stats["raw_rows_scanned"] += int(len(chunk))
        chunk = chunk[chunk["hadm_id"].notna()].copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(union_hadm)]
        stats["cohort_hadm_rows"] += int(len(chunk))
        if chunk.empty:
            if max_chunks and stats["chunks"] >= max_chunks:
                break
            continue

        chunk = chunk.merge(cohort_windows, on="hadm_id", how="inner")
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["endtime"] = pd.to_datetime(chunk["endtime"], errors="coerce")
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="coerce")
        chunk["available_time"] = available_time_from_source(chunk, proc_source)
        chunk["procedure_group"] = chunk["itemid"].map(category_map).fillna("other")
        duration = (chunk["endtime"] - chunk["starttime"]).dt.total_seconds() / 60
        value = pd.to_numeric(chunk["value"], errors="coerce")
        value_is_minutes = chunk["valueuom"].astype("string").str.lower().eq("min")
        chunk["duration_minutes"] = duration.where(duration.notna(), value.where(value_is_minutes, 0)).fillna(0)
        chunk["duration_minutes"] = chunk["duration_minutes"].clip(lower=0, upper=30 * 24 * 60)

        base = chunk["available_time"].notna() & (chunk["available_time"] >= chunk["admittime"])
        chunk = chunk[base].copy()
        if chunk.empty:
            if max_chunks and stats["chunks"] >= max_chunks:
                break
            continue

        part_24h = chunk[chunk["available_time"] <= chunk["window_24h_end"]].copy()
        stats["time_window_24h_rows"] += int(len(part_24h))
        if not part_24h.empty:
            update_state(state_24h, part_24h)

        part_discharge = chunk[chunk["available_time"] <= chunk["window_discharge_end"]].copy()
        stats["time_window_discharge_rows"] += int(len(part_discharge))
        if not part_discharge.empty:
            update_state(state_discharge, part_discharge)

        if max_chunks and stats["chunks"] >= max_chunks:
            break

    features_24h = state_to_features(state_24h, cohort_hadm, "proc24h")
    features_discharge = state_to_features(state_discharge, cohort_hadm, "procdischarge")
    availability = []
    for cohort in STUDIES:
        availability.append(summarize_availability(features_24h[cohort], cohort, "proc24h", "first_24h"))
        availability.append(summarize_availability(features_discharge[cohort], cohort, "procdischarge", "admission_to_discharge"))
    return features_24h, features_discharge, pd.concat(availability, ignore_index=True), stats


def write_report(stats: dict[str, int], availability: pd.DataFrame, output: Path) -> None:
    rows = [
        "| Cohort | Window | Group | HADM with group | Percent | Total events | Total minutes | Median events |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in availability.itertuples(index=False):
        rows.append(
            f"| {row.cohort} | {row.window} | {row.procedure_group} | {int(row.hadm_with_group)} | "
            f"{row.hadm_with_group_percent:.2%} | {int(row.total_events)} | {row.total_minutes:.1f} | "
            f"{row.median_events_among_all:.1f} |"
        )
    text = f"""# MIMIC Chronic Disease ICU Procedure Feature Extraction Report

## Purpose

这个报告记录四个慢病队列的 ICU procedure-event features。Procedure events 来自 `icu/procedureevents.csv.gz`，具有 start/store/end time，因此可以按 first-24h 和 admission-to-discharge 两个时间窗抽取。

## Time Windows

- First 24h: `admittime <= available_time <= min(admittime + 24h, dischtime)`
- Discharge-safe: `admittime <= available_time <= dischtime`
- Available time: `starttime` when present, otherwise `storetime`
- Window config: `configs/feature_window_specs.json`

## Scan Stats

- chunks: {stats["chunks"]}
- raw procedureevents rows scanned: {stats["raw_rows_scanned"]}
- chronic cohort HADM rows: {stats["cohort_hadm_rows"]}
- first-24h rows: {stats["time_window_24h_rows"]}
- admission-to-discharge rows: {stats["time_window_discharge_rows"]}

## Availability

{chr(10).join(rows)}

## Leakage Note

The `proc24h_*` files are safe for a 24-hour in-hospital prediction time or later. The `procdischarge_*` files are safe only for discharge-time prediction and must not be reused for admission-time or 24-hour prediction.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--window-spec", type=Path, default=DEFAULT_FEATURE_WINDOW_SPEC)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_chunks is not None and not args.dry_run:
        raise SystemExit("--max-chunks is only allowed with --dry-run so partial scans cannot overwrite full outputs.")
    window_spec = load_feature_window_spec(args.window_spec)
    processed = args.project_root / "data" / "processed"
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    cohort_windows = load_all_cohorts(args.project_root, window_spec)
    features_24h, features_discharge, availability, stats = process_procedureevents(
        args.mimic_root,
        cohort_windows,
        window_spec,
        args.chunksize,
        args.max_chunks,
    )
    if args.dry_run:
        print("MIMIC chronic disease procedure dry run complete")
        print(f"chunks={stats['chunks']}")
        print(f"first_24h_rows={stats['time_window_24h_rows']}")
        print(f"discharge_rows={stats['time_window_discharge_rows']}")
        return

    for cohort in STUDIES:
        features_24h[cohort].to_csv(processed / f"mimic_{cohort}_procedure_features_24h.csv", index=False)
        features_discharge[cohort].to_csv(processed / f"mimic_{cohort}_procedure_features_discharge.csv", index=False)
    availability.to_csv(tables / "chronic_disease_procedure_feature_availability.csv", index=False)
    pd.DataFrame([stats]).to_csv(tables / "chronic_disease_procedure_extraction_stats.csv", index=False)
    write_report(stats, availability, reports / "chronic_disease_procedure_feature_report.md")

    print("MIMIC chronic disease procedure features extracted")
    print(f"chunks={stats['chunks']}")
    print(f"first_24h_rows={stats['time_window_24h_rows']}")
    print(f"discharge_rows={stats['time_window_discharge_rows']}")


if __name__ == "__main__":
    main()
