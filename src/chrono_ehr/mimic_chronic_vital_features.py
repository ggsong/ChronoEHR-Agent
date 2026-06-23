#!/usr/bin/env python3
"""Extract first-24h and discharge-time vital-sign features for chronic cohorts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from feature_window_spec_loader import (
    DEFAULT_FEATURE_WINDOW_SPEC,
    load_feature_window_spec,
    vital_itemid_map,
    vital_names,
    vital_plausible_ranges,
)
from mimic_ckd_lab_itemids import DEFAULT_MIMIC_ROOT, DEFAULT_PROJECT


STUDIES = {
    "diabetes": {
        "cohort_path": "data/processed/mimic_diabetes_readmission_cohort.csv",
        "prefix": "vital",
    },
    "ckd": {
        "cohort_path": "data/processed/mimic_ckd_readmission_cohort.csv",
        "prefix": "vital",
    },
    "heart_failure": {
        "cohort_path": "data/processed/mimic_heart_failure_readmission_cohort.csv",
        "prefix": "vital",
    },
    "hypertension": {
        "cohort_path": "data/processed/mimic_hypertension_readmission_cohort.csv",
        "prefix": "vital",
    },
}

VITAL_ITEMIDS = {
    220045: "heart_rate",
    220179: "sbp",
    220180: "dbp",
    220181: "mbp",
    220210: "respiratory_rate",
    220277: "spo2",
    223761: "temperature_c",
    223762: "temperature_c",
}

VITALS = ["heart_rate", "sbp", "dbp", "mbp", "respiratory_rate", "spo2", "temperature_c"]

PLAUSIBLE_RANGES = {
    "heart_rate": (20, 250),
    "sbp": (40, 300),
    "dbp": (20, 200),
    "mbp": (30, 220),
    "respiratory_rate": (4, 80),
    "spo2": (40, 100),
    "temperature_c": (25, 45),
}


def apply_window_spec(path: Path) -> None:
    global VITAL_ITEMIDS, VITALS, PLAUSIBLE_RANGES
    spec = load_feature_window_spec(path)
    VITAL_ITEMIDS = vital_itemid_map(spec)
    VITALS = vital_names(spec)
    PLAUSIBLE_RANGES = vital_plausible_ranges(spec)


def empty_state() -> dict[str, dict[int, dict[str, dict]]]:
    return defaultdict(lambda: defaultdict(dict))


def load_cohort(project_root: Path, cohort: str, relative_path: str) -> pd.DataFrame:
    path = project_root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort file for {cohort}: {path}")
    df = pd.read_csv(path, usecols=["hadm_id", "admittime", "dischtime"], parse_dates=["admittime", "dischtime"])
    df = df.dropna(subset=["hadm_id", "admittime", "dischtime"]).copy()
    df["hadm_id"] = df["hadm_id"].astype(int)
    df["cohort"] = cohort
    df["window_24h_end"] = df["admittime"] + pd.Timedelta(hours=24)
    df["window_24h_end"] = df[["window_24h_end", "dischtime"]].min(axis=1)
    return df[["cohort", "hadm_id", "admittime", "window_24h_end", "dischtime"]]


def load_all_cohorts(project_root: Path) -> pd.DataFrame:
    parts = [load_cohort(project_root, cohort, spec["cohort_path"]) for cohort, spec in STUDIES.items()]
    return pd.concat(parts, ignore_index=True)


def normalize_values(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["vital_name"] = chunk["itemid"].map(VITAL_ITEMIDS)
    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
    fahrenheit = chunk["itemid"].eq(223761)
    chunk.loc[fahrenheit, "valuenum"] = (chunk.loc[fahrenheit, "valuenum"] - 32) * 5 / 9

    keep = pd.Series(True, index=chunk.index)
    for vital, (low, high) in PLAUSIBLE_RANGES.items():
        mask = chunk["vital_name"].eq(vital)
        keep &= ~mask | chunk["valuenum"].between(low, high, inclusive="both")
    return chunk[keep].copy()


def update_state(state: dict[str, dict[int, dict[str, dict]]], grouped: pd.DataFrame, latest: pd.DataFrame) -> None:
    latest_lookup = {
        (str(row.cohort), int(row.hadm_id), str(row.vital_name)): (row.available_time, float(row.valuenum))
        for row in latest.itertuples(index=False)
    }
    for row in grouped.itertuples(index=False):
        cohort = str(row.cohort)
        hadm_id = int(row.hadm_id)
        vital = str(row.vital_name)
        current = state[cohort][hadm_id].get(vital)
        latest_time, latest_value = latest_lookup[(cohort, hadm_id, vital)]
        if current is None:
            state[cohort][hadm_id][vital] = {
                "count": int(row.count),
                "sum": float(row.sum),
                "min": float(row.min),
                "max": float(row.max),
                "warning_count": int(row.warning_count),
                "last_time": latest_time,
                "last": latest_value,
            }
            continue
        current["count"] += int(row.count)
        current["sum"] += float(row.sum)
        current["min"] = min(float(current["min"]), float(row.min))
        current["max"] = max(float(current["max"]), float(row.max))
        current["warning_count"] += int(row.warning_count)
        if pd.isna(current["last_time"]) or latest_time >= current["last_time"]:
            current["last_time"] = latest_time
            current["last"] = latest_value


def state_to_features(state: dict[str, dict[int, dict[str, dict]]], cohort_hadm: dict[str, set[int]], prefix: str) -> dict[str, pd.DataFrame]:
    outputs = {}
    for cohort, hadm_ids in cohort_hadm.items():
        rows = []
        cohort_state = state.get(cohort, {})
        for hadm_id in sorted(hadm_ids):
            row = {"hadm_id": int(hadm_id)}
            by_vital = cohort_state.get(int(hadm_id), {})
            for vital in VITALS:
                slot = by_vital.get(vital)
                col = f"{prefix}_{vital}"
                if slot is None:
                    row[f"{col}_count"] = 0
                    row[f"{col}_mean"] = None
                    row[f"{col}_min"] = None
                    row[f"{col}_max"] = None
                    row[f"{col}_last"] = None
                    row[f"{col}_warning_count"] = 0
                    row[f"{col}_has"] = 0
                else:
                    count = int(slot["count"])
                    row[f"{col}_count"] = count
                    row[f"{col}_mean"] = float(slot["sum"]) / count if count else None
                    row[f"{col}_min"] = slot["min"]
                    row[f"{col}_max"] = slot["max"]
                    row[f"{col}_last"] = slot["last"]
                    row[f"{col}_warning_count"] = int(slot["warning_count"])
                    row[f"{col}_has"] = int(count > 0)
            rows.append(row)
        outputs[cohort] = pd.DataFrame(rows)
    return outputs


def summarize_availability(features: pd.DataFrame, cohort: str, prefix: str, window: str) -> pd.DataFrame:
    rows = []
    total = len(features)
    for vital in VITALS:
        has_col = f"{prefix}_{vital}_has"
        count_col = f"{prefix}_{vital}_count"
        rows.append(
            {
                "cohort": cohort,
                "window": window,
                "vital": vital,
                "hadm_with_vital": int(features[has_col].sum()),
                "hadm_with_vital_percent": float(features[has_col].mean()) if total else 0.0,
                "total_measurements": int(features[count_col].sum()),
                "median_measurements_among_all": float(features[count_col].median()) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def process_chartevents(
    mimic_root: Path,
    all_cohorts: pd.DataFrame,
    chunksize: int,
    max_chunks: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    chartevents_path = mimic_root / "icu" / "chartevents.csv.gz"
    cohort_windows = all_cohorts.copy()
    cohort_hadm = {
        cohort: set(group["hadm_id"].astype(int))
        for cohort, group in cohort_windows.groupby("cohort", sort=False)
    }
    union_hadm = set(cohort_windows["hadm_id"].astype(int))

    state_24h = empty_state()
    state_discharge = empty_state()
    stats = {
        "chunks": 0,
        "raw_rows_scanned": 0,
        "target_item_rows": 0,
        "cohort_hadm_rows": 0,
        "numeric_rows": 0,
        "plausible_rows": 0,
        "time_window_24h_rows": 0,
        "time_window_discharge_rows": 0,
    }

    usecols = ["hadm_id", "charttime", "storetime", "itemid", "valuenum", "warning"]
    for chunk in pd.read_csv(
        chartevents_path,
        compression="gzip",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        stats["chunks"] += 1
        stats["raw_rows_scanned"] += int(len(chunk))

        chunk = chunk[chunk["itemid"].isin(VITAL_ITEMIDS)]
        stats["target_item_rows"] += int(len(chunk))
        if chunk.empty:
            if max_chunks and stats["chunks"] >= max_chunks:
                break
            continue

        chunk = chunk[chunk["hadm_id"].notna()].copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(union_hadm)]
        stats["cohort_hadm_rows"] += int(len(chunk))
        if chunk.empty:
            if max_chunks and stats["chunks"] >= max_chunks:
                break
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="coerce")
        chunk["available_time"] = chunk["storetime"].where(chunk["storetime"].notna(), chunk["charttime"])
        chunk = normalize_values(chunk)
        chunk["warning"] = pd.to_numeric(chunk["warning"], errors="coerce").fillna(0).astype(int)
        chunk = chunk[chunk["available_time"].notna() & chunk["valuenum"].notna()].copy()
        stats["numeric_rows"] += int(len(chunk))
        if chunk.empty:
            if max_chunks and stats["chunks"] >= max_chunks:
                break
            continue

        chunk = chunk.merge(cohort_windows, on="hadm_id", how="inner")
        chunk = chunk[chunk["available_time"] >= chunk["admittime"]].copy()
        stats["plausible_rows"] += int(len(chunk))
        if chunk.empty:
            if max_chunks and stats["chunks"] >= max_chunks:
                break
            continue

        for state, end_col, stat_key in [
            (state_24h, "window_24h_end", "time_window_24h_rows"),
            (state_discharge, "dischtime", "time_window_discharge_rows"),
        ]:
            part = chunk[chunk["available_time"] <= chunk[end_col]].copy()
            stats[stat_key] += int(len(part))
            if part.empty:
                continue
            grouped = (
                part.groupby(["cohort", "hadm_id", "vital_name"], sort=False)
                .agg(
                    count=("valuenum", "size"),
                    sum=("valuenum", "sum"),
                    min=("valuenum", "min"),
                    max=("valuenum", "max"),
                    warning_count=("warning", "sum"),
                )
                .reset_index()
            )
            latest = (
                part.sort_values(["cohort", "hadm_id", "vital_name", "available_time"])
                .groupby(["cohort", "hadm_id", "vital_name"], sort=False)
                .tail(1)[["cohort", "hadm_id", "vital_name", "available_time", "valuenum"]]
            )
            update_state(state, grouped, latest)

        if max_chunks and stats["chunks"] >= max_chunks:
            break

    features_24h = state_to_features(state_24h, cohort_hadm, "vital24h")
    features_discharge = state_to_features(state_discharge, cohort_hadm, "vitaldischarge")
    availability = []
    for cohort in STUDIES:
        availability.append(summarize_availability(features_24h[cohort], cohort, "vital24h", "first_24h"))
        availability.append(
            summarize_availability(features_discharge[cohort], cohort, "vitaldischarge", "admission_to_discharge")
        )
    return features_24h, features_discharge, pd.concat(availability, ignore_index=True), stats


def write_report(stats: dict[str, int], availability: pd.DataFrame, output: Path) -> None:
    lines = [
        "| Cohort | Window | Vital | HADM with vital | Percent | Total measurements | Median measurements |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in availability.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.window} | {row.vital} | {int(row.hadm_with_vital)} | "
            f"{row.hadm_with_vital_percent:.2%} | {int(row.total_measurements)} | "
            f"{row.median_measurements_among_all:.1f} |"
        )

    item_lines = [
        "| Vital | Item IDs | Notes |",
        "|---|---|---|",
        "| heart_rate | 220045 | Heart Rate |",
        "| sbp | 220179 | Non-invasive systolic blood pressure |",
        "| dbp | 220180 | Non-invasive diastolic blood pressure |",
        "| mbp | 220181 | Non-invasive mean blood pressure |",
        "| respiratory_rate | 220210 | Respiratory Rate |",
        "| spo2 | 220277 | O2 saturation pulseoxymetry |",
        "| temperature_c | 223761, 223762 | Fahrenheit values converted to Celsius |",
    ]

    text = f"""# MIMIC Chronic Disease Vital-Sign Feature Extraction Report

## Purpose

这个报告记录四个慢病队列的 ICU charted vital-sign features。它服务于 ChronoEHR-Agent 的时间点感知建模：同一个 vital sign 在 first-24h 和 admission-to-discharge 两个窗口内分别聚合，避免把出院前信息错误用于入院时预测。

## Time Windows

- First 24h: `admittime <= available_time <= min(admittime + 24h, dischtime)`
- Discharge-safe: `admittime <= available_time <= dischtime`
- Available time: `storetime` when present, otherwise `charttime`

## Included Vital Signs

{chr(10).join(item_lines)}

## Scan Stats

- chunks: {stats["chunks"]}
- raw chartevents rows scanned: {stats["raw_rows_scanned"]}
- target item rows: {stats["target_item_rows"]}
- chronic cohort HADM rows: {stats["cohort_hadm_rows"]}
- numeric rows: {stats["numeric_rows"]}
- plausible rows after admission: {stats["plausible_rows"]}
- first-24h rows: {stats["time_window_24h_rows"]}
- admission-to-discharge rows: {stats["time_window_discharge_rows"]}

## Availability

{chr(10).join(lines)}

## Leakage Note

The `vital24h_*` files are only safe for a 24-hour in-hospital prediction time or later. The `vitaldischarge_*` files are safe for discharge-time prediction and must not be reused for admission-time prediction.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--window-spec", type=Path, default=DEFAULT_FEATURE_WINDOW_SPEC)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-chunks", type=int, help="Development-only limit for testing the extractor.")
    parser.add_argument("--dry-run", action="store_true", help="Scan limited chunks and print stats without writing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_chunks is not None and not args.dry_run:
        raise SystemExit("--max-chunks is only allowed with --dry-run so partial scans cannot overwrite full outputs.")
    apply_window_spec(args.window_spec)
    processed = args.project_root / "data" / "processed"
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    all_cohorts = load_all_cohorts(args.project_root)
    features_24h, features_discharge, availability, stats = process_chartevents(
        args.mimic_root,
        all_cohorts,
        args.chunksize,
        args.max_chunks,
    )
    if args.dry_run:
        print("MIMIC chronic disease vital dry run complete")
        print(f"chunks={stats['chunks']}")
        print(f"first_24h_rows={stats['time_window_24h_rows']}")
        print(f"discharge_rows={stats['time_window_discharge_rows']}")
        return
    for cohort in STUDIES:
        features_24h[cohort].to_csv(processed / f"mimic_{cohort}_vital_features_24h.csv", index=False)
        features_discharge[cohort].to_csv(processed / f"mimic_{cohort}_vital_features_discharge.csv", index=False)
    availability.to_csv(tables / "chronic_disease_vital_feature_availability.csv", index=False)
    pd.DataFrame([stats]).to_csv(tables / "chronic_disease_vital_extraction_stats.csv", index=False)
    write_report(stats, availability, reports / "chronic_disease_vital_feature_report.md")

    print("MIMIC chronic disease vital features extracted")
    print(f"chunks={stats['chunks']}")
    print(f"first_24h_rows={stats['time_window_24h_rows']}")
    print(f"discharge_rows={stats['time_window_discharge_rows']}")


if __name__ == "__main__":
    main()
