#!/usr/bin/env python3
"""Extract first-24h lab features for the MIMIC diabetes cohort."""

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
from mimic_diabetes_lab_features import (
    DEFAULT_MIMIC_ROOT,
    DEFAULT_PROJECT,
    LABS,
    TARGET_ITEMIDS,
    empty_state,
    get_slot,
    summarize_lab_availability,
    update_state,
)


def load_cohort(project_root: Path, window_spec: dict | None = None) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "mimic_diabetes_readmission_cohort.csv"
    usecols = ["hadm_id", "admittime", "dischtime"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["admittime", "dischtime"])
    return add_window_end(df, "first_24h", spec=window_spec)


def rename_lab24h_columns(features: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in features.columns:
        if column.startswith("lab_"):
            renamed[column] = "lab24h_" + column.removeprefix("lab_")
    return features.rename(columns=renamed)


def process_labevents(
    mimic_root: Path,
    cohort: pd.DataFrame,
    window_spec: dict | None = None,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    window_spec = window_spec or load_feature_window_spec()
    lab_source = source_spec(window_spec, "labs")
    labevents_path = mimic_root / "hosp" / "labevents.csv.gz"
    cohort_times = cohort.set_index("hadm_id")[["admittime", "window_end"]]
    cohort_hadm = set(cohort_times.index.astype(int))
    target_itemids = set(TARGET_ITEMIDS)

    state = empty_state()
    stats = {
        "chunks": 0,
        "raw_rows_scanned": 0,
        "target_item_rows": 0,
        "cohort_hadm_rows": 0,
        "time_window_rows": 0,
        "numeric_rows": 0,
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

        chunk = chunk[chunk["itemid"].isin(target_itemids)]
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
        chunk["lab_name"] = chunk["itemid"].map(TARGET_ITEMIDS)

        in_window = (
            chunk["available_time"].notna()
            & chunk["valuenum"].notna()
            & (chunk["available_time"] >= chunk["admittime"])
            & (chunk["available_time"] <= chunk["window_end"])
        )
        chunk = chunk[in_window]
        stats["time_window_rows"] += int(len(chunk))
        stats["numeric_rows"] += int(len(chunk))
        if chunk.empty:
            continue

        flag = chunk["flag"].astype("string").fillna("").str.lower()
        chunk["is_abnormal"] = flag.ne("").astype(int)

        grouped = (
            chunk.groupby(["hadm_id", "lab_name"], sort=False)
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
            chunk.sort_values(["hadm_id", "lab_name", "available_time"])
            .groupby(["hadm_id", "lab_name"], sort=False)
            .tail(1)[["hadm_id", "lab_name", "available_time", "valuenum"]]
        )
        update_state(state, grouped, latest)

    rows = []
    for hadm_id in sorted(cohort_hadm):
        row = {"hadm_id": int(hadm_id)}
        by_lab = state.get(int(hadm_id), {})
        for lab in LABS:
            slot = by_lab.get(lab)
            prefix = f"lab_{lab}"
            if slot is None:
                row[f"{prefix}_count"] = 0
                row[f"{prefix}_mean"] = None
                row[f"{prefix}_min"] = None
                row[f"{prefix}_max"] = None
                row[f"{prefix}_last"] = None
                row[f"{prefix}_abnormal_count"] = 0
                row[f"{prefix}_has"] = 0
            else:
                count = int(slot["count"])
                row[f"{prefix}_count"] = count
                row[f"{prefix}_mean"] = float(slot["sum"]) / count if count else None
                row[f"{prefix}_min"] = slot["min"]
                row[f"{prefix}_max"] = slot["max"]
                row[f"{prefix}_last"] = slot["last"]
                row[f"{prefix}_abnormal_count"] = int(slot["abnormal_count"])
                row[f"{prefix}_has"] = int(count > 0)
        rows.append(row)

    return rename_lab24h_columns(pd.DataFrame(rows)), stats


def summarize_lab24h_availability(features: pd.DataFrame) -> pd.DataFrame:
    normalized = features.rename(columns={column: column.replace("lab24h_", "lab_") for column in features.columns})
    availability = summarize_lab_availability(normalized)
    availability.insert(0, "window", "first_24h")
    return availability


def write_report(stats: dict[str, int], availability: pd.DataFrame, report_path: Path) -> None:
    lines = [
        "| Lab | HADM with lab | Percent | Total measurements | Median measurements |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in availability.itertuples(index=False):
        lines.append(
            f"| {row.lab} | {int(row.hadm_with_lab)} | {row.hadm_with_lab_percent:.2%} | "
            f"{int(row.total_measurements)} | {row.median_measurements_among_all:.1f} |"
        )
    text = f"""# MIMIC 糖尿病入院后 24 小时化验特征抽取报告

## 时间点规则

- 预测时间点：`admittime + 24h`
- 化验可用时间：优先使用 `storetime`，缺失时使用 `charttime`
- 纳入化验：`admittime <= available_time <= min(admittime + 24h, dischtime)`
- 时间窗配置：`configs/feature_window_specs.json` 中的 `first_24h`
- 只抽取 index admission 内、目标 itemid 的数值型化验

## 扫描统计

- chunks：{stats["chunks"]}
- 原始 labevents 行数：{stats["raw_rows_scanned"]}
- 目标 itemid 行数：{stats["target_item_rows"]}
- 属于糖尿病 cohort hadm 的行数：{stats["cohort_hadm_rows"]}
- 通过前 24 小时时间窗的数值化验行数：{stats["time_window_rows"]}

## 化验覆盖

{chr(10).join(lines)}

## 说明

这些特征用于 inhospital 24h prediction。它们不能用于真正的 admission-time prediction，因为 admission-time prediction 不能预先知道入院后 24 小时内的化验结果。
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
    processed_dir = args.project_root / "data" / "processed"
    tables_dir = args.project_root / "outputs" / "tables"
    reports_dir = args.project_root / "outputs" / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    window_spec = load_feature_window_spec(args.window_spec)
    cohort = load_cohort(args.project_root, window_spec)
    features, stats = process_labevents(args.mimic_root, cohort, window_spec=window_spec, chunksize=args.chunksize)
    availability = summarize_lab24h_availability(features)

    features.to_csv(processed_dir / "mimic_diabetes_lab_features_24h.csv", index=False)
    availability.to_csv(tables_dir / "mimic_diabetes_lab24h_feature_availability.csv", index=False)
    pd.DataFrame([stats]).to_csv(tables_dir / "mimic_diabetes_lab24h_extraction_stats.csv", index=False)
    write_report(stats, availability, reports_dir / "mimic_diabetes_lab24h_feature_report.md")

    print("MIMIC diabetes first-24h lab features extracted")
    print(f"time_window_rows={stats['time_window_rows']}")
    print(availability.to_string(index=False))


if __name__ == "__main__":
    main()
