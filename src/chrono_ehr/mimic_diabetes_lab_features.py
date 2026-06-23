#!/usr/bin/env python3
"""Extract discharge-time-safe lab features for the MIMIC diabetes cohort."""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MIMIC_ROOT = Path(os.environ.get("MIMIC_IV_ROOT", "~/mimic-iv-3.1")).expanduser()

TARGET_ITEMIDS = {
    50809: "glucose",
    50931: "glucose",
    52027: "glucose",
    52569: "glucose",
    50852: "hba1c",
    50854: "hba1c_absolute",
    50912: "creatinine",
    52024: "creatinine",
    52546: "creatinine",
    51006: "bun",
    52647: "bun",
}

LABS = ["glucose", "hba1c", "hba1c_absolute", "creatinine", "bun"]


def load_cohort(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "mimic_diabetes_readmission_cohort.csv"
    usecols = ["hadm_id", "admittime", "dischtime"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["admittime", "dischtime"])
    return df


def empty_state() -> dict[int, dict[str, dict]]:
    return {}


def get_slot(state: dict[int, dict[str, dict]], hadm_id: int, lab: str) -> dict:
    by_lab = state.setdefault(int(hadm_id), {})
    return by_lab.setdefault(
        lab,
        {
            "count": 0,
            "sum": 0.0,
            "min": None,
            "max": None,
            "abnormal_count": 0,
            "last_time": None,
            "last": None,
        },
    )


def update_state(state: dict[int, dict[str, dict]], grouped: pd.DataFrame, latest: pd.DataFrame) -> None:
    for row in grouped.itertuples(index=False):
        slot = get_slot(state, int(row.hadm_id), str(row.lab_name))
        slot["count"] += int(row.count)
        slot["sum"] += float(row.sum)
        slot["min"] = float(row.min) if slot["min"] is None else min(slot["min"], float(row.min))
        slot["max"] = float(row.max) if slot["max"] is None else max(slot["max"], float(row.max))
        slot["abnormal_count"] += int(row.abnormal_count)

    for row in latest.itertuples(index=False):
        slot = get_slot(state, int(row.hadm_id), str(row.lab_name))
        row_time = row.available_time
        if slot["last_time"] is None or row_time > slot["last_time"]:
            slot["last_time"] = row_time
            slot["last"] = float(row.valuenum)


def process_labevents(
    mimic_root: Path,
    cohort: pd.DataFrame,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    labevents_path = mimic_root / "hosp" / "labevents.csv.gz"
    cohort_times = cohort.set_index("hadm_id")[["admittime", "dischtime"]]
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
        chunk["available_time"] = chunk["storetime"].where(chunk["storetime"].notna(), chunk["charttime"])
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk["lab_name"] = chunk["itemid"].map(TARGET_ITEMIDS)

        in_window = (
            chunk["available_time"].notna()
            & chunk["valuenum"].notna()
            & (chunk["available_time"] >= chunk["admittime"])
            & (chunk["available_time"] <= chunk["dischtime"])
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

    return pd.DataFrame(rows), stats


def summarize_lab_availability(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(features)
    for lab in LABS:
        has_col = f"lab_{lab}_has"
        count_col = f"lab_{lab}_count"
        rows.append(
            {
                "lab": lab,
                "hadm_with_lab": int(features[has_col].sum()),
                "hadm_with_lab_percent": float(features[has_col].mean()) if total else 0.0,
                "total_measurements": int(features[count_col].sum()),
                "median_measurements_among_all": float(features[count_col].median()),
            }
        )
    return pd.DataFrame(rows)


def write_report(stats: dict[str, int], availability: pd.DataFrame, report_path: Path) -> None:
    lines = ["| Lab | HADM with lab | Percent | Total measurements | Median measurements |", "|---|---:|---:|---:|---:|"]
    for row in availability.itertuples(index=False):
        lines.append(
            f"| {row.lab} | {int(row.hadm_with_lab)} | {row.hadm_with_lab_percent:.2%} | "
            f"{int(row.total_measurements)} | {row.median_measurements_among_all:.1f} |"
        )
    availability_md = "\n".join(lines)
    text = f"""# MIMIC 糖尿病出院前化验特征抽取报告

## 时间点规则

- 预测时间点：`dischtime`
- 化验可用时间：优先使用 `storetime`，缺失时使用 `charttime`
- 纳入化验：`admittime <= available_time <= dischtime`
- 只抽取 index admission 内、目标 itemid 的数值型化验

## 扫描统计

- chunks：{stats["chunks"]}
- 原始 labevents 行数：{stats["raw_rows_scanned"]}
- 目标 itemid 行数：{stats["target_item_rows"]}
- 属于糖尿病 cohort hadm 的行数：{stats["cohort_hadm_rows"]}
- 通过入院到出院时间窗的数值化验行数：{stats["time_window_rows"]}

## 化验覆盖

{availability_md}

## 说明

这些特征可以用于 discharge-time prediction。它们不能直接迁移到 admission-time prediction；如果未来改成入院时预测，需要把时间窗改为入院前或入院后固定早期窗口。
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
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

    cohort = load_cohort(args.project_root)
    features, stats = process_labevents(args.mimic_root, cohort, chunksize=args.chunksize)
    availability = summarize_lab_availability(features)

    features.to_csv(processed_dir / "mimic_diabetes_lab_features.csv", index=False)
    availability.to_csv(tables_dir / "mimic_diabetes_lab_feature_availability.csv", index=False)
    pd.DataFrame([stats]).to_csv(tables_dir / "mimic_diabetes_lab_extraction_stats.csv", index=False)
    write_report(stats, availability, reports_dir / "mimic_diabetes_lab_feature_report.md")

    print("MIMIC diabetes lab features extracted")
    print(f"time_window_rows={stats['time_window_rows']}")
    print(availability.to_string(index=False))


if __name__ == "__main__":
    main()
