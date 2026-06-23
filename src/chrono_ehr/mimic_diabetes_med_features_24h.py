#!/usr/bin/env python3
"""Extract first-24h diabetes medication features."""

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
from mimic_diabetes_med_features import (
    DEFAULT_MIMIC_ROOT,
    DEFAULT_PROJECT,
    MED_CLASSES,
    classify_drug,
    init_row,
)


def load_cohort(project_root: Path, window_spec: dict | None = None) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "mimic_diabetes_readmission_cohort.csv"
    usecols = ["hadm_id", "admittime", "dischtime"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["admittime", "dischtime"])
    return add_window_end(df, "first_24h", spec=window_spec)


def rename_med24h_columns(features: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in features.columns:
        if column.startswith("med_"):
            renamed[column] = "med24h_" + column.removeprefix("med_")
    return features.rename(columns=renamed)


def process_prescriptions_24h(
    mimic_root: Path,
    cohort: pd.DataFrame,
    window_spec: dict | None = None,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    window_spec = window_spec or load_feature_window_spec()
    med_source = source_spec(window_spec, "diabetes_medications")
    prescriptions_path = mimic_root / "hosp" / "prescriptions.csv.gz"
    cohort_times = cohort.set_index("hadm_id")[["admittime", "window_end"]]
    cohort_hadm = set(cohort_times.index.astype(int))
    state = {int(hadm_id): init_row(int(hadm_id)) for hadm_id in cohort_hadm}

    stats = {
        "chunks": 0,
        "raw_rows_scanned": 0,
        "diabetes_med_rows": 0,
        "cohort_hadm_rows": 0,
        "time_window_rows": 0,
    }

    usecols = ["hadm_id", "starttime", "stoptime", "drug", "route"]
    for chunk in pd.read_csv(
        prescriptions_path,
        compression="gzip",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        stats["chunks"] += 1
        stats["raw_rows_scanned"] += int(len(chunk))
        chunk["med_class"] = chunk["drug"].map(classify_drug)
        chunk = chunk[chunk["med_class"].notna()]
        stats["diabetes_med_rows"] += int(len(chunk))
        if chunk.empty:
            continue

        chunk = chunk[chunk["hadm_id"].notna()]
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(cohort_hadm)]
        stats["cohort_hadm_rows"] += int(len(chunk))
        if chunk.empty:
            continue

        chunk = chunk.merge(cohort_times, left_on="hadm_id", right_index=True, how="left")
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["stoptime"] = pd.to_datetime(chunk["stoptime"], errors="coerce")
        chunk["available_time"] = available_time_from_source(chunk, med_source)

        in_window = (
            chunk["available_time"].notna()
            & (chunk["available_time"] <= chunk["window_end"])
            & (
                chunk["stoptime"].isna()
                | (chunk["stoptime"] >= chunk["admittime"])
                | (chunk["available_time"] >= chunk["admittime"])
            )
        )
        chunk = chunk[in_window]
        stats["time_window_rows"] += int(len(chunk))
        if chunk.empty:
            continue

        grouped = chunk.groupby(["hadm_id", "med_class"], sort=False).size().reset_index(name="count")
        for row in grouped.itertuples(index=False):
            hadm_id = int(row.hadm_id)
            med_class = str(row.med_class)
            count = int(row.count)
            state[hadm_id][f"med_{med_class}_count"] += count
            state[hadm_id][f"med_{med_class}_has"] = 1
            state[hadm_id]["med_any_diabetes_count"] += count
            state[hadm_id]["med_any_diabetes_has"] = 1

    features = rename_med24h_columns(pd.DataFrame(state.values()).sort_values("hadm_id"))
    availability_rows = []
    total = len(features)
    for med_class in [*MED_CLASSES.keys(), "any_diabetes"]:
        has_col = f"med24h_{med_class}_has"
        count_col = f"med24h_{med_class}_count"
        availability_rows.append(
            {
                "window": "first_24h",
                "med_class": med_class,
                "hadm_with_med": int(features[has_col].sum()),
                "hadm_with_med_percent": float(features[has_col].mean()) if total else 0.0,
                "total_orders": int(features[count_col].sum()),
                "median_orders_among_all": float(features[count_col].median()),
            }
        )
    return features, pd.DataFrame(availability_rows), stats


def write_report(stats: dict[str, int], availability: pd.DataFrame, report_path: Path) -> None:
    lines = [
        "| Medication class | HADM with med | Percent | Total orders | Median orders |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in availability.itertuples(index=False):
        lines.append(
            f"| {row.med_class} | {int(row.hadm_with_med)} | {row.hadm_with_med_percent:.2%} | "
            f"{int(row.total_orders)} | {row.median_orders_among_all:.1f} |"
        )
    text = f"""# MIMIC 糖尿病入院后 24 小时用药特征抽取报告

## 时间点规则

- 预测时间点：`admittime + 24h`
- 用药可用时间：优先使用 `starttime`，缺失时使用 `stoptime`
- 纳入用药：处方时间不晚于 `min(admittime + 24h, dischtime)`，且处方时间与 index admission 有交集
- 时间窗配置：`configs/feature_window_specs.json` 中的 `first_24h`
- 数据来源：`hosp/prescriptions.csv.gz`

## 扫描统计

- chunks：{stats["chunks"]}
- 原始 prescriptions 行数：{stats["raw_rows_scanned"]}
- 糖尿病相关药物行数：{stats["diabetes_med_rows"]}
- 属于糖尿病 cohort hadm 的药物行数：{stats["cohort_hadm_rows"]}
- 通过前 24 小时时间窗的药物行数：{stats["time_window_rows"]}

## 用药覆盖

{chr(10).join(lines)}

## 注意

这些特征用于 inhospital 24h prediction，不能用于真正的 admission-time prediction。第一版仍按关键词识别药物类别，正式研究前需要进一步核对 insulin 等药物的具体用途。
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
    features, availability, stats = process_prescriptions_24h(args.mimic_root, cohort, window_spec=window_spec, chunksize=args.chunksize)
    features.to_csv(processed_dir / "mimic_diabetes_med_features_24h.csv", index=False)
    availability.to_csv(tables_dir / "mimic_diabetes_med24h_feature_availability.csv", index=False)
    pd.DataFrame([stats]).to_csv(tables_dir / "mimic_diabetes_med24h_extraction_stats.csv", index=False)
    write_report(stats, availability, reports_dir / "mimic_diabetes_med24h_feature_report.md")

    print("MIMIC diabetes first-24h medication features extracted")
    print(f"time_window_rows={stats['time_window_rows']}")
    print(availability.to_string(index=False))


if __name__ == "__main__":
    main()
