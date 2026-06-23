#!/usr/bin/env python3
"""Extract discharge-time-safe diabetes medication features."""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MIMIC_ROOT = Path(os.environ.get("MIMIC_IV_ROOT", "~/mimic-iv-3.1")).expanduser()

MED_CLASSES = {
    "insulin": ["insulin", "humalog", "novolog", "glargine", "levemir", "lispro", "aspart"],
    "metformin": ["metformin", "glucophage"],
    "sulfonylurea": ["glipizide", "glyburide", "glimepiride"],
    "dpp4": ["sitagliptin", "linagliptin", "alogliptin", "saxagliptin"],
    "tzd": ["pioglitazone", "rosiglitazone"],
    "glp1": ["liraglutide", "semaglutide", "dulaglutide", "exenatide"],
    "sglt2": ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"],
    "alpha_glucosidase": ["acarbose", "miglitol"],
}


def load_cohort(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "mimic_diabetes_readmission_cohort.csv"
    usecols = ["hadm_id", "admittime", "dischtime"]
    return pd.read_csv(path, usecols=usecols, parse_dates=["admittime", "dischtime"])


def classify_drug(drug: str) -> str | None:
    text = str(drug).lower()
    for med_class, terms in MED_CLASSES.items():
        if any(term in text for term in terms):
            return med_class
    return None


def init_row(hadm_id: int) -> dict:
    row = {"hadm_id": int(hadm_id)}
    for med_class in MED_CLASSES:
        row[f"med_{med_class}_count"] = 0
        row[f"med_{med_class}_has"] = 0
    row["med_any_diabetes_count"] = 0
    row["med_any_diabetes_has"] = 0
    return row


def process_prescriptions(
    mimic_root: Path,
    cohort: pd.DataFrame,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    prescriptions_path = mimic_root / "hosp" / "prescriptions.csv.gz"
    cohort_times = cohort.set_index("hadm_id")[["admittime", "dischtime"]]
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
        available_time = chunk["starttime"].where(chunk["starttime"].notna(), chunk["stoptime"])
        chunk["available_time"] = available_time

        in_window = (
            chunk["available_time"].notna()
            & (chunk["available_time"] <= chunk["dischtime"])
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

    features = pd.DataFrame(state.values()).sort_values("hadm_id")
    availability_rows = []
    total = len(features)
    for med_class in [*MED_CLASSES.keys(), "any_diabetes"]:
        has_col = f"med_{med_class}_has"
        count_col = f"med_{med_class}_count"
        availability_rows.append(
            {
                "med_class": med_class,
                "hadm_with_med": int(features[has_col].sum()),
                "hadm_with_med_percent": float(features[has_col].mean()) if total else 0.0,
                "total_orders": int(features[count_col].sum()),
                "median_orders_among_all": float(features[count_col].median()),
            }
        )
    return features, pd.DataFrame(availability_rows), stats


def write_report(stats: dict[str, int], availability: pd.DataFrame, report_path: Path) -> None:
    lines = ["| Medication class | HADM with med | Percent | Total orders | Median orders |", "|---|---:|---:|---:|---:|"]
    for row in availability.itertuples(index=False):
        lines.append(
            f"| {row.med_class} | {int(row.hadm_with_med)} | {row.hadm_with_med_percent:.2%} | "
            f"{int(row.total_orders)} | {row.median_orders_among_all:.1f} |"
        )
    text = f"""# MIMIC 糖尿病出院前用药特征抽取报告

## 时间点规则

- 预测时间点：`dischtime`
- 用药可用时间：优先使用 `starttime`，缺失时使用 `stoptime`
- 纳入用药：处方开始时间不晚于出院，且处方时间与 index admission 有交集
- 数据来源：`hosp/prescriptions.csv.gz`

## 扫描统计

- chunks：{stats["chunks"]}
- 原始 prescriptions 行数：{stats["raw_rows_scanned"]}
- 糖尿病相关药物行数：{stats["diabetes_med_rows"]}
- 属于糖尿病 cohort hadm 的药物行数：{stats["cohort_hadm_rows"]}
- 通过出院前时间窗的药物行数：{stats["time_window_rows"]}

## 用药覆盖

{chr(10).join(lines)}

## 注意

第一版按药名关键词识别药物类别，主要用于 demo 和 baseline。正式论文前需要进一步核对药品映射，例如部分 insulin 可能用于高钾血症处理，不一定代表常规降糖治疗。
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
    features, availability, stats = process_prescriptions(args.mimic_root, cohort, chunksize=args.chunksize)
    features.to_csv(processed_dir / "mimic_diabetes_med_features.csv", index=False)
    availability.to_csv(tables_dir / "mimic_diabetes_med_feature_availability.csv", index=False)
    pd.DataFrame([stats]).to_csv(tables_dir / "mimic_diabetes_med_extraction_stats.csv", index=False)
    write_report(stats, availability, reports_dir / "mimic_diabetes_med_feature_report.md")

    print("MIMIC diabetes medication features extracted")
    print(f"time_window_rows={stats['time_window_rows']}")
    print(availability.to_string(index=False))


if __name__ == "__main__":
    main()

