#!/usr/bin/env python3
"""Extract first-24h and discharge-time general medication features for chronic cohorts."""

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

MEDICATION_CLASSES = [
    "insulin",
    "oral_antidiabetic",
    "loop_diuretic",
    "thiazide_diuretic",
    "beta_blocker",
    "acei_arb",
    "calcium_channel_blocker",
    "vasopressor_inotrope",
    "antibiotic",
    "anticoagulant",
    "heparin_flush",
    "antiplatelet",
    "statin",
    "steroid",
    "opioid",
    "sedative",
    "bronchodilator",
    "ppi_h2blocker",
    "electrolyte",
    "iv_fluid",
]

MED_PATTERNS = {
    "insulin": r"\binsulin\b|glargine|lispro|aspart|detemir|regular\s+insulin|humulin|novolog|lantus",
    "oral_antidiabetic": r"metformin|glyburide|glipizide|glimepiride|sitagliptin|linagliptin|pioglitazone|rosiglitazone|canagliflozin|dapagliflozin|empagliflozin|acarbose",
    "loop_diuretic": r"furosemide|bumetanide|torsemide|ethacrynic",
    "thiazide_diuretic": r"hydrochlorothiazide|chlorthalidone|metolazone|chlorothiazide",
    "beta_blocker": r"metoprolol|carvedilol|atenolol|labetalol|propranolol|bisoprolol|esmolol|nadolol",
    "acei_arb": r"lisinopril|losartan|valsartan|captopril|enalapril|ramipril|irbesartan|olmesartan|benazepril|quinapril|candesartan",
    "calcium_channel_blocker": r"amlodipine|diltiazem|verapamil|nifedipine|nicardipine|clevidipine",
    "vasopressor_inotrope": r"norepinephrine|epinephrine|phenylephrine|vasopressin|dopamine|dobutamine|milrinone",
    "antibiotic": r"vancomycin|cefepime|ceftriaxone|cefazolin|piperacillin|tazobactam|meropenem|ertapenem|imipenem|levofloxacin|ciprofloxacin|azithromycin|metronidazole|linezolid|daptomycin|clindamycin|gentamicin|tobramycin|ampicillin|amoxicillin",
    "anticoagulant": r"\bheparin\b|warfarin|enoxaparin|apixaban|rivaroxaban|fondaparinux|argatroban|bivalirudin|dabigatran",
    "antiplatelet": r"aspirin|clopidogrel|ticagrelor|prasugrel|dipyridamole",
    "statin": r"atorvastatin|simvastatin|rosuvastatin|pravastatin|lovastatin|fluvastatin",
    "steroid": r"prednisone|methylprednisolone|hydrocortisone|dexamethasone|prednisolone|solu-medrol",
    "opioid": r"morphine|hydromorphone|fentanyl|oxycodone|hydrocodone|methadone|tramadol|codeine",
    "sedative": r"propofol|lorazepam|midazolam|dexmedetomidine|diazepam|clonazepam|haloperidol|quetiapine",
    "bronchodilator": r"albuterol|ipratropium|tiotropium|levalbuterol|budesonide|fluticasone|formoterol|salmeterol",
    "ppi_h2blocker": r"pantoprazole|omeprazole|lansoprazole|esomeprazole|famotidine|ranitidine",
    "electrolyte": r"potassium|magnesium|calcium\s+gluconate|calcium\s+chloride|sodium\s+bicarbonate|phosphate",
    "iv_fluid": r"sodium\s+chloride|normal\s+saline|lactated\s+ringer|dextrose|d5w|sterile\s+water",
}

COMPILED_PATTERNS = {name: re.compile(pattern, flags=re.IGNORECASE) for name, pattern in MED_PATTERNS.items()}
HEPARIN_FLUSH_PATTERN = re.compile(r"\bheparin\b.*\bflush\b|\bflush\b.*\bheparin\b", flags=re.IGNORECASE)
IV_ROUTE_PATTERN = re.compile(r"\biv\b|intravenous|ivpb|iv drip|iv push|central", flags=re.IGNORECASE)


def classify_drug(drug: str | float | None) -> list[str]:
    if drug is None or pd.isna(drug):
        return []
    text = str(drug)
    classes = []
    if HEPARIN_FLUSH_PATTERN.search(text):
        classes.append("heparin_flush")
    for med_class, pattern in COMPILED_PATTERNS.items():
        if med_class == "anticoagulant" and HEPARIN_FLUSH_PATTERN.search(text):
            continue
        if pattern.search(text):
            classes.append(med_class)
    return sorted(set(classes))


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


def empty_state() -> dict[str, dict[int, dict[str, dict[str, int]]]]:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "iv_count": 0})))


def update_state(state: dict[str, dict[int, dict[str, dict[str, int]]]], part: pd.DataFrame) -> None:
    grouped = (
        part.groupby(["cohort", "hadm_id", "med_class"], sort=False)
        .agg(count=("med_class", "size"), iv_count=("is_iv_route", "sum"))
        .reset_index()
    )
    total = (
        part.groupby(["cohort", "hadm_id"], sort=False)
        .agg(count=("med_class", "size"), iv_count=("is_iv_route", "sum"))
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        slot = state[str(row.cohort)][int(row.hadm_id)][str(row.med_class)]
        slot["count"] += int(row.count)
        slot["iv_count"] += int(row.iv_count)
    for row in total.itertuples(index=False):
        slot = state[str(row.cohort)][int(row.hadm_id)]["any_general_med"]
        slot["count"] += int(row.count)
        slot["iv_count"] += int(row.iv_count)


def state_to_features(
    state: dict[str, dict[int, dict[str, dict[str, int]]]],
    cohort_hadm: dict[str, set[int]],
    prefix: str,
) -> dict[str, pd.DataFrame]:
    outputs = {}
    classes = [*MEDICATION_CLASSES, "any_general_med"]
    for cohort, hadm_ids in cohort_hadm.items():
        rows = []
        cohort_state = state.get(cohort, {})
        for hadm_id in sorted(hadm_ids):
            row = {"hadm_id": int(hadm_id)}
            by_class = cohort_state.get(int(hadm_id), {})
            for med_class in classes:
                slot = by_class.get(med_class, {"count": 0, "iv_count": 0})
                col = f"{prefix}_{med_class}"
                count = int(slot["count"])
                iv_count = int(slot["iv_count"])
                row[f"{col}_count"] = count
                row[f"{col}_has"] = int(count > 0)
                row[f"{col}_iv_count"] = iv_count
                row[f"{col}_iv_has"] = int(iv_count > 0)
            rows.append(row)
        outputs[cohort] = pd.DataFrame(rows)
    return outputs


def summarize_availability(features: pd.DataFrame, cohort: str, prefix: str, window: str) -> pd.DataFrame:
    rows = []
    total = len(features)
    for med_class in [*MEDICATION_CLASSES, "any_general_med"]:
        has_col = f"{prefix}_{med_class}_has"
        count_col = f"{prefix}_{med_class}_count"
        iv_has_col = f"{prefix}_{med_class}_iv_has"
        iv_count_col = f"{prefix}_{med_class}_iv_count"
        rows.append(
            {
                "cohort": cohort,
                "window": window,
                "med_class": med_class,
                "hadm_with_med": int(features[has_col].sum()),
                "hadm_with_med_percent": float(features[has_col].mean()) if total else 0.0,
                "total_orders": int(features[count_col].sum()),
                "median_orders_among_all": float(features[count_col].median()) if total else 0.0,
                "hadm_with_iv_med": int(features[iv_has_col].sum()),
                "total_iv_orders": int(features[iv_count_col].sum()),
            }
        )
    return pd.DataFrame(rows)


def process_prescriptions(
    mimic_root: Path,
    cohort_windows: pd.DataFrame,
    window_spec: dict,
    chunksize: int,
    max_chunks: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    prescriptions_path = mimic_root / "hosp" / "prescriptions.csv.gz"
    med_source = source_spec(window_spec, "general_medications")
    cohort_hadm = {cohort: set(group["hadm_id"].astype(int)) for cohort, group in cohort_windows.groupby("cohort", sort=False)}
    union_hadm = set(cohort_windows["hadm_id"].astype(int))
    state_24h = empty_state()
    state_discharge = empty_state()
    stats = {
        "chunks": 0,
        "raw_rows_scanned": 0,
        "classified_med_rows": 0,
        "cohort_hadm_rows": 0,
        "time_window_24h_rows": 0,
        "time_window_discharge_rows": 0,
    }

    usecols = ["hadm_id", "starttime", "stoptime", "drug", "route"]
    for chunk in pd.read_csv(prescriptions_path, compression="gzip", usecols=usecols, chunksize=chunksize, low_memory=False):
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

        chunk["med_class"] = chunk["drug"].map(classify_drug)
        chunk = chunk[chunk["med_class"].map(bool)].copy()
        stats["classified_med_rows"] += int(len(chunk))
        if chunk.empty:
            if max_chunks and stats["chunks"] >= max_chunks:
                break
            continue

        chunk = chunk.explode("med_class")
        chunk = chunk.merge(cohort_windows, on="hadm_id", how="inner")
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["stoptime"] = pd.to_datetime(chunk["stoptime"], errors="coerce")
        chunk["available_time"] = available_time_from_source(chunk, med_source)
        chunk["is_iv_route"] = chunk["route"].astype("string").fillna("").map(lambda route: int(bool(IV_ROUTE_PATTERN.search(route))))

        overlaps_admission = (
            chunk["available_time"].notna()
            & (chunk["available_time"] <= chunk["window_discharge_end"])
            & (chunk["stoptime"].isna() | (chunk["stoptime"] >= chunk["admittime"]) | (chunk["available_time"] >= chunk["admittime"]))
        )
        chunk = chunk[overlaps_admission].copy()
        if chunk.empty:
            if max_chunks and stats["chunks"] >= max_chunks:
                break
            continue

        effective_time = chunk["available_time"].where(chunk["available_time"] >= chunk["admittime"], chunk["admittime"])
        part_24h = chunk[effective_time <= chunk["window_24h_end"]].copy()
        stats["time_window_24h_rows"] += int(len(part_24h))
        if not part_24h.empty:
            update_state(state_24h, part_24h)

        part_discharge = chunk[effective_time <= chunk["window_discharge_end"]].copy()
        stats["time_window_discharge_rows"] += int(len(part_discharge))
        if not part_discharge.empty:
            update_state(state_discharge, part_discharge)

        if max_chunks and stats["chunks"] >= max_chunks:
            break

    features_24h = state_to_features(state_24h, cohort_hadm, "genmed24h")
    features_discharge = state_to_features(state_discharge, cohort_hadm, "genmeddischarge")
    availability = []
    for cohort in STUDIES:
        availability.append(summarize_availability(features_24h[cohort], cohort, "genmed24h", "first_24h"))
        availability.append(summarize_availability(features_discharge[cohort], cohort, "genmeddischarge", "admission_to_discharge"))
    return features_24h, features_discharge, pd.concat(availability, ignore_index=True), stats


def write_report(stats: dict[str, int], availability: pd.DataFrame, output: Path) -> None:
    rows = [
        "| Cohort | Window | Medication class | HADM with med | Percent | Total orders | HADM with IV med | Total IV orders | Median orders |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in availability.itertuples(index=False):
        rows.append(
            f"| {row.cohort} | {row.window} | {row.med_class} | {int(row.hadm_with_med)} | "
            f"{row.hadm_with_med_percent:.2%} | {int(row.total_orders)} | {int(row.hadm_with_iv_med)} | "
            f"{int(row.total_iv_orders)} | {row.median_orders_among_all:.1f} |"
        )
    text = f"""# MIMIC Chronic Disease General Medication Feature Extraction Report

## Purpose

这个报告记录四个慢病队列的通用 medication features。Medication orders 来自 `hosp/prescriptions.csv.gz`，按药名关键词归入常见治疗类别，并分别生成 first-24h 和 admission-to-discharge 两个时间窗。

## Time Windows

- First 24h: `admittime <= effective_available_time <= min(admittime + 24h, dischtime)`
- Discharge-safe: `admittime <= effective_available_time <= dischtime`
- Available time: `starttime` when present, otherwise `stoptime`
- If an order starts before admission but overlaps the admission, its effective available time is treated as admission time.
- Window config: `configs/feature_window_specs.json`

## Medication Classes

`{", ".join(MEDICATION_CLASSES)}`

`heparin flush` is separated from therapeutic anticoagulant exposure to reduce a common medication-feature proxy problem.

## Scan Stats

- chunks: {stats["chunks"]}
- raw prescriptions rows scanned: {stats["raw_rows_scanned"]}
- chronic cohort HADM rows: {stats["cohort_hadm_rows"]}
- classified medication rows before explode: {stats["classified_med_rows"]}
- first-24h rows after class explode: {stats["time_window_24h_rows"]}
- admission-to-discharge rows after class explode: {stats["time_window_discharge_rows"]}

## Availability

{chr(10).join(rows)}

## Leakage Note

The `genmed24h_*` files are safe for a 24-hour in-hospital prediction time or later. The `genmeddischarge_*` files are safe only for discharge-time prediction and must not be reused for admission-time or 24-hour prediction.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--window-spec", type=Path, default=DEFAULT_FEATURE_WINDOW_SPEC)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
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
    features_24h, features_discharge, availability, stats = process_prescriptions(
        args.mimic_root,
        cohort_windows,
        window_spec,
        args.chunksize,
        args.max_chunks,
    )
    if args.dry_run:
        print("MIMIC chronic disease medication dry run complete")
        print(f"chunks={stats['chunks']}")
        print(f"first_24h_rows={stats['time_window_24h_rows']}")
        print(f"discharge_rows={stats['time_window_discharge_rows']}")
        return

    for cohort in STUDIES:
        features_24h[cohort].to_csv(processed / f"mimic_{cohort}_general_med_features_24h.csv", index=False)
        features_discharge[cohort].to_csv(processed / f"mimic_{cohort}_general_med_features_discharge.csv", index=False)
    availability.to_csv(tables / "chronic_disease_general_med_feature_availability.csv", index=False)
    pd.DataFrame([stats]).to_csv(tables / "chronic_disease_general_med_extraction_stats.csv", index=False)
    write_report(stats, availability, reports / "chronic_disease_general_med_feature_report.md")

    print("MIMIC chronic disease general medication features extracted")
    print(f"chunks={stats['chunks']}")
    print(f"first_24h_rows={stats['time_window_24h_rows']}")
    print(f"discharge_rows={stats['time_window_discharge_rows']}")


if __name__ == "__main__":
    main()
