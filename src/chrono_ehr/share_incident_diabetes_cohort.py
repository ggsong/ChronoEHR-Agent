#!/usr/bin/env python3
"""Build a SHARE wave 1 to wave 2/wave 4 incident diabetes cohort skeleton."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from share_data_readiness import SHARE_CANDIDATE_ROOTS, choose_root, find_harmonized_file


ID_COLUMNS = ["mergeid", "hhid", "country"]
WAVE_COLUMNS = ["inw1", "inw2", "inw4"]
BASELINE_FEATURES = [
    "r1agey",
    "ragender",
    "raedyrs",
    "raeducl",
    "r1smokev",
    "r1smoken",
    "r1drink3m",
    "r1drinkx",
    "r1bmi",
    "r1hibpe",
    "r1hearte",
    "r1stroke",
    "r1lunge",
    "r1adla",
    "r1iadla",
    "r1mobilb",
    "r1depress",
    "r1wtresp",
]
BASELINE_DIABETES_COLUMNS = ["r1diabe", "r1rxdiab"]
FOLLOWUP_WAVE2_COLUMNS = ["r2diabe", "r2rxdiab"]
FOLLOWUP_WAVE4_COLUMNS = ["r4diabe", "r4rxdiab"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--share-root", type=Path, help="Optional explicit SHARE root.")
    parser.add_argument("--min-baseline-age", type=float, default=50.0)
    return parser.parse_args()


def available_columns(path: Path) -> set[str]:
    reader = pd.io.stata.StataReader(path)
    return set(reader.variable_labels())


def binary_yes(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    flags = []
    for column in columns:
        if column in df:
            flags.append(pd.to_numeric(df[column], errors="coerce").eq(1))
    if not flags:
        return pd.Series(False, index=df.index)
    return pd.concat(flags, axis=1).any(axis=1)


def binary_known(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    flags = []
    for column in columns:
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce")
            flags.append(values.isin([0, 1]))
    if not flags:
        return pd.Series(False, index=df.index)
    return pd.concat(flags, axis=1).any(axis=1)


def split_for_person(person_id: object) -> str:
    digest = hashlib.md5(str(person_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def read_harmonized(path: Path) -> pd.DataFrame:
    wanted = ID_COLUMNS + WAVE_COLUMNS + BASELINE_FEATURES + BASELINE_DIABETES_COLUMNS + FOLLOWUP_WAVE2_COLUMNS + FOLLOWUP_WAVE4_COLUMNS
    present = [column for column in wanted if column in available_columns(path)]
    return pd.read_stata(path, columns=present, convert_categoricals=False)


def build_cohort(raw: pd.DataFrame, min_baseline_age: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = raw.copy()
    cohort["person_id"] = cohort["mergeid"].astype(str)
    cohort["baseline_present"] = pd.to_numeric(cohort.get("inw1"), errors="coerce").eq(1) if "inw1" in cohort else cohort["r1agey"].notna()
    cohort["followup_wave2_present"] = pd.to_numeric(cohort.get("inw2"), errors="coerce").eq(1) if "inw2" in cohort else binary_known(cohort, FOLLOWUP_WAVE2_COLUMNS)
    cohort["followup_wave4_present"] = pd.to_numeric(cohort.get("inw4"), errors="coerce").eq(1) if "inw4" in cohort else binary_known(cohort, FOLLOWUP_WAVE4_COLUMNS)
    cohort["baseline_age_years"] = pd.to_numeric(cohort["r1agey"], errors="coerce")
    cohort["baseline_age_ge_min"] = cohort["baseline_age_years"].ge(min_baseline_age)
    cohort["baseline_diabetes_known"] = binary_known(cohort, BASELINE_DIABETES_COLUMNS)
    cohort["baseline_diabetes_any"] = binary_yes(cohort, BASELINE_DIABETES_COLUMNS)
    cohort["followup_wave2_diabetes_known"] = binary_known(cohort, FOLLOWUP_WAVE2_COLUMNS)
    cohort["followup_wave2_diabetes_any"] = binary_yes(cohort, FOLLOWUP_WAVE2_COLUMNS)
    cohort["followup_wave4_diabetes_known"] = binary_known(cohort, FOLLOWUP_WAVE4_COLUMNS)
    cohort["followup_wave4_diabetes_any"] = binary_yes(cohort, FOLLOWUP_WAVE4_COLUMNS)
    cohort["any_followup_diabetes_known"] = cohort["followup_wave2_diabetes_known"] | cohort["followup_wave4_diabetes_known"]
    cohort["incident_diabetes_wave2"] = cohort["followup_wave2_diabetes_any"].astype(int)
    cohort["incident_diabetes_wave4"] = cohort["followup_wave4_diabetes_any"].astype(int)
    cohort["incident_diabetes_wave2_or_wave4"] = (
        cohort["followup_wave2_diabetes_any"] | cohort["followup_wave4_diabetes_any"]
    ).astype(int)

    exclusions = [{"step": "raw_harmonized_rows", "remaining": len(cohort), "excluded": 0}]
    filters = [
        ("valid_person_id", cohort["person_id"].notna() & cohort["person_id"].ne("") & cohort["person_id"].ne("nan")),
        ("baseline_wave_present", cohort["baseline_present"]),
        ("baseline_age_at_least_min", cohort["baseline_age_ge_min"]),
        ("baseline_diabetes_known", cohort["baseline_diabetes_known"]),
        ("exclude_prevalent_baseline_diabetes", ~cohort["baseline_diabetes_any"]),
        ("at_least_one_followup_known", cohort["any_followup_diabetes_known"]),
    ]
    keep = pd.Series(True, index=cohort.index)
    for step, mask in filters:
        before = int(keep.sum())
        keep = keep & mask.fillna(False)
        exclusions.append({"step": step, "remaining": int(keep.sum()), "excluded": before - int(keep.sum())})

    cohort = cohort[keep].copy()
    cohort["split"] = cohort["person_id"].map(split_for_person)
    cohort["prediction_anchor_wave"] = "share_wave1"
    cohort["outcome_window"] = "share_wave2_or_wave4"

    rename = {
        "mergeid": "share_mergeid",
        "hhid": "household_id",
        "ragender": "sex",
        "raedyrs": "education_years",
        "raeducl": "education_level",
        "r1smokev": "baseline_smoked_ever",
        "r1smoken": "baseline_smokes_now",
        "r1drink3m": "baseline_drinks_recently",
        "r1drinkx": "baseline_drink_frequency",
        "r1bmi": "baseline_bmi",
        "r1hibpe": "baseline_hypertension",
        "r1hearte": "baseline_heart_problem",
        "r1stroke": "baseline_stroke",
        "r1lunge": "baseline_lung_disease",
        "r1adla": "baseline_adl_limitations",
        "r1iadla": "baseline_iadl_limitations",
        "r1mobilb": "baseline_mobility_limitations",
        "r1depress": "baseline_depression",
        "r1wtresp": "baseline_person_weight",
    }
    cohort = cohort.rename(columns=rename)
    columns = [
        "person_id",
        "household_id",
        "country",
        "split",
        "prediction_anchor_wave",
        "outcome_window",
        "baseline_age_years",
        "sex",
        "education_years",
        "education_level",
        "baseline_smoked_ever",
        "baseline_smokes_now",
        "baseline_drinks_recently",
        "baseline_drink_frequency",
        "baseline_bmi",
        "baseline_hypertension",
        "baseline_heart_problem",
        "baseline_stroke",
        "baseline_lung_disease",
        "baseline_adl_limitations",
        "baseline_iadl_limitations",
        "baseline_mobility_limitations",
        "baseline_depression",
        "baseline_person_weight",
        "followup_wave2_present",
        "followup_wave4_present",
        "followup_wave2_diabetes_known",
        "followup_wave4_diabetes_known",
        "incident_diabetes_wave2",
        "incident_diabetes_wave4",
        "incident_diabetes_wave2_or_wave4",
        "baseline_diabetes_any",
    ]
    columns = [column for column in columns if column in cohort]
    return cohort[columns].sort_values("person_id"), pd.DataFrame(exclusions)


def summarize(cohort: pd.DataFrame, exclusions: pd.DataFrame, min_baseline_age: float) -> pd.DataFrame:
    rows = [
        {"metric": "eligible_persons", "value": len(cohort)},
        {"metric": "unique_persons", "value": cohort["person_id"].nunique()},
        {"metric": "min_baseline_age", "value": min_baseline_age},
        {"metric": "mean_baseline_age", "value": round(float(cohort["baseline_age_years"].mean()), 3)},
        {"metric": "incident_diabetes_events", "value": int(cohort["incident_diabetes_wave2_or_wave4"].sum())},
        {"metric": "incident_diabetes_rate", "value": round(float(cohort["incident_diabetes_wave2_or_wave4"].mean()), 4)},
        {"metric": "followup_wave2_known", "value": int(cohort["followup_wave2_diabetes_known"].sum())},
        {"metric": "followup_wave4_known", "value": int(cohort["followup_wave4_diabetes_known"].sum())},
        {"metric": "excluded_prevalent_baseline_diabetes", "value": int(exclusions.loc[exclusions["step"].eq("exclude_prevalent_baseline_diabetes"), "excluded"].iloc[0])},
    ]
    for split, group in cohort.groupby("split"):
        rows.append({"metric": f"{split}_persons", "value": len(group)})
        rows.append({"metric": f"{split}_event_rate", "value": round(float(group["incident_diabetes_wave2_or_wave4"].mean()), 4)})
    return pd.DataFrame(rows)


def wave_outcome_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave, known_col, event_col in [
        ("wave2", "followup_wave2_diabetes_known", "incident_diabetes_wave2"),
        ("wave4", "followup_wave4_diabetes_known", "incident_diabetes_wave4"),
    ]:
        known_mask = cohort[known_col].astype(bool)
        rows.append(
            {
                "wave": wave,
                "known": int(known_mask.sum()),
                "events": int(cohort[event_col].sum()),
                "event_rate": round(float(cohort.loc[known_mask, event_col].mean()), 4) if known_mask.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for item in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(
    project_root: Path,
    harmonized_path: Path,
    cohort: pd.DataFrame,
    exclusions: pd.DataFrame,
    summary: pd.DataFrame,
    wave_summary: pd.DataFrame,
) -> Path:
    output = project_root / "outputs" / "reports" / "share_incident_diabetes_cohort_skeleton.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# SHARE Incident Diabetes Cohort Skeleton

- Harmonized source: `{harmonized_path}`
- Prediction anchor: SHARE wave 1
- Outcome window: wave 2 or wave 4
- Boundary: cohort skeleton only; no model training and no clinical recommendation.

## Cohort Summary

{markdown_table(summary)}

## Sample Flow

{markdown_table(exclusions)}

## Wave Outcome Summary

{markdown_table(wave_summary)}

## Exported Columns

{", ".join(cohort.columns)}
"""
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    selected_root, _ = choose_root(args.share_root)
    if selected_root is None:
        selected_root = args.share_root or next((root for root in SHARE_CANDIDATE_ROOTS if root.exists()), None)
    harmonized_path = find_harmonized_file(selected_root)
    if harmonized_path is None:
        raise FileNotFoundError("No harmonized SHARE .dta file found. Set SHARE_ROOT or run --share-readiness.")

    raw = read_harmonized(harmonized_path)
    cohort, exclusions = build_cohort(raw, args.min_baseline_age)
    summary = summarize(cohort, exclusions, args.min_baseline_age)
    wave_summary = wave_outcome_summary(cohort)

    data_dir = args.project_root / "data" / "processed"
    table_dir = args.project_root / "outputs" / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = data_dir / "share_incident_diabetes_cohort_skeleton.csv"
    exclusions_path = table_dir / "share_incident_diabetes_sample_flow.csv"
    summary_path = table_dir / "share_incident_diabetes_cohort_summary.csv"
    wave_summary_path = table_dir / "share_incident_diabetes_wave_outcome_summary.csv"
    cohort.to_csv(cohort_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)
    summary.to_csv(summary_path, index=False)
    wave_summary.to_csv(wave_summary_path, index=False)
    report = write_report(args.project_root, harmonized_path, cohort, exclusions, summary, wave_summary)
    print(f"SHARE incident diabetes cohort rows: {len(cohort)}")
    print(f"Events: {int(cohort['incident_diabetes_wave2_or_wave4'].sum())}")
    print(f"Wrote {cohort_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
