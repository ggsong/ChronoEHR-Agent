#!/usr/bin/env python3
"""Build a CHARLS 2011 baseline to 2013/2015 incident diabetes cohort skeleton."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from charls_data_readiness import CHARLS_CANDIDATE_ROOTS, choose_root
from charls_wave_variable_map import CORE_VARIABLES, find_harmonized_file
from mimic_diabetes_baseline import DEFAULT_PROJECT


ID_COLUMNS = ["ID", "householdID", "communityID", "hhid", "pn"]
BASELINE_FEATURES = [
    "r1agey",
    "ragender",
    "raeduc_c",
    "raeducl",
    "r1smokev",
    "r1smoken",
    "r1drinkl",
    "r1drinkn_c",
    "r1mbmi",
    "r1hibpe",
    "r1hearte",
    "r1stroke",
    "r1lunge",
    "r1kidneye",
]
WAVE_COLUMNS = ["inw1", "inw2", "inw3"]
BASELINE_DIABETES_COLUMNS = ["r1diabe", "r1rxdiab_c", "r1rxdiab", "r1rxdiabi"]
FOLLOWUP_2013_COLUMNS = ["r2diabe", "r2rxdiab_c", "r2rxdiab", "r2rxdiabi"]
FOLLOWUP_2015_COLUMNS = ["r3diabe", "r3rxdiab_c", "r3rxdiab", "r3rxdiabi"]
FORBIDDEN_BASELINE_FEATURES = set(FOLLOWUP_2013_COLUMNS + FOLLOWUP_2015_COLUMNS + ["inw2", "inw3"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--charls-root", type=Path, help="Optional explicit CHARLS root.")
    parser.add_argument("--min-baseline-age", type=float, default=45.0)
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
    wanted = ID_COLUMNS + WAVE_COLUMNS + BASELINE_FEATURES + BASELINE_DIABETES_COLUMNS + FOLLOWUP_2013_COLUMNS + FOLLOWUP_2015_COLUMNS
    present = [column for column in wanted if column in available_columns(path)]
    return pd.read_stata(path, columns=present, convert_categoricals=False)


def build_cohort(raw: pd.DataFrame, min_baseline_age: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = raw.copy()
    cohort["person_id"] = cohort["ID"].astype(str)
    cohort["baseline_present"] = pd.to_numeric(cohort.get("inw1"), errors="coerce").eq(1) if "inw1" in cohort else cohort["r1agey"].notna()
    cohort["followup_2013_present"] = pd.to_numeric(cohort.get("inw2"), errors="coerce").eq(1) if "inw2" in cohort else binary_known(cohort, FOLLOWUP_2013_COLUMNS)
    cohort["followup_2015_present"] = pd.to_numeric(cohort.get("inw3"), errors="coerce").eq(1) if "inw3" in cohort else binary_known(cohort, FOLLOWUP_2015_COLUMNS)
    cohort["baseline_age_years"] = pd.to_numeric(cohort["r1agey"], errors="coerce")
    cohort["baseline_age_ge_min"] = cohort["baseline_age_years"].ge(min_baseline_age)
    cohort["baseline_diabetes_known"] = binary_known(cohort, BASELINE_DIABETES_COLUMNS)
    cohort["baseline_diabetes_any"] = binary_yes(cohort, BASELINE_DIABETES_COLUMNS)
    cohort["followup_2013_diabetes_known"] = binary_known(cohort, FOLLOWUP_2013_COLUMNS)
    cohort["followup_2013_diabetes_any"] = binary_yes(cohort, FOLLOWUP_2013_COLUMNS)
    cohort["followup_2015_diabetes_known"] = binary_known(cohort, FOLLOWUP_2015_COLUMNS)
    cohort["followup_2015_diabetes_any"] = binary_yes(cohort, FOLLOWUP_2015_COLUMNS)
    cohort["any_followup_diabetes_known"] = cohort["followup_2013_diabetes_known"] | cohort["followup_2015_diabetes_known"]
    cohort["incident_diabetes_2013"] = cohort["followup_2013_diabetes_any"].astype(int)
    cohort["incident_diabetes_2015"] = cohort["followup_2015_diabetes_any"].astype(int)
    cohort["incident_diabetes_2013_or_2015"] = (
        cohort["followup_2013_diabetes_any"] | cohort["followup_2015_diabetes_any"]
    ).astype(int)

    exclusions = []
    exclusions.append({"step": "raw_harmonized_rows", "remaining": len(cohort), "excluded": 0})
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
    cohort["prediction_anchor_wave"] = "2011_wave1"
    cohort["outcome_window"] = "2013_wave2_or_2015_wave3"

    rename = {
        "householdID": "household_id",
        "communityID": "community_id",
        "ragender": "sex",
        "raeduc_c": "education_category",
        "raeducl": "education_level",
        "r1smokev": "baseline_smoked_ever",
        "r1smoken": "baseline_smokes_now",
        "r1drinkl": "baseline_drinks_last_year",
        "r1drinkn_c": "baseline_drink_frequency",
        "r1mbmi": "baseline_bmi",
        "r1hibpe": "baseline_hypertension",
        "r1hearte": "baseline_heart_problem",
        "r1stroke": "baseline_stroke",
        "r1lunge": "baseline_lung_disease",
        "r1kidneye": "baseline_kidney_disease",
    }
    cohort = cohort.rename(columns=rename)
    columns = [
        "person_id",
        "household_id",
        "community_id",
        "split",
        "prediction_anchor_wave",
        "outcome_window",
        "baseline_age_years",
        "sex",
        "education_category",
        "education_level",
        "baseline_smoked_ever",
        "baseline_smokes_now",
        "baseline_drinks_last_year",
        "baseline_drink_frequency",
        "baseline_bmi",
        "baseline_hypertension",
        "baseline_heart_problem",
        "baseline_stroke",
        "baseline_lung_disease",
        "baseline_kidney_disease",
        "followup_2013_present",
        "followup_2015_present",
        "followup_2013_diabetes_known",
        "followup_2015_diabetes_known",
        "incident_diabetes_2013",
        "incident_diabetes_2015",
        "incident_diabetes_2013_or_2015",
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
        {"metric": "incident_diabetes_events", "value": int(cohort["incident_diabetes_2013_or_2015"].sum())},
        {"metric": "incident_diabetes_rate", "value": round(float(cohort["incident_diabetes_2013_or_2015"].mean()), 4)},
        {"metric": "followup_2013_known", "value": int(cohort["followup_2013_diabetes_known"].sum())},
        {"metric": "followup_2015_known", "value": int(cohort["followup_2015_diabetes_known"].sum())},
        {"metric": "excluded_prevalent_baseline_diabetes", "value": int(exclusions.loc[exclusions["step"].eq("exclude_prevalent_baseline_diabetes"), "excluded"].iloc[0])},
    ]
    for split, group in cohort.groupby("split"):
        rows.append({"metric": f"{split}_persons", "value": len(group)})
        rows.append({"metric": f"{split}_event_rate", "value": round(float(group["incident_diabetes_2013_or_2015"].mean()), 4)})
    return pd.DataFrame(rows)


def wave_outcome_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "wave": "2013_wave2",
                "known": int(cohort["followup_2013_diabetes_known"].sum()),
                "events": int(cohort["incident_diabetes_2013"].sum()),
                "event_rate": round(float(cohort.loc[cohort["followup_2013_diabetes_known"], "incident_diabetes_2013"].mean()), 4),
            },
            {
                "wave": "2015_wave3",
                "known": int(cohort["followup_2015_diabetes_known"].sum()),
                "events": int(cohort["incident_diabetes_2015"].sum()),
                "event_rate": round(float(cohort.loc[cohort["followup_2015_diabetes_known"], "incident_diabetes_2015"].mean()), 4),
            },
        ]
    )


def markdown_table(df: pd.DataFrame) -> str:
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
    output = project_root / "outputs" / "reports" / "charls_incident_diabetes_cohort_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# CHARLS Incident Diabetes Cohort Skeleton

- Source table: `{harmonized_path}`
- Output cohort: `data/processed/charls_incident_diabetes_cohort.csv`
- Prediction anchor: 2011 Wave 1 baseline.
- Outcome window: 2013 Wave 2 or 2015 Wave 3.
- Boundary: cohort construction only; no model training and no clinical recommendation.

## Cohort Definition

Respondents must be present in 2011, be at least the configured baseline age, have known baseline diabetes status, have no baseline diabetes by diagnosis or diabetes medication, and have at least one known follow-up diabetes status in 2013 or 2015.

## Summary

{markdown_table(summary)}

## Wave Outcomes

{markdown_table(wave_summary)}

## Exclusions

{markdown_table(exclusions)}

## Leakage Boundary

- Baseline features are restricted to 2011/static variables listed in the wave map.
- Follow-up variables are labels only: `{", ".join(sorted(FORBIDDEN_BASELINE_FEATURES))}`.
- `baseline_diabetes_any` is retained only for validation and should be all `False` in the cohort.
"""
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    selected_root, _ = choose_root(args.charls_root)
    if selected_root is None:
        selected_root = next((root for root in CHARLS_CANDIDATE_ROOTS if root.exists()), None)
    harmonized_path = find_harmonized_file(selected_root)
    if harmonized_path is None:
        raise FileNotFoundError("No harmonized CHARLS .dta file found. Run --charls-wave-map after downloading data.")

    raw = read_harmonized(harmonized_path)
    cohort, exclusions = build_cohort(raw, args.min_baseline_age)
    summary = summarize(cohort, exclusions, args.min_baseline_age)
    wave_summary = wave_outcome_summary(cohort)

    processed_dir = args.project_root / "data" / "processed"
    table_dir = args.project_root / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = processed_dir / "charls_incident_diabetes_cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    exclusions.to_csv(table_dir / "charls_incident_diabetes_exclusions.csv", index=False)
    summary.to_csv(table_dir / "charls_incident_diabetes_cohort_summary.csv", index=False)
    wave_summary.to_csv(table_dir / "charls_incident_diabetes_wave_outcome_summary.csv", index=False)
    report = write_report(args.project_root, harmonized_path, cohort, exclusions, summary, wave_summary)
    print(f"CHARLS incident diabetes cohort rows: {len(cohort)}")
    print(f"Events: {int(cohort['incident_diabetes_2013_or_2015'].sum())}")
    print(f"Wrote {cohort_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
