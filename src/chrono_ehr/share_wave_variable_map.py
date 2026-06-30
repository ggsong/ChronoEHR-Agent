#!/usr/bin/env python3
"""Build a SHARE harmonized wave-variable map for incident diabetes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from share_data_readiness import SHARE_CANDIDATE_ROOTS, choose_root, find_harmonized_file


CORE_VARIABLES = [
    ("person_id", "mergeid", "all", "id", "cross-wave person identifier", "allowed"),
    ("household_id", "hhid", "all", "id", "household identifier", "allowed"),
    ("country", "country", "baseline/static", "baseline_feature", "country identifier", "allowed"),
    ("in_wave1", "inw1", "wave1", "baseline_membership", "baseline wave participation", "allowed"),
    ("in_wave2", "inw2", "wave2", "followup_status", "follow-up participation", "do_not_use_as_baseline_feature"),
    ("in_wave4", "inw4", "wave4", "followup_status", "follow-up participation", "do_not_use_as_baseline_feature"),
    ("baseline_age", "r1agey", "wave1", "baseline_feature", "age at baseline", "allowed"),
    ("sex", "ragender", "baseline/static", "baseline_feature", "respondent gender", "allowed"),
    ("education_years", "raedyrs", "baseline/static", "baseline_feature", "years of education", "allowed"),
    ("education_level", "raeducl", "baseline/static", "baseline_feature", "harmonized education level", "allowed"),
    ("baseline_smoked_ever", "r1smokev", "wave1", "baseline_feature", "baseline smoking history", "allowed"),
    ("baseline_smokes_now", "r1smoken", "wave1", "baseline_feature", "baseline current smoking", "allowed"),
    ("baseline_drinks_recently", "r1drink3m", "wave1", "baseline_feature", "baseline alcohol use", "allowed"),
    ("baseline_drink_frequency", "r1drinkx", "wave1", "baseline_feature", "baseline drinking frequency", "allowed"),
    ("baseline_bmi", "r1bmi", "wave1", "baseline_feature", "baseline BMI", "allowed"),
    ("baseline_hypertension", "r1hibpe", "wave1", "baseline_feature", "baseline high blood pressure history", "allowed"),
    ("baseline_heart_problem", "r1hearte", "wave1", "baseline_feature", "baseline heart problem history", "allowed"),
    ("baseline_stroke", "r1stroke", "wave1", "baseline_feature", "baseline stroke history", "allowed"),
    ("baseline_lung_disease", "r1lunge", "wave1", "baseline_feature", "baseline lung disease history", "allowed"),
    ("baseline_adl_limitations", "r1adla", "wave1", "baseline_feature", "baseline ADL limitations", "allowed"),
    ("baseline_iadl_limitations", "r1iadla", "wave1", "baseline_feature", "baseline IADL limitations", "allowed"),
    ("baseline_mobility_limitations", "r1mobilb", "wave1", "baseline_feature", "baseline mobility limitations", "allowed"),
    ("baseline_depression", "r1depress", "wave1", "baseline_feature", "baseline depression indicator", "allowed"),
    ("baseline_person_weight", "r1wtresp", "wave1", "survey_design", "baseline person-level weight", "allowed"),
    ("baseline_diabetes_diagnosis", "r1diabe", "wave1", "baseline_exclusion", "exclude prevalent baseline diabetes", "label_boundary_only"),
    ("baseline_diabetes_medication", "r1rxdiab", "wave1", "baseline_exclusion", "baseline diabetes medication", "label_boundary_only"),
    ("followup_wave2_diabetes_diagnosis", "r2diabe", "wave2", "outcome", "incident diabetes component", "forbidden_as_baseline_feature"),
    ("followup_wave2_diabetes_medication", "r2rxdiab", "wave2", "outcome", "incident diabetes medication component", "forbidden_as_baseline_feature"),
    ("followup_wave4_diabetes_diagnosis", "r4diabe", "wave4", "outcome", "incident diabetes component", "forbidden_as_baseline_feature"),
    ("followup_wave4_diabetes_medication", "r4rxdiab", "wave4", "outcome", "incident diabetes medication component", "forbidden_as_baseline_feature"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--share-root", type=Path, help="Optional explicit SHARE root.")
    return parser.parse_args()


def metadata(path: Path) -> tuple[list[str], dict[str, str]]:
    reader = pd.io.stata.StataReader(path)
    labels = reader.variable_labels()
    return list(labels), labels


def build_map(harmonized_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns, labels = metadata(harmonized_path)
    column_set = set(columns)
    rows = []
    for concept, variable, wave, role, description, leakage_status in CORE_VARIABLES:
        present = variable in column_set
        rows.append(
            {
                "concept": concept,
                "source_file": str(harmonized_path),
                "variable": variable,
                "wave": wave,
                "role": role,
                "description": description,
                "variable_label": labels.get(variable, ""),
                "present": present,
                "usable": present and leakage_status != "forbidden_as_baseline_feature",
                "leakage_status": leakage_status,
            }
        )
    inventory = pd.DataFrame(
        [
            {
                "source_file": str(harmonized_path),
                "variable": variable,
                "variable_label": labels.get(variable, ""),
                "matched_core_map": variable in {item[1] for item in CORE_VARIABLES},
            }
            for variable in columns
        ]
    )
    return pd.DataFrame(rows), inventory


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for item in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, selected_root: Path | None, harmonized_path: Path, wave_map: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "share_wave_variable_map.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        wave_map.groupby(["wave", "role", "leakage_status"], dropna=False)
        .agg(variables=("variable", "count"), present=("present", "sum"))
        .reset_index()
    )
    text = f"""# SHARE Wave Variable Map

- Selected SHARE root: `{selected_root}`
- Harmonized source: `{harmonized_path}`
- Boundary: maps variables and leakage roles only; no model training and no clinical recommendation.

## Summary

{markdown_table(summary)}

## Core Map

{markdown_table(wave_map)}

## Next Step

Use this map to build the wave 1 baseline -> wave 2/wave 4 incident diabetes cohort skeleton. Variables marked `forbidden_as_baseline_feature` may define outcomes but must not enter baseline feature matrices.
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
        raise FileNotFoundError("No harmonized SHARE .dta file found. Set SHARE_ROOT or run --share-readiness after downloading data.")

    wave_map, inventory = build_map(harmonized_path)
    table_dir = args.project_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    map_path = table_dir / "share_wave_variable_map.csv"
    inv_path = table_dir / "share_harmonized_variable_inventory.csv"
    wave_map.to_csv(map_path, index=False)
    inventory.to_csv(inv_path, index=False)
    report = write_report(args.project_root, selected_root, harmonized_path, wave_map)
    missing = int((~wave_map["present"]).sum())
    print(f"SHARE wave map variables: {len(wave_map)}")
    print(f"Missing mapped variables: {missing}")
    print(f"Wrote {map_path}")
    print(f"Wrote {report}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
