#!/usr/bin/env python3
"""Build a concrete CHARLS harmonized wave-variable map for incident diabetes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from charls_data_readiness import CHARLS_CANDIDATE_ROOTS, choose_root
from mimic_diabetes_baseline import DEFAULT_PROJECT


HARMONIZED_NAMES = ["H_CHARLS_D_Data.dta", "H_CHARLS_C_Data.dta", "H_CHARLS_B_Data.dta"]
CORE_VARIABLES = [
    ("person_id", "ID", "all", "id", "cross-wave person identifier", "allowed"),
    ("household_id", "householdID", "all", "id", "household identifier", "allowed"),
    ("community_id", "communityID", "all", "id", "community identifier", "allowed"),
    ("in_wave_2011", "inw1", "2011_wave1", "followup_status", "baseline wave participation", "allowed"),
    ("in_wave_2013", "inw2", "2013_wave2", "followup_status", "follow-up participation", "do_not_use_as_baseline_feature"),
    ("in_wave_2015", "inw3", "2015_wave3", "followup_status", "follow-up participation", "do_not_use_as_baseline_feature"),
    ("baseline_age", "r1agey", "2011_wave1", "baseline_feature", "age at baseline", "allowed"),
    ("sex", "ragender", "baseline/static", "baseline_feature", "respondent gender", "allowed"),
    ("education", "raeduc_c", "baseline/static", "baseline_feature", "respondent education category", "allowed"),
    ("education_harmonized", "raeducl", "baseline/static", "baseline_feature", "harmonized education level", "allowed"),
    ("baseline_smoked_ever", "r1smokev", "2011_wave1", "baseline_feature", "baseline smoking history", "allowed"),
    ("baseline_smokes_now", "r1smoken", "2011_wave1", "baseline_feature", "baseline current smoking", "allowed"),
    ("baseline_drinks_last_year", "r1drinkl", "2011_wave1", "baseline_feature", "baseline alcohol use", "allowed"),
    ("baseline_drink_frequency", "r1drinkn_c", "2011_wave1", "baseline_feature", "baseline drinking frequency", "allowed"),
    ("baseline_bmi", "r1mbmi", "2011_wave1", "baseline_feature", "baseline measured BMI", "allowed"),
    ("baseline_hypertension", "r1hibpe", "2011_wave1", "baseline_feature", "baseline high blood pressure history", "allowed"),
    ("baseline_heart_problem", "r1hearte", "2011_wave1", "baseline_feature", "baseline heart problem history", "allowed"),
    ("baseline_stroke", "r1stroke", "2011_wave1", "baseline_feature", "baseline stroke history", "allowed"),
    ("baseline_lung_disease", "r1lunge", "2011_wave1", "baseline_feature", "baseline lung disease history", "allowed"),
    ("baseline_kidney_disease", "r1kidneye", "2011_wave1", "baseline_feature", "baseline kidney disease history", "allowed"),
    ("baseline_diabetes_diagnosis", "r1diabe", "2011_wave1", "baseline_exclusion", "exclude prevalent baseline diabetes", "label_boundary_only"),
    ("baseline_diabetes_any_med", "r1rxdiab_c", "2011_wave1", "baseline_exclusion", "baseline diabetes medication", "label_boundary_only"),
    ("baseline_diabetes_modern_med", "r1rxdiab", "2011_wave1", "baseline_exclusion", "baseline modern diabetes medication", "label_boundary_only"),
    ("baseline_diabetes_insulin", "r1rxdiabi", "2011_wave1", "baseline_exclusion", "baseline insulin use", "label_boundary_only"),
    ("followup_2013_diabetes_diagnosis", "r2diabe", "2013_wave2", "outcome", "incident diabetes component", "forbidden_as_baseline_feature"),
    ("followup_2013_diabetes_any_med", "r2rxdiab_c", "2013_wave2", "outcome", "incident diabetes medication component", "forbidden_as_baseline_feature"),
    ("followup_2013_diabetes_modern_med", "r2rxdiab", "2013_wave2", "outcome", "incident diabetes medication component", "forbidden_as_baseline_feature"),
    ("followup_2013_diabetes_insulin", "r2rxdiabi", "2013_wave2", "outcome", "incident diabetes insulin component", "forbidden_as_baseline_feature"),
    ("followup_2015_diabetes_diagnosis", "r3diabe", "2015_wave3", "outcome", "incident diabetes component", "forbidden_as_baseline_feature"),
    ("followup_2015_diabetes_any_med", "r3rxdiab_c", "2015_wave3", "outcome", "incident diabetes medication component", "forbidden_as_baseline_feature"),
    ("followup_2015_diabetes_modern_med", "r3rxdiab", "2015_wave3", "outcome", "incident diabetes medication component", "forbidden_as_baseline_feature"),
    ("followup_2015_diabetes_insulin", "r3rxdiabi", "2015_wave3", "outcome", "incident diabetes insulin component", "forbidden_as_baseline_feature"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--charls-root", type=Path, help="Optional explicit CHARLS root.")
    return parser.parse_args()


def find_harmonized_file(root: Path | None) -> Path | None:
    if root is None or not root.exists():
        return None
    for name in HARMONIZED_NAMES:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    matches = sorted(path for path in root.rglob("*.dta") if "harmon" in str(path).lower())
    return matches[0] if matches else None


def metadata(path: Path) -> tuple[list[str], dict[str, str]]:
    reader = pd.io.stata.StataReader(path)
    labels = reader.variable_labels()
    return list(labels), labels


def build_map(harmonized_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns, labels = metadata(harmonized_path)
    rows = []
    for concept, variable, wave, role, description, leakage_status in CORE_VARIABLES:
        present = variable in labels
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
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for item in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, selected_root: Path | None, harmonized_path: Path, wave_map: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "charls_wave_variable_map.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        wave_map.groupby(["wave", "role", "leakage_status"], dropna=False)
        .agg(variables=("variable", "count"), present=("present", "sum"))
        .reset_index()
    )
    text = f"""# CHARLS Wave Variable Map

- Selected CHARLS root: `{selected_root}`
- Harmonized source: `{harmonized_path}`
- Boundary: maps variables and leakage roles only; no model training and no clinical recommendation.

## Summary

{markdown_table(summary)}

## Core Map

{markdown_table(wave_map)}

## Next Step

Use this map to build the 2011 baseline -> 2013/2015 incident diabetes cohort. Variables marked `forbidden_as_baseline_feature` may define outcomes but must not enter baseline feature matrices.
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
        raise FileNotFoundError("No harmonized CHARLS .dta file found. Run --charls-readiness after downloading data.")

    wave_map, inventory = build_map(harmonized_path)
    table_dir = args.project_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    map_path = table_dir / "charls_wave_variable_map.csv"
    inv_path = table_dir / "charls_harmonized_variable_inventory.csv"
    wave_map.to_csv(map_path, index=False)
    inventory.to_csv(inv_path, index=False)
    report = write_report(args.project_root, selected_root, harmonized_path, wave_map)
    missing = int((~wave_map["present"]).sum())
    print(f"CHARLS wave map variables: {len(wave_map)}")
    print(f"Missing mapped variables: {missing}")
    print(f"Wrote {map_path}")
    print(f"Wrote {report}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
