#!/usr/bin/env python3
"""Build a CHARLS 2011 baseline feature matrix for incident diabetes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


FEATURE_MAP = {
    "baseline_age_years": "charls_baseline_age_years",
    "sex": "charls_baseline_sex_code",
    "education_category": "charls_baseline_education_category",
    "education_level": "charls_baseline_education_level",
    "baseline_smoked_ever": "charls_baseline_smoked_ever",
    "baseline_smokes_now": "charls_baseline_smokes_now",
    "baseline_drinks_last_year": "charls_baseline_drinks_last_year",
    "baseline_drink_frequency": "charls_baseline_drink_frequency",
    "baseline_bmi": "charls_baseline_bmi",
    "baseline_hypertension": "charls_baseline_hypertension",
    "baseline_heart_problem": "charls_baseline_heart_problem",
    "baseline_stroke": "charls_baseline_stroke",
    "baseline_lung_disease": "charls_baseline_lung_disease",
    "baseline_kidney_disease": "charls_baseline_kidney_disease",
}
ID_COLUMNS = ["person_id", "household_id", "community_id", "split"]
LABEL_COLUMN = "incident_diabetes_2013_or_2015"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def build_feature_matrix(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [column for column in [*ID_COLUMNS, LABEL_COLUMN, *FEATURE_MAP] if column not in cohort.columns]
    if missing:
        raise ValueError(f"cohort is missing required columns: {missing}")
    matrix = cohort[[*ID_COLUMNS, LABEL_COLUMN]].copy()
    manifest_rows = []
    missing_rows = []
    for source, output in FEATURE_MAP.items():
        values = pd.to_numeric(cohort[source], errors="coerce")
        matrix[output] = values
        matrix[f"{output}_missing"] = values.isna().astype(int)
        missing_rows.append(
            {
                "feature": output,
                "source_column": source,
                "missing_count": int(values.isna().sum()),
                "missing_rate": round(float(values.isna().mean()), 4),
                "nonmissing_count": int(values.notna().sum()),
            }
        )
        manifest_rows.append(
            {
                "feature": output,
                "source_column": source,
                "role": "baseline_feature",
                "wave": "2011_wave1_or_static",
                "allowed_as_feature": True,
                "imputation_note": "Raw numeric/coded value; pair with *_missing indicator before modeling.",
            }
        )
        manifest_rows.append(
            {
                "feature": f"{output}_missing",
                "source_column": source,
                "role": "baseline_missingness_indicator",
                "wave": "2011_wave1_or_static",
                "allowed_as_feature": True,
                "imputation_note": "Missingness indicator generated from baseline source column.",
            }
        )
    return matrix.sort_values("person_id"), pd.DataFrame(manifest_rows), pd.DataFrame(missing_rows)


def summarize(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "rows", "value": len(matrix)},
        {"metric": "unique_persons", "value": matrix["person_id"].nunique()},
        {"metric": "feature_columns", "value": len([c for c in matrix.columns if c.startswith("charls_baseline_")])},
        {"metric": "events", "value": int(matrix[LABEL_COLUMN].sum())},
        {"metric": "event_rate", "value": round(float(matrix[LABEL_COLUMN].mean()), 4)},
    ]
    for split, group in matrix.groupby("split"):
        rows.append({"metric": f"{split}_rows", "value": len(group)})
        rows.append({"metric": f"{split}_event_rate", "value": round(float(group[LABEL_COLUMN].mean()), 4)})
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for item in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, summary: pd.DataFrame, missingness: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "charls_baseline_features_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# CHARLS Baseline Feature Matrix

- Source cohort: `data/processed/charls_incident_diabetes_cohort.csv`
- Output matrix: `data/processed/charls_incident_diabetes_baseline_features.csv`
- Prediction anchor: 2011 Wave 1 baseline.
- Boundary: baseline feature construction only; no model training and no clinical recommendation.

## Summary

{markdown_table(summary)}

## Missingness

{markdown_table(missingness)}

## Feature Manifest

{markdown_table(manifest)}

## Leakage Boundary

Only columns with the `charls_baseline_` prefix are candidate features. Follow-up indicators and 2013/2015 outcome components stay out of the matrix except for the single label column `{LABEL_COLUMN}`.
"""
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    cohort_path = args.project_root / "data" / "processed" / "charls_incident_diabetes_cohort.csv"
    cohort = read_csv(cohort_path)
    if cohort.empty:
        raise FileNotFoundError(f"Missing CHARLS cohort: {cohort_path}")
    matrix, manifest, missingness = build_feature_matrix(cohort)
    summary = summarize(matrix)

    processed_dir = args.project_root / "data" / "processed"
    table_dir = args.project_root / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = processed_dir / "charls_incident_diabetes_baseline_features.csv"
    matrix.to_csv(matrix_path, index=False)
    manifest.to_csv(table_dir / "charls_baseline_feature_manifest.csv", index=False)
    missingness.to_csv(table_dir / "charls_baseline_feature_missingness.csv", index=False)
    summary.to_csv(table_dir / "charls_baseline_feature_summary.csv", index=False)
    report = write_report(args.project_root, summary, missingness, manifest)
    print(f"CHARLS baseline feature matrix rows: {len(matrix)}")
    print(f"Feature columns: {len([c for c in matrix.columns if c.startswith('charls_baseline_')])}")
    print(f"Wrote {matrix_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
