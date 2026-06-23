#!/usr/bin/env python3
"""Validate the eICU temporal mortality cohort skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_COLUMNS = {
    "stay_id",
    "patient_id",
    "split",
    "age_years",
    "icu_admission_prediction_offset",
    "first_24h_prediction_offset",
    "unit_los_minutes",
    "eligible_admission_prediction",
    "eligible_first_24h_prediction",
    "hospital_mortality",
    "icu_mortality",
}


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


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    cohort_path = project_root / "data" / "processed" / "eicu_temporal_mortality_cohort.csv"
    summary_path = project_root / "outputs" / "tables" / "eicu_temporal_mortality_cohort_summary.csv"
    cohort = read_csv(cohort_path)
    summary = read_csv(summary_path)
    rows = [
        row("cohort_exists", "PASS" if not cohort.empty else "FAIL", str(cohort_path), f"rows={len(cohort)}"),
        row("summary_exists", "PASS" if not summary.empty else "FAIL", str(summary_path), f"rows={len(summary)}"),
    ]
    if cohort.empty:
        return pd.DataFrame(rows)

    missing = sorted(REQUIRED_COLUMNS - set(cohort.columns))
    rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(cohort_path), "missing=" + ",".join(missing)))
    if missing:
        return pd.DataFrame(rows)

    rows.append(row("adult_only", "PASS" if cohort["age_years"].min() >= 18 else "FAIL", str(cohort_path), f"min_age={cohort['age_years'].min()}"))
    label_values = sorted(cohort["hospital_mortality"].dropna().unique().tolist())
    rows.append(row("binary_hospital_mortality", "PASS" if set(label_values).issubset({0, 1}) and len(label_values) == 2 else "FAIL", str(cohort_path), f"values={label_values}"))
    rows.append(row("admission_offset_zero", "PASS" if cohort["icu_admission_prediction_offset"].eq(0).all() else "FAIL", str(cohort_path), "all admission offsets should be 0"))
    rows.append(row("first_24h_offset_1440", "PASS" if cohort["first_24h_prediction_offset"].eq(1440).all() else "FAIL", str(cohort_path), "all first-24h offsets should be 1440 minutes"))
    rows.append(row("positive_icu_los", "PASS" if cohort["unit_los_minutes"].gt(0).all() else "FAIL", str(cohort_path), f"min_los={cohort['unit_los_minutes'].min()}"))
    eligible_match = cohort["eligible_first_24h_prediction"].astype(bool).eq(cohort["unit_los_minutes"].ge(1440))
    rows.append(row("first_24h_eligibility_matches_los", "PASS" if eligible_match.all() else "FAIL", str(cohort_path), f"bad_rows={int((~eligible_match).sum())}"))
    split_values = set(cohort["split"].astype(str).unique())
    rows.append(row("expected_splits", "PASS" if split_values == {"train", "validation", "test"} else "FAIL", str(cohort_path), "splits=" + ",".join(sorted(split_values))))
    patient_split_counts = cohort.groupby("patient_id")["split"].nunique()
    overlap = int(patient_split_counts.gt(1).sum())
    rows.append(row("no_patient_split_overlap", "PASS" if overlap == 0 else "FAIL", str(cohort_path), f"overlap_patients={overlap}"))
    forbidden_raw = {"hospitaldischargestatus", "unitdischargestatus"}
    present_forbidden_raw = sorted(forbidden_raw & set(cohort.columns))
    rows.append(row("raw_outcome_strings_removed", "PASS" if not present_forbidden_raw else "FAIL", str(cohort_path), "present=" + ",".join(present_forbidden_raw)))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = validate(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "eicu_temporal_mortality_cohort_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "eicu_temporal_mortality_cohort_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# eICU Temporal Mortality Cohort Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates cohort construction only; no model training or clinical recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"eICU cohort validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
