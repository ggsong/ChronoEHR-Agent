#!/usr/bin/env python3
"""Validate the CHARLS incident diabetes cohort skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_COLUMNS = {
    "person_id",
    "split",
    "prediction_anchor_wave",
    "outcome_window",
    "baseline_age_years",
    "followup_2013_diabetes_known",
    "followup_2015_diabetes_known",
    "incident_diabetes_2013",
    "incident_diabetes_2015",
    "incident_diabetes_2013_or_2015",
    "baseline_diabetes_any",
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
    cohort_path = project_root / "data" / "processed" / "charls_incident_diabetes_cohort.csv"
    summary_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_cohort_summary.csv"
    wave_summary_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_wave_outcome_summary.csv"
    report_path = project_root / "outputs" / "reports" / "charls_incident_diabetes_cohort_report.md"
    cohort = read_csv(cohort_path)
    summary = read_csv(summary_path)
    wave_summary = read_csv(wave_summary_path)
    rows = [
        row("cohort_exists", "PASS" if not cohort.empty else "FAIL", str(cohort_path), f"rows={len(cohort)}"),
        row("summary_exists", "PASS" if not summary.empty else "FAIL", str(summary_path), f"rows={len(summary)}"),
        row("wave_summary_exists", "PASS" if not wave_summary.empty else "FAIL", str(wave_summary_path), f"rows={len(wave_summary)}"),
        row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
    ]
    if cohort.empty:
        return pd.DataFrame(rows)

    missing = sorted(REQUIRED_COLUMNS - set(cohort.columns))
    rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(cohort_path), "missing=" + ",".join(missing)))
    if missing:
        return pd.DataFrame(rows)

    rows.append(row("unique_person_id", "PASS" if cohort["person_id"].nunique() == len(cohort) else "FAIL", str(cohort_path), f"unique={cohort['person_id'].nunique()} rows={len(cohort)}"))
    rows.append(row("baseline_age_min_45", "PASS" if cohort["baseline_age_years"].min() >= 45 else "FAIL", str(cohort_path), f"min_age={cohort['baseline_age_years'].min()}"))
    baseline_diabetes_sum = int(cohort["baseline_diabetes_any"].astype(bool).sum())
    rows.append(row("baseline_diabetes_excluded", "PASS" if baseline_diabetes_sum == 0 else "FAIL", str(cohort_path), f"baseline_diabetes_rows={baseline_diabetes_sum}"))
    label_values = sorted(cohort["incident_diabetes_2013_or_2015"].dropna().unique().tolist())
    rows.append(row("binary_incident_label", "PASS" if set(label_values).issubset({0, 1}) and len(label_values) == 2 else "FAIL", str(cohort_path), f"values={label_values}"))
    followup_known = cohort["followup_2013_diabetes_known"].astype(bool) | cohort["followup_2015_diabetes_known"].astype(bool)
    rows.append(row("followup_known_for_all", "PASS" if followup_known.all() else "FAIL", str(cohort_path), f"bad_rows={int((~followup_known).sum())}"))
    split_values = set(cohort["split"].astype(str).unique())
    rows.append(row("expected_splits", "PASS" if split_values == {"train", "validation", "test"} else "FAIL", str(cohort_path), "splits=" + ",".join(sorted(split_values))))
    forbidden = {"r2diabe", "r3diabe", "r2rxdiab_c", "r3rxdiab_c", "inw2", "inw3"}
    present_forbidden = sorted(forbidden & set(cohort.columns))
    rows.append(row("future_raw_fields_not_exported", "PASS" if not present_forbidden else "FAIL", str(cohort_path), "present=" + ",".join(present_forbidden)))
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
    table_path = args.project_root / "outputs" / "tables" / "charls_incident_diabetes_cohort_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "charls_incident_diabetes_cohort_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CHARLS Incident Diabetes Cohort Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates cohort construction only; no model training or clinical recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CHARLS cohort validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
