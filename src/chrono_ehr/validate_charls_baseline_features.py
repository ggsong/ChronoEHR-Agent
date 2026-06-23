#!/usr/bin/env python3
"""Validate the CHARLS baseline feature matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


LABEL_COLUMN = "incident_diabetes_2013_or_2015"
REQUIRED_COLUMNS = {"person_id", "split", LABEL_COLUMN}
FORBIDDEN_TOKENS = ["followup", "2013", "2015", "r2", "r3", "diabetes_2013", "diabetes_2015"]


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


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    return [column for column in matrix.columns if column.startswith("charls_baseline_")]


def validate(project_root: Path) -> pd.DataFrame:
    cohort_path = project_root / "data" / "processed" / "charls_incident_diabetes_cohort.csv"
    matrix_path = project_root / "data" / "processed" / "charls_incident_diabetes_baseline_features.csv"
    manifest_path = project_root / "outputs" / "tables" / "charls_baseline_feature_manifest.csv"
    missingness_path = project_root / "outputs" / "tables" / "charls_baseline_feature_missingness.csv"
    report_path = project_root / "outputs" / "reports" / "charls_baseline_features_report.md"
    cohort = read_csv(cohort_path)
    matrix = read_csv(matrix_path)
    manifest = read_csv(manifest_path)
    missingness = read_csv(missingness_path)
    rows = [
        row("matrix_exists", "PASS" if not matrix.empty else "FAIL", str(matrix_path), f"rows={len(matrix)}"),
        row("manifest_exists", "PASS" if not manifest.empty else "FAIL", str(manifest_path), f"rows={len(manifest)}"),
        row("missingness_exists", "PASS" if not missingness.empty else "FAIL", str(missingness_path), f"rows={len(missingness)}"),
        row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
    ]
    if matrix.empty:
        return pd.DataFrame(rows)

    missing = sorted(REQUIRED_COLUMNS - set(matrix.columns))
    rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(matrix_path), "missing=" + ",".join(missing)))
    if missing:
        return pd.DataFrame(rows)

    features = feature_columns(matrix)
    rows.append(row("feature_columns_present", "PASS" if features else "FAIL", str(matrix_path), f"features={len(features)}"))
    rows.append(row("row_count_matches_cohort", "PASS" if not cohort.empty and len(matrix) == len(cohort) else "FAIL", str(matrix_path), f"matrix={len(matrix)} cohort={len(cohort)}"))
    rows.append(row("unique_person_id", "PASS" if matrix["person_id"].nunique() == len(matrix) else "FAIL", str(matrix_path), f"unique={matrix['person_id'].nunique()} rows={len(matrix)}"))
    label_values = sorted(matrix[LABEL_COLUMN].dropna().unique().tolist())
    rows.append(row("binary_label", "PASS" if set(label_values).issubset({0, 1}) and len(label_values) == 2 else "FAIL", str(matrix_path), f"values={label_values}"))
    split_values = set(matrix["split"].astype(str).unique())
    rows.append(row("expected_splits", "PASS" if split_values == {"train", "validation", "test"} else "FAIL", str(matrix_path), "splits=" + ",".join(sorted(split_values))))
    bad_feature_tokens = [
        column
        for column in features
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    rows.append(row("no_future_tokens_in_feature_columns", "PASS" if not bad_feature_tokens else "FAIL", str(matrix_path), "bad=" + ",".join(bad_feature_tokens[:20])))
    non_feature_payload = sorted(
        set(matrix.columns)
        - {"person_id", "household_id", "community_id", "split", LABEL_COLUMN}
        - set(features)
    )
    rows.append(row("only_expected_payload_columns", "PASS" if not non_feature_payload else "FAIL", str(matrix_path), "extra=" + ",".join(non_feature_payload)))
    if not manifest.empty and {"feature", "role", "allowed_as_feature"}.issubset(manifest.columns):
        disallowed = manifest[manifest["allowed_as_feature"].astype(str).str.lower().ne("true")]
        rows.append(row("manifest_allows_only_baseline_features", "PASS" if disallowed.empty else "FAIL", str(manifest_path), f"disallowed={len(disallowed)}"))
        missing_manifest = sorted(set(features) - set(manifest["feature"].astype(str)))
        rows.append(row("all_features_in_manifest", "PASS" if not missing_manifest else "FAIL", str(manifest_path), "missing=" + ",".join(missing_manifest[:20])))
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
    table_path = args.project_root / "outputs" / "tables" / "charls_baseline_features_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "charls_baseline_features_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CHARLS Baseline Feature Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates baseline feature construction only; no model training.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CHARLS baseline feature checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
