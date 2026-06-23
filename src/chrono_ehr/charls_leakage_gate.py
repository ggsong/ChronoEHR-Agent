#!/usr/bin/env python3
"""Run leakage gates for the CHARLS incident diabetes baseline slice."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


CRITICAL = "critical"
WARNING = "warning"
PASS = "pass"
LABEL_COLUMN = "incident_diabetes_2013_or_2015"
ALLOWED_NON_FEATURE_COLUMNS = {"person_id", "household_id", "community_id", "split", LABEL_COLUMN}
FORBIDDEN_PATTERNS = [
    "followup",
    "2013",
    "2015",
    "wave2",
    "wave3",
    "r2",
    "r3",
    "inw2",
    "inw3",
    "future",
    "outcome_window",
    "baseline_diabetes_any",
    "incident_diabetes_2013",
    "incident_diabetes_2015",
]


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


def row(check: str, severity: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "severity": severity, "status": status, "evidence": evidence, "detail": detail}


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    return [column for column in matrix.columns if column.startswith("charls_baseline_")]


def audit(project_root: Path) -> pd.DataFrame:
    matrix_path = project_root / "data" / "processed" / "charls_incident_diabetes_baseline_features.csv"
    manifest_path = project_root / "outputs" / "tables" / "charls_baseline_feature_manifest.csv"
    validation_path = project_root / "outputs" / "tables" / "charls_baseline_features_validation.csv"
    wave_map_path = project_root / "outputs" / "tables" / "charls_wave_variable_map.csv"
    cohort_path = project_root / "data" / "processed" / "charls_incident_diabetes_cohort.csv"

    matrix = read_csv(matrix_path)
    manifest = read_csv(manifest_path)
    validation = read_csv(validation_path)
    wave_map = read_csv(wave_map_path)
    cohort = read_csv(cohort_path)

    rows = [
        row("feature_matrix_exists", PASS if not matrix.empty else CRITICAL, "allowed" if not matrix.empty else "blocked", str(matrix_path), f"rows={len(matrix)}"),
        row("feature_validation_passed", PASS if not validation.empty and validation["status"].eq("PASS").all() else CRITICAL, "allowed" if not validation.empty and validation["status"].eq("PASS").all() else "blocked", str(validation_path), f"rows={len(validation)}"),
    ]
    if matrix.empty:
        return pd.DataFrame(rows)

    features = feature_columns(matrix)
    rows.append(row("feature_prefixes_are_baseline_only", PASS if features and len(features) == len(set(features)) else CRITICAL, "allowed" if features and len(features) == len(set(features)) else "blocked", str(matrix_path), f"features={len(features)}"))
    bad_features = [column for column in features if any(pattern in column.lower() for pattern in FORBIDDEN_PATTERNS)]
    rows.append(row("no_future_or_outcome_feature_names", PASS if not bad_features else CRITICAL, "allowed" if not bad_features else "blocked", str(matrix_path), "bad=" + ",".join(bad_features[:20])))
    extra_payload = sorted(set(matrix.columns) - ALLOWED_NON_FEATURE_COLUMNS - set(features))
    rows.append(row("no_unclassified_payload_columns", PASS if not extra_payload else CRITICAL, "allowed" if not extra_payload else "blocked", str(matrix_path), "extra=" + ",".join(extra_payload[:20])))
    label_values = sorted(matrix[LABEL_COLUMN].dropna().unique().tolist()) if LABEL_COLUMN in matrix else []
    rows.append(row("label_is_single_allowed_outcome", PASS if set(label_values).issubset({0, 1}) and len(label_values) == 2 else CRITICAL, "allowed" if set(label_values).issubset({0, 1}) and len(label_values) == 2 else "blocked", str(matrix_path), f"label_values={label_values}"))

    if not manifest.empty and {"feature", "role", "wave", "allowed_as_feature"}.issubset(manifest.columns):
        bad_manifest = manifest[
            ~manifest["feature"].astype(str).isin(features)
            | ~manifest["role"].astype(str).str.startswith("baseline", na=False)
            | manifest["wave"].astype(str).str.contains("2013|2015|wave2|wave3", case=False, regex=True, na=False)
            | manifest["allowed_as_feature"].astype(str).str.lower().ne("true")
        ]
        rows.append(row("manifest_declares_baseline_only", PASS if bad_manifest.empty else CRITICAL, "allowed" if bad_manifest.empty else "blocked", str(manifest_path), f"bad_rows={len(bad_manifest)}"))
    else:
        rows.append(row("feature_manifest_present", CRITICAL, "blocked", str(manifest_path), "missing manifest columns"))

    if not wave_map.empty and {"role", "leakage_status", "variable"}.issubset(wave_map.columns):
        future = wave_map[wave_map["leakage_status"].astype(str).str.contains("forbidden|do_not_use", case=False, regex=True, na=False)]
        leaked_names = sorted(set(future["variable"].astype(str)) & set(matrix.columns))
        rows.append(row("wave_map_forbidden_variables_absent", PASS if not leaked_names else CRITICAL, "allowed" if not leaked_names else "blocked", str(wave_map_path), "leaked=" + ",".join(leaked_names)))
    else:
        rows.append(row("wave_map_present", WARNING, "review", str(wave_map_path), "missing wave map"))

    if not cohort.empty:
        rows.append(row("matrix_rows_match_cohort", PASS if len(matrix) == len(cohort) else CRITICAL, "allowed" if len(matrix) == len(cohort) else "blocked", str(cohort_path), f"matrix={len(matrix)} cohort={len(cohort)}"))
        if "baseline_diabetes_any" in cohort:
            prevalent = int(cohort["baseline_diabetes_any"].astype(bool).sum())
            rows.append(row("baseline_prevalent_diabetes_excluded", PASS if prevalent == 0 else CRITICAL, "allowed" if prevalent == 0 else "blocked", str(cohort_path), f"prevalent_rows={prevalent}"))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "severity", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    blocked = checks[checks["status"].eq("blocked")]
    table_path = args.project_root / "outputs" / "tables" / "charls_leakage_gate.csv"
    report_path = args.project_root / "outputs" / "reports" / "charls_leakage_gate_report.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CHARLS Incident Diabetes Leakage Gate

- Overall status: `{"PASS" if blocked.empty else "FAIL"}`
- Checks: {len(checks)}
- Blocked checks: {len(blocked)}
- Boundary: research data timing/leakage audit only; no model training or clinical recommendation.

## Interpretation

This gate blocks 2013/2015 follow-up variables, future wave indicators, raw outcome components, and baseline prevalent diabetes flags from entering the 2011 baseline feature matrix.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CHARLS leakage gate checks: {len(checks)}")
    print(f"Blocked: {len(blocked)}")
    print(f"Wrote {report_path}")
    if not blocked.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
