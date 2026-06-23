#!/usr/bin/env python3
"""Run leakage gates for the eICU first-24h mortality slice."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


CRITICAL = "critical"
WARNING = "warning"
PASS = "pass"

FORBIDDEN_FEATURE_PATTERNS = [
    "hospitaldischargestatus",
    "unitdischargestatus",
    "hospitaldischargeoffset",
    "unitdischargeoffset",
    "actualhospitalmortality",
    "actualicumortality",
    "icu_mortality",
    "discharge",
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


def forbidden_columns(columns: list[str]) -> list[str]:
    bad = []
    for column in columns:
        lowered = column.lower()
        if column == "hospital_mortality":
            continue
        if any(pattern in lowered for pattern in FORBIDDEN_FEATURE_PATTERNS):
            bad.append(column)
    return bad


def audit(project_root: Path) -> pd.DataFrame:
    cohort_path = project_root / "data" / "processed" / "eicu_temporal_mortality_cohort.csv"
    matrix_path = project_root / "data" / "processed" / "eicu_first24h_feature_matrix_skeleton.csv"
    stats_path = project_root / "outputs" / "tables" / "eicu_temporal_features_24h_extraction_stats.csv"
    validation_path = project_root / "outputs" / "tables" / "eicu_temporal_features_24h_validation.csv"
    role_path = project_root / "outputs" / "tables" / "external_field_role_catalog.csv"

    cohort = read_csv(cohort_path)
    matrix = read_csv(matrix_path)
    stats = read_csv(stats_path)
    validation = read_csv(validation_path)
    roles = read_csv(role_path)

    rows = [
        row("cohort_exists", PASS if not cohort.empty else CRITICAL, "allowed" if not cohort.empty else "blocked", str(cohort_path), f"rows={len(cohort)}"),
        row("feature_matrix_exists", PASS if not matrix.empty else CRITICAL, "allowed" if not matrix.empty else "blocked", str(matrix_path), f"rows={len(matrix)}"),
        row("feature_validation_passed", PASS if not validation.empty and validation["status"].eq("PASS").all() else CRITICAL, "allowed" if not validation.empty and validation["status"].eq("PASS").all() else "blocked", str(validation_path), f"rows={len(validation)}"),
    ]
    if matrix.empty:
        return pd.DataFrame(rows)

    bad = forbidden_columns(matrix.columns.tolist())
    rows.append(row("no_discharge_or_outcome_proxy_features", PASS if not bad else CRITICAL, "allowed" if not bad else "blocked", str(matrix_path), "bad=" + ",".join(bad[:20])))

    feature_cols = [column for column in matrix.columns if column not in {"stay_id", "patient_id", "split", "hospital_mortality"}]
    non_temporal = [column for column in feature_cols if not column.startswith(("eicu_lab24h_", "eicu_vital24h_"))]
    rows.append(row("feature_prefixes_encode_first24h_window", PASS if not non_temporal else CRITICAL, "allowed" if not non_temporal else "blocked", str(matrix_path), f"non_temporal={len(non_temporal)}"))

    if not stats.empty:
        max_offset = pd.to_numeric(stats["max_included_offset"], errors="coerce").max()
        min_offset = pd.to_numeric(stats["min_included_offset"], errors="coerce").min()
        rows.append(row("no_events_after_prediction_window", PASS if pd.notna(max_offset) and max_offset <= 1440 else CRITICAL, "allowed" if pd.notna(max_offset) and max_offset <= 1440 else "blocked", str(stats_path), f"max_offset={max_offset}"))
        rows.append(row("no_pre_admission_events_in_first24h_features", PASS if pd.notna(min_offset) and min_offset >= 0 else CRITICAL, "allowed" if pd.notna(min_offset) and min_offset >= 0 else "blocked", str(stats_path), f"min_offset={min_offset}"))
    else:
        rows.append(row("feature_offset_stats_exist", CRITICAL, "blocked", str(stats_path), "missing stats"))

    if not cohort.empty:
        first24 = int(cohort["eligible_first_24h_prediction"].astype(bool).sum()) if "eligible_first_24h_prediction" in cohort else 0
        rows.append(row("first24h_target_population_defined", PASS if first24 == len(matrix) else CRITICAL, "allowed" if first24 == len(matrix) else "blocked", str(cohort_path), f"eligible={first24}, matrix={len(matrix)}"))
        overlap = int(cohort.groupby("patient_id")["split"].nunique().gt(1).sum()) if {"patient_id", "split"}.issubset(cohort.columns) else -1
        rows.append(row("no_patient_overlap_across_splits", PASS if overlap == 0 else CRITICAL, "allowed" if overlap == 0 else "blocked", str(cohort_path), f"overlap={overlap}"))

    if not roles.empty:
        critical = roles[(roles["dataset"].eq("eICU")) & (roles["leakage_risk"].eq("critical"))]
        unsafe = critical[~critical["prediction_time_use"].astype(str).str.contains("forbidden|reference", case=False, na=False)]
        rows.append(row("critical_role_fields_forbidden", PASS if unsafe.empty else CRITICAL, "allowed" if unsafe.empty else "blocked", str(role_path), f"unsafe_critical_fields={len(unsafe)}"))
    else:
        rows.append(row("external_field_role_catalog_present", WARNING, "review", str(role_path), "role catalog missing; run --external-field-role-catalog"))
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
    table_path = args.project_root / "outputs" / "tables" / "eicu_leakage_gate.csv"
    report_path = args.project_root / "outputs" / "reports" / "eicu_leakage_gate_report.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# eICU First-24h Leakage Gate

- Overall status: `{"PASS" if blocked.empty else "FAIL"}`
- Checks: {len(checks)}
- Blocked checks: {len(blocked)}
- Boundary: research data timing/leakage audit only; no clinical recommendation.

## Interpretation

This gate asks whether a first-24h mortality prediction feature matrix accidentally used future information. It blocks discharge status, discharge offsets, outcome proxies, and event offsets after 1440 minutes.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"eICU leakage gate checks: {len(checks)}")
    print(f"Blocked: {len(blocked)}")
    print(f"Wrote {report_path}")
    if not blocked.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
