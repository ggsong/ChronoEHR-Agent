#!/usr/bin/env python3
"""Validate eICU first-24h temporal feature skeletons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


FORBIDDEN_PATTERNS = [
    "hospitaldischargestatus",
    "unitdischargestatus",
    "hospitaldischargeoffset",
    "unitdischargeoffset",
    "hospital_los",
    "unit_los",
    "icu_mortality",
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


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def bad_columns(columns: list[str]) -> list[str]:
    bad = []
    for column in columns:
        lowered = column.lower()
        if any(pattern in lowered for pattern in FORBIDDEN_PATTERNS):
            bad.append(column)
    return bad


def validate(project_root: Path) -> pd.DataFrame:
    cohort_path = project_root / "data" / "processed" / "eicu_temporal_mortality_cohort.csv"
    matrix_path = project_root / "data" / "processed" / "eicu_first24h_feature_matrix_skeleton.csv"
    lab_path = project_root / "data" / "processed" / "eicu_lab_features_24h.csv"
    vital_path = project_root / "data" / "processed" / "eicu_vital_features_24h.csv"
    availability_path = project_root / "outputs" / "tables" / "eicu_temporal_features_24h_availability.csv"
    stats_path = project_root / "outputs" / "tables" / "eicu_temporal_features_24h_extraction_stats.csv"

    cohort = read_csv(cohort_path)
    matrix = read_csv(matrix_path)
    lab = read_csv(lab_path)
    vital = read_csv(vital_path)
    availability = read_csv(availability_path)
    stats = read_csv(stats_path)

    rows = [
        row("feature_matrix_exists", "PASS" if not matrix.empty else "FAIL", str(matrix_path), f"rows={len(matrix)}"),
        row("lab_features_exist", "PASS" if not lab.empty else "FAIL", str(lab_path), f"rows={len(lab)}"),
        row("vital_features_exist", "PASS" if not vital.empty else "FAIL", str(vital_path), f"rows={len(vital)}"),
        row("availability_exists", "PASS" if not availability.empty else "FAIL", str(availability_path), f"rows={len(availability)}"),
        row("stats_exists", "PASS" if not stats.empty else "FAIL", str(stats_path), f"rows={len(stats)}"),
    ]
    if cohort.empty or matrix.empty or lab.empty or vital.empty or stats.empty:
        return pd.DataFrame(rows)

    eligible = cohort[cohort["eligible_first_24h_prediction"].astype(bool)]
    rows.append(row("matrix_rows_match_eligible_stays", "PASS" if len(matrix) == len(eligible) else "FAIL", str(matrix_path), f"matrix={len(matrix)}, eligible={len(eligible)}"))
    rows.append(row("lab_rows_match_eligible_stays", "PASS" if len(lab) == len(eligible) else "FAIL", str(lab_path), f"lab={len(lab)}, eligible={len(eligible)}"))
    rows.append(row("vital_rows_match_eligible_stays", "PASS" if len(vital) == len(eligible) else "FAIL", str(vital_path), f"vital={len(vital)}, eligible={len(eligible)}"))

    missing_ids = set(eligible["stay_id"].astype(int)) - set(matrix["stay_id"].astype(int))
    rows.append(row("all_eligible_stays_represented", "PASS" if not missing_ids else "FAIL", str(matrix_path), f"missing={len(missing_ids)}"))

    bad = bad_columns(matrix.columns.tolist())
    allowed_labels = {"hospital_mortality"}
    bad = [column for column in bad if column not in allowed_labels]
    rows.append(row("no_forbidden_discharge_or_outcome_features", "PASS" if not bad else "FAIL", str(matrix_path), "bad=" + ",".join(bad[:20])))

    feature_columns = [column for column in matrix.columns if column not in {"stay_id", "patient_id", "split", "hospital_mortality"}]
    prefix_ok = all(column.startswith(("eicu_lab24h_", "eicu_vital24h_")) for column in feature_columns)
    rows.append(row("feature_prefixes_are_temporal", "PASS" if prefix_ok else "FAIL", str(matrix_path), f"feature_columns={len(feature_columns)}"))

    max_offset = pd.to_numeric(stats["max_included_offset"], errors="coerce").max()
    min_offset = pd.to_numeric(stats["min_included_offset"], errors="coerce").min()
    rows.append(row("included_offsets_not_after_24h", "PASS" if pd.notna(max_offset) and max_offset <= 1440 else "FAIL", str(stats_path), f"max_offset={max_offset}"))
    rows.append(row("included_offsets_not_before_admission", "PASS" if pd.notna(min_offset) and min_offset >= 0 else "FAIL", str(stats_path), f"min_offset={min_offset}"))

    measurement_rows = int(pd.to_numeric(stats["numeric_rows"], errors="coerce").fillna(0).sum())
    rows.append(row("nonzero_measurements", "PASS" if measurement_rows > 0 else "FAIL", str(stats_path), f"numeric_rows={measurement_rows}"))
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
    table_path = args.project_root / "outputs" / "tables" / "eicu_temporal_features_24h_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "eicu_temporal_features_24h_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# eICU First-24h Temporal Feature Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates feature timing and skeleton files only; no model training or clinical recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"eICU first-24h feature checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
