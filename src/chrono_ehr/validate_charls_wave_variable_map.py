#!/usr/bin/env python3
"""Validate the concrete CHARLS wave-variable map."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_CONCEPTS = {
    "person_id",
    "baseline_age",
    "baseline_diabetes_diagnosis",
    "followup_2013_diabetes_diagnosis",
    "followup_2015_diabetes_diagnosis",
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
    map_path = project_root / "outputs" / "tables" / "charls_wave_variable_map.csv"
    inv_path = project_root / "outputs" / "tables" / "charls_harmonized_variable_inventory.csv"
    report_path = project_root / "outputs" / "reports" / "charls_wave_variable_map.md"
    wave_map = read_csv(map_path)
    inventory = read_csv(inv_path)
    rows = [
        row("wave_map_exists", "PASS" if not wave_map.empty else "FAIL", str(map_path), f"rows={len(wave_map)}"),
        row("inventory_exists", "PASS" if not inventory.empty else "FAIL", str(inv_path), f"rows={len(inventory)}"),
        row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
    ]
    if wave_map.empty:
        return pd.DataFrame(rows)

    required_columns = {"concept", "variable", "wave", "role", "present", "leakage_status"}
    missing_columns = sorted(required_columns - set(wave_map.columns))
    rows.append(row("required_columns", "PASS" if not missing_columns else "FAIL", str(map_path), "missing=" + ",".join(missing_columns)))
    if missing_columns:
        return pd.DataFrame(rows)

    present_concepts = set(wave_map.loc[wave_map["present"].astype(bool), "concept"].astype(str))
    missing_concepts = sorted(REQUIRED_CONCEPTS - present_concepts)
    rows.append(row("required_concepts_present", "PASS" if not missing_concepts else "FAIL", str(map_path), "missing=" + ",".join(missing_concepts)))
    future_wave = wave_map["wave"].astype(str).str.contains("2013|2015", regex=True, na=False)
    future_roles = wave_map["role"].astype(str).eq("outcome") | (wave_map["role"].astype(str).eq("followup_status") & future_wave)
    bad_future = wave_map[future_roles & ~wave_map["leakage_status"].astype(str).isin(["forbidden_as_baseline_feature", "do_not_use_as_baseline_feature"])]
    rows.append(row("future_wave_marked_forbidden", "PASS" if bad_future.empty else "FAIL", str(map_path), f"bad_rows={len(bad_future)}"))
    baseline_features = wave_map[wave_map["role"].astype(str).eq("baseline_feature")]
    future_baseline = baseline_features[baseline_features["wave"].astype(str).str.contains("2013|2015", regex=True, na=False)]
    rows.append(row("baseline_features_not_future_wave", "PASS" if future_baseline.empty else "FAIL", str(map_path), f"bad_rows={len(future_baseline)}"))
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
    table_path = args.project_root / "outputs" / "tables" / "charls_wave_variable_map_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "charls_wave_variable_map_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# CHARLS Wave Variable Map Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates variable mapping and leakage roles only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"CHARLS wave-map validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
