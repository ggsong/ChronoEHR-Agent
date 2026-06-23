#!/usr/bin/env python3
"""Validate the external field-role catalog for eICU and CHARLS."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_DATASETS = {"eICU", "CHARLS"}
REQUIRED_ROLES = {"id", "prediction_time", "outcome", "feature_source"}


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


def audit(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "tables" / "external_field_role_catalog.csv"
    summary_path = project_root / "outputs" / "tables" / "external_field_role_summary.csv"
    catalog = read_csv(path)
    summary = read_csv(summary_path)
    rows = [
        row("catalog_exists", "PASS" if not catalog.empty else "FAIL", str(path), f"rows={len(catalog)}"),
        row("summary_exists", "PASS" if not summary.empty else "FAIL", str(summary_path), f"rows={len(summary)}"),
    ]
    if catalog.empty:
        return pd.DataFrame(rows)
    required_columns = {
        "dataset",
        "study_id",
        "chrono_concept",
        "field_role",
        "source_table_or_wave",
        "candidate_field",
        "time_available",
        "prediction_time_use",
        "leakage_risk",
        "leakage_policy",
        "raw_data_status",
    }
    missing = sorted(required_columns - set(catalog.columns))
    rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(path), "missing=" + ",".join(missing)))

    datasets = set(catalog["dataset"].astype(str))
    rows.append(row("required_datasets", "PASS" if REQUIRED_DATASETS.issubset(datasets) else "FAIL", str(path), "datasets=" + ",".join(sorted(datasets))))

    for dataset in sorted(REQUIRED_DATASETS):
        subset = catalog[catalog["dataset"].astype(str).eq(dataset)]
        roles = set(subset["field_role"].astype(str))
        rows.append(
            row(
                f"{dataset}_required_roles",
                "PASS" if REQUIRED_ROLES.issubset(roles) else "FAIL",
                str(path),
                "roles=" + ",".join(sorted(roles)),
            )
        )

    policy_empty = catalog["leakage_policy"].fillna("").astype(str).str.strip().eq("").sum()
    rows.append(row("leakage_policy_nonempty", "PASS" if policy_empty == 0 else "FAIL", str(path), f"empty={policy_empty}"))

    critical = catalog[catalog["leakage_risk"].astype(str).eq("critical")]
    rows.append(row("critical_leakage_fields_present", "PASS" if not critical.empty else "FAIL", str(path), f"critical={len(critical)}"))
    bad_critical = critical[~critical["prediction_time_use"].astype(str).str.contains("forbidden", na=False)]
    rows.append(row("critical_fields_forbidden", "PASS" if bad_critical.empty else "FAIL", str(path), f"bad={len(bad_critical)}"))

    pending = catalog[catalog["raw_data_status"].astype(str).eq("data_pending")]
    rows.append(row("pending_status_allowed", "PASS", str(path), f"data_pending={len(pending)}"))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "external_field_role_catalog_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_field_role_catalog_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# External Field Role Catalog Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: field-role mapping validation only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External field-role catalog checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
