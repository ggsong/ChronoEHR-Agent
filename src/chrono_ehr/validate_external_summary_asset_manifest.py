#!/usr/bin/env python3
"""Validate the formal external-summary asset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_ASSETS = {
    "start_here_external_technical_summary",
    "external_benchmark_summary",
    "external_calibration_decision_summary",
    "external_model_selection_rationale",
    "external_subgroup_robustness_summary",
    "external_threshold_band_sensitivity",
    "external_calibration_method_rationale",
    "table_s13_external_benchmark_summary",
    "table_s14_external_benchmark_hard_metrics",
    "table_s15_external_technical_summary",
    "table_s16_external_calibration_decision_summary",
    "table_s17_external_model_selection_rationale",
    "table_s18_external_subgroup_robustness_summary",
    "table_s19_external_threshold_band_sensitivity",
    "table_s20_external_calibration_method_rationale",
    "external_metric_consistency_audit",
    "external_metric_consistency_audit_validation",
    "external_subgroup_robustness_summary_validation",
    "external_threshold_band_sensitivity_validation",
    "external_calibration_method_rationale_validation",
    "boundary_cdsl_full_stay",
    "boundary_eicu_task_scope",
    "boundary_charls_scope",
    "boundary_decision_curve_scope",
}

REQUIRED_SECTIONS = {
    "01_start_here",
    "02_main_tables",
    "03_supplementary_tables",
    "04_validation_evidence",
    "05_internal_audit_sources",
    "06_figures",
    "07_boundary_statements",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "external_summary_asset_manifest.csv"
    report_path = project_root / "outputs" / "reports" / "external_summary_asset_manifest.md"
    table = read_csv(table_path)
    rows = [
        row("manifest_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("manifest_report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)

    required_columns = {
        "asset_id",
        "package_section",
        "audience_role",
        "formal_role",
        "table_number",
        "path",
        "source_command",
        "validation_command",
        "boundary_note",
        "exists",
        "size_bytes",
        "sha256",
        "status",
    }
    missing_columns = sorted(required_columns - set(table.columns))
    rows.append(row("required_columns_present", "PASS" if not missing_columns else "FAIL", str(table_path), "missing=" + ",".join(missing_columns)))
    asset_ids = set(table["asset_id"].astype(str))
    missing_assets = sorted(REQUIRED_ASSETS - asset_ids)
    rows.append(row("required_assets_present", "PASS" if not missing_assets else "FAIL", str(table_path), "missing=" + ",".join(missing_assets)))
    sections = set(table["package_section"].astype(str))
    missing_sections = sorted(REQUIRED_SECTIONS - sections)
    rows.append(row("required_sections_present", "PASS" if not missing_sections else "FAIL", str(table_path), "missing=" + ",".join(missing_sections)))
    failures = table[table["status"].astype(str).ne("PASS")]
    rows.append(row("all_assets_pass", "PASS" if failures.empty else "FAIL", str(table_path), f"failures={len(failures)}"))
    rows.append(row("enough_assets_for_handoff", "PASS" if len(table) >= 25 else "FAIL", str(table_path), f"assets={len(table)}"))

    roles = set(table["formal_role"].astype(str))
    expected_roles = {"main_summary_table", "supplementary_table", "validation_report", "cross_table_audit", "source_table", "figure", "boundary_statement"}
    missing_roles = sorted(expected_roles - roles)
    rows.append(row("formal_roles_cover_package", "PASS" if not missing_roles else "FAIL", str(table_path), "missing=" + ",".join(missing_roles)))

    boundary_text = " ".join(table["boundary_note"].astype(str))
    boundary_tokens = [
        "naive upper-reference",
        "not chronic readmission external validation",
        "longitudinal cohort extension",
        "clinical action threshold",
    ]
    missing_tokens = [token for token in boundary_tokens if token.lower() not in boundary_text.lower()]
    rows.append(row("boundary_statements_fixed", "PASS" if not missing_tokens else "FAIL", str(table_path), "missing=" + ",".join(missing_tokens)))

    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rows.append(row("report_declares_research_boundary", "PASS" if "research model evaluation handoff package only" in report_text else "FAIL", str(report_path), "research-only boundary"))
    rows.append(row("report_has_pass_status", "PASS" if "Overall status: `PASS`" in report_text else "FAIL", str(report_path), "overall PASS"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_summary_asset_manifest_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_summary_asset_manifest_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Summary Asset Manifest Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research handoff-manifest outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External summary asset manifest validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
