#!/usr/bin/env python3
"""Validate the concrete external handoff package."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


PACKAGE_DIR = Path("outputs/external_handoff_package")
PACKAGE_ZIP = Path("outputs/external_handoff_package.zip")
REQUIRED_SECTIONS = {
    "01_start_here",
    "02_main_tables",
    "03_supplementary_tables",
    "04_validation_evidence",
    "05_internal_audit_sources",
    "06_figures",
}
REQUIRED_ASSETS = {
    "start_here_external_technical_summary",
    "start_here_external_technical_report",
    "external_benchmark_summary",
    "external_calibration_decision_summary",
    "external_model_selection_rationale",
    "external_subgroup_robustness_summary",
    "external_threshold_band_sensitivity",
    "external_calibration_method_rationale",
    "external_metric_consistency_audit",
    "external_metric_consistency_audit_report",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def validate(project_root: Path) -> pd.DataFrame:
    package_root = project_root / PACKAGE_DIR
    table_path = project_root / "outputs" / "tables" / "external_handoff_package_manifest.csv"
    report_path = project_root / "outputs" / "reports" / "external_handoff_package.md"
    zip_path = project_root / PACKAGE_ZIP
    package_manifest_path = package_root / "package_manifest.csv"
    readme_path = package_root / "README.md"
    boundary_path = package_root / "boundary_statements.md"
    manifest = read_csv(table_path)

    rows = [
        row("package_directory_exists", "PASS" if package_root.exists() and package_root.is_dir() else "FAIL", str(package_root), "package directory"),
        row("package_manifest_exists", "PASS" if not manifest.empty else "FAIL", str(table_path), f"rows={len(manifest)}"),
        row("package_internal_manifest_exists", "PASS" if exists(package_manifest_path) else "FAIL", str(package_manifest_path), "package copy of manifest"),
        row("package_report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), "handoff package report"),
        row("package_readme_exists", "PASS" if exists(readme_path) else "FAIL", str(readme_path), "package README"),
        row("boundary_statements_exists", "PASS" if exists(boundary_path) else "FAIL", str(boundary_path), "boundary statements file"),
        row("zip_archive_exists", "PASS" if exists(zip_path) else "FAIL", str(zip_path), f"size={zip_path.stat().st_size if zip_path.exists() else 0}"),
    ]
    if manifest.empty:
        return pd.DataFrame(rows)

    required_columns = {
        "asset_id",
        "package_section",
        "audience_role",
        "formal_role",
        "table_number",
        "source_path",
        "package_path",
        "source_sha256",
        "package_sha256",
        "size_bytes",
        "status",
    }
    missing = sorted(required_columns - set(manifest.columns))
    rows.append(row("required_columns_present", "PASS" if not missing else "FAIL", str(table_path), "missing=" + ",".join(missing)))
    rows.append(row("enough_packaged_files", "PASS" if len(manifest) >= 20 else "FAIL", str(table_path), f"files={len(manifest)}"))
    assets = set(manifest["asset_id"].astype(str))
    missing_assets = sorted(REQUIRED_ASSETS - assets)
    rows.append(row("required_assets_packaged", "PASS" if not missing_assets else "FAIL", str(table_path), "missing=" + ",".join(missing_assets)))
    sections = set(manifest["package_section"].astype(str))
    missing_sections = sorted(REQUIRED_SECTIONS - sections)
    rows.append(row("required_sections_packaged", "PASS" if not missing_sections else "FAIL", str(table_path), "missing=" + ",".join(missing_sections)))

    file_failures = []
    hash_failures = []
    for _, item in manifest.iterrows():
        package_file = package_root / str(item["package_path"])
        if not exists(package_file):
            file_failures.append(str(item["asset_id"]))
            continue
        if sha256(package_file) != str(item["package_sha256"]):
            hash_failures.append(str(item["asset_id"]))
    rows.append(row("all_package_files_exist", "PASS" if not file_failures else "FAIL", str(package_root), "missing=" + ",".join(file_failures)))
    rows.append(row("package_hashes_match_manifest", "PASS" if not hash_failures else "FAIL", str(table_path), "bad=" + ",".join(hash_failures)))

    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    boundary_text = boundary_path.read_text(encoding="utf-8") if boundary_path.exists() else ""
    rows.append(row("readme_has_opening_order", "PASS" if "Recommended opening order" in readme_text else "FAIL", str(readme_path), "opening order"))
    rows.append(row("boundary_mentions_research_only", "PASS" if "research model evaluation only" in boundary_text else "FAIL", str(boundary_path), "research-only boundary"))
    forbidden = ["recommended treatment", "ready for clinical deployment"]
    rows.append(row("package_avoids_clinical_claims", "PASS" if not any(token in (readme_text + boundary_text).lower() for token in forbidden) else "FAIL", str(package_root), "forbidden clinical wording absent"))

    zip_entries: list[str] = []
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            zip_entries = archive.namelist()
    expected_zip_entries = [str(PACKAGE_DIR / "README.md"), str(PACKAGE_DIR / "package_manifest.csv"), str(PACKAGE_DIR / "boundary_statements.md")]
    rows.append(row("zip_contains_package_control_files", "PASS" if all(item in zip_entries for item in expected_zip_entries) else "FAIL", str(zip_path), f"entries={len(zip_entries)}"))
    rows.append(row("zip_contains_all_manifest_files", "PASS" if len(zip_entries) >= len(manifest) + 3 else "FAIL", str(zip_path), f"entries={len(zip_entries)}; manifest_files={len(manifest)}"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_handoff_package_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_handoff_package_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Handoff Package Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research handoff package outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External handoff package validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
