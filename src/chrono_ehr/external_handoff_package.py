#!/usr/bin/env python3
"""Build a concrete external handoff package directory and zip archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


PACKAGE_DIR = Path("outputs/external_handoff_package")
PACKAGE_ZIP = Path("outputs/external_handoff_package.zip")


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


def clean_package_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_name(asset_id: str, source_path: str) -> str:
    suffix = Path(source_path).suffix
    return f"{asset_id}{suffix}"


def build_package(project_root: Path) -> pd.DataFrame:
    manifest_path = project_root / "outputs" / "tables" / "external_summary_asset_manifest.csv"
    manifest = read_csv(manifest_path)
    if manifest.empty:
        raise FileNotFoundError("Missing external_summary_asset_manifest.csv")

    package_root = project_root / PACKAGE_DIR
    clean_package_dir(package_root)
    copied_rows: list[dict[str, object]] = []

    copyable = manifest[
        manifest["status"].astype(str).eq("PASS")
        & manifest["exists"].astype(bool)
        & manifest["formal_role"].astype(str).ne("boundary_statement")
    ].copy()
    for _, asset in copyable.iterrows():
        source_relative = str(asset["path"])
        source = project_root / source_relative
        section = str(asset["package_section"])
        target_dir = package_root / section
        target_dir.mkdir(parents=True, exist_ok=True)
        target_relative = Path(section) / safe_name(str(asset["asset_id"]), source_relative)
        target = package_root / target_relative
        shutil.copy2(source, target)
        copied_rows.append(
            {
                "asset_id": str(asset["asset_id"]),
                "package_section": section,
                "audience_role": str(asset["audience_role"]),
                "formal_role": str(asset["formal_role"]),
                "table_number": "" if pd.isna(asset["table_number"]) else str(asset["table_number"]),
                "source_path": source_relative,
                "package_path": str(target_relative),
                "source_sha256": sha256(source),
                "package_sha256": sha256(target),
                "size_bytes": int(target.stat().st_size),
                "status": "PASS",
            }
        )

    package_manifest = pd.DataFrame(copied_rows)
    boundary = manifest[manifest["formal_role"].astype(str).eq("boundary_statement")].copy()
    write_package_readme(project_root, package_root, package_manifest, boundary)
    write_boundary_statements(package_root, boundary)

    package_manifest_path = package_root / "package_manifest.csv"
    package_manifest.to_csv(package_manifest_path, index=False)
    report_path = project_root / "outputs" / "reports" / "external_handoff_package.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(package_report(package_manifest, boundary), encoding="utf-8")

    table_path = project_root / "outputs" / "tables" / "external_handoff_package_manifest.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    package_manifest.to_csv(table_path, index=False)
    build_zip(project_root, package_root, project_root / PACKAGE_ZIP)
    return package_manifest


def write_boundary_statements(package_root: Path, boundary: pd.DataFrame) -> None:
    lines = [
        "# Boundary Statements",
        "",
        "This handoff package is for research model evaluation only. It does not provide diagnosis, treatment, clinical deployment guidance, or clinical action threshold recommendations.",
        "",
    ]
    for _, row in boundary.iterrows():
        lines.append(f"- {row['asset_id']}: {row['boundary_note']}")
    (package_root / "boundary_statements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_package_readme(project_root: Path, package_root: Path, package_manifest: pd.DataFrame, boundary: pd.DataFrame) -> None:
    by_section = package_manifest.groupby("package_section")["asset_id"].count().to_dict() if not package_manifest.empty else {}
    lines = [
        "# ChronoEHR External Handoff Package",
        "",
        "Boundary: research model evaluation handoff package only; no diagnosis, treatment, clinical deployment, or clinical action threshold recommendation.",
        "",
        "Recommended opening order:",
        "",
        "1. `01_start_here/start_here_external_technical_summary.csv`",
        "2. `01_start_here/start_here_external_technical_report.md`",
        "3. `02_main_tables/external_benchmark_summary.csv`",
        "4. `04_validation_evidence/external_metric_consistency_audit_report.md`",
        "",
        "Package sections:",
        "",
    ]
    for section, count in sorted(by_section.items()):
        lines.append(f"- `{section}`: {count} files")
    lines.extend(["", "Fixed boundary statements:", ""])
    for _, row in boundary.iterrows():
        lines.append(f"- {row['boundary_note']}")
    lines.extend(
        [
            "",
            "Generated from:",
            "",
            "- `outputs/tables/external_summary_asset_manifest.csv`",
            "- `outputs/tables/external_metric_consistency_audit.csv`",
            "- `outputs/reports/external_summary_asset_manifest.md`",
            "",
            f"Project root: `{project_root}`",
        ]
    )
    (package_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_report(package_manifest: pd.DataFrame, boundary: pd.DataFrame) -> str:
    section_counts = package_manifest.groupby("package_section")["asset_id"].count().to_dict() if not package_manifest.empty else {}
    lines = [
        "# External Handoff Package",
        "",
        "- Overall status: `PASS`",
        f"- Packaged files: {len(package_manifest)}",
        f"- Boundary statements: {len(boundary)}",
        f"- Zip archive: `{PACKAGE_ZIP}`",
        "- Boundary: research model evaluation handoff package only; no diagnosis, treatment, clinical deployment, or clinical action threshold recommendation.",
        "",
        "## Sections",
        "",
    ]
    for section, count in sorted(section_counts.items()):
        lines.append(f"- `{section}`: {count} files")
    lines.extend(["", "## Package Manifest", "", markdown_table(package_manifest)])
    return "\n".join(lines) + "\n"


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["package_section", "asset_id", "formal_role", "table_number", "package_path", "status"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def build_zip(project_root: Path, package_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(project_root))


def main() -> None:
    args = parse_args()
    package_manifest = build_package(args.project_root)
    print(f"External handoff packaged files: {len(package_manifest)}")
    print(f"Wrote {args.project_root / PACKAGE_DIR}")
    print(f"Wrote {args.project_root / PACKAGE_ZIP}")


if __name__ == "__main__":
    main()
