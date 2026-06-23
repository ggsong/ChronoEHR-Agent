#!/usr/bin/env python3
"""Inspect registered ChronoEHR-Agent studies without running analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT / "configs" / "study_registry.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def active_study(registry: dict[str, Any], requested: str | None) -> dict[str, Any] | None:
    study_id = requested or registry.get("active_study")
    for study in registry.get("studies", []):
        if study.get("id") == study_id:
            return study
    return None


def status_line(label: str, value: Any) -> str:
    return f"- {label}: {value}"


def format_core_metrics(manifest: dict[str, Any]) -> list[str]:
    metrics = manifest.get("metrics", {})
    cohort = metrics.get("cohort", {})
    lines = [
        status_line("Pipeline status", f"`{manifest.get('status', 'unknown')}`"),
        status_line("Missing outputs", manifest.get("missing_outputs", [])),
    ]
    if cohort:
        lines.extend(
            [
                status_line("Final index admissions", f"{cohort.get('final_index_admissions', 'NA'):,}"),
                status_line("Final subjects", f"{cohort.get('final_subjects', 'NA'):,}"),
                status_line("30-day readmission rate", f"{cohort.get('readmission_30d_rate', 0):.2%}"),
            ]
        )
    return lines


def format_test_performance(manifest: dict[str, Any]) -> str:
    rows = manifest.get("metrics", {}).get("test_performance", [])
    if not rows:
        return "No test performance found."
    lines = [
        "| Feature set | AUROC | AUPRC | Brier |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature_set']} | {row['AUROC']:.4f} | {row['AUPRC']:.4f} | {row['Brier_score']:.4f} |"
        )
    return "\n".join(lines)


def format_planned(registry: dict[str, Any]) -> list[str]:
    planned = registry.get("planned_studies", [])
    if not planned:
        return ["- None"]
    lines = []
    for study in planned:
        details = [
            f"status={study.get('status', 'planned')}",
            f"priority={study.get('priority', 'NA')}",
        ]
        if study.get("config"):
            details.append(f"config={study['config']}")
        if study.get("next_step"):
            details.append(f"next={study['next_step']}")
        lines.append(f"- `{study.get('id', 'NA')}`: " + "; ".join(details))
    return lines


def format_registered(registry: dict[str, Any]) -> list[str]:
    studies = registry.get("studies", [])
    if not studies:
        return ["- None"]
    active = registry.get("active_study")
    lines = []
    for study in studies:
        marker = "active" if study.get("id") == active else "registered"
        details = [
            f"{marker}",
            f"status={study.get('status', 'NA')}",
            f"cohort={study.get('cohort', 'NA')}",
            f"config={study.get('config', 'NA')}",
            f"pipeline={study.get('pipeline', 'NA')}",
        ]
        if study.get("next_step"):
            details.append(f"next={study['next_step']}")
        lines.append(f"- `{study.get('id', 'NA')}`: " + "; ".join(details))
    return lines


def write_inspection(project_root: Path, registry_path: Path, study_id: str | None, output_path: Path) -> None:
    registry = load_json(registry_path)
    study = active_study(registry, study_id)
    manifest = load_json(project_root / "outputs" / "pipeline_manifest.json")
    package = load_json(project_root / "outputs" / "study_package.json")

    lines = [
        "# ChronoEHR-Agent Study Inspection",
        "",
        "## Active Study",
        "",
    ]
    if study is None:
        lines.append("- No matching registered study found.")
    else:
        lines.extend(
            [
                status_line("Study id", f"`{study.get('id', 'NA')}`"),
                status_line("Status", f"`{study.get('status', 'NA')}`"),
                status_line("Config", f"`{study.get('config', 'NA')}`"),
                status_line("Pipeline", f"`{study.get('pipeline', 'NA')}`"),
                status_line("Cohort", study.get("cohort", "NA")),
                status_line("Outcome", study.get("outcome", "NA")),
            ]
        )

    lines.extend(["", "## Current Run Status", ""])
    lines.extend(format_core_metrics(manifest))
    if package:
        lines.append(status_line("Study package", "`outputs/reports/study_package.md`"))

    lines.extend(["", "## Test Performance", "", format_test_performance(manifest)])

    lines.extend(["", "## Registered Studies", ""])
    lines.extend(format_registered(registry))

    lines.extend(["", "## Planned Studies", ""])
    lines.extend(format_planned(registry))

    lines.extend(
        [
            "",
            "## Useful Commands",
            "",
            "```bash",
            f"cd {project_root}",
            "python3 src/chrono_ehr/run_study.py --list",
            "python3 src/chrono_ehr/run_study.py --skip-existing --no-expensive",
            "python3 src/chrono_ehr/run_diabetes_demo.py --only manifest study_package inspect_study validate_registry validate_config",
            "```",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--study")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "reports" / "study_inspection.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_inspection(args.project_root, args.registry, args.study, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
