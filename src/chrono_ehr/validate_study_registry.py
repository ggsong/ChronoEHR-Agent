#!/usr/bin/env python3
"""Validate study registry references for registered and planned studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_study_config import PROJECT, load_yaml_with_ruby


REGISTRY = PROJECT / "configs" / "study_registry.json"


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_path(project_root: Path, relative_path: str | None, issues: list[str], label: str) -> bool:
    if not relative_path:
        issues.append(f"Missing {label} path")
        return False
    path = project_root / relative_path
    if not path.exists():
        issues.append(f"Missing {label}: {relative_path}")
        return False
    return True


def validate_config(project_root: Path, relative_path: str, issues: list[str]) -> None:
    path = project_root / relative_path
    try:
        config = load_yaml_with_ruby(path)
    except Exception as exc:  # noqa: BLE001 - report parser error to user-facing report
        issues.append(f"Config cannot be parsed as YAML: {relative_path} ({exc})")
        return
    for key in ["dataset", "cohort_definition", "outcome", "prediction_times", "feature_sets", "outputs"]:
        if key not in config:
            issues.append(f"Config {relative_path} missing top-level `{key}`")


def validate_study(project_root: Path, study: dict[str, Any], planned: bool) -> dict[str, Any]:
    issues: list[str] = []
    notes: list[str] = []
    config_path = study.get("config")
    requires_scaffold_files = not planned or study.get("status") == "scaffolded" or bool(config_path)
    if requires_scaffold_files and check_path(project_root, config_path, issues, "config") and config_path:
        validate_config(project_root, config_path, issues)
    elif planned and not config_path:
        notes.append("No config yet; planned-only entry.")
    if planned:
        if study.get("protocol"):
            check_path(project_root, study.get("protocol"), issues, "protocol")
        if study.get("feature_time_map"):
            check_path(project_root, study.get("feature_time_map"), issues, "feature_time_map")
    else:
        check_path(project_root, study.get("pipeline"), issues, "pipeline")
    return {
        "id": study.get("id", "NA"),
        "status": study.get("status", "NA"),
        "planned": planned,
        "issues": issues,
        "notes": notes,
    }


def write_report(results: list[dict[str, Any]], output_path: Path) -> None:
    errors = sum(len(result["issues"]) for result in results)
    lines = [
        "# Study Registry Validation",
        "",
        f"- Status: `{'PASS' if errors == 0 else 'FAIL'}`",
        f"- Studies checked: {len(results)}",
        f"- Issues: {errors}",
        "",
        "## Study Checks",
        "",
    ]
    for result in results:
        lines.append(f"### {result['id']}")
        lines.append("")
        lines.append(f"- Registry status: `{result['status']}`")
        lines.append(f"- Planned/scaffolded entry: {result['planned']}")
        if result["issues"]:
            for issue in result["issues"]:
                lines.append(f"- **ISSUE**: {issue}")
        else:
            lines.append("- No issues found.")
        for note in result.get("notes", []):
            lines.append(f"- Note: {note}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    results = []
    for study in registry.get("studies", []):
        results.append(validate_study(args.project_root, study, planned=False))
    for study in registry.get("planned_studies", []):
        results.append(validate_study(args.project_root, study, planned=True))
    output = args.project_root / "outputs" / "reports" / "study_registry_validation_report.md"
    write_report(results, output)
    issue_count = sum(len(result["issues"]) for result in results)
    print(f"Study registry validation: issues={issue_count}")
    print(f"Wrote {output}")
    if issue_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
