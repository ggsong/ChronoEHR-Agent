#!/usr/bin/env python3
"""Validate prediction-time model specs against processed feature files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from prediction_time_spec_loader import DEFAULT_SPEC_PATH, load_prediction_time_config, load_raw_config


def read_columns(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns)


def validate_study(project_root: Path, study_key: str, spec_path: Path) -> list[str]:
    issues = []
    try:
        config = load_prediction_time_config(study_key, spec_path)
    except Exception as exc:  # noqa: BLE001 - user-facing validation report
        return [f"Cannot load `{study_key}`: {exc}"]

    cohort_path = project_root / config["cohort_path"]
    if not cohort_path.exists():
        return [f"Missing cohort file for `{study_key}`: {config['cohort_path']}"]

    base_columns = read_columns(cohort_path)
    for spec in config["specs"]:
        available_columns = set(base_columns)
        for relative_path in spec.get("extra_feature_files", []):
            feature_path = project_root / relative_path
            if not feature_path.exists():
                issues.append(f"`{study_key}.{spec['feature_set']}` missing extra feature file: {relative_path}")
                continue
            available_columns.update(read_columns(feature_path))

        required = set(spec["numeric_features"] + spec["categorical_features"] + ["subject_id", "hadm_id", "readmission_30d", "split"])
        missing = sorted(required - available_columns)
        if missing:
            issues.append(f"`{study_key}.{spec['feature_set']}` missing columns: {missing}")
    return issues


def write_report(results: dict[str, list[str]], output_path: Path) -> None:
    issue_count = sum(len(issues) for issues in results.values())
    lines = [
        "# Prediction-Time Spec Validation Report",
        "",
        f"- Status: `{'PASS' if issue_count == 0 else 'FAIL'}`",
        f"- Studies checked: {len(results)}",
        f"- Issues: {issue_count}",
        "",
    ]
    for study_key, issues in results.items():
        lines.append(f"## {study_key}")
        lines.append("")
        if issues:
            for issue in issues:
                lines.append(f"- **ISSUE**: {issue}")
        else:
            lines.append("- No issues found.")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_raw_config(args.spec)
    results = {
        study_key: validate_study(args.project_root, study_key, args.spec)
        for study_key in raw.get("studies", {})
    }
    output = args.project_root / "outputs" / "reports" / "prediction_time_spec_validation_report.md"
    write_report(results, output)
    issue_count = sum(len(issues) for issues in results.values())
    print(f"Prediction-time spec validation: issues={issue_count}")
    print(f"Wrote {output}")
    if issue_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
