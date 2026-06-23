#!/usr/bin/env python3
"""Validate ChronoEHR-Agent study configuration files without extra Python deps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]

REQUIRED_TOP_LEVEL = [
    "dataset",
    "cohort_definition",
    "outcome",
    "prediction_times",
    "feature_sets",
    "models",
    "metrics",
    "split_strategy",
    "outputs",
]

REQUIRED_DATASET_KEYS = ["name", "type", "root_path", "tables", "id_columns"]
REQUIRED_OUTCOME_KEYS = ["name", "type", "prediction_time", "definition"]
REQUIRED_SPLIT_KEYS = ["method", "group_column"]


def load_yaml_with_ruby(path: Path) -> dict[str, Any]:
    ruby = (
        "require 'yaml'; require 'json'; "
        "data = YAML.load_file(ARGV[0]); "
        "puts JSON.generate(data)"
    )
    result = subprocess.run(
        ["ruby", "-e", ruby, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def add_issue(issues: list[dict[str, str]], severity: str, message: str) -> None:
    issues.append({"severity": severity, "message": message})


def require_keys(section: dict[str, Any], keys: list[str], prefix: str, issues: list[dict[str, str]]) -> None:
    for key in keys:
        if key not in section:
            add_issue(issues, "error", f"Missing `{prefix}.{key}`")


def flatten_output_paths(outputs: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(outputs, str):
        paths.append(outputs)
    elif isinstance(outputs, dict):
        for value in outputs.values():
            paths.extend(flatten_output_paths(value))
    elif isinstance(outputs, list):
        for value in outputs:
            paths.extend(flatten_output_paths(value))
    return paths


def validate(config: dict[str, Any], project_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    issues: list[dict[str, str]] = []
    summary: dict[str, Any] = {}

    for key in REQUIRED_TOP_LEVEL:
        if key not in config:
            add_issue(issues, "error", f"Missing top-level section `{key}`")

    dataset = config.get("dataset", {})
    if isinstance(dataset, dict):
        require_keys(dataset, REQUIRED_DATASET_KEYS, "dataset", issues)
        root = Path(os.path.expandvars(str(dataset.get("root_path", "")))).expanduser()
        summary["dataset_root"] = str(root)
        if not root.exists():
            add_issue(issues, "error", f"Dataset root does not exist: {root}")
        tables = dataset.get("tables", {})
        if isinstance(tables, dict):
            missing_tables = []
            for table_name, relative_path in tables.items():
                table_path = root / str(relative_path)
                if not table_path.exists():
                    missing_tables.append(f"{table_name}: {relative_path}")
            summary["dataset_tables_checked"] = len(tables)
            summary["dataset_tables_missing"] = len(missing_tables)
            for item in missing_tables:
                add_issue(issues, "error", f"Missing dataset table `{item}`")
        else:
            add_issue(issues, "error", "`dataset.tables` must be a mapping")
    else:
        add_issue(issues, "error", "`dataset` must be a mapping")

    outcome = config.get("outcome", {})
    if isinstance(outcome, dict):
        require_keys(outcome, REQUIRED_OUTCOME_KEYS, "outcome", issues)
        if outcome.get("type") != "binary":
            add_issue(issues, "warning", "Current demo tooling is optimized for binary outcomes")
        if "followup_window_days" not in outcome:
            add_issue(issues, "warning", "No `outcome.followup_window_days` found")
    else:
        add_issue(issues, "error", "`outcome` must be a mapping")

    split = config.get("split_strategy", {})
    if isinstance(split, dict):
        require_keys(split, REQUIRED_SPLIT_KEYS, "split_strategy", issues)
        method = str(split.get("method", ""))
        if "patient" not in method and "subject" not in method:
            add_issue(issues, "warning", "Split strategy may not be patient-level")
    else:
        add_issue(issues, "error", "`split_strategy` must be a mapping")

    feature_sets = config.get("feature_sets", {})
    forbidden = []
    enabled_sets = []
    if isinstance(feature_sets, dict):
        for name, spec in feature_sets.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("enabled") is True:
                enabled_sets.append(name)
            if "forbidden" in name or "high_risk" in name:
                forbidden.extend(spec.get("variables", []))
    summary["enabled_feature_sets"] = enabled_sets
    summary["forbidden_or_high_risk_variables"] = forbidden
    if not forbidden:
        add_issue(issues, "warning", "No forbidden/high-risk feature list found")

    required_metrics = config.get("metrics", {}).get("required", []) if isinstance(config.get("metrics"), dict) else []
    for metric in ["AUROC", "AUPRC"]:
        if metric not in required_metrics:
            add_issue(issues, "warning", f"Required metrics should include `{metric}`")

    output_paths = flatten_output_paths(config.get("outputs", {}))
    missing_outputs = []
    existing_outputs = []
    for relative_path in output_paths:
        path = project_root / relative_path
        if path.exists():
            existing_outputs.append(relative_path)
        else:
            missing_outputs.append(relative_path)
    summary["outputs_listed"] = len(output_paths)
    summary["outputs_existing"] = len(existing_outputs)
    summary["outputs_missing"] = len(missing_outputs)
    summary["missing_output_paths"] = missing_outputs

    return issues, summary


def write_report(config_path: Path, issues: list[dict[str, str]], summary: dict[str, Any], output_path: Path) -> None:
    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    status = "PASS" if error_count == 0 else "FAIL"
    lines = [
        "# Study Config Validation Report",
        "",
        f"- Config: `{config_path}`",
        f"- Status: `{status}`",
        f"- Errors: {error_count}",
        f"- Warnings: {warning_count}",
        "",
        "## Summary",
        "",
        f"- Dataset root: `{summary.get('dataset_root', 'NA')}`",
        f"- Dataset tables checked: {summary.get('dataset_tables_checked', 0)}",
        f"- Dataset tables missing: {summary.get('dataset_tables_missing', 0)}",
        f"- Enabled feature sets: {', '.join(summary.get('enabled_feature_sets', [])) or 'None'}",
        f"- Forbidden/high-risk variables: {len(summary.get('forbidden_or_high_risk_variables', []))}",
        f"- Outputs listed: {summary.get('outputs_listed', 0)}",
        f"- Outputs existing: {summary.get('outputs_existing', 0)}",
        f"- Outputs missing: {summary.get('outputs_missing', 0)}",
        "",
        "## Issues",
        "",
    ]
    if issues:
        for issue in issues:
            lines.append(f"- **{issue['severity'].upper()}**: {issue['message']}")
    else:
        lines.append("- No issues found.")
    if summary.get("missing_output_paths"):
        lines.extend(["", "## Missing Outputs", ""])
        for path in summary["missing_output_paths"]:
            lines.append(f"- `{path}`")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=PROJECT / "configs" / "diabetes_mimic_readmission.yaml")
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_with_ruby(args.config)
    issues, summary = validate(config, args.project_root)
    output_path = args.project_root / "outputs" / "reports" / "study_config_validation_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.config, issues, summary, output_path)

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    print(f"Study config validation: errors={error_count}, warnings={warning_count}")
    print(f"Wrote {output_path}")
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
