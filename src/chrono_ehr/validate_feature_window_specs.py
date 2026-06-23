#!/usr/bin/env python3
"""Validate feature-window specs against local MIMIC and generated feature files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from feature_window_spec_loader import DEFAULT_FEATURE_WINDOW_SPEC, load_feature_window_spec
from mimic_diabetes_baseline import DEFAULT_PROJECT
from mimic_ckd_lab_itemids import DEFAULT_MIMIC_ROOT


REQUIRED_WINDOWS = {"admission_baseline", "first_24h", "admission_to_discharge", "followup_30d"}
REQUIRED_WINDOW_KEYS = {"meaning", "start", "end", "usable_for_prediction_times", "forbidden_for_prediction_times", "leakage_note"}
REQUIRED_SOURCE_KEYS = {"mimic_table", "available_time", "window_outputs"}


def add_issue(issues: list[dict[str, str]], severity: str, message: str) -> None:
    issues.append({"severity": severity, "message": message})


def read_columns(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns)


def validate_windows(spec: dict[str, Any], issues: list[dict[str, str]]) -> None:
    windows = spec.get("windows", {})
    missing = REQUIRED_WINDOWS - set(windows)
    for window in sorted(missing):
        add_issue(issues, "error", f"Missing required window `{window}`")
    for name, window in windows.items():
        missing_keys = REQUIRED_WINDOW_KEYS - set(window)
        for key in sorted(missing_keys):
            add_issue(issues, "error", f"Window `{name}` missing `{key}`")
        if name == "followup_30d" and window.get("usable_for_prediction_times"):
            add_issue(issues, "error", "`followup_30d` must not be usable for prediction features")
        if name == "admission_to_discharge" and "admission" not in window.get("forbidden_for_prediction_times", []):
            add_issue(issues, "warning", "`admission_to_discharge` should be forbidden for admission prediction")


def validate_cohorts(project_root: Path, spec: dict[str, Any], issues: list[dict[str, str]]) -> None:
    for cohort, cohort_spec in spec.get("cohorts", {}).items():
        path = project_root / cohort_spec.get("cohort_path", "")
        if not path.exists():
            add_issue(issues, "error", f"Cohort `{cohort}` missing file: {path}")
            continue
        columns = read_columns(path)
        for column in ["hadm_id", "admittime", "dischtime", "split", "readmission_30d"]:
            if column not in columns:
                add_issue(issues, "error", f"Cohort `{cohort}` missing required column `{column}`")
        allowed = set(cohort_spec.get("allowed_prediction_times", []))
        for required in ["admission", "inhospital_24h", "discharge"]:
            if required not in allowed:
                add_issue(issues, "warning", f"Cohort `{cohort}` does not list prediction time `{required}`")


def validate_sources(project_root: Path, mimic_root: Path, spec: dict[str, Any], issues: list[dict[str, str]]) -> None:
    windows = set(spec.get("windows", {}))
    cohorts = set(spec.get("cohorts", {}))
    for source_name, source in spec.get("feature_sources", {}).items():
        for key in REQUIRED_SOURCE_KEYS:
            if key not in source:
                add_issue(issues, "error", f"Feature source `{source_name}` missing `{key}`")
        mimic_table = mimic_root / source.get("mimic_table", "")
        if not mimic_table.exists():
            add_issue(issues, "error", f"Feature source `{source_name}` missing MIMIC table: {mimic_table}")
        dictionary = source.get("item_dictionary")
        if dictionary and not (mimic_root / dictionary).exists():
            add_issue(issues, "error", f"Feature source `{source_name}` missing item dictionary: {dictionary}")
        available = source.get("available_time", {})
        if "primary" not in available:
            add_issue(issues, "error", f"Feature source `{source_name}` missing available_time.primary")

        for output in source.get("window_outputs", []):
            cohort = output.get("cohort")
            window = output.get("window")
            relative_path = output.get("path")
            if cohort not in cohorts:
                add_issue(issues, "error", f"Feature source `{source_name}` output references unknown cohort `{cohort}`")
            if window not in windows:
                add_issue(issues, "error", f"Feature source `{source_name}` output references unknown window `{window}`")
            if not relative_path:
                add_issue(issues, "error", f"Feature source `{source_name}` output missing path")
                continue
            output_path = project_root / relative_path
            if not output_path.exists():
                add_issue(issues, "warning", f"Feature source `{source_name}` output not generated yet: {relative_path}")
                continue
            columns = read_columns(output_path)
            if "hadm_id" not in columns:
                add_issue(issues, "error", f"Feature output `{relative_path}` missing `hadm_id`")


def validate_vital_itemids(mimic_root: Path, spec: dict[str, Any], issues: list[dict[str, str]]) -> None:
    source = spec.get("feature_sources", {}).get("vital_signs", {})
    dictionary = source.get("item_dictionary")
    itemids = {int(itemid) for itemid in source.get("itemids", {})}
    if not dictionary or not itemids:
        return
    path = mimic_root / dictionary
    if not path.exists():
        return
    d_items = pd.read_csv(path, usecols=["itemid", "label"])
    available = set(d_items["itemid"].astype(int))
    missing = sorted(itemids - available)
    for itemid in missing:
        add_issue(issues, "error", f"Vital itemid `{itemid}` not found in `{dictionary}`")


def write_report(
    spec_path: Path,
    project_root: Path,
    mimic_root: Path,
    spec: dict[str, Any],
    issues: list[dict[str, str]],
    output_path: Path,
) -> None:
    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    status = "PASS" if error_count == 0 else "FAIL"
    rows = []
    for source_name, source in spec.get("feature_sources", {}).items():
        for item in source.get("window_outputs", []):
            path = project_root / item.get("path", "")
            rows.append(
                f"| {source_name} | {item.get('cohort')} | {item.get('window')} | "
                f"`{item.get('path')}` | {'yes' if path.exists() else 'no'} |"
            )
    lines = [
        "# Feature Window Spec Validation Report",
        "",
        f"- Spec: `{spec_path}`",
        f"- Project root: `{project_root}`",
        f"- MIMIC root: `{mimic_root}`",
        f"- Status: `{status}`",
        f"- Errors: {error_count}",
        f"- Warnings: {warning_count}",
        f"- Windows: {len(spec.get('windows', {}))}",
        f"- Feature sources: {len(spec.get('feature_sources', {}))}",
        f"- Cohorts: {len(spec.get('cohorts', {}))}",
        "",
        "## Window Summary",
        "",
        "| Window | Start | End | Usable prediction times | Forbidden prediction times |",
        "|---|---|---|---|---|",
    ]
    for name, window in spec.get("windows", {}).items():
        lines.append(
            f"| {name} | `{window.get('start')}` | `{window.get('end')}` | "
            f"{', '.join(window.get('usable_for_prediction_times', [])) or 'none'} | "
            f"{', '.join(window.get('forbidden_for_prediction_times', [])) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Generated Feature Outputs",
            "",
            "| Source | Cohort | Window | Path | Exists |",
            "|---|---|---|---|---|",
            *rows,
            "",
            "## Issues",
            "",
        ]
    )
    if issues:
        for issue in issues:
            lines.append(f"- **{issue['severity'].upper()}**: {issue['message']}")
    else:
        lines.append("- No issues found.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_FEATURE_WINDOW_SPEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_feature_window_spec(args.spec)
    issues: list[dict[str, str]] = []
    validate_windows(spec, issues)
    validate_cohorts(args.project_root, spec, issues)
    validate_sources(args.project_root, args.mimic_root, spec, issues)
    validate_vital_itemids(args.mimic_root, spec, issues)
    output_path = args.project_root / "outputs" / "reports" / "feature_window_spec_validation_report.md"
    write_report(args.spec, args.project_root, args.mimic_root, spec, issues, output_path)
    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    print(f"Feature-window spec validation: errors={error_count}, warnings={warning_count}")
    print(f"Wrote {output_path}")
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
