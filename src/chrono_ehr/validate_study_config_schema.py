#!/usr/bin/env python3
"""Validate registry study configs as research-workflow schemas."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = DEFAULT_PROJECT / "configs" / "study_registry.json"

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
REQUIRED_DATASET_KEYS = ["name", "type", "root_path", "id_columns"]
REQUIRED_OUTCOME_KEYS = ["name", "type"]
REQUIRED_SPLIT_KEYS = ["method", "group_column", "train_size", "validation_size", "test_size"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_yaml_with_ruby(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    ruby = "require 'yaml'; require 'json'; data = YAML.load_file(ARGV[0]); puts JSON.generate(data)"
    result = subprocess.run(["ruby", "-e", ruby, str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def row(study_id: str, config_path: str, section: str, check: str, status: str, detail: str) -> dict[str, str]:
    return {
        "study_id": study_id,
        "config_path": config_path,
        "section": section,
        "check": check,
        "status": status,
        "detail": detail,
    }


def add(rows: list[dict[str, str]], study_id: str, config_path: str, section: str, check: str, passed: bool, detail: str, fail_status: str = "FAIL") -> None:
    rows.append(row(study_id, config_path, section, check, "PASS" if passed else fail_status, detail))


def flatten_output_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        paths: list[str] = []
        for item in value.values():
            paths.extend(flatten_output_paths(item))
        return paths
    if isinstance(value, list):
        paths: list[str] = []
        for item in value:
            paths.extend(flatten_output_paths(item))
        return paths
    return []


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def table_mapping(dataset: dict[str, Any]) -> dict[str, Any]:
    tables = dataset.get("tables")
    if isinstance(tables, dict):
        return tables
    files = dataset.get("files")
    if isinstance(files, dict):
        return files
    return {}


def validate_config(project_root: Path, study: dict[str, Any], planned: bool) -> list[dict[str, str]]:
    study_id = str(study.get("id", "unknown"))
    config_relative = str(study.get("config", ""))
    config_path = project_root / config_relative
    rows: list[dict[str, str]] = []
    config = load_yaml_with_ruby(config_path)
    add(rows, study_id, config_relative, "file", "config_exists", bool(config), config_relative)
    if not config:
        return rows

    for key in REQUIRED_TOP_LEVEL:
        add(rows, study_id, config_relative, "top_level", f"has_{key}", key in config, key)

    dataset = section(config, "dataset")
    for key in REQUIRED_DATASET_KEYS:
        add(rows, study_id, config_relative, "dataset", f"has_{key}", key in dataset, key)
    table_map = table_mapping(dataset)
    add(rows, study_id, config_relative, "dataset", "has_tables_or_files", bool(table_map), f"count={len(table_map)}")
    id_columns = dataset.get("id_columns", {}) if isinstance(dataset.get("id_columns"), dict) else {}
    non_null_ids = [key for key, value in id_columns.items() if value not in (None, "")]
    id_ok = bool(non_null_ids) if planned else len(non_null_ids) == len(id_columns) and bool(non_null_ids)
    add(rows, study_id, config_relative, "dataset", "id_columns_named", id_ok, f"named={len(non_null_ids)} total={len(id_columns)}", fail_status="WARN" if planned else "FAIL")
    root = Path(os.path.expandvars(str(dataset.get("root_path", "")))).expanduser()
    add(rows, study_id, config_relative, "dataset", "dataset_root_exists", root.exists(), str(root), fail_status="WARN" if planned else "FAIL")
    existing_tables = 0
    checked_tables = 0
    for _, relative in table_map.items():
        if relative in (None, ""):
            continue
        checked_tables += 1
        if (root / str(relative)).exists():
            existing_tables += 1
    add(
        rows,
        study_id,
        config_relative,
        "dataset",
        "declared_raw_files_exist",
        existing_tables == checked_tables and checked_tables > 0,
        f"existing={existing_tables} checked={checked_tables}",
        fail_status="WARN" if planned else "FAIL",
    )

    cohort = section(config, "cohort_definition")
    add(rows, study_id, config_relative, "cohort_definition", "has_population", bool(cohort.get("population")), "population")
    add(rows, study_id, config_relative, "cohort_definition", "has_inclusion_criteria", bool(cohort.get("inclusion_criteria")), "inclusion_criteria")
    add(rows, study_id, config_relative, "cohort_definition", "has_exclusion_criteria", bool(cohort.get("exclusion_criteria")), "exclusion_criteria")
    code_rules = [key for key in cohort if str(key).endswith("_code_rules")]
    if "mimic" in study_id:
        add(rows, study_id, config_relative, "cohort_definition", "has_icd_code_rules", bool(code_rules), ", ".join(code_rules))

    outcome = section(config, "outcome")
    for key in REQUIRED_OUTCOME_KEYS:
        add(rows, study_id, config_relative, "outcome", f"has_{key}", key in outcome, key)
    add(rows, study_id, config_relative, "outcome", "binary_or_declared", bool(outcome.get("type")), str(outcome.get("type", "")))
    has_window = any(key in outcome for key in ["followup_window_days", "primary_followup_wave", "secondary_followup_wave", "candidate_columns"])
    add(rows, study_id, config_relative, "outcome", "has_followup_or_source_window", has_window, "followup_window/source columns")

    prediction_times = config.get("prediction_times", {})
    add(rows, study_id, config_relative, "prediction_times", "non_empty", isinstance(prediction_times, dict) and bool(prediction_times), f"type={type(prediction_times).__name__}")

    feature_sets = section(config, "feature_sets")
    enabled_sets = [name for name, spec in feature_sets.items() if isinstance(spec, dict) and spec.get("enabled") is True]
    add(rows, study_id, config_relative, "feature_sets", "has_enabled_feature_set", bool(enabled_sets), ", ".join(enabled_sets))
    feature_payload_ok = True
    for name, spec in feature_sets.items():
        if not isinstance(spec, dict):
            feature_payload_ok = False
            continue
        payload_keys = {
            "variables",
            "variable_examples",
            "source_tables",
            "aggregations",
            "windows_hours",
            "description",
            "warning",
            "allowed_wave",
            "allowed_offset",
        }
        if not payload_keys.intersection(spec):
            feature_payload_ok = False
    add(rows, study_id, config_relative, "feature_sets", "feature_sets_have_payload", feature_payload_ok and bool(feature_sets), f"sets={len(feature_sets)}")
    forbidden_top = config.get("forbidden_or_high_risk_features")
    forbidden_nested = any("forbidden" in str(name) or "high_risk" in str(name) for name in feature_sets)
    add(rows, study_id, config_relative, "feature_sets", "has_forbidden_or_high_risk_list", bool(forbidden_top) or forbidden_nested, "forbidden/high-risk")

    metrics = config.get("metrics", {})
    required_metrics = metrics.get("required", []) if isinstance(metrics, dict) else []
    add(rows, study_id, config_relative, "metrics", "has_auroc", "AUROC" in required_metrics, ", ".join(map(str, required_metrics)))
    add(rows, study_id, config_relative, "metrics", "has_auprc", "AUPRC" in required_metrics, ", ".join(map(str, required_metrics)))

    split = section(config, "split_strategy")
    for key in REQUIRED_SPLIT_KEYS:
        add(rows, study_id, config_relative, "split_strategy", f"has_{key}", key in split, key)
    sizes = [split.get("train_size"), split.get("validation_size"), split.get("test_size")]
    if all(isinstance(value, int | float) for value in sizes):
        total = float(sum(sizes))
        add(rows, study_id, config_relative, "split_strategy", "split_sizes_sum_to_one", abs(total - 1.0) < 0.001, f"sum={total:.3f}")
    patient_level = any(token in str(split.get("method", "")).lower() for token in ["patient", "subject", "person"])
    add(rows, study_id, config_relative, "split_strategy", "patient_or_person_level", patient_level, str(split.get("method", "")))

    outputs = flatten_output_paths(config.get("outputs", {}))
    relative_outputs = [path for path in outputs if path and not Path(path).is_absolute()]
    add(rows, study_id, config_relative, "outputs", "has_declared_outputs", bool(outputs), f"count={len(outputs)}")
    add(rows, study_id, config_relative, "outputs", "outputs_are_relative", len(relative_outputs) == len(outputs), f"relative={len(relative_outputs)} total={len(outputs)}")
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["study_id", "checks", "pass", "warn", "fail", "status"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in item) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, rows: pd.DataFrame) -> Path:
    table_path = project_root / "outputs" / "tables" / "study_config_schema_validation.csv"
    report_path = project_root / "outputs" / "reports" / "study_config_schema_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(table_path, index=False)
    summary = (
        rows.groupby("study_id")
        .agg(
            checks=("check", "count"),
            pass_count=("status", lambda values: int((values == "PASS").sum())),
            warn=("status", lambda values: int((values == "WARN").sum())),
            fail=("status", lambda values: int((values == "FAIL").sum())),
        )
        .reset_index()
        .rename(columns={"pass_count": "pass"})
    )
    summary["status"] = summary["fail"].apply(lambda value: "PASS" if int(value) == 0 else "FAIL")
    failures = rows[rows["status"].eq("FAIL")]
    warnings = rows[rows["status"].eq("WARN")]
    report_path.write_text(
        f"""# Study Config Schema Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(rows)}
- Failures: {len(failures)}
- Warnings: {len(warnings)}
- Boundary: local research workflow schema validation only; no medical QA, diagnosis, or treatment recommendation.

## Summary

{markdown_table(summary)}

## Failures

{markdown_detail(failures) if not failures.empty else "No failures."}

## Warnings

{markdown_detail(warnings) if not warnings.empty else "No warnings."}
""",
        encoding="utf-8",
    )
    return report_path


def markdown_detail(df: pd.DataFrame) -> str:
    lines = []
    for item in df.itertuples(index=False):
        lines.append(f"- `{item.study_id}` `{item.section}.{item.check}`: {item.detail}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    registry = load_json(args.registry)
    all_rows: list[dict[str, str]] = []
    for study in registry.get("studies", []):
        all_rows.extend(validate_config(args.project_root, study, planned=False))
    for study in registry.get("planned_studies", []):
        all_rows.extend(validate_config(args.project_root, study, planned=True))
    rows = pd.DataFrame(all_rows)
    report = write_outputs(args.project_root, rows)
    failures = int(rows["status"].eq("FAIL").sum()) if not rows.empty else 1
    warnings = int(rows["status"].eq("WARN").sum()) if not rows.empty else 0
    print(f"Study config schema validation checks: {len(rows)}")
    print(f"Failures: {failures}")
    print(f"Warnings: {warnings}")
    print(f"Wrote {report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
