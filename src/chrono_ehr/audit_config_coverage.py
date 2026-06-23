#!/usr/bin/env python3
"""Audit how much of the main workflow is described by configuration files."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = DEFAULT_PROJECT / "configs" / "study_registry.json"
MIMIC_STUDIES = {
    "mimic_iv_3_1_diabetes_readmission",
    "mimic_iv_ckd_readmission",
    "mimic_iv_heart_failure_readmission",
    "mimic_iv_hypertension_readmission",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_yaml_with_ruby(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    ruby = "require 'yaml'; require 'json'; data = YAML.load_file(ARGV[0]); puts JSON.generate(data)"
    result = subprocess.run(["ruby", "-e", ruby, str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def status(has_definition: bool, runner_uses_config: bool) -> str:
    if has_definition and runner_uses_config:
        return "CONFIG_DRIVEN"
    if has_definition:
        return "CONFIGURED_BUT_RUNNER_PARTLY_HARDCODED"
    return "MISSING_CONFIG"


def runner_text(project_root: Path, study: dict[str, Any]) -> str:
    pipeline = project_root / study.get("pipeline", "")
    if not pipeline.exists():
        return ""
    related = [pipeline]
    cohort = study.get("cohort", "")
    related.extend(sorted((project_root / "src" / "chrono_ehr").glob(f"*{cohort}*.py")))
    if study.get("id", "") in MIMIC_STUDIES:
        related.extend(
            [
                project_root / "src" / "chrono_ehr" / "mimic_diagnosis_cohort_builder.py",
                project_root / "src" / "chrono_ehr" / "study_config_loader.py",
            ]
        )
    text = []
    for path in dict.fromkeys(related):
        try:
            text.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return "\n".join(text)


def configured_sections(config: dict[str, Any]) -> dict[str, bool]:
    cohort = config.get("cohort_definition", {}) if isinstance(config.get("cohort_definition"), dict) else {}
    code_rules = any(key.endswith("_code_rules") or key in {"diabetes_code_rules"} for key in cohort)
    outcome = config.get("outcome", {}) if isinstance(config.get("outcome"), dict) else {}
    prediction_times = config.get("prediction_times", {}) if isinstance(config.get("prediction_times"), dict) else {}
    feature_sets = config.get("feature_sets", {}) if isinstance(config.get("feature_sets"), dict) else {}
    forbidden = any("forbidden" in key or "high_risk" in key for key in feature_sets)
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs"), dict) else {}
    return {
        "dataset_tables": bool(config.get("dataset", {}).get("tables")) if isinstance(config.get("dataset"), dict) else False,
        "cohort_code_rules": code_rules,
        "inclusion_exclusion": bool(cohort.get("inclusion_criteria")) and bool(cohort.get("exclusion_criteria")),
        "outcome_window": bool(outcome.get("name")) and bool(outcome.get("followup_window_days")),
        "prediction_times": bool(prediction_times),
        "feature_sets": bool(feature_sets),
        "forbidden_features": forbidden,
        "models_metrics_split": all(key in config for key in ["models", "metrics", "split_strategy"]),
        "declared_outputs": bool(outputs),
    }


def runner_config_usage(text: str, study: dict[str, Any]) -> dict[str, bool]:
    config_name = Path(study.get("config", "")).name
    has_config_ref = (
        config_name in text
        or "--config" in text
        or "load_yaml" in text
        or "load_cohort_code_rules" in text
        or "spec_code_prefixes" in text
        or "build_diagnosis_readmission_cohort" in text
        or "load_prediction_time_config" in text
    )
    return {
        "dataset_tables": has_config_ref,
        "cohort_code_rules": has_config_ref and (
            "code_rules" in text or "code_rule_key" in text or "spec_code_prefixes" in text
        ),
        "inclusion_exclusion": has_config_ref,
        "outcome_window": has_config_ref,
        "prediction_times": "prediction_time_model_specs" in text or "load_prediction_time_config" in text or "prediction_time" in text,
        "feature_sets": "prediction_time_model_specs" in text or "load_prediction_time_config" in text or "feature_sets" in text,
        "forbidden_features": "forbidden" in text or "leakage" in text,
        "models_metrics_split": "prediction_time_model_specs" in text or "load_prediction_time_config" in text or "split" in text,
        "declared_outputs": "STEPS" in text or has_config_ref,
    }


def collect_rows(project_root: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in registry.get("studies", []):
        study_id = study.get("id", "")
        config_path = project_root / study.get("config", "")
        config = load_yaml_with_ruby(config_path)
        definitions = configured_sections(config)
        usage = runner_config_usage(runner_text(project_root, study), study)
        for area, has_definition in definitions.items():
            uses_config = usage.get(area, False)
            rows.append(
                {
                    "study_id": study_id,
                    "cohort": study.get("cohort", ""),
                    "area": area,
                    "config_path": str(config_path.relative_to(project_root)) if config_path.exists() else study.get("config", ""),
                    "has_config_definition": has_definition,
                    "runner_appears_config_driven": uses_config,
                    "status": status(has_definition, uses_config),
                    "action": action_for(area, has_definition, uses_config, study_id),
                }
            )
    rows.extend(shared_spec_rows(project_root))
    return rows


def action_for(area: str, has_definition: bool, uses_config: bool, study_id: str) -> str:
    if not has_definition:
        return f"Add `{area}` to the study config before extending `{study_id}`."
    if not uses_config:
        return f"Keep current behavior for v0.1, then migrate `{area}` from script constants to config loading."
    return "No immediate action."


def shared_spec_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared = [
        ("feature_window_specs", "configs/feature_window_specs.json", "Temporal Feature Agent"),
        ("prediction_time_model_specs", "configs/prediction_time_model_specs.json", "Benchmark Agent"),
        ("agent_action_catalog", "configs/agent_action_catalog.json", "Study Registry / Runner"),
        ("agent_demo_workflows", "configs/agent_demo_workflows.json", "Study Registry / Runner"),
        ("agent_entrypoints", "configs/agent_entrypoints.json", "Study Registry / Runner"),
    ]
    for area, relative, owner in shared:
        path = project_root / relative
        rows.append(
            {
                "study_id": "shared",
                "cohort": owner,
                "area": area,
                "config_path": relative,
                "has_config_definition": path.exists() and path.stat().st_size > 0,
                "runner_appears_config_driven": True,
                "status": "CONFIG_DRIVEN" if path.exists() and path.stat().st_size > 0 else "MISSING_CONFIG",
                "action": "No immediate action." if path.exists() else f"Create {relative}.",
            }
        )
    return rows


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        details.groupby("study_id")
        .agg(
            areas=("area", "count"),
            config_driven=("status", lambda values: int((values == "CONFIG_DRIVEN").sum())),
            configured_partial=("status", lambda values: int((values == "CONFIGURED_BUT_RUNNER_PARTLY_HARDCODED").sum())),
            missing_config=("status", lambda values: int((values == "MISSING_CONFIG").sum())),
        )
        .reset_index()
    )
    grouped["config_definition_coverage"] = (
        (grouped["areas"] - grouped["missing_config"]) / grouped["areas"] * 100
    ).round(1)
    grouped["runner_config_driven_coverage"] = (grouped["config_driven"] / grouped["areas"] * 100).round(1)
    grouped["overall_status"] = grouped.apply(overall_status, axis=1)
    return grouped.sort_values(["overall_status", "config_definition_coverage", "study_id"], ascending=[True, False, True])


def overall_status(row: pd.Series) -> str:
    if row["missing_config"] > 0:
        return "CONFIG_GAPS"
    if row["configured_partial"] > 0:
        return "PARTIAL_RUNNER_MIGRATION"
    return "CONFIG_DRIVEN"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.astype(object).where(pd.notna(df), "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, details: pd.DataFrame, summary: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "config_coverage_audit.md"
    partial = details[details["status"].ne("CONFIG_DRIVEN")]
    output.write_text(
        f"""# Config Coverage Audit

- Purpose: show which study definitions are already represented in config and which runner behaviors remain partly hard-coded.
- Boundary: local research workflow maintainability audit only; no medical QA, diagnosis, or treatment recommendation.

## Summary

{markdown_table(summary)}

## Gaps And Migration Items

{markdown_table(partial[["study_id", "cohort", "area", "status", "action"]]) if not partial.empty else "All checked areas are config-driven."}

## Interpretation

- `CONFIG_DRIVEN`: definition exists and the relevant runner appears to consume config/spec files.
- `CONFIGURED_BUT_RUNNER_PARTLY_HARDCODED`: acceptable for v0.1, but should be migrated gradually.
- `MISSING_CONFIG`: should be fixed before expanding that study.
""",
        encoding="utf-8",
    )
    return output


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    details = pd.DataFrame(collect_rows(args.project_root, registry))
    summary = summarize(details)
    table_dir = args.project_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(table_dir / "config_coverage_audit.csv", index=False)
    summary.to_csv(table_dir / "config_coverage_summary.csv", index=False)
    report = write_report(args.project_root, details, summary)
    print(f"Wrote {report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
