#!/usr/bin/env python3
"""Generate a study package summary from registry, config, and pipeline outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_study_config import PROJECT, flatten_output_paths, load_yaml_with_ruby


REGISTRY = PROJECT / "configs" / "study_registry.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_study(registry: dict[str, Any], study_id: str | None) -> dict[str, Any]:
    selected_id = study_id or registry.get("active_study")
    for study in registry.get("studies", []):
        if study.get("id") == selected_id:
            return study
    available = ", ".join(study.get("id", "") for study in registry.get("studies", [])) or "none"
    raise SystemExit(f"Unknown study id `{selected_id}`. Available studies: {available}")


def file_status(project_root: Path, relative_path: str) -> dict[str, Any]:
    path = project_root / relative_path
    if not path.exists():
        return {"path": relative_path, "exists": False}
    return {
        "path": relative_path,
        "exists": True,
        "size_bytes": path.stat().st_size,
    }


def config_summary(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    cohort = config.get("cohort_definition", {})
    outcome = config.get("outcome", {})
    prediction_times = config.get("prediction_times", {})
    primary_prediction = prediction_times.get("primary", {}) if isinstance(prediction_times, dict) else {}
    output_paths = flatten_output_paths(config.get("outputs", {}))
    return {
        "dataset_name": dataset.get("name", "NA"),
        "dataset_root": dataset.get("root_path", "NA"),
        "dataset_tables": dataset.get("tables", {}),
        "study_name": cohort.get("study_name", "NA"),
        "population": cohort.get("population", "NA"),
        "outcome_name": outcome.get("name", "NA"),
        "outcome_definition": outcome.get("definition", "NA"),
        "followup_window_days": outcome.get("followup_window_days", "NA"),
        "prediction_time": outcome.get("prediction_time", "NA"),
        "primary_prediction_name": primary_prediction.get("name", "NA") if isinstance(primary_prediction, dict) else "NA",
        "outputs_listed": len(output_paths),
    }


def markdown_performance_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No model performance found."
    lines = [
        "| Feature set | AUROC | AUPRC | Brier | PPV | NPV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature_set']} | {row['AUROC']:.4f} | {row['AUPRC']:.4f} | "
            f"{row['Brier_score']:.4f} | {row['PPV']:.4f} | {row['NPV']:.4f} |"
        )
    return "\n".join(lines)


def markdown_prediction_time_table(project_root: Path) -> str:
    table_path = project_root / "outputs" / "tables" / "mimic_diabetes_prediction_time_model_performance.csv"
    if not table_path.exists():
        return "No prediction-time model table found."
    import pandas as pd

    df = pd.read_csv(table_path)
    tests = df[df["split"].eq("test")].copy()
    if tests.empty:
        return "No test split rows found."
    order = [
        "admission_safe_minimal",
        "inhospital_24h_lab_minimal",
        "inhospital_24h_lab_med_minimal",
        "discharge_safe_minimal",
    ]
    tests["sort_order"] = tests["feature_set"].map({name: i for i, name in enumerate(order)}).fillna(99)
    tests = tests.sort_values(["sort_order", "feature_set"])
    lines = [
        "| Feature set | Prediction time | AUROC | AUPRC | Brier |",
        "|---|---|---:|---:|---:|",
    ]
    for row in tests.itertuples(index=False):
        lines.append(
            f"| {row.feature_set} | {row.prediction_time} | {row.AUROC:.4f} | "
            f"{row.AUPRC:.4f} | {row.Brier_score:.4f} |"
        )
    return "\n".join(lines)


def build_package(project_root: Path, registry_path: Path, study_id: str | None) -> dict[str, Any]:
    registry = load_json(registry_path)
    study = find_study(registry, study_id)
    config_path = project_root / study["config"]
    config = load_yaml_with_ruby(config_path)
    manifest_path = project_root / "outputs" / "pipeline_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}

    output_paths = flatten_output_paths(config.get("outputs", {}))
    self_outputs = {"outputs/study_package.json", "outputs/reports/study_package.md"}
    outputs = [file_status(project_root, path) for path in output_paths if path not in self_outputs]
    missing_outputs = [item["path"] for item in outputs if not item["exists"]]

    return {
        "study": study,
        "config_path": str(config_path),
        "config_summary": config_summary(config),
        "pipeline_manifest_path": str(manifest_path),
        "pipeline_status": manifest.get("status", "unknown"),
        "missing_outputs": missing_outputs,
        "outputs_checked": len(outputs),
        "metrics": manifest.get("metrics", {}),
        "recommended_next_steps": manifest.get("recommended_next_steps", []),
        "key_reports": [
            "outputs/reports/mimic_diabetes_methods_results_draft.md",
            "outputs/reports/mimic_diabetes_prediction_time_model_report.md",
            "outputs/reports/mimic_diabetes_prediction_time_figure_report.md",
            "outputs/reports/mimic_diabetes_comprehensive_audit_report.md",
            "outputs/reports/mimic_diabetes_feature_time_audit_report.md",
            "outputs/reports/mimic_diabetes_leakage_sensitivity_report.md",
            "outputs/reports/study_config_validation_report.md",
            "outputs/reports/study_registry_validation_report.md",
            "outputs/reports/study_inspection.md",
            "outputs/reports/pipeline_status.md",
        ],
    }


def write_json(package: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(package: dict[str, Any], project_root: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    study = package["study"]
    cfg = package["config_summary"]
    metrics = package.get("metrics", {})
    cohort = metrics.get("cohort", {})
    missing_outputs = package.get("missing_outputs", [])
    missing_text = "None" if not missing_outputs else ", ".join(f"`{item}`" for item in missing_outputs)
    key_reports = [report for report in package["key_reports"] if (project_root / report).exists()]

    text = f"""# ChronoEHR-Agent Study Package

## Study Identity

- Study id: `{study.get("id", "NA")}`
- Status: `{study.get("status", "NA")}`
- Config: `{study.get("config", "NA")}`
- Dataset root: `{cfg["dataset_root"]}`
- Cohort: `{study.get("cohort", "NA")}`
- Outcome: `{study.get("outcome", cfg["outcome_name"])}`

## Study Design Snapshot

- Study name: {cfg["study_name"]}
- Population: {cfg["population"]}
- Outcome definition: {cfg["outcome_definition"]}
- Follow-up window: {cfg["followup_window_days"]} days
- Primary prediction time: `{cfg["prediction_time"]}`
- Primary prediction name: `{cfg["primary_prediction_name"]}`

## Pipeline Status

- Pipeline status: `{package["pipeline_status"]}`
- Config outputs checked before package self-files: {package["outputs_checked"]}
- Missing outputs: {missing_text}

## Cohort Summary

- Final index admissions: {cohort.get("final_index_admissions", "NA")}
- Final subjects: {cohort.get("final_subjects", "NA")}
- 30-day readmission count: {cohort.get("readmission_30d_count", "NA")}
- 30-day readmission rate: {cohort.get("readmission_30d_rate", 0):.2%}

## Core Model Performance

{markdown_performance_table(metrics.get("test_performance", []))}

## Prediction-Time Model Comparison

{markdown_prediction_time_table(project_root)}

## Key Reports

{chr(10).join(f"- `{report}`" for report in key_reports)}

## Recommended Next Steps

{chr(10).join(f"- {step}" for step in package.get("recommended_next_steps", []))}

## Reproducibility Commands

```bash
cd {project_root}
python3 src/chrono_ehr/run_study.py --list
python3 src/chrono_ehr/run_study.py --skip-existing --no-expensive
python3 src/chrono_ehr/run_diabetes_demo.py --only manifest validate_config
```
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--study", help="Study id. Defaults to active_study in registry.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = build_package(args.project_root, args.registry, args.study)
    json_path = args.project_root / "outputs" / "study_package.json"
    md_path = args.project_root / "outputs" / "reports" / "study_package.md"
    write_json(package, json_path)
    write_markdown(package, args.project_root, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
