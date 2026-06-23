#!/usr/bin/env python3
"""Run the MIMIC diabetes demo pipeline with resumable steps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]

STEPS = [
    {
        "name": "cohort",
        "script": "mimic_diabetes_cohort.py",
        "outputs": ["data/processed/mimic_diabetes_readmission_cohort.csv"],
        "expensive": False,
    },
    {
        "name": "lab_features",
        "script": "mimic_diabetes_lab_features.py",
        "outputs": ["data/processed/mimic_diabetes_lab_features.csv"],
        "expensive": True,
    },
    {
        "name": "lab24h_features",
        "script": "mimic_diabetes_lab_features_24h.py",
        "outputs": ["data/processed/mimic_diabetes_lab_features_24h.csv"],
        "expensive": True,
    },
    {
        "name": "med_features",
        "script": "mimic_diabetes_med_features.py",
        "outputs": ["data/processed/mimic_diabetes_med_features.csv"],
        "expensive": True,
    },
    {
        "name": "med24h_features",
        "script": "mimic_diabetes_med_features_24h.py",
        "outputs": ["data/processed/mimic_diabetes_med_features_24h.csv"],
        "expensive": True,
    },
    {
        "name": "baseline",
        "script": "mimic_diabetes_baseline.py",
        "outputs": ["outputs/tables/mimic_diabetes_model_performance.csv"],
        "expensive": False,
    },
    {
        "name": "figures",
        "script": "mimic_diabetes_figures.py",
        "outputs": ["outputs/figures/mimic_diabetes_roc_curve.png"],
        "expensive": False,
    },
    {
        "name": "leakage_sensitivity",
        "script": "mimic_diabetes_leakage_sensitivity.py",
        "outputs": ["outputs/tables/mimic_diabetes_leakage_sensitivity.csv"],
        "expensive": False,
    },
    {
        "name": "feature_time_audit",
        "script": "audit_feature_time_map.py",
        "outputs": ["outputs/tables/mimic_diabetes_feature_time_audit.csv"],
        "expensive": False,
    },
    {
        "name": "prediction_time_comparison",
        "script": "compare_prediction_times.py",
        "outputs": ["outputs/tables/mimic_diabetes_prediction_time_comparison.csv"],
        "expensive": False,
    },
    {
        "name": "prediction_time_models",
        "script": "mimic_diabetes_prediction_time_models.py",
        "outputs": ["outputs/tables/mimic_diabetes_prediction_time_model_performance.csv"],
        "expensive": False,
    },
    {
        "name": "prediction_time_figures",
        "script": "mimic_diabetes_prediction_time_figures.py",
        "outputs": ["outputs/figures/mimic_diabetes_prediction_time_auroc.png"],
        "expensive": False,
    },
    {
        "name": "outcome_sensitivity",
        "script": "mimic_diabetes_outcome_sensitivity.py",
        "outputs": ["outputs/tables/mimic_diabetes_outcome_sensitivity.csv"],
        "expensive": False,
    },
    {
        "name": "uncertainty",
        "script": "mimic_diabetes_model_uncertainty.py",
        "outputs": ["outputs/tables/mimic_diabetes_model_performance_bootstrap_ci.csv"],
        "expensive": False,
    },
    {
        "name": "report",
        "script": "mimic_diabetes_report.py",
        "outputs": ["outputs/reports/mimic_diabetes_methods_results_draft.md"],
        "expensive": False,
    },
    {
        "name": "comprehensive_audit",
        "script": "mimic_diabetes_comprehensive_audit.py",
        "outputs": ["outputs/reports/mimic_diabetes_comprehensive_audit_report.md"],
        "expensive": False,
    },
    {
        "name": "manifest",
        "script": "mimic_diabetes_manifest.py",
        "outputs": ["outputs/pipeline_manifest.json", "outputs/reports/pipeline_status.md"],
        "expensive": False,
    },
    {
        "name": "study_package",
        "script": "generate_study_package.py",
        "outputs": ["outputs/study_package.json", "outputs/reports/study_package.md"],
        "expensive": False,
    },
    {
        "name": "inspect_study",
        "script": "inspect_study.py",
        "outputs": ["outputs/reports/study_inspection.md"],
        "expensive": False,
    },
    {
        "name": "validate_registry",
        "script": "validate_study_registry.py",
        "outputs": ["outputs/reports/study_registry_validation_report.md"],
        "expensive": False,
    },
    {
        "name": "validate_config",
        "script": "validate_study_config.py",
        "outputs": ["outputs/reports/study_config_validation_report.md"],
        "expensive": False,
    },
]


def outputs_exist(step: dict, project: Path) -> bool:
    return all((project / output).exists() for output in step["outputs"])


def run_step(step: dict, project: Path) -> None:
    script_path = project / "src" / "chrono_ehr" / step["script"]
    cmd = [sys.executable, str(script_path)]
    print(f"\n=== Running {step['name']} ===")
    subprocess.run(cmd, cwd=project, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a step when its expected outputs already exist.",
    )
    parser.add_argument(
        "--no-expensive",
        action="store_true",
        help="Skip expensive raw-table scans such as labs and medications.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[step["name"] for step in STEPS],
        help="Run only selected step names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.only) if args.only else None
    for step in STEPS:
        if selected is not None and step["name"] not in selected:
            continue
        if args.no_expensive and step["expensive"]:
            print(f"Skipping expensive step {step['name']}")
            continue
        if args.skip_existing and outputs_exist(step, args.project_root):
            print(f"Skipping existing step {step['name']}")
            continue
        run_step(step, args.project_root)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
