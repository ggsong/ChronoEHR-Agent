#!/usr/bin/env python3
"""Run the MIMIC heart failure demo pipeline with resumable steps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]

STEPS = [
    {
        "name": "cohort",
        "script": "mimic_heart_failure_cohort.py",
        "outputs": ["data/processed/mimic_heart_failure_readmission_cohort.csv"],
        "expensive": False,
    },
    {
        "name": "lab_features",
        "script": "mimic_heart_failure_lab_features.py",
        "outputs": [
            "data/processed/mimic_heart_failure_lab_features_24h.csv",
            "data/processed/mimic_heart_failure_lab_features_discharge.csv",
            "outputs/reports/mimic_heart_failure_lab_feature_report.md",
        ],
        "expensive": True,
    },
    {
        "name": "feature_time_audit",
        "script": "mimic_heart_failure_feature_time_audit.py",
        "outputs": ["outputs/reports/mimic_heart_failure_feature_time_audit_report.md"],
        "expensive": False,
    },
    {
        "name": "prediction_time_models",
        "script": "mimic_heart_failure_prediction_time_models.py",
        "outputs": ["outputs/tables/mimic_heart_failure_prediction_time_model_performance.csv"],
        "expensive": False,
    },
    {
        "name": "prediction_time_figures",
        "script": "mimic_heart_failure_prediction_time_figures.py",
        "outputs": [
            "outputs/figures/mimic_heart_failure_prediction_time_auroc.png",
            "outputs/figures/mimic_heart_failure_prediction_time_auprc.png",
        ],
        "expensive": False,
    },
    {
        "name": "outcome_sensitivity",
        "script": "mimic_heart_failure_outcome_sensitivity.py",
        "outputs": ["outputs/tables/mimic_heart_failure_outcome_sensitivity.csv"],
        "expensive": False,
    },
    {
        "name": "leakage_sensitivity",
        "script": "mimic_heart_failure_leakage_sensitivity.py",
        "outputs": ["outputs/tables/mimic_heart_failure_leakage_sensitivity.csv"],
        "expensive": False,
    },
    {
        "name": "validate_registry",
        "script": "validate_study_registry.py",
        "outputs": ["outputs/reports/study_registry_validation_report.md"],
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
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-expensive", action="store_true")
    parser.add_argument("--only", nargs="*", choices=[step["name"] for step in STEPS])
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
    print("\nHeart failure pipeline complete.")


if __name__ == "__main__":
    main()
