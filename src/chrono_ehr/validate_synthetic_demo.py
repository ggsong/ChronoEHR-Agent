#!/usr/bin/env python3
"""Validate synthetic ChronoEHR-Agent demo artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
REQUIRED_COHORT_COLUMNS = {
    "synthetic_admission_id",
    "prediction_time",
    "age",
    "prior_admissions_count",
    "length_of_stay_days",
    "abnormal_lab_count",
    "medication_change_count",
    "readmission_30d",
    "split",
    "synthetic_risk_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_dir = args.project_root / "outputs" / "demo"
    cohort_path = demo_dir / "synthetic_cohort.csv"
    metrics_path = demo_dir / "synthetic_demo_metrics.csv"
    report_path = demo_dir / "synthetic_demo_report.md"
    missing_files = [path for path in [cohort_path, metrics_path, report_path] if not path.exists()]
    if missing_files:
        raise SystemExit("Missing synthetic demo artifacts: " + ", ".join(str(path) for path in missing_files))

    cohort = pd.read_csv(cohort_path)
    metrics = pd.read_csv(metrics_path)
    missing_columns = sorted(REQUIRED_COHORT_COLUMNS - set(cohort.columns))
    if missing_columns:
        raise SystemExit("Synthetic cohort missing columns: " + ", ".join(missing_columns))
    if len(cohort) < 50:
        raise SystemExit("Synthetic cohort is unexpectedly small.")
    if not set(cohort["split"]).issuperset({"train", "test"}):
        raise SystemExit("Synthetic cohort must contain train and test splits.")
    if not cohort["synthetic_risk_score"].between(0, 1).all():
        raise SystemExit("Synthetic risk scores must be in [0, 1].")
    if not set(metrics["split"]).issuperset({"train", "test"}):
        raise SystemExit("Synthetic metrics must contain train and test rows.")

    print(f"Synthetic demo validation PASS: {len(cohort)} rows, {len(metrics)} metric rows.")


if __name__ == "__main__":
    main()
