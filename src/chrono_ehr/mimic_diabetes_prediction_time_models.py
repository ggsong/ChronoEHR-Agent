#!/usr/bin/env python3
"""Compare diabetes admission, 24h, and discharge-time logistic baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from mimic_diabetes_baseline import DEFAULT_PROJECT
from prediction_time_model_tools import run_prediction_time_models
from prediction_time_spec_loader import load_prediction_time_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_prediction_time_config("diabetes")
    outputs = config["outputs"]
    performance = run_prediction_time_models(
        project_root=args.project_root,
        cohort_path=config["cohort_path"],
        specs=config["specs"],
        performance_path=args.project_root / outputs["performance"],
        coefficient_path=args.project_root / outputs["coefficients"],
        prediction_path=args.project_root / outputs["predictions"],
        report_path=args.project_root / outputs["report"],
        report_options=config["report"],
    )

    print(f"Wrote {args.project_root / outputs['performance']}")
    print(f"Wrote {args.project_root / outputs['report']}")
    for row in performance[performance["split"].eq("test")].itertuples(index=False):
        print(f"{row.feature_set}: AUROC={row.AUROC:.4f} AUPRC={row.AUPRC:.4f}")


if __name__ == "__main__":
    main()
