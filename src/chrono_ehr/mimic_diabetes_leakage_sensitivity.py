#!/usr/bin/env python3
"""Run leakage sensitivity for the MIMIC diabetes demo."""

from __future__ import annotations

from pathlib import Path

from sensitivity_tools import run_leakage_sensitivity


PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    out = run_leakage_sensitivity(
        cohort_path=PROJECT / "data" / "processed" / "mimic_diabetes_readmission_cohort.csv",
        performance_path=PROJECT / "outputs" / "tables" / "mimic_diabetes_model_performance.csv",
        output_table=PROJECT / "outputs" / "tables" / "mimic_diabetes_leakage_sensitivity.csv",
        output_report=PROJECT / "outputs" / "reports" / "mimic_diabetes_leakage_sensitivity_report.md",
        valid_feature_set="lab_augmented",
        valid_scenario="valid_lab_augmented_logistic",
        report_title="MIMIC 糖尿病 Leakage Sensitivity Report",
        intro="这个报告演示为什么 ChronoEHR-Agent 必须做 leakage audit。",
        valid_note="合法模型：只用出院前可用特征。",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
