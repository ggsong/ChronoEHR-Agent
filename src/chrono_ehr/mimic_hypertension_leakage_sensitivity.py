#!/usr/bin/env python3
"""Run leakage sensitivity for the MIMIC hypertension demo."""

from __future__ import annotations

from mimic_diabetes_baseline import DEFAULT_PROJECT
from sensitivity_tools import run_leakage_sensitivity


def main() -> None:
    out = run_leakage_sensitivity(
        cohort_path=DEFAULT_PROJECT / "data" / "processed" / "mimic_hypertension_readmission_cohort.csv",
        performance_path=DEFAULT_PROJECT / "outputs" / "tables" / "mimic_hypertension_prediction_time_model_performance.csv",
        output_table=DEFAULT_PROJECT / "outputs" / "tables" / "mimic_hypertension_leakage_sensitivity.csv",
        output_report=DEFAULT_PROJECT / "outputs" / "reports" / "mimic_hypertension_leakage_sensitivity_report.md",
        valid_feature_set="discharge_lab_minimal",
        valid_scenario="valid_discharge_logistic",
        report_title="MIMIC Hypertension Leakage Sensitivity Report",
        intro="这个报告演示高血压再入院预测中未来信息泄漏如何抬高模型表现。",
        valid_note="合法模型：只使用出院前可用的住院过程变量和化验特征。",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
