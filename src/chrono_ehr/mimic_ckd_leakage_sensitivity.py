#!/usr/bin/env python3
"""Run leakage sensitivity for the MIMIC CKD demo."""

from __future__ import annotations

from mimic_diabetes_baseline import DEFAULT_PROJECT
from sensitivity_tools import run_leakage_sensitivity


def main() -> None:
    out = run_leakage_sensitivity(
        cohort_path=DEFAULT_PROJECT / "data" / "processed" / "mimic_ckd_readmission_cohort.csv",
        performance_path=DEFAULT_PROJECT / "outputs" / "tables" / "mimic_ckd_prediction_time_model_performance.csv",
        output_table=DEFAULT_PROJECT / "outputs" / "tables" / "mimic_ckd_leakage_sensitivity.csv",
        output_report=DEFAULT_PROJECT / "outputs" / "reports" / "mimic_ckd_leakage_sensitivity_report.md",
        valid_feature_set="discharge_lab_minimal",
        valid_scenario="valid_discharge_lab_logistic",
        report_title="MIMIC CKD Leakage Sensitivity Report",
        intro="这个报告演示为什么 ChronoEHR-Agent 必须做 leakage audit。CKD 合法模型只使用出院前可用信息；错误示范则故意使用随访后的未来信息。",
        valid_note="合法模型：只使用出院前可用特征和出院前化验。",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
