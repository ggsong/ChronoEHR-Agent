#!/usr/bin/env python3
"""Run outcome-definition sensitivity for the MIMIC heart failure demo."""

from __future__ import annotations

from mimic_diabetes_baseline import DEFAULT_PROJECT
from sensitivity_tools import run_outcome_sensitivity


def main() -> None:
    out = run_outcome_sensitivity(
        cohort_path=DEFAULT_PROJECT / "data" / "processed" / "mimic_heart_failure_readmission_cohort.csv",
        output_table=DEFAULT_PROJECT / "outputs" / "tables" / "mimic_heart_failure_outcome_sensitivity.csv",
        output_type_table=DEFAULT_PROJECT / "outputs" / "tables" / "mimic_heart_failure_30d_readmission_next_type.csv",
        output_report=DEFAULT_PROJECT / "outputs" / "reports" / "mimic_heart_failure_outcome_sensitivity_report.md",
        report_title="MIMIC Heart Failure 30 天再入院结局敏感性分析",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
