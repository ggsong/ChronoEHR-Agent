#!/usr/bin/env python3
"""Run outcome-definition sensitivity for the MIMIC diabetes demo."""

from __future__ import annotations

from pathlib import Path

from sensitivity_tools import run_outcome_sensitivity


PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    out = run_outcome_sensitivity(
        cohort_path=PROJECT / "data" / "processed" / "mimic_diabetes_readmission_cohort.csv",
        output_table=PROJECT / "outputs" / "tables" / "mimic_diabetes_outcome_sensitivity.csv",
        output_type_table=PROJECT / "outputs" / "tables" / "mimic_diabetes_30d_readmission_next_type.csv",
        output_report=PROJECT / "outputs" / "reports" / "mimic_diabetes_outcome_sensitivity_report.md",
        report_title="MIMIC 糖尿病 30 天再入院结局敏感性分析",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
