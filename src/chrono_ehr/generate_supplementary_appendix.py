#!/usr/bin/env python3
"""Generate a manuscript-ready supplementary appendix from ChronoEHR outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from generate_cross_cohort_methods_results import COHORT_LABELS, fmt_pct, label_cohort, markdown_table
from mimic_diabetes_baseline import DEFAULT_PROJECT


TABLES = {
    "cohort": "outputs/tables/chronic_disease_benchmark_cohort_summary.csv",
    "ablation": "outputs/tables/chronic_disease_feature_group_ablation_summary.csv",
    "repeated_concepts": "outputs/tables/chronic_disease_repeated_feature_concepts.csv",
    "selected": "outputs/tables/chronic_disease_selected_feature_set_supplementary_table.csv",
    "leakage_gate": "outputs/tables/prediction_time_leakage_gate.csv",
    "model": "outputs/tables/chronic_disease_manuscript_model_table.csv",
    "ed_los": "outputs/tables/chronic_disease_ed_los_sensitivity_comparison.csv",
    "threshold": "outputs/tables/chronic_disease_threshold_analysis.csv",
    "decision_curve": "outputs/tables/chronic_disease_decision_curve.csv",
    "subgroup": "outputs/tables/chronic_disease_subgroup_performance.csv",
    "cdsl_baselines": "outputs/tables/cdsl_traditional_baselines_metrics.csv",
    "cdsl_audit": "outputs/tables/cdsl_leakage_audit.csv",
}


def read_optional(project_root: Path, key: str) -> pd.DataFrame:
    path = project_root / TABLES[key]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["cohort_label"] = out["cohort"].map(label_cohort)
    out["readmission_30d_rate_pct"] = out["readmission_30d_rate"].map(fmt_pct)
    return out[
        [
            "cohort_label",
            "final_index_admissions",
            "final_subjects",
            "readmission_30d_count",
            "readmission_30d_rate_pct",
        ]
    ]


def format_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "stage",
        "group_added",
        "comparisons",
        "cohorts_improved_AUROC",
        "cohorts_improved_AUPRC",
        "mean_delta_AUROC",
        "mean_delta_AUPRC",
        "mean_delta_Brier",
    ]
    return df[cols].copy()


def format_repeated_concepts(df: pd.DataFrame, max_rows: int = 30) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "feature_group",
        "concept",
        "n_cohorts",
        "appearances",
        "prediction_times",
        "mean_abs_coefficient",
        "positive_count",
        "negative_count",
    ]
    return df[cols].head(max_rows).copy()


def format_selected_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "cohort_label",
        "prediction_time",
        "selected_features",
        "full_AUROC",
        "selected_AUROC",
        "delta_AUROC",
        "full_AUPRC",
        "selected_AUPRC",
        "delta_AUPRC",
        "full_Brier",
        "selected_Brier",
        "delta_Brier",
        "mean_absolute_calibration_error",
        "max_absolute_calibration_error",
    ]
    return df[cols].copy()


def format_leakage_gate_warnings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    warnings = df[df["severity"].eq("warning")].copy()
    if warnings.empty:
        return warnings
    grouped = (
        warnings.groupby(["study", "prediction_time", "check_type", "status", "object", "reason"], dropna=False)
        .agg(
            n_feature_sets=("feature_set", "nunique"),
            feature_sets=("feature_set", lambda values: ", ".join(sorted(set(map(str, values)))[:8])),
        )
        .reset_index()
    )
    grouped["cohort_label"] = grouped["study"].map(lambda value: COHORT_LABELS.get(str(value), str(value)))
    return grouped[
        [
            "cohort_label",
            "prediction_time",
            "check_type",
            "status",
            "object",
            "n_feature_sets",
            "feature_sets",
            "reason",
        ]
    ]


def format_ed_los_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "cohort_label",
        "source_feature_set",
        "n",
        "events",
        "original_AUROC",
        "no_ed_los_AUROC",
        "delta_AUROC",
        "original_AUPRC",
        "no_ed_los_AUPRC",
        "delta_AUPRC",
        "delta_Brier",
    ]
    return df[cols].copy()


def format_threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "cohort_label",
        "prediction_time",
        "alert_rate",
        "alerts",
        "risk_threshold",
        "ppv",
        "recall",
        "specificity",
        "npv",
        "lift_vs_event_rate",
    ]
    return df[cols].copy()


def format_decision_curve_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    focus = df[df["threshold_probability"].isin([0.10, 0.20, 0.30])].copy()
    cols = [
        "cohort_label",
        "prediction_time",
        "threshold_probability",
        "alerts",
        "alert_rate",
        "ppv",
        "recall",
        "model_net_benefit",
        "treat_all_net_benefit",
        "net_benefit_advantage",
        "preferred_strategy",
    ]
    return focus[cols].copy()


def format_subgroup_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "cohort_label",
        "prediction_time",
        "subgroup_variable",
        "subgroup_value",
        "n",
        "events",
        "event_rate",
        "AUROC",
        "AUPRC",
        "Brier_score",
        "top10_ppv",
        "top10_recall",
    ]
    return df[cols].copy()


def format_cdsl_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    test = df[df["split"].eq("test")].copy()
    if test.empty:
        return test
    test["benchmark_role"] = test["feature_set"].map(
        {
            "admission_demographics": "admission-time baseline",
            "first_24h_vitals_labs": "early in-hospital 24h window",
            "first_48h_vitals_labs": "early in-hospital 48h window",
            "full_stay_naive_reference": "full-stay reference; not early prediction",
        }
    )
    cols = [
        "feature_set",
        "benchmark_role",
        "model",
        "n",
        "events",
        "event_rate",
        "AUROC",
        "AUPRC",
        "Brier",
        "feature_count",
    ]
    return test[cols].sort_values(["feature_set", "model"]).copy()


def format_cdsl_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "check",
        "status",
        "severity",
        "evidence",
        "beginner_interpretation",
        "recommended_action",
    ]
    return df[cols].copy()


def write_csv_exports(project_root: Path, tables: dict[str, pd.DataFrame]) -> None:
    out_dir = project_root / "outputs" / "tables" / "supplementary_appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def write_report(project_root: Path, output: Path) -> None:
    cohort = format_cohort_table(read_optional(project_root, "cohort"))
    ablation = format_ablation_table(read_optional(project_root, "ablation"))
    repeated = format_repeated_concepts(read_optional(project_root, "repeated_concepts"))
    selected = format_selected_table(read_optional(project_root, "selected"))
    leakage = format_leakage_gate_warnings(read_optional(project_root, "leakage_gate"))
    ed_los = format_ed_los_table(read_optional(project_root, "ed_los"))
    threshold = format_threshold_table(read_optional(project_root, "threshold"))
    decision_curve = format_decision_curve_table(read_optional(project_root, "decision_curve"))
    subgroup = format_subgroup_table(read_optional(project_root, "subgroup"))
    cdsl_baselines = format_cdsl_baseline_table(read_optional(project_root, "cdsl_baselines"))
    cdsl_audit = format_cdsl_audit_table(read_optional(project_root, "cdsl_audit"))
    model = read_optional(project_root, "model")

    exports = {
        "table_s1_cohort_summary": cohort,
        "table_s2_model_baselines": model,
        "table_s3_feature_group_ablation": ablation,
        "table_s4_repeated_feature_concepts": repeated,
        "table_s5_selected_feature_set_comparison": selected,
        "table_s6_leakage_gate_warnings": leakage,
        "table_s7_ed_los_sensitivity": ed_los,
        "table_s8_threshold_analysis": threshold,
        "table_s9_decision_curve": decision_curve,
        "table_s10_subgroup_performance": subgroup,
        "table_s11_cdsl_external_benchmark": cdsl_baselines,
        "table_s12_cdsl_leakage_audit": cdsl_audit,
    }
    write_csv_exports(project_root, exports)

    text = f"""# Supplementary Appendix

本补充材料由 ChronoEHR-Agent 自动生成，用于支持慢病 EHR time-aware prediction benchmark。它只总结本地 EHR 数据研究结果，不提供医学诊疗建议。

## Table S1. Cohort Summary

{markdown_table(cohort, list(cohort.columns), display_names={
    "cohort_label": "Cohort",
    "final_index_admissions": "Index admissions",
    "final_subjects": "Patients",
    "readmission_30d_count": "30-day readmissions",
    "readmission_30d_rate_pct": "30-day readmission rate",
})}

## Table S2. Model Baseline Summary

{markdown_table(model, list(model.columns))}

## Table S3. Feature Group Ablation

{markdown_table(ablation, list(ablation.columns))}

## Table S4. Repeated Feature Concepts

{markdown_table(repeated, list(repeated.columns))}

## Table S5. Full Vs Selected Feature Sets

{markdown_table(selected, list(selected.columns))}

## Table S6. Leakage Gate Warnings

{markdown_table(leakage, list(leakage.columns))}

## Table S7. ED Length-of-Stay Sensitivity

{markdown_table(ed_los, list(ed_los.columns))}

## Table S8. Threshold And Alert-Burden Analysis

{markdown_table(threshold, list(threshold.columns))}

## Table S9. Decision-Curve Net Benefit

{markdown_table(decision_curve, list(decision_curve.columns))}

## Table S10. Subgroup Performance

{markdown_table(subgroup, list(subgroup.columns))}

## Table S11. CDSL External Temporal Benchmark

CDSL is included as an external EHR time-series benchmark for method validation. It is not a direct external validation cohort for chronic-disease 30-day readmission because its task is COVID hospitalization mortality/LOS.

{markdown_table(cdsl_baselines, list(cdsl_baselines.columns))}

## Table S12. CDSL Leakage Audit

This audit summarizes whether the CDSL temporal benchmark accidentally used labels, LOS, discharge time, or records outside the intended prediction window as model features.

{markdown_table(cdsl_audit, list(cdsl_audit.columns))}

## Notes For Manuscript Use

- Table S1 supports cohort description.
- Table S2 supports traditional baseline comparison.
- Table S3 supports the claim that labs and broad medications are more stable feature groups than ICU vitals/procedures.
- Table S4 supports selected feature set construction.
- Table S5 supports the sensitivity analysis showing selected feature sets retain most performance with fewer variables.
- Table S6 documents non-critical leakage-gate warnings; current warnings mainly involve conditional availability of `ed_los_hours` for 24h prediction.
- Table S7 supports the sensitivity analysis showing 24h model performance is not materially dependent on `ed_los_hours`.
- Table S8 supports fixed alert-burden interpretation for final 24h and discharge models.
- Table S9 supports net-benefit interpretation across common threshold probabilities.
- Table S10 supports subgroup performance summaries and future heterogeneity/fairness checks.
- Table S11 supports external time-aware EHR method validation on CDSL.
- Table S12 documents CDSL-specific leakage warnings and boundary records.

## CSV Exports

The individual supplementary tables are also exported under:

`outputs/tables/supplementary_appendix/`
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.project_root / "outputs" / "reports" / "chronic_disease_supplementary_appendix.md"
    write_report(args.project_root, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
