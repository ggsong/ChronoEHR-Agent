#!/usr/bin/env python3
"""Audit study-level capabilities for ChronoEHR-Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


DEFAULT_REGISTRY = DEFAULT_PROJECT / "configs" / "study_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def exists(project_root: Path, relative: str) -> bool:
    path = project_root / relative
    return path.exists() and path.stat().st_size > 0


def cohort_prefix(study: dict[str, Any]) -> str:
    cohort = study.get("cohort", "")
    mapping = {
        "diabetes": "mimic_diabetes",
        "ckd": "mimic_ckd",
        "heart_failure": "mimic_heart_failure",
        "hypertension": "mimic_hypertension",
    }
    return mapping.get(cohort, cohort)


def capability_row(study_id: str, capability: str, status: str, evidence: str, note: str) -> dict[str, str]:
    return {
        "study_id": study_id,
        "capability": capability,
        "status": status,
        "evidence": evidence,
        "note": note,
    }


def mimic_capabilities(project_root: Path, study: dict[str, Any]) -> list[dict[str, str]]:
    study_id = study.get("id", "")
    prefix = cohort_prefix(study)
    checks = [
        ("config", study.get("config", ""), "study YAML config exists"),
        ("runner", study.get("pipeline", ""), "study runner exists"),
        ("cohort_summary", f"outputs/tables/{prefix}_cohort_summary.csv", "cohort can be counted"),
        ("table1", f"outputs/tables/{prefix}_table1_basic.csv", "basic descriptive Table 1 exists"),
        ("split_summary", f"outputs/tables/{prefix}_split_summary.csv", "train/validation/test split summary exists"),
        ("feature_missingness", f"outputs/tables/{prefix}_feature_missingness.csv", "feature missingness table exists"),
        ("feature_time_audit", f"outputs/reports/{prefix}_feature_time_audit_report.md", "time availability audit exists"),
        ("leakage_audit", f"outputs/reports/{prefix}_leakage_audit_report.md", "leakage audit report exists"),
        ("leakage_sensitivity", f"outputs/tables/{prefix}_leakage_sensitivity.csv", "leakage sensitivity table exists"),
        ("outcome_sensitivity", f"outputs/tables/{prefix}_outcome_sensitivity.csv", "outcome sensitivity table exists"),
        ("prediction_time_models", f"outputs/tables/{prefix}_prediction_time_model_performance.csv", "prediction-time model table exists"),
        ("prediction_time_figures", f"outputs/reports/{prefix}_prediction_time_figure_report.md", "prediction-time figure report exists"),
        ("test_predictions", f"outputs/tables/{prefix}_test_predictions.csv", "test-set predictions exist"),
        ("methods_results_draft", f"outputs/reports/{prefix}_methods_results_draft.md", "single-study draft exists"),
    ]
    rows = []
    for capability, evidence, note in checks:
        status = "PASS" if evidence and exists(project_root, evidence) else "MISSING"
        rows.append(capability_row(study_id, capability, status, evidence, note))

    lab_evidence = f"outputs/tables/{prefix}_lab_feature_availability.csv"
    rows.append(
        capability_row(
            study_id,
            "lab_features",
            "PASS" if exists(project_root, lab_evidence) else "MISSING",
            lab_evidence,
            "lab feature availability table exists",
        )
    )
    return rows


def shared_capabilities(project_root: Path) -> list[dict[str, str]]:
    checks = [
        ("cross_cohort_benchmark", "outputs/tables/chronic_disease_prediction_time_benchmark.csv"),
        ("traditional_baselines", "outputs/tables/chronic_disease_model_baseline_comparison.csv"),
        ("calibration", "outputs/tables/chronic_disease_model_calibration_summary.csv"),
        ("feature_group_ablation", "outputs/tables/chronic_disease_feature_group_ablation_summary.csv"),
        ("selected_feature_sets", "outputs/tables/chronic_disease_selected_feature_set_comparison.csv"),
        ("ed_los_sensitivity", "outputs/tables/chronic_disease_ed_los_sensitivity_comparison.csv"),
        ("threshold_analysis", "outputs/tables/chronic_disease_threshold_analysis.csv"),
        ("decision_curve", "outputs/tables/chronic_disease_decision_curve.csv"),
        ("subgroup_analysis", "outputs/tables/chronic_disease_subgroup_performance.csv"),
        ("external_bootstrap_ci", "outputs/tables/external_model_bootstrap_ci.csv"),
        ("external_calibration_decision", "outputs/tables/cdsl_calibration_decision_curve_validation.csv"),
        ("external_subgroup_performance", "outputs/tables/external_subgroup_performance_validation.csv"),
        ("external_model_recalibration", "outputs/tables/external_model_comparison_recalibration_validation.csv"),
        ("summary_figures", "outputs/reports/chronic_disease_summary_figures_report.md"),
        ("manuscript_draft", "outputs/reports/chronic_disease_methods_results_draft.md"),
        ("supplementary_appendix", "outputs/reports/chronic_disease_supplementary_appendix.md"),
        ("word_package", "outputs/reports/ChronoEHR_Methods_Results_Draft.docx"),
    ]
    rows = []
    for capability, evidence in checks:
        rows.append(
            capability_row(
                "cross_cohort",
                capability,
                "PASS" if exists(project_root, evidence) else "MISSING",
                evidence,
                "shared manuscript or benchmark capability",
            )
        )
    return rows


def external_status_by_dataset(project_root: Path) -> dict[str, str]:
    summary = read_csv(project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv")
    if summary.empty or not {"dataset", "local_status"}.issubset(summary.columns):
        return {}
    return dict(zip(summary["dataset"].astype(str), summary["local_status"].astype(str)))


def planned_capabilities(project_root: Path, study: dict[str, Any], external_statuses: dict[str, str]) -> list[dict[str, str]]:
    study_id = study.get("id", "")
    checks = [
        ("config_template", study.get("config", ""), "planned-study config template exists"),
        ("protocol", study.get("protocol", ""), "planned-study protocol exists"),
        ("feature_time_map", study.get("feature_time_map", ""), "planned-study time map exists"),
    ]
    if study_id == "eicu_temporal_mortality":
        checks.extend(
            [
                ("readiness_script", "src/chrono_ehr/eicu_data_readiness.py", "readiness script exists"),
                ("readiness_report", "outputs/reports/eicu_data_readiness_report.md", "readiness report exists"),
                ("schema_mapping", "outputs/tables/eicu_schema_mapping_draft.csv", "schema mapping draft exists"),
                ("cohort", "data/processed/eicu_temporal_mortality_cohort.csv", "first-24h mortality cohort exists"),
                ("cohort_validation", "outputs/tables/eicu_temporal_mortality_cohort_validation.csv", "cohort validation exists"),
                ("temporal_features", "data/processed/eicu_first24h_feature_matrix_skeleton.csv", "first-24h lab/vital feature matrix exists"),
                ("temporal_feature_validation", "outputs/tables/eicu_temporal_features_24h_validation.csv", "temporal feature validation exists"),
                ("leakage_gate", "outputs/tables/eicu_leakage_gate.csv", "eICU leakage gate exists"),
                ("logistic_baseline", "outputs/tables/eicu_first24h_logistic_baseline_metrics.csv", "first-24h logistic baseline metrics exist"),
                ("logistic_baseline_validation", "outputs/tables/eicu_first24h_logistic_baseline_validation.csv", "baseline validation exists"),
                ("model_comparison", "outputs/tables/eicu_first24h_model_comparison_metrics.csv", "eICU first-24h model-comparison metrics exist"),
                ("model_comparison_validation", "outputs/tables/eicu_first24h_model_comparison_validation.csv", "eICU model-comparison validation exists"),
                ("baseline_figures", "outputs/reports/eicu_baseline_figures_report.md", "ROC/PR/calibration figure report exists"),
                ("baseline_figures_validation", "outputs/tables/eicu_baseline_figures_validation.csv", "baseline figure validation exists"),
                ("probability_recalibration", "outputs/tables/eicu_probability_recalibration_metrics.csv", "eICU probability recalibration metrics exist"),
                ("probability_recalibration_validation", "outputs/tables/eicu_probability_recalibration_validation.csv", "eICU probability recalibration validation exists"),
                ("external_benchmark_summary", "outputs/tables/external_benchmark_summary_table.csv", "CDSL/eICU external benchmark summary exists"),
            ]
        )
    elif study_id == "charls_incident_diabetes":
        checks.extend(
            [
                ("readiness_script", "src/chrono_ehr/charls_data_readiness.py", "readiness script exists"),
                ("readiness_report", "outputs/reports/charls_data_readiness_report.md", "readiness report exists"),
                ("schema_mapping", "outputs/tables/charls_schema_mapping_draft.csv", "schema mapping draft exists"),
                ("wave_variable_map", "outputs/tables/charls_wave_variable_map.csv", "concrete harmonized wave-variable map exists"),
                ("wave_variable_map_validation", "outputs/tables/charls_wave_variable_map_validation.csv", "wave-variable map validation exists"),
                ("incident_diabetes_cohort", "data/processed/charls_incident_diabetes_cohort.csv", "2011 baseline to 2013/2015 incident diabetes cohort exists"),
                ("incident_diabetes_cohort_validation", "outputs/tables/charls_incident_diabetes_cohort_validation.csv", "incident diabetes cohort validation exists"),
                ("baseline_features", "data/processed/charls_incident_diabetes_baseline_features.csv", "2011 baseline feature matrix exists"),
                ("baseline_feature_validation", "outputs/tables/charls_baseline_features_validation.csv", "baseline feature matrix validation exists"),
                ("leakage_gate", "outputs/tables/charls_leakage_gate.csv", "CHARLS leakage gate exists"),
                ("logistic_baseline", "outputs/tables/charls_incident_diabetes_logistic_baseline_metrics.csv", "CHARLS incident diabetes logistic baseline metrics exist"),
                ("logistic_baseline_validation", "outputs/tables/charls_incident_diabetes_logistic_baseline_validation.csv", "CHARLS logistic baseline validation exists"),
                ("sensitivity_analysis", "outputs/tables/charls_incident_diabetes_sensitivity_metrics.csv", "CHARLS sensitivity analysis metrics exist"),
                ("sensitivity_validation", "outputs/tables/charls_incident_diabetes_sensitivity_validation.csv", "CHARLS sensitivity validation exists"),
                ("model_comparison", "outputs/tables/charls_incident_diabetes_model_comparison_metrics.csv", "CHARLS traditional model-comparison metrics exist"),
                ("model_comparison_validation", "outputs/tables/charls_incident_diabetes_model_comparison_validation.csv", "CHARLS model-comparison validation exists"),
                ("calibration_decision", "outputs/tables/charls_calibration_summary.csv", "CHARLS calibration and decision-curve summary exists"),
                ("calibration_decision_validation", "outputs/tables/charls_calibration_decision_curve_validation.csv", "CHARLS calibration/decision validation exists"),
                ("probability_recalibration", "outputs/tables/charls_probability_recalibration_metrics.csv", "CHARLS probability recalibration metrics exist"),
                ("probability_recalibration_validation", "outputs/tables/charls_probability_recalibration_validation.csv", "CHARLS probability recalibration validation exists"),
            ]
        )
    rows = []
    for capability, evidence, note in checks:
        status = "PASS" if evidence and exists(project_root, evidence) else "MISSING"
        rows.append(capability_row(study_id, capability, status, evidence, note))

    external_dataset = "eICU" if study_id == "eicu_temporal_mortality" else "CHARLS" if study_id == "charls_incident_diabetes" else ""
    external_status = external_statuses.get(external_dataset, "")
    ready_statuses = {"READY", "READY_FOR_COHORT_CODE", "COHORT_READY", "FEATURE_READY", "BASELINE_READY", "READY_FOR_PROTOCOL_CODE"}
    raw_status = "PASS" if external_status in ready_statuses else "DATA_PENDING"
    raw_note = (
        f"external readiness status is {external_status or 'UNKNOWN'}; rerun readiness after files change"
        if raw_status == "PASS"
        else "raw data must be confirmed by readiness script before cohort/model code"
    )
    rows.append(
        capability_row(
            study_id,
            "raw_data_ready",
            raw_status,
            study.get("dataset_root", ""),
            raw_note,
        )
    )
    return rows


def summarize(rows: list[dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("study_id")
        .agg(
            capabilities=("capability", "count"),
            pass_count=("status", lambda values: int((values == "PASS").sum())),
            missing_count=("status", lambda values: int((values == "MISSING").sum())),
            data_pending_count=("status", lambda values: int((values == "DATA_PENDING").sum())),
        )
        .reset_index()
    )
    grouped["completion_percent"] = (grouped["pass_count"] / grouped["capabilities"] * 100).round(1)
    grouped["overall_status"] = grouped.apply(overall_status, axis=1)
    return grouped.sort_values(["overall_status", "completion_percent", "study_id"], ascending=[True, False, True])


def overall_status(row: pd.Series) -> str:
    if row["data_pending_count"] > 0:
        return "DATA_PENDING"
    if row["missing_count"] == 0:
        return "COMPLETE"
    if row["pass_count"] >= row["capabilities"] * 0.7:
        return "USABLE_WITH_GAPS"
    return "INCOMPLETE"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.astype(object).where(pd.notna(df), "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, details: pd.DataFrame, summary: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "study_capability_audit.md"
    incomplete = details[details["status"].ne("PASS")]
    text = f"""# Study Capability Audit

这个报告按 study 检查 ChronoEHR-Agent 是否已经具备研究执行能力。它关注的是本地研究流程：cohort、Table 1、时间点特征、leakage audit、prediction-time model、sensitivity analysis、报告草稿和外部扩展准备；不是医学问答，也不是诊疗建议。

## Study Summary

{markdown_table(summary)}

## Non-PASS Items

{markdown_table(incomplete) if not incomplete.empty else "All checked capabilities are PASS."}

## Interpretation

- MIMIC-IV 糖尿病是主 demo，应该作为当前最稳的展示和写作骨架。
- CKD、心衰和高血压已经适合作为 cross-cohort replication，用来证明 Agent 不是只服务一个疾病。
- eICU 已经按外部 ICU benchmark 路线推进到 first-24h cohort、temporal features、leakage gate、logistic baseline、figures/calibration 和 probability recalibration；它不是慢病再入院外部验证。
- CHARLS 已从 protocol/config/readiness 推进到 harmonized wave-variable map、2011 baseline -> 2013/2015 incident diabetes cohort skeleton、baseline feature matrix、leakage gate、logistic baseline、sensitivity analysis、calibration、decision curve 和 probability recalibration。
- `cross_cohort` 行代表跨队列共同能力，例如传统 baseline、calibration、decision curve、subgroup analysis、Word package。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    external_statuses = external_status_by_dataset(args.project_root)
    rows: list[dict[str, str]] = []
    for study in registry.get("studies", []):
        rows.extend(mimic_capabilities(args.project_root, study))
    rows.extend(shared_capabilities(args.project_root))
    for study in registry.get("planned_studies", []):
        rows.extend(planned_capabilities(args.project_root, study, external_statuses))

    details = pd.DataFrame(rows)
    summary = summarize(rows)
    detail_path = args.project_root / "outputs" / "tables" / "study_capability_audit.csv"
    summary_path = args.project_root / "outputs" / "tables" / "study_capability_summary.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = write_report(args.project_root, details, summary)
    print(f"Wrote {report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
