#!/usr/bin/env python3
"""Audit whether ChronoEHR-Agent demo deliverables are ready to hand off."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_FILES = {
    "core_reports": [
        "outputs/reports/agent_progress_report.md",
        "outputs/reports/study_capability_audit.md",
        "outputs/reports/pipeline_step_introspection.md",
        "outputs/reports/agent_control_panel.md",
        "outputs/reports/agent_control_routing_validation.md",
        "outputs/reports/agent_entrypoints.md",
        "outputs/reports/mainline_mvp_validation.md",
        "outputs/reports/config_coverage_audit.md",
        "outputs/reports/config_migration_backlog.md",
        "outputs/reports/config_code_rules_validation.md",
        "outputs/reports/study_config_schema_validation.md",
        "outputs/reports/agent_task_plan.md",
        "outputs/reports/agent_task_execution_validation.md",
        "outputs/reports/agent_task_scenario_library.md",
        "outputs/reports/agent_task_scenario_library_validation.md",
        "outputs/reports/agent_task_router_validation.md",
        "outputs/reports/agent_action_catalog_validation.md",
        "outputs/reports/agent_entrypoints_validation.md",
        "outputs/reports/agent_command_lint.md",
        "outputs/reports/agent_control_consistency_audit.md",
        "outputs/reports/agent_dependency_audit.md",
        "outputs/reports/agent_doc_command_audit.md",
        "outputs/reports/agent_handoff_checklist.md",
        "outputs/reports/agent_boundary_audit.md",
        "outputs/reports/agent_artifact_freshness.md",
        "outputs/reports/agent_doctor.md",
        "outputs/reports/agent_doctor_validation.md",
        "outputs/reports/agent_progress_score.md",
        "outputs/reports/agent_progress_score_validation.md",
        "outputs/reports/agent_status_card.md",
        "outputs/reports/agent_status_card_validation.md",
        "outputs/reports/agent_demo_workflow_diabetes.md",
        "outputs/reports/agent_demo_workflow_validation.md",
        "outputs/reports/agent_self_check.md",
        "outputs/reports/agent_recovery_plan.md",
        "outputs/reports/agent_runbook.md",
        "outputs/reports/agent_runbook_validation.md",
        "outputs/reports/agent_runbook_confirmation_validation.md",
        "outputs/reports/agent_runbook_state_machine.md",
        "outputs/reports/agent_runbook_state_machine_validation.md",
        "outputs/reports/agent_runbook_retry_plan.md",
        "outputs/reports/agent_runbook_retry_plan_validation.md",
        "outputs/reports/agent_next_tasks.md",
        "outputs/reports/agent_next_tasks_validation.md",
        "outputs/reports/agent_task_queue.md",
        "outputs/reports/agent_task_queue_validation.md",
        "outputs/reports/agent_task_queue_execution.md",
        "outputs/reports/agent_task_queue_execution_validation.md",
        "outputs/reports/agent_cooldown_fingerprint_validation.md",
        "outputs/reports/agent_state_validation.md",
        "outputs/reports/report_preset_discovery.md",
        "outputs/reports/manuscript_asset_manifest.md",
        "outputs/reports/chronic_disease_methods_results_english_brief.md",
        "outputs/reports/english_brief_quality_audit.md",
        "outputs/reports/english_brief_docx_audit.md",
        "outputs/reports/chronic_disease_methods_results_draft.md",
        "outputs/reports/chronic_disease_supplementary_appendix.md",
        "outputs/reports/mimic_heart_failure_methods_results_draft.md",
        "outputs/reports/mimic_hypertension_methods_results_draft.md",
        "outputs/reports/prediction_time_leakage_gate_report.md",
        "outputs/reports/chronic_disease_ed_los_sensitivity_report.md",
        "outputs/reports/chronic_disease_threshold_analysis_report.md",
        "outputs/reports/chronic_disease_decision_curve_report.md",
        "outputs/reports/chronic_disease_subgroup_performance_report.md",
        "outputs/reports/chronic_disease_summary_figures_report.md",
    ],
    "word_package": [
        "outputs/reports/ChronoEHR_Methods_Results_Draft.docx",
        "outputs/reports/ChronoEHR_Supplementary_Appendix.docx",
        "outputs/reports/ChronoEHR_Methods_Results_Brief.docx",
        "outputs/reports/ChronoEHR_Supplementary_Brief.docx",
        "outputs/reports/ChronoEHR_Methods_Results_English_Brief.docx",
        "outputs/reports/rendered_methods_results/page-1.png",
        "outputs/reports/rendered_supplementary_appendix/page-1.png",
        "outputs/reports/rendered_methods_results_brief/page-1.png",
        "outputs/reports/rendered_supplementary_brief/page-1.png",
        "outputs/reports/rendered_english_brief/page-1.png",
        "outputs/reports/rendered_english_brief/page-2.png",
        "outputs/reports/rendered_english_brief/page-3.png",
        "outputs/reports/rendered_english_brief/ChronoEHR_Methods_Results_English_Brief.pdf",
    ],
    "key_tables": [
        "outputs/tables/prediction_time_leakage_gate_action_items.csv",
        "outputs/tables/chronic_disease_ed_los_sensitivity_comparison.csv",
        "outputs/tables/chronic_disease_threshold_analysis.csv",
        "outputs/tables/chronic_disease_decision_curve.csv",
        "outputs/tables/chronic_disease_subgroup_performance.csv",
        "outputs/tables/study_capability_audit.csv",
        "outputs/tables/study_capability_summary.csv",
        "outputs/tables/pipeline_step_introspection.csv",
        "outputs/tables/pipeline_step_summary.csv",
        "outputs/tables/agent_control_study_state.csv",
        "outputs/tables/agent_control_external_state.csv",
        "outputs/tables/agent_control_recommended_actions.csv",
        "outputs/tables/agent_control_routing_validation.csv",
        "outputs/tables/agent_entrypoints.csv",
        "outputs/tables/mainline_mvp_validation.csv",
        "outputs/tables/config_coverage_audit.csv",
        "outputs/tables/config_coverage_summary.csv",
        "outputs/tables/config_migration_backlog.csv",
        "outputs/tables/config_code_rules_validation.csv",
        "outputs/tables/study_config_schema_validation.csv",
        "outputs/tables/agent_task_plan.csv",
        "outputs/tables/agent_task_scenario.csv",
        "outputs/tables/agent_task_deferred_actions.csv",
        "outputs/tables/agent_task_execution.csv",
        "outputs/tables/agent_task_post_run_refresh.csv",
        "outputs/tables/agent_task_execution_validation.csv",
        "outputs/tables/agent_task_scenario_library.csv",
        "outputs/tables/agent_task_scenario_examples.csv",
        "outputs/tables/agent_task_scenario_library_validation.csv",
        "outputs/tables/agent_task_router_validation.csv",
        "outputs/tables/agent_action_catalog_validation.csv",
        "outputs/tables/agent_entrypoints_validation.csv",
        "outputs/tables/agent_command_lint.csv",
        "outputs/tables/agent_control_consistency_audit.csv",
        "outputs/tables/agent_dependency_audit.csv",
        "outputs/tables/agent_doc_command_audit.csv",
        "outputs/tables/agent_handoff_checklist.csv",
        "outputs/tables/agent_boundary_audit.csv",
        "outputs/tables/agent_artifact_freshness.csv",
        "outputs/tables/agent_doctor.csv",
        "outputs/tables/agent_doctor_validation.csv",
        "outputs/tables/agent_progress_score.csv",
        "outputs/tables/agent_progress_score_validation.csv",
        "outputs/tables/agent_status_card.csv",
        "outputs/tables/agent_status_card_validation.csv",
        "outputs/tables/agent_demo_workflow_diabetes.csv",
        "outputs/tables/agent_demo_workflow_validation.csv",
        "outputs/tables/agent_self_check.csv",
        "outputs/tables/agent_recovery_plan.csv",
        "outputs/tables/agent_recovery_execution.csv",
        "outputs/tables/agent_runbook.csv",
        "outputs/tables/agent_runbook_execution.csv",
        "outputs/tables/agent_runbook_phase_summary.csv",
        "outputs/tables/agent_runbook_validation.csv",
        "outputs/tables/agent_runbook_confirmation_validation.csv",
        "outputs/tables/agent_runbook_state_machine.csv",
        "outputs/tables/agent_runbook_state_machine_validation.csv",
        "outputs/tables/agent_runbook_retry_plan.csv",
        "outputs/tables/agent_runbook_retry_plan_validation.csv",
        "outputs/tables/agent_next_tasks.csv",
        "outputs/tables/agent_next_tasks_validation.csv",
        "outputs/tables/agent_task_queue.csv",
        "outputs/tables/agent_task_queue_validation.csv",
        "outputs/tables/agent_task_queue_execution.csv",
        "outputs/tables/agent_task_queue_execution_validation.csv",
        "outputs/tables/agent_cooldown_fingerprint_validation.csv",
        "outputs/tables/agent_state_validation.csv",
        "outputs/tables/report_preset_discovery.csv",
        "outputs/tables/manuscript_asset_manifest.csv",
        "outputs/tables/english_brief_key_results.csv",
        "outputs/tables/english_brief_quality_audit.csv",
        "outputs/tables/english_brief_docx_audit.csv",
        "outputs/tables/supplementary_appendix/table_s10_subgroup_performance.csv",
    ],
    "figures": [
        "outputs/figures/chronic_disease_decision_curve_net_benefit.png",
        "outputs/figures/chronic_disease_subgroup_mean_auroc.png",
        "outputs/figures/chronic_disease_subgroup_event_rate_top10_ppv.png",
    ],
    "configs": [
        "configs/manuscript_export_template.json",
        "configs/manuscript_export_brief.json",
        "configs/manuscript_export_english_brief.json",
        "configs/report_text_templates_english_brief.json",
        "configs/agent_action_catalog.json",
        "configs/agent_demo_workflows.json",
        "configs/agent_entrypoints.json",
        "configs/agent_cooldown_fingerprint.json",
        "docs/english_manuscript_preset_plan.md",
        "docs/mainline_mvp_definition.md",
        "docs/quickstart_usage.md",
        "configs/feature_window_specs.json",
        "configs/prediction_time_model_specs.json",
        "configs/study_registry.json",
    ],
    "agent_state": [
        "outputs/state/agent_state.json",
        "outputs/state/agent_state.md",
        "outputs/state/agent_task_history.csv",
        "outputs/state/agent_task_queue_execution_history.csv",
        "outputs/state/agent_demo_workflow_diabetes.json",
        "outputs/state/agent_runbook_state_machine.json",
        "outputs/state/agent_runbook_phase_history.csv",
    ],
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def status_row(category: str, check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {
        "category": category,
        "check": check,
        "status": status,
        "evidence": evidence,
        "detail": detail,
    }


def audit_required_files(project_root: Path) -> list[dict[str, str]]:
    rows = []
    for category, files in REQUIRED_FILES.items():
        for relative in files:
            path = project_root / relative
            if path.exists() and path.stat().st_size > 0:
                detail = f"exists; size={path.stat().st_size} bytes"
                status = "PASS"
            else:
                detail = "missing or empty"
                status = "FAIL"
            rows.append(status_row(category, relative, status, relative, detail))
    return rows


def audit_validation_reports(project_root: Path) -> list[dict[str, str]]:
    rows = []
    feature_report = read_text(project_root / "outputs/reports/feature_window_spec_validation_report.md")
    pred_report = read_text(project_root / "outputs/reports/prediction_time_spec_validation_report.md")
    leakage_report = read_text(project_root / "outputs/reports/prediction_time_leakage_gate_report.md")

    rows.append(
        status_row(
            "validation",
            "feature_window_specs",
            "PASS" if "Status: `PASS`" in feature_report and "Errors: 0" in feature_report and "Warnings: 0" in feature_report else "FAIL",
            "outputs/reports/feature_window_spec_validation_report.md",
            "requires PASS, Errors: 0, Warnings: 0",
        )
    )
    rows.append(
        status_row(
            "validation",
            "prediction_time_specs",
            "PASS" if "Status: `PASS`" in pred_report and "Issues: 0" in pred_report else "FAIL",
            "outputs/reports/prediction_time_spec_validation_report.md",
            "requires PASS and Issues: 0",
        )
    )
    rows.append(
        status_row(
            "validation",
            "leakage_gate_critical",
            "PASS" if "Status: `PASS`" in leakage_report and "Critical issues: 0" in leakage_report else "FAIL",
            "outputs/reports/prediction_time_leakage_gate_report.md",
            "requires PASS and Critical issues: 0",
        )
    )
    warning_match = re.search(r"Warnings:\s*(\d+)", leakage_report)
    warnings = int(warning_match.group(1)) if warning_match else -1
    rows.append(
        status_row(
            "validation",
            "leakage_gate_warnings_documented",
            "PASS" if warnings == 15 and "Action Items" in leakage_report and "ed_los_hours" in leakage_report else "FAIL",
            "outputs/reports/prediction_time_leakage_gate_report.md",
            f"warnings={warnings}; expected documented ed_los_hours action items",
        )
    )
    return rows


def audit_leakage_action_items(project_root: Path) -> list[dict[str, str]]:
    rows = []
    action_path = project_root / "outputs/tables/prediction_time_leakage_gate_action_items.csv"
    ed_path = project_root / "outputs/tables/chronic_disease_ed_los_sensitivity_comparison.csv"
    methods = read_text(project_root / "outputs/reports/chronic_disease_methods_results_draft.md")
    supplement = read_text(project_root / "outputs/reports/chronic_disease_supplementary_appendix.md")
    action_items = pd.read_csv(action_path) if action_path.exists() else pd.DataFrame()
    ed = pd.read_csv(ed_path) if ed_path.exists() else pd.DataFrame()
    rows.append(
        status_row(
            "leakage_action_items",
            "action_items_exist",
            "PASS" if len(action_items) >= 3 and action_items.astype(str).to_string().find("ed_los_hours") >= 0 else "FAIL",
            "outputs/tables/prediction_time_leakage_gate_action_items.csv",
            f"rows={len(action_items)}; expects ed_los_hours action items",
        )
    )
    mean_abs_auroc = float(ed["delta_AUROC"].abs().mean()) if not ed.empty and "delta_AUROC" in ed else float("nan")
    mean_abs_auprc = float(ed["delta_AUPRC"].abs().mean()) if not ed.empty and "delta_AUPRC" in ed else float("nan")
    sensitivity_pass = len(ed) == 17 and mean_abs_auroc < 0.005 and mean_abs_auprc < 0.005
    rows.append(
        status_row(
            "leakage_action_items",
            "ed_los_sensitivity_response",
            "PASS" if sensitivity_pass else "FAIL",
            "outputs/tables/chronic_disease_ed_los_sensitivity_comparison.csv",
            f"rows={len(ed)}; mean_abs_delta_AUROC={mean_abs_auroc:.4f}; mean_abs_delta_AUPRC={mean_abs_auprc:.4f}",
        )
    )
    rows.append(
        status_row(
            "leakage_action_items",
            "methods_and_appendix_document_response",
            "PASS" if "ED length-of-stay" in methods and "Table S7" in supplement else "FAIL",
            "outputs/reports/chronic_disease_methods_results_draft.md; outputs/reports/chronic_disease_supplementary_appendix.md",
            "requires ED LOS section in Methods/Results and S7 in appendix",
        )
    )
    return rows


def audit_docx_pages(project_root: Path) -> list[dict[str, str]]:
    expected = {
        "rendered_methods_results": 6,
        "rendered_supplementary_appendix": 11,
        "rendered_methods_results_brief": 5,
        "rendered_supplementary_brief": 8,
        "rendered_english_brief": 3,
    }
    rows = []
    for directory, expected_pages in expected.items():
        pages = sorted((project_root / "outputs/reports" / directory).glob("page-*.png"))
        status = "PASS" if len(pages) == expected_pages and all(page.stat().st_size > 0 for page in pages) else "FAIL"
        rows.append(
            status_row(
                "word_render",
                directory,
                status,
                f"outputs/reports/{directory}/",
                f"pages={len(pages)}; expected={expected_pages}",
            )
        )
    return rows


def audit_optional_external_benchmark(project_root: Path) -> list[dict[str, str]]:
    optional_files = [
        "docs/external_validation_plan.md",
        "docs/multidatabase_expansion_plan.md",
        "outputs/reports/next_study_action_plan.md",
        "outputs/reports/next_study_action_plan_validation.md",
        "outputs/reports/external_benchmark_readiness_summary.md",
        "outputs/reports/external_benchmark_summary_table.md",
        "outputs/reports/external_benchmark_summary_validation.md",
        "outputs/reports/external_technical_summary.md",
        "outputs/reports/external_technical_summary_validation.md",
        "outputs/reports/external_calibration_decision_summary.md",
        "outputs/reports/external_calibration_decision_summary_validation.md",
        "outputs/reports/external_model_selection_rationale.md",
        "outputs/reports/external_model_selection_rationale_validation.md",
        "outputs/reports/external_metric_consistency_audit.md",
        "outputs/reports/external_metric_consistency_audit_validation.md",
        "outputs/reports/external_summary_asset_manifest.md",
        "outputs/reports/external_summary_asset_manifest_validation.md",
        "outputs/reports/external_handoff_package.md",
        "outputs/reports/external_handoff_package_validation.md",
        "outputs/reports/external_model_bootstrap_ci.md",
        "outputs/reports/external_model_bootstrap_ci_validation.md",
        "outputs/reports/external_subgroup_performance.md",
        "outputs/reports/external_subgroup_performance_validation.md",
        "outputs/reports/external_subgroup_bootstrap_ci.md",
        "outputs/reports/external_subgroup_bootstrap_ci_validation.md",
        "outputs/reports/external_subgroup_robustness_summary.md",
        "outputs/reports/external_subgroup_robustness_summary_validation.md",
        "outputs/reports/external_threshold_band_sensitivity.md",
        "outputs/reports/external_threshold_band_sensitivity_validation.md",
        "outputs/reports/external_calibration_method_rationale.md",
        "outputs/reports/external_calibration_method_rationale_validation.md",
        "outputs/tables/next_study_action_plan.csv",
        "outputs/tables/next_study_action_plan_validation.csv",
        "outputs/tables/external_benchmark_readiness_summary.csv",
        "outputs/tables/external_benchmark_summary_table.csv",
        "outputs/tables/external_benchmark_hard_metrics_table.csv",
        "outputs/tables/external_benchmark_summary_validation.csv",
        "outputs/tables/external_technical_summary_table.csv",
        "outputs/tables/external_technical_summary_validation.csv",
        "outputs/tables/external_calibration_decision_summary.csv",
        "outputs/tables/external_calibration_decision_summary_validation.csv",
        "outputs/tables/external_model_selection_rationale.csv",
        "outputs/tables/external_model_selection_rationale_validation.csv",
        "outputs/tables/external_metric_consistency_audit.csv",
        "outputs/tables/external_metric_consistency_audit_validation.csv",
        "outputs/tables/external_summary_asset_manifest.csv",
        "outputs/tables/external_summary_asset_manifest_validation.csv",
        "outputs/tables/external_handoff_package_manifest.csv",
        "outputs/tables/external_handoff_package_validation.csv",
        "outputs/tables/external_model_bootstrap_ci.csv",
        "outputs/tables/external_model_bootstrap_ci_validation.csv",
        "outputs/tables/external_subgroup_performance.csv",
        "outputs/tables/external_subgroup_performance_validation.csv",
        "outputs/tables/external_subgroup_bootstrap_ci.csv",
        "outputs/tables/external_subgroup_bootstrap_ci_validation.csv",
        "outputs/tables/external_subgroup_robustness_summary.csv",
        "outputs/tables/external_subgroup_robustness_summary_validation.csv",
        "outputs/tables/external_threshold_band_sensitivity.csv",
        "outputs/tables/external_threshold_band_sensitivity_validation.csv",
        "outputs/tables/external_calibration_method_rationale.csv",
        "outputs/tables/external_calibration_method_rationale_validation.csv",
        "outputs/tables/external_model_comparison_recalibration_predictions.csv",
        "outputs/tables/external_model_comparison_recalibration_calibrators.csv",
        "outputs/tables/external_model_comparison_recalibration_metrics.csv",
        "outputs/tables/external_model_comparison_recalibration_deciles.csv",
        "outputs/tables/external_model_comparison_recalibration_summary.csv",
        "outputs/tables/external_model_comparison_recalibration_decision_curve.csv",
        "outputs/tables/external_model_comparison_recalibration_validation.csv",
        "outputs/reports/cdsl_external_validation_readiness_report.md",
        "outputs/reports/cdsl_temporal_benchmark_report.md",
        "outputs/reports/cdsl_leakage_audit_report.md",
        "outputs/reports/cdsl_traditional_baselines_report.md",
        "outputs/reports/cdsl_summary_figures_report.md",
        "outputs/reports/cdsl_calibration_decision_curve_report.md",
        "outputs/reports/cdsl_calibration_decision_curve_validation.md",
        "outputs/reports/external_model_comparison_recalibration.md",
        "outputs/reports/external_model_comparison_recalibration_validation.md",
        "outputs/reports/eicu_data_readiness_report.md",
        "outputs/reports/eicu_temporal_mortality_cohort_report.md",
        "outputs/reports/eicu_temporal_mortality_cohort_validation.md",
        "outputs/reports/eicu_temporal_features_24h_report.md",
        "outputs/reports/eicu_temporal_features_24h_validation.md",
        "outputs/reports/eicu_leakage_gate_report.md",
        "outputs/reports/eicu_first24h_logistic_baseline_report.md",
        "outputs/reports/eicu_first24h_logistic_baseline_validation.md",
        "outputs/reports/eicu_first24h_model_comparison_report.md",
        "outputs/reports/eicu_first24h_model_comparison_validation.md",
        "outputs/reports/eicu_baseline_figures_report.md",
        "outputs/reports/eicu_baseline_figures_validation.md",
        "outputs/reports/eicu_probability_recalibration_report.md",
        "outputs/reports/eicu_probability_recalibration_validation.md",
        "outputs/reports/charls_data_readiness_report.md",
        "outputs/reports/charls_wave_variable_map.md",
        "outputs/reports/charls_wave_variable_map_validation.md",
        "outputs/reports/charls_incident_diabetes_cohort_report.md",
        "outputs/reports/charls_incident_diabetes_cohort_validation.md",
        "outputs/reports/charls_baseline_features_report.md",
        "outputs/reports/charls_baseline_features_validation.md",
        "outputs/reports/charls_leakage_gate_report.md",
        "outputs/reports/charls_incident_diabetes_logistic_baseline_report.md",
        "outputs/reports/charls_incident_diabetes_logistic_baseline_validation.md",
        "outputs/reports/charls_incident_diabetes_sensitivity_report.md",
        "outputs/reports/charls_incident_diabetes_sensitivity_validation.md",
        "outputs/reports/charls_incident_diabetes_model_comparison_report.md",
        "outputs/reports/charls_incident_diabetes_model_comparison_validation.md",
        "outputs/reports/charls_calibration_decision_curve_report.md",
        "outputs/reports/charls_calibration_decision_curve_validation.md",
        "outputs/reports/charls_probability_recalibration_report.md",
        "outputs/reports/charls_probability_recalibration_validation.md",
        "outputs/reports/external_field_role_catalog.md",
        "outputs/reports/external_field_role_catalog_validation.md",
        "data/processed/eicu_temporal_mortality_cohort.csv",
        "data/processed/eicu_lab_features_24h.csv",
        "data/processed/eicu_vital_features_24h.csv",
        "data/processed/eicu_first24h_feature_matrix_skeleton.csv",
        "outputs/tables/cdsl_traditional_baselines_metrics.csv",
        "outputs/tables/cdsl_traditional_baselines_predictions.csv",
        "outputs/tables/cdsl_calibration_deciles.csv",
        "outputs/tables/cdsl_calibration_summary.csv",
        "outputs/tables/cdsl_decision_curve.csv",
        "outputs/tables/cdsl_calibration_decision_curve_validation.csv",
        "outputs/tables/cdsl_leakage_audit.csv",
        "outputs/tables/eicu_data_readiness_expected_tables.csv",
        "outputs/tables/eicu_temporal_mortality_cohort_summary.csv",
        "outputs/tables/eicu_temporal_mortality_cohort_exclusions.csv",
        "outputs/tables/eicu_temporal_mortality_split_summary.csv",
        "outputs/tables/eicu_temporal_mortality_cohort_validation.csv",
        "outputs/tables/eicu_temporal_features_24h_availability.csv",
        "outputs/tables/eicu_temporal_features_24h_extraction_stats.csv",
        "outputs/tables/eicu_temporal_features_24h_validation.csv",
        "outputs/tables/eicu_leakage_gate.csv",
        "outputs/tables/eicu_first24h_logistic_baseline_metrics.csv",
        "outputs/tables/eicu_first24h_logistic_baseline_predictions.csv",
        "outputs/tables/eicu_first24h_logistic_baseline_coefficients.csv",
        "outputs/tables/eicu_first24h_logistic_baseline_validation.csv",
        "outputs/tables/eicu_first24h_model_comparison_metrics.csv",
        "outputs/tables/eicu_first24h_model_comparison_predictions.csv",
        "outputs/tables/eicu_first24h_model_comparison_importances.csv",
        "outputs/tables/eicu_first24h_model_comparison_validation.csv",
        "outputs/tables/eicu_first24h_calibration_deciles.csv",
        "outputs/tables/eicu_first24h_calibration_summary.csv",
        "outputs/tables/eicu_baseline_figures_validation.csv",
        "outputs/tables/eicu_probability_recalibration_predictions.csv",
        "outputs/tables/eicu_probability_recalibration_metrics.csv",
        "outputs/tables/eicu_probability_recalibration_deciles.csv",
        "outputs/tables/eicu_probability_recalibration_summary.csv",
        "outputs/tables/eicu_probability_recalibration_decision_curve.csv",
        "outputs/tables/eicu_probability_recalibration_validation.csv",
        "outputs/tables/charls_wave_detection.csv",
        "outputs/tables/charls_wave_variable_map.csv",
        "outputs/tables/charls_harmonized_variable_inventory.csv",
        "outputs/tables/charls_wave_variable_map_validation.csv",
        "data/processed/charls_incident_diabetes_cohort.csv",
        "outputs/tables/charls_incident_diabetes_exclusions.csv",
        "outputs/tables/charls_incident_diabetes_cohort_summary.csv",
        "outputs/tables/charls_incident_diabetes_wave_outcome_summary.csv",
        "outputs/tables/charls_incident_diabetes_cohort_validation.csv",
        "data/processed/charls_incident_diabetes_baseline_features.csv",
        "outputs/tables/charls_baseline_feature_manifest.csv",
        "outputs/tables/charls_baseline_feature_missingness.csv",
        "outputs/tables/charls_baseline_feature_summary.csv",
        "outputs/tables/charls_baseline_features_validation.csv",
        "outputs/tables/charls_leakage_gate.csv",
        "outputs/tables/charls_incident_diabetes_logistic_baseline_metrics.csv",
        "outputs/tables/charls_incident_diabetes_logistic_baseline_predictions.csv",
        "outputs/tables/charls_incident_diabetes_logistic_baseline_coefficients.csv",
        "outputs/tables/charls_incident_diabetes_logistic_baseline_validation.csv",
        "outputs/tables/charls_incident_diabetes_sensitivity_metrics.csv",
        "outputs/tables/charls_incident_diabetes_sensitivity_predictions.csv",
        "outputs/tables/charls_incident_diabetes_sensitivity_coefficients.csv",
        "outputs/tables/charls_incident_diabetes_sensitivity_validation.csv",
        "outputs/tables/charls_incident_diabetes_model_comparison_metrics.csv",
        "outputs/tables/charls_incident_diabetes_model_comparison_predictions.csv",
        "outputs/tables/charls_incident_diabetes_model_comparison_importances.csv",
        "outputs/tables/charls_incident_diabetes_model_comparison_validation.csv",
        "outputs/tables/charls_calibration_deciles.csv",
        "outputs/tables/charls_calibration_summary.csv",
        "outputs/tables/charls_decision_curve.csv",
        "outputs/tables/charls_calibration_decision_curve_validation.csv",
        "outputs/tables/charls_probability_recalibration_predictions.csv",
        "outputs/tables/charls_probability_recalibration_metrics.csv",
        "outputs/tables/charls_probability_recalibration_deciles.csv",
        "outputs/tables/charls_probability_recalibration_summary.csv",
        "outputs/tables/charls_probability_recalibration_decision_curve.csv",
        "outputs/tables/charls_probability_recalibration_validation.csv",
        "outputs/tables/external_field_role_catalog.csv",
        "outputs/tables/external_field_role_summary.csv",
        "outputs/tables/external_field_role_catalog_validation.csv",
        "outputs/figures/cdsl_temporal_benchmark_auroc.png",
        "outputs/figures/cdsl_temporal_benchmark_auprc.png",
        "outputs/figures/eicu_first24h_logistic_roc.png",
        "outputs/figures/eicu_first24h_logistic_precision_recall.png",
        "outputs/figures/eicu_first24h_logistic_calibration_deciles.png",
        "outputs/tables/supplementary_appendix/table_s11_cdsl_external_benchmark.csv",
        "outputs/tables/supplementary_appendix/table_s12_cdsl_leakage_audit.csv",
        "outputs/tables/supplementary_appendix/table_s13_external_benchmark_summary.csv",
        "outputs/tables/supplementary_appendix/table_s14_external_benchmark_hard_metrics.csv",
        "outputs/tables/supplementary_appendix/table_s18_external_subgroup_robustness_summary.csv",
        "outputs/tables/supplementary_appendix/table_s19_external_threshold_band_sensitivity.csv",
        "outputs/tables/supplementary_appendix/table_s20_external_calibration_method_rationale.csv",
        "outputs/external_handoff_package.zip",
    ]
    existing = [relative for relative in optional_files if (project_root / relative).exists()]
    missing = [relative for relative in optional_files if relative not in existing]
    summary_path = project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv"
    readiness_detail = ""
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        readiness_detail = "; statuses=" + ", ".join(
            f"{row.dataset}:{row.local_status}" for row in summary[["dataset", "local_status"]].itertuples(index=False)
        )
    detail = f"optional external benchmark files present={len(existing)}/{len(optional_files)}{readiness_detail}"
    if missing:
        detail += "; missing optional files: " + ", ".join(missing[:4])
    return [
        status_row(
            "optional_external_benchmark",
            "external_benchmark_readiness_nonblocking",
            "PASS",
            "outputs/reports/external_benchmark_readiness_summary.md",
            detail,
        )
    ]


def audit(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    rows.extend(audit_required_files(project_root))
    rows.extend(audit_validation_reports(project_root))
    rows.extend(audit_leakage_action_items(project_root))
    rows.extend(audit_docx_pages(project_root))
    rows.extend(audit_optional_external_benchmark(project_root))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["category", "check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, output: Path) -> None:
    failures = summary[summary["status"].ne("PASS")]
    overall = "PASS" if failures.empty else "FAIL"
    by_status = summary["status"].value_counts().to_dict()
    text = f"""# ChronoEHR-Agent Delivery Readiness Audit

- Overall status: `{overall}`
- Checks: {len(summary)}
- PASS: {by_status.get("PASS", 0)}
- FAIL: {by_status.get("FAIL", 0)}

This audit checks whether the current local ChronoEHR-Agent demo has the expected leakage action-item response, ED LOS sensitivity analysis, validation reports, Word package, supplementary tables, configured exports, and summary figures. Optional CDSL external benchmark artifacts are recorded as non-blocking checks. It is a research-tool delivery audit, not a clinical recommendation.

## Check Table

{markdown_table(summary)}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "delivery_readiness_audit.csv"
    report_path = args.project_root / "outputs" / "reports" / "delivery_readiness_audit.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    write_report(summary, report_path)
    failures = int(summary["status"].ne("PASS").sum())
    print(f"Delivery readiness checks: {len(summary)}")
    print(f"Failures: {failures}")
    print(f"Wrote {report_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
