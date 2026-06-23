#!/usr/bin/env python3
"""Audit freshness of core ChronoEHR-Agent control artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


FRESHNESS_RULES = [
    {
        "artifact_id": "agent_entrypoints",
        "inputs": ["src/chrono_ehr/generate_agent_entrypoints.py", "configs/agent_entrypoints.json"],
        "outputs": ["outputs/reports/agent_entrypoints.md", "outputs/tables/agent_entrypoints.csv"],
    },
    {
        "artifact_id": "agent_entrypoints_validation",
        "inputs": [
            "src/chrono_ehr/validate_agent_entrypoints.py",
            "src/chrono_ehr/run_study.py",
            "configs/agent_entrypoints.json",
            "outputs/reports/agent_entrypoints.md",
            "outputs/tables/agent_entrypoints.csv",
        ],
        "outputs": ["outputs/reports/agent_entrypoints_validation.md", "outputs/tables/agent_entrypoints_validation.csv"],
    },
    {
        "artifact_id": "agent_action_catalog_validation",
        "inputs": ["src/chrono_ehr/validate_agent_action_catalog.py", "configs/agent_action_catalog.json"],
        "outputs": ["outputs/reports/agent_action_catalog_validation.md", "outputs/tables/agent_action_catalog_validation.csv"],
    },
    {
        "artifact_id": "agent_command_lint",
        "inputs": [
            "src/chrono_ehr/agent_command_linter.py",
            "src/chrono_ehr/run_study.py",
            "src/chrono_ehr/agent_self_check.py",
            "configs/agent_entrypoints.json",
            "configs/agent_action_catalog.json",
            "configs/agent_demo_workflows.json",
            "outputs/state/agent_state.json",
            "outputs/tables/next_study_action_plan.csv",
        ],
        "outputs": ["outputs/reports/agent_command_lint.md", "outputs/tables/agent_command_lint.csv"],
    },
    {
        "artifact_id": "agent_boundary_audit",
        "inputs": ["src/chrono_ehr/agent_boundary_audit.py", "docs", "configs", "outputs/reports/agent_progress_report.md"],
        "outputs": ["outputs/reports/agent_boundary_audit.md", "outputs/tables/agent_boundary_audit.csv"],
    },
    {
        "artifact_id": "agent_state",
        "inputs": [
            "src/chrono_ehr/build_agent_state.py",
            "src/chrono_ehr/agent_cooldown_fingerprint.py",
            "configs/agent_cooldown_fingerprint.json",
            "outputs/tables/study_capability_summary.csv",
            "outputs/tables/external_benchmark_readiness_summary.csv",
            "outputs/tables/agent_next_tasks.csv",
            "outputs/state/agent_task_queue_execution_history.csv",
            "outputs/state/agent_runbook_state_machine.json",
        ],
        "outputs": ["outputs/state/agent_state.json", "outputs/state/agent_state.md"],
    },
    {
        "artifact_id": "agent_control_consistency_audit",
        "inputs": [
            "src/chrono_ehr/agent_control_consistency_audit.py",
            "src/chrono_ehr/agent_control_panel.py",
            "src/chrono_ehr/agent_self_check.py",
            "src/chrono_ehr/run_study.py",
            "src/chrono_ehr/delivery_readiness_audit.py",
            "src/chrono_ehr/validate_mainline_mvp.py",
            "configs/agent_entrypoints.json",
            "configs/agent_action_catalog.json",
            "docs/quickstart_usage.md",
            "outputs/tables/external_benchmark_readiness_summary.csv",
            "outputs/tables/study_capability_summary.csv",
            "outputs/tables/agent_control_external_state.csv",
        ],
        "outputs": [
            "outputs/reports/agent_control_consistency_audit.md",
            "outputs/tables/agent_control_consistency_audit.csv",
        ],
    },
    {
        "artifact_id": "agent_dependency_audit",
        "inputs": [
            "src/chrono_ehr/agent_dependency_audit.py",
            "src/chrono_ehr/agent_self_check.py",
            "src/chrono_ehr/agent_doctor.py",
            "src/chrono_ehr/agent_artifact_freshness.py",
            "src/chrono_ehr/run_study.py",
        ],
        "outputs": ["outputs/reports/agent_dependency_audit.md", "outputs/tables/agent_dependency_audit.csv"],
    },
    {
        "artifact_id": "agent_doc_command_audit",
        "inputs": [
            "src/chrono_ehr/agent_doc_command_audit.py",
            "src/chrono_ehr/run_study.py",
            "README.md",
            "docs/quickstart_usage.md",
        ],
        "outputs": ["outputs/reports/agent_doc_command_audit.md", "outputs/tables/agent_doc_command_audit.csv"],
    },
    {
        "artifact_id": "agent_handoff_checklist",
        "inputs": [
            "src/chrono_ehr/agent_handoff_checklist.py",
            "src/chrono_ehr/run_study.py",
            "README.md",
            "docs/quickstart_usage.md",
            "docs/resume_state.md",
            "outputs/reports/agent_progress_score.md",
            "outputs/tables/agent_progress_score.csv",
            "outputs/reports/agent_next_tasks.md",
            "outputs/tables/agent_next_tasks.csv",
            "outputs/state/agent_state.json",
            "outputs/state/agent_state.md",
        ],
        "outputs": ["outputs/reports/agent_handoff_checklist.md", "outputs/tables/agent_handoff_checklist.csv"],
    },
    {
        "artifact_id": "agent_task_execution_validation",
        "inputs": [
            "src/chrono_ehr/validate_agent_task_execution.py",
            "src/chrono_ehr/agent_task_router.py",
            "outputs/reports/agent_task_plan.md",
            "outputs/tables/agent_task_plan.csv",
            "outputs/tables/agent_task_scenario.csv",
            "outputs/tables/agent_task_deferred_actions.csv",
            "outputs/tables/agent_task_execution.csv",
            "outputs/tables/agent_task_post_run_refresh.csv",
        ],
        "outputs": [
            "outputs/reports/agent_task_execution_validation.md",
            "outputs/tables/agent_task_execution_validation.csv",
        ],
    },
    {
        "artifact_id": "agent_task_scenario_library_validation",
        "inputs": [
            "src/chrono_ehr/agent_task_scenario_library.py",
            "src/chrono_ehr/validate_agent_task_scenarios.py",
            "src/chrono_ehr/agent_task_router.py",
            "outputs/reports/agent_task_scenario_library.md",
            "outputs/tables/agent_task_scenario_library.csv",
            "outputs/tables/agent_task_scenario_examples.csv",
        ],
        "outputs": [
            "outputs/reports/agent_task_scenario_library_validation.md",
            "outputs/tables/agent_task_scenario_library_validation.csv",
        ],
    },
    {
        "artifact_id": "agent_cooldown_fingerprint_validation",
        "inputs": [
            "src/chrono_ehr/agent_cooldown_fingerprint.py",
            "src/chrono_ehr/validate_agent_cooldown_fingerprint.py",
            "configs/agent_cooldown_fingerprint.json",
        ],
        "outputs": [
            "outputs/reports/agent_cooldown_fingerprint_validation.md",
            "outputs/tables/agent_cooldown_fingerprint_validation.csv",
        ],
    },
    {
        "artifact_id": "agent_next_tasks_validation",
        "inputs": [
            "src/chrono_ehr/agent_next_task_planner.py",
            "src/chrono_ehr/validate_agent_next_tasks.py",
            "src/chrono_ehr/agent_cooldown_fingerprint.py",
            "configs/agent_cooldown_fingerprint.json",
            "src/chrono_ehr/agent_task_router.py",
            "outputs/reports/agent_next_tasks.md",
            "outputs/tables/agent_next_tasks.csv",
            "outputs/tables/agent_task_scenario_library.csv",
        ],
        "outputs": [
            "outputs/reports/agent_next_tasks_validation.md",
            "outputs/tables/agent_next_tasks_validation.csv",
        ],
    },
    {
        "artifact_id": "agent_task_queue_validation",
        "inputs": [
            "src/chrono_ehr/agent_task_queue.py",
            "src/chrono_ehr/validate_agent_task_queue.py",
            "src/chrono_ehr/agent_cooldown_fingerprint.py",
            "configs/agent_cooldown_fingerprint.json",
            "outputs/reports/agent_task_queue.md",
            "outputs/tables/agent_task_queue.csv",
            "outputs/tables/agent_next_tasks.csv",
        ],
        "outputs": [
            "outputs/reports/agent_task_queue_validation.md",
            "outputs/tables/agent_task_queue_validation.csv",
        ],
    },
    {
        "artifact_id": "agent_task_queue_execution_validation",
        "inputs": [
            "src/chrono_ehr/agent_task_queue_runner.py",
            "src/chrono_ehr/validate_agent_task_queue_execution.py",
            "outputs/reports/agent_task_queue_execution.md",
            "outputs/tables/agent_task_queue_execution.csv",
            "outputs/state/agent_task_queue_execution_history.csv",
            "outputs/tables/agent_task_queue.csv",
        ],
        "outputs": [
            "outputs/reports/agent_task_queue_execution_validation.md",
            "outputs/tables/agent_task_queue_execution_validation.csv",
        ],
    },
    {
        "artifact_id": "agent_progress_score_validation",
        "inputs": [
            "src/chrono_ehr/validate_agent_progress_score.py",
            "outputs/reports/agent_progress_score.md",
            "outputs/tables/agent_progress_score.csv",
        ],
        "outputs": ["outputs/reports/agent_progress_score_validation.md", "outputs/tables/agent_progress_score_validation.csv"],
    },
    {
        "artifact_id": "agent_state_validation",
        "inputs": ["src/chrono_ehr/validate_agent_state.py", "outputs/state/agent_state.json"],
        "outputs": ["outputs/reports/agent_state_validation.md", "outputs/tables/agent_state_validation.csv"],
    },
    {
        "artifact_id": "agent_runbook_retry_plan",
        "inputs": [
            "src/chrono_ehr/agent_runbook_retry_planner.py",
            "outputs/tables/agent_runbook_state_machine.csv",
        ],
        "outputs": ["outputs/reports/agent_runbook_retry_plan.md", "outputs/tables/agent_runbook_retry_plan.csv"],
    },
    {
        "artifact_id": "agent_runbook_retry_plan_validation",
        "inputs": [
            "src/chrono_ehr/validate_agent_runbook_retry_plan.py",
            "outputs/tables/agent_runbook_retry_plan.csv",
        ],
        "outputs": [
            "outputs/reports/agent_runbook_retry_plan_validation.md",
            "outputs/tables/agent_runbook_retry_plan_validation.csv",
        ],
    },
    {
        "artifact_id": "next_study_action_plan_validation",
        "inputs": [
            "src/chrono_ehr/validate_next_study_plan.py",
            "src/chrono_ehr/next_study_planner.py",
            "outputs/tables/next_study_action_plan.csv",
            "outputs/reports/next_study_action_plan.md",
            "outputs/tables/external_benchmark_readiness_summary.csv",
        ],
        "outputs": [
            "outputs/reports/next_study_action_plan_validation.md",
            "outputs/tables/next_study_action_plan_validation.csv",
        ],
    },
    {
        "artifact_id": "external_benchmark_summary_validation",
        "inputs": [
            "src/chrono_ehr/external_benchmark_summary_table.py",
            "src/chrono_ehr/validate_external_benchmark_summary_table.py",
            "outputs/tables/external_benchmark_summary_table.csv",
            "outputs/tables/external_benchmark_hard_metrics_table.csv",
        ],
        "outputs": [
            "outputs/reports/external_benchmark_summary_validation.md",
            "outputs/tables/external_benchmark_summary_validation.csv",
        ],
    },
    {
        "artifact_id": "external_technical_summary_validation",
        "inputs": [
            "src/chrono_ehr/external_technical_summary.py",
            "src/chrono_ehr/validate_external_technical_summary.py",
            "outputs/tables/external_benchmark_summary_table.csv",
            "outputs/tables/external_benchmark_hard_metrics_table.csv",
            "outputs/tables/external_subgroup_bootstrap_ci.csv",
            "outputs/tables/cdsl_decision_curve.csv",
            "outputs/tables/eicu_probability_recalibration_decision_curve.csv",
            "outputs/tables/charls_probability_recalibration_decision_curve.csv",
            "outputs/tables/external_model_comparison_recalibration_decision_curve.csv",
            "outputs/tables/external_technical_summary_table.csv",
        ],
        "outputs": [
            "outputs/reports/external_technical_summary_validation.md",
            "outputs/tables/external_technical_summary_validation.csv",
        ],
    },
    {
        "artifact_id": "external_calibration_decision_summary_validation",
        "inputs": [
            "src/chrono_ehr/external_calibration_decision_summary.py",
            "src/chrono_ehr/validate_external_calibration_decision_summary.py",
            "outputs/tables/cdsl_calibration_summary.csv",
            "outputs/tables/eicu_probability_recalibration_summary.csv",
            "outputs/tables/charls_probability_recalibration_summary.csv",
            "outputs/tables/external_model_comparison_recalibration_summary.csv",
            "outputs/tables/cdsl_decision_curve.csv",
            "outputs/tables/eicu_probability_recalibration_decision_curve.csv",
            "outputs/tables/charls_probability_recalibration_decision_curve.csv",
            "outputs/tables/external_model_comparison_recalibration_decision_curve.csv",
            "outputs/tables/external_calibration_decision_summary.csv",
        ],
        "outputs": [
            "outputs/reports/external_calibration_decision_summary_validation.md",
            "outputs/tables/external_calibration_decision_summary_validation.csv",
        ],
    },
    {
        "artifact_id": "external_model_selection_rationale_validation",
        "inputs": [
            "src/chrono_ehr/external_model_selection_rationale.py",
            "src/chrono_ehr/validate_external_model_selection_rationale.py",
            "outputs/tables/external_benchmark_summary_table.csv",
            "outputs/tables/external_benchmark_hard_metrics_table.csv",
            "outputs/tables/external_model_selection_rationale.csv",
        ],
        "outputs": [
            "outputs/reports/external_model_selection_rationale_validation.md",
            "outputs/tables/external_model_selection_rationale_validation.csv",
        ],
    },
    {
        "artifact_id": "external_metric_consistency_audit_validation",
        "inputs": [
            "src/chrono_ehr/external_metric_consistency_audit.py",
            "src/chrono_ehr/validate_external_metric_consistency_audit.py",
            "outputs/tables/external_benchmark_summary_table.csv",
            "outputs/tables/external_benchmark_hard_metrics_table.csv",
            "outputs/tables/external_technical_summary_table.csv",
            "outputs/tables/external_calibration_decision_summary.csv",
            "outputs/tables/external_model_selection_rationale.csv",
            "outputs/tables/external_model_bootstrap_ci.csv",
            "outputs/tables/external_metric_consistency_audit.csv",
        ],
        "outputs": [
            "outputs/reports/external_metric_consistency_audit_validation.md",
            "outputs/tables/external_metric_consistency_audit_validation.csv",
        ],
    },
    {
        "artifact_id": "external_summary_asset_manifest_validation",
        "inputs": [
            "src/chrono_ehr/external_summary_asset_manifest.py",
            "src/chrono_ehr/validate_external_summary_asset_manifest.py",
            "outputs/tables/external_technical_summary_table.csv",
            "outputs/tables/external_benchmark_summary_table.csv",
            "outputs/tables/external_calibration_decision_summary.csv",
            "outputs/tables/external_model_selection_rationale.csv",
            "outputs/tables/external_metric_consistency_audit.csv",
            "outputs/tables/external_summary_asset_manifest.csv",
        ],
        "outputs": [
            "outputs/reports/external_summary_asset_manifest_validation.md",
            "outputs/tables/external_summary_asset_manifest_validation.csv",
        ],
    },
    {
        "artifact_id": "external_handoff_package_validation",
        "inputs": [
            "src/chrono_ehr/external_handoff_package.py",
            "src/chrono_ehr/validate_external_handoff_package.py",
            "outputs/tables/external_summary_asset_manifest.csv",
            "outputs/tables/external_metric_consistency_audit.csv",
            "outputs/tables/external_handoff_package_manifest.csv",
            "outputs/reports/external_handoff_package.md",
            "outputs/external_handoff_package.zip",
        ],
        "outputs": [
            "outputs/reports/external_handoff_package_validation.md",
            "outputs/tables/external_handoff_package_validation.csv",
        ],
    },
    {
        "artifact_id": "external_subgroup_robustness_summary_validation",
        "inputs": [
            "src/chrono_ehr/external_subgroup_robustness_summary.py",
            "src/chrono_ehr/validate_external_subgroup_robustness_summary.py",
            "outputs/tables/external_benchmark_summary_table.csv",
            "outputs/tables/external_subgroup_bootstrap_ci.csv",
            "outputs/tables/external_subgroup_robustness_summary.csv",
        ],
        "outputs": [
            "outputs/reports/external_subgroup_robustness_summary_validation.md",
            "outputs/tables/external_subgroup_robustness_summary_validation.csv",
        ],
    },
    {
        "artifact_id": "external_threshold_band_sensitivity_validation",
        "inputs": [
            "src/chrono_ehr/external_threshold_band_sensitivity.py",
            "src/chrono_ehr/validate_external_threshold_band_sensitivity.py",
            "outputs/tables/external_benchmark_summary_table.csv",
            "outputs/tables/cdsl_decision_curve.csv",
            "outputs/tables/eicu_probability_recalibration_decision_curve.csv",
            "outputs/tables/charls_probability_recalibration_decision_curve.csv",
            "outputs/tables/external_model_comparison_recalibration_decision_curve.csv",
            "outputs/tables/external_threshold_band_sensitivity.csv",
        ],
        "outputs": [
            "outputs/reports/external_threshold_band_sensitivity_validation.md",
            "outputs/tables/external_threshold_band_sensitivity_validation.csv",
        ],
    },
    {
        "artifact_id": "external_calibration_method_rationale_validation",
        "inputs": [
            "src/chrono_ehr/external_calibration_method_rationale.py",
            "src/chrono_ehr/validate_external_calibration_method_rationale.py",
            "outputs/tables/external_benchmark_summary_table.csv",
            "outputs/tables/external_calibration_decision_summary.csv",
            "outputs/tables/external_calibration_method_rationale.csv",
        ],
        "outputs": [
            "outputs/reports/external_calibration_method_rationale_validation.md",
            "outputs/tables/external_calibration_method_rationale_validation.csv",
        ],
    },
    {
        "artifact_id": "external_bootstrap_ci_validation",
        "inputs": [
            "src/chrono_ehr/external_model_bootstrap_ci.py",
            "src/chrono_ehr/validate_external_model_bootstrap_ci.py",
            "outputs/tables/cdsl_traditional_baselines_predictions.csv",
            "outputs/tables/eicu_first24h_logistic_baseline_predictions.csv",
            "outputs/tables/eicu_first24h_model_comparison_predictions.csv",
            "outputs/tables/charls_probability_recalibration_predictions.csv",
            "outputs/tables/charls_incident_diabetes_model_comparison_predictions.csv",
            "outputs/tables/external_model_comparison_recalibration_predictions.csv",
            "outputs/tables/external_model_bootstrap_ci.csv",
        ],
        "outputs": [
            "outputs/reports/external_model_bootstrap_ci_validation.md",
            "outputs/tables/external_model_bootstrap_ci_validation.csv",
        ],
    },
    {
        "artifact_id": "cdsl_calibration_decision_validation",
        "inputs": [
            "src/chrono_ehr/cdsl_calibration_decision_curve.py",
            "src/chrono_ehr/validate_cdsl_calibration_decision_curve.py",
            "outputs/tables/cdsl_traditional_baselines_predictions.csv",
            "outputs/tables/cdsl_calibration_deciles.csv",
            "outputs/tables/cdsl_calibration_summary.csv",
            "outputs/tables/cdsl_decision_curve.csv",
        ],
        "outputs": [
            "outputs/reports/cdsl_calibration_decision_curve_validation.md",
            "outputs/tables/cdsl_calibration_decision_curve_validation.csv",
        ],
    },
    {
        "artifact_id": "external_subgroup_performance_validation",
        "inputs": [
            "src/chrono_ehr/external_subgroup_performance.py",
            "src/chrono_ehr/validate_external_subgroup_performance.py",
            "outputs/tables/cdsl_traditional_baselines_predictions.csv",
            "outputs/tables/eicu_probability_recalibration_predictions.csv",
            "outputs/tables/eicu_first24h_model_comparison_predictions.csv",
            "outputs/tables/charls_probability_recalibration_predictions.csv",
            "outputs/tables/charls_incident_diabetes_model_comparison_predictions.csv",
            "outputs/tables/external_model_comparison_recalibration_predictions.csv",
            "outputs/tables/external_subgroup_performance.csv",
        ],
        "outputs": [
            "outputs/reports/external_subgroup_performance_validation.md",
            "outputs/tables/external_subgroup_performance_validation.csv",
        ],
    },
    {
        "artifact_id": "external_subgroup_bootstrap_ci_validation",
        "inputs": [
            "src/chrono_ehr/external_subgroup_bootstrap_ci.py",
            "src/chrono_ehr/validate_external_subgroup_bootstrap_ci.py",
            "outputs/tables/cdsl_traditional_baselines_predictions.csv",
            "outputs/tables/eicu_probability_recalibration_predictions.csv",
            "outputs/tables/eicu_first24h_model_comparison_predictions.csv",
            "outputs/tables/charls_probability_recalibration_predictions.csv",
            "outputs/tables/charls_incident_diabetes_model_comparison_predictions.csv",
            "outputs/tables/external_model_comparison_recalibration_predictions.csv",
            "outputs/tables/external_subgroup_performance.csv",
            "outputs/tables/external_subgroup_bootstrap_ci.csv",
        ],
        "outputs": [
            "outputs/reports/external_subgroup_bootstrap_ci_validation.md",
            "outputs/tables/external_subgroup_bootstrap_ci_validation.csv",
        ],
    },
    {
        "artifact_id": "external_model_comparison_recalibration_validation",
        "inputs": [
            "src/chrono_ehr/external_model_comparison_recalibration.py",
            "src/chrono_ehr/validate_external_model_comparison_recalibration.py",
            "outputs/tables/eicu_first24h_model_comparison_predictions.csv",
            "outputs/tables/charls_incident_diabetes_model_comparison_predictions.csv",
            "outputs/tables/external_model_comparison_recalibration_predictions.csv",
            "outputs/tables/external_model_comparison_recalibration_metrics.csv",
            "outputs/tables/external_model_comparison_recalibration_deciles.csv",
            "outputs/tables/external_model_comparison_recalibration_summary.csv",
            "outputs/tables/external_model_comparison_recalibration_decision_curve.csv",
        ],
        "outputs": [
            "outputs/reports/external_model_comparison_recalibration_validation.md",
            "outputs/tables/external_model_comparison_recalibration_validation.csv",
        ],
    },
    {
        "artifact_id": "eicu_baseline_figures_validation",
        "inputs": [
            "src/chrono_ehr/validate_eicu_baseline_figures.py",
            "outputs/tables/eicu_first24h_calibration_summary.csv",
        ],
        "outputs": ["outputs/reports/eicu_baseline_figures_validation.md", "outputs/tables/eicu_baseline_figures_validation.csv"],
    },
    {
        "artifact_id": "eicu_probability_recalibration_validation",
        "inputs": [
            "src/chrono_ehr/eicu_probability_recalibration.py",
            "src/chrono_ehr/validate_eicu_probability_recalibration.py",
            "outputs/tables/eicu_first24h_logistic_baseline_predictions.csv",
            "outputs/tables/eicu_probability_recalibration_predictions.csv",
            "outputs/tables/eicu_probability_recalibration_metrics.csv",
            "outputs/tables/eicu_probability_recalibration_deciles.csv",
            "outputs/tables/eicu_probability_recalibration_summary.csv",
            "outputs/tables/eicu_probability_recalibration_decision_curve.csv",
        ],
        "outputs": [
            "outputs/reports/eicu_probability_recalibration_validation.md",
            "outputs/tables/eicu_probability_recalibration_validation.csv",
        ],
    },
    {
        "artifact_id": "eicu_model_comparison_validation",
        "inputs": [
            "src/chrono_ehr/eicu_first24h_model_comparison.py",
            "src/chrono_ehr/validate_eicu_first24h_model_comparison.py",
            "data/processed/eicu_first24h_feature_matrix_skeleton.csv",
            "outputs/tables/eicu_leakage_gate.csv",
            "outputs/tables/eicu_first24h_model_comparison_metrics.csv",
            "outputs/tables/eicu_first24h_model_comparison_predictions.csv",
            "outputs/tables/eicu_first24h_model_comparison_importances.csv",
        ],
        "outputs": [
            "outputs/reports/eicu_first24h_model_comparison_validation.md",
            "outputs/tables/eicu_first24h_model_comparison_validation.csv",
        ],
    },
    {
        "artifact_id": "charls_wave_variable_map_validation",
        "inputs": [
            "src/chrono_ehr/charls_wave_variable_map.py",
            "src/chrono_ehr/validate_charls_wave_variable_map.py",
            "outputs/tables/charls_wave_variable_map.csv",
            "outputs/tables/charls_harmonized_variable_inventory.csv",
        ],
        "outputs": [
            "outputs/reports/charls_wave_variable_map_validation.md",
            "outputs/tables/charls_wave_variable_map_validation.csv",
        ],
    },
    {
        "artifact_id": "charls_incident_diabetes_cohort_validation",
        "inputs": [
            "src/chrono_ehr/charls_incident_diabetes_cohort.py",
            "src/chrono_ehr/validate_charls_incident_diabetes_cohort.py",
            "outputs/tables/charls_wave_variable_map.csv",
            "data/processed/charls_incident_diabetes_cohort.csv",
            "outputs/tables/charls_incident_diabetes_cohort_summary.csv",
            "outputs/tables/charls_incident_diabetes_wave_outcome_summary.csv",
        ],
        "outputs": [
            "outputs/reports/charls_incident_diabetes_cohort_validation.md",
            "outputs/tables/charls_incident_diabetes_cohort_validation.csv",
        ],
    },
    {
        "artifact_id": "charls_baseline_features_validation",
        "inputs": [
            "src/chrono_ehr/charls_baseline_features.py",
            "src/chrono_ehr/validate_charls_baseline_features.py",
            "data/processed/charls_incident_diabetes_cohort.csv",
            "data/processed/charls_incident_diabetes_baseline_features.csv",
            "outputs/tables/charls_baseline_feature_manifest.csv",
            "outputs/tables/charls_baseline_feature_missingness.csv",
        ],
        "outputs": [
            "outputs/reports/charls_baseline_features_validation.md",
            "outputs/tables/charls_baseline_features_validation.csv",
        ],
    },
    {
        "artifact_id": "charls_leakage_gate",
        "inputs": [
            "src/chrono_ehr/charls_leakage_gate.py",
            "data/processed/charls_incident_diabetes_baseline_features.csv",
            "outputs/tables/charls_baseline_features_validation.csv",
            "outputs/tables/charls_wave_variable_map.csv",
        ],
        "outputs": [
            "outputs/reports/charls_leakage_gate_report.md",
            "outputs/tables/charls_leakage_gate.csv",
        ],
    },
    {
        "artifact_id": "charls_logistic_baseline_validation",
        "inputs": [
            "src/chrono_ehr/charls_incident_diabetes_baseline.py",
            "src/chrono_ehr/validate_charls_incident_diabetes_baseline.py",
            "data/processed/charls_incident_diabetes_baseline_features.csv",
            "outputs/tables/charls_leakage_gate.csv",
            "outputs/tables/charls_incident_diabetes_logistic_baseline_metrics.csv",
            "outputs/tables/charls_incident_diabetes_logistic_baseline_predictions.csv",
            "outputs/tables/charls_incident_diabetes_logistic_baseline_coefficients.csv",
        ],
        "outputs": [
            "outputs/reports/charls_incident_diabetes_logistic_baseline_validation.md",
            "outputs/tables/charls_incident_diabetes_logistic_baseline_validation.csv",
        ],
    },
    {
        "artifact_id": "charls_sensitivity_validation",
        "inputs": [
            "src/chrono_ehr/charls_incident_diabetes_sensitivity.py",
            "src/chrono_ehr/validate_charls_incident_diabetes_sensitivity.py",
            "data/processed/charls_incident_diabetes_baseline_features.csv",
            "data/processed/charls_incident_diabetes_cohort.csv",
            "outputs/tables/charls_leakage_gate.csv",
            "outputs/tables/charls_incident_diabetes_sensitivity_metrics.csv",
            "outputs/tables/charls_incident_diabetes_sensitivity_predictions.csv",
            "outputs/tables/charls_incident_diabetes_sensitivity_coefficients.csv",
        ],
        "outputs": [
            "outputs/reports/charls_incident_diabetes_sensitivity_validation.md",
            "outputs/tables/charls_incident_diabetes_sensitivity_validation.csv",
        ],
    },
    {
        "artifact_id": "charls_model_comparison_validation",
        "inputs": [
            "src/chrono_ehr/charls_incident_diabetes_model_comparison.py",
            "src/chrono_ehr/validate_charls_incident_diabetes_model_comparison.py",
            "data/processed/charls_incident_diabetes_baseline_features.csv",
            "outputs/tables/charls_leakage_gate.csv",
            "outputs/tables/charls_incident_diabetes_model_comparison_metrics.csv",
            "outputs/tables/charls_incident_diabetes_model_comparison_predictions.csv",
            "outputs/tables/charls_incident_diabetes_model_comparison_importances.csv",
        ],
        "outputs": [
            "outputs/reports/charls_incident_diabetes_model_comparison_validation.md",
            "outputs/tables/charls_incident_diabetes_model_comparison_validation.csv",
        ],
    },
    {
        "artifact_id": "charls_calibration_decision_validation",
        "inputs": [
            "src/chrono_ehr/charls_calibration_decision_curve.py",
            "src/chrono_ehr/validate_charls_calibration_decision_curve.py",
            "outputs/tables/charls_incident_diabetes_logistic_baseline_predictions.csv",
            "outputs/tables/charls_calibration_deciles.csv",
            "outputs/tables/charls_calibration_summary.csv",
            "outputs/tables/charls_decision_curve.csv",
        ],
        "outputs": [
            "outputs/reports/charls_calibration_decision_curve_validation.md",
            "outputs/tables/charls_calibration_decision_curve_validation.csv",
        ],
    },
    {
        "artifact_id": "charls_probability_recalibration_validation",
        "inputs": [
            "src/chrono_ehr/charls_probability_recalibration.py",
            "src/chrono_ehr/validate_charls_probability_recalibration.py",
            "outputs/tables/charls_incident_diabetes_logistic_baseline_predictions.csv",
            "outputs/tables/charls_probability_recalibration_predictions.csv",
            "outputs/tables/charls_probability_recalibration_metrics.csv",
            "outputs/tables/charls_probability_recalibration_deciles.csv",
            "outputs/tables/charls_probability_recalibration_summary.csv",
            "outputs/tables/charls_probability_recalibration_decision_curve.csv",
        ],
        "outputs": [
            "outputs/reports/charls_probability_recalibration_validation.md",
            "outputs/tables/charls_probability_recalibration_validation.csv",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def newest_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_mtime
    mtimes = [item.stat().st_mtime for item in path.rglob("*") if item.is_file()]
    return max(mtimes) if mtimes else path.stat().st_mtime


def fmt_ts(value: float | None) -> str:
    if value is None:
        return ""
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


def row(
    artifact_id: str,
    output: str,
    status: str,
    newest_input: float | None,
    output_mtime: float | None,
    detail: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "output": output,
        "status": status,
        "newest_input_time": fmt_ts(newest_input),
        "output_time": fmt_ts(output_mtime),
        "detail": detail,
    }


def audit(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for rule in FRESHNESS_RULES:
        artifact_id = str(rule["artifact_id"])
        input_times: dict[str, float | None] = {
            relative: newest_mtime(project_root / relative) for relative in rule.get("inputs", [])
        }
        missing_inputs = [relative for relative, mtime in input_times.items() if mtime is None]
        newest_input = max((mtime for mtime in input_times.values() if mtime is not None), default=None)
        for output in rule.get("outputs", []):
            output_path = project_root / output
            output_mtime = newest_mtime(output_path)
            if missing_inputs:
                status = "FAIL"
                detail = "missing inputs: " + ", ".join(missing_inputs)
            elif output_mtime is None:
                status = "FAIL"
                detail = "missing output"
            elif newest_input is not None and output_mtime + 1 < newest_input:
                status = "STALE"
                newest_inputs = [
                    relative for relative, mtime in input_times.items() if mtime is not None and abs(mtime - newest_input) < 0.001
                ]
                detail = "older than newest input: " + ", ".join(newest_inputs)
            else:
                status = "PASS"
                detail = "fresh"
            rows.append(row(artifact_id, output, status, newest_input, output_mtime, detail))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["artifact_id", "output", "status", "newest_input_time", "output_time", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_artifact_freshness.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_artifact_freshness.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Artifact Freshness Audit

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Artifact groups: {len(FRESHNESS_RULES)}
- Checks: {len(checks)}
- Failures or stale outputs: {len(failures)}
- Boundary: checks reproducibility freshness for local Agent control outputs only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent artifact freshness checks: {len(checks)}")
    print(f"Failures or stale outputs: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
