#!/usr/bin/env python3
"""Run lightweight health checks for ChronoEHR-Agent itself."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]

CHECKS = [
    {
        "id": "study_capabilities",
        "category": "registry",
        "command": "python3 src/chrono_ehr/run_study.py --study-capabilities",
    },
    {
        "id": "pipeline_steps",
        "category": "registry",
        "command": "python3 src/chrono_ehr/run_study.py --pipeline-steps",
    },
    {
        "id": "feature_windows",
        "category": "temporal_features",
        "command": "python3 src/chrono_ehr/run_study.py --validate-feature-windows",
    },
    {
        "id": "leakage_gate",
        "category": "leakage",
        "command": "python3 src/chrono_ehr/run_study.py --leakage-gate",
    },
    {
        "id": "extractor_window_usage",
        "category": "temporal_features",
        "command": "python3 src/chrono_ehr/run_study.py --audit-extractor-windows",
    },
    {
        "id": "external_readiness",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --external-readiness-summary",
    },
    {
        "id": "external_benchmark_summary_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary",
    },
    {
        "id": "external_technical_summary_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-technical-summary",
    },
    {
        "id": "external_calibration_decision_summary_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-calibration-decision-summary",
    },
    {
        "id": "external_model_selection_rationale_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-model-selection-rationale",
    },
    {
        "id": "external_metric_consistency_audit_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-metric-consistency-audit",
    },
    {
        "id": "external_summary_asset_manifest_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-summary-asset-manifest",
    },
    {
        "id": "external_handoff_package_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-handoff-package",
    },
    {
        "id": "external_subgroup_robustness_summary_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-subgroup-robustness-summary",
    },
    {
        "id": "external_threshold_band_sensitivity_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-threshold-band-sensitivity",
    },
    {
        "id": "external_calibration_method_rationale_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-calibration-method-rationale",
    },
    {
        "id": "external_bootstrap_ci_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-bootstrap-ci",
    },
    {
        "id": "cdsl_calibration_decision_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-cdsl-calibration-decision",
    },
    {
        "id": "external_subgroup_performance_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-subgroup-performance",
    },
    {
        "id": "external_subgroup_bootstrap_ci_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-subgroup-bootstrap-ci",
    },
    {
        "id": "external_model_comparison_recalibration_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-model-comparison-recalibration",
    },
    {
        "id": "external_field_role_catalog",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --external-field-role-catalog",
    },
    {
        "id": "external_field_role_catalog_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-external-field-role-catalog",
    },
    {
        "id": "next_study_plan",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --next-study-plan",
    },
    {
        "id": "next_study_plan_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-next-study-plan",
    },
    {
        "id": "eicu_cohort_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-eicu-cohort",
    },
    {
        "id": "eicu_temporal_features_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-eicu-temporal-features",
    },
    {
        "id": "eicu_leakage_gate",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --eicu-leakage-gate",
    },
    {
        "id": "eicu_logistic_baseline_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-eicu-logistic-baseline",
    },
    {
        "id": "eicu_model_comparison_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-eicu-model-comparison",
    },
    {
        "id": "eicu_baseline_figures_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-eicu-baseline-figures",
    },
    {
        "id": "eicu_probability_recalibration_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-eicu-probability-recalibration",
    },
    {
        "id": "charls_wave_map_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-charls-wave-map",
    },
    {
        "id": "charls_incident_diabetes_cohort_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-charls-incident-diabetes-cohort",
    },
    {
        "id": "charls_baseline_features_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-charls-baseline-features",
    },
    {
        "id": "charls_leakage_gate",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --charls-leakage-gate",
    },
    {
        "id": "charls_logistic_baseline_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-charls-logistic-baseline",
    },
    {
        "id": "charls_sensitivity_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-charls-sensitivity",
    },
    {
        "id": "charls_model_comparison_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-charls-model-comparison",
    },
    {
        "id": "charls_calibration_decision_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-charls-calibration-decision",
    },
    {
        "id": "charls_probability_recalibration_validation",
        "category": "external",
        "command": "python3 src/chrono_ehr/run_study.py --validate-charls-probability-recalibration",
    },
    {
        "id": "agent_routing",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-control",
    },
    {
        "id": "agent_task_router",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-task-router",
    },
    {
        "id": "agent_task_execution_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-task-execution",
    },
    {
        "id": "agent_task_scenarios",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-task-scenarios",
    },
    {
        "id": "agent_task_scenario_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-task-scenarios",
    },
    {
        "id": "agent_action_catalog",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-action-catalog",
    },
    {
        "id": "agent_control_panel",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-control --agent-goal status",
    },
    {
        "id": "agent_entrypoints",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-entrypoints",
    },
    {
        "id": "agent_entrypoints_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-entrypoints",
    },
    {
        "id": "config_coverage_audit",
        "category": "registry",
        "command": "python3 src/chrono_ehr/run_study.py --config-coverage-audit",
    },
    {
        "id": "config_migration_backlog",
        "category": "registry",
        "command": "python3 src/chrono_ehr/run_study.py --config-migration-backlog",
    },
    {
        "id": "config_code_rules_validation",
        "category": "registry",
        "command": "python3 src/chrono_ehr/run_study.py --validate-config-code-rules",
    },
    {
        "id": "study_config_schema_validation",
        "category": "registry",
        "command": "python3 src/chrono_ehr/run_study.py --validate-study-config-schema",
    },
    {
        "id": "agent_demo_workflow",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-demo-workflow",
    },
    {
        "id": "agent_recovery_plan",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-recovery-plan",
    },
    {
        "id": "agent_runbook_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-runbook",
    },
    {
        "id": "agent_runbook_confirmation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-runbook-confirmation",
    },
    {
        "id": "agent_runbook_state",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-runbook-state",
    },
    {
        "id": "agent_runbook_state_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-runbook-state",
    },
    {
        "id": "agent_runbook_retry_plan",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-runbook-retry-plan",
    },
    {
        "id": "agent_runbook_retry_plan_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-runbook-retry-plan",
    },
    {
        "id": "agent_next_tasks",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-next-tasks",
    },
    {
        "id": "agent_next_tasks_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-next-tasks",
    },
    {
        "id": "agent_task_queue",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-task-queue",
    },
    {
        "id": "agent_task_queue_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-task-queue",
    },
    {
        "id": "agent_task_queue_run",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-task-queue-run",
    },
    {
        "id": "agent_task_queue_execution_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-task-queue-execution",
    },
    {
        "id": "agent_cooldown_fingerprint_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-cooldown-fingerprint",
    },
    {
        "id": "agent_state",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-state",
    },
    {
        "id": "agent_state_validation",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --validate-agent-state",
    },
    {
        "id": "agent_command_lint",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-command-lint",
    },
    {
        "id": "agent_control_consistency",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-control-consistency",
    },
    {
        "id": "agent_dependency_audit",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-dependency-audit",
    },
    {
        "id": "agent_doc_command_audit",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-doc-command-audit",
    },
    {
        "id": "agent_handoff_checklist",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-handoff-checklist",
    },
    {
        "id": "agent_boundary_audit",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-boundary-audit",
    },
    {
        "id": "agent_artifact_freshness",
        "category": "agent_control",
        "command": "python3 src/chrono_ehr/run_study.py --agent-artifact-freshness",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def run_checks(project_root: Path) -> pd.DataFrame:
    rows = []
    for check in CHECKS:
        completed = subprocess.run(shlex.split(check["command"]), cwd=project_root, text=True, capture_output=True)
        rows.append(
            {
                **check,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "returncode": completed.returncode,
                "detail": (completed.stdout + completed.stderr).strip()[-1500:],
            }
        )
        if completed.returncode != 0:
            break
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["id", "category", "status", "command", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, checks: pd.DataFrame) -> Path:
    table_path = project_root / "outputs" / "tables" / "agent_self_check.csv"
    report_path = project_root / "outputs" / "reports" / "agent_self_check.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    text = f"""# Agent Self-Check

- Overall status: `{"PASS" if failures.empty and len(checks) == len(CHECKS) else "FAIL"}`
- Checks run: {len(checks)}/{len(CHECKS)}
- Failures: {len(failures)}
- Boundary: lightweight local workflow health check only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    checks = run_checks(args.project_root)
    report = write_outputs(args.project_root, checks)
    failures = int((checks["status"] != "PASS").sum())
    print(f"Wrote {report}")
    print(f"Agent self-checks run: {len(checks)}/{len(CHECKS)}")
    print(f"Failures: {failures}")
    if failures or len(checks) != len(CHECKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
