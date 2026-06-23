#!/usr/bin/env python3
"""Audit consistency across ChronoEHR-Agent control-layer registries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from agent_doctor import DOCTOR_STEPS
from agent_self_check import CHECKS
from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_ENTRYPOINT_IDS = {
    "agent_state",
    "validate_agent_state",
    "agent_self_check",
    "agent_status_card",
    "agent_progress_score",
    "validate_agent_progress_score",
    "validate_agent_status_card",
    "agent_task_scenarios",
    "validate_agent_task_scenarios",
    "agent_task_queue",
    "validate_agent_task_queue",
    "agent_task_queue_run",
    "validate_agent_task_queue_execution",
    "validate_agent_task_execution",
    "agent_doctor",
    "validate_agent_doctor",
    "validate_agent_entrypoints",
    "agent_command_lint",
    "agent_boundary_audit",
    "agent_artifact_freshness",
    "agent_dependency_audit",
    "agent_doc_command_audit",
    "agent_handoff_checklist",
}
REQUIRED_ACTION_IDS = {
    "agent_state_validation",
    "agent_status_card",
    "agent_progress_score",
    "agent_progress_score_validation",
    "agent_status_card_validation",
    "agent_task_execution_validation",
    "agent_task_scenarios",
    "agent_task_scenario_validation",
    "agent_task_queue",
    "agent_task_queue_validation",
    "agent_task_queue_run",
    "agent_task_queue_execution_validation",
    "agent_doctor",
    "agent_doctor_validation",
    "agent_task_execution_validation",
    "agent_entrypoints_validation",
    "agent_command_lint",
    "agent_boundary_audit",
    "agent_artifact_freshness",
    "agent_dependency_audit",
    "agent_doc_command_audit",
    "agent_handoff_checklist",
    "delivery_readiness",
}
REQUIRED_SELF_CHECK_IDS = {
    "agent_state",
    "agent_state_validation",
    "agent_task_scenarios",
    "agent_task_scenario_validation",
    "agent_task_queue",
    "agent_task_queue_validation",
    "agent_task_queue_run",
    "agent_task_queue_execution_validation",
    "agent_entrypoints_validation",
    "agent_command_lint",
    "agent_boundary_audit",
    "agent_artifact_freshness",
    "agent_dependency_audit",
    "agent_doc_command_audit",
    "agent_handoff_checklist",
}
REQUIRED_RUN_STUDY_FLAGS = {
    "--agent-state",
    "--validate-agent-state",
    "--agent-self-check",
    "--agent-status-card",
    "--agent-progress-score",
    "--validate-agent-progress-score",
    "--validate-agent-status-card",
    "--agent-task-scenarios",
    "--validate-agent-task-scenarios",
    "--agent-task-queue",
    "--validate-agent-task-queue",
    "--agent-task-queue-run",
    "--validate-agent-task-queue-execution",
    "--validate-agent-task-execution",
    "--agent-doctor",
    "--validate-agent-doctor",
    "--validate-agent-entrypoints",
    "--agent-command-lint",
    "--agent-boundary-audit",
    "--agent-artifact-freshness",
    "--agent-dependency-audit",
    "--agent-doc-command-audit",
    "--agent-handoff-checklist",
}
REQUIRED_DELIVERY_TOKENS = {
    "outputs/reports/agent_doctor.md",
    "outputs/tables/agent_doctor.csv",
    "outputs/reports/agent_doctor_validation.md",
    "outputs/tables/agent_doctor_validation.csv",
}
REQUIRED_MVP_TOKENS = {
    "agent_doctor_passed",
    "agent_doctor_validation_passed",
    "outputs/tables/agent_doctor.csv",
    "outputs/tables/agent_doctor_validation.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def command_ids_from_entrypoints(config: dict[str, Any]) -> set[str]:
    return {
        str(command.get("id", ""))
        for group in config.get("groups", [])
        for command in group.get("commands", [])
        if command.get("id")
    }


def action_ids_from_catalog(config: dict[str, Any]) -> set[str]:
    return {str(action.get("id", "")) for action in config.get("actions", []) if action.get("id")}


def run_study_flags(source: str) -> set[str]:
    return set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', source))


def required_present(name: str, required: set[str], observed: set[str], evidence: str) -> dict[str, str]:
    missing = sorted(required - observed)
    return row(name, "PASS" if not missing else "FAIL", evidence, "missing=" + ",".join(missing))


def audit(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    entrypoints_path = project_root / "configs" / "agent_entrypoints.json"
    catalog_path = project_root / "configs" / "agent_action_catalog.json"
    run_study_path = project_root / "src" / "chrono_ehr" / "run_study.py"
    delivery_path = project_root / "src" / "chrono_ehr" / "delivery_readiness_audit.py"
    mvp_path = project_root / "src" / "chrono_ehr" / "validate_mainline_mvp.py"
    quickstart_path = project_root / "docs" / "quickstart_usage.md"
    external_summary_path = project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv"
    capability_summary_path = project_root / "outputs" / "tables" / "study_capability_summary.csv"
    control_external_path = project_root / "outputs" / "tables" / "agent_control_external_state.csv"

    entrypoints = read_json(entrypoints_path)
    catalog = read_json(catalog_path)
    run_source = read_text(run_study_path)
    delivery_source = read_text(delivery_path)
    mvp_source = read_text(mvp_path)
    quickstart = read_text(quickstart_path)
    external_summary = read_csv(external_summary_path)
    capability_summary = read_csv(capability_summary_path)
    control_external = read_csv(control_external_path)

    entry_ids = command_ids_from_entrypoints(entrypoints)
    action_ids = action_ids_from_catalog(catalog)
    self_check_ids = {str(check.get("id", "")) for check in CHECKS}
    flags = run_study_flags(run_source)
    doctor_step_ids = {str(step.get("id", "")) for step in DOCTOR_STEPS}

    rows.append(row("entrypoint_ids_unique", "PASS" if len(entry_ids) == sum(len(g.get("commands", [])) for g in entrypoints.get("groups", [])) else "FAIL", str(entrypoints_path), f"ids={len(entry_ids)}"))
    rows.append(row("action_ids_unique", "PASS" if len(action_ids) == len(catalog.get("actions", [])) else "FAIL", str(catalog_path), f"ids={len(action_ids)}"))
    rows.append(required_present("required_control_entrypoints_present", REQUIRED_ENTRYPOINT_IDS, entry_ids, str(entrypoints_path)))
    rows.append(required_present("required_control_actions_present", REQUIRED_ACTION_IDS, action_ids, str(catalog_path)))
    rows.append(required_present("required_self_check_items_present", REQUIRED_SELF_CHECK_IDS, self_check_ids, "src/chrono_ehr/agent_self_check.py"))
    rows.append(required_present("required_run_study_flags_present", REQUIRED_RUN_STUDY_FLAGS, flags, str(run_study_path)))

    ordered_self_checks = [str(check.get("id", "")) for check in CHECKS]
    expected_order = [
        "agent_entrypoints_validation",
        "agent_state",
        "agent_state_validation",
        "agent_command_lint",
        "agent_dependency_audit",
        "agent_doc_command_audit",
        "agent_handoff_checklist",
        "agent_boundary_audit",
        "agent_artifact_freshness",
    ]
    observed_order = [item for item in ordered_self_checks if item in expected_order]
    rows.append(row("self_check_dependency_order", "PASS" if observed_order == expected_order else "FAIL", "src/chrono_ehr/agent_self_check.py", f"observed={observed_order}"))

    rows.append(row("doctor_uses_expected_steps", "PASS" if doctor_step_ids == {"agent_self_check", "delivery_readiness", "agent_artifact_freshness"} else "FAIL", "src/chrono_ehr/agent_doctor.py", f"doctor_steps={sorted(doctor_step_ids)}"))
    rows.append(row("doctor_validation_not_recursive", "PASS" if "--validate-agent-doctor" not in {str(step.get("command", "")) for step in DOCTOR_STEPS} else "FAIL", "src/chrono_ehr/agent_doctor.py", "doctor must not call its own validator"))
    rows.append(row("run_study_dispatches_doctor", "PASS" if "agent_doctor.py" in run_source and "validate_agent_doctor.py" in run_source else "FAIL", str(run_study_path), "expects doctor and validator dispatch"))

    missing_delivery = sorted(token for token in REQUIRED_DELIVERY_TOKENS if token not in delivery_source)
    rows.append(row("delivery_readiness_tracks_doctor_outputs", "PASS" if not missing_delivery else "FAIL", str(delivery_path), "missing=" + ",".join(missing_delivery)))
    missing_mvp = sorted(token for token in REQUIRED_MVP_TOKENS if token not in mvp_source)
    rows.append(row("mainline_mvp_tracks_doctor_outputs", "PASS" if not missing_mvp else "FAIL", str(mvp_path), "missing=" + ",".join(missing_mvp)))

    quickstart_tokens = {"--agent-doctor", "--validate-agent-doctor", "agent_doctor.md", "agent_doctor_validation.md"}
    missing_quickstart = sorted(token for token in quickstart_tokens if token not in quickstart)
    rows.append(row("quickstart_documents_doctor", "PASS" if not missing_quickstart else "FAIL", str(quickstart_path), "missing=" + ",".join(missing_quickstart)))

    if not external_summary.empty and not capability_summary.empty:
        external_status = dict(zip(external_summary["dataset"].astype(str), external_summary["local_status"].astype(str)))
        capability_status = dict(zip(capability_summary["study_id"].astype(str), capability_summary["overall_status"].astype(str)))
        eicu_external = external_status.get("eICU", "")
        eicu_capability = capability_status.get("eicu_temporal_mortality", "")
        eicu_advanced = eicu_external in {"READY_FOR_COHORT_CODE", "COHORT_READY", "FEATURE_READY", "BASELINE_READY", "READY"}
        rows.append(
            row(
                "eicu_external_and_capability_status_consistent",
                "PASS" if not eicu_advanced or eicu_capability != "DATA_PENDING" else "FAIL",
                f"{external_summary_path}; {capability_summary_path}",
                f"external={eicu_external}; capability={eicu_capability}",
            )
        )
        charls_external = external_status.get("CHARLS", "")
        charls_capability = capability_status.get("charls_incident_diabetes", "")
        rows.append(
            row(
                "charls_external_and_capability_status_consistent",
                "PASS" if charls_external != "DATA_PENDING" or charls_capability == "DATA_PENDING" else "FAIL",
                f"{external_summary_path}; {capability_summary_path}",
                f"external={charls_external}; capability={charls_capability}",
            )
        )
    else:
        rows.append(
            row(
                "external_capability_status_sources_exist",
                "FAIL",
                f"{external_summary_path}; {capability_summary_path}",
                f"external_rows={len(external_summary)}; capability_rows={len(capability_summary)}",
            )
        )

    if not control_external.empty and {"dataset", "local_status", "command"}.issubset(control_external.columns):
        eicu = control_external[control_external["dataset"].astype(str).eq("eICU")]
        if not eicu.empty:
            eicu_status = str(eicu.iloc[0]["local_status"])
            eicu_command = str(eicu.iloc[0]["command"])
            rows.append(
                row(
                    "control_panel_eicu_command_matches_status",
                    "PASS"
                    if eicu_status != "BASELINE_READY" or "--validate-external-benchmark-summary" in eicu_command
                    else "FAIL",
                    str(control_external_path),
                    f"status={eicu_status}; command={eicu_command}",
                )
            )
        charls = control_external[control_external["dataset"].astype(str).eq("CHARLS")]
        if not charls.empty:
            charls_status = str(charls.iloc[0]["local_status"])
            charls_command = str(charls.iloc[0]["command"])
            rows.append(
                row(
                    "control_panel_charls_command_matches_status",
                    "PASS" if charls_status != "DATA_PENDING" or "--charls-readiness" in charls_command else "FAIL",
                    str(control_external_path),
                    f"status={charls_status}; command={charls_command}",
                )
            )
    else:
        rows.append(
            row(
                "control_panel_external_state_exists",
                "FAIL",
                str(control_external_path),
                f"rows={len(control_external)}",
            )
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_control_consistency_audit.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_control_consistency_audit.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Control Consistency Audit

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: checks local Agent control-layer consistency only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent control consistency checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
