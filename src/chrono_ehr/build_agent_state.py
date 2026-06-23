#!/usr/bin/env python3
"""Build a persistent local state snapshot for ChronoEHR-Agent."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = DEFAULT_PROJECT / "configs" / "study_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def delivery_state(project_root: Path) -> dict[str, Any]:
    readiness = read_csv(project_root / "outputs" / "tables" / "delivery_readiness_audit.csv")
    if readiness.empty:
        return {"overall": "UNKNOWN", "checks": 0, "failures": None}
    failures = int((readiness["status"] != "PASS").sum())
    return {"overall": "PASS" if failures == 0 else "FAIL", "checks": len(readiness), "failures": failures}


def external_state(project_root: Path) -> list[dict[str, Any]]:
    external = read_csv(project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv")
    if external.empty:
        return []
    keep = ["dataset", "local_status", "recommended_first_task", "critical_blocker"]
    return external[[column for column in keep if column in external.columns]].fillna("").to_dict(orient="records")


def task_history(project_root: Path) -> list[dict[str, Any]]:
    history = read_csv(project_root / "outputs" / "state" / "agent_task_history.csv")
    if history.empty:
        return []
    keep = [
        "timestamp",
        "task",
        "goal_type",
        "risk_mode",
        "budget_mode",
        "selected_actions",
        "deferred_actions",
        "executed_actions",
        "failures",
        "post_refresh_steps",
        "post_refresh_failures",
    ]
    return history[[column for column in keep if column in history.columns]].tail(5).fillna("").to_dict(orient="records")


def last_task_execution(project_root: Path) -> dict[str, Any]:
    history = read_csv(project_root / "outputs" / "state" / "agent_task_history.csv")
    scenario = read_csv(project_root / "outputs" / "tables" / "agent_task_scenario.csv")
    execution = read_csv(project_root / "outputs" / "tables" / "agent_task_execution.csv")
    refresh = read_csv(project_root / "outputs" / "tables" / "agent_task_post_run_refresh.csv")
    validation = read_csv(project_root / "outputs" / "tables" / "agent_task_execution_validation.csv")
    if history.empty:
        return {"available": False}
    last = history.tail(1).fillna("").to_dict(orient="records")[0]
    executed = int(execution["status"].isin(["PASS", "FAIL"]).sum()) if not execution.empty and "status" in execution else 0
    execution_failures = int((execution["status"] == "FAIL").sum()) if not execution.empty and "status" in execution else 0
    refresh_steps = int(refresh["status"].isin(["PASS", "FAIL"]).sum()) if not refresh.empty and "status" in refresh else 0
    refresh_failures = int((refresh["status"] == "FAIL").sum()) if not refresh.empty and "status" in refresh else 0
    validation_failures = int((validation["status"] != "PASS").sum()) if not validation.empty and "status" in validation else None
    executed_ids = execution["id"].astype(str).tolist() if not execution.empty and "id" in execution else []
    refresh_ids = refresh["id"].astype(str).tolist() if not refresh.empty and "id" in refresh else []
    scenario_row = scenario.tail(1).fillna("").to_dict(orient="records")[0] if not scenario.empty else {}
    return {
        "available": True,
        "task": last.get("task", ""),
        "timestamp": last.get("timestamp", ""),
        "scenario_id": scenario_row.get("scenario_id", ""),
        "scenario_title": scenario_row.get("title", ""),
        "scenario_next_step_hint": scenario_row.get("next_step_hint", ""),
        "goal_type": last.get("goal_type", ""),
        "risk_mode": last.get("risk_mode", ""),
        "budget_mode": last.get("budget_mode", ""),
        "selected_actions": int(float(last.get("selected_actions", 0) or 0)),
        "deferred_actions": int(float(last.get("deferred_actions", 0) or 0)),
        "executed_actions": executed,
        "execution_failures": execution_failures,
        "post_refresh_steps": refresh_steps,
        "post_refresh_failures": refresh_failures,
        "validation_failures": validation_failures,
        "executed_action_ids": executed_ids,
        "post_refresh_ids": refresh_ids,
        "report": last.get("report", ""),
    }


def runbook_summary(project_root: Path) -> dict[str, Any]:
    runbook = read_csv(project_root / "outputs" / "tables" / "agent_runbook.csv")
    execution = read_csv(project_root / "outputs" / "tables" / "agent_runbook_execution.csv")
    phase_summary = read_csv(project_root / "outputs" / "tables" / "agent_runbook_phase_summary.csv")
    if runbook.empty:
        return {"available": False}
    phase_counts = runbook["phase"].value_counts().to_dict() if "phase" in runbook else {}
    failures = int((execution["status"] == "FAIL").sum()) if not execution.empty and "status" in execution else 0
    executed = int(execution["status"].isin(["PASS", "FAIL"]).sum()) if not execution.empty and "status" in execution else 0
    keep = ["phase", "risk_level", "planned_actions", "executed_actions", "failed_actions", "status", "next_gate"]
    phase_state = (
        phase_summary[[column for column in keep if column in phase_summary.columns]].fillna("").to_dict(orient="records")
        if not phase_summary.empty
        else []
    )
    return {
        "available": True,
        "actions": len(runbook),
        "phase_counts": phase_counts,
        "executed_actions": executed,
        "execution_failures": failures,
        "phase_state": phase_state,
    }


def next_tasks(project_root: Path) -> list[dict[str, Any]]:
    tasks = read_csv(project_root / "outputs" / "tables" / "agent_next_tasks.csv")
    if tasks.empty:
        return []
    keep = [
        "priority",
        "scenario_id",
        "execution_mode",
        "completion_status",
        "last_success_run_id",
        "cooldown_fingerprint_status",
        "cooldown_reason",
        "cooldown_policy_summary",
        "next_task",
        "suggested_agent_task",
        "suggested_safe_refresh_command",
        "command",
        "reason",
    ]
    return tasks[[column for column in keep if column in tasks.columns]].head(5).fillna("").to_dict(orient="records")


def queue_execution_history(project_root: Path) -> dict[str, Any]:
    history = read_csv(project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv")
    if history.empty:
        return {"available": False, "rows": 0}
    latest_run_id = str(history["run_id"].dropna().astype(str).iloc[-1]) if "run_id" in history else ""
    latest = history[history["run_id"].astype(str).eq(latest_run_id)] if latest_run_id else history.tail(1)
    failures = int(latest["execution_status"].astype(str).eq("FAIL").sum()) if "execution_status" in latest else 0
    q003_success = history[
        history["queue_id"].astype(str).eq("Q003")
        & history["scenario_id"].astype(str).eq("agent_control_focus")
        & history["execution_status"].astype(str).eq("PASS")
    ]
    keep = [
        "run_id",
        "run_mode",
        "queue_filter",
        "scenario_filter",
        "queue_id",
        "scenario_id",
        "queue_status",
        "execution_status",
        "started_at",
        "cooldown_fingerprint_status",
        "command",
    ]
    return {
        "available": True,
        "rows": len(history),
        "latest_run_id": latest_run_id,
        "latest_run_mode": str(latest.iloc[0].get("run_mode", "")) if not latest.empty else "",
        "latest_queue_filter": str(latest.iloc[0].get("queue_filter", "")) if not latest.empty else "",
        "latest_scenario_filter": str(latest.iloc[0].get("scenario_filter", "")) if not latest.empty else "",
        "latest_rows": len(latest),
        "latest_failures": failures,
        "agent_control_focus_success_rows": len(q003_success),
        "latest_items": latest[[column for column in keep if column in latest.columns]].fillna("").to_dict(orient="records"),
    }


def current_queue(project_root: Path) -> list[dict[str, Any]]:
    queue = read_csv(project_root / "outputs" / "tables" / "agent_task_queue.csv")
    if queue.empty:
        return []
    keep = [
        "queue_id",
        "priority",
        "queue_status",
        "scenario_id",
        "next_task",
        "last_success_run_id",
        "cooldown_fingerprint_status",
        "cooldown_reason",
        "cooldown_policy_summary",
    ]
    return queue[[column for column in keep if column in queue.columns]].head(10).fillna("").to_dict(orient="records")


def cooldown_policy(project_root: Path) -> dict[str, Any]:
    config = read_json(project_root / "configs" / "agent_cooldown_fingerprint.json")
    inputs = config.get("fingerprint_inputs", []) if isinstance(config, dict) else []
    excluded = config.get("excluded_volatile_patterns", []) if isinstance(config, dict) else []
    queue = read_csv(project_root / "outputs" / "tables" / "agent_task_queue.csv")
    history = read_csv(project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv")
    q003 = (
        queue[queue["scenario_id"].astype(str).eq("agent_control_focus")].tail(1)
        if not queue.empty and "scenario_id" in queue.columns
        else pd.DataFrame()
    )
    q003_success = (
        history[
            history["scenario_id"].astype(str).eq("agent_control_focus")
            & history["execution_status"].astype(str).eq("PASS")
        ].tail(1)
        if not history.empty and {"scenario_id", "execution_status"}.issubset(history.columns)
        else pd.DataFrame()
    )
    return {
        "available": bool(config),
        "config_version": config.get("version", ""),
        "config_path": "configs/agent_cooldown_fingerprint.json",
        "input_count": len(inputs) if isinstance(inputs, list) else 0,
        "volatile_outputs_excluded": "outputs/" in excluded if isinstance(excluded, list) else False,
        "q003_queue_status": str(q003.iloc[0].get("queue_status", "")) if not q003.empty else "",
        "q003_cooldown_status": str(q003.iloc[0].get("cooldown_fingerprint_status", "")) if not q003.empty else "",
        "q003_last_success_run_id": str(q003_success.iloc[0].get("run_id", "")) if not q003_success.empty else "",
    }


def active_focus(last_task: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_id = str(last_task.get("scenario_id", "")) if last_task.get("available") else ""
    if scenario_id == "agent_control_focus":
        matching_next = next((item for item in tasks if item.get("scenario_id") == "agent_control_focus"), {})
        return {
            "focus_id": "agent_control_focus",
            "summary": "先完善 Agent 控制层，不做汇报材料。",
            "source": "last_agent_task",
            "task": last_task.get("task", ""),
            "safe_next_command": matching_next.get(
                "suggested_safe_refresh_command",
                "python3 src/chrono_ehr/run_study.py --agent-task \"先完善 Agent 控制层，不要做汇报材料\" --agent-task-execute-safe --agent-task-post-run-refresh",
            ),
            "guardrails": [
                "Do not start report, manuscript, slide, or mentor-update materials unless explicitly requested.",
                "Do not train or recalibrate models from this focus alone.",
                "Do not run expensive raw-table scans from this focus alone.",
                "Keep work inside local Agent control, validation, state, queue, doctor, freshness, and handoff modules.",
            ],
        }
    if scenario_id:
        return {
            "focus_id": scenario_id,
            "summary": str(last_task.get("scenario_next_step_hint", "")),
            "source": "last_agent_task",
            "task": last_task.get("task", ""),
            "safe_next_command": "",
            "guardrails": [
                "Keep the task inside the local EHR research workflow boundary.",
                "Do not provide medical QA, diagnosis, or treatment recommendation.",
            ],
        }
    matching_next = next((item for item in tasks if item.get("execution_mode") == "safe_auto_allowed"), {})
    return {
        "focus_id": str(matching_next.get("scenario_id", "validation_first") or "validation_first"),
        "summary": str(matching_next.get("next_task", "Run validation-first Agent checks.")),
        "source": "next_tasks",
        "task": str(matching_next.get("suggested_agent_task", "")),
        "safe_next_command": str(matching_next.get("suggested_safe_refresh_command", "")),
        "guardrails": [
            "Keep the task inside the local EHR research workflow boundary.",
            "Do not provide medical QA, diagnosis, or treatment recommendation.",
        ],
    }


def runbook_state_machine(project_root: Path) -> dict[str, Any]:
    payload = read_json(project_root / "outputs" / "state" / "agent_runbook_state_machine.json")
    if not payload:
        return {"available": False}
    summary = payload.get("summary", {})
    phases = payload.get("phases", [])
    return {
        "available": True,
        "summary": summary,
        "phases": phases,
    }


def known_boundaries(external: list[dict[str, Any]]) -> list[str]:
    status_by_dataset = {str(item.get("dataset", "")): str(item.get("local_status", "")) for item in external}
    eicu_status = status_by_dataset.get("eICU", "")
    if eicu_status in {"BASELINE_READY", "FEATURE_READY", "COHORT_READY", "READY_FOR_COHORT_CODE"}:
        eicu_boundary = "eICU is an external ICU mortality benchmark; do not describe it as chronic readmission external validation."
    else:
        eicu_boundary = "eICU should enter the workflow only after required raw CSV tables are confirmed locally."
    return [
        "CDSL is an external temporal-method benchmark, not direct external validation for MIMIC chronic readmission.",
        eicu_boundary,
        "CHARLS is a longitudinal chronic-disease cohort extension, not an EHR external validation dataset.",
        "Report generation is useful later, but current development priority is Agent workflow control.",
    ]


def study_state(project_root: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    capability = read_csv(project_root / "outputs" / "tables" / "study_capability_summary.csv")
    pipeline = read_csv(project_root / "outputs" / "tables" / "pipeline_step_summary.csv")
    capability_by_id = capability.set_index("study_id").to_dict(orient="index") if not capability.empty else {}
    pipeline_by_id = pipeline.set_index("study_id").to_dict(orient="index") if not pipeline.empty else {}
    rows = []
    for study in registry.get("studies", []):
        study_id = study.get("id", "")
        cap = capability_by_id.get(study_id, {})
        pipe = pipeline_by_id.get(study_id, {})
        rows.append(
            {
                "study_id": study_id,
                "cohort": study.get("cohort", ""),
                "registry_status": study.get("status", ""),
                "capability_status": cap.get("overall_status", "UNKNOWN"),
                "capability_completion": cap.get("completion_percent", ""),
                "pipeline_completion": pipe.get("completion_percent", ""),
                "safe_rerun_command": pipe.get("recommended_safe_rerun", ""),
            }
        )
    return rows


def build_state(project_root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    studies = study_state(project_root, registry)
    primary = registry.get("active_study", "mimic_iv_3_1_diabetes_readmission")
    primary_row = next((study for study in studies if study["study_id"] == primary), {})
    delivery = delivery_state(project_root)
    external = external_state(project_root)
    last_task = last_task_execution(project_root)
    tasks = next_tasks(project_root)
    queue_history = queue_execution_history(project_root)
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "boundary": "Local research workflow state only; not medical QA, diagnosis, or treatment recommendation.",
        "primary_demo": {
            "study_id": primary,
            "cohort": primary_row.get("cohort", "diabetes"),
            "reason": "Most stable vertical slice for ChronoEHR-Agent.",
            "safe_rerun_command": primary_row.get(
                "safe_rerun_command",
                "python3 src/chrono_ehr/run_study.py --study mimic_iv_3_1_diabetes_readmission --skip-existing --no-expensive",
            ),
        },
        "delivery_readiness": delivery,
        "studies": studies,
        "external_datasets": external,
        "active_focus": active_focus(last_task, tasks),
        "recent_tasks": task_history(project_root),
        "last_task_execution": last_task,
        "current_queue": current_queue(project_root),
        "cooldown_policy": cooldown_policy(project_root),
        "queue_execution_history": queue_history,
        "last_runbook": runbook_summary(project_root),
        "runbook_state_machine": runbook_state_machine(project_root),
        "next_tasks": tasks,
        "known_boundaries": known_boundaries(external),
        "recommended_next_commands": [
            "python3 src/chrono_ehr/run_study.py --agent-doctor",
            "python3 src/chrono_ehr/run_study.py --validate-agent-state",
            "python3 src/chrono_ehr/run_study.py --agent-next-tasks",
            "python3 src/chrono_ehr/run_study.py --agent-control --agent-goal status",
            "python3 src/chrono_ehr/run_study.py --agent-control-consistency",
        ],
    }


def write_markdown(project_root: Path, state: dict[str, Any]) -> Path:
    output = project_root / "outputs" / "state" / "agent_state.md"
    lines = [
        "# ChronoEHR-Agent State",
        "",
        f"- Updated at: `{state['updated_at']}`",
        f"- Boundary: {state['boundary']}",
        f"- Primary demo: `{state['primary_demo']['study_id']}` ({state['primary_demo']['cohort']})",
        f"- Delivery readiness: `{state['delivery_readiness']['overall']}` "
        f"({state['delivery_readiness']['checks']} checks, failures={state['delivery_readiness']['failures']})",
        f"- Active focus: `{state.get('active_focus', {}).get('focus_id', '')}` "
        f"{state.get('active_focus', {}).get('summary', '')}",
        "",
        "## Active Focus",
        "",
        f"- Focus id: `{state.get('active_focus', {}).get('focus_id', '')}`",
        f"- Summary: {state.get('active_focus', {}).get('summary', '')}",
        f"- Source: `{state.get('active_focus', {}).get('source', '')}`",
        f"- Safe next command: `{state.get('active_focus', {}).get('safe_next_command', '')}`",
        "",
        "### Guardrails",
        "",
    ]
    for item in state.get("active_focus", {}).get("guardrails", []):
        lines.append(f"- {item}")
    lines.extend([
        "## Studies",
        "",
        "| study_id | cohort | capability_status | capability_completion | pipeline_completion |",
        "|---|---|---|---:|---:|",
    ])
    for item in state["studies"]:
        lines.append(
            f"| {item['study_id']} | {item['cohort']} | {item['capability_status']} | "
            f"{item['capability_completion']} | {item['pipeline_completion']} |"
        )
    lines.extend(["", "## External Datasets", "", "| dataset | status | blocker |", "|---|---|---|"])
    for item in state["external_datasets"]:
        lines.append(f"| {item.get('dataset', '')} | {item.get('local_status', '')} | {item.get('critical_blocker', '')} |")
    lines.extend(["", "## Recommended Next Commands", ""])
    for command in state["recommended_next_commands"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Current Task Queue", "", "| queue_id | status | scenario | last_success | next_task |", "|---|---|---|---|---|"])
    for item in state.get("current_queue", []):
        lines.append(
            f"| {item.get('queue_id', '')} | {item.get('queue_status', '')} | {item.get('scenario_id', '')} | "
            f"{item.get('last_success_run_id', '')} | {item.get('next_task', '')} |"
        )
    queue_history = state.get("queue_execution_history", {})
    lines.extend(["", "## Queue Execution History", ""])
    if not queue_history.get("available"):
        lines.append("No queue execution history has been recorded yet.")
    else:
        lines.extend(
            [
                f"- Rows: {queue_history.get('rows', 0)}",
                f"- Latest run id: `{queue_history.get('latest_run_id', '')}`",
                f"- Latest mode/filter: `{queue_history.get('latest_run_mode', '')}` / "
                f"`{queue_history.get('latest_queue_filter', '')}` / `{queue_history.get('latest_scenario_filter', '')}`",
                f"- Latest failures: {queue_history.get('latest_failures', 0)}",
                f"- Preserved Q003 agent-control successes: {queue_history.get('agent_control_focus_success_rows', 0)}",
                "",
                "| queue_id | scenario_id | status | command |",
                "|---|---|---|---|",
            ]
        )
        for item in queue_history.get("latest_items", []):
            lines.append(
                f"| {item.get('queue_id', '')} | {item.get('scenario_id', '')} | "
                f"{item.get('execution_status', '')} | `{item.get('command', '')}` |"
            )
    last_task = state.get("last_task_execution", {})
    lines.extend(["", "## Last Agent Task Execution", ""])
    if not last_task.get("available"):
        lines.append("No natural-language agent task has been recorded yet.")
    else:
        lines.extend(
            [
                f"- Task: `{last_task.get('task', '')}`",
                f"- Scenario: `{last_task.get('scenario_id', '')}` ({last_task.get('scenario_title', '')})",
                f"- Goal/risk/budget: `{last_task.get('goal_type', '')}` / `{last_task.get('risk_mode', '')}` / `{last_task.get('budget_mode', '')}`",
                f"- Selected actions: {last_task.get('selected_actions', 0)}",
                f"- Deferred actions: {last_task.get('deferred_actions', 0)}",
                f"- Executed actions: {last_task.get('executed_actions', 0)}",
                f"- Execution failures: {last_task.get('execution_failures', 0)}",
                f"- Post-refresh steps: {last_task.get('post_refresh_steps', 0)}",
                f"- Post-refresh failures: {last_task.get('post_refresh_failures', 0)}",
                f"- Validation failures: {last_task.get('validation_failures', '')}",
                f"- Scenario next step hint: {last_task.get('scenario_next_step_hint', '')}",
                f"- Report: `{last_task.get('report', '')}`",
            ]
        )
    lines.extend(["", "## Recent Agent Tasks", "", "| timestamp | task | goal | risk | budget | selected | deferred | executed | failures | refresh | refresh failures |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for item in state["recent_tasks"]:
        lines.append(
            f"| {item.get('timestamp', '')} | {item.get('task', '')} | {item.get('goal_type', '')} | "
            f"{item.get('risk_mode', '')} | {item.get('budget_mode', '')} | {item.get('selected_actions', '')} | "
            f"{item.get('deferred_actions', '')} | {item.get('executed_actions', '')} | {item.get('failures', '')} | "
            f"{item.get('post_refresh_steps', '')} | {item.get('post_refresh_failures', '')} |"
        )
    runbook = state["last_runbook"]
    lines.extend(["", "## Last Runbook", ""])
    if not runbook.get("available"):
        lines.append("No runbook generated yet.")
    else:
        lines.append(f"- Actions: {runbook['actions']}")
        lines.append(f"- Phase counts: `{json.dumps(runbook['phase_counts'], ensure_ascii=False)}`")
        lines.append(f"- Executed actions: {runbook['executed_actions']}")
        lines.append(f"- Execution failures: {runbook['execution_failures']}")
        lines.extend(["", "| phase | status | planned | executed | next_gate |", "|---|---|---:|---:|---|"])
        for item in runbook.get("phase_state", []):
            lines.append(
                f"| {item.get('phase', '')} | {item.get('status', '')} | {item.get('planned_actions', '')} | "
                f"{item.get('executed_actions', '')} | {item.get('next_gate', '')} |"
            )
    machine = state.get("runbook_state_machine", {})
    lines.extend(["", "## Runbook State Machine", ""])
    if not machine.get("available"):
        lines.append("No runbook state machine generated yet.")
    else:
        summary = machine.get("summary", {})
        lines.append(f"- Overall status: `{summary.get('overall_status', '')}`")
        lines.append(f"- Next phase: `{summary.get('next_phase', '')}`")
        lines.append(f"- Next command: `{summary.get('next_command', '')}`")
        lines.extend(
            [
                "",
                "| phase | gate_status | can_execute_now | confirmation_required | attempts |",
                "|---|---|---|---|---:|",
            ]
        )
        for item in machine.get("phases", []):
            lines.append(
                f"| {item.get('phase', '')} | {item.get('gate_status', '')} | {item.get('can_execute_now', '')} | "
                f"{item.get('confirmation_required', '')} | {item.get('attempt_count', '')} |"
            )
    lines.extend(
        [
            "",
            "## Suggested Next Tasks",
            "",
            "| priority | next_task | cooldown_policy_summary | command |",
            "|---|---|---|---|",
        ]
    )
    for item in state["next_tasks"]:
        lines.append(
            f"| {item.get('priority', '')} | {item.get('next_task', '')} | "
            f"{item.get('cooldown_policy_summary', '')} | `{item.get('command', '')}` |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    state = build_state(args.project_root, registry)
    output_dir = args.project_root / "outputs" / "state"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "agent_state.json"
    json_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = write_markdown(args.project_root, state)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Delivery readiness: {state['delivery_readiness']['overall']} ({state['delivery_readiness']['checks']} checks)")


if __name__ == "__main__":
    main()
