#!/usr/bin/env python3
"""Build a machine-readable state machine for Agent runbook phases."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = "我要去睡觉，电脑不关，可以多跑一些但不要重训模型"
PHASE_ORDER = [
    "phase_1_safe_checks",
    "phase_2_expensive_non_model",
    "phase_3_model_requires_confirmation",
    "phase_4_report_deferred",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def safe_task_text(task: str) -> str:
    return task.replace('"', '\\"')


def phase_command(phase: str, task: str) -> str:
    task_text = safe_task_text(task)
    if phase == "phase_1_safe_checks":
        return (
            f'python3 src/chrono_ehr/run_study.py --agent-runbook "{task_text}" '
            "--agent-runbook-execute-safe-phase --agent-runbook-post-phase-refresh"
        )
    if phase == "phase_2_expensive_non_model":
        return (
            f'python3 src/chrono_ehr/run_study.py --agent-runbook "{task_text}" '
            "--agent-runbook-execute-expensive-phase --confirm-expensive --agent-runbook-post-phase-refresh"
        )
    if phase == "phase_3_model_requires_confirmation":
        return "LOCKED: model phase needs a separate explicit model-running confirmation."
    return "DEFERRED: report phase is deferred while polishing the Agent workflow."


def append_phase_history(
    project_root: Path,
    task: str,
    goal_type: str,
    risk_mode: str,
    budget_mode: str,
    phase_summary: pd.DataFrame,
    execution: pd.DataFrame,
) -> Path:
    state_dir = project_root / "outputs" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    output = state_dir / "agent_runbook_phase_history.csv"
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    confirmations_by_phase: dict[str, str] = {}
    if not execution.empty and {"phase", "confirmation"}.issubset(execution.columns):
        for phase, subset in execution.groupby("phase"):
            confirmations = sorted(str(value) for value in subset["confirmation"].dropna().unique())
            confirmations_by_phase[str(phase)] = ",".join(confirmations)
    rows = []
    for item in phase_summary.to_dict(orient="records"):
        executed_actions = int(item.get("executed_actions", 0))
        failed_actions = int(item.get("failed_actions", 0))
        rows.append(
            {
                "timestamp": timestamp,
                "task": task,
                "goal_type": goal_type,
                "risk_mode": risk_mode,
                "budget_mode": budget_mode,
                "phase": item.get("phase", ""),
                "risk_level": item.get("risk_level", ""),
                "planned_actions": item.get("planned_actions", 0),
                "executed_actions": executed_actions,
                "failed_actions": failed_actions,
                "phase_status": item.get("status", ""),
                "event_type": "execution" if executed_actions or failed_actions else "plan",
                "confirmation": confirmations_by_phase.get(str(item.get("phase", "")), ""),
            }
        )
    new_history = pd.DataFrame(rows)
    if output.exists():
        history = read_csv(output)
        history = pd.concat([history, new_history], ignore_index=True).tail(500)
    else:
        history = new_history
    history.to_csv(output, index=False)
    return output


def history_attempts(history: pd.DataFrame, phase: str) -> tuple[int, str]:
    if history.empty or "phase" not in history or "event_type" not in history:
        return 0, ""
    subset = history[(history["phase"].astype(str) == phase) & (history["event_type"].astype(str) == "execution")]
    if subset.empty:
        return 0, ""
    last_attempt_at = str(subset.iloc[-1].get("timestamp", ""))
    return len(subset), last_attempt_at


def gate_state(phase: str, risk_level: str, phase_status: str, failed_actions: int) -> tuple[str, str, str, str]:
    if phase_status == "completed":
        return "complete", "NO", "NO", "Phase completed."
    if failed_actions > 0 or phase_status == "failed_needs_recovery":
        return "recovery_required", "NO", "NO", "Phase has failed actions; run recovery before continuing."
    if phase == "phase_1_safe_checks":
        return "open", "YES", "NO", "Safe local checks can run without extra confirmation."
    if phase == "phase_2_expensive_non_model":
        return "confirmation_required", "YES_WITH_CONFIRMATION", "YES", "Large-table non-model work requires explicit confirmation."
    if phase == "phase_3_model_requires_confirmation":
        return "locked", "NO", "YES", "Model execution is outside the current automatic runbook."
    if phase == "phase_4_report_deferred":
        return "deferred", "NO", "YES", "Report generation is not the current Agent-polishing priority."
    if phase_status == "not_applicable":
        return "not_applicable", "NO", "NO", "No planned actions for this phase."
    return "unknown", "NO", "NO", "State machine could not classify this phase."


def build_state(project_root: Path, task: str = DEFAULT_TASK) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_summary = read_csv(project_root / "outputs" / "tables" / "agent_runbook_phase_summary.csv")
    execution = read_csv(project_root / "outputs" / "tables" / "agent_runbook_execution.csv")
    history = read_csv(project_root / "outputs" / "state" / "agent_runbook_phase_history.csv")
    if phase_summary.empty:
        return [], {
            "overall_status": "NO_RUNBOOK",
            "next_phase": "",
            "next_command": "",
            "blocked_reason": "No runbook phase summary found.",
        }

    rows = []
    for phase in PHASE_ORDER:
        subset = phase_summary[phase_summary["phase"].astype(str) == phase] if "phase" in phase_summary else pd.DataFrame()
        if subset.empty:
            continue
        item = subset.iloc[0].to_dict()
        risk_level = str(item.get("risk_level", ""))
        phase_status = str(item.get("status", ""))
        planned_actions = int(item.get("planned_actions", 0))
        executed_actions = int(item.get("executed_actions", 0))
        failed_actions = int(item.get("failed_actions", 0))
        attempt_count, last_attempt_at = history_attempts(history, phase)
        gate, can_execute_now, confirmation_required, reason = gate_state(phase, risk_level, phase_status, failed_actions)
        confirmation_received = "NO"
        if not execution.empty and {"phase", "confirmation"}.issubset(execution.columns):
            phase_execution = execution[execution["phase"].astype(str) == phase]
            if not phase_execution.empty and phase_execution["confirmation"].astype(str).str.len().gt(0).any():
                confirmation_received = "YES"
        next_command = phase_command(phase, task) if planned_actions else ""
        if gate in {"complete", "not_applicable"}:
            next_command = ""
        if gate == "recovery_required":
            next_command = "python3 src/chrono_ehr/run_study.py --agent-recovery-plan --agent-recovery-execute-safe"
        rows.append(
            {
                "phase": phase,
                "risk_level": risk_level,
                "planned_actions": planned_actions,
                "executed_actions": executed_actions,
                "failed_actions": failed_actions,
                "phase_status": phase_status,
                "gate_status": gate,
                "can_execute_now": can_execute_now,
                "confirmation_required": confirmation_required,
                "confirmation_received": confirmation_received,
                "attempt_count": attempt_count,
                "last_attempt_at": last_attempt_at,
                "next_allowed_command": next_command,
                "reason": reason,
            }
        )

    next_row = next((row for row in rows if row["gate_status"] in {"recovery_required", "open", "confirmation_required"}), None)
    locked = [row for row in rows if row["gate_status"] in {"locked", "deferred"} and row["planned_actions"]]
    if next_row:
        overall = "READY"
        next_phase = next_row["phase"]
        next_command = next_row["next_allowed_command"]
        blocked_reason = next_row["reason"]
    elif locked:
        overall = "BLOCKED"
        next_phase = locked[0]["phase"]
        next_command = locked[0]["next_allowed_command"]
        blocked_reason = locked[0]["reason"]
    else:
        overall = "COMPLETE_OR_NOT_APPLICABLE"
        next_phase = ""
        next_command = ""
        blocked_reason = "No runnable phase remains."
    summary = {
        "overall_status": overall,
        "next_phase": next_phase,
        "next_command": next_command,
        "blocked_reason": blocked_reason,
    }
    return rows, summary


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "phase",
        "gate_status",
        "can_execute_now",
        "confirmation_required",
        "attempt_count",
        "next_allowed_command",
        "reason",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    table_path = project_root / "outputs" / "tables" / "agent_runbook_state_machine.csv"
    report_path = project_root / "outputs" / "reports" / "agent_runbook_state_machine.md"
    json_path = project_root / "outputs" / "state" / "agent_runbook_state_machine.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(table_path, index=False)
    json_path.write_text(json.dumps({"summary": summary, "phases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        f"""# Agent Runbook State Machine

- Overall status: `{summary.get("overall_status", "")}`
- Next phase: `{summary.get("next_phase", "")}`
- Boundary: local research workflow phase control only; no medical QA, diagnosis, or treatment recommendation.

## Phase Gates

{markdown_table(df) if not df.empty else "No runbook state available."}
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    rows, summary = build_state(args.project_root, args.task)
    report = write_outputs(args.project_root, rows, summary)
    print(f"Wrote {report}")
    print(f"Runbook state: {summary.get('overall_status', '')}")
    print(f"Next phase: {summary.get('next_phase', '')}")


if __name__ == "__main__":
    main()
