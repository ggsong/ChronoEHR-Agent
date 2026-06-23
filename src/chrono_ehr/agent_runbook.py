#!/usr/bin/env python3
"""Build a phased runbook for longer ChronoEHR-Agent work sessions."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from agent_task_router import infer_budget_mode, infer_goal_type, infer_risk_mode, read_json, select_actions, execute_selected
from agent_runbook_state_machine import append_phase_history, build_state as build_runbook_machine_state, write_outputs as write_machine_outputs


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = DEFAULT_PROJECT / "configs" / "agent_action_catalog.json"
PHASE_ORDER = [
    ("phase_1_safe_checks", "safe", "run with --agent-runbook-execute-safe-phase"),
    ("phase_2_expensive_non_model", "expensive", "requires --agent-runbook-execute-expensive-phase --confirm-expensive"),
    ("phase_3_model_requires_confirmation", "model", "requires a later explicit model-running command"),
    ("phase_4_report_deferred", "report", "deferred while polishing the Agent itself"),
]
EXECUTION_COLUMNS = [
    "id",
    "status",
    "risk_level",
    "duration_seconds",
    "returncode",
    "command",
    "detail",
    "phase",
    "confirmation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--execute-safe-phase",
        action="store_true",
        help="Execute only the safe phase. Expensive/model/report phases remain planned.",
    )
    parser.add_argument(
        "--execute-expensive-phase",
        action="store_true",
        help="Execute the expensive non-model phase only when --confirm-expensive is also set.",
    )
    parser.add_argument(
        "--confirm-expensive",
        action="store_true",
        help="Required with --execute-expensive-phase to confirm large-table non-model work.",
    )
    parser.add_argument(
        "--post-phase-refresh",
        action="store_true",
        help="After phase execution, refresh recovery plan, next tasks, and state.",
    )
    return parser.parse_args()


def phase_for_risk(risk_level: str) -> str:
    if risk_level == "safe":
        return "phase_1_safe_checks"
    if risk_level == "expensive":
        return "phase_2_expensive_non_model"
    if risk_level == "model":
        return "phase_3_model_requires_confirmation"
    return "phase_4_report_deferred"


def build_runbook(catalog: dict, task: str) -> tuple[pd.DataFrame, str, str, str]:
    goal_type = infer_goal_type(task)
    risk_mode = infer_risk_mode(task, "auto")
    budget_mode = infer_budget_mode(task)
    actions = select_actions(catalog, goal_type, risk_mode, budget_mode)
    if actions.empty:
        return actions, goal_type, risk_mode, budget_mode
    runbook = actions.copy()
    runbook["phase"] = runbook["risk_level"].map(phase_for_risk)
    runbook["execution_policy"] = runbook["risk_level"].map(
        {
            "safe": "may_execute_with_execute_safe_phase",
            "expensive": "plan_only_requires_user_or_night_run_confirmation",
            "model": "plan_only_requires_explicit_model_confirmation",
            "report": "deferred_while_polishing_agent",
        }
    )
    return runbook, goal_type, risk_mode, budget_mode


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No matching actions."
    display = df[columns].astype(object).where(pd.notna(df[columns]), "")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def build_phase_summary(runbook: pd.DataFrame, execution: pd.DataFrame) -> pd.DataFrame:
    execution = execution if not execution.empty else pd.DataFrame(columns=EXECUTION_COLUMNS)
    rows = []
    for phase, risk_level, gate in PHASE_ORDER:
        planned = runbook[runbook["phase"].eq(phase)] if not runbook.empty and "phase" in runbook else pd.DataFrame()
        executed = (
            execution[execution["phase"].eq(phase) & execution["status"].isin(["PASS", "FAIL"])]
            if not execution.empty and {"phase", "status"}.issubset(execution.columns)
            else pd.DataFrame()
        )
        failures = int((executed["status"] == "FAIL").sum()) if not executed.empty and "status" in executed else 0
        if planned.empty:
            phase_status = "not_applicable"
        elif failures:
            phase_status = "failed_needs_recovery"
        elif len(executed) == len(planned):
            phase_status = "completed"
        elif len(executed) > 0:
            phase_status = "partial"
        elif risk_level == "safe":
            phase_status = "planned_ready"
        elif risk_level == "expensive":
            phase_status = "waiting_for_explicit_confirmation"
        elif risk_level == "model":
            phase_status = "locked_model_confirmation_required"
        else:
            phase_status = "deferred"
        rows.append(
            {
                "phase": phase,
                "risk_level": risk_level,
                "planned_actions": len(planned),
                "executed_actions": len(executed),
                "failed_actions": failures,
                "status": phase_status,
                "next_gate": gate,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    project_root: Path,
    task: str,
    goal_type: str,
    risk_mode: str,
    budget_mode: str,
    runbook: pd.DataFrame,
    execution: pd.DataFrame,
) -> Path:
    table_dir = project_root / "outputs" / "tables"
    report_dir = project_root / "outputs" / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if execution.empty:
        execution = pd.DataFrame(columns=EXECUTION_COLUMNS)
    phase_summary = build_phase_summary(runbook, execution)
    runbook.to_csv(table_dir / "agent_runbook.csv", index=False)
    execution.to_csv(table_dir / "agent_runbook_execution.csv", index=False)
    phase_summary.to_csv(table_dir / "agent_runbook_phase_summary.csv", index=False)
    append_phase_history(project_root, task, goal_type, risk_mode, budget_mode, phase_summary, execution)
    machine_rows, machine_summary = build_runbook_machine_state(project_root, task)
    write_machine_outputs(project_root, machine_rows, machine_summary)
    if execution.empty:
        execution_text = "Not executed."
    else:
        execution_columns = [
            column
            for column in ["phase", "id", "status", "risk_level", "confirmation", "duration_seconds", "returncode", "command", "detail"]
            if column in execution.columns
        ]
        execution_text = markdown_table(execution, execution_columns)
    output = report_dir / "agent_runbook.md"
    output.write_text(
        f"""# Agent Runbook

- User task: `{task}`
- Goal type: `{goal_type}`
- Risk mode: `{risk_mode}`
- Budget mode: `{budget_mode}`
- Boundary: local research workflow runbook only; no medical QA, diagnosis, or treatment recommendation.

## Phased Actions

{markdown_table(runbook, ["phase", "id", "module", "risk_level", "execution_policy", "command", "description"])}

## Safe Phase Execution

{execution_text}

## Phase State

{markdown_table(phase_summary, ["phase", "risk_level", "planned_actions", "executed_actions", "failed_actions", "status", "next_gate"])}

## Notes

- Safe phase can be executed automatically when requested.
- Expensive phase is executed only with explicit `--execute-expensive-phase --confirm-expensive`.
- Model and report phases require explicit later confirmation.
""",
        encoding="utf-8",
    )
    state_dir = project_root / "outputs" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    phase_counts = runbook["phase"].value_counts().to_dict() if not runbook.empty and "phase" in runbook else {}
    executed = int(execution["status"].isin(["PASS", "FAIL"]).sum()) if not execution.empty and "status" in execution else 0
    failures = int((execution["status"] == "FAIL").sum()) if not execution.empty and "status" in execution else 0
    history_row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "task": task,
                "goal_type": goal_type,
                "risk_mode": risk_mode,
                "budget_mode": budget_mode,
                "actions": len(runbook),
                "phase_counts": str(phase_counts),
                "executed_actions": executed,
                "failures": failures,
                "report": str(output.relative_to(project_root)),
            }
        ]
    )
    history_path = state_dir / "agent_runbook_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, history_row], ignore_index=True).tail(100)
    else:
        history = history_row
    history.to_csv(history_path, index=False)
    return output


def annotate_execution(execution: pd.DataFrame, runbook: pd.DataFrame, phase: str, confirmation: str) -> pd.DataFrame:
    if execution.empty:
        return execution
    phase_by_id = runbook.set_index("id")["phase"].to_dict() if "id" in runbook and "phase" in runbook else {}
    execution = execution.copy()
    execution["phase"] = execution["id"].map(phase_by_id).fillna(phase)
    execution["confirmation"] = confirmation
    return execution


def refresh_after_phase(project_root: Path) -> None:
    import subprocess
    import sys

    commands = [
        ["--agent-recovery-plan"],
        ["--agent-next-tasks"],
        ["--agent-state"],
    ]
    for command in commands:
        subprocess.run([sys.executable, str(project_root / "src" / "chrono_ehr" / "run_study.py"), *command], cwd=project_root, check=True)


def main() -> None:
    args = parse_args()
    catalog = read_json(args.catalog)
    runbook, goal_type, risk_mode, budget_mode = build_runbook(catalog, args.task)
    execution = pd.DataFrame()
    if args.execute_safe_phase and not runbook.empty:
        execution = pd.DataFrame(execute_selected(args.project_root, runbook[runbook["risk_level"].eq("safe")]))
        execution = annotate_execution(execution, runbook, "phase_1_safe_checks", "safe_phase")
    if args.execute_expensive_phase:
        if not args.confirm_expensive:
            raise SystemExit("Refusing to execute expensive phase without --confirm-expensive.")
        if not runbook.empty:
            expensive = runbook[runbook["risk_level"].eq("expensive")]
            expensive_execution = pd.DataFrame(execute_selected(args.project_root, expensive, allowed_risk_levels=("expensive",)))
            expensive_execution = annotate_execution(
                expensive_execution,
                runbook,
                "phase_2_expensive_non_model",
                "confirmed_expensive",
            )
            execution = pd.concat([execution, expensive_execution], ignore_index=True)
    output = write_outputs(args.project_root, args.task, goal_type, risk_mode, budget_mode, runbook, execution)
    if args.post_phase_refresh:
        refresh_after_phase(args.project_root)
    print(f"Wrote {output}")
    print(f"Goal type: {goal_type}")
    print(f"Risk mode: {risk_mode}")
    print(f"Budget mode: {budget_mode}")
    print(f"Runbook actions: {len(runbook)}")
    if not execution.empty:
        print(f"Phase executions: {len(execution)}; failures={int((execution['status'] == 'FAIL').sum())}")
        if (execution["status"] == "FAIL").any():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
