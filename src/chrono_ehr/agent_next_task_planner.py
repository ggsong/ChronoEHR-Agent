#!/usr/bin/env python3
"""Recommend the next natural-language Agent task from current local state."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agent_cooldown_fingerprint import cooldown_fingerprint


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def last_success_for(history: pd.DataFrame, scenario_id: str, command: str, fingerprint: str) -> dict[str, str]:
    if history.empty or not command:
        return {}
    required = {"scenario_id", "command", "execution_status", "run_id", "started_at"}
    if not required.issubset(history.columns):
        return {}
    matches = history[
        history["scenario_id"].astype(str).eq(str(scenario_id))
        & history["command"].astype(str).eq(str(command))
        & history["execution_status"].astype(str).eq("PASS")
    ]
    if matches.empty:
        return {}
    latest = matches.tail(1).fillna("").to_dict(orient="records")[0]
    last_fingerprint = str(latest.get("cooldown_fingerprint", ""))
    if fingerprint and last_fingerprint:
        fingerprint_status = "matched_success" if fingerprint == last_fingerprint else "changed_since_success"
    elif fingerprint:
        fingerprint_status = "legacy_success_missing_fingerprint"
    else:
        fingerprint_status = "fingerprint_unavailable"
    return {
        "last_success_run_id": str(latest.get("run_id", "")),
        "last_success_at": str(latest.get("started_at", "")),
        "last_success_fingerprint": last_fingerprint,
        "cooldown_fingerprint_status": fingerprint_status,
    }


def cooldown_policy_summary(
    execution_mode: str,
    fingerprint_status: str,
    last_success_run_id: str,
    suggested_safe_refresh_command: str,
) -> str:
    if execution_mode == "manual_confirmation_required":
        return "Manual confirmation required; cooldown is not applied to this item."
    if execution_mode == "recently_completed" and fingerprint_status == "matched_success":
        return (
            "Cooldown active: matching safe-auto PASS history found for the current code/config fingerprint; "
            "rerun only after Agent code/config changes or explicit user request."
        )
    if last_success_run_id and fingerprint_status == "changed_since_success":
        return "Safe rerun allowed: previous PASS exists, but the cooldown fingerprint changed since that success."
    if last_success_run_id and fingerprint_status == "legacy_success_missing_fingerprint":
        return "Safe rerun allowed: previous PASS exists, but that legacy success did not record a cooldown fingerprint."
    if suggested_safe_refresh_command:
        return "Safe-auto open: no matching PASS history exists for the current cooldown fingerprint."
    return "Plan-only item; cooldown is not applied."


def build_next_tasks(project_root: Path) -> pd.DataFrame:
    self_check = read_csv(project_root / "outputs" / "tables" / "agent_self_check.csv")
    recovery = read_csv(project_root / "outputs" / "tables" / "agent_recovery_plan.csv")
    runbook = read_csv(project_root / "outputs" / "tables" / "agent_runbook.csv")
    runbook_execution = read_csv(project_root / "outputs" / "tables" / "agent_runbook_execution.csv")
    phase_summary = read_csv(project_root / "outputs" / "tables" / "agent_runbook_phase_summary.csv")
    state_machine = read_csv(project_root / "outputs" / "tables" / "agent_runbook_state_machine.csv")
    retry_plan = read_csv(project_root / "outputs" / "tables" / "agent_runbook_retry_plan.csv")
    external = read_csv(project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv")
    scenario_examples = read_csv(project_root / "outputs" / "tables" / "agent_task_scenario_examples.csv")
    queue_history = read_csv(project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv")

    rows = []
    known_scenarios = (
        set(scenario_examples["scenario_id"].astype(str))
        if not scenario_examples.empty and "scenario_id" in scenario_examples
        else {
            "low_quota_self_check",
            "night_run_safe_plus_deferred",
            "external_readiness_first",
            "validation_first",
            "model_or_report_explicit",
        }
    )

    def add_task(
        priority: str,
        next_task: str,
        command: str,
        reason: str,
        scenario_id: str,
        suggested_agent_task: str,
        execution_mode: str | None = None,
        execution_boundary: str | None = None,
    ) -> None:
        if scenario_id not in known_scenarios:
            scenario_id = "validation_first"
        quoted_task = shlex.quote(suggested_agent_task)
        suggested_agent_command = "python3 src/chrono_ehr/run_study.py --agent-task " + quoted_task
        if execution_mode is None:
            execution_mode = "manual_confirmation_required" if "--confirm-expensive" in command else "safe_auto_allowed"
        if execution_boundary is None:
            execution_boundary = (
                "Requires explicit confirmation because the recommended command crosses an expensive-phase gate."
                if execution_mode == "manual_confirmation_required"
                else "Safe actions may be executed with post-run refresh; expensive/model/report actions remain deferred."
            )
        suggested_safe_refresh_command = (
            ""
            if execution_mode == "manual_confirmation_required"
            else suggested_agent_command + " --agent-task-execute-safe --agent-task-post-run-refresh"
        )
        fingerprint = ""
        fingerprint_status = "not_applicable"
        missing_inputs = ""
        input_paths = ""
        if suggested_safe_refresh_command:
            fingerprint_info = cooldown_fingerprint(
                project_root,
                scenario_id,
                suggested_agent_task,
                suggested_safe_refresh_command,
            )
            fingerprint = fingerprint_info["cooldown_fingerprint"]
            missing_inputs = fingerprint_info["cooldown_missing_inputs"]
            input_paths = fingerprint_info["cooldown_fingerprint_inputs"]
            fingerprint_status = "no_success"
        success = last_success_for(queue_history, scenario_id, suggested_safe_refresh_command, fingerprint)
        completion_status = "open"
        last_success_run_id = ""
        last_success_at = ""
        last_success_fingerprint = ""
        cooldown_reason = ""
        if execution_mode == "manual_confirmation_required":
            completion_status = "waiting_confirmation"
        elif success and success.get("cooldown_fingerprint_status") == "matched_success":
            execution_mode = "recently_completed"
            execution_boundary = "Already completed as a safe-auto queue item; do not repeat unless inputs change or the user asks to rerun."
            suggested_safe_refresh_command = ""
            completion_status = "completed"
            last_success_run_id = success.get("last_success_run_id", "")
            last_success_at = success.get("last_success_at", "")
            last_success_fingerprint = success.get("last_success_fingerprint", "")
            fingerprint_status = "matched_success"
            cooldown_reason = "safe-auto command already has a recorded PASS with matching cooldown fingerprint"
        elif success:
            last_success_run_id = success.get("last_success_run_id", "")
            last_success_at = success.get("last_success_at", "")
            last_success_fingerprint = success.get("last_success_fingerprint", "")
            fingerprint_status = success.get("cooldown_fingerprint_status", "changed_since_success")
            cooldown_reason = "previous PASS exists, but cooldown inputs changed or legacy success lacks a fingerprint"
        cooldown_summary = cooldown_policy_summary(
            execution_mode,
            fingerprint_status,
            last_success_run_id,
            suggested_safe_refresh_command,
        )
        rows.append(
            {
                "priority": priority,
                "scenario_id": scenario_id,
                "next_task": next_task,
                "execution_mode": execution_mode,
                "execution_boundary": execution_boundary,
                "completion_status": completion_status,
                "last_success_run_id": last_success_run_id,
                "last_success_at": last_success_at,
                "cooldown_fingerprint": fingerprint,
                "last_success_fingerprint": last_success_fingerprint,
                "cooldown_fingerprint_status": fingerprint_status,
                "cooldown_reason": cooldown_reason,
                "cooldown_policy_summary": cooldown_summary,
                "cooldown_missing_inputs": missing_inputs,
                "cooldown_fingerprint_inputs": input_paths,
                "suggested_agent_task": suggested_agent_task,
                "suggested_agent_command": suggested_agent_command,
                "suggested_safe_refresh_command": suggested_safe_refresh_command,
                "command": command,
                "reason": reason,
            }
        )

    if self_check.empty or (not self_check.empty and (self_check["status"] != "PASS").any()):
        add_task(
            "P1",
            "运行 agent 自检并生成恢复计划",
            "python3 src/chrono_ehr/run_study.py --agent-self-check && python3 src/chrono_ehr/run_study.py --agent-recovery-plan",
            "Agent self-check is missing or has failures.",
            "validation_first",
            "先运行轻量验证和恢复计划，不要跑模型",
        )
        return pd.DataFrame(rows)

    if not recovery.empty and not (len(recovery) == 1 and str(recovery.iloc[0]["failed_item"]) == "none"):
        add_task(
            "P1",
            "执行安全恢复动作",
            "python3 src/chrono_ehr/run_study.py --agent-recovery-plan --agent-recovery-execute-safe",
            "Recovery plan contains failed items.",
            "validation_first",
            "执行安全恢复动作，然后刷新 Agent 状态",
        )
        return pd.DataFrame(rows)

    if not retry_plan.empty and {"retry_status", "retry_command", "priority", "phase"}.issubset(retry_plan.columns):
        actionable = retry_plan[
            retry_plan["retry_status"].astype(str).isin(["RECOVERY_FIRST", "READY_TO_RETRY", "CONFIRMATION_REQUIRED", "REBUILD_STATE"])
        ].copy()
        actionable = actionable[actionable["retry_command"].astype(str).str.len().gt(0)]
        if not actionable.empty:
            item = actionable.iloc[0]
            status = str(item["retry_status"])
            label = {
                "RECOVERY_FIRST": "执行 runbook recovery-first 恢复动作",
                "READY_TO_RETRY": "重试 runbook safe phase",
                "CONFIRMATION_REQUIRED": "确认后重试 runbook expensive phase",
                "REBUILD_STATE": "重建 runbook state machine",
            }.get(status, "继续 runbook retry/resume")
            scenario_id = (
                "night_run_safe_plus_deferred"
                if status in {"CONFIRMATION_REQUIRED", "READY_TO_RETRY"} or "EXPENSIVE" in status
                else "validation_first"
            )
            add_task(
                str(item.get("priority", "P2")),
                label,
                str(item["retry_command"]),
                f"Retry planner selected {item.get('phase', '')} ({status}).",
                scenario_id,
                "我要去睡觉，继续 runbook，但不要自动重训模型",
                "manual_confirmation_required" if status == "CONFIRMATION_REQUIRED" else "safe_auto_allowed",
            )

    if not state_machine.empty and {"gate_status", "next_allowed_command", "phase"}.issubset(state_machine.columns):
        runnable = state_machine[state_machine["gate_status"].astype(str).isin(["recovery_required", "open", "confirmation_required"])]
        if not runnable.empty:
            next_phase = str(runnable.iloc[0]["phase"])
            gate_status = str(runnable.iloc[0]["gate_status"])
            priority = "P1" if gate_status in {"recovery_required", "open"} else "P2"
            label = {
                "recovery_required": "执行 runbook 阶段恢复动作",
                "open": "执行 runbook safe phase",
                "confirmation_required": "确认是否执行夜间 runbook 的 expensive phase",
            }.get(gate_status, "继续 runbook 下一阶段")
            scenario_id = "night_run_safe_plus_deferred" if gate_status == "confirmation_required" else "validation_first"
            add_task(
                priority,
                label,
                str(runnable.iloc[0]["next_allowed_command"]),
                f"Runbook state machine next phase is {next_phase} ({gate_status}).",
                scenario_id,
                "继续 runbook 阶段执行，但保留高风险动作到确认门",
                "manual_confirmation_required" if gate_status == "confirmation_required" else "safe_auto_allowed",
            )

    elif not runbook.empty and "risk_level" in runbook:
        expensive = runbook[runbook["risk_level"].eq("expensive")]
        expensive_phase = (
            phase_summary[phase_summary["risk_level"].eq("expensive")]
            if not phase_summary.empty and "risk_level" in phase_summary
            else pd.DataFrame()
        )
        expensive_executed = (
            not expensive_phase.empty
            and "status" in expensive_phase
            and expensive_phase["status"].astype(str).isin(["completed", "partial", "failed_needs_recovery"]).any()
        )
        if expensive_phase.empty:
            expensive_executed = (
                not runbook_execution.empty
                and "risk_level" in runbook_execution
                and "status" in runbook_execution
                and not runbook_execution[
                    runbook_execution["risk_level"].eq("expensive") & runbook_execution["status"].isin(["PASS", "FAIL"])
                ].empty
            )
        if not expensive.empty and not expensive_executed:
            add_task(
                "P2",
                "确认是否执行夜间 runbook 的 expensive phase",
                "python3 src/chrono_ehr/run_study.py --agent-runbook \"我要去睡觉，电脑不关，可以多跑一些但不要重训模型\" --agent-runbook-execute-expensive-phase --confirm-expensive --agent-runbook-post-phase-refresh",
                f"Last runbook has {len(expensive)} planned expensive non-model actions.",
                "night_run_safe_plus_deferred",
                "我要去睡觉，电脑不关，可以多跑一些但不要重训模型",
                "manual_confirmation_required",
            )
        elif expensive_executed:
            add_task(
                "P2",
                "检查 expensive phase 执行后是否需要 recovery",
                "python3 src/chrono_ehr/run_study.py --agent-recovery-plan && python3 src/chrono_ehr/run_study.py --agent-state",
                "Last runbook has executed expensive non-model actions.",
                "validation_first",
                "检查刚才长任务执行后有没有失败，并刷新 Agent 状态",
            )

    if not external.empty:
        pending = external[external["local_status"].astype(str).str.contains("DATA_PENDING", na=False)]
        if not pending.empty:
            pending_names = "/".join(pending["dataset"].astype(str).tolist())
            add_task(
                "P3",
                f"复查 {pending_names} 数据是否已经下载到固定路径",
                "python3 src/chrono_ehr/run_study.py --external-readiness-summary",
                f"External datasets remain data-pending: {pending_names}.",
                "external_readiness_first",
                f"复查 {pending_names} 外部数据 readiness，不要重训模型",
            )

    add_task(
        "P3",
        "继续完善 Agent 控制层：扩展本地审计、队列、状态摘要和恢复记忆",
        "python3 src/chrono_ehr/run_study.py --agent-task \"先完善 Agent 控制层，不要做汇报材料\"",
        "All current health checks pass.",
        "agent_control_focus",
        "先完善 Agent 控制层，不要做汇报材料",
    )
    tasks = pd.DataFrame(rows)
    if not tasks.empty and "command" in tasks:
        tasks = tasks.drop_duplicates(subset=["command"], keep="first").reset_index(drop=True)
    return tasks


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "priority",
        "scenario_id",
        "next_task",
        "execution_mode",
        "execution_boundary",
        "completion_status",
        "last_success_run_id",
        "last_success_at",
        "cooldown_fingerprint",
        "last_success_fingerprint",
        "cooldown_fingerprint_status",
        "cooldown_reason",
        "cooldown_policy_summary",
        "cooldown_missing_inputs",
        "suggested_agent_task",
        "suggested_agent_command",
        "suggested_safe_refresh_command",
        "command",
        "reason",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    display = df[columns].astype(object).where(pd.notna(df[columns]), "")
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tasks = build_next_tasks(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "agent_next_tasks.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_next_tasks.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Next Tasks

- Items: {len(tasks)}
- Boundary: local research workflow planning only; no medical QA, diagnosis, or treatment recommendation.

## Next Task Table

{markdown_table(tasks)}
""",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    print(tasks[["priority", "next_task"]].to_string(index=False))


if __name__ == "__main__":
    main()
