#!/usr/bin/env python3
"""Validate Agent next-task recommendations and safety boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agent_task_router import infer_budget_mode, infer_goal_type, infer_risk_mode, infer_task_scenario


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


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path) -> pd.DataFrame:
    tasks_path = project_root / "outputs" / "tables" / "agent_next_tasks.csv"
    external_path = project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv"
    scenarios_path = project_root / "outputs" / "tables" / "agent_task_scenario_library.csv"
    tasks = read_csv(tasks_path)
    external = read_csv(external_path)
    scenarios = read_csv(scenarios_path)
    rows = [row("next_tasks_exist", "PASS" if not tasks.empty else "FAIL", str(tasks_path), f"rows={len(tasks)}")]
    if tasks.empty:
        return pd.DataFrame(rows)

    required = {
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
    }
    missing = sorted(required - set(tasks.columns))
    rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(tasks_path), "missing=" + ",".join(missing)))

    scenario_ids = set(scenarios["scenario_id"].astype(str)) if not scenarios.empty and "scenario_id" in scenarios else set()
    task_scenarios = set(tasks["scenario_id"].astype(str)) if "scenario_id" in tasks else set()
    unknown_scenarios = sorted(task_scenarios - scenario_ids)
    rows.append(
        row(
            "next_task_scenarios_known",
            "PASS" if scenario_ids and not unknown_scenarios else "FAIL",
            f"{tasks_path}; {scenarios_path}",
            "unknown=" + ",".join(unknown_scenarios),
        )
    )

    empty_suggested = int(tasks["suggested_agent_task"].fillna("").astype(str).eq("").sum()) if "suggested_agent_task" in tasks else len(tasks)
    rows.append(
        row(
            "suggested_agent_tasks_present",
            "PASS" if empty_suggested == 0 else "FAIL",
            str(tasks_path),
            f"empty={empty_suggested}",
        )
    )
    empty_suggested_commands = int(tasks["suggested_agent_command"].fillna("").astype(str).eq("").sum()) if "suggested_agent_command" in tasks else len(tasks)
    rows.append(
        row(
            "suggested_agent_commands_present",
            "PASS" if empty_suggested_commands == 0 else "FAIL",
            str(tasks_path),
            f"empty={empty_suggested_commands}",
        )
    )

    allowed_modes = {"safe_auto_allowed", "manual_confirmation_required", "plan_only", "recently_completed"}
    modes = set(tasks["execution_mode"].astype(str)) if "execution_mode" in tasks else set()
    rows.append(
        row(
            "execution_modes_known",
            "PASS" if modes and modes.issubset(allowed_modes) else "FAIL",
            str(tasks_path),
            "modes=" + ",".join(sorted(modes)),
        )
    )

    if {"execution_mode", "suggested_safe_refresh_command"}.issubset(tasks.columns):
        safe_auto = tasks[tasks["execution_mode"].astype(str).eq("safe_auto_allowed")]
        manual = tasks[tasks["execution_mode"].astype(str).eq("manual_confirmation_required")]
        recently_completed = tasks[tasks["execution_mode"].astype(str).eq("recently_completed")]
        safe_refresh_ok = (
            safe_auto.empty
            or (
                safe_auto["suggested_safe_refresh_command"].fillna("").astype(str).str.contains("--agent-task-execute-safe", na=False).all()
                and safe_auto["suggested_safe_refresh_command"].fillna("").astype(str).str.contains("--agent-task-post-run-refresh", na=False).all()
            )
        )
        manual_refresh_blank = manual.empty or manual["suggested_safe_refresh_command"].fillna("").astype(str).eq("").all()
        rows.append(
            row(
                "safe_auto_recommendations_have_refresh_command",
                "PASS" if safe_refresh_ok else "FAIL",
                str(tasks_path),
                f"safe_auto_rows={len(safe_auto)}",
            )
        )
        rows.append(
            row(
                "manual_confirmation_has_no_safe_refresh_command",
                "PASS" if manual_refresh_blank else "FAIL",
                str(tasks_path),
                f"manual_rows={len(manual)}",
            )
        )
        completed_refresh_blank = recently_completed.empty or recently_completed["suggested_safe_refresh_command"].fillna("").astype(str).eq("").all()
        rows.append(
            row(
                "recently_completed_has_no_safe_refresh_command",
                "PASS" if completed_refresh_blank else "FAIL",
                str(tasks_path),
                f"recently_completed_rows={len(recently_completed)}",
            )
        )
        completed_has_memory = (
            recently_completed.empty
            or (
                recently_completed["last_success_run_id"].fillna("").astype(str).str.len().gt(0).all()
                and recently_completed["last_success_at"].fillna("").astype(str).str.len().gt(0).all()
                and recently_completed["cooldown_fingerprint"].fillna("").astype(str).str.len().gt(0).all()
                and recently_completed["last_success_fingerprint"].fillna("").astype(str).str.len().gt(0).all()
                and recently_completed["cooldown_fingerprint_status"].fillna("").astype(str).eq("matched_success").all()
                and recently_completed["cooldown_reason"].fillna("").astype(str).str.len().gt(0).all()
            )
        )
        rows.append(
            row(
                "recently_completed_has_success_memory",
                "PASS" if completed_has_memory else "FAIL",
                str(tasks_path),
                f"recently_completed_rows={len(recently_completed)}",
            )
        )
        summary_present = tasks["cooldown_policy_summary"].fillna("").astype(str).str.len().gt(0).all()
        rows.append(
            row(
                "cooldown_policy_summary_present",
                "PASS" if summary_present else "FAIL",
                str(tasks_path),
                f"rows={len(tasks)}",
            )
        )
        completed_summary_ok = (
            recently_completed.empty
            or (
                recently_completed["cooldown_policy_summary"]
                .fillna("")
                .astype(str)
                .str.contains("Cooldown active", case=False, na=False)
                .all()
                and recently_completed["cooldown_policy_summary"]
                .fillna("")
                .astype(str)
                .str.contains("matching safe-auto PASS", case=False, na=False)
                .all()
            )
        )
        rows.append(
            row(
                "recently_completed_summary_explains_cooldown",
                "PASS" if completed_summary_ok else "FAIL",
                str(tasks_path),
                f"recently_completed_rows={len(recently_completed)}",
            )
        )
        safe_fingerprints = (
            safe_auto.empty
            or (
                safe_auto["cooldown_fingerprint"].fillna("").astype(str).str.len().ge(32).all()
                and safe_auto["cooldown_fingerprint_status"].fillna("").astype(str).ne("").all()
            )
        )
        rows.append(
            row(
                "safe_auto_recommendations_have_cooldown_fingerprint",
                "PASS" if safe_fingerprints else "FAIL",
                str(tasks_path),
                f"safe_auto_rows={len(safe_auto)}",
            )
        )
        invalidated_success_open = tasks[
            tasks["execution_mode"].astype(str).eq("safe_auto_allowed")
            & tasks["last_success_run_id"].fillna("").astype(str).str.len().gt(0)
            & tasks["cooldown_fingerprint_status"].fillna("").astype(str).isin(
                ["changed_since_success", "legacy_success_missing_fingerprint"]
            )
        ]
        rows.append(
            row(
                "changed_or_legacy_success_keeps_task_open",
                "PASS"
                if invalidated_success_open.empty
                or invalidated_success_open["suggested_safe_refresh_command"].fillna("").astype(str).str.len().gt(0).all()
                else "FAIL",
                str(tasks_path),
                f"invalidated_success_open_rows={len(invalidated_success_open)}",
            )
        )
        invalidated_summary_ok = (
            invalidated_success_open.empty
            or (
                invalidated_success_open["cooldown_policy_summary"]
                .fillna("")
                .astype(str)
                .str.contains("Safe rerun allowed", case=False, na=False)
                .all()
                and invalidated_success_open["cooldown_policy_summary"]
                .fillna("")
                .astype(str)
                .str.contains("changed|legacy", case=False, na=False, regex=True)
                .all()
            )
        )
        rows.append(
            row(
                "invalidated_success_summary_explains_rerun",
                "PASS" if invalidated_summary_ok else "FAIL",
                str(tasks_path),
                f"invalidated_success_open_rows={len(invalidated_success_open)}",
            )
        )
        manual_summary_ok = (
            manual.empty
            or manual["cooldown_policy_summary"]
            .fillna("")
            .astype(str)
            .str.contains("Manual confirmation", case=False, na=False)
            .all()
        )
        rows.append(
            row(
                "manual_summary_explains_no_cooldown",
                "PASS" if manual_summary_ok else "FAIL",
                str(tasks_path),
                f"manual_rows={len(manual)}",
            )
        )

    if {"scenario_id", "suggested_agent_task"}.issubset(tasks.columns):
        mismatches = []
        for item in tasks[["scenario_id", "suggested_agent_task"]].itertuples(index=False):
            task = str(item.suggested_agent_task)
            goal = infer_goal_type(task)
            risk = infer_risk_mode(task, "auto")
            budget = infer_budget_mode(task)
            inferred = infer_task_scenario(task, goal, risk, budget)["scenario_id"]
            if inferred != str(item.scenario_id):
                mismatches.append(f"{item.scenario_id}->{inferred}:{task}")
        rows.append(
            row(
                "suggested_agent_tasks_route_to_declared_scenario",
                "PASS" if not mismatches else "FAIL",
                str(tasks_path),
                "mismatches=" + "; ".join(mismatches[:3]),
            )
        )

    commands = tasks["command"].astype(str) if "command" in tasks else pd.Series(dtype=str)
    duplicate_commands = int(commands.duplicated().sum()) if not commands.empty else 0
    rows.append(
        row(
            "no_duplicate_next_task_commands",
            "PASS" if duplicate_commands == 0 else "FAIL",
            str(tasks_path),
            f"duplicate_commands={duplicate_commands}",
        )
    )
    history = read_csv(project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv")
    if not history.empty and {"queue_id", "scenario_id", "execution_status"}.issubset(history.columns):
        q003_success = history[
            history["queue_id"].astype(str).eq("Q003")
            & history["scenario_id"].astype(str).eq("agent_control_focus")
            & history["execution_status"].astype(str).eq("PASS")
        ]
        agent_control_rows = tasks[tasks["scenario_id"].astype(str).eq("agent_control_focus")]
        cooled = agent_control_rows[agent_control_rows["execution_mode"].astype(str).eq("recently_completed")]
        matching_success = pd.DataFrame()
        if not agent_control_rows.empty and "cooldown_fingerprint" in agent_control_rows and "cooldown_fingerprint" in history:
            current_fingerprints = set(agent_control_rows["cooldown_fingerprint"].fillna("").astype(str))
            matching_success = q003_success[q003_success["cooldown_fingerprint"].fillna("").astype(str).isin(current_fingerprints)]
        rows.append(
            row(
                "agent_control_success_cools_down_next_task",
                "PASS" if matching_success.empty or not cooled.empty else "FAIL",
                str(tasks_path),
                f"q003_success_rows={len(q003_success)}; matching_fingerprint_success_rows={len(matching_success)}; cooled_rows={len(cooled)}",
            )
        )
    safe_rows = tasks[
        tasks["next_task"].astype(str).str.contains("safe phase", case=False, na=False)
        | commands.str.contains("--agent-runbook-execute-safe-phase", na=False)
    ]
    if not safe_rows.empty:
        rows.append(
            row(
                "safe_recommendation_executes_safe_phase",
                "PASS"
                if safe_rows["command"].astype(str).str.contains("--agent-runbook-execute-safe-phase", na=False).all()
                else "FAIL",
                str(tasks_path),
                f"rows={len(safe_rows)}",
            )
        )
        rows.append(
            row(
                "safe_recommendation_refreshes_state",
                "PASS"
                if safe_rows["command"].astype(str).str.contains("--agent-runbook-post-phase-refresh", na=False).all()
                else "FAIL",
                str(tasks_path),
                f"rows={len(safe_rows)}",
            )
        )
    expensive_rows = tasks[
        tasks["next_task"].astype(str).str.contains("expensive phase", case=False, na=False)
        | commands.str.contains("--agent-runbook-execute-expensive-phase", na=False)
    ]
    if not expensive_rows.empty:
        complete_confirmation = expensive_rows["command"].astype(str).str.contains("--confirm-expensive", na=False).all()
        post_refresh = expensive_rows["command"].astype(str).str.contains("--agent-runbook-post-phase-refresh", na=False).all()
        manual_mode = (
            "execution_mode" in expensive_rows
            and expensive_rows["execution_mode"].astype(str).eq("manual_confirmation_required").all()
        )
        rows.append(
            row(
                "expensive_recommendation_requires_confirmation",
                "PASS" if complete_confirmation else "FAIL",
                str(tasks_path),
                f"rows={len(expensive_rows)}",
            )
        )
        rows.append(
            row(
                "expensive_recommendation_is_manual_mode",
                "PASS" if manual_mode else "FAIL",
                str(tasks_path),
                f"rows={len(expensive_rows)}",
            )
        )
        rows.append(
            row(
                "expensive_recommendation_refreshes_state",
                "PASS" if post_refresh else "FAIL",
                str(tasks_path),
                f"rows={len(expensive_rows)}",
            )
        )

    risky_auto = commands[
        commands.str.contains("--confirm-model|--run-report-phase|--agent-runbook-execute-model", regex=True, na=False)
    ]
    rows.append(
        row(
            "no_model_or_report_auto_execution",
            "PASS" if risky_auto.empty else "FAIL",
            str(tasks_path),
            f"risky_commands={len(risky_auto)}",
        )
    )
    if not external.empty and {"dataset", "local_status"}.issubset(external.columns):
        status_by_dataset = dict(zip(external["dataset"].astype(str), external["local_status"].astype(str)))
        task_text = " ".join(tasks["next_task"].astype(str).tolist() + tasks["reason"].astype(str).tolist())
        has_recovery_priority = tasks["priority"].astype(str).eq("P1").any() and (
            task_text.find("self-check") >= 0 or task_text.find("恢复") >= 0 or task_text.find("Recovery") >= 0
        )
        eicu_pending = status_by_dataset.get("eICU") == "DATA_PENDING"
        charls_pending = status_by_dataset.get("CHARLS") == "DATA_PENDING"
        rows.append(
            row(
                "pending_dataset_text_does_not_include_nonpending_eicu",
                "PASS" if eicu_pending or "eICU/CHARLS" not in task_text else "FAIL",
                str(tasks_path),
                f"eICU={status_by_dataset.get('eICU', '')}; task_text_contains_eICU_CHARLS={'eICU/CHARLS' in task_text}",
            )
        )
        rows.append(
            row(
                "pending_charls_task_present_when_charls_pending",
                "PASS" if not charls_pending or "CHARLS" in task_text or has_recovery_priority else "FAIL",
                str(tasks_path),
                f"CHARLS={status_by_dataset.get('CHARLS', '')}; recovery_priority={has_recovery_priority}",
            )
        )
        if charls_pending and "scenario_id" in tasks:
            charls_rows = tasks[
                tasks["next_task"].astype(str).str.contains("CHARLS", na=False)
                | tasks["reason"].astype(str).str.contains("CHARLS", na=False)
                | tasks["suggested_agent_task"].astype(str).str.contains("CHARLS", na=False)
            ]
            rows.append(
                row(
                    "pending_charls_task_uses_external_scenario",
                    "PASS"
                    if charls_rows.empty
                    or charls_rows["scenario_id"].astype(str).eq("external_readiness_first").all()
                    else "FAIL",
                    str(tasks_path),
                    f"charls_rows={len(charls_rows)}",
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
    table_path = args.project_root / "outputs" / "tables" / "agent_next_tasks_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_next_tasks_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Next Tasks Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This validation checks that next-task recommendations remain bounded to local research workflow control.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent next-task checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
