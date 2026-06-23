#!/usr/bin/env python3
"""Validate natural-language task routing for ChronoEHR-Agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent_task_router import (
    AGENT_CONTROL_FOCUS_ACTION_IDS,
    DEFAULT_CATALOG,
    POST_REFRESH_COLUMNS,
    POST_RUN_REFRESH_COMMANDS,
    SCENARIO_COLUMNS,
    deferred_actions,
    infer_budget_mode,
    infer_goal_type,
    infer_risk_mode,
    infer_task_scenario,
    read_json,
    select_actions,
)


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]

CASES = [
    ("检查糖尿病 demo 有没有时间点泄漏", "leakage", "safe", "normal"),
    ("我要去睡觉，可以多跑一些但不要重训模型", "demo", "expensive", "night_run"),
    ("帮我看 eICU 和 CHARLS readiness", "external", "safe", "normal"),
    ("重训 random forest baseline", "model", "model", "normal"),
    ("现在先不要 report，打磨 agent", "status", "safe", "normal"),
    ("导出英文 docx", "report", "report", "normal"),
    ("我快没额度了，只剩2%，做一点简单自检", "status", "safe", "low_quota"),
    ("继续做 eICU 外部验证准备，不要重训模型", "external", "safe", "normal"),
    ("继续打磨 Agent 控制层，先做验证和状态卡", "status", "safe", "normal"),
    ("先完善agent，不要做汇报材料", "status", "safe", "normal"),
]

SCENARIO_CASES = [
    ("我快没额度了，只剩2%，做一点简单自检", "low_quota_self_check"),
    ("我要去睡觉，可以多跑一些但不要重训模型", "night_run_safe_plus_deferred"),
    ("继续做 eICU 外部验证准备，不要重训模型", "external_readiness_first"),
    ("先暂停一下，做一下检查。从主线来看有没有偏离？", "validation_first"),
    ("先完善 Agent 控制层，不要做汇报材料", "agent_control_focus"),
    ("重训 random forest baseline", "model_or_report_explicit"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "task",
        "expected_goal",
        "actual_goal",
        "expected_risk",
        "actual_risk",
        "expected_budget",
        "actual_budget",
        "status",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = []
    for task, expected_goal, expected_risk, expected_budget in CASES:
        actual_goal = infer_goal_type(task)
        actual_risk = infer_risk_mode(task, "auto")
        actual_budget = infer_budget_mode(task)
        rows.append(
            {
                "task": task,
                "expected_goal": expected_goal,
                "actual_goal": actual_goal,
                "expected_risk": expected_risk,
                "actual_risk": actual_risk,
                "expected_budget": expected_budget,
                "actual_budget": actual_budget,
                "status": "PASS"
                if actual_goal == expected_goal and actual_risk == expected_risk and actual_budget == expected_budget
                else "FAIL",
            }
        )
    for task, expected_scenario in SCENARIO_CASES:
        actual_goal = infer_goal_type(task)
        actual_risk = infer_risk_mode(task, "auto")
        actual_budget = infer_budget_mode(task)
        actual_scenario = infer_task_scenario(task, actual_goal, actual_risk, actual_budget)["scenario_id"]
        rows.append(
            {
                "task": "__scenario__ " + task,
                "expected_goal": expected_scenario,
                "actual_goal": actual_scenario,
                "expected_risk": actual_risk,
                "actual_risk": actual_risk,
                "expected_budget": actual_budget,
                "actual_budget": actual_budget,
                "status": "PASS" if actual_scenario == expected_scenario else "FAIL",
                "detail": f"goal={actual_goal}; risk={actual_risk}; budget={actual_budget}",
            }
        )
    catalog = read_json(DEFAULT_CATALOG)
    external_actions = select_actions(catalog, "external", "safe", "normal")
    external_deferred = deferred_actions(catalog, "external", "safe")
    required_plan_columns = {"phase", "execution_policy", "policy_reason"}
    missing_plan_columns = sorted(required_plan_columns - set(external_actions.columns))
    rows.append(
        {
            "task": "__external_plan_schema__",
            "expected_goal": "external",
            "actual_goal": "external",
            "expected_risk": "safe",
            "actual_risk": "safe",
            "expected_budget": "normal",
            "actual_budget": "normal",
            "status": "PASS" if not missing_plan_columns else "FAIL",
            "detail": "missing_columns=" + ",".join(missing_plan_columns),
        }
    )
    deferred_ids = set(external_deferred["id"].astype(str)) if not external_deferred.empty and "id" in external_deferred else set()
    rows.append(
        {
            "task": "__external_deferred_risk_gates__",
            "expected_goal": "external",
            "actual_goal": "external",
            "expected_risk": "safe",
            "actual_risk": "safe",
            "expected_budget": "normal",
            "actual_budget": "normal",
            "status": "PASS" if {"eicu_temporal_features", "eicu_logistic_baseline"}.issubset(deferred_ids) else "FAIL",
            "detail": "deferred_ids=" + ",".join(sorted(deferred_ids)),
        }
    )
    refresh_ids = {item[0] for item in POST_RUN_REFRESH_COMMANDS}
    required_refresh_ids = {
        "agent_recovery_plan",
        "agent_next_tasks",
        "agent_state",
        "agent_state_validation",
        "agent_handoff_checklist",
        "agent_progress_score",
        "agent_progress_score_validation",
        "agent_command_lint",
        "agent_control_consistency",
    }
    rows.append(
        {
            "task": "__post_run_refresh_chain__",
            "expected_goal": "status",
            "actual_goal": "status",
            "expected_risk": "safe",
            "actual_risk": "safe",
            "expected_budget": "normal",
            "actual_budget": "normal",
            "status": "PASS" if required_refresh_ids.issubset(refresh_ids) else "FAIL",
            "detail": "missing=" + ",".join(sorted(required_refresh_ids - refresh_ids)),
        }
    )
    missing_refresh_columns = sorted(set(POST_REFRESH_COLUMNS) - {"id", "status", "command", "started_at", "duration_seconds", "returncode", "detail"})
    rows.append(
        {
            "task": "__post_run_refresh_schema__",
            "expected_goal": "status",
            "actual_goal": "status",
            "expected_risk": "safe",
            "actual_risk": "safe",
            "expected_budget": "normal",
            "actual_budget": "normal",
            "status": "PASS" if not missing_refresh_columns and len(POST_REFRESH_COLUMNS) == 7 else "FAIL",
            "detail": "columns=" + ",".join(POST_REFRESH_COLUMNS),
        }
    )
    rows.append(
        {
            "task": "__scenario_schema__",
            "expected_goal": "scenario",
            "actual_goal": "scenario",
            "expected_risk": "safe",
            "actual_risk": "safe",
            "expected_budget": "normal",
            "actual_budget": "normal",
            "status": "PASS"
            if set(SCENARIO_COLUMNS)
            == {
                "scenario_id",
                "title",
                "when_to_use",
                "execution_style",
                "auto_run_policy",
                "next_step_hint",
                "example_task",
            }
            else "FAIL",
            "detail": "columns=" + ",".join(SCENARIO_COLUMNS),
        }
    )
    agent_actions = select_actions(catalog, "status", "safe", "normal", "agent_control_focus")
    agent_ids = set(agent_actions["id"].astype(str)) if not agent_actions.empty and "id" in agent_actions else set()
    disallowed_agent_ids = sorted(agent_ids - AGENT_CONTROL_FOCUS_ACTION_IDS)
    rows.append(
        {
            "task": "__agent_control_focus_action_boundary__",
            "expected_goal": "agent_control_focus",
            "actual_goal": "agent_control_focus",
            "expected_risk": "safe",
            "actual_risk": "safe",
            "expected_budget": "normal",
            "actual_budget": "normal",
            "status": "PASS" if agent_ids and not disallowed_agent_ids else "FAIL",
            "detail": "selected=" + ",".join(sorted(agent_ids)) + "; disallowed=" + ",".join(disallowed_agent_ids),
        }
    )
    agent_high_risk = (
        agent_actions["risk_level"].astype(str).isin(["expensive", "model", "report"]).sum()
        if not agent_actions.empty and "risk_level" in agent_actions
        else 0
    )
    agent_bad_commands = (
        agent_actions["command"].fillna("").astype(str).str.contains(
            "--confirm-expensive|--agent-runbook-execute-expensive-phase|--agent-runbook-execute-model|docx|manuscript",
            regex=True,
            na=False,
        )
        if not agent_actions.empty and "command" in agent_actions
        else pd.Series(dtype=bool)
    )
    rows.append(
        {
            "task": "__agent_control_focus_no_report_model_or_expensive__",
            "expected_goal": "agent_control_focus",
            "actual_goal": "agent_control_focus",
            "expected_risk": "safe",
            "actual_risk": "safe",
            "expected_budget": "normal",
            "actual_budget": "normal",
            "status": "PASS" if int(agent_high_risk) == 0 and int(agent_bad_commands.sum()) == 0 else "FAIL",
            "detail": f"high_risk={int(agent_high_risk)}; bad_commands={int(agent_bad_commands.sum())}",
        }
    )
    run_study = (args.project_root / "src" / "chrono_ehr" / "run_study.py").read_text(encoding="utf-8")
    rows.append(
        {
            "task": "__post_run_refresh_flag_registered__",
            "expected_goal": "status",
            "actual_goal": "status",
            "expected_risk": "safe",
            "actual_risk": "safe",
            "expected_budget": "normal",
            "actual_budget": "normal",
            "status": "PASS" if "--agent-task-post-run-refresh" in run_study and "--post-run-refresh" in run_study else "FAIL",
            "detail": "requires run_study flag and forwarded script flag",
        }
    )
    checks = pd.DataFrame(rows)
    table_path = args.project_root / "outputs" / "tables" / "agent_task_router_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_task_router_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Task Router Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This internal validation checks whether short Chinese/English research tasks route to the expected local workflow goal and risk mode. It is not a clinical decision check.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent task router checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
