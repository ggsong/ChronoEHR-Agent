#!/usr/bin/env python3
"""Build a short human-readable ChronoEHR-Agent status card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


STATUS_SOURCES = [
    ("agent_progress_score", "outputs/tables/agent_progress_score.csv"),
    ("mainline_mvp", "outputs/tables/mainline_mvp_validation.csv"),
    ("agent_self_check", "outputs/tables/agent_self_check.csv"),
    ("agent_doctor", "outputs/tables/agent_doctor.csv"),
    ("delivery_readiness", "outputs/tables/delivery_readiness_audit.csv"),
    ("artifact_freshness", "outputs/tables/agent_artifact_freshness.csv"),
    ("command_lint", "outputs/tables/agent_command_lint.csv"),
    ("boundary_audit", "outputs/tables/agent_boundary_audit.csv"),
    ("dependency_audit", "outputs/tables/agent_dependency_audit.csv"),
    ("doc_command_audit", "outputs/tables/agent_doc_command_audit.csv"),
    ("handoff_checklist", "outputs/tables/agent_handoff_checklist.csv"),
    ("task_execution_validation", "outputs/tables/agent_task_execution_validation.csv"),
    ("task_scenario_validation", "outputs/tables/agent_task_scenario_library_validation.csv"),
    ("task_queue_validation", "outputs/tables/agent_task_queue_validation.csv"),
    ("task_queue_execution_validation", "outputs/tables/agent_task_queue_execution_validation.csv"),
    ("cooldown_fingerprint_validation", "outputs/tables/agent_cooldown_fingerprint_validation.csv"),
    ("entrypoints_validation", "outputs/tables/agent_entrypoints_validation.csv"),
    ("next_study_validation", "outputs/tables/next_study_action_plan_validation.csv"),
    ("runbook_retry_validation", "outputs/tables/agent_runbook_retry_plan_validation.csv"),
]


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summarize_status_table(project_root: Path) -> pd.DataFrame:
    rows = []
    for item_id, relative in STATUS_SOURCES:
        path = project_root / relative
        df = read_csv(path)
        if df.empty:
            rows.append(
                {
                    "item": item_id,
                    "status": "MISSING",
                    "checks": 0,
                    "failures": "",
                    "evidence": relative,
                    "detail": "missing or empty",
                }
            )
            continue
        if item_id == "agent_progress_score" and {"component", "status", "weighted_points"}.issubset(df.columns):
            total = df[df["component"].astype(str).eq("TOTAL")]
            total_status = str(total.iloc[0]["status"]) if not total.empty else "MISSING"
            total_score = float(total.iloc[0]["weighted_points"]) if not total.empty else 0.0
            failures = 0 if total_status == "PASS" else 1
            detail = f"{total_score:.1f}/100 ({total_status})"
            checks = len(df)
        else:
            failures = int((df["status"] != "PASS").sum()) if "status" in df.columns else 0
            detail = "all checks passed" if failures == 0 else f"{failures} failing rows"
            checks = len(df)
        rows.append(
            {
                "item": item_id,
                "status": "PASS" if failures == 0 else "FAIL",
                "checks": checks,
                "failures": failures,
                "evidence": relative,
                "detail": detail,
            }
        )

    external = read_csv(project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv")
    if not external.empty and {"dataset", "local_status"}.issubset(external.columns):
        for item in external.to_dict(orient="records"):
            dataset = str(item.get("dataset", ""))
            status = str(item.get("local_status", ""))
            rows.append(
                {
                    "item": f"external_{dataset.lower()}",
                    "status": "PASS",
                    "checks": "",
                    "failures": "",
                    "evidence": "outputs/tables/external_benchmark_readiness_summary.csv",
                    "detail": status,
                }
            )
    return pd.DataFrame(rows)


def summarize_next_tasks(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "tables" / "agent_next_tasks.csv"
    tasks = read_csv(path)
    if tasks.empty:
        return pd.DataFrame(
            columns=[
                "priority",
                "scenario_id",
                "execution_mode",
                "completion_status",
                "last_success_run_id",
                "cooldown_fingerprint_status",
                "cooldown_policy_summary",
                "next_task",
                "suggested_agent_task",
                "command",
            ]
        )
    columns = [
        column
        for column in [
            "priority",
            "scenario_id",
            "execution_mode",
            "completion_status",
            "last_success_run_id",
            "cooldown_fingerprint_status",
            "cooldown_policy_summary",
            "next_task",
            "suggested_agent_task",
            "command",
        ]
        if column in tasks.columns
    ]
    return tasks[columns].head(5)


def summarize_last_task(state: dict) -> pd.DataFrame:
    last_task = state.get("last_task_execution", {}) if isinstance(state, dict) else {}
    if not last_task.get("available"):
        return pd.DataFrame(columns=["task", "scenario", "goal", "risk", "budget", "selected", "executed", "failures", "refresh", "refresh_failures"])
    return pd.DataFrame(
        [
            {
                "task": last_task.get("task", ""),
                "scenario": last_task.get("scenario_id", ""),
                "goal": last_task.get("goal_type", ""),
                "risk": last_task.get("risk_mode", ""),
                "budget": last_task.get("budget_mode", ""),
                "selected": last_task.get("selected_actions", 0),
                "executed": last_task.get("executed_actions", 0),
                "failures": last_task.get("execution_failures", 0),
                "refresh": last_task.get("post_refresh_steps", 0),
                "refresh_failures": last_task.get("post_refresh_failures", 0),
            }
        ]
    )


def summarize_active_focus(state: dict) -> pd.DataFrame:
    active = state.get("active_focus", {}) if isinstance(state, dict) else {}
    if not active.get("focus_id"):
        return pd.DataFrame(columns=["focus_id", "summary", "source", "safe_next_command"])
    return pd.DataFrame(
        [
            {
                "focus_id": active.get("focus_id", ""),
                "summary": active.get("summary", ""),
                "source": active.get("source", ""),
                "safe_next_command": active.get("safe_next_command", ""),
            }
        ]
    )


def summarize_queue_history(state: dict) -> pd.DataFrame:
    history = state.get("queue_execution_history", {}) if isinstance(state, dict) else {}
    if not history.get("available"):
        return pd.DataFrame(
            columns=[
                "latest_run_id",
                "latest_mode",
                "latest_filter",
                "latest_rows",
                "latest_failures",
                "q003_success_rows",
            ]
        )
    return pd.DataFrame(
        [
            {
                "latest_run_id": history.get("latest_run_id", ""),
                "latest_mode": history.get("latest_run_mode", ""),
                "latest_filter": history.get("latest_queue_filter", ""),
                "latest_rows": history.get("latest_rows", 0),
                "latest_failures": history.get("latest_failures", 0),
                "q003_success_rows": history.get("agent_control_focus_success_rows", 0),
            }
        ]
    )


def summarize_cooldown_policy(project_root: Path) -> pd.DataFrame:
    config_path = project_root / "configs" / "agent_cooldown_fingerprint.json"
    config = read_json(config_path)
    queue = read_csv(project_root / "outputs" / "tables" / "agent_task_queue.csv")
    history = read_csv(project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv")
    inputs = config.get("fingerprint_inputs", []) if isinstance(config, dict) else []
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
    return pd.DataFrame(
        [
            {
                "config_version": config.get("version", ""),
                "input_count": len(inputs) if isinstance(inputs, list) else 0,
                "volatile_outputs_excluded": "yes" if "outputs/" in config.get("excluded_volatile_patterns", []) else "no",
                "q003_queue_status": str(q003.iloc[0].get("queue_status", "")) if not q003.empty else "",
                "q003_cooldown_status": str(q003.iloc[0].get("cooldown_fingerprint_status", "")) if not q003.empty else "",
                "q003_last_success_run_id": str(q003_success.iloc[0].get("run_id", "")) if not q003_success.empty else "",
            }
        ]
    )


def summarize_current_queue(project_root: Path) -> pd.DataFrame:
    queue = read_csv(project_root / "outputs" / "tables" / "agent_task_queue.csv")
    if queue.empty:
        return pd.DataFrame(
            columns=[
                "queue_id",
                "queue_status",
                "scenario_id",
                "last_success_run_id",
                "cooldown_fingerprint_status",
                "cooldown_policy_summary",
                "next_task",
            ]
        )
    columns = [
        column
        for column in [
            "queue_id",
            "queue_status",
            "scenario_id",
            "last_success_run_id",
            "cooldown_fingerprint_status",
            "cooldown_policy_summary",
            "next_task",
        ]
        if column in queue.columns
    ]
    return queue[columns].head(5)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].astype(object).where(pd.notna(df[columns]), "").itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    status = summarize_status_table(args.project_root)
    next_tasks = summarize_next_tasks(args.project_root)
    state = read_json(args.project_root / "outputs" / "state" / "agent_state.json")
    last_task = summarize_last_task(state)
    active_focus = summarize_active_focus(state)
    queue_history = summarize_queue_history(state)
    cooldown_policy = summarize_cooldown_policy(args.project_root)
    current_queue = summarize_current_queue(args.project_root)
    failures = int(status["status"].eq("FAIL").sum()) if not status.empty else 1
    missing = int(status["status"].eq("MISSING").sum()) if not status.empty else 1
    overall = "PASS" if failures == 0 and missing == 0 else "ATTENTION_NEEDED"

    table_path = args.project_root / "outputs" / "tables" / "agent_status_card.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_status_card.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(table_path, index=False)
    progress = state.get("progress_summary", {}) if isinstance(state, dict) else {}
    report_path.write_text(
        f"""# ChronoEHR-Agent Status Card

- Overall status: `{overall}`
- Mainline: local EHR research tool for time-aware chronic-disease prediction.
- Boundary: not a medical QA system, not a diagnosis system, not a treatment recommendation system.
- Current stable demo: MIMIC-IV diabetes 30-day readmission.
- v0.1 note: {progress.get("short_status", "MVP validation is tracked by mainline_mvp_validation.")}

## Health Checks

{markdown_table(status)}

## Last Task

{markdown_table(last_task)}

## Active Focus

{markdown_table(active_focus)}

## Queue Execution History

{markdown_table(queue_history)}

## Cooldown Policy

{markdown_table(cooldown_policy)}

## Current Queue

{markdown_table(current_queue)}

## Next Tasks

{markdown_table(next_tasks)}

## Best Next Commands

```bash
python3 src/chrono_ehr/run_study.py --agent-doctor
python3 src/chrono_ehr/run_study.py --agent-next-tasks
python3 src/chrono_ehr/run_study.py --next-study-plan
```
""",
        encoding="utf-8",
    )
    print(f"Agent status card overall: {overall}")
    print(f"Wrote {report_path}")
    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
