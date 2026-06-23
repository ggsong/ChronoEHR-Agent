#!/usr/bin/env python3
"""Validate the Agent task queue safety boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agent_task_queue import QUEUE_COLUMNS
from mimic_diabetes_baseline import DEFAULT_PROJECT


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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path) -> pd.DataFrame:
    queue_path = project_root / "outputs" / "tables" / "agent_task_queue.csv"
    report_path = project_root / "outputs" / "reports" / "agent_task_queue.md"
    next_tasks_path = project_root / "outputs" / "tables" / "agent_next_tasks.csv"
    queue = read_csv(queue_path)
    next_tasks = read_csv(next_tasks_path)
    report = read_text(report_path)
    rows = [
        row("task_queue_table_exists", "PASS" if not queue.empty else "FAIL", str(queue_path), f"rows={len(queue)}"),
        row("task_queue_report_exists", "PASS" if bool(report) else "FAIL", str(report_path), f"chars={len(report)}"),
    ]
    if queue.empty:
        return pd.DataFrame(rows)

    missing = sorted(set(QUEUE_COLUMNS) - set(queue.columns))
    rows.append(row("task_queue_columns", "PASS" if not missing else "FAIL", str(queue_path), "missing=" + ",".join(missing)))

    if not next_tasks.empty:
        rows.append(
            row(
                "queue_matches_next_task_count",
                "PASS" if len(queue) == len(next_tasks) else "FAIL",
                f"{queue_path}; {next_tasks_path}",
                f"queue={len(queue)}; next_tasks={len(next_tasks)}",
            )
        )

    manual = queue[queue["queue_status"].astype(str).eq("WAITING_CONFIRMATION")]
    ready = queue[queue["queue_status"].astype(str).eq("READY_SAFE_AUTO")]
    recently_completed = queue[queue["queue_status"].astype(str).eq("RECENTLY_COMPLETED")]
    allowed_statuses = {"READY_SAFE_AUTO", "WAITING_CONFIRMATION", "PLAN_ONLY", "RECENTLY_COMPLETED"}
    bad_statuses = sorted(set(queue["queue_status"].astype(str)) - allowed_statuses)
    rows.append(row("task_queue_status_values", "PASS" if not bad_statuses else "FAIL", str(queue_path), "bad_statuses=" + ",".join(bad_statuses)))
    rows.append(
        row(
            "manual_items_have_no_safe_auto_command",
            "PASS" if manual.empty or manual["safe_auto_command"].fillna("").astype(str).eq("").all() else "FAIL",
            str(queue_path),
            f"manual_rows={len(manual)}",
        )
    )
    rows.append(
        row(
            "manual_items_have_confirmation_command",
            "PASS" if manual.empty or manual["manual_confirmation_command"].fillna("").astype(str).str.contains("--confirm-expensive", na=False).all() else "FAIL",
            str(queue_path),
            f"manual_rows={len(manual)}",
        )
    )
    rows.append(
        row(
            "safe_auto_items_have_safe_refresh_command",
            "PASS"
            if ready.empty
            or (
                ready["safe_auto_command"].fillna("").astype(str).str.contains("--agent-task-execute-safe", na=False).all()
                and ready["safe_auto_command"].fillna("").astype(str).str.contains("--agent-task-post-run-refresh", na=False).all()
            )
            else "FAIL",
            str(queue_path),
            f"ready_rows={len(ready)}",
        )
    )
    rows.append(
        row(
            "recently_completed_items_have_success_memory",
            "PASS"
            if recently_completed.empty
            or (
                recently_completed["last_success_run_id"].fillna("").astype(str).str.len().gt(0).all()
                and recently_completed["last_success_at"].fillna("").astype(str).str.len().gt(0).all()
                and recently_completed["cooldown_fingerprint"].fillna("").astype(str).str.len().ge(32).all()
                and recently_completed["last_success_fingerprint"].fillna("").astype(str).str.len().ge(32).all()
                and recently_completed["cooldown_fingerprint_status"].fillna("").astype(str).eq("matched_success").all()
                and recently_completed["safe_auto_command"].fillna("").astype(str).eq("").all()
            )
            else "FAIL",
            str(queue_path),
            f"recently_completed_rows={len(recently_completed)}",
        )
    )
    rows.append(
        row(
            "recently_completed_items_explain_cooldown",
            "PASS"
            if recently_completed.empty
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
            else "FAIL",
            str(queue_path),
            f"recently_completed_rows={len(recently_completed)}",
        )
    )
    rows.append(
        row(
            "ready_safe_auto_items_have_current_fingerprint",
            "PASS"
            if ready.empty
            or (
                ready["cooldown_fingerprint"].fillna("").astype(str).str.len().ge(32).all()
                and ready["cooldown_fingerprint_status"].fillna("").astype(str).ne("").all()
            )
            else "FAIL",
            str(queue_path),
            f"ready_rows={len(ready)}",
        )
    )
    cooled_agent_control = queue[
        queue["scenario_id"].astype(str).eq("agent_control_focus")
        & queue["queue_status"].astype(str).eq("RECENTLY_COMPLETED")
    ]
    history = read_csv(project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv")
    matching_agent_control_success = pd.DataFrame()
    if not history.empty and "cooldown_fingerprint" in history and "cooldown_fingerprint" in queue:
        agent_control_fingerprints = set(
            queue[queue["scenario_id"].astype(str).eq("agent_control_focus")]["cooldown_fingerprint"].fillna("").astype(str)
        )
        matching_agent_control_success = history[
            history["scenario_id"].astype(str).eq("agent_control_focus")
            & history["execution_status"].astype(str).eq("PASS")
            & history["cooldown_fingerprint"].fillna("").astype(str).isin(agent_control_fingerprints)
        ]
    expects_agent_control_cooldown = not matching_agent_control_success.empty
    rows.append(
        row(
            "agent_control_focus_recent_success_cools_down",
            "PASS" if not expects_agent_control_cooldown or not cooled_agent_control.empty else "FAIL",
            str(queue_path),
            f"matching_fingerprint_success_rows={len(matching_agent_control_success)}",
        )
    )
    risky_auto = queue["safe_auto_command"].fillna("").astype(str).str.contains("--confirm-expensive|--agent-runbook-execute-expensive-phase|--agent-runbook-execute-model", regex=True, na=False)
    rows.append(row("safe_auto_commands_do_not_cross_high_risk_gates", "PASS" if int(risky_auto.sum()) == 0 else "FAIL", str(queue_path), f"risky_auto={int(risky_auto.sum())}"))
    rows.append(
        row(
            "task_queue_report_boundary",
            "PASS" if "no medical QA" in report and "treatment recommendation" in report else "FAIL",
            str(report_path),
            "expects non-clinical boundary wording",
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
    table_path = args.project_root / "outputs" / "tables" / "agent_task_queue_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_task_queue_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Task Queue Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local Agent task queue safety only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent task queue checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
