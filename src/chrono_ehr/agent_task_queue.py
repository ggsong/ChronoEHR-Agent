#!/usr/bin/env python3
"""Build a queue from scenario-aware Agent next-task recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agent_cooldown_fingerprint import cooldown_fingerprint
from mimic_diabetes_baseline import DEFAULT_PROJECT


QUEUE_COLUMNS = [
    "queue_id",
    "priority",
    "queue_status",
    "scenario_id",
    "execution_mode",
    "next_task",
    "suggested_agent_task",
    "safe_auto_command",
    "manual_confirmation_command",
    "last_success_run_id",
    "last_success_at",
    "cooldown_fingerprint",
    "last_success_fingerprint",
    "cooldown_fingerprint_status",
    "cooldown_reason",
    "cooldown_policy_summary",
    "cooldown_missing_inputs",
    "reason",
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


def cell(item: dict[str, object], key: str) -> str:
    value = item.get(key, "")
    if pd.isna(value):
        return ""
    return str(value)


def queue_status(execution_mode: str) -> str:
    if execution_mode == "manual_confirmation_required":
        return "WAITING_CONFIRMATION"
    if execution_mode == "safe_auto_allowed":
        return "READY_SAFE_AUTO"
    if execution_mode == "recently_completed":
        return "RECENTLY_COMPLETED"
    return "PLAN_ONLY"


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


def build_queue(project_root: Path) -> pd.DataFrame:
    tasks = read_csv(project_root / "outputs" / "tables" / "agent_next_tasks.csv")
    history = read_csv(project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv")
    if tasks.empty:
        return pd.DataFrame(columns=QUEUE_COLUMNS)
    rows = []
    for idx, item in enumerate(tasks.to_dict(orient="records"), start=1):
        execution_mode = cell(item, "execution_mode") or "plan_only"
        status = queue_status(execution_mode)
        scenario_id = cell(item, "scenario_id")
        safe_auto_command = cell(item, "suggested_safe_refresh_command") if status == "READY_SAFE_AUTO" else ""
        cooldown_hash = cell(item, "cooldown_fingerprint")
        if safe_auto_command and not cooldown_hash:
            fingerprint_info = cooldown_fingerprint(
                project_root,
                scenario_id,
                cell(item, "suggested_agent_task"),
                safe_auto_command,
            )
            cooldown_hash = fingerprint_info["cooldown_fingerprint"]
            cooldown_missing_inputs = fingerprint_info["cooldown_missing_inputs"]
        else:
            cooldown_missing_inputs = cell(item, "cooldown_missing_inputs")
        success = {
            "last_success_run_id": cell(item, "last_success_run_id"),
            "last_success_at": cell(item, "last_success_at"),
            "last_success_fingerprint": cell(item, "last_success_fingerprint"),
            "cooldown_fingerprint_status": cell(item, "cooldown_fingerprint_status"),
        }
        cooldown_reason = cell(item, "cooldown_reason")
        cooldown_summary = cell(item, "cooldown_policy_summary")
        history_success = last_success_for(history, str(scenario_id), str(safe_auto_command), cooldown_hash)
        if history_success.get("cooldown_fingerprint_status") == "matched_success" or not success["last_success_run_id"]:
            success = history_success
        if status == "READY_SAFE_AUTO" and success and success.get("cooldown_fingerprint_status") == "matched_success":
            status = "RECENTLY_COMPLETED"
            safe_auto_command = ""
            cooldown_reason = "safe-auto command already has a recorded PASS with matching cooldown fingerprint"
            cooldown_summary = (
                "Cooldown active: matching safe-auto PASS history found for the current code/config fingerprint; "
                "rerun only after Agent code/config changes or explicit user request."
            )
        elif status == "READY_SAFE_AUTO" and success and success.get("last_success_run_id"):
            cooldown_reason = "previous PASS exists, but cooldown inputs changed or legacy success lacks a fingerprint"
            if success.get("cooldown_fingerprint_status") == "changed_since_success":
                cooldown_summary = "Safe rerun allowed: previous PASS exists, but the cooldown fingerprint changed since that success."
            elif success.get("cooldown_fingerprint_status") == "legacy_success_missing_fingerprint":
                cooldown_summary = (
                    "Safe rerun allowed: previous PASS exists, but that legacy success did not record a cooldown fingerprint."
                )
        rows.append(
            {
                "queue_id": f"Q{idx:03d}",
                "priority": cell(item, "priority"),
                "queue_status": status,
                "scenario_id": scenario_id,
                "execution_mode": execution_mode,
                "next_task": cell(item, "next_task"),
                "suggested_agent_task": cell(item, "suggested_agent_task"),
                "safe_auto_command": safe_auto_command if status == "READY_SAFE_AUTO" else "",
                "manual_confirmation_command": cell(item, "command") if status == "WAITING_CONFIRMATION" else "",
                "last_success_run_id": success.get("last_success_run_id", ""),
                "last_success_at": success.get("last_success_at", ""),
                "cooldown_fingerprint": cooldown_hash,
                "last_success_fingerprint": success.get("last_success_fingerprint", ""),
                "cooldown_fingerprint_status": success.get("cooldown_fingerprint_status", cell(item, "cooldown_fingerprint_status")),
                "cooldown_reason": cooldown_reason,
                "cooldown_policy_summary": cooldown_summary,
                "cooldown_missing_inputs": cooldown_missing_inputs,
                "reason": cell(item, "reason"),
            }
        )
    return pd.DataFrame(rows, columns=QUEUE_COLUMNS)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No queued tasks."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].astype(object).where(pd.notna(df[columns]), "").itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    queue = build_queue(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "agent_task_queue.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_task_queue.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(table_path, index=False)
    ready = int(queue["queue_status"].eq("READY_SAFE_AUTO").sum()) if not queue.empty else 0
    waiting = int(queue["queue_status"].eq("WAITING_CONFIRMATION").sum()) if not queue.empty else 0
    completed = int(queue["queue_status"].eq("RECENTLY_COMPLETED").sum()) if not queue.empty else 0
    report_path.write_text(
        f"""# Agent Task Queue

- Items: {len(queue)}
- Ready safe-auto items: {ready}
- Waiting confirmation items: {waiting}
- Recently completed items: {completed}
- Boundary: local research workflow queue only; no medical QA, diagnosis, or treatment recommendation.

## Queue Table

{markdown_table(queue)}
""",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    if not queue.empty:
        print(queue[["queue_id", "priority", "queue_status", "scenario_id", "next_task"]].to_string(index=False))


if __name__ == "__main__":
    main()
