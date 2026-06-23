#!/usr/bin/env python3
"""Plan or execute READY_SAFE_AUTO items from the Agent task queue."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXECUTION_COLUMNS = [
    "queue_id",
    "scenario_id",
    "queue_status",
    "execution_status",
    "started_at",
    "duration_seconds",
    "returncode",
    "cooldown_fingerprint",
    "cooldown_fingerprint_status",
    "command",
    "detail",
]
HISTORY_COLUMNS = [
    "run_id",
    "run_mode",
    "queue_filter",
    "scenario_filter",
    *EXECUTION_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--execute-safe",
        action="store_true",
        help="Execute READY_SAFE_AUTO queue items. WAITING_CONFIRMATION items are always skipped.",
    )
    parser.add_argument(
        "--queue-id",
        action="append",
        default=[],
        help="Limit the run to one queue id, such as Q003. Can be passed more than once.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=[],
        help="Limit the run to one scenario id, such as agent_control_focus. Can be passed more than once.",
    )
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


def filtered_queue(queue: pd.DataFrame, queue_ids: list[str], scenario_ids: list[str]) -> pd.DataFrame:
    if queue.empty:
        return queue
    selected = queue.copy()
    if queue_ids:
        wanted = {item.strip() for item in queue_ids if item.strip()}
        selected = selected[selected["queue_id"].astype(str).isin(wanted)]
    if scenario_ids:
        wanted = {item.strip() for item in scenario_ids if item.strip()}
        selected = selected[selected["scenario_id"].astype(str).isin(wanted)]
    return selected


def execution_rows(project_root: Path, execute_safe: bool, queue_ids: list[str], scenario_ids: list[str]) -> list[dict[str, str]]:
    queue = read_csv(project_root / "outputs" / "tables" / "agent_task_queue.csv")
    queue = filtered_queue(queue, queue_ids, scenario_ids)
    rows: list[dict[str, str]] = []
    for item in queue.to_dict(orient="records"):
        queue_id = str(item.get("queue_id", ""))
        scenario_id = str(item.get("scenario_id", ""))
        queue_status = str(item.get("queue_status", ""))
        command = cell(item, "safe_auto_command")
        cooldown_hash = cell(item, "cooldown_fingerprint")
        cooldown_status = cell(item, "cooldown_fingerprint_status")
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        if queue_status == "WAITING_CONFIRMATION":
            rows.append(
                {
                    "queue_id": queue_id,
                    "scenario_id": scenario_id,
                    "queue_status": queue_status,
                    "execution_status": "SKIPPED_CONFIRMATION_REQUIRED",
                    "started_at": started_at,
                    "duration_seconds": "0.000",
                    "returncode": "",
                    "cooldown_fingerprint": cooldown_hash,
                    "cooldown_fingerprint_status": cooldown_status,
                    "command": cell(item, "manual_confirmation_command"),
                    "detail": "manual confirmation required; not executed by queue safe runner",
                }
            )
            continue
        if queue_status == "RECENTLY_COMPLETED":
            rows.append(
                {
                    "queue_id": queue_id,
                    "scenario_id": scenario_id,
                    "queue_status": queue_status,
                    "execution_status": "SKIPPED_RECENTLY_COMPLETED",
                    "started_at": started_at,
                    "duration_seconds": "0.000",
                    "returncode": "",
                    "cooldown_fingerprint": cooldown_hash,
                    "cooldown_fingerprint_status": cooldown_status,
                    "command": cell(item, "safe_auto_command"),
                    "detail": "safe-auto command already has a recorded PASS with matching cooldown fingerprint",
                }
            )
            continue
        if queue_status != "READY_SAFE_AUTO":
            rows.append(
                {
                    "queue_id": queue_id,
                    "scenario_id": scenario_id,
                    "queue_status": queue_status,
                    "execution_status": "SKIPPED_PLAN_ONLY",
                    "started_at": started_at,
                    "duration_seconds": "0.000",
                    "returncode": "",
                    "cooldown_fingerprint": cooldown_hash,
                    "cooldown_fingerprint_status": cooldown_status,
                    "command": command,
                    "detail": "not a ready safe-auto queue item",
                }
            )
            continue
        if not command:
            rows.append(
                {
                    "queue_id": queue_id,
                    "scenario_id": scenario_id,
                    "queue_status": queue_status,
                    "execution_status": "SKIPPED_MISSING_COMMAND",
                    "started_at": started_at,
                    "duration_seconds": "0.000",
                    "returncode": "",
                    "cooldown_fingerprint": cooldown_hash,
                    "cooldown_fingerprint_status": cooldown_status,
                    "command": command,
                    "detail": "READY_SAFE_AUTO item has no safe_auto_command",
                }
            )
            continue
        if not execute_safe:
            rows.append(
                {
                    "queue_id": queue_id,
                    "scenario_id": scenario_id,
                    "queue_status": queue_status,
                    "execution_status": "PLANNED_SAFE_AUTO",
                    "started_at": started_at,
                    "duration_seconds": "0.000",
                    "returncode": "",
                    "cooldown_fingerprint": cooldown_hash,
                    "cooldown_fingerprint_status": cooldown_status,
                    "command": command,
                    "detail": "dry run only; pass --agent-task-queue-execute-safe to execute",
                }
            )
            continue
        start = time.monotonic()
        completed = subprocess.run(shlex.split(command), cwd=project_root, text=True, capture_output=True)
        duration = time.monotonic() - start
        execution_status = "PASS" if completed.returncode == 0 else "FAIL"
        rows.append(
            {
                "queue_id": queue_id,
                "scenario_id": scenario_id,
                "queue_status": queue_status,
                "execution_status": execution_status,
                "started_at": started_at,
                "duration_seconds": f"{duration:.3f}",
                "returncode": str(completed.returncode),
                "cooldown_fingerprint": cooldown_hash,
                "cooldown_fingerprint_status": "recorded_success" if execution_status == "PASS" else cooldown_status,
                "command": command,
                "detail": (completed.stdout + completed.stderr).strip()[-1000:],
            }
        )
        if completed.returncode != 0:
            break
    return rows


def append_history(project_root: Path, execution: pd.DataFrame, args: argparse.Namespace, run_id: str) -> Path:
    history_path = project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if execution.empty:
        if not history_path.exists():
            pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(history_path, index=False)
        return history_path
    history_rows = execution.copy()
    history_rows.insert(0, "scenario_filter", ", ".join(args.scenario_id) if args.scenario_id else "ALL")
    history_rows.insert(0, "queue_filter", ", ".join(args.queue_id) if args.queue_id else "ALL")
    history_rows.insert(0, "run_mode", "execute_safe" if args.execute_safe else "dry_run")
    history_rows.insert(0, "run_id", run_id)
    history_rows = history_rows[HISTORY_COLUMNS]
    existing = read_csv(history_path)
    if not existing.empty:
        missing_existing = [column for column in HISTORY_COLUMNS if column not in existing.columns]
        for column in missing_existing:
            existing[column] = ""
        history = pd.concat([existing[HISTORY_COLUMNS], history_rows], ignore_index=True)
    else:
        history = history_rows
    history.tail(500).to_csv(history_path, index=False)
    return history_path


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No queue execution rows."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].astype(object).where(pd.notna(df[columns]), "").itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    rows = execution_rows(args.project_root, args.execute_safe, args.queue_id, args.scenario_id)
    execution = pd.DataFrame(rows, columns=EXECUTION_COLUMNS)
    table_path = args.project_root / "outputs" / "tables" / "agent_task_queue_execution.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_task_queue_execution.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    execution.to_csv(table_path, index=False)
    history_path = append_history(args.project_root, execution, args, run_id)
    failures = int(execution["execution_status"].eq("FAIL").sum()) if not execution.empty else 0
    report_path.write_text(
        f"""# Agent Task Queue Execution

- Run id: `{run_id}`
- Mode: `{"execute_safe" if args.execute_safe else "dry_run"}`
- Queue filter: `{", ".join(args.queue_id) if args.queue_id else "ALL"}`
- Scenario filter: `{", ".join(args.scenario_id) if args.scenario_id else "ALL"}`
- Rows: {len(execution)}
- Failures: {failures}
- History: `{history_path.relative_to(args.project_root)}`
- Boundary: executes only READY_SAFE_AUTO local research workflow tasks when explicitly requested; no medical QA, diagnosis, or treatment recommendation.

## Execution Table

{markdown_table(execution)}
""",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
