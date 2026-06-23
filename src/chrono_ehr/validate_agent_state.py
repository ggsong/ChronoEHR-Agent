#!/usr/bin/env python3
"""Validate the persistent ChronoEHR-Agent state snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    state_path = project_root / "outputs" / "state" / "agent_state.json"
    md_path = project_root / "outputs" / "state" / "agent_state.md"
    state = read_json(state_path)
    rows = [
        row("agent_state_json_exists", "PASS" if state_path.exists() and state_path.stat().st_size > 0 else "FAIL", str(state_path), f"keys={len(state)}"),
        row("agent_state_md_exists", "PASS" if md_path.exists() and md_path.stat().st_size > 0 else "FAIL", str(md_path), f"size={md_path.stat().st_size if md_path.exists() else 0}"),
    ]
    if not state:
        return pd.DataFrame(rows)

    rows.append(
        row(
            "boundary_declared",
            "PASS" if "not medical QA" in str(state.get("boundary", "")) or "not medical" in str(state.get("boundary", "")) else "FAIL",
            str(state_path),
            str(state.get("boundary", "")),
        )
    )
    delivery = state.get("delivery_readiness", {})
    rows.append(
        row(
            "delivery_readiness_pass",
            "PASS" if delivery.get("overall") == "PASS" and int(delivery.get("failures", 1) or 0) == 0 else "FAIL",
            str(state_path),
            str(delivery),
        )
    )
    external = state.get("external_datasets", [])
    status_by_dataset = {str(item.get("dataset", "")): str(item.get("local_status", "")) for item in external}
    rows.append(
        row(
            "external_statuses_present",
            "PASS" if {"CDSL", "eICU", "CHARLS"} <= set(status_by_dataset) else "FAIL",
            str(state_path),
            str(status_by_dataset),
        )
    )
    boundaries = " ".join(str(item) for item in state.get("known_boundaries", []))
    eicu_status = status_by_dataset.get("eICU", "")
    stale_eicu_text = "eICU is data-pending" in boundaries if eicu_status != "DATA_PENDING" else False
    rows.append(
        row(
            "eicu_boundary_not_stale",
            "PASS" if not stale_eicu_text else "FAIL",
            str(state_path),
            f"eICU status={eicu_status}; boundaries={boundaries}",
        )
    )
    rows.append(
        row(
            "external_boundary_notes_present",
            "PASS"
            if "CDSL is an external temporal-method benchmark" in boundaries
            and "CHARLS is a longitudinal chronic-disease cohort extension" in boundaries
            else "FAIL",
            str(state_path),
            boundaries,
        )
    )
    recommended = state.get("recommended_next_commands", [])
    rows.append(
        row(
            "recommended_commands_are_safe_entrypoints",
            "PASS" if recommended and all(str(command).startswith("python3 src/chrono_ehr/run_study.py --") for command in recommended) else "FAIL",
            str(state_path),
            " | ".join(str(command) for command in recommended),
        )
    )
    recommended_text = " | ".join(str(command) for command in recommended)
    rows.append(
        row(
            "recommended_commands_include_doctor",
            "PASS" if "--agent-doctor" in recommended_text else "FAIL",
            str(state_path),
            recommended_text,
        )
    )
    rows.append(
        row(
            "recommended_commands_include_control_consistency",
            "PASS" if "--agent-control-consistency" in recommended_text else "FAIL",
            str(state_path),
            recommended_text,
        )
    )
    next_tasks = state.get("next_tasks", [])
    rows.append(
        row(
            "next_tasks_embedded",
            "PASS" if next_tasks else "FAIL",
            str(state_path),
            f"next_tasks={len(next_tasks)}",
        )
    )
    rows.append(
        row(
            "next_tasks_cooldown_policy_summary_embedded",
            "PASS"
            if next_tasks and all(str(item.get("cooldown_policy_summary", "")).strip() for item in next_tasks)
            else "FAIL",
            str(state_path),
            str(next_tasks[:3]),
        )
    )
    active_focus = state.get("active_focus", {})
    rows.append(
        row(
            "active_focus_embedded",
            "PASS" if active_focus.get("focus_id") and active_focus.get("summary") else "FAIL",
            str(state_path),
            str(active_focus),
        )
    )
    focus_guardrails = " ".join(str(item) for item in active_focus.get("guardrails", []))
    rows.append(
        row(
            "active_focus_guardrails_present",
            "PASS"
            if "medical QA" in focus_guardrails
            or ("report" in focus_guardrails and "model" in focus_guardrails and "expensive" in focus_guardrails)
            else "FAIL",
            str(state_path),
            focus_guardrails,
        )
    )
    last_task = state.get("last_task_execution", {})
    rows.append(
        row(
            "last_task_execution_embedded",
            "PASS" if last_task.get("available") is True else "FAIL",
            str(state_path),
            str(last_task),
        )
    )
    rows.append(
        row(
            "last_task_execution_has_no_failures",
            "PASS"
            if last_task.get("available") is True
            and int(last_task.get("execution_failures", 1) or 0) == 0
            and int(last_task.get("post_refresh_failures", 1) or 0) == 0
            and int(last_task.get("validation_failures", 1) or 0) == 0
            else "FAIL",
            str(state_path),
            str(last_task),
        )
    )
    rows.append(
        row(
            "last_task_execution_scenario_recorded",
            "PASS" if last_task.get("available") is True and bool(last_task.get("scenario_id")) else "FAIL",
            str(state_path),
            str(last_task),
        )
    )
    if last_task.get("scenario_id") == "agent_control_focus":
        rows.append(
            row(
                "agent_control_focus_persisted_as_active_focus",
                "PASS"
                if active_focus.get("focus_id") == "agent_control_focus"
                and "不做汇报材料" in str(active_focus.get("summary", ""))
                else "FAIL",
                str(state_path),
                str(active_focus),
            )
        )
    queue_history = state.get("queue_execution_history", {})
    rows.append(
        row(
            "queue_execution_history_embedded",
            "PASS" if queue_history.get("available") is True and int(queue_history.get("rows", 0) or 0) > 0 else "FAIL",
            str(state_path),
            str(queue_history),
        )
    )
    rows.append(
        row(
            "latest_queue_execution_has_no_failures",
            "PASS"
            if queue_history.get("available") is True and int(queue_history.get("latest_failures", 1) or 0) == 0
            else "FAIL",
            str(state_path),
            str(queue_history),
        )
    )
    rows.append(
        row(
            "agent_control_queue_success_count_recorded",
            "PASS"
            if queue_history.get("available") is True and "agent_control_focus_success_rows" in queue_history
            else "FAIL",
            str(state_path),
            str(queue_history),
        )
    )
    current_queue = state.get("current_queue", [])
    rows.append(
        row(
            "current_queue_embedded",
            "PASS" if current_queue else "FAIL",
            str(state_path),
            f"queue_rows={len(current_queue)}",
        )
    )
    agent_control_queue_items = [
        item for item in current_queue if str(item.get("scenario_id", "")) == "agent_control_focus"
    ]
    expects_agent_control_cooldown = any(
        str(item.get("cooldown_fingerprint_status", "")) == "matched_success" for item in agent_control_queue_items
    )
    if expects_agent_control_cooldown:
        cooled = [item for item in agent_control_queue_items if str(item.get("queue_status", "")) == "RECENTLY_COMPLETED"]
        rows.append(
            row(
                "current_queue_agent_control_cooldown",
                "PASS" if cooled else "FAIL",
                str(state_path),
                str(current_queue),
            )
        )
    cooldown_policy = state.get("cooldown_policy", {})
    rows.append(
        row(
            "cooldown_policy_embedded",
            "PASS"
            if cooldown_policy.get("available") is True
            and str(cooldown_policy.get("config_version", "")).strip()
            and int(cooldown_policy.get("input_count", 0) or 0) > 0
            else "FAIL",
            str(state_path),
            str(cooldown_policy),
        )
    )
    rows.append(
        row(
            "cooldown_policy_excludes_volatile_outputs",
            "PASS" if cooldown_policy.get("volatile_outputs_excluded") is True else "FAIL",
            str(state_path),
            str(cooldown_policy),
        )
    )
    if cooldown_policy.get("q003_last_success_run_id") and (
        cooldown_policy.get("q003_queue_status") or cooldown_policy.get("q003_cooldown_status")
    ):
        q003_queue_status = str(cooldown_policy.get("q003_queue_status", ""))
        q003_cooldown_status = str(cooldown_policy.get("q003_cooldown_status", ""))
        q003_policy_valid = (
            q003_queue_status == "RECENTLY_COMPLETED"
            and q003_cooldown_status == "matched_success"
        ) or (
            q003_queue_status == "READY_SAFE_AUTO"
            and q003_cooldown_status in {"changed_since_success", "legacy_success_missing_fingerprint"}
        )
        rows.append(
            row(
                "cooldown_policy_q003_matches_queue",
                "PASS" if q003_policy_valid else "FAIL",
                str(state_path),
                str(cooldown_policy),
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
    checks = validate(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_state_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_state_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent State Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local Agent memory/state only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent state checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
