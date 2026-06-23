#!/usr/bin/env python3
"""Plan retry/resume commands from runbook state-machine outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
MAX_AUTO_SAFE_ATTEMPTS = 3
MAX_EXPENSIVE_ATTEMPTS = 2
POST_RETRY_REFRESH_COMMAND = (
    "python3 src/chrono_ehr/run_study.py --agent-recovery-plan && "
    "python3 src/chrono_ehr/run_study.py --agent-runbook-state && "
    "python3 src/chrono_ehr/run_study.py --agent-next-tasks && "
    "python3 src/chrono_ehr/run_study.py --agent-state"
)


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


def next_retry_for_row(item: dict[str, Any]) -> dict[str, str]:
    phase = str(item.get("phase", ""))
    gate = str(item.get("gate_status", ""))
    command = str(item.get("next_allowed_command", ""))
    attempts = int(float(item.get("attempt_count", 0) or 0))
    risk = str(item.get("risk_level", ""))
    failed = int(float(item.get("failed_actions", 0) or 0))

    if gate == "recovery_required" or failed > 0:
        return {
            "phase": phase,
            "risk_level": risk,
            "retry_status": "RECOVERY_FIRST",
            "priority": "P1",
            "attempt_count": str(attempts),
            "max_recommended_attempts": str(MAX_AUTO_SAFE_ATTEMPTS),
            "requires_confirmation": "NO",
            "retry_command": "python3 src/chrono_ehr/run_study.py --agent-recovery-plan --agent-recovery-execute-safe",
            "post_retry_command": POST_RETRY_REFRESH_COMMAND,
            "reason": "This phase has failed actions; run safe recovery before retrying the phase.",
        }
    if gate == "open":
        blocked = attempts >= MAX_AUTO_SAFE_ATTEMPTS
        return {
            "phase": phase,
            "risk_level": risk,
            "retry_status": "BLOCKED_MAX_ATTEMPTS" if blocked else "READY_TO_RETRY",
            "priority": "P1" if not blocked else "P2",
            "attempt_count": str(attempts),
            "max_recommended_attempts": str(MAX_AUTO_SAFE_ATTEMPTS),
            "requires_confirmation": "NO",
            "retry_command": "" if blocked else command,
            "post_retry_command": POST_RETRY_REFRESH_COMMAND,
            "reason": "Safe phase can be retried without extra confirmation." if not blocked else "Safe phase reached the recommended retry limit; inspect logs before continuing.",
        }
    if gate == "confirmation_required":
        blocked = attempts >= MAX_EXPENSIVE_ATTEMPTS
        return {
            "phase": phase,
            "risk_level": risk,
            "retry_status": "CONFIRMATION_REQUIRED" if not blocked else "BLOCKED_MAX_ATTEMPTS",
            "priority": "P2",
            "attempt_count": str(attempts),
            "max_recommended_attempts": str(MAX_EXPENSIVE_ATTEMPTS),
            "requires_confirmation": "YES",
            "retry_command": "" if blocked else command,
            "post_retry_command": POST_RETRY_REFRESH_COMMAND,
            "reason": "Expensive non-model phase requires explicit confirmation." if not blocked else "Expensive phase reached the recommended retry limit; inspect outputs before continuing.",
        }
    if gate in {"locked", "deferred"}:
        return {
            "phase": phase,
            "risk_level": risk,
            "retry_status": "LOCKED_OR_DEFERRED",
            "priority": "P3",
            "attempt_count": str(attempts),
            "max_recommended_attempts": "0",
            "requires_confirmation": "YES",
            "retry_command": command if command.startswith(("LOCKED", "DEFERRED")) else "",
            "post_retry_command": "",
            "reason": "This phase is intentionally not retried automatically.",
        }
    return {
        "phase": phase,
        "risk_level": risk,
        "retry_status": "NO_RETRY_NEEDED",
        "priority": "P3",
        "attempt_count": str(attempts),
        "max_recommended_attempts": "0",
        "requires_confirmation": "NO",
        "retry_command": "",
        "post_retry_command": "",
        "reason": "Phase is complete, not applicable, or has no runnable gate.",
    }


def build_retry_plan(project_root: Path) -> pd.DataFrame:
    state = read_csv(project_root / "outputs" / "tables" / "agent_runbook_state_machine.csv")
    if state.empty:
        return pd.DataFrame(
            [
                {
                    "phase": "missing_state_machine",
                    "risk_level": "safe",
                    "retry_status": "REBUILD_STATE",
                    "priority": "P1",
                    "attempt_count": "0",
                    "max_recommended_attempts": str(MAX_AUTO_SAFE_ATTEMPTS),
                    "requires_confirmation": "NO",
                    "retry_command": "python3 src/chrono_ehr/run_study.py --agent-runbook-state",
                    "post_retry_command": POST_RETRY_REFRESH_COMMAND,
                    "reason": "Runbook state machine output is missing.",
                }
            ]
        )
    rows = [next_retry_for_row(item) for item in state.to_dict(orient="records")]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "phase",
        "retry_status",
        "priority",
        "attempt_count",
        "requires_confirmation",
        "retry_command",
        "reason",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, plan: pd.DataFrame) -> Path:
    table_path = project_root / "outputs" / "tables" / "agent_runbook_retry_plan.csv"
    report_path = project_root / "outputs" / "reports" / "agent_runbook_retry_plan.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(table_path, index=False)
    actionable = plan[plan["retry_status"].isin(["RECOVERY_FIRST", "READY_TO_RETRY", "CONFIRMATION_REQUIRED", "REBUILD_STATE"])]
    report_path.write_text(
        f"""# Agent Runbook Retry Plan

- Actionable items: {len(actionable)}
- Boundary: local workflow retry/resume planning only; no medical QA, diagnosis, or treatment recommendation.

## Retry Table

{markdown_table(plan)}
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    plan = build_retry_plan(args.project_root)
    report = write_outputs(args.project_root, plan)
    actionable = int(plan["retry_status"].isin(["RECOVERY_FIRST", "READY_TO_RETRY", "CONFIRMATION_REQUIRED", "REBUILD_STATE"]).sum())
    print(f"Wrote {report}")
    print(f"Runbook retry actions: {actionable}")


if __name__ == "__main__":
    main()
