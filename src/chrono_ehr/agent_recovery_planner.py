#!/usr/bin/env python3
"""Plan minimal recovery actions from Agent self-check and readiness failures."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from agent_self_check import CHECKS


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]

RECOVERY_COMMANDS = {str(check["id"]): str(check["command"]) for check in CHECKS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--execute-safe-recovery",
        action="store_true",
        help="Execute planned P1/P2 recovery commands that are known safe local checks.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def self_check_recovery(project_root: Path) -> list[dict[str, str]]:
    checks = read_csv(project_root / "outputs" / "tables" / "agent_self_check.csv")
    if checks.empty:
        return [
            {
                "source": "agent_self_check",
                "failed_item": "missing_self_check",
                "priority": "P1",
                "recovery_command": "python3 src/chrono_ehr/run_study.py --agent-self-check",
                "reason": "Self-check output is missing.",
            }
        ]
    rows = []
    for item in checks[checks["status"].ne("PASS")].to_dict(orient="records"):
        check_id = str(item.get("id", ""))
        rows.append(
            {
                "source": "agent_self_check",
                "failed_item": check_id,
                "priority": "P1",
                "recovery_command": RECOVERY_COMMANDS.get(check_id, "python3 src/chrono_ehr/run_study.py --agent-self-check"),
                "reason": str(item.get("detail", ""))[:300],
            }
        )
    return rows


def readiness_recovery(project_root: Path) -> list[dict[str, str]]:
    readiness = read_csv(project_root / "outputs" / "tables" / "delivery_readiness_audit.csv")
    if readiness.empty:
        return [
            {
                "source": "delivery_readiness",
                "failed_item": "missing_delivery_readiness",
                "priority": "P1",
                "recovery_command": "python3 src/chrono_ehr/run_study.py --delivery-readiness",
                "reason": "Delivery readiness output is missing.",
            }
        ]
    rows = []
    for item in readiness[readiness["status"].ne("PASS")].head(10).to_dict(orient="records"):
        category = str(item.get("category", ""))
        check = str(item.get("check", ""))
        if "agent_control" in check or "agent_task" in check or "agent_action" in check:
            command = "python3 src/chrono_ehr/run_study.py --agent-self-check"
        elif "feature_window" in check:
            command = "python3 src/chrono_ehr/run_study.py --validate-feature-windows"
        elif "leakage" in check:
            command = "python3 src/chrono_ehr/run_study.py --leakage-gate"
        elif category == "word_package" or "docx" in check.lower():
            command = "python3 src/chrono_ehr/run_study.py --delivery-readiness"
        else:
            command = "python3 src/chrono_ehr/run_study.py --delivery-readiness"
        rows.append(
            {
                "source": "delivery_readiness",
                "failed_item": check,
                "priority": "P2",
                "recovery_command": command,
                "reason": str(item.get("detail", ""))[:300],
            }
        )
    return rows


def build_recovery_plan(project_root: Path) -> pd.DataFrame:
    rows = self_check_recovery(project_root) + readiness_recovery(project_root)
    if not rows:
        rows = [
            {
                "source": "agent_recovery_planner",
                "failed_item": "none",
                "priority": "P3",
                "recovery_command": "python3 src/chrono_ehr/run_study.py --agent-task \"继续打磨 agent\"",
                "reason": "No failures detected; continue with Agent workflow improvements.",
            }
        ]
    return pd.DataFrame(rows)


def execute_recovery(project_root: Path, plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in plan.to_dict(orient="records"):
        command = str(item.get("recovery_command", ""))
        priority = str(item.get("priority", ""))
        failed_item = str(item.get("failed_item", ""))
        if failed_item == "none":
            rows.append({**item, "execution_status": "SKIPPED", "returncode": "", "duration_seconds": "0.000", "execution_detail": "no failure"})
            continue
        if priority not in {"P1", "P2"} or "run_study.py --" not in command:
            rows.append({**item, "execution_status": "SKIPPED", "returncode": "", "duration_seconds": "0.000", "execution_detail": "not safe recovery"})
            continue
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        start = time.monotonic()
        completed = subprocess.run(shlex.split(command), cwd=project_root, text=True, capture_output=True)
        duration = time.monotonic() - start
        rows.append(
            {
                **item,
                "execution_status": "PASS" if completed.returncode == 0 else "FAIL",
                "returncode": str(completed.returncode),
                "started_at": started_at,
                "duration_seconds": f"{duration:.3f}",
                "execution_detail": (completed.stdout + completed.stderr).strip()[-1000:],
            }
        )
        if completed.returncode != 0:
            break
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is None:
        columns = ["source", "failed_item", "priority", "recovery_command", "reason"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, plan: pd.DataFrame, execution: pd.DataFrame) -> Path:
    table_path = project_root / "outputs" / "tables" / "agent_recovery_plan.csv"
    execution_path = project_root / "outputs" / "tables" / "agent_recovery_execution.csv"
    report_path = project_root / "outputs" / "reports" / "agent_recovery_plan.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(table_path, index=False)
    execution.to_csv(execution_path, index=False)
    has_failure = not (len(plan) == 1 and plan.iloc[0]["failed_item"] == "none")
    if execution.empty:
        execution_text = "Not executed."
    else:
        execution_display = execution.rename(columns={"reason": "recovery_reason"})
        execution_text = markdown_table(
            execution_display,
            [
                "source",
                "failed_item",
                "priority",
                "recovery_command",
                "execution_status",
                "duration_seconds",
                "recovery_reason",
                "execution_detail",
            ],
        )
    report_path.write_text(
        f"""# Agent Recovery Plan

- Recovery needed: `{"YES" if has_failure else "NO"}`
- Items: {len(plan)}
- Boundary: local workflow recovery only; no medical QA, diagnosis, or treatment recommendation.

## Recovery Table

{markdown_table(plan)}

## Execution

{execution_text}
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    plan = build_recovery_plan(args.project_root)
    execution = execute_recovery(args.project_root, plan) if args.execute_safe_recovery else pd.DataFrame()
    report = write_outputs(args.project_root, plan, execution)
    print(f"Wrote {report}")
    print(f"Recovery items: {len(plan)}")
    if not execution.empty:
        failures = int((execution["execution_status"] == "FAIL").sum())
        print(f"Recovery executions: {len(execution)}; failures={failures}")
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
