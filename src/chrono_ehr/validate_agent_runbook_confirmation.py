#!/usr/bin/env python3
"""Validate that expensive runbook execution requires explicit confirmation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
TASK = "我要去睡觉，电脑不关，可以多跑一些但不要重训模型"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(args.project_root / "src" / "chrono_ehr" / "agent_runbook.py"),
        "--project-root",
        str(args.project_root),
        "--task",
        TASK,
        "--execute-expensive-phase",
    ]
    completed = subprocess.run(command, cwd=args.project_root, text=True, capture_output=True)
    expected_message = "Refusing to execute expensive phase without --confirm-expensive."
    checks = pd.DataFrame(
        [
            {
                "check": "expensive_without_confirmation_rejected",
                "status": "PASS" if completed.returncode != 0 and expected_message in (completed.stdout + completed.stderr) else "FAIL",
                "detail": (completed.stdout + completed.stderr).strip()[-1000:],
            }
        ]
    )
    table_path = args.project_root / "outputs" / "tables" / "agent_runbook_confirmation_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_runbook_confirmation_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Runbook Confirmation Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This validation checks that expensive non-model runbook execution is blocked unless explicitly confirmed.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent runbook confirmation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
