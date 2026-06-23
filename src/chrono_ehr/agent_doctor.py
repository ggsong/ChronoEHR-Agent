#!/usr/bin/env python3
"""Run a final local health check bundle for ChronoEHR-Agent."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


DOCTOR_STEPS = [
    {
        "id": "agent_self_check",
        "command": "python3 src/chrono_ehr/run_study.py --agent-self-check",
        "purpose": "Run the full local Agent health-check queue.",
    },
    {
        "id": "delivery_readiness",
        "command": "python3 src/chrono_ehr/run_study.py --delivery-readiness",
        "purpose": "Refresh the project-level delivery readiness gate after self-check outputs are written.",
    },
    {
        "id": "agent_artifact_freshness",
        "command": "python3 src/chrono_ehr/run_study.py --agent-artifact-freshness",
        "purpose": "Confirm core Agent control artifacts are newer than their producers and inputs.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def run_doctor(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for step in DOCTOR_STEPS:
        started = time.perf_counter()
        completed = subprocess.run(
            shlex.split(step["command"]),
            cwd=project_root,
            text=True,
            capture_output=True,
        )
        elapsed = round(time.perf_counter() - started, 2)
        detail = (completed.stdout + completed.stderr).strip()[-2000:]
        rows.append(
            {
                **step,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "detail": detail,
            }
        )
        if completed.returncode != 0:
            break
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["id", "status", "elapsed_seconds", "command", "purpose", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, checks: pd.DataFrame) -> Path:
    table_path = project_root / "outputs" / "tables" / "agent_doctor.csv"
    report_path = project_root / "outputs" / "reports" / "agent_doctor.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Doctor

- Overall status: `{"PASS" if failures.empty and len(checks) == len(DOCTOR_STEPS) else "FAIL"}`
- Steps run: {len(checks)}/{len(DOCTOR_STEPS)}
- Failures: {len(failures)}
- Boundary: local EHR research-tool health check only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    checks = run_doctor(args.project_root)
    report = write_outputs(args.project_root, checks)
    failures = int((checks["status"] != "PASS").sum())
    print(f"Agent doctor steps run: {len(checks)}/{len(DOCTOR_STEPS)}")
    print(f"Failures: {failures}")
    print(f"Wrote {report}")
    if failures or len(checks) != len(DOCTOR_STEPS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
