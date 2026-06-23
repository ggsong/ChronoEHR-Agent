#!/usr/bin/env python3
"""Run a configured one-click ChronoEHR-Agent demo workflow."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = DEFAULT_PROJECT / "configs" / "agent_demo_workflows.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workflow", default=None, help="Workflow id from configs/agent_demo_workflows.json.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the workflow plan without executing commands.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_workflow(config: dict[str, Any], workflow_id: str | None) -> tuple[str, dict[str, Any]]:
    selected = workflow_id or config.get("default_workflow")
    workflows = config.get("workflows", {})
    if selected not in workflows:
        available = ", ".join(workflows) or "none"
        raise SystemExit(f"Unknown demo workflow `{selected}`. Available workflows: {available}")
    return selected, workflows[selected]


def run_step(project_root: Path, step: dict[str, Any], plan_only: bool) -> dict[str, Any]:
    command = str(step.get("command", ""))
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    if plan_only:
        return {
            **step,
            "started_at": started_at,
            "status": "PLANNED",
            "returncode": "",
            "duration_seconds": "0.000",
            "detail": "plan-only mode",
        }
    start = time.monotonic()
    completed = subprocess.run(shlex.split(command), cwd=project_root, text=True, capture_output=True)
    duration = time.monotonic() - start
    return {
        **step,
        "started_at": started_at,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "duration_seconds": f"{duration:.3f}",
        "detail": (completed.stdout + completed.stderr).strip()[-1500:],
    }


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["id", "module", "risk_level", "status", "duration_seconds", "command", "purpose"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_outputs(
    project_root: Path,
    workflow_id: str,
    workflow: dict[str, Any],
    results: pd.DataFrame,
    plan_only: bool,
) -> Path:
    table_dir = project_root / "outputs" / "tables"
    report_dir = project_root / "outputs" / "reports"
    state_dir = project_root / "outputs" / "state"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    table_path = table_dir / "agent_demo_workflow_diabetes.csv"
    report_path = report_dir / "agent_demo_workflow_diabetes.md"
    summary_path = state_dir / "agent_demo_workflow_diabetes.json"
    results.to_csv(table_path, index=False)
    failures = results[results["status"].eq("FAIL")]
    summary = {
        "workflow_id": workflow_id,
        "title": workflow.get("title", ""),
        "study_id": workflow.get("study_id", ""),
        "cohort": workflow.get("cohort", ""),
        "plan_only": plan_only,
        "steps": len(results),
        "failures": len(failures),
        "overall_status": "PASS" if failures.empty else "FAIL",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        f"""# Diabetes Agent Demo Workflow

- Workflow id: `{workflow_id}`
- Title: {workflow.get("title", "")}
- Study: `{workflow.get("study_id", "")}`
- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Steps: {len(results)}
- Failures: {len(failures)}
- Mode: `{"plan_only" if plan_only else "executed"}`
- Boundary: {workflow.get("boundary", "Local research workflow only.")}

## What This Demo Shows

This one-click workflow refreshes the diabetes vertical slice, time-window checks, leakage audit, pipeline state, Agent runbook state, next-task plan, control panel, and persistent Agent state. It is not a medical QA, diagnosis, or treatment system.

## Step Results

{markdown_table(results)}
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    workflow_id, workflow = select_workflow(config, args.workflow)
    rows = []
    for step in workflow.get("steps", []):
        print(f"Running demo step: {step.get('id', '')}", flush=True)
        result = run_step(args.project_root, step, args.plan_only)
        rows.append(result)
        if result["status"] == "FAIL":
            break
    results = pd.DataFrame(rows)
    report = write_outputs(args.project_root, workflow_id, workflow, results, args.plan_only)
    print(f"Wrote {report}")
    failures = int((results["status"] == "FAIL").sum()) if not results.empty else 0
    print(f"Demo workflow steps: {len(results)}")
    print(f"Failures: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
