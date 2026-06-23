#!/usr/bin/env python3
"""Run a no-data agent behavior trace for public ChronoEHR-Agent demos."""

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
DEFAULT_TASK = "check whether the synthetic EHR demo is ready for release"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    return parser.parse_args()


def run_step(project_root: Path, step_id: str, role: str, command: str, rationale: str) -> dict[str, Any]:
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    start = time.monotonic()
    completed = subprocess.run(shlex.split(command), cwd=project_root, text=True, capture_output=True)
    duration = time.monotonic() - start
    detail = (completed.stdout + completed.stderr).strip()
    return {
        "step_id": step_id,
        "role": role,
        "rationale": rationale,
        "command": command,
        "started_at": started_at,
        "duration_seconds": round(duration, 3),
        "returncode": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "detail": detail[-1200:],
    }


def summarize_artifacts(project_root: Path) -> dict[str, Any]:
    demo_dir = project_root / "outputs" / "demo"
    contract_path = demo_dir / "synthetic_ehr_contract.csv"
    cohort_path = demo_dir / "synthetic_cohort.csv"
    metrics_path = demo_dir / "synthetic_demo_metrics.csv"
    summary: dict[str, Any] = {
        "synthetic_raw_tables": sorted(path.name for path in (demo_dir / "raw").glob("*.csv")) if (demo_dir / "raw").exists() else [],
        "contract_checks": 0,
        "contract_failures": 0,
        "cohort_rows": 0,
        "metric_rows": 0,
    }
    if contract_path.exists():
        contract = pd.read_csv(contract_path)
        summary["contract_checks"] = int(len(contract))
        summary["contract_failures"] = int((contract["status"] != "PASS").sum()) if "status" in contract.columns else len(contract)
    if cohort_path.exists():
        summary["cohort_rows"] = int(len(pd.read_csv(cohort_path)))
    if metrics_path.exists():
        summary["metric_rows"] = int(len(pd.read_csv(metrics_path)))
    return summary


def write_markdown(trace: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Agent Demo Trace",
        "",
        f"- Task: `{trace['task']}`",
        f"- Interpreted intent: `{trace['interpreted_intent']}`",
        f"- Risk policy: `{trace['risk_policy']}`",
        f"- Overall status: `{trace['overall_status']}`",
        "",
        "## Steps",
        "",
        "| step | role | status | command | rationale |",
        "|---|---|---|---|---|",
    ]
    for step in trace["steps"]:
        lines.append(
            f"| {step['step_id']} | {step['role']} | {step['status']} | `{step['command']}` | {step['rationale']} |"
        )
    artifacts = trace["artifact_summary"]
    lines.extend(
        [
            "",
            "## Artifact Summary",
            "",
            f"- Synthetic raw tables: {', '.join(artifacts['synthetic_raw_tables'])}",
            f"- Contract checks: {artifacts['contract_checks']}",
            f"- Contract failures: {artifacts['contract_failures']}",
            f"- Cohort rows: {artifacts['cohort_rows']}",
            f"- Metric rows: {artifacts['metric_rows']}",
            "",
            "## Boundary",
            "",
            "This trace demonstrates local research workflow control only. It does not perform diagnosis, treatment recommendation, or patient-care decision support.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    table_dir = args.project_root / "outputs" / "tables"
    report_dir = args.project_root / "outputs" / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        run_step(
            args.project_root,
            "S1",
            "planner",
            "python3 src/chrono_ehr/run_study.py --list",
            "Inspect registered studies before choosing a no-data route.",
        ),
        run_step(
            args.project_root,
            "S2",
            "executor",
            "python3 src/chrono_ehr/run_study.py --synthetic-demo",
            "Generate synthetic EHR raw tables and derived cohort artifacts without controlled data.",
        ),
        run_step(
            args.project_root,
            "S3",
            "auditor",
            "python3 src/chrono_ehr/run_study.py --validate-synthetic-demo",
            "Validate raw-table schema, temporal boundaries, labels, metrics, and data contract outputs.",
        ),
        run_step(
            args.project_root,
            "S4",
            "release_guard",
            "python3 scripts/release_audit.py --project-root .",
            "Confirm public release files avoid local absolute paths and controlled-data directories are ignored.",
        ),
    ]
    overall_status = "PASS" if all(step["status"] == "PASS" for step in steps) else "FAIL"
    trace = {
        "task": args.task,
        "interpreted_intent": "public no-data readiness check",
        "risk_policy": "safe-only; no real clinical data, model training, report export, or expensive phase",
        "overall_status": overall_status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "steps": steps,
        "artifact_summary": summarize_artifacts(args.project_root),
    }
    (table_dir / "agent_demo_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    pd.DataFrame(steps).to_csv(table_dir / "agent_demo_trace.csv", index=False)
    write_markdown(trace, report_dir / "agent_demo_trace.md")
    print(f"Agent demo trace {overall_status}: {len(steps)} steps.")
    if overall_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
