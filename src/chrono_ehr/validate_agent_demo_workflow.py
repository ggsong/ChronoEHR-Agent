#!/usr/bin/env python3
"""Validate configured one-click Agent demo workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = DEFAULT_PROJECT / "configs" / "agent_demo_workflows.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    workflows = config.get("workflows", {})
    rows.append(row("workflow_config_exists", "PASS" if workflows else "FAIL", "configs/agent_demo_workflows.json", f"workflows={len(workflows)}"))
    diabetes = workflows.get("diabetes_v0", {})
    steps = diabetes.get("steps", [])
    rows.append(row("diabetes_workflow_defined", "PASS" if steps else "FAIL", "configs/agent_demo_workflows.json", f"steps={len(steps)}"))
    if steps:
        commands = pd.Series([str(step.get("command", "")) for step in steps])
        risky = commands[commands.str.contains("--confirm-expensive|--calibrated|--random-forest-baseline|--gradient-boosting", regex=True, na=False)]
        rows.append(row("diabetes_workflow_safe_by_default", "PASS" if risky.empty else "FAIL", "configs/agent_demo_workflows.json", f"risky_commands={len(risky)}"))
        has_pipeline = commands.str.contains("--study mimic_iv_3_1_diabetes_readmission --skip-existing --no-expensive", regex=False).any()
        rows.append(row("diabetes_safe_pipeline_step_present", "PASS" if has_pipeline else "FAIL", "configs/agent_demo_workflows.json", f"present={has_pipeline}"))
        has_leakage = commands.str.contains("--leakage-gate", regex=False).any()
        rows.append(row("leakage_gate_step_present", "PASS" if has_leakage else "FAIL", "configs/agent_demo_workflows.json", f"present={has_leakage}"))
        has_state = commands.str.contains("--agent-state", regex=False).any()
        rows.append(row("agent_state_step_present", "PASS" if has_state else "FAIL", "configs/agent_demo_workflows.json", f"present={has_state}"))

    output = read_csv(project_root / "outputs" / "tables" / "agent_demo_workflow_diabetes.csv")
    rows.append(row("diabetes_workflow_output_exists", "PASS" if not output.empty else "FAIL", "outputs/tables/agent_demo_workflow_diabetes.csv", f"rows={len(output)}"))
    if not output.empty:
        failures = int((output["status"] == "FAIL").sum()) if "status" in output else -1
        rows.append(row("diabetes_workflow_last_run_passed", "PASS" if failures == 0 else "FAIL", "outputs/tables/agent_demo_workflow_diabetes.csv", f"failures={failures}"))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    checks = audit(args.project_root, config)
    table_path = args.project_root / "outputs" / "tables" / "agent_demo_workflow_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_demo_workflow_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Demo Workflow Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This validation checks the configured one-click diabetes demo workflow and its most recent local run.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent demo workflow checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
