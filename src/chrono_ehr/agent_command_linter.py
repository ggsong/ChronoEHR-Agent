#!/usr/bin/env python3
"""Lint ChronoEHR-Agent command strings across configs and state outputs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from agent_self_check import CHECKS
from mimic_diabetes_baseline import DEFAULT_PROJECT


COMMAND_PREFIX = ["python3", "src/chrono_ehr/run_study.py"]
COMPOUND_TOKENS = {"&&", ";", "||"}
AUTO_EXECUTED_SOURCES = {
    "agent_action_catalog",
    "agent_demo_workflow",
    "agent_entrypoints",
    "agent_self_check",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
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


def known_run_study_flags(project_root: Path) -> set[str]:
    source = (project_root / "src" / "chrono_ehr" / "run_study.py").read_text(encoding="utf-8")
    return set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', source))


def command_rows(project_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    entrypoints = read_json(project_root / "configs" / "agent_entrypoints.json")
    for group in entrypoints.get("groups", []):
        for item in group.get("commands", []):
            rows.append(
                {
                    "source": "agent_entrypoints",
                    "id": str(item.get("id", "")),
                    "risk_level": str(item.get("risk_level", "")),
                    "command": str(item.get("command", "")),
                    "compound_allowed": "False",
                }
            )

    catalog = read_json(project_root / "configs" / "agent_action_catalog.json")
    for item in catalog.get("actions", []):
        rows.append(
            {
                "source": "agent_action_catalog",
                "id": str(item.get("id", "")),
                "risk_level": str(item.get("risk_level", "")),
                "command": str(item.get("command", "")),
                "compound_allowed": "False",
            }
        )

    demo = read_json(project_root / "configs" / "agent_demo_workflows.json")
    for workflow_id, workflow in demo.get("workflows", {}).items():
        for item in workflow.get("steps", []):
            rows.append(
                {
                    "source": "agent_demo_workflow",
                    "id": f"{workflow_id}:{item.get('id', '')}",
                    "risk_level": str(item.get("risk_level", "")),
                    "command": str(item.get("command", "")),
                    "compound_allowed": "False",
                }
            )

    for item in CHECKS:
        rows.append(
            {
                "source": "agent_self_check",
                "id": str(item.get("id", "")),
                "risk_level": "safe",
                "command": str(item.get("command", "")),
                "compound_allowed": "False",
            }
        )

    for path, source, id_col, compound_allowed in [
        (project_root / "outputs" / "tables" / "agent_next_tasks.csv", "agent_next_tasks", "next_task", True),
        (project_root / "outputs" / "tables" / "next_study_action_plan.csv", "next_study_action_plan", "study_id", True),
        (project_root / "outputs" / "tables" / "agent_recovery_plan.csv", "agent_recovery_plan", "failed_item", False),
        (project_root / "outputs" / "tables" / "agent_runbook_retry_plan.csv", "agent_runbook_retry_plan", "phase", False),
    ]:
        df = read_csv(path)
        command_col = "command" if "command" in df.columns else "recovery_command" if "recovery_command" in df.columns else "retry_command"
        if df.empty or command_col not in df.columns:
            continue
        for item in df.to_dict(orient="records"):
            command = item.get(command_col, "")
            if pd.isna(command) or not str(command).strip():
                continue
            rows.append(
                {
                    "source": source,
                    "id": str(item.get(id_col, "")),
                    "risk_level": str(item.get("risk_level", "")),
                    "command": str(command),
                    "compound_allowed": str(bool(compound_allowed)),
                }
            )

    state = read_json(project_root / "outputs" / "state" / "agent_state.json")
    for index, command in enumerate(state.get("recommended_next_commands", []), start=1):
        if pd.isna(command) or not str(command).strip():
            continue
        rows.append(
            {
                "source": "agent_state_recommended_commands",
                "id": f"recommended_{index}",
                "risk_level": "safe",
                "command": str(command),
                "compound_allowed": "False",
            }
        )
    return rows


def split_components(command: str) -> list[str]:
    parts = re.split(r"\s*(?:&&|\|\||;)\s*", command.strip())
    return [part for part in parts if part]


def lint_command(item: dict[str, str], flags: set[str]) -> list[dict[str, str]]:
    source = item["source"]
    command = item["command"]
    compound_allowed = item["compound_allowed"].lower() == "true"
    tokens = shlex.split(command) if command else []
    rows: list[dict[str, str]] = []

    has_compound = any(token in COMPOUND_TOKENS for token in tokens) or bool(re.search(r"\s(&&|\|\||;)\s", command))
    if has_compound and source in AUTO_EXECUTED_SOURCES:
        rows.append({**item, "check": "no_compound_for_auto_source", "status": "FAIL", "detail": "auto-executed source must not use shell chaining"})
    elif has_compound and not compound_allowed:
        rows.append({**item, "check": "compound_allowed", "status": "FAIL", "detail": "compound command is not allowed for this source"})
    else:
        rows.append({**item, "check": "compound_policy", "status": "PASS", "detail": f"compound={has_compound}; allowed={compound_allowed}"})

    components = split_components(command)
    if not components:
        rows.append({**item, "check": "command_nonempty", "status": "FAIL", "detail": "empty command"})
        return rows

    for component_index, component in enumerate(components, start=1):
        component_tokens = shlex.split(component)
        prefix_ok = component_tokens[:2] == COMMAND_PREFIX
        rows.append(
            {
                **item,
                "check": f"component_{component_index}_prefix",
                "status": "PASS" if prefix_ok else "FAIL",
                "detail": " ".join(component_tokens[:2]) if component_tokens else "empty component",
            }
        )
        used_flags = [token for token in component_tokens[2:] if token.startswith("--")]
        unknown = sorted(set(used_flags) - flags)
        rows.append(
            {
                **item,
                "check": f"component_{component_index}_flags_known",
                "status": "PASS" if not unknown else "FAIL",
                "detail": "unknown=" + ",".join(unknown) if unknown else "flags=" + ",".join(used_flags),
            }
        )
    return rows


def lint(project_root: Path) -> pd.DataFrame:
    flags = known_run_study_flags(project_root)
    rows: list[dict[str, str]] = []
    for item in command_rows(project_root):
        rows.extend(lint_command(item, flags))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["source", "id", "check", "status", "command", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = lint(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_command_lint.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_command_lint.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Command Lint

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Commands linted: {checks[["source", "id", "command"]].drop_duplicates().shape[0]}
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local workflow commands only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent command lint checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
