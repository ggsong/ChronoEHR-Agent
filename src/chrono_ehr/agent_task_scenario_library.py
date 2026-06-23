#!/usr/bin/env python3
"""Export the natural-language Agent task scenario library."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent_task_router import (
    DEFAULT_CATALOG,
    SCENARIO_COLUMNS,
    TASK_SCENARIOS,
    deferred_actions,
    infer_budget_mode,
    infer_goal_type,
    infer_risk_mode,
    infer_task_scenario,
    read_json,
    select_actions,
)
from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    return parser.parse_args()


def scenario_rows() -> pd.DataFrame:
    rows = []
    for scenario_id, scenario in TASK_SCENARIOS.items():
        rows.append({"scenario_id": scenario_id, **scenario})
    return pd.DataFrame(rows, columns=SCENARIO_COLUMNS)


def example_rows(catalog: dict) -> pd.DataFrame:
    rows = []
    for scenario_id, scenario in TASK_SCENARIOS.items():
        task = scenario["example_task"]
        goal = infer_goal_type(task)
        risk = infer_risk_mode(task, "auto")
        budget = infer_budget_mode(task)
        inferred = infer_task_scenario(task, goal, risk, budget)["scenario_id"]
        selected = select_actions(catalog, goal, risk, budget)
        deferred = deferred_actions(catalog, goal, risk)
        selected_ids = ", ".join(selected["id"].astype(str).head(5).tolist()) if not selected.empty and "id" in selected else ""
        deferred_ids = ", ".join(deferred["id"].astype(str).head(5).tolist()) if not deferred.empty and "id" in deferred else ""
        rows.append(
            {
                "scenario_id": scenario_id,
                "example_task": task,
                "inferred_scenario": inferred,
                "goal_type": goal,
                "risk_mode": risk,
                "budget_mode": budget,
                "selected_actions": len(selected),
                "deferred_actions": len(deferred),
                "top_selected_action_ids": selected_ids,
                "top_deferred_action_ids": deferred_ids,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    catalog = read_json(args.catalog)
    df = scenario_rows()
    examples = example_rows(catalog)
    table_path = args.project_root / "outputs" / "tables" / "agent_task_scenario_library.csv"
    examples_path = args.project_root / "outputs" / "tables" / "agent_task_scenario_examples.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_task_scenario_library.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)
    examples.to_csv(examples_path, index=False)
    report_path.write_text(
        f"""# Agent Task Scenario Library

- Scenarios: {len(df)}
- Scope: local EHR research workflow planning for ChronoEHR-Agent.
- Boundary: not medical QA, diagnosis, or treatment recommendation.

This library explains how `--agent-task` interprets common user situations before choosing commands. It is meant for research workflow control, especially when the user has low quota, wants an overnight run, is preparing external data, or wants validation-first checks.

## Scenario Table

{markdown_table(df)}

## Example Dry Runs

These examples show how each scenario is interpreted before any command is executed.

{markdown_table(examples)}
""",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    print(df[["scenario_id", "title"]].to_string(index=False))


if __name__ == "__main__":
    main()
