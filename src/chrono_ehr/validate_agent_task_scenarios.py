#!/usr/bin/env python3
"""Validate the Agent task scenario library."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agent_task_router import SCENARIO_COLUMNS, TASK_SCENARIOS, infer_budget_mode, infer_goal_type, infer_risk_mode, infer_task_scenario
from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_SCENARIOS = {
    "low_quota_self_check",
    "night_run_safe_plus_deferred",
    "external_readiness_first",
    "validation_first",
    "agent_control_focus",
    "model_or_report_explicit",
}
ROUTING_CASES = [
    ("我快没额度了，只剩2%，做一点简单自检", "low_quota_self_check"),
    ("我要去睡觉，可以多跑一些但不要重训模型", "night_run_safe_plus_deferred"),
    ("继续做 eICU 外部验证准备，不要重训模型", "external_readiness_first"),
    ("先暂停一下，做一下检查。从主线来看有没有偏离？", "validation_first"),
    ("先完善 Agent 控制层，不要做汇报材料", "agent_control_focus"),
    ("重训 random forest baseline", "model_or_report_explicit"),
]


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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "agent_task_scenario_library.csv"
    examples_path = project_root / "outputs" / "tables" / "agent_task_scenario_examples.csv"
    report_path = project_root / "outputs" / "reports" / "agent_task_scenario_library.md"
    table = read_csv(table_path)
    examples = read_csv(examples_path)
    report = read_text(report_path)
    rows = [
        row("scenario_library_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("scenario_examples_table_exists", "PASS" if not examples.empty else "FAIL", str(examples_path), f"rows={len(examples)}"),
        row("scenario_library_report_exists", "PASS" if bool(report) else "FAIL", str(report_path), f"chars={len(report)}"),
    ]
    missing_columns = sorted(set(SCENARIO_COLUMNS) - set(table.columns))
    rows.append(row("scenario_library_columns", "PASS" if not missing_columns else "FAIL", str(table_path), "missing=" + ",".join(missing_columns)))

    configured_ids = set(TASK_SCENARIOS)
    table_ids = set(table["scenario_id"].astype(str)) if "scenario_id" in table else set()
    rows.append(row("expected_scenarios_configured", "PASS" if EXPECTED_SCENARIOS == configured_ids else "FAIL", "src/chrono_ehr/agent_task_router.py", f"configured={sorted(configured_ids)}"))
    rows.append(row("expected_scenarios_exported", "PASS" if EXPECTED_SCENARIOS == table_ids else "FAIL", str(table_path), f"exported={sorted(table_ids)}"))

    example_required = {
        "scenario_id",
        "example_task",
        "inferred_scenario",
        "goal_type",
        "risk_mode",
        "budget_mode",
        "selected_actions",
        "deferred_actions",
        "top_selected_action_ids",
        "top_deferred_action_ids",
    }
    missing_example_columns = sorted(example_required - set(examples.columns))
    rows.append(row("scenario_examples_columns", "PASS" if not missing_example_columns else "FAIL", str(examples_path), "missing=" + ",".join(missing_example_columns)))
    if not examples.empty and {"scenario_id", "inferred_scenario"}.issubset(examples.columns):
        mismatches = int((examples["scenario_id"].astype(str) != examples["inferred_scenario"].astype(str)).sum())
        rows.append(row("scenario_examples_match_inference", "PASS" if mismatches == 0 else "FAIL", str(examples_path), f"mismatches={mismatches}"))
    else:
        rows.append(row("scenario_examples_match_inference", "FAIL", str(examples_path), "missing columns or empty examples"))

    if not table.empty:
        required_text_columns = ["title", "when_to_use", "execution_style", "auto_run_policy", "next_step_hint", "example_task"]
        empty_cells = int(table[required_text_columns].fillna("").astype(str).eq("").sum().sum()) if set(required_text_columns).issubset(table.columns) else 1
        rows.append(row("scenario_text_fields_complete", "PASS" if empty_cells == 0 else "FAIL", str(table_path), f"empty_cells={empty_cells}"))
    else:
        rows.append(row("scenario_text_fields_complete", "FAIL", str(table_path), "empty table"))

    for task, expected in ROUTING_CASES:
        goal = infer_goal_type(task)
        risk = infer_risk_mode(task, "auto")
        budget = infer_budget_mode(task)
        actual = infer_task_scenario(task, goal, risk, budget)["scenario_id"]
        rows.append(
            row(
                "scenario_routing_" + expected,
                "PASS" if actual == expected else "FAIL",
                "src/chrono_ehr/agent_task_router.py",
                f"task={task}; actual={actual}; goal={goal}; risk={risk}; budget={budget}",
            )
        )

    rows.append(
        row(
            "scenario_report_boundary",
            "PASS" if "Example Dry Runs" in report and "not medical QA" in report and "treatment recommendation" in report else "FAIL",
            str(report_path),
            "expects example dry runs and non-clinical boundary wording",
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
    checks = audit(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_task_scenario_library_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_task_scenario_library_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Task Scenario Library Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local research workflow scenarios only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent task scenario checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
