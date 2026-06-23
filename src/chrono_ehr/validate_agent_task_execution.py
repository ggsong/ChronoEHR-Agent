#!/usr/bin/env python3
"""Validate the latest natural-language Agent task execution outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agent_task_router import EXECUTION_COLUMNS, POST_REFRESH_COLUMNS, SCENARIO_COLUMNS
from mimic_diabetes_baseline import DEFAULT_PROJECT


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
    plan_path = project_root / "outputs" / "tables" / "agent_task_plan.csv"
    deferred_path = project_root / "outputs" / "tables" / "agent_task_deferred_actions.csv"
    scenario_path = project_root / "outputs" / "tables" / "agent_task_scenario.csv"
    execution_path = project_root / "outputs" / "tables" / "agent_task_execution.csv"
    refresh_path = project_root / "outputs" / "tables" / "agent_task_post_run_refresh.csv"
    report_path = project_root / "outputs" / "reports" / "agent_task_plan.md"

    plan = read_csv(plan_path)
    deferred = read_csv(deferred_path)
    scenario = read_csv(scenario_path)
    execution = read_csv(execution_path)
    refresh = read_csv(refresh_path)
    report = read_text(report_path)

    rows = [
        row("task_plan_exists", "PASS" if not plan.empty else "FAIL", str(plan_path), f"rows={len(plan)}"),
        row("task_scenario_table_exists", "PASS" if not scenario.empty else "FAIL", str(scenario_path), f"rows={len(scenario)}"),
        row("task_execution_table_exists", "PASS" if execution_path.exists() else "FAIL", str(execution_path), f"rows={len(execution)}"),
        row("task_post_refresh_table_exists", "PASS" if refresh_path.exists() else "FAIL", str(refresh_path), f"rows={len(refresh)}"),
        row("task_report_exists", "PASS" if bool(report) else "FAIL", str(report_path), f"chars={len(report)}"),
    ]

    missing_plan = sorted({"id", "risk_level", "phase", "execution_policy", "command"} - set(plan.columns))
    rows.append(row("task_plan_required_columns", "PASS" if not missing_plan else "FAIL", str(plan_path), "missing=" + ",".join(missing_plan)))

    missing_scenario = sorted(set(SCENARIO_COLUMNS) - set(scenario.columns))
    rows.append(row("task_scenario_required_columns", "PASS" if not missing_scenario else "FAIL", str(scenario_path), "missing=" + ",".join(missing_scenario)))

    missing_execution = sorted(set(EXECUTION_COLUMNS) - set(execution.columns))
    rows.append(
        row(
            "task_execution_required_columns",
            "PASS" if not missing_execution else "FAIL",
            str(execution_path),
            "missing=" + ",".join(missing_execution),
        )
    )

    missing_refresh = sorted(set(POST_REFRESH_COLUMNS) - set(refresh.columns))
    rows.append(
        row(
            "task_post_refresh_required_columns",
            "PASS" if not missing_refresh else "FAIL",
            str(refresh_path),
            "missing=" + ",".join(missing_refresh),
        )
    )

    if not execution.empty and "risk_level" in execution:
        non_safe = sorted(set(execution.loc[execution["risk_level"].astype(str).ne("safe"), "id"].astype(str)))
        rows.append(row("executed_actions_are_safe_only", "PASS" if not non_safe else "FAIL", str(execution_path), "non_safe=" + ",".join(non_safe)))
        failures = int((execution["status"].astype(str) == "FAIL").sum()) if "status" in execution else len(execution)
        rows.append(row("executed_actions_all_pass", "PASS" if failures == 0 else "FAIL", str(execution_path), f"failures={failures}"))
    else:
        rows.append(row("executed_actions_are_safe_only", "PASS", str(execution_path), "no actions executed"))
        rows.append(row("executed_actions_all_pass", "PASS", str(execution_path), "no actions executed"))

    if not refresh.empty:
        refresh_failures = int((refresh["status"].astype(str) == "FAIL").sum()) if "status" in refresh else len(refresh)
        rows.append(row("post_run_refresh_all_pass", "PASS" if refresh_failures == 0 else "FAIL", str(refresh_path), f"failures={refresh_failures}"))
        refresh_ids = set(refresh["id"].astype(str)) if "id" in refresh else set()
        required_refresh = {"agent_recovery_plan", "agent_next_tasks", "agent_state", "agent_handoff_checklist", "agent_command_lint"}
        missing_refresh_ids = sorted(required_refresh - refresh_ids)
        rows.append(
            row(
                "post_run_refresh_core_steps_present",
                "PASS" if not missing_refresh_ids else "FAIL",
                str(refresh_path),
                "missing=" + ",".join(missing_refresh_ids),
            )
        )
    else:
        rows.append(row("post_run_refresh_all_pass", "PASS", str(refresh_path), "post-run refresh not requested"))
        rows.append(row("post_run_refresh_core_steps_present", "PASS", str(refresh_path), "post-run refresh not requested"))

    if not deferred.empty and {"risk_level", "execution_policy"}.issubset(deferred.columns):
        high_risk = deferred["risk_level"].astype(str).isin(["expensive", "model", "report"])
        policy_ok = deferred.loc[high_risk, "execution_policy"].astype(str).ne("auto_safe_if_requested").all()
        rows.append(
            row(
                "deferred_high_risk_actions_not_auto_safe",
                "PASS" if policy_ok else "FAIL",
                str(deferred_path),
                f"high_risk_deferred={int(high_risk.sum())}",
            )
        )
    else:
        rows.append(row("deferred_high_risk_actions_not_auto_safe", "PASS", str(deferred_path), f"rows={len(deferred)}"))

    rows.append(
        row(
            "task_report_documents_post_refresh",
            "PASS" if "Post-Run Refresh" in report and "Task Scenario" in report and "no medical QA" in report else "FAIL",
            str(report_path),
            "expects Task Scenario, Post-Run Refresh, and non-clinical boundary",
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
    table_path = args.project_root / "outputs" / "tables" / "agent_task_execution_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_task_execution_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Task Execution Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local Agent task execution only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent task execution checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
