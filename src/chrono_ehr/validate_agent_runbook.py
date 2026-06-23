#!/usr/bin/env python3
"""Validate Agent runbook phase policies and execution boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
EXPECTED_POLICIES = {
    "safe": "may_execute_with_execute_safe_phase",
    "expensive": "plan_only_requires_user_or_night_run_confirmation",
    "model": "plan_only_requires_explicit_model_confirmation",
    "report": "deferred_while_polishing_agent",
}


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


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path) -> pd.DataFrame:
    runbook_path = project_root / "outputs" / "tables" / "agent_runbook.csv"
    execution_path = project_root / "outputs" / "tables" / "agent_runbook_execution.csv"
    phase_summary_path = project_root / "outputs" / "tables" / "agent_runbook_phase_summary.csv"
    runbook = read_csv(runbook_path)
    execution = read_csv(execution_path)
    phase_summary = read_csv(phase_summary_path)
    rows = [
        row("runbook_exists", "PASS" if not runbook.empty else "FAIL", str(runbook_path), f"rows={len(runbook)}"),
        row("execution_table_exists", "PASS" if execution_path.exists() else "FAIL", str(execution_path), f"rows={len(execution)}"),
        row(
            "phase_summary_exists",
            "PASS" if not phase_summary.empty else "FAIL",
            str(phase_summary_path),
            f"rows={len(phase_summary)}",
        ),
    ]
    if not runbook.empty:
        required = {"phase", "risk_level", "execution_policy", "command"}
        missing = sorted(required - set(runbook.columns))
        rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(runbook_path), "missing=" + ",".join(missing)))
        for risk, policy in EXPECTED_POLICIES.items():
            subset = runbook[runbook["risk_level"].eq(risk)]
            if subset.empty:
                continue
            bad = subset[~subset["execution_policy"].eq(policy)]
            rows.append(row(f"policy:{risk}", "PASS" if bad.empty else "FAIL", str(runbook_path), f"bad_rows={len(bad)}"))
        if not execution.empty and "risk_level" in execution:
            executed = execution[execution["status"].isin(["PASS", "FAIL"])]
            forbidden_executed = executed[execution["risk_level"].isin(["model", "report"])]
            rows.append(
                row(
                    "model_report_not_executed",
                    "PASS" if forbidden_executed.empty else "FAIL",
                    str(execution_path),
                    f"forbidden_executed={len(forbidden_executed)}",
                )
            )
            expensive_executed = executed[execution["risk_level"].eq("expensive")]
            if not expensive_executed.empty:
                bad_confirmation = expensive_executed[
                    ~expensive_executed.get("confirmation", pd.Series(index=expensive_executed.index, dtype=str)).eq("confirmed_expensive")
                ]
                rows.append(
                    row(
                        "expensive_requires_confirmation",
                        "PASS" if bad_confirmation.empty else "FAIL",
                        str(execution_path),
                        f"bad_confirmation={len(bad_confirmation)}",
                    )
                )
            failed = execution[execution["status"].eq("FAIL")]
            rows.append(row("execution_failures_absent", "PASS" if failed.empty else "FAIL", str(execution_path), f"failures={len(failed)}"))
        if not phase_summary.empty:
            required_phases = {
                "phase_1_safe_checks",
                "phase_2_expensive_non_model",
                "phase_3_model_requires_confirmation",
                "phase_4_report_deferred",
            }
            missing_phases = sorted(required_phases - set(phase_summary["phase"].astype(str)))
            rows.append(
                row(
                    "phase_summary_complete",
                    "PASS" if not missing_phases else "FAIL",
                    str(phase_summary_path),
                    "missing=" + ",".join(missing_phases),
                )
            )
            risky_completed = phase_summary[
                phase_summary["risk_level"].isin(["model", "report"]) & phase_summary["status"].eq("completed")
            ]
            rows.append(
                row(
                    "phase_summary_keeps_model_report_locked",
                    "PASS" if risky_completed.empty else "FAIL",
                    str(phase_summary_path),
                    f"completed_locked_phases={len(risky_completed)}",
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
    table_path = args.project_root / "outputs" / "tables" / "agent_runbook_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_runbook_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Runbook Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This validation checks whether phased runbooks preserve execution boundaries. It is not a clinical decision check.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent runbook validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
