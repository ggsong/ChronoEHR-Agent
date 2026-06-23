#!/usr/bin/env python3
"""Validate Agent runbook retry/resume recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


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
    path = project_root / "outputs" / "tables" / "agent_runbook_retry_plan.csv"
    plan = read_csv(path)
    rows = [row("retry_plan_exists", "PASS" if not plan.empty else "FAIL", str(path), f"rows={len(plan)}")]
    if plan.empty:
        return pd.DataFrame(rows)
    required = {
        "phase",
        "retry_status",
        "priority",
        "attempt_count",
        "max_recommended_attempts",
        "requires_confirmation",
        "retry_command",
        "post_retry_command",
        "reason",
    }
    missing = sorted(required - set(plan.columns))
    rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(path), "missing=" + ",".join(missing)))
    runnable = plan[plan["retry_status"].astype(str).isin(["RECOVERY_FIRST", "READY_TO_RETRY", "CONFIRMATION_REQUIRED", "REBUILD_STATE"])]
    if not runnable.empty:
        post = runnable["post_retry_command"].astype(str)
        required_refresh_tokens = [
            "--agent-recovery-plan",
            "--agent-runbook-state",
            "--agent-next-tasks",
            "--agent-state",
        ]
        missing_refresh = [token for token in required_refresh_tokens if not post.str.contains(token, regex=False, na=False).all()]
        rows.append(
            row(
                "post_retry_refreshes_recovery_state_and_tasks",
                "PASS" if not missing_refresh else "FAIL",
                str(path),
                "missing=" + ",".join(missing_refresh) if missing_refresh else f"runnable_rows={len(runnable)}",
            )
        )

    safe = plan[plan["phase"].astype(str).eq("phase_1_safe_checks")]
    if not safe.empty:
        rows.append(
            row(
                "safe_retry_no_confirmation",
                "PASS" if safe["requires_confirmation"].astype(str).eq("NO").all() else "FAIL",
                str(path),
                f"rows={len(safe)}",
            )
        )
        runnable_safe = safe[safe["retry_status"].astype(str).eq("READY_TO_RETRY")]
        if not runnable_safe.empty:
            rows.append(
                row(
                    "safe_retry_uses_safe_phase_command",
                    "PASS" if runnable_safe["retry_command"].astype(str).str.contains("--agent-runbook-execute-safe-phase", na=False).all() else "FAIL",
                    str(path),
                    f"rows={len(runnable_safe)}",
                )
            )

    expensive = plan[plan["phase"].astype(str).eq("phase_2_expensive_non_model")]
    if not expensive.empty:
        rows.append(
            row(
                "expensive_retry_requires_confirmation",
                "PASS" if expensive["requires_confirmation"].astype(str).eq("YES").all() else "FAIL",
                str(path),
                f"rows={len(expensive)}",
            )
        )
        runnable_expensive = expensive[expensive["retry_status"].astype(str).eq("CONFIRMATION_REQUIRED")]
        if not runnable_expensive.empty:
            command = runnable_expensive["retry_command"].astype(str)
            rows.append(
                row(
                    "expensive_retry_command_confirmed",
                    "PASS" if command.str.contains("--confirm-expensive", na=False).all() else "FAIL",
                    str(path),
                    f"rows={len(runnable_expensive)}",
                )
            )
            rows.append(
                row(
                    "expensive_retry_refreshes_state",
                    "PASS" if command.str.contains("--agent-runbook-post-phase-refresh", na=False).all() else "FAIL",
                    str(path),
                    f"rows={len(runnable_expensive)}",
                )
            )

    risky = plan[plan["phase"].astype(str).isin(["phase_3_model_requires_confirmation", "phase_4_report_deferred"])]
    if not risky.empty:
        auto_retry = risky[risky["retry_status"].astype(str).isin(["READY_TO_RETRY", "CONFIRMATION_REQUIRED"])]
        rows.append(
            row(
                "model_report_not_auto_retryable",
                "PASS" if auto_retry.empty else "FAIL",
                str(path),
                f"auto_retry={len(auto_retry)}",
            )
        )
    risky_command = plan["retry_command"].astype(str).str.contains("--confirm-model|--run-report-phase|--agent-runbook-execute-model", regex=True, na=False)
    rows.append(row("no_model_report_retry_command", "PASS" if not risky_command.any() else "FAIL", str(path), f"risky={int(risky_command.sum())}"))
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
    table_path = args.project_root / "outputs" / "tables" / "agent_runbook_retry_plan_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_runbook_retry_plan_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Runbook Retry Plan Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This validation checks retry/resume recommendations for local workflow safety boundaries.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent runbook retry-plan checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
