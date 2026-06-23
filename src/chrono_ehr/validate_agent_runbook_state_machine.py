#!/usr/bin/env python3
"""Validate runbook state-machine phase gates."""

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
    state_path = project_root / "outputs" / "tables" / "agent_runbook_state_machine.csv"
    history_path = project_root / "outputs" / "state" / "agent_runbook_phase_history.csv"
    state = read_csv(state_path)
    history = read_csv(history_path)
    rows = [
        row("state_machine_exists", "PASS" if not state.empty else "FAIL", str(state_path), f"rows={len(state)}"),
        row("phase_history_exists", "PASS" if not history.empty else "FAIL", str(history_path), f"rows={len(history)}"),
    ]
    if state.empty:
        return pd.DataFrame(rows)

    required = {
        "phase",
        "risk_level",
        "phase_status",
        "gate_status",
        "can_execute_now",
        "confirmation_required",
        "attempt_count",
        "next_allowed_command",
    }
    missing = sorted(required - set(state.columns))
    rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(state_path), "missing=" + ",".join(missing)))

    expensive = state[state["risk_level"].astype(str).eq("expensive")]
    if not expensive.empty:
        commands = expensive["next_allowed_command"].astype(str)
        rows.append(
            row(
                "expensive_gate_requires_confirmation",
                "PASS"
                if expensive["confirmation_required"].astype(str).eq("YES").all()
                and commands.str.contains("--confirm-expensive", na=False).all()
                else "FAIL",
                str(state_path),
                f"rows={len(expensive)}",
            )
        )
        rows.append(
            row(
                "expensive_gate_refreshes_after_phase",
                "PASS" if commands.str.contains("--agent-runbook-post-phase-refresh", na=False).all() else "FAIL",
                str(state_path),
                f"rows={len(expensive)}",
            )
        )

    locked = state[state["risk_level"].astype(str).isin(["model", "report"])]
    if not locked.empty:
        auto_open = locked[locked["can_execute_now"].astype(str).isin(["YES", "YES_WITH_CONFIRMATION"])]
        rows.append(
            row(
                "model_report_not_auto_open",
                "PASS" if auto_open.empty else "FAIL",
                str(state_path),
                f"auto_open={len(auto_open)}",
            )
        )

    bad_commands = state[
        state["next_allowed_command"].astype(str).str.contains("--agent-runbook-execute-model|--run-report-phase", regex=True, na=False)
    ]
    rows.append(
        row(
            "no_unimplemented_phase_commands",
            "PASS" if bad_commands.empty else "FAIL",
            str(state_path),
            f"bad_commands={len(bad_commands)}",
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
    table_path = args.project_root / "outputs" / "tables" / "agent_runbook_state_machine_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_runbook_state_machine_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Runbook State-Machine Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This validation checks whether runbook phase gates remain explicit and bounded.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent runbook state-machine checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
