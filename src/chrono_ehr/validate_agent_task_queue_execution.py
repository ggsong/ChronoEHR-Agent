#!/usr/bin/env python3
"""Validate Agent task queue execution boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agent_task_queue_runner import EXECUTION_COLUMNS, HISTORY_COLUMNS
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
    execution_path = project_root / "outputs" / "tables" / "agent_task_queue_execution.csv"
    history_path = project_root / "outputs" / "state" / "agent_task_queue_execution_history.csv"
    report_path = project_root / "outputs" / "reports" / "agent_task_queue_execution.md"
    execution = read_csv(execution_path)
    history = read_csv(history_path)
    report = read_text(report_path)
    rows = [
        row("queue_execution_table_exists", "PASS" if not execution.empty else "FAIL", str(execution_path), f"rows={len(execution)}"),
        row("queue_execution_history_exists", "PASS" if not history.empty else "FAIL", str(history_path), f"rows={len(history)}"),
        row("queue_execution_report_exists", "PASS" if bool(report) else "FAIL", str(report_path), f"chars={len(report)}"),
    ]
    if execution.empty:
        return pd.DataFrame(rows)
    missing = sorted(set(EXECUTION_COLUMNS) - set(execution.columns))
    rows.append(row("queue_execution_columns", "PASS" if not missing else "FAIL", str(execution_path), "missing=" + ",".join(missing)))
    missing_history = sorted(set(HISTORY_COLUMNS) - set(history.columns))
    rows.append(row("queue_execution_history_columns", "PASS" if not missing_history else "FAIL", str(history_path), "missing=" + ",".join(missing_history)))
    ready_like = execution[
        execution["queue_status"].astype(str).eq("READY_SAFE_AUTO")
        | execution["execution_status"].astype(str).isin(["PASS", "FAIL", "PLANNED_SAFE_AUTO"])
    ]
    ready_like_fingerprints = (
        ready_like.empty
        or (
            ready_like["cooldown_fingerprint"].fillna("").astype(str).str.len().ge(32).all()
            and ready_like["cooldown_fingerprint_status"].fillna("").astype(str).ne("").all()
        )
    )
    rows.append(
        row(
            "ready_queue_execution_rows_have_cooldown_fingerprint",
            "PASS" if ready_like_fingerprints else "FAIL",
            str(execution_path),
            f"ready_like_rows={len(ready_like)}",
        )
    )
    scenario_recorded = "scenario_id" in execution.columns and execution["scenario_id"].fillna("").astype(str).ne("").all()
    rows.append(
        row(
            "queue_execution_scenario_recorded",
            "PASS" if scenario_recorded else "FAIL",
            str(execution_path),
            "expects scenario_id for filtered queue execution recovery",
        )
    )
    manual = execution[execution["queue_status"].astype(str).eq("WAITING_CONFIRMATION")]
    rows.append(
        row(
            "manual_confirmation_items_not_executed",
            "PASS" if manual.empty or manual["execution_status"].astype(str).eq("SKIPPED_CONFIRMATION_REQUIRED").all() else "FAIL",
            str(execution_path),
            f"manual_rows={len(manual)}",
        )
    )
    executed = execution[execution["execution_status"].astype(str).isin(["PASS", "FAIL"])]
    non_ready_executed = executed[~executed["queue_status"].astype(str).eq("READY_SAFE_AUTO")]
    rows.append(
        row(
            "only_ready_safe_auto_items_execute",
            "PASS" if non_ready_executed.empty else "FAIL",
            str(execution_path),
            f"non_ready_executed={len(non_ready_executed)}",
        )
    )
    high_risk_auto = execution["command"].fillna("").astype(str).str.contains("--confirm-expensive|--agent-runbook-execute-expensive-phase|--agent-runbook-execute-model", regex=True, na=False) & execution["execution_status"].astype(str).isin(["PASS", "FAIL", "PLANNED_SAFE_AUTO"])
    rows.append(
        row(
            "queue_safe_execution_never_crosses_high_risk_gate",
            "PASS" if int(high_risk_auto.sum()) == 0 else "FAIL",
            str(execution_path),
            f"high_risk_auto={int(high_risk_auto.sum())}",
        )
    )
    failures = int(execution["execution_status"].astype(str).eq("FAIL").sum())
    rows.append(row("queue_execution_has_no_failures", "PASS" if failures == 0 else "FAIL", str(execution_path), f"failures={failures}"))
    if not history.empty and not missing_history:
        current_keys = execution[["queue_id", "scenario_id", "started_at", "execution_status", "command"]].astype(str)
        history_keys = history[["queue_id", "scenario_id", "started_at", "execution_status", "command"]].astype(str)
        merged = current_keys.merge(history_keys.drop_duplicates(), how="left", indicator=True)
        missing_current = int((merged["_merge"] != "both").sum())
        rows.append(
            row(
                "queue_history_records_current_execution",
                "PASS" if missing_current == 0 else "FAIL",
                str(history_path),
                f"missing_current_rows={missing_current}",
            )
        )
        history_high_risk_auto = history["command"].fillna("").astype(str).str.contains(
            "--confirm-expensive|--agent-runbook-execute-expensive-phase|--agent-runbook-execute-model",
            regex=True,
            na=False,
        ) & history["execution_status"].astype(str).isin(["PASS", "FAIL", "PLANNED_SAFE_AUTO"])
        rows.append(
            row(
                "queue_history_never_crosses_high_risk_gate",
                "PASS" if int(history_high_risk_auto.sum()) == 0 else "FAIL",
                str(history_path),
                f"high_risk_history_rows={int(history_high_risk_auto.sum())}",
            )
        )
        q003_success = history[
            history["queue_id"].astype(str).eq("Q003")
            & history["scenario_id"].astype(str).eq("agent_control_focus")
            & history["execution_status"].astype(str).eq("PASS")
        ]
        current_q003_success = execution[
            execution["queue_id"].astype(str).eq("Q003")
            & execution["scenario_id"].astype(str).eq("agent_control_focus")
            & execution["execution_status"].astype(str).eq("PASS")
        ]
        rows.append(
            row(
                "queue_history_preserves_agent_control_success",
                "PASS" if current_q003_success.empty or not q003_success.empty else "FAIL",
                str(history_path),
                f"current_q003_success={len(current_q003_success)}; q003_success_rows={len(q003_success)}",
            )
        )
    else:
        rows.append(row("queue_history_records_current_execution", "FAIL", str(history_path), "history missing or schema incomplete"))
        rows.append(row("queue_history_never_crosses_high_risk_gate", "FAIL", str(history_path), "history missing or schema incomplete"))
        rows.append(row("queue_history_preserves_agent_control_success", "FAIL", str(history_path), "history missing or schema incomplete"))
    rows.append(
        row(
            "queue_execution_report_boundary",
            "PASS" if "no medical QA" in report and "treatment recommendation" in report else "FAIL",
            str(report_path),
            "expects non-clinical boundary wording",
        )
    )
    run_study = (project_root / "src" / "chrono_ehr" / "run_study.py").read_text(encoding="utf-8")
    runner = (project_root / "src" / "chrono_ehr" / "agent_task_queue_runner.py").read_text(encoding="utf-8")
    rows.append(
        row(
            "queue_runner_filter_flags_registered",
            "PASS"
            if "--agent-task-queue-id" in run_study
            and "--agent-task-queue-scenario" in run_study
            and "--queue-id" in runner
            and "--scenario-id" in runner
            and "agent_task_queue_execution_history.csv" in runner
            else "FAIL",
            "src/chrono_ehr/run_study.py; src/chrono_ehr/agent_task_queue_runner.py",
            "expects queue_id/scenario_id filters and append-only history",
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
    table_path = args.project_root / "outputs" / "tables" / "agent_task_queue_execution_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_task_queue_execution_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Task Queue Execution Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local queue execution safety only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent task queue execution checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
