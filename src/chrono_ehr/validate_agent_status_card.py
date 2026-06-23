#!/usr/bin/env python3
"""Validate the short ChronoEHR-Agent status card output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_ITEMS = {
    "agent_progress_score",
    "mainline_mvp",
    "agent_self_check",
    "agent_doctor",
    "delivery_readiness",
    "artifact_freshness",
    "command_lint",
    "boundary_audit",
    "dependency_audit",
    "doc_command_audit",
    "handoff_checklist",
    "task_execution_validation",
    "task_scenario_validation",
    "task_queue_validation",
    "task_queue_execution_validation",
    "cooldown_fingerprint_validation",
    "entrypoints_validation",
    "next_study_validation",
    "runbook_retry_validation",
    "external_cdsl",
    "external_eicu",
    "external_charls",
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "agent_status_card.csv"
    report_path = project_root / "outputs" / "reports" / "agent_status_card.md"
    table = read_csv(table_path)
    report = read_text(report_path)
    rows = [
        row("status_card_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("status_card_report_exists", "PASS" if bool(report) else "FAIL", str(report_path), f"chars={len(report)}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)

    required_columns = {"item", "status", "evidence", "detail"}
    missing_columns = sorted(required_columns - set(table.columns))
    rows.append(row("required_columns", "PASS" if not missing_columns else "FAIL", str(table_path), "missing=" + ",".join(missing_columns)))

    items = set(table["item"].astype(str)) if "item" in table else set()
    missing_items = sorted(REQUIRED_ITEMS - items)
    rows.append(row("required_items_present", "PASS" if not missing_items else "FAIL", str(table_path), "missing=" + ",".join(missing_items)))

    failures = int((table["status"].astype(str) != "PASS").sum()) if "status" in table else len(table)
    rows.append(row("all_status_rows_pass", "PASS" if failures == 0 else "FAIL", str(table_path), f"non_pass={failures}"))

    detail_by_item = dict(zip(table["item"].astype(str), table["detail"].astype(str))) if {"item", "detail"}.issubset(table.columns) else {}
    rows.append(row("external_eicu_boundary_current", "PASS" if detail_by_item.get("external_eicu") == "BASELINE_READY" else "FAIL", str(table_path), f"eICU={detail_by_item.get('external_eicu', '')}"))
    allowed_charls_statuses = {"DATA_PENDING", "READY_FOR_PROTOCOL_CODE", "READY_FOR_PROTOCOL_DRAFT", "READY"}
    rows.append(
        row(
            "external_charls_status_allowed",
            "PASS" if detail_by_item.get("external_charls") in allowed_charls_statuses else "FAIL",
            str(table_path),
            f"CHARLS={detail_by_item.get('external_charls', '')}",
        )
    )

    rows.append(row("report_overall_pass", "PASS" if "Overall status: `PASS`" in report else "FAIL", str(report_path), "expects Overall status PASS"))
    rows.append(row("report_boundary_declared", "PASS" if "not a medical QA system" in report and "not a treatment recommendation system" in report else "FAIL", str(report_path), "expects non-clinical boundary"))
    rows.append(row("report_last_task_present", "PASS" if "## Last Task" in report and "refresh_failures" in report else "FAIL", str(report_path), "expects Last Task section with refresh summary"))
    rows.append(row("report_last_task_scenario_present", "PASS" if "## Last Task" in report and "scenario" in report else "FAIL", str(report_path), "expects Last Task scenario column"))
    rows.append(row("report_active_focus_present", "PASS" if "## Active Focus" in report and "focus_id" in report else "FAIL", str(report_path), "expects Active Focus section"))
    rows.append(row("report_queue_history_present", "PASS" if "## Queue Execution History" in report and "q003_success_rows" in report else "FAIL", str(report_path), "expects queue execution history section"))
    rows.append(
        row(
            "report_cooldown_policy_present",
            "PASS"
            if "## Cooldown Policy" in report
            and "config_version" in report
            and "q003_cooldown_status" in report
            and "volatile_outputs_excluded" in report
            else "FAIL",
            str(report_path),
            "expects cooldown policy section with config and Q003 cooldown summary",
        )
    )
    rows.append(
        row(
            "report_current_queue_present",
            "PASS"
            if "## Current Queue" in report
            and "cooldown_fingerprint_status" in report
            and "cooldown_policy_summary" in report
            else "FAIL",
            str(report_path),
            "expects current queue section with cooldown fingerprint status and policy summary",
        )
    )
    rows.append(
        row(
            "report_next_tasks_policy_summary_present",
            "PASS" if "## Next Tasks" in report and "cooldown_policy_summary" in report else "FAIL",
            str(report_path),
            "expects next tasks section with cooldown policy summary",
        )
    )
    rows.append(row("report_best_commands_present", "PASS" if "--agent-doctor" in report and "--agent-next-tasks" in report else "FAIL", str(report_path), "expects best next commands"))
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
    table_path = args.project_root / "outputs" / "tables" / "agent_status_card_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_status_card_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Status Card Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local Agent status reporting only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent status-card checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
