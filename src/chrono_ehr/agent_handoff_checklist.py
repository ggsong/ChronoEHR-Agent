#!/usr/bin/env python3
"""Build a lightweight handoff checklist for resuming ChronoEHR-Agent work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_FILES = [
    "README.md",
    "docs/quickstart_usage.md",
    "docs/resume_state.md",
    "docs/mainline_mvp_definition.md",
    "outputs/reports/agent_progress_score.md",
    "outputs/tables/agent_progress_score.csv",
    "outputs/reports/agent_doctor.md",
    "outputs/tables/agent_doctor.csv",
    "outputs/reports/agent_next_tasks.md",
    "outputs/tables/agent_next_tasks.csv",
    "outputs/state/agent_state.md",
    "outputs/state/agent_state.json",
]

REQUIRED_COMMANDS = [
    "--agent-doctor",
    "--agent-status-card",
    "--agent-progress-score",
    "--validate-agent-progress-score",
    "--agent-next-tasks",
    "--diabetes-agent-demo",
    "--external-readiness-summary",
]

BOUNDARY_TERMS = [
    "不是医学问答",
    "不是临床诊疗建议",
    "not a medical QA",
    "no medical QA",
    "no diagnosis",
    "no treatment recommendation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def status_row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit_required_files(project_root: Path) -> list[dict[str, str]]:
    rows = []
    for relative in REQUIRED_FILES:
        path = project_root / relative
        status = "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL"
        detail = f"size={path.stat().st_size}" if path.exists() else "missing"
        rows.append(status_row(f"file_exists:{relative}", status, relative, detail))
    return rows


def audit_resume_docs(project_root: Path) -> list[dict[str, str]]:
    rows = []
    quickstart = read_text(project_root / "docs" / "quickstart_usage.md")
    readme = read_text(project_root / "README.md")
    resume = read_text(project_root / "docs" / "resume_state.md")
    combined = "\n".join([quickstart, readme, resume])

    for command in REQUIRED_COMMANDS:
        rows.append(
            status_row(
                f"handoff_command_documented:{command}",
                "PASS" if command in combined else "FAIL",
                "README.md; docs/quickstart_usage.md; docs/resume_state.md",
                f"requires {command}",
            )
        )

    boundary_hits = [term for term in BOUNDARY_TERMS if term in combined]
    rows.append(
        status_row(
            "handoff_boundary_statement_present",
            "PASS" if len(boundary_hits) >= 2 else "FAIL",
            "README.md; docs/quickstart_usage.md; docs/resume_state.md",
            "matched=" + ",".join(boundary_hits),
        )
    )
    rows.append(
        status_row(
            "resume_state_mentions_current_progress_score",
            "PASS" if "98.7/100" in resume and "agent-progress-score" in resume else "FAIL",
            "docs/resume_state.md",
            "expects current score and command",
        )
    )
    return rows


def audit_status_outputs(project_root: Path) -> list[dict[str, str]]:
    rows = []
    progress = pd.read_csv(project_root / "outputs" / "tables" / "agent_progress_score.csv")
    next_tasks = pd.read_csv(project_root / "outputs" / "tables" / "agent_next_tasks.csv")
    state = read_json(project_root / "outputs" / "state" / "agent_state.json")

    total = progress[progress["component"].astype(str).eq("TOTAL")]
    score = float(total.iloc[0]["weighted_points"]) if not total.empty else 0.0
    rows.append(
        status_row(
            "progress_score_resumable_mvp",
            "PASS" if score >= 95.0 else "FAIL",
            "outputs/tables/agent_progress_score.csv",
            f"score={score:.1f}",
        )
    )
    commands = next_tasks["command"].astype(str).tolist() if "command" in next_tasks else []
    resume_command_present = any(
        token in item
        for item in commands
        for token in ["--agent-self-check", "--agent-doctor", "--agent-next-tasks", "--agent-runbook", "--external-readiness-summary"]
    )
    rows.append(
        status_row(
            "next_tasks_have_resume_commands",
            "PASS" if commands and resume_command_present else "FAIL",
            "outputs/tables/agent_next_tasks.csv",
            f"commands={len(commands)}",
        )
    )
    recommended = state.get("recommended_next_commands", [])
    rows.append(
        status_row(
            "agent_state_has_recommended_commands",
            "PASS" if recommended and any("--agent-doctor" in str(item) for item in recommended) else "FAIL",
            "outputs/state/agent_state.json",
            f"recommended={len(recommended)}",
        )
    )
    rows.append(
        status_row(
            "agent_state_boundary_is_research_tool",
            "PASS" if "medical" in str(state.get("boundary", "")).lower() and "treatment" in str(state.get("boundary", "")).lower() else "FAIL",
            "outputs/state/agent_state.json",
            str(state.get("boundary", "")),
        )
    )
    last_task = state.get("last_task_execution", {})
    rows.append(
        status_row(
            "last_task_execution_recorded",
            "PASS" if last_task.get("available") is True else "FAIL",
            "outputs/state/agent_state.json",
            str(last_task),
        )
    )
    rows.append(
        status_row(
            "last_task_execution_clean",
            "PASS"
            if last_task.get("available") is True
            and int(last_task.get("execution_failures", 1) or 0) == 0
            and int(last_task.get("post_refresh_failures", 1) or 0) == 0
            and int(last_task.get("validation_failures", 1) or 0) == 0
            else "FAIL",
            "outputs/state/agent_state.json",
            str(last_task),
        )
    )
    rows.append(
        status_row(
            "last_task_execution_scenario_recorded",
            "PASS" if last_task.get("available") is True and bool(last_task.get("scenario_id")) else "FAIL",
            "outputs/state/agent_state.json",
            str(last_task),
        )
    )
    return rows


def audit(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    rows.extend(audit_required_files(project_root))
    rows.extend(audit_resume_docs(project_root))
    rows.extend(audit_status_outputs(project_root))
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
    table_path = args.project_root / "outputs" / "tables" / "agent_handoff_checklist.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_handoff_checklist.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Handoff Checklist

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: local work-resume checklist only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent handoff checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
