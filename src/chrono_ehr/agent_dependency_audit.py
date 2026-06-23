#!/usr/bin/env python3
"""Audit control-layer dependency boundaries to prevent circular gates."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from agent_artifact_freshness import FRESHNESS_RULES
from agent_doctor import DOCTOR_STEPS
from agent_self_check import CHECKS
from mimic_diabetes_baseline import DEFAULT_PROJECT


FORBIDDEN_SELF_CHECK_FLAGS = {
    "--agent-doctor",
    "--agent-self-check",
    "--agent-status-card",
    "--delivery-readiness",
    "--validate-agent-doctor",
    "--validate-mainline-mvp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def run_study_flags(project_root: Path) -> set[str]:
    source = read_text(project_root / "src" / "chrono_ehr" / "run_study.py")
    return set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', source))


def audit(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    self_commands = [str(item.get("command", "")) for item in CHECKS]
    self_text = "\n".join(self_commands)
    forbidden = sorted(flag for flag in FORBIDDEN_SELF_CHECK_FLAGS if flag in self_text)
    rows.append(row("self_check_commands_present", "PASS" if self_commands else "FAIL", "src/chrono_ehr/agent_self_check.py", f"commands={len(self_commands)}"))
    rows.append(
        row(
            "self_check_has_no_terminal_gate_cycle",
            "PASS" if not forbidden else "FAIL",
            "src/chrono_ehr/agent_self_check.py",
            "forbidden=" + ",".join(forbidden),
        )
    )

    self_ids = {str(item.get("id", "")) for item in CHECKS}
    rows.append(row("mvp_not_inside_self_check", "PASS" if "mainline_mvp_validation" not in self_ids else "FAIL", "src/chrono_ehr/agent_self_check.py", f"ids={len(self_ids)}"))
    rows.append(row("delivery_not_inside_self_check", "PASS" if "delivery_readiness" not in self_ids else "FAIL", "src/chrono_ehr/agent_self_check.py", f"ids={len(self_ids)}"))
    rows.append(row("status_card_not_inside_self_check", "PASS" if "agent_status_card" not in self_ids else "FAIL", "src/chrono_ehr/agent_self_check.py", f"ids={len(self_ids)}"))
    self_order = [str(item.get("id", "")) for item in CHECKS]
    try:
        state_index = self_order.index("agent_state")
        handoff_index = self_order.index("agent_handoff_checklist")
        freshness_index = self_order.index("agent_artifact_freshness")
        handoff_order_ok = state_index < handoff_index < freshness_index
    except ValueError:
        handoff_order_ok = False
    rows.append(
        row(
            "handoff_checklist_order_avoids_stale_state",
            "PASS" if handoff_order_ok else "FAIL",
            "src/chrono_ehr/agent_self_check.py",
            f"order={self_order}",
        )
    )

    direct_status_card_rules = [
        rule
        for rule in FRESHNESS_RULES
        if str(rule.get("artifact_id", "")) == "agent_status_card"
        or "outputs/reports/agent_status_card.md" in rule.get("outputs", [])
        or "outputs/tables/agent_status_card.csv" in rule.get("outputs", [])
    ]
    rows.append(
        row(
            "status_card_not_in_freshness_cycle",
            "PASS" if not direct_status_card_rules else "FAIL",
            "src/chrono_ehr/agent_artifact_freshness.py",
            f"direct_status_card_rules={len(direct_status_card_rules)}",
        )
    )
    status_card_validation_rules = [
        rule
        for rule in FRESHNESS_RULES
        if str(rule.get("artifact_id", "")) == "agent_status_card_validation"
    ]
    rows.append(
        row(
            "status_card_validation_not_in_freshness_cycle",
            "PASS" if not status_card_validation_rules else "FAIL",
            "src/chrono_ehr/agent_artifact_freshness.py",
            f"status_card_validation_rules={len(status_card_validation_rules)}",
        )
    )

    doctor_ids = [str(item.get("id", "")) for item in DOCTOR_STEPS]
    expected_doctor = ["agent_self_check", "delivery_readiness", "agent_artifact_freshness"]
    rows.append(
        row(
            "doctor_layer_order",
            "PASS" if doctor_ids == expected_doctor else "FAIL",
            "src/chrono_ehr/agent_doctor.py",
            f"doctor_ids={doctor_ids}",
        )
    )

    flags = run_study_flags(project_root)
    rows.append(
        row(
            "dependency_audit_flag_registered",
            "PASS" if "--agent-dependency-audit" in flags else "FAIL",
            "src/chrono_ehr/run_study.py",
            "--agent-dependency-audit",
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
    table_path = args.project_root / "outputs" / "tables" / "agent_dependency_audit.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_dependency_audit.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Dependency Audit

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: audits local Agent control dependencies only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent dependency audit checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
