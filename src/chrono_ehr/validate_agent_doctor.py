#!/usr/bin/env python3
"""Validate the ChronoEHR-Agent doctor health-check bundle."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agent_doctor import DOCTOR_STEPS
from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_STEPS = [
    ("agent_self_check", "python3 src/chrono_ehr/run_study.py --agent-self-check"),
    ("delivery_readiness", "python3 src/chrono_ehr/run_study.py --delivery-readiness"),
    ("agent_artifact_freshness", "python3 src/chrono_ehr/run_study.py --agent-artifact-freshness"),
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


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def command_is_simple_run_study(command: str) -> bool:
    tokens = shlex.split(command)
    return tokens[:2] == ["python3", "src/chrono_ehr/run_study.py"] and len([token for token in tokens if token.startswith("--")]) == 1


def audit(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    configured = [(str(step.get("id", "")), str(step.get("command", ""))) for step in DOCTOR_STEPS]
    rows.append(
        row(
            "configured_steps_match_expected",
            "PASS" if configured == EXPECTED_STEPS else "FAIL",
            "src/chrono_ehr/agent_doctor.py",
            f"configured={configured}",
        )
    )
    rows.append(
        row(
            "configured_commands_are_simple",
            "PASS" if all(command_is_simple_run_study(command) for _, command in configured) else "FAIL",
            "src/chrono_ehr/agent_doctor.py",
            "expects one run_study flag per doctor step",
        )
    )

    doctor_path = project_root / "outputs" / "tables" / "agent_doctor.csv"
    doctor = read_csv(doctor_path)
    rows.append(row("doctor_output_exists", "PASS" if not doctor.empty else "FAIL", str(doctor_path), f"rows={len(doctor)}"))
    if doctor.empty:
        return pd.DataFrame(rows)

    output_steps = list(zip(doctor.get("id", pd.Series(dtype=str)).astype(str), doctor.get("command", pd.Series(dtype=str)).astype(str)))
    rows.append(
        row(
            "doctor_output_steps_match_expected",
            "PASS" if output_steps == EXPECTED_STEPS else "FAIL",
            str(doctor_path),
            f"output_steps={output_steps}",
        )
    )
    failures = int(doctor["status"].ne("PASS").sum()) if "status" in doctor else -1
    rows.append(row("doctor_last_run_passed", "PASS" if failures == 0 else "FAIL", str(doctor_path), f"failures={failures}"))
    elapsed_ok = "elapsed_seconds" in doctor and pd.to_numeric(doctor["elapsed_seconds"], errors="coerce").notna().all()
    rows.append(row("doctor_records_elapsed_seconds", "PASS" if elapsed_ok else "FAIL", str(doctor_path), f"elapsed_ok={elapsed_ok}"))
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
    table_path = args.project_root / "outputs" / "tables" / "agent_doctor_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_doctor_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Doctor Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local Agent health-check orchestration only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent doctor validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
