#!/usr/bin/env python3
"""Validate the ChronoEHR-Agent v0.1 mainline MVP gate."""

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


def exists(project_root: Path, relative: str) -> bool:
    path = project_root / relative
    return path.exists() and path.stat().st_size > 0


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit_docs(project_root: Path) -> list[dict[str, str]]:
    docs = [
        "docs/mainline_mvp_definition.md",
        "docs/quickstart_usage.md",
        "docs/agent_design.md",
        "docs/study_protocol.md",
    ]
    return [
        row(f"doc:{Path(doc).name}", "PASS" if exists(project_root, doc) else "FAIL", doc, "required MVP documentation")
        for doc in docs
    ]


def audit_demo_workflow(project_root: Path) -> list[dict[str, str]]:
    workflow = read_csv(project_root / "outputs" / "tables" / "agent_demo_workflow_diabetes.csv")
    validation = read_csv(project_root / "outputs" / "tables" / "agent_demo_workflow_validation.csv")
    rows = [
        row("diabetes_demo_workflow_exists", "PASS" if not workflow.empty else "FAIL", "outputs/tables/agent_demo_workflow_diabetes.csv", f"rows={len(workflow)}"),
    ]
    if not workflow.empty:
        failures = int((workflow["status"] == "FAIL").sum()) if "status" in workflow else -1
        safe_by_default = (
            "command" in workflow
            and not workflow["command"].astype(str).str.contains("--confirm-expensive|--random-forest-baseline|--gradient-boosting", regex=True, na=False).any()
        )
        rows.append(row("diabetes_demo_workflow_passed", "PASS" if failures == 0 else "FAIL", "outputs/tables/agent_demo_workflow_diabetes.csv", f"failures={failures}"))
        rows.append(row("diabetes_demo_safe_by_default", "PASS" if safe_by_default else "FAIL", "outputs/tables/agent_demo_workflow_diabetes.csv", f"safe={safe_by_default}"))
    rows.append(
        row(
            "diabetes_demo_workflow_validation",
            "PASS" if not validation.empty and (validation["status"] == "PASS").all() else "FAIL",
            "outputs/tables/agent_demo_workflow_validation.csv",
            f"rows={len(validation)}",
        )
    )
    return rows


def audit_studies(project_root: Path) -> list[dict[str, str]]:
    summary = read_csv(project_root / "outputs" / "tables" / "study_capability_summary.csv")
    rows = [row("study_capability_summary_exists", "PASS" if not summary.empty else "FAIL", "outputs/tables/study_capability_summary.csv", f"rows={len(summary)}")]
    expected_complete = {
        "mimic_iv_3_1_diabetes_readmission",
        "mimic_iv_ckd_readmission",
        "mimic_iv_heart_failure_readmission",
        "mimic_iv_hypertension_readmission",
        "cross_cohort",
    }
    if not summary.empty:
        complete = set(summary[summary["overall_status"].eq("COMPLETE")]["study_id"].astype(str))
        missing = sorted(expected_complete - complete)
        rows.append(row("mimic_chronic_studies_complete", "PASS" if not missing else "FAIL", "outputs/tables/study_capability_summary.csv", "missing=" + ",".join(missing)))
    return rows


def audit_agent(project_root: Path) -> list[dict[str, str]]:
    self_check = read_csv(project_root / "outputs" / "tables" / "agent_self_check.csv")
    doctor = read_csv(project_root / "outputs" / "tables" / "agent_doctor.csv")
    doctor_validation = read_csv(project_root / "outputs" / "tables" / "agent_doctor_validation.csv")
    control_consistency = read_csv(project_root / "outputs" / "tables" / "agent_control_consistency_audit.csv")
    readiness = read_csv(project_root / "outputs" / "tables" / "delivery_readiness_audit.csv")
    runbook_state = read_csv(project_root / "outputs" / "tables" / "agent_runbook_state_machine.csv")
    next_tasks = read_csv(project_root / "outputs" / "tables" / "agent_next_tasks.csv")
    self_check_core = self_check
    if not self_check.empty and "id" in self_check:
        # Keep this tolerant of older self-check outputs that included the MVP
        # gate before the self-check/MVP circular dependency was removed.
        self_check_core = self_check[self_check["id"].astype(str).ne("mainline_mvp_validation")]
    rows = [
        row("agent_self_check_core_passed", "PASS" if not self_check_core.empty and (self_check_core["status"] == "PASS").all() else "FAIL", "outputs/tables/agent_self_check.csv", f"rows={len(self_check)}, core_rows={len(self_check_core)}"),
        row("agent_doctor_passed", "PASS" if not doctor.empty and (doctor["status"] == "PASS").all() else "FAIL", "outputs/tables/agent_doctor.csv", f"rows={len(doctor)}"),
        row("agent_doctor_validation_passed", "PASS" if not doctor_validation.empty and (doctor_validation["status"] == "PASS").all() else "FAIL", "outputs/tables/agent_doctor_validation.csv", f"rows={len(doctor_validation)}"),
        row("agent_control_consistency_passed", "PASS" if not control_consistency.empty and (control_consistency["status"] == "PASS").all() else "FAIL", "outputs/tables/agent_control_consistency_audit.csv", f"rows={len(control_consistency)}"),
        row("delivery_readiness_passed", "PASS" if not readiness.empty and (readiness["status"] == "PASS").all() else "FAIL", "outputs/tables/delivery_readiness_audit.csv", f"rows={len(readiness)}"),
        row("runbook_state_machine_exists", "PASS" if not runbook_state.empty else "FAIL", "outputs/tables/agent_runbook_state_machine.csv", f"rows={len(runbook_state)}"),
        row("next_tasks_exist", "PASS" if not next_tasks.empty else "FAIL", "outputs/tables/agent_next_tasks.csv", f"rows={len(next_tasks)}"),
    ]
    if not runbook_state.empty:
        bad_model = runbook_state[
            runbook_state["risk_level"].astype(str).isin(["model", "report"])
            & runbook_state["can_execute_now"].astype(str).isin(["YES", "YES_WITH_CONFIRMATION"])
        ]
        rows.append(row("model_report_not_auto_open", "PASS" if bad_model.empty else "FAIL", "outputs/tables/agent_runbook_state_machine.csv", f"bad_rows={len(bad_model)}"))
    return rows


def audit_external(project_root: Path) -> list[dict[str, str]]:
    external = read_csv(project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv")
    rows = [row("external_readiness_summary_exists", "PASS" if not external.empty else "FAIL", "outputs/tables/external_benchmark_readiness_summary.csv", f"rows={len(external)}")]
    if not external.empty:
        statuses = dict(zip(external["dataset"].astype(str), external["local_status"].astype(str)))
        rows.append(row("cdsl_ready", "PASS" if statuses.get("CDSL") == "READY" else "FAIL", "outputs/tables/external_benchmark_readiness_summary.csv", f"CDSL={statuses.get('CDSL', '')}"))
        eicu_ok = statuses.get("eICU") in {"READY", "READY_FOR_COHORT_CODE", "COHORT_READY", "FEATURE_READY", "BASELINE_READY", "DATA_PENDING"}
        charls_ok = statuses.get("CHARLS") in {"READY", "READY_FOR_PROTOCOL_CODE", "DATA_PENDING"}
        rows.append(row("eicu_status_allowed", "PASS" if eicu_ok else "FAIL", "outputs/tables/external_benchmark_readiness_summary.csv", f"eICU={statuses.get('eICU', '')}"))
        rows.append(row("charls_status_allowed", "PASS" if charls_ok else "FAIL", "outputs/tables/external_benchmark_readiness_summary.csv", f"CHARLS={statuses.get('CHARLS', '')}"))
    return rows


def audit(project_root: Path) -> pd.DataFrame:
    rows = []
    rows.extend(audit_docs(project_root))
    rows.extend(audit_demo_workflow(project_root))
    rows.extend(audit_studies(project_root))
    rows.extend(audit_agent(project_root))
    rows.extend(audit_external(project_root))
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
    table_path = args.project_root / "outputs" / "tables" / "mainline_mvp_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "mainline_mvp_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Mainline MVP Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local research-tool readiness only; no medical QA, diagnosis, or treatment recommendation.

## Interpretation

`PASS` means ChronoEHR-Agent v0.1 has a working local diabetes demo, time-aware/leakage checks, Agent control layer, recovery/state artifacts, and documented usage. eICU and CHARLS may remain `DATA_PENDING` at this stage, or move into a planned-code-ready status when local files/templates are available.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Mainline MVP checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
