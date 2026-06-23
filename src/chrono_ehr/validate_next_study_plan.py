#!/usr/bin/env python3
"""Validate the next-study action plan against current readiness state."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

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
    plan_path = project_root / "outputs" / "tables" / "next_study_action_plan.csv"
    report_path = project_root / "outputs" / "reports" / "next_study_action_plan.md"
    external_path = project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv"
    plan = read_csv(plan_path)
    external = read_csv(external_path)
    report = read_text(report_path)
    rows = [
        row("next_study_plan_exists", "PASS" if not plan.empty else "FAIL", str(plan_path), f"rows={len(plan)}"),
        row("next_study_report_exists", "PASS" if report else "FAIL", str(report_path), f"chars={len(report)}"),
    ]
    if plan.empty:
        return pd.DataFrame(rows)

    required = {"dataset", "status", "priority", "recommended_action", "command"}
    missing = sorted(required - set(plan.columns))
    rows.append(row("required_columns", "PASS" if not missing else "FAIL", str(plan_path), "missing=" + ",".join(missing)))

    if not external.empty and {"dataset", "local_status"}.issubset(external.columns):
        status_by_dataset = dict(zip(external["dataset"].astype(str), external["local_status"].astype(str)))
        plan_by_dataset = {str(item["dataset"]): item for item in plan.to_dict(orient="records")}
        for dataset in ["CDSL", "eICU", "CHARLS"]:
            rows.append(
                row(
                    f"{dataset.lower()}_row_present",
                    "PASS" if dataset in plan_by_dataset else "FAIL",
                    str(plan_path),
                    f"external_status={status_by_dataset.get(dataset, '')}",
                )
            )
        cdsl = plan_by_dataset.get("CDSL", {})
        rows.append(
            row(
                "cdsl_ready_uses_external_summary_validation",
                "PASS"
                if status_by_dataset.get("CDSL") != "READY"
                or "--validate-external-benchmark-summary" in str(cdsl.get("command", ""))
                else "FAIL",
                str(plan_path),
                f"status={status_by_dataset.get('CDSL', '')}; command={cdsl.get('command', '')}",
            )
        )
        eicu = plan_by_dataset.get("eICU", {})
        rows.append(
            row(
                "eicu_baseline_ready_uses_external_summary_validation",
                "PASS"
                if status_by_dataset.get("eICU") != "BASELINE_READY"
                or "--validate-external-benchmark-summary" in str(eicu.get("command", ""))
                else "FAIL",
                str(plan_path),
                f"status={status_by_dataset.get('eICU', '')}; command={eicu.get('command', '')}",
            )
        )
        charls = plan_by_dataset.get("CHARLS", {})
        rows.append(
            row(
                "charls_data_pending_uses_readiness",
                "PASS"
                if status_by_dataset.get("CHARLS") != "DATA_PENDING"
                or "--charls-readiness" in str(charls.get("command", ""))
                else "FAIL",
                str(plan_path),
                f"status={status_by_dataset.get('CHARLS', '')}; command={charls.get('command', '')}",
            )
        )

    stale_text = "eICU 和 CHARLS 已经有 protocol/config/checklist/readiness 脚本；真正写 cohort/model 代码前" in report
    rows.append(row("report_does_not_group_eicu_as_data_pending", "PASS" if not stale_text else "FAIL", str(report_path), f"stale_text={stale_text}"))
    rows.append(
        row(
            "report_declares_eicu_boundary",
            "PASS" if "eICU 已经推进到 baseline-ready ICU mortality benchmark" in report else "FAIL",
            str(report_path),
            "expects eICU baseline-ready boundary text",
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
    table_path = args.project_root / "outputs" / "tables" / "next_study_action_plan_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "next_study_action_plan_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Next Study Action Plan Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local research-planning recommendations only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Next-study plan checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
