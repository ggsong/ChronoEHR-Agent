#!/usr/bin/env python3
"""Generate a capability card for the public ChronoEHR-Agent demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def status_row(capability: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"capability": capability, "status": status, "evidence": evidence, "detail": detail}


def build_rows(project_root: Path) -> list[dict[str, str]]:
    demo_dir = project_root / "outputs" / "demo"
    contract_path = demo_dir / "synthetic_ehr_contract.csv"
    cohort_path = demo_dir / "synthetic_cohort.csv"
    metrics_path = demo_dir / "synthetic_demo_metrics.csv"
    trace_path = project_root / "outputs" / "tables" / "agent_demo_trace.json"
    catalog = read_json(project_root / "configs" / "agent_action_catalog.json")
    entrypoints = read_json(project_root / "configs" / "agent_entrypoints.json")

    rows = []
    rows.append(
        status_row(
            "unified_agent_entrypoint",
            "PASS" if (project_root / "src" / "chrono_ehr" / "run_study.py").exists() else "FAIL",
            "src/chrono_ehr/run_study.py",
            "Single command-line dispatcher for study, audit, demo, and agent-control actions.",
        )
    )
    action_count = len(catalog.get("actions", []))
    rows.append(
        status_row(
            "task_action_catalog",
            "PASS" if action_count > 0 else "FAIL",
            "configs/agent_action_catalog.json",
            f"{action_count} cataloged actions with risk levels and commands.",
        )
    )
    entrypoint_groups = entrypoints.get("groups", entrypoints.get("sections", []))
    entrypoint_count = sum(len(section.get("commands", [])) for section in entrypoint_groups)
    rows.append(
        status_row(
            "stable_entrypoints",
            "PASS" if entrypoint_count > 0 else "FAIL",
            "configs/agent_entrypoints.json",
            f"{entrypoint_count} documented entrypoint commands.",
        )
    )
    raw_tables = sorted(path.name for path in (demo_dir / "raw").glob("*.csv")) if (demo_dir / "raw").exists() else []
    rows.append(
        status_row(
            "synthetic_ehr_raw_tables",
            "PASS" if len(raw_tables) >= 5 else "FAIL",
            "outputs/demo/raw/*.csv",
            ", ".join(raw_tables) if raw_tables else "raw tables missing; run --synthetic-demo",
        )
    )
    if contract_path.exists():
        contract = pd.read_csv(contract_path)
        failures = int((contract["status"] != "PASS").sum()) if "status" in contract.columns else len(contract)
        rows.append(
            status_row(
                "data_contract_audit",
                "PASS" if failures == 0 and len(contract) > 0 else "FAIL",
                "outputs/demo/synthetic_ehr_contract.csv",
                f"{len(contract)} checks, {failures} failures.",
            )
        )
    else:
        rows.append(status_row("data_contract_audit", "FAIL", "outputs/demo/synthetic_ehr_contract.csv", "missing; run --synthetic-demo"))
    cohort_rows = len(pd.read_csv(cohort_path)) if cohort_path.exists() else 0
    metric_rows = len(pd.read_csv(metrics_path)) if metrics_path.exists() else 0
    rows.append(
        status_row(
            "cohort_feature_metric_pipeline",
            "PASS" if cohort_rows > 0 and metric_rows > 0 else "FAIL",
            "outputs/demo/synthetic_cohort.csv; outputs/demo/synthetic_demo_metrics.csv",
            f"{cohort_rows} cohort rows, {metric_rows} metric rows.",
        )
    )
    trace = read_json(trace_path)
    rows.append(
        status_row(
            "agent_behavior_trace",
            "PASS" if trace.get("overall_status") == "PASS" else "FAIL",
            "outputs/tables/agent_demo_trace.json",
            f"{len(trace.get('steps', []))} trace steps; status={trace.get('overall_status', 'missing')}.",
        )
    )
    rows.append(
        status_row(
            "release_safety_guard",
            "PASS" if (project_root / "scripts" / "release_audit.py").exists() and (project_root / ".gitignore").exists() else "FAIL",
            "scripts/release_audit.py; .gitignore",
            "Public release audit and controlled-data ignore rules are present.",
        )
    )
    rows.append(
        status_row(
            "medical_agent_positioning",
            "PASS" if (project_root / "docs" / "RELATED_WORK.md").exists() else "FAIL",
            "docs/RELATED_WORK.md",
            "Explicitly positions the project relative to medical-agent evaluation and audit work.",
        )
    )
    rows.append(
        status_row(
            "ci_smoke_test",
            "PASS" if (project_root / ".github" / "workflows" / "ci.yml").exists() else "FAIL",
            ".github/workflows/ci.yml",
            "GitHub Actions smoke test is configured for no-data validation.",
        )
    )
    return rows


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    overall = "PASS" if df["status"].eq("PASS").all() else "FAIL"
    lines = [
        "# ChronoEHR-Agent Capability Card",
        "",
        f"- Overall status: `{overall}`",
        f"- Capabilities checked: {len(df)}",
        "",
        "| capability | status | evidence | detail |",
        "|---|---|---|---|",
    ]
    for row in df.to_dict(orient="records"):
        lines.append(f"| {row['capability']} | {row['status']} | `{row['evidence']}` | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This card demonstrates agent engineering capability for local EHR research workflows. It does not claim clinical decision support or diagnostic performance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    table_dir = args.project_root / "outputs" / "tables"
    report_dir = args.project_root / "outputs" / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_rows(args.project_root))
    df.to_csv(table_dir / "agent_capability_card.csv", index=False)
    write_markdown(df, report_dir / "agent_capability_card.md")
    overall = "PASS" if df["status"].eq("PASS").all() else "FAIL"
    print(f"Agent capability card {overall}: {len(df)} capabilities.")
    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
