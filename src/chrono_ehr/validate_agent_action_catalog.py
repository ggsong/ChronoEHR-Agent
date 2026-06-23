#!/usr/bin/env python3
"""Validate the ChronoEHR-Agent action catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = DEFAULT_PROJECT / "configs" / "agent_action_catalog.json"
VALID_RISKS = {"safe", "expensive", "model", "report"}
VALID_GOALS = {"status", "leakage", "external", "demo", "model", "report"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(catalog_path: Path) -> pd.DataFrame:
    catalog = read_json(catalog_path)
    actions = catalog.get("actions", [])
    rows = [
        row("catalog_exists", "PASS" if catalog_path.exists() and catalog_path.stat().st_size > 0 else "FAIL", str(catalog_path), "Catalog file must exist."),
        row("actions_present", "PASS" if actions else "FAIL", str(catalog_path), f"actions={len(actions)}"),
    ]
    ids = [str(action.get("id", "")) for action in actions]
    duplicate_ids = sorted([item for item, count in Counter(ids).items() if count > 1])
    rows.append(row("unique_action_ids", "PASS" if not duplicate_ids else "FAIL", str(catalog_path), "duplicates=" + ", ".join(duplicate_ids)))

    for action in actions:
        action_id = str(action.get("id", ""))
        risk = str(action.get("risk_level", ""))
        goals = set(action.get("goal_types", []))
        command = str(action.get("command", ""))
        rows.append(row(f"{action_id}:risk", "PASS" if risk in VALID_RISKS else "FAIL", action_id, risk))
        rows.append(row(f"{action_id}:goals", "PASS" if goals and goals <= VALID_GOALS else "FAIL", action_id, ",".join(sorted(goals))))
        rows.append(
            row(
                f"{action_id}:command",
                "PASS" if command.startswith("python3 src/chrono_ehr/run_study.py ") else "FAIL",
                action_id,
                command,
            )
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, checks: pd.DataFrame) -> Path:
    table_path = project_root / "outputs" / "tables" / "agent_action_catalog_validation.csv"
    report_path = project_root / "outputs" / "reports" / "agent_action_catalog_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    report_path.write_text(
        f"""# Agent Action Catalog Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This validation checks the local Agent action catalog used for task routing and risk control. It is not a clinical decision check.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    checks = audit(args.catalog)
    report = write_outputs(args.project_root, checks)
    failures = int((checks["status"] != "PASS").sum())
    print(f"Agent action catalog checks: {len(checks)}")
    print(f"Failures: {failures}")
    print(f"Wrote {report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
