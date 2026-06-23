#!/usr/bin/env python3
"""Validate ChronoEHR-Agent control-panel goal routing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent_control_panel import normalize_goal


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]

CASES = [
    ("status", "status"),
    ("继续完善我们的 agent", "status"),
    ("检查时间点和特征泄漏", "leakage"),
    ("leakage audit", "leakage"),
    ("先做验证和状态卡", "status"),
    ("做外部验证 eICU CHARLS", "external"),
    ("CDSL external benchmark", "external"),
    ("糖尿病 demo", "demo"),
    ("MIMIC 主线", "demo"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["input", "expected_goal_type", "actual_goal_type", "status"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, checks: pd.DataFrame) -> Path:
    table_path = project_root / "outputs" / "tables" / "agent_control_routing_validation.csv"
    report_path = project_root / "outputs" / "reports" / "agent_control_routing_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    failures = checks[checks["status"].ne("PASS")]
    text = f"""# Agent Control Routing Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This internal validation checks whether short user goals are routed to the expected local research-workflow mode. It is not a medical QA or clinical decision check.

## Check Table

{markdown_table(checks)}
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    rows = []
    for text, expected in CASES:
        actual = normalize_goal(text)
        status = "PASS" if actual == expected else "FAIL"
        rows.append(
            {
                "input": text,
                "expected_goal_type": expected,
                "actual_goal_type": actual,
                "status": status,
            }
        )
    checks = pd.DataFrame(rows)
    report_path = write_outputs(args.project_root, checks)
    failures = []
    for row in checks.itertuples(index=False):
        if row.status != "PASS":
            failures.append((row.input, row.expected_goal_type, row.actual_goal_type))
    print(f"Agent control routing checks: {len(CASES)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    for text, expected, actual in failures:
        print(f"FAIL: {text!r}: expected={expected}, actual={actual}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
