#!/usr/bin/env python3
"""Audit documented run_study commands in README and quickstart docs."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


DOCS_TO_AUDIT = [
    "README.md",
    "docs/quickstart_usage.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def known_run_study_flags(project_root: Path) -> set[str]:
    source = read_text(project_root / "src" / "chrono_ehr" / "run_study.py")
    return set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', source))


def documented_commands(project_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in DOCS_TO_AUDIT:
        path = project_root / relative
        for line_no, line in enumerate(read_text(path).splitlines(), start=1):
            stripped = line.strip()
            if "src/chrono_ehr/run_study.py" not in stripped:
                continue
            backtick_commands = [
                match
                for match in re.findall(r"`([^`]*src/chrono_ehr/run_study\.py[^`]*)`", stripped)
                if "src/chrono_ehr/run_study.py" in match
            ]
            if backtick_commands:
                for command in backtick_commands:
                    rows.append({"doc": relative, "line": str(line_no), "command": command.strip()})
            else:
                rows.append({"doc": relative, "line": str(line_no), "command": stripped.lstrip("- ").strip()})
    return rows


def audit_command(item: dict[str, str], flags: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    command = item["command"]
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return [{**item, "check": "shell_parse", "status": "FAIL", "detail": str(exc)}]

    rows.append({**item, "check": "shell_parse", "status": "PASS", "detail": f"tokens={len(tokens)}"})
    script_indices = [index for index, token in enumerate(tokens) if token == "src/chrono_ehr/run_study.py"]
    rows.append(
        {
            **item,
            "check": "run_study_script_present",
            "status": "PASS" if script_indices else "FAIL",
            "detail": f"indices={script_indices}",
        }
    )
    if not script_indices:
        return rows
    used_flags = [token for token in tokens[script_indices[0] + 1 :] if token.startswith("--")]
    unknown = sorted(set(used_flags) - flags)
    rows.append(
        {
            **item,
            "check": "flags_known",
            "status": "PASS" if not unknown else "FAIL",
            "detail": "flags=" + ",".join(used_flags) if not unknown else "unknown=" + ",".join(unknown),
        }
    )
    return rows


def audit(project_root: Path) -> pd.DataFrame:
    flags = known_run_study_flags(project_root)
    commands = documented_commands(project_root)
    rows: list[dict[str, str]] = []
    rows.append(
        {
            "doc": ",".join(DOCS_TO_AUDIT),
            "line": "",
            "command": "",
            "check": "documented_commands_present",
            "status": "PASS" if commands else "FAIL",
            "detail": f"commands={len(commands)}",
        }
    )
    for item in commands:
        rows.extend(audit_command(item, flags))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["doc", "line", "check", "status", "command", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_doc_command_audit.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_doc_command_audit.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Documentation Command Audit

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local documentation commands only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent doc-command checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
