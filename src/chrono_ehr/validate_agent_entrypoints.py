#!/usr/bin/env python3
"""Validate stable ChronoEHR-Agent entrypoint configuration and outputs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = DEFAULT_PROJECT / "configs" / "agent_entrypoints.json"
VALID_RISKS = {"safe", "expensive", "model", "report"}
COMMAND_PREFIX = ["python3", "src/chrono_ehr/run_study.py"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def configured_commands(config: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in config.get("groups", []):
        group_name = str(group.get("group", ""))
        for command in group.get("commands", []):
            rows.append(
                {
                    "group": group_name,
                    "id": str(command.get("id", "")),
                    "risk_level": str(command.get("risk_level", "")),
                    "command": str(command.get("command", "")),
                    "description": str(command.get("description", "")),
                }
            )
    return rows


def known_run_study_flags(project_root: Path) -> set[str]:
    source = read_text(project_root / "src" / "chrono_ehr" / "run_study.py")
    return set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', source))


def audit(project_root: Path, config_path: Path) -> pd.DataFrame:
    config = read_json(config_path)
    commands = configured_commands(config)
    flags = known_run_study_flags(project_root)
    rows = [
        row(
            "entrypoint_config_exists",
            "PASS" if config_path.exists() and config_path.stat().st_size > 0 else "FAIL",
            str(config_path),
            "config must exist and be non-empty",
        ),
        row("entrypoint_groups_present", "PASS" if config.get("groups") else "FAIL", str(config_path), f"groups={len(config.get('groups', []))}"),
        row("entrypoint_commands_present", "PASS" if commands else "FAIL", str(config_path), f"commands={len(commands)}"),
    ]

    ids = [item["id"] for item in commands]
    duplicate_ids = sorted([item for item, count in Counter(ids).items() if count > 1])
    rows.append(row("unique_entrypoint_ids", "PASS" if not duplicate_ids else "FAIL", str(config_path), "duplicates=" + ",".join(duplicate_ids)))

    for item in commands:
        command_id = item["id"] or "<missing_id>"
        command = item["command"]
        tokens = shlex.split(command) if command else []
        prefix_ok = tokens[:2] == COMMAND_PREFIX
        used_flags = [token for token in tokens[2:] if token.startswith("--")]
        unknown = sorted(set(used_flags) - flags)
        rows.append(row(f"{command_id}:risk", "PASS" if item["risk_level"] in VALID_RISKS else "FAIL", command_id, item["risk_level"]))
        rows.append(row(f"{command_id}:description", "PASS" if item["description"].strip() else "FAIL", command_id, "description present"))
        rows.append(row(f"{command_id}:prefix", "PASS" if prefix_ok else "FAIL", command_id, command))
        rows.append(row(f"{command_id}:flags_known", "PASS" if not unknown else "FAIL", command_id, "unknown=" + ",".join(unknown) if unknown else "flags=" + ",".join(used_flags)))

    table_path = project_root / "outputs" / "tables" / "agent_entrypoints.csv"
    report_path = project_root / "outputs" / "reports" / "agent_entrypoints.md"
    generated = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    rows.append(row("generated_entrypoint_table_exists", "PASS" if not generated.empty else "FAIL", str(table_path), f"rows={len(generated)}"))
    rows.append(row("generated_entrypoint_report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "report must exist"))
    if not generated.empty and "id" in generated.columns:
        generated_ids = set(generated["id"].astype(str))
        missing = sorted(set(ids) - generated_ids)
        extra = sorted(generated_ids - set(ids))
        rows.append(row("generated_table_matches_config_ids", "PASS" if not missing and not extra else "FAIL", str(table_path), f"missing={missing}; extra={extra}"))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root, args.config)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_entrypoints_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_entrypoints_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Entrypoints Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local research-tool commands only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent entrypoint checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
