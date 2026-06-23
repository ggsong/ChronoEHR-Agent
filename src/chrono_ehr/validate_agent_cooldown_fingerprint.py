#!/usr/bin/env python3
"""Validate Agent cooldown fingerprint configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agent_cooldown_fingerprint import DEFAULT_CONFIG, cooldown_input_paths
from mimic_diabetes_baseline import DEFAULT_PROJECT


VOLATILE_PREFIXES = ("outputs/",)
REQUIRED_INPUTS = {
    "src/chrono_ehr/agent_cooldown_fingerprint.py",
    "src/chrono_ehr/agent_next_task_planner.py",
    "src/chrono_ehr/agent_task_queue.py",
    "src/chrono_ehr/agent_task_queue_runner.py",
    "configs/agent_action_catalog.json",
    "configs/agent_entrypoints.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path, config_path: Path) -> pd.DataFrame:
    config = read_json(config_path)
    inputs = cooldown_input_paths(project_root)
    rows = [
        row(
            "cooldown_config_exists",
            "PASS" if config_path.exists() and config_path.stat().st_size > 0 else "FAIL",
            str(config_path),
            "config must exist and be non-empty",
        ),
        row("version_present", "PASS" if str(config.get("version", "")).strip() else "FAIL", str(config_path), str(config.get("version", ""))),
        row("boundary_declared", "PASS" if "no medical QA" in str(config.get("boundary", "")) else "FAIL", str(config_path), str(config.get("boundary", ""))),
        row("fingerprint_inputs_present", "PASS" if inputs else "FAIL", str(config_path), f"inputs={len(inputs)}"),
    ]
    duplicates = sorted({item for item in inputs if inputs.count(item) > 1})
    rows.append(row("fingerprint_inputs_unique", "PASS" if not duplicates else "FAIL", str(config_path), "duplicates=" + ",".join(duplicates)))

    non_relative = [item for item in inputs if item.startswith("/") or ".." in Path(item).parts]
    rows.append(row("fingerprint_inputs_are_relative", "PASS" if not non_relative else "FAIL", str(config_path), "bad=" + ",".join(non_relative)))

    missing = [item for item in inputs if not (project_root / item).is_file()]
    rows.append(row("fingerprint_inputs_exist", "PASS" if not missing else "FAIL", str(config_path), "missing=" + ",".join(missing)))

    volatile = [item for item in inputs if item.startswith(VOLATILE_PREFIXES)]
    rows.append(
        row(
            "volatile_outputs_excluded",
            "PASS" if not volatile else "FAIL",
            str(config_path),
            "volatile_inputs=" + ",".join(volatile),
        )
    )

    allowed_prefix = [item for item in inputs if not (item.startswith("src/") or item.startswith("configs/"))]
    rows.append(
        row(
            "fingerprint_inputs_are_code_or_config",
            "PASS" if not allowed_prefix else "FAIL",
            str(config_path),
            "unexpected_prefix=" + ",".join(allowed_prefix),
        )
    )

    missing_required = sorted(REQUIRED_INPUTS - set(inputs))
    rows.append(row("required_control_inputs_present", "PASS" if not missing_required else "FAIL", str(config_path), "missing=" + ",".join(missing_required)))

    source = (project_root / "src" / "chrono_ehr" / "agent_cooldown_fingerprint.py").read_text(encoding="utf-8")
    rows.append(
        row(
            "config_file_affects_fingerprint",
            "PASS" if "config=" in source and str(DEFAULT_CONFIG) in source else "FAIL",
            "src/chrono_ehr/agent_cooldown_fingerprint.py",
            "expects config file digest in cooldown fingerprint",
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
    config_path = args.config if args.config else args.project_root / DEFAULT_CONFIG
    checks = audit(args.project_root, config_path)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_cooldown_fingerprint_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_cooldown_fingerprint_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Cooldown Fingerprint Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local Agent cooldown configuration only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent cooldown fingerprint checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
