#!/usr/bin/env python3
"""Generate a concise command index for stable ChronoEHR-Agent entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = DEFAULT_PROJECT / "configs" / "agent_entrypoints.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect_rows(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for group in config.get("groups", []):
        for command in group.get("commands", []):
            rows.append({"group": group.get("group", ""), **command})
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["group", "id", "risk_level", "command", "description"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    rows = collect_rows(config)
    table_path = args.project_root / "outputs" / "tables" / "agent_entrypoints.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_entrypoints.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Entrypoints

- Commands: {len(rows)}
- Boundary: stable local research-tool entrypoints only; no medical QA, diagnosis, or treatment recommendation.

## Command Index

{markdown_table(rows) if not rows.empty else "No commands configured."}
""",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    print(rows[["group", "id", "risk_level"]].to_string(index=False))


if __name__ == "__main__":
    main()
