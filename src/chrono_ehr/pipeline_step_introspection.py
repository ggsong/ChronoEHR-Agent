#!/usr/bin/env python3
"""Inspect registered pipeline steps without running them."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


DEFAULT_REGISTRY = DEFAULT_PROJECT / "configs" / "study_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def import_steps(script_path: Path) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pipeline script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    steps = getattr(module, "STEPS", None)
    if not isinstance(steps, list):
        return []
    return steps


def output_status(project_root: Path, outputs: list[str]) -> tuple[str, str]:
    if not outputs:
        return "NO_OUTPUT_DECLARED", ""
    existing = [output for output in outputs if (project_root / output).exists()]
    missing = [output for output in outputs if output not in existing]
    if not missing:
        return "COMPLETE", ""
    if existing:
        return "PARTIAL", "; ".join(missing)
    return "MISSING", "; ".join(missing)


def collect_rows(project_root: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in registry.get("studies", []):
        pipeline = study.get("pipeline", "")
        script_path = project_root / pipeline
        if not script_path.exists():
            rows.append(
                {
                    "study_id": study.get("id", ""),
                    "cohort": study.get("cohort", ""),
                    "step_order": None,
                    "step_name": "pipeline_script",
                    "script": pipeline,
                    "expensive": None,
                    "n_outputs": 0,
                    "output_status": "MISSING_PIPELINE",
                    "missing_outputs": pipeline,
                    "run_only_command": "",
                }
            )
            continue
        for index, step in enumerate(import_steps(script_path), start=1):
            outputs = step.get("outputs", [])
            status, missing = output_status(project_root, outputs)
            name = step.get("name", "")
            rows.append(
                {
                    "study_id": study.get("id", ""),
                    "cohort": study.get("cohort", ""),
                    "step_order": index,
                    "step_name": name,
                    "script": step.get("script", ""),
                    "expensive": bool(step.get("expensive", False)),
                    "n_outputs": len(outputs),
                    "output_status": status,
                    "missing_outputs": missing,
                    "run_only_command": f"python3 src/chrono_ehr/run_study.py --study {study.get('id', '')} --only {name}",
                }
            )
    return rows


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    grouped = (
        details.groupby(["study_id", "cohort"])
        .agg(
            n_steps=("step_name", "count"),
            expensive_steps=("expensive", lambda values: int(pd.Series(values).fillna(False).sum())),
            complete_steps=("output_status", lambda values: int((values == "COMPLETE").sum())),
            partial_steps=("output_status", lambda values: int((values == "PARTIAL").sum())),
            missing_steps=("output_status", lambda values: int((values == "MISSING").sum())),
        )
        .reset_index()
    )
    grouped["completion_percent"] = (grouped["complete_steps"] / grouped["n_steps"] * 100).round(1)
    grouped["recommended_safe_rerun"] = grouped["study_id"].map(
        lambda study_id: f"python3 src/chrono_ehr/run_study.py --study {study_id} --skip-existing --no-expensive"
    )
    return grouped.sort_values(["completion_percent", "study_id"], ascending=[False, True])


def markdown_table(df: pd.DataFrame) -> str:
    display = df.astype(object).where(pd.notna(df), "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, details: pd.DataFrame, summary: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "pipeline_step_introspection.md"
    missing = details[details["output_status"].isin(["MISSING", "PARTIAL", "MISSING_PIPELINE"])]
    text = f"""# Pipeline Step Introspection

这个报告读取每个 registered study runner 里的 `STEPS`，但不运行任何分析。它的作用是告诉你：每个队列 pipeline 有哪些步骤、哪些步骤会扫大表比较费时、哪些输出已经存在、如果只想补某一步应该运行什么命令。

## Study-Level Summary

{markdown_table(summary)}

## Missing Or Partial Step Outputs

{markdown_table(missing) if not missing.empty else "All declared step outputs are present."}

## Interpretation

- `expensive=True` 通常表示会扫描 MIMIC 大表，例如 `labevents`、`prescriptions` 或 ICU 事件表；外出吃饭或睡觉前可以跑，平时调试可加 `--no-expensive`。
- `--skip-existing` 适合接着上次结果继续，不会重复生成已经存在的步骤输出。
- `--only STEP_NAME` 适合只补某一个失败或缺失步骤。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    details = pd.DataFrame(collect_rows(args.project_root, registry))
    summary = summarize(details)
    table_dir = args.project_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(table_dir / "pipeline_step_introspection.csv", index=False)
    summary.to_csv(table_dir / "pipeline_step_summary.csv", index=False)
    report = write_report(args.project_root, details, summary)
    print(f"Wrote {report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
