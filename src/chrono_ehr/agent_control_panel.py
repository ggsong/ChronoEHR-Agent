#!/usr/bin/env python3
"""Local control panel for ChronoEHR-Agent research workflows."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = DEFAULT_PROJECT / "configs" / "study_registry.json"


SAFE_CHECK_COMMANDS = [
    "python3 src/chrono_ehr/run_study.py --study-capabilities",
    "python3 src/chrono_ehr/run_study.py --pipeline-steps",
    "python3 src/chrono_ehr/run_study.py --validate-feature-windows",
    "python3 src/chrono_ehr/run_study.py --leakage-gate",
    "python3 src/chrono_ehr/run_study.py --audit-extractor-windows",
    "python3 src/chrono_ehr/run_study.py --external-readiness-summary",
    "python3 src/chrono_ehr/run_study.py --delivery-readiness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--goal",
        default="status",
        help=(
            "Free-text goal or shortcut. Examples: status, resume, leakage, external, "
            "demo, diabetes, eicu, charls."
        ),
    )
    parser.add_argument(
        "--execute-safe-checks",
        action="store_true",
        help="Run only lightweight self-check commands before building the control panel.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_safe_checks(project_root: Path) -> None:
    for command in SAFE_CHECK_COMMANDS:
        subprocess.run(command.split(), cwd=project_root, check=True)


def normalize_goal(goal: str) -> str:
    text = goal.lower()
    if re.search(r"leak|泄漏|时间点|time|audit|审计", text):
        return "leakage"
    if re.search(r"external|外部|eicu|charls|cdsl", text):
        return "external"
    if re.search(r"demo|糖尿病|diabetes|mimic|主线", text):
        return "demo"
    if re.search(r"resume|继续|完善|状态|status|agent|self", text):
        return "status"
    return "status"


def study_rows(project_root: Path, registry: dict[str, Any]) -> pd.DataFrame:
    capability = read_csv(project_root / "outputs" / "tables" / "study_capability_summary.csv")
    pipeline = read_csv(project_root / "outputs" / "tables" / "pipeline_step_summary.csv")
    capability_by_id = capability.set_index("study_id").to_dict(orient="index") if not capability.empty else {}
    pipeline_by_id = pipeline.set_index("study_id").to_dict(orient="index") if not pipeline.empty else {}
    rows = []
    for study in registry.get("studies", []):
        study_id = study.get("id", "")
        cap = capability_by_id.get(study_id, {})
        pipe = pipeline_by_id.get(study_id, {})
        rows.append(
            {
                "study_id": study_id,
                "cohort": study.get("cohort", ""),
                "registry_status": study.get("status", ""),
                "capability_status": cap.get("overall_status", "UNKNOWN"),
                "capability_completion": cap.get("completion_percent", ""),
                "pipeline_completion": pipe.get("completion_percent", ""),
                "safe_rerun_command": pipe.get(
                    "recommended_safe_rerun",
                    f"python3 src/chrono_ehr/run_study.py --study {study_id} --skip-existing --no-expensive",
                ),
                "next_step": study.get("next_step", ""),
            }
        )
    return pd.DataFrame(rows)


def external_rows(project_root: Path) -> pd.DataFrame:
    external = read_csv(project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv")
    if external.empty:
        return pd.DataFrame(
            [
                {
                    "dataset": "external",
                    "local_status": "NOT_GENERATED",
                    "recommended_first_task": "",
                    "critical_blocker": "Run external readiness summary first.",
                    "command": "python3 src/chrono_ehr/run_study.py --external-readiness-summary",
                }
            ]
        )

    def command_for(row: pd.Series) -> str:
        dataset = str(row.get("dataset", ""))
        status = str(row.get("local_status", ""))
        if dataset == "CDSL":
            return "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary"
        if dataset == "eICU":
            if status == "BASELINE_READY":
                return "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary"
            if status == "FEATURE_READY":
                return "python3 src/chrono_ehr/run_study.py --eicu-logistic-baseline"
            if status == "COHORT_READY":
                return "python3 src/chrono_ehr/run_study.py --eicu-temporal-features"
            if status == "READY_FOR_COHORT_CODE":
                return "python3 src/chrono_ehr/run_study.py --eicu-cohort"
            return "python3 src/chrono_ehr/run_study.py --eicu-readiness"
        if dataset == "CHARLS":
            return "python3 src/chrono_ehr/run_study.py --charls-readiness"
        return "python3 src/chrono_ehr/run_study.py --external-readiness-summary"

    rows = external.copy()
    rows["command"] = rows.apply(command_for, axis=1)
    return rows


def readiness_row(project_root: Path) -> dict[str, Any]:
    readiness = read_csv(project_root / "outputs" / "tables" / "delivery_readiness_audit.csv")
    if readiness.empty:
        return {"overall": "UNKNOWN", "checks": 0, "failures": "", "command": "python3 src/chrono_ehr/run_study.py --delivery-readiness"}
    failures = int((readiness["status"] != "PASS").sum())
    return {
        "overall": "PASS" if failures == 0 else "FAIL",
        "checks": len(readiness),
        "failures": failures,
        "command": "python3 src/chrono_ehr/run_study.py --delivery-readiness",
    }


def action_rows(goal_type: str, studies: pd.DataFrame, external: pd.DataFrame, readiness: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if goal_type == "leakage":
        rows.extend(
            [
                {
                    "priority": "P1",
                    "agent_module": "Leakage Audit Agent",
                    "action": "Refresh prediction-time leakage gate and feature-window validation.",
                    "why": "先保证时间点边界正确，再考虑模型或写作。",
                    "command": "python3 src/chrono_ehr/run_study.py --validate-feature-windows && python3 src/chrono_ehr/run_study.py --leakage-gate && python3 src/chrono_ehr/run_study.py --audit-extractor-windows",
                    "safe_to_run_now": True,
                },
                {
                    "priority": "P2",
                    "agent_module": "Benchmark Agent",
                    "action": "Rerun ED LOS sensitivity if time-window specs change.",
                    "why": "这是当前最明确的边界变量敏感性检查。",
                    "command": "python3 src/chrono_ehr/run_study.py --ed-los-sensitivity",
                    "safe_to_run_now": True,
                },
            ]
        )
    elif goal_type == "external":
        for item in external.to_dict(orient="records"):
            status = item.get("local_status", "")
            priority = "P1" if item.get("dataset") == "CDSL" and status == "READY" else "P2"
            rows.append(
                {
                    "priority": priority,
                    "agent_module": "External Benchmark Agent",
                    "action": f"Handle {item.get('dataset', '')} readiness/status.",
                    "why": item.get("critical_blocker", "") or "Keep external validation boundaries explicit.",
                    "command": item.get("command", ""),
                    "safe_to_run_now": True,
                }
            )
    elif goal_type == "demo":
        diabetes = studies[studies["cohort"].eq("diabetes")]
        command = (
            diabetes.iloc[0]["safe_rerun_command"]
            if not diabetes.empty
            else "python3 src/chrono_ehr/run_study.py --study mimic_iv_3_1_diabetes_readmission --skip-existing --no-expensive"
        )
        rows.extend(
            [
                {
                    "priority": "P1",
                    "agent_module": "Study Design/Cohort/Benchmark Agents",
                    "action": "Refresh the MIMIC diabetes vertical slice without expensive scans.",
                    "why": "糖尿病仍然是最稳的第一版 demo 主线。",
                    "command": command,
                    "safe_to_run_now": True,
                },
                {
                    "priority": "P2",
                    "agent_module": "Study Registry / Runner",
                    "action": "Confirm all registered pipelines still have complete declared outputs.",
                    "why": "防止脚本增加后输出清单和真实文件状态脱节。",
                    "command": "python3 src/chrono_ehr/run_study.py --pipeline-steps",
                    "safe_to_run_now": True,
                },
            ]
        )
    else:
        rows.extend(
            [
                {
                    "priority": "P1",
                    "agent_module": "Study Registry / Runner",
                    "action": "Refresh lightweight Agent self-checks.",
                    "why": f"当前 delivery readiness 为 {readiness['overall']}，{readiness['checks']} checks，failures={readiness['failures']}。",
                    "command": "python3 src/chrono_ehr/run_study.py --study-capabilities && python3 src/chrono_ehr/run_study.py --pipeline-steps && python3 src/chrono_ehr/run_study.py --delivery-readiness",
                    "safe_to_run_now": True,
                },
                {
                    "priority": "P2",
                    "agent_module": "Leakage Audit Agent",
                    "action": "Refresh time-window and leakage checks before further model work.",
                    "why": "ChronoEHR 的核心差异化是时间点意识和 leakage audit。",
                    "command": "python3 src/chrono_ehr/run_study.py --validate-feature-windows && python3 src/chrono_ehr/run_study.py --leakage-gate",
                    "safe_to_run_now": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No rows."
    display = df[columns].astype(object).where(pd.notna(df[columns]), "")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_control_panel(
    project_root: Path,
    goal: str,
    goal_type: str,
    studies: pd.DataFrame,
    external: pd.DataFrame,
    actions: pd.DataFrame,
    readiness: dict[str, Any],
) -> Path:
    output = project_root / "outputs" / "reports" / "agent_control_panel.md"
    text = f"""# ChronoEHR-Agent Control Panel

- User goal: `{goal}`
- Interpreted goal type: `{goal_type}`
- Boundary: local research workflow control only; no medical QA, diagnosis, or treatment recommendation.
- Delivery readiness: `{readiness["overall"]}` ({readiness["checks"]} checks, failures={readiness["failures"]})

## Recommended Actions

{markdown_table(actions, ["priority", "agent_module", "action", "why", "command", "safe_to_run_now"])}

## Registered Study State

{markdown_table(studies, ["study_id", "cohort", "registry_status", "capability_status", "capability_completion", "pipeline_completion", "safe_rerun_command"])}

## External Dataset State

{markdown_table(external, ["dataset", "local_status", "recommended_first_task", "critical_blocker", "command"])}

## How To Use This

- 先看 `Recommended Actions`，它是当前目标下最适合执行的动作。
- `safe_to_run_now=True` 表示命令不应该重新扫描 MIMIC 大表，适合日常打磨。
- 真正要跑大表或重训模型时，再进入具体 study runner 或 benchmark command。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    if args.execute_safe_checks:
        run_safe_checks(args.project_root)
    registry = read_json(args.registry)
    goal_type = normalize_goal(args.goal)
    studies = study_rows(args.project_root, registry)
    external = external_rows(args.project_root)
    readiness = readiness_row(args.project_root)
    actions = action_rows(goal_type, studies, external, readiness)

    table_dir = args.project_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    studies.to_csv(table_dir / "agent_control_study_state.csv", index=False)
    external.to_csv(table_dir / "agent_control_external_state.csv", index=False)
    actions.to_csv(table_dir / "agent_control_recommended_actions.csv", index=False)
    output = write_control_panel(args.project_root, args.goal, goal_type, studies, external, actions, readiness)

    print(f"Wrote {output}")
    print(actions[["priority", "agent_module", "action"]].to_string(index=False))


if __name__ == "__main__":
    main()
