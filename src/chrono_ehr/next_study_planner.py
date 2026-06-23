#!/usr/bin/env python3
"""Recommend the next practical ChronoEHR-Agent study actions."""

from __future__ import annotations

import argparse
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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def existing_study_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    status_rank = {
        "complete_demo": 1,
        "model_ready": 2,
        "planned_data_pending": 4,
    }
    rows = []
    for study in registry.get("studies", []):
        status = study.get("status", "unknown")
        if status == "complete_demo":
            recommendation = "Use as primary demo and manuscript backbone."
            blocker = ""
            priority = "P1"
        elif status == "model_ready":
            recommendation = "Keep as cross-cohort replication; refresh if the shared feature pipeline changes."
            blocker = ""
            priority = "P2"
        else:
            recommendation = study.get("next_step", "Review study status before running.")
            blocker = "Unknown completion status."
            priority = "P3"
        rows.append(
            {
                "dataset": "MIMIC-IV",
                "study_id": study.get("id", ""),
                "status": status,
                "priority": priority,
                "rank": status_rank.get(status, 3),
                "recommended_action": recommendation,
                "blocker": blocker,
                "command": f"python3 src/chrono_ehr/run_study.py --study {study.get('id', '')}",
            }
        )
    return rows


def external_rows(project_root: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    external = read_csv(project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv")
    planned_by_dataset = {
        "eICU": next((item for item in registry.get("planned_studies", []) if item.get("id") == "eicu_temporal_mortality"), {}),
        "CHARLS": next((item for item in registry.get("planned_studies", []) if item.get("id") == "charls_incident_diabetes"), {}),
    }
    if external.empty:
        return [
            {
                "dataset": "external",
                "study_id": "external_readiness_summary",
                "status": "NOT_GENERATED",
                "priority": "P1",
                "rank": 1,
                "recommended_action": "Generate the external readiness summary.",
                "blocker": "",
                "command": "python3 src/chrono_ehr/run_study.py --external-readiness-summary",
            }
        ]

    rows = []
    for item in external.to_dict(orient="records"):
        dataset = item.get("dataset", "")
        status = item.get("local_status", "")
        if dataset == "CDSL" and status == "READY":
            recommendation = "Use as supplementary external method validation; report early-window metrics separately from full-stay reference."
            blocker = ""
            priority = "P1"
            rank = 2
            command = "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary"
        elif dataset == "eICU":
            planned = planned_by_dataset["eICU"]
            if status in {"READY_FOR_COHORT_CODE", "COHORT_READY", "FEATURE_READY", "BASELINE_READY"}:
                cohort_ready = (project_root / "data" / "processed" / "eicu_temporal_mortality_cohort.csv").exists()
                feature_ready = (project_root / "data" / "processed" / "eicu_first24h_feature_matrix_skeleton.csv").exists()
                baseline_ready = (project_root / "outputs" / "tables" / "eicu_first24h_logistic_baseline_metrics.csv").exists()
                figures_ready = (project_root / "outputs" / "tables" / "eicu_first24h_calibration_summary.csv").exists()
                summary_ready = (project_root / "outputs" / "tables" / "external_benchmark_summary_table.csv").exists()
                if baseline_ready and figures_ready and summary_ready:
                    recommendation = "Use the CDSL/eICU external benchmark summary table as the concise external-method evidence table."
                    command = "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary"
                elif baseline_ready and figures_ready:
                    recommendation = "Use CDSL and eICU outputs to build a concise external benchmark summary table."
                    command = "python3 src/chrono_ehr/run_study.py --external-benchmark-summary && python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary"
                elif baseline_ready:
                    recommendation = "Use the eICU baseline to add calibration summary and ROC/PR figures."
                    command = "python3 src/chrono_ehr/run_study.py --eicu-baseline-figures && python3 src/chrono_ehr/run_study.py --validate-eicu-baseline-figures"
                elif feature_ready:
                    recommendation = "Use the validated eICU first-24h feature matrix to define baseline specs and run a lightweight traditional baseline."
                    command = "python3 src/chrono_ehr/run_study.py --eicu-logistic-baseline && python3 src/chrono_ehr/run_study.py --validate-eicu-logistic-baseline"
                elif cohort_ready:
                    recommendation = "Use the validated eICU cohort to start first-24h lab/vital feature extraction skeleton and leakage gate."
                    command = "python3 src/chrono_ehr/run_study.py --eicu-temporal-features && python3 src/chrono_ehr/run_study.py --eicu-leakage-gate"
                else:
                    recommendation = "Use the external field-role catalog to build an eICU first-24h mortality cohort skeleton."
                    command = "python3 src/chrono_ehr/run_study.py --eicu-cohort && python3 src/chrono_ehr/run_study.py --validate-eicu-cohort"
                blocker = ""
                priority = "P1"
                rank = 2
            else:
                recommendation = "Wait for raw CSVs, then rerun readiness before writing cohort/model code."
                blocker = str(item.get("critical_blocker", "") or "eICU raw CSVs not confirmed locally.")
                priority = "P2"
                rank = 4
                command = planned.get("readiness_command", "python3 src/chrono_ehr/run_study.py --eicu-readiness")
        elif dataset == "CHARLS":
            planned = planned_by_dataset["CHARLS"]
            recommendation = "Wait for approved waves, then build wave variable map for incident diabetes."
            blocker = str(item.get("critical_blocker", "") or "CHARLS waves not confirmed locally.")
            priority = "P2"
            rank = 4
            command = planned.get("readiness_command", "python3 src/chrono_ehr/run_study.py --charls-readiness")
        else:
            recommendation = str(item.get("next_action", "") or "Review this dataset manually.")
            blocker = str(item.get("critical_blocker", "") or "")
            priority = "P3"
            rank = 3
            command = "python3 src/chrono_ehr/run_study.py --external-readiness-summary"
        rows.append(
            {
                "dataset": dataset,
                "study_id": item.get("recommended_first_task", ""),
                "status": status,
                "priority": priority,
                "rank": rank,
                "recommended_action": recommendation,
                "blocker": blocker,
                "command": command,
            }
        )
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    display = df.drop(columns=["rank"], errors="ignore")
    display = display.astype(object).where(pd.notna(display), "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, actions: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "next_study_action_plan.md"
    top = actions.sort_values(["rank", "priority", "dataset"]).iloc[0]
    text = f"""# Next Study Action Plan

这个报告是 ChronoEHR-Agent 的本地研究路线 planner。它不会给临床诊疗建议，只根据本地数据、已完成输出和 readiness 状态，建议下一步最实际的研究开发动作。

## Highest-Value Next Action

- 推荐：{top["recommended_action"]}
- 对应数据/任务：{top["dataset"]} / {top["study_id"]}
- 当前状态：{top["status"]}
- 可运行命令：`{top["command"]}`

## Action Table

{markdown_table(actions)}

## Practical Interpretation

- 现在最稳的主线仍然是 MIMIC-IV 慢病 30 天再入院 demo，因为它已经有 cohort、time-aware features、leakage audit、traditional baselines 和 manuscript outputs。
- CDSL 已经可以作为外部方法验证补充，但不能写成慢病再入院外部验证。
- eICU 已经推进到 baseline-ready ICU mortality benchmark；不要把它写成慢病再入院外部验证。
- CHARLS 已经有 protocol/config/checklist/readiness 脚本；真正写 wave-based cohort/model 代码前，需要先把批准后的 wave 数据放到固定路径并重新跑 readiness。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    rows = existing_study_rows(registry) + external_rows(args.project_root, registry)
    actions = pd.DataFrame(rows).sort_values(["rank", "priority", "dataset", "study_id"])
    table_path = args.project_root / "outputs" / "tables" / "next_study_action_plan.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    actions.to_csv(table_path, index=False)
    report_path = write_report(args.project_root, actions)
    print(f"Wrote {report_path}")
    print(actions[["priority", "dataset", "study_id", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
