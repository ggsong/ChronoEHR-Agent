#!/usr/bin/env python3
"""Route a natural-language research task to ChronoEHR-Agent actions."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agent_control_panel import normalize_goal


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = DEFAULT_PROJECT / "configs" / "agent_action_catalog.json"

RISK_ORDER = {"safe": 0, "expensive": 1, "model": 2, "report": 3}
BUDGET_ACTION_LIMITS = {"low_quota": 4, "normal": 8, "night_run": 20}
AGENT_CONTROL_FOCUS_ACTION_IDS = {
    "agent_control_status",
    "agent_state",
    "agent_control_routing_validation",
    "agent_command_lint",
    "agent_doctor",
    "agent_doctor_validation",
    "agent_status_card",
    "agent_status_card_validation",
    "agent_progress_score",
    "agent_progress_score_validation",
    "agent_task_execution_validation",
    "agent_task_scenarios",
    "agent_task_scenario_validation",
    "agent_task_queue",
    "agent_task_queue_validation",
    "agent_task_queue_run",
    "agent_task_queue_execution_validation",
    "agent_cooldown_fingerprint_validation",
    "agent_entrypoints_validation",
    "agent_control_consistency",
    "agent_dependency_audit",
    "agent_doc_command_audit",
    "agent_handoff_checklist",
    "agent_boundary_audit",
    "agent_artifact_freshness",
    "delivery_readiness",
}
EXECUTION_COLUMNS = ["id", "status", "risk_level", "command", "started_at", "duration_seconds", "returncode", "detail"]
POST_REFRESH_COLUMNS = ["id", "status", "command", "started_at", "duration_seconds", "returncode", "detail"]
SCENARIO_COLUMNS = [
    "scenario_id",
    "title",
    "when_to_use",
    "execution_style",
    "auto_run_policy",
    "next_step_hint",
    "example_task",
]
TASK_SCENARIOS = {
    "low_quota_self_check": {
        "title": "低额度轻量自检",
        "when_to_use": "用户说快没额度、只剩少量额度、想做一点简单检查。",
        "execution_style": "只选择少量 safe actions；优先验证状态、命令和最近任务执行结果。",
        "auto_run_policy": "可以配合 --agent-task-execute-safe 自动运行；不碰大表、不训练模型、不生成正式报告。",
        "next_step_hint": "额度恢复后先看 status card、agent_state 和 recovery plan，再决定是否继续长任务。",
        "example_task": "我快没额度了，只剩2%，做一点简单自检",
    },
    "night_run_safe_plus_deferred": {
        "title": "睡觉长跑规划",
        "when_to_use": "用户说要睡觉、电脑不关、可以多跑一些，但未明确允许重训模型。",
        "execution_style": "允许规划更多步骤；safe actions 可自动执行，expensive/model actions 必须留在 deferred actions。",
        "auto_run_policy": "默认不自动扫大表或训练模型；需要通过 runbook confirmation gate 显式确认。",
        "next_step_hint": "睡醒后先看 runbook state machine、handoff checklist 和 artifact freshness。",
        "example_task": "我要去睡觉，可以多跑一些但不要重训模型",
    },
    "external_readiness_first": {
        "title": "外部数据 readiness 优先",
        "when_to_use": "任务提到 eICU、CHARLS、CDSL、外部验证或数据库下载状态。",
        "execution_style": "先做路径、schema、字段角色、已完成输出的 readiness/summary；模型或特征重算放到后续确认。",
        "auto_run_policy": "readiness 和 summary 属于 safe；大表特征抽取和 baseline 训练保留在 deferred actions。",
        "next_step_hint": "如果 readiness 显示 DATA_PENDING，就先补路径或等待授权数据；不要硬写模型结果。",
        "example_task": "继续做 eICU 外部验证准备，不要重训模型",
    },
    "validation_first": {
        "title": "验证优先修复",
        "when_to_use": "用户说检查、验证、有没有偏离主线、下一步做什么，或任务没有明确要求重训/报告。",
        "execution_style": "优先运行 validation、lint、boundary、freshness、doctor 和 state refresh。",
        "auto_run_policy": "safe validation 可以自动运行；失败后先生成 recovery plan，而不是扩大任务范围。",
        "next_step_hint": "若全部 PASS，再进入更具体的 cohort、feature、外部验证或报告工作。",
        "example_task": "先暂停一下，做一下检查。从主线来看有没有偏离？",
    },
    "agent_control_focus": {
        "title": "Agent 控制层优先",
        "when_to_use": "用户明确要求完善、打磨或继续做 Agent 本体、控制层、队列、状态卡、自检、runbook 或恢复机制。",
        "execution_style": "只选择 Agent 控制层、验证、状态、队列、freshness、doctor 和 handoff 相关 safe actions；不主动转向汇报材料或论文写作。",
        "auto_run_policy": "可以配合 --agent-task-execute-safe 自动运行 safe 控制层检查；report/model/expensive actions 必须保持 deferred 或等待明确任务。",
        "next_step_hint": "先看 task queue、next tasks、state、doctor 和 freshness，再决定是否继续扩展控制层审计。",
        "example_task": "先完善 Agent 控制层，不要做汇报材料",
    },
    "model_or_report_explicit": {
        "title": "模型或报告显式任务",
        "when_to_use": "用户明确说重训模型、baseline、Random Forest、XGBoost、导出 Word、manuscript 或 report。",
        "execution_style": "可以生成模型或报告计划；是否执行取决于风险门和用户确认。",
        "auto_run_policy": "模型和正式报告不由 --agent-task-execute-safe 自动执行。",
        "next_step_hint": "先确认数据、特征窗口、泄漏审计和 baseline 输出是否新鲜，再运行模型/报告命令。",
        "example_task": "重训 random forest baseline",
    },
}
POST_RUN_REFRESH_COMMANDS = [
    ("agent_recovery_plan", "python3 src/chrono_ehr/run_study.py --agent-recovery-plan"),
    ("agent_next_tasks", "python3 src/chrono_ehr/run_study.py --agent-next-tasks"),
    ("agent_state", "python3 src/chrono_ehr/run_study.py --agent-state"),
    ("agent_state_validation", "python3 src/chrono_ehr/run_study.py --validate-agent-state"),
    ("agent_handoff_checklist", "python3 src/chrono_ehr/run_study.py --agent-handoff-checklist"),
    ("agent_progress_score", "python3 src/chrono_ehr/run_study.py --agent-progress-score"),
    ("agent_progress_score_validation", "python3 src/chrono_ehr/run_study.py --validate-agent-progress-score"),
    ("agent_command_lint", "python3 src/chrono_ehr/run_study.py --agent-command-lint"),
    ("agent_control_consistency", "python3 src/chrono_ehr/run_study.py --agent-control-consistency"),
]
ACTION_PRIORITY = {
    "agent_control_status": 10,
    "agent_task_scenarios": 11,
    "agent_task_scenario_validation": 12,
    "agent_task_queue": 13,
    "agent_task_queue_validation": 14,
    "agent_task_queue_run": 15,
    "agent_task_queue_execution_validation": 16,
    "agent_cooldown_fingerprint_validation": 17,
    "agent_state": 18,
    "agent_state_validation": 19,
    "agent_control_routing_validation": 20,
    "agent_control_consistency": 20,
    "agent_command_lint": 21,
    "agent_boundary_audit": 22,
    "agent_handoff_checklist": 23,
    "agent_artifact_freshness": 24,
    "agent_doctor": 25,
    "delivery_readiness": 26,
    "study_capability_audit": 20,
    "pipeline_step_introspection": 21,
    "feature_window_validation": 30,
    "prediction_time_leakage_gate": 31,
    "extractor_window_usage_audit": 32,
    "external_readiness_summary": 40,
    "validate_external_benchmark_summary": 41,
    "external_benchmark_summary": 42,
    "eicu_readiness": 43,
    "eicu_cohort": 44,
    "validate_eicu_cohort": 45,
    "validate_eicu_temporal_features": 46,
    "eicu_leakage_gate": 47,
    "validate_eicu_logistic_baseline": 48,
    "validate_eicu_baseline_figures": 49,
    "eicu_baseline_figures": 50,
    "charls_readiness": 51,
    "cdsl_leakage_audit": 52,
    "eicu_temporal_features": 70,
    "eicu_logistic_baseline": 90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--task", required=True, help="Natural-language user task.")
    parser.add_argument(
        "--risk-mode",
        choices=["safe", "expensive", "model", "report", "auto"],
        default="auto",
        help="Maximum action risk level allowed in the plan.",
    )
    parser.add_argument(
        "--execute-safe",
        action="store_true",
        help="Execute only selected safe actions. Expensive/model/report actions are never executed by this flag.",
    )
    parser.add_argument(
        "--post-run-refresh",
        action="store_true",
        help="After --execute-safe, refresh recovery plan, next tasks, state, handoff checklist, and status card.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_goal_type(task: str) -> str:
    text = task.lower()
    model_negated = re.search(r"不要.*(模型|重训|训练)|不.*(重训|训练)|no model|without model", text)
    report_negated = re.search(r"不要.*(report|报告|论文|word|docx)|先不.*(report|报告|论文|word|docx)|no report|without report", text)
    external_mentioned = re.search(r"external|外部|eicu|charls|cdsl", text)
    leakage_mentioned = re.search(r"leak|泄漏|时间点|time|audit|审计", text)
    if external_mentioned and leakage_mentioned:
        return "external"
    if re.search(r"leak|泄漏|时间点|time|audit|审计", text):
        return "leakage"
    if re.search(r"external|外部|eicu|charls|cdsl", text):
        return "external"
    if re.search(r"睡觉|一晚|overnight|多跑|跑久|大表|扫表|expensive", text):
        return "demo"
    if re.search(r"model|模型|baseline|random forest|xgboost|lightgbm|重训|训练", text) and not model_negated:
        return "model"
    if re.search(r"report|论文|word|docx|写作|manuscript", text) and not report_negated:
        return "report"
    return normalize_goal(task)


def infer_risk_mode(task: str, requested: str) -> str:
    if requested != "auto":
        return requested
    text = task.lower()
    model_negated = re.search(r"不要.*(模型|重训|训练)|不.*(重训|训练)|no model|without model", text)
    report_negated = re.search(r"不要.*(report|报告|论文|word|docx)|先不.*(report|报告|论文|word|docx)|no report|without report", text)
    if re.search(r"睡觉|一晚|overnight|多跑|跑久|大表|扫表|expensive", text):
        return "expensive"
    if re.search(r"模型|重训|训练|baseline|random forest|xgboost|lightgbm|model", text) and not model_negated:
        return "model"
    if re.search(r"报告|论文|word|docx|manuscript|report", text) and not report_negated:
        return "report"
    return "safe"


def infer_budget_mode(task: str) -> str:
    text = task.lower()
    if re.search(r"没额度|快没额度|只剩|2%|低额度|low quota|low budget|快结束", text):
        return "low_quota"
    if re.search(r"睡觉|一晚|overnight|电脑不会关|多跑|跑久|长跑", text):
        return "night_run"
    return "normal"


def infer_task_summary(task: str, goal_type: str, risk_mode: str, budget_mode: str) -> dict[str, str]:
    scenario = infer_task_scenario(task, goal_type, risk_mode, budget_mode)
    return {
        "task": task,
        "goal_type": goal_type,
        "risk_mode": risk_mode,
        "budget_mode": budget_mode,
        "scenario_id": scenario["scenario_id"],
        "plain_language": (
            "这是一个本地 EHR 研究工作流任务。Agent 会优先选择轻量检查和已有结果验证；"
            "涉及扫大表、训练模型或生成正式报告的步骤会被标注为更高风险。"
        ),
        "boundary": "No medical QA, diagnosis, or treatment recommendation.",
    }


def infer_task_scenario(task: str, goal_type: str, risk_mode: str, budget_mode: str) -> dict[str, str]:
    text = task.lower()
    if budget_mode == "low_quota":
        scenario_id = "low_quota_self_check"
    elif budget_mode == "night_run":
        scenario_id = "night_run_safe_plus_deferred"
    elif goal_type == "external":
        scenario_id = "external_readiness_first"
    elif re.search(r"完善.*agent|打磨.*agent|agent.*控制|控制层|队列|task queue|runbook|状态卡|自检|doctor|freshness|handoff", text):
        scenario_id = "agent_control_focus"
    elif risk_mode in {"model", "report"}:
        scenario_id = "model_or_report_explicit"
    elif re.search(r"检查|验证|偏离|下一步|状态|status|validate|check|audit", text):
        scenario_id = "validation_first"
    else:
        scenario_id = "validation_first"
    return {"scenario_id": scenario_id, **TASK_SCENARIOS[scenario_id]}


def infer_action_phase(action: dict[str, Any]) -> str:
    action_id = str(action.get("id", ""))
    module = str(action.get("module", ""))
    risk_level = str(action.get("risk_level", "safe"))
    if action_id.startswith("agent_") or "control" in action_id:
        return "agent_control"
    if "readiness" in action_id or "capability" in action_id or "introspection" in action_id:
        return "readiness"
    if "leakage" in action_id or "window" in action_id or "time" in action_id:
        return "temporal_leakage"
    if "cohort" in action_id:
        return "cohort"
    if risk_level == "model" or "baseline" in action_id:
        return "modeling"
    if risk_level == "report" or "Report" in module or "docx" in action_id:
        return "reporting"
    if risk_level == "expensive":
        return "expensive_feature_refresh"
    return "safe_validation"


def execution_policy_for(action: dict[str, Any], risk_mode: str, budget_mode: str) -> tuple[str, str]:
    risk_level = str(action.get("risk_level", "safe"))
    if risk_level == "safe":
        return "auto_safe_if_requested", "can run with --agent-task-execute-safe"
    if risk_level == "expensive":
        return "confirm_before_run", "may scan large raw tables; run through runbook or explicit confirmation"
    if risk_level == "model":
        return "confirm_before_modeling", "may train/recalibrate models; not executed by safe mode"
    if risk_level == "report":
        return "defer_unless_requested", "report writing is lower priority while polishing the Agent"
    return "manual_review", f"risk_mode={risk_mode}; budget_mode={budget_mode}"


def annotate_actions(actions: pd.DataFrame, risk_mode: str, budget_mode: str) -> pd.DataFrame:
    if actions.empty:
        return actions
    annotated = actions.copy()
    phases = []
    policies = []
    reasons = []
    for action in annotated.to_dict(orient="records"):
        phases.append(infer_action_phase(action))
        policy, reason = execution_policy_for(action, risk_mode, budget_mode)
        policies.append(policy)
        reasons.append(reason)
    annotated["phase"] = phases
    annotated["execution_policy"] = policies
    annotated["policy_reason"] = reasons
    return annotated


def action_allowed_for_scenario(action: dict[str, Any], scenario_id: str) -> bool:
    if scenario_id != "agent_control_focus":
        return True
    return str(action.get("id", "")) in AGENT_CONTROL_FOCUS_ACTION_IDS


def select_actions(
    catalog: dict[str, Any],
    goal_type: str,
    risk_mode: str,
    budget_mode: str,
    scenario_id: str = "",
) -> pd.DataFrame:
    max_risk = RISK_ORDER[risk_mode]
    rows = []
    for action in catalog.get("actions", []):
        if not action_allowed_for_scenario(action, scenario_id):
            continue
        action_risk = action.get("risk_level", "safe")
        if RISK_ORDER[action_risk] > max_risk:
            continue
        if goal_type not in action.get("goal_types", []) and goal_type != "status":
            continue
        if goal_type == "status" and "status" not in action.get("goal_types", []):
            continue
        rows.append(action)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["risk_rank"] = df["risk_level"].map(RISK_ORDER)
    df["priority_rank"] = df["id"].map(ACTION_PRIORITY).fillna(100).astype(int)
    df["execute_selected"] = df["risk_level"].eq("safe")
    df["budget_mode"] = budget_mode
    df = df.sort_values(["risk_rank", "priority_rank", "module", "id"]).reset_index(drop=True)
    action_limit = 10 if scenario_id == "agent_control_focus" else BUDGET_ACTION_LIMITS.get(budget_mode, BUDGET_ACTION_LIMITS["normal"])
    df = df.head(action_limit)
    return annotate_actions(df, risk_mode, budget_mode)


def deferred_actions(catalog: dict[str, Any], goal_type: str, risk_mode: str, scenario_id: str = "") -> pd.DataFrame:
    max_risk = RISK_ORDER[risk_mode]
    rows = []
    for action in catalog.get("actions", []):
        if not action_allowed_for_scenario(action, scenario_id):
            continue
        if goal_type not in action.get("goal_types", []) and not (goal_type == "status" and "status" in action.get("goal_types", [])):
            continue
        action_risk = action.get("risk_level", "safe")
        if RISK_ORDER[action_risk] <= max_risk:
            continue
        rows.append({**action, "defer_reason": f"risk_level={action_risk} exceeds risk_mode={risk_mode}"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["risk_rank"] = df["risk_level"].map(RISK_ORDER)
    df["priority_rank"] = df["id"].map(ACTION_PRIORITY).fillna(100).astype(int)
    df = df.sort_values(["risk_rank", "priority_rank", "module", "id"]).reset_index(drop=True)
    return annotate_actions(df.head(8), risk_mode, "normal")


def execute_selected(project_root: Path, actions: pd.DataFrame, allowed_risk_levels: tuple[str, ...] = ("safe",)) -> list[dict[str, str]]:
    rows = []
    for action in actions.to_dict(orient="records"):
        command = str(action["command"])
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        start = time.monotonic()
        risk_level = str(action.get("risk_level", ""))
        if risk_level not in allowed_risk_levels:
            rows.append(
                {
                    "id": action["id"],
                    "status": "SKIPPED",
                    "risk_level": risk_level,
                    "command": command,
                    "started_at": started_at,
                    "duration_seconds": "0.000",
                    "returncode": "",
                    "detail": "risk level not allowed for this execution mode",
                }
            )
            continue
        completed = subprocess.run(shlex.split(command), cwd=project_root, text=True, capture_output=True)
        duration = time.monotonic() - start
        rows.append(
            {
                "id": action["id"],
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "risk_level": risk_level,
                "command": command,
                "started_at": started_at,
                "duration_seconds": f"{duration:.3f}",
                "returncode": str(completed.returncode),
                "detail": (completed.stdout + completed.stderr).strip()[-1000:],
            }
        )
        if completed.returncode != 0:
            break
    return rows


def post_run_refresh(project_root: Path) -> list[dict[str, str]]:
    rows = []
    for refresh_id, command in POST_RUN_REFRESH_COMMANDS:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        start = time.monotonic()
        completed = subprocess.run(shlex.split(command), cwd=project_root, text=True, capture_output=True)
        duration = time.monotonic() - start
        rows.append(
            {
                "id": refresh_id,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "command": command,
                "started_at": started_at,
                "duration_seconds": f"{duration:.3f}",
                "returncode": str(completed.returncode),
                "detail": (completed.stdout + completed.stderr).strip()[-1000:],
            }
        )
        if completed.returncode != 0:
            break
    return rows


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No matching actions."
    display = df[columns].astype(object).where(pd.notna(df[columns]), "")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_outputs(
    project_root: Path,
    task: str,
    goal_type: str,
    risk_mode: str,
    budget_mode: str,
    actions: pd.DataFrame,
    deferred: pd.DataFrame,
    execution_rows: list[dict[str, str]],
    post_refresh_rows: list[dict[str, str]],
    replace_latest_history: bool = False,
) -> Path:
    table_dir = project_root / "outputs" / "tables"
    report_dir = project_root / "outputs" / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    actions.to_csv(table_dir / "agent_task_plan.csv", index=False)
    deferred.to_csv(table_dir / "agent_task_deferred_actions.csv", index=False)
    execution = pd.DataFrame(execution_rows, columns=EXECUTION_COLUMNS)
    execution.to_csv(table_dir / "agent_task_execution.csv", index=False)
    post_refresh = pd.DataFrame(post_refresh_rows, columns=POST_REFRESH_COLUMNS)
    post_refresh.to_csv(table_dir / "agent_task_post_run_refresh.csv", index=False)
    output = report_dir / "agent_task_plan.md"
    profile = infer_task_summary(task, goal_type, risk_mode, budget_mode)
    scenario = infer_task_scenario(task, goal_type, risk_mode, budget_mode)
    scenario_df = pd.DataFrame([{column: scenario.get(column, "") for column in SCENARIO_COLUMNS}])
    scenario_df.to_csv(table_dir / "agent_task_scenario.csv", index=False)
    execution_text = "Not executed." if execution.empty else markdown_table(
        execution, ["id", "status", "risk_level", "duration_seconds", "returncode", "command", "detail"]
    )
    post_refresh_text = "Not requested." if post_refresh.empty else markdown_table(
        post_refresh, ["id", "status", "duration_seconds", "returncode", "command", "detail"]
    )
    deferred_text = "No deferred actions." if deferred.empty else markdown_table(
        deferred, ["id", "module", "risk_level", "phase", "execution_policy", "command", "defer_reason"]
    )
    text = f"""# Agent Task Plan

- User task: `{task}`
- Interpreted goal type: `{goal_type}`
- Risk mode: `{risk_mode}`
- Budget mode: `{budget_mode}`
- Scenario: `{profile["scenario_id"]}` ({scenario["title"]})
- Boundary: local research workflow routing only; no medical QA, diagnosis, or treatment recommendation.
- Plain-language summary: {profile["plain_language"]}

## Task Scenario

{markdown_table(scenario_df, SCENARIO_COLUMNS)}

## Selected Actions

{markdown_table(actions, ["id", "module", "risk_level", "phase", "execution_policy", "priority_rank", "command", "description"])}

## Deferred Actions

{deferred_text}

## Execution

{execution_text}

## Post-Run Refresh

{post_refresh_text}

## Interpretation

- `safe` actions inspect existing outputs, configs, or lightweight audits.
- `expensive` actions may scan large raw tables.
- `model` actions may train or recalibrate models.
- `report` actions create manuscript/report artifacts and are not prioritized while polishing the Agent itself.
- `low_quota` keeps the plan short and safe; `night_run` may include expensive non-model actions when the task allows them.
- `execution_policy` explains whether an action can run automatically under `--agent-task-execute-safe` or needs explicit confirmation.
- `post-run refresh` updates recovery plan, next tasks, persistent state, handoff checklist, progress score, command lint, and control consistency after safe execution. Final status-card refresh should run after any outer queue runner writes its execution result.
"""
    output.write_text(text, encoding="utf-8")
    state_dir = project_root / "outputs" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    failures = int((execution["status"] == "FAIL").sum()) if not execution.empty and "status" in execution else 0
    refresh_failures = int((post_refresh["status"] == "FAIL").sum()) if not post_refresh.empty and "status" in post_refresh else 0
    executed = int(execution["status"].isin(["PASS", "FAIL"]).sum()) if not execution.empty and "status" in execution else 0
    history_row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "task": task,
                "goal_type": goal_type,
                "risk_mode": risk_mode,
                "budget_mode": budget_mode,
                "selected_actions": len(actions),
                "deferred_actions": len(deferred),
                "executed_actions": executed,
                "failures": failures,
                "post_refresh_steps": len(post_refresh),
                "post_refresh_failures": refresh_failures,
                "report": str(output.relative_to(project_root)),
            }
        ]
    )
    history_path = state_dir / "agent_task_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        if replace_latest_history and not history.empty and str(history.tail(1).iloc[0].get("task", "")) == task:
            history = history.iloc[:-1]
        history = pd.concat([history, history_row], ignore_index=True).tail(100)
    else:
        history = history_row
    history.to_csv(history_path, index=False)
    return output


def main() -> None:
    args = parse_args()
    catalog = read_json(args.catalog)
    goal_type = infer_goal_type(args.task)
    risk_mode = infer_risk_mode(args.task, args.risk_mode)
    budget_mode = infer_budget_mode(args.task)
    scenario_id = infer_task_scenario(args.task, goal_type, risk_mode, budget_mode)["scenario_id"]
    actions = select_actions(catalog, goal_type, risk_mode, budget_mode, scenario_id)
    deferred = deferred_actions(catalog, goal_type, risk_mode, scenario_id)
    execution_rows: list[dict[str, str]] = []
    post_refresh_rows: list[dict[str, str]] = []
    if args.execute_safe:
        unsafe_selected = actions[actions["risk_level"].ne("safe")] if not actions.empty else pd.DataFrame()
        if not unsafe_selected.empty:
            print("Only safe actions will be executed; higher-risk actions remain planned.", file=sys.stderr)
        execution_rows = execute_selected(args.project_root, actions)
        execution_failed = any(row.get("status") == "FAIL" for row in execution_rows)
    output = write_outputs(
        args.project_root,
        args.task,
        goal_type,
        risk_mode,
        budget_mode,
        actions,
        deferred,
        execution_rows,
        post_refresh_rows,
    )
    if args.execute_safe and args.post_run_refresh and not any(row.get("status") == "FAIL" for row in execution_rows):
        post_refresh_rows = post_run_refresh(args.project_root)
        output = write_outputs(
            args.project_root,
            args.task,
            goal_type,
            risk_mode,
            budget_mode,
            actions,
            deferred,
            execution_rows,
            post_refresh_rows,
            replace_latest_history=True,
        )
        finalize_commands = [
            "python3 src/chrono_ehr/run_study.py --agent-state",
            "python3 src/chrono_ehr/run_study.py --validate-agent-state",
        ]
        for command in finalize_commands:
            completed = subprocess.run(shlex.split(command), cwd=args.project_root, text=True, capture_output=True)
            if completed.returncode != 0:
                print((completed.stdout + completed.stderr).strip(), file=sys.stderr)
                raise SystemExit(completed.returncode)
    print(f"Wrote {output}")
    print(f"Goal type: {goal_type}")
    print(f"Risk mode: {risk_mode}")
    print(f"Budget mode: {budget_mode}")
    print(f"Selected actions: {len(actions)}")
    print(f"Deferred actions: {len(deferred)}")
    print(f"Post-run refresh steps: {len(post_refresh_rows)}")
    if any(row.get("status") == "FAIL" for row in execution_rows + post_refresh_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
