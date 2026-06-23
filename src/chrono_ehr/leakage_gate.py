#!/usr/bin/env python3
"""Block prediction-time models when feature timing rules are violated."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

from feature_window_spec_loader import DEFAULT_FEATURE_WINDOW_SPEC, load_feature_window_spec
from mimic_diabetes_baseline import DEFAULT_PROJECT
from prediction_time_spec_loader import DEFAULT_SPEC_PATH, load_prediction_time_config, load_raw_config


CRITICAL = "critical"
WARNING = "warning"
PASS = "pass"

FEATURE_TIME_MAPS = {
    "diabetes": "docs/mimic_diabetes_feature_time_map.csv",
    "ckd": "docs/mimic_ckd_feature_time_map.csv",
    "heart_failure": "docs/mimic_heart_failure_feature_time_map.csv",
    "hypertension": "docs/mimic_hypertension_feature_time_map.csv",
}

USABILITY_COLUMNS = {
    "admission": "usable_for_admission_prediction",
    "inhospital_24h": "usable_for_inhospital_prediction",
    "discharge": "usable_for_discharge_prediction",
}

GENERATED_FEATURE_PREFIXES = (
    "lab24h_",
    "med24h_",
    "ckdlab24h_",
    "hflab24h_",
    "htnlab24h_",
    "vital24h_",
    "proc24h_",
    "genmed24h_",
    "labdischarge_",
    "meddischarge_",
    "ckdlabdischarge_",
    "hflabdischarge_",
    "htnlabdischarge_",
    "vitaldischarge_",
    "procdischarge_",
    "genmeddischarge_",
)

ALWAYS_FORBIDDEN_PATTERNS = [
    r"readmission",
    r"days_to_next",
    r"next_admission",
    r"followup",
    r"outcome",
    r"label",
    r"death_time",
    r"deathtime",
    r"\bdod\b",
]

ADMISSION_FORBIDDEN_PATTERNS = [
    r"ed_los_hours",
    r"length_of_stay",
    r"\blos\b",
    r"dischtime",
    r"lab24h",
    r"med24h",
    r"vital24h",
    r"proc24h",
    r"genmed24h",
    r"labdischarge",
    r"vitaldischarge",
    r"procdischarge",
    r"genmeddischarge",
]

INHOSPITAL_24H_FORBIDDEN_PATTERNS = [
    r"length_of_stay",
    r"\blos\b",
    r"dischtime",
    r"labdischarge",
    r"vitaldischarge",
    r"procdischarge",
    r"genmeddischarge",
]


def normalize_path(path: str | Path) -> str:
    return str(path).strip().lstrip("./")


def build_feature_output_index(window_spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    index = {}
    for source_name, source in window_spec.get("feature_sources", {}).items():
        for output in source.get("window_outputs", []):
            relative_path = output.get("path")
            if not relative_path:
                continue
            index[normalize_path(relative_path)] = {
                "source": source_name,
                "window": str(output.get("window", "")),
                "cohort": str(output.get("cohort", "")),
            }
    return index


def regex_matches(patterns: list[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def is_generated_window_feature(feature: str) -> bool:
    return feature.startswith(GENERATED_FEATURE_PREFIXES)


def forbidden_patterns_for_prediction_time(prediction_time: str) -> list[str]:
    patterns = list(ALWAYS_FORBIDDEN_PATTERNS)
    if prediction_time == "admission":
        patterns.extend(ADMISSION_FORBIDDEN_PATTERNS)
    elif prediction_time == "inhospital_24h":
        patterns.extend(INHOSPITAL_24H_FORBIDDEN_PATTERNS)
    return patterns


def read_feature_time_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["variable_name"]: row for row in csv.DictReader(handle)}


def feature_time_map_path(project_root: Path, study_key: str) -> Path | None:
    relative = FEATURE_TIME_MAPS.get(study_key)
    if relative is None:
        return None
    return project_root / relative


def audit_spec_feature_windows(
    study_key: str,
    spec: dict[str, Any],
    window_spec: dict[str, Any],
    output_index: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    prediction_time = str(spec.get("prediction_time", ""))
    feature_set = str(spec.get("feature_set", ""))
    windows = window_spec.get("windows", {})

    for relative_path in spec.get("extra_feature_files", []):
        normalized = normalize_path(relative_path)
        output = output_index.get(normalized)
        if output is None:
            rows.append(
                {
                    "study": study_key,
                    "feature_set": feature_set,
                    "prediction_time": prediction_time,
                    "check_type": "feature_file_window",
                    "severity": WARNING,
                    "status": "unknown_feature_file",
                    "object": normalized,
                    "reason": "Feature file is not listed in configs/feature_window_specs.json, so timing cannot be audited from the shared registry.",
                }
            )
            continue
        window_name = output["window"]
        window = windows.get(window_name, {})
        forbidden = set(window.get("forbidden_for_prediction_times", []))
        usable = set(window.get("usable_for_prediction_times", []))
        if prediction_time in forbidden or (usable and prediction_time not in usable):
            rows.append(
                {
                    "study": study_key,
                    "feature_set": feature_set,
                    "prediction_time": prediction_time,
                    "check_type": "feature_file_window",
                    "severity": CRITICAL,
                    "status": "blocked",
                    "object": normalized,
                    "reason": f"Feature file belongs to `{window_name}` window and is not legal for `{prediction_time}` prediction.",
                }
            )
        else:
            rows.append(
                {
                    "study": study_key,
                    "feature_set": feature_set,
                    "prediction_time": prediction_time,
                    "check_type": "feature_file_window",
                    "severity": PASS,
                    "status": "allowed",
                    "object": normalized,
                    "reason": f"Feature file window `{window_name}` is legal for `{prediction_time}` prediction.",
                }
            )
    return rows


def audit_spec_feature_names(study_key: str, spec: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    prediction_time = str(spec.get("prediction_time", ""))
    feature_set = str(spec.get("feature_set", ""))
    patterns = forbidden_patterns_for_prediction_time(prediction_time)
    feature_names = [*spec.get("numeric_features", []), *spec.get("categorical_features", [])]

    for feature in feature_names:
        matches = regex_matches(patterns, str(feature))
        if not matches:
            continue
        rows.append(
            {
                "study": study_key,
                "feature_set": feature_set,
                "prediction_time": prediction_time,
                "check_type": "feature_name",
                "severity": CRITICAL,
                "status": "blocked",
                "object": str(feature),
                "reason": f"Feature name matches forbidden timing/leakage pattern(s): {', '.join(matches)}.",
            }
        )
    return rows


def audit_spec_variable_time_map(project_root: Path, study_key: str, spec: dict[str, Any]) -> list[dict[str, str]]:
    prediction_time = str(spec.get("prediction_time", ""))
    feature_set = str(spec.get("feature_set", ""))
    usability_column = USABILITY_COLUMNS.get(prediction_time)
    path = feature_time_map_path(project_root, study_key)

    if usability_column is None:
        return [
            {
                "study": study_key,
                "feature_set": feature_set,
                "prediction_time": prediction_time,
                "check_type": "variable_time_map",
                "severity": WARNING,
                "status": "unknown_prediction_time",
                "object": prediction_time,
                "reason": "当前 prediction time 没有映射到 feature_time_map 的可用性列，需要补充 gate 规则。",
            }
        ]

    if path is None or not path.exists():
        return [
            {
                "study": study_key,
                "feature_set": feature_set,
                "prediction_time": prediction_time,
                "check_type": "variable_time_map",
                "severity": WARNING,
                "status": "missing_feature_time_map",
                "object": str(path or study_key),
                "reason": "当前队列没有配置 feature_time_map CSV，因此不能做变量级时间点审计。",
            }
        ]

    feature_map = read_feature_time_map(path)
    rows = []
    feature_names = [*spec.get("numeric_features", []), *spec.get("categorical_features", [])]
    for feature in feature_names:
        feature = str(feature)
        if is_generated_window_feature(feature):
            continue
        row = feature_map.get(feature)
        if row is None:
            rows.append(
                {
                    "study": study_key,
                    "feature_set": feature_set,
                    "prediction_time": prediction_time,
                    "check_type": "variable_time_map",
                    "severity": WARNING,
                    "status": "needs_manual_review",
                    "object": feature,
                    "reason": f"该显式变量没有出现在 `{path.relative_to(project_root)}` 中；需要先补充时间点标注，再信任这个模型。",
                }
            )
            continue

        usable = row.get(usability_column, "").strip().lower()
        risk = row.get("leakage_risk", "").strip().lower()
        if usable == "yes" and risk != "critical":
            severity = PASS
            status = "allowed"
            reason = f"feature_time_map 标注该变量可用于 `{prediction_time}`；leakage_risk={risk or 'unknown'}。"
        elif usable == "conditional" or risk == "medium":
            severity = WARNING
            status = "conditional_review"
            reason = "该变量在当前 prediction time 属于 conditional；需要在 Methods 中说明具体取值窗口，并确认预测时已经可见。"
        else:
            severity = CRITICAL
            status = "blocked"
            reason = f"feature_time_map 标注该变量不能用于 `{prediction_time}`；usable={usable or 'unknown'}，leakage_risk={risk or 'unknown'}。"

        rows.append(
            {
                "study": study_key,
                "feature_set": feature_set,
                "prediction_time": prediction_time,
                "check_type": "variable_time_map",
                "severity": severity,
                "status": status,
                "object": feature,
                "reason": reason,
            }
        )
    return rows


def audit_patient_split(project_root: Path, study_key: str, cohort_path: str) -> list[dict[str, str]]:
    path = project_root / cohort_path
    if not path.exists():
        return [
            {
                "study": study_key,
                "feature_set": "cohort",
                "prediction_time": "all",
                "check_type": "patient_split",
                "severity": CRITICAL,
                "status": "blocked",
                "object": cohort_path,
                "reason": "Cohort file is missing, so patient-level split cannot be audited.",
            }
        ]
    df = pd.read_csv(path, usecols=["subject_id", "split"], low_memory=False).dropna()
    split_subjects = {split: set(group["subject_id"].astype(str)) for split, group in df.groupby("split")}
    rows = []
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        overlap = split_subjects.get(left, set()) & split_subjects.get(right, set())
        rows.append(
            {
                "study": study_key,
                "feature_set": "cohort",
                "prediction_time": "all",
                "check_type": "patient_split",
                "severity": CRITICAL if overlap else PASS,
                "status": "blocked" if overlap else "allowed",
                "object": f"{left}_vs_{right}",
                "reason": f"{len(overlap)} overlapping patients between {left} and {right}.",
            }
        )
    return rows


def audit_study(project_root: Path, study_key: str, spec_path: Path, window_spec: dict[str, Any]) -> list[dict[str, str]]:
    config = load_prediction_time_config(study_key, spec_path)
    output_index = build_feature_output_index(window_spec)
    rows = audit_patient_split(project_root, study_key, config["cohort_path"])
    for spec in config["specs"]:
        rows.extend(audit_spec_feature_windows(study_key, spec, window_spec, output_index))
        rows.extend(audit_spec_feature_names(study_key, spec))
        rows.extend(audit_spec_variable_time_map(project_root, study_key, spec))
    return rows


def audit_all(project_root: Path, spec_path: Path, window_spec_path: Path) -> pd.DataFrame:
    raw = load_raw_config(spec_path)
    window_spec = load_feature_window_spec(window_spec_path)
    rows = []
    for study_key in raw.get("studies", {}):
        rows.extend(audit_study(project_root, study_key, spec_path, window_spec))
    return pd.DataFrame(rows)


def critical_issues(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    return rows[rows["severity"].eq(CRITICAL)].copy()


def enforce_spec_gate(project_root: Path, study_key: str, spec: dict[str, Any], window_spec_path: Path = DEFAULT_FEATURE_WINDOW_SPEC) -> None:
    window_spec = load_feature_window_spec(window_spec_path)
    output_index = build_feature_output_index(window_spec)
    rows = [
        *audit_spec_feature_windows(study_key, spec, window_spec, output_index),
        *audit_spec_feature_names(study_key, spec),
        *audit_spec_variable_time_map(project_root, study_key, spec),
    ]
    issues = [row for row in rows if row["severity"] == CRITICAL]
    if issues:
        details = "; ".join(f"{row['object']}: {row['reason']}" for row in issues[:5])
        raise ValueError(f"Leakage gate blocked `{study_key}.{spec.get('feature_set')}`. {details}")


def markdown_table(rows: pd.DataFrame) -> str:
    columns = ["study", "feature_set", "prediction_time", "check_type", "severity", "status", "object", "reason"]
    if rows.empty:
        return "No checks were generated."
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def warning_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "severity" not in rows.columns:
        return pd.DataFrame()
    warnings = rows[rows["severity"].eq(WARNING)].copy()
    if warnings.empty:
        return warnings
    grouped = (
        warnings.groupby(["study", "prediction_time", "check_type", "status", "object", "reason"], dropna=False)
        .agg(
            n_feature_sets=("feature_set", "nunique"),
            feature_sets=("feature_set", lambda values: ", ".join(sorted(set(map(str, values)))[:6])),
        )
        .reset_index()
    )
    return grouped[
        ["study", "prediction_time", "check_type", "status", "object", "n_feature_sets", "feature_sets", "reason"]
    ]


def markdown_warning_summary(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No leakage-gate warnings found."
    columns = ["study", "prediction_time", "check_type", "status", "object", "n_feature_sets", "feature_sets", "reason"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def markdown_generic_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No action items generated."
    columns = list(rows.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def beginner_interpretation(rows: pd.DataFrame) -> str:
    issue_count = len(critical_issues(rows))
    warning_count = int(rows["severity"].eq(WARNING).sum()) if not rows.empty else 0
    if issue_count:
        return (
            "本次结果是 `BLOCKED`，意思是至少有一个特征或特征文件违反了预测时间点规则。"
            "在修复前不建议继续使用这些模型结果写论文，因为模型可能已经看到了未来信息。"
        )
    if warning_count:
        return (
            "本次结果是 `PASS with warnings`。这表示当前没有发现必须阻断建模的严重泄漏，"
            "但有些变量需要人工解释清楚。warning 不是说模型一定错了，而是提醒 Methods 中必须写明这个变量在预测时是否已经可见。"
        )
    return (
        "本次结果是 `PASS`。这表示当前规则没有发现 critical leakage。"
        "不过这不是绝对证明模型没有任何偏倚；它只说明已定义的时间窗、危险特征名、患者级切分和 feature_time_map 检查通过。"
    )


def action_items(rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["priority", "finding", "affected_objects", "plain_language_reason", "recommended_action"]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    items: list[dict[str, str]] = []
    blocked = critical_issues(rows)
    if not blocked.empty:
        grouped = (
            blocked.groupby(["check_type", "status"], dropna=False)
            .agg(
                n=("object", "count"),
                affected_objects=("object", lambda values: ", ".join(sorted(set(map(str, values)))[:8])),
            )
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            items.append(
                {
                    "priority": "P0",
                    "finding": f"{row.check_type}: {row.status}",
                    "affected_objects": row.affected_objects,
                    "plain_language_reason": "这些对象违反了预测时间点或泄漏规则，可能让模型看到预测时还不知道的信息。",
                    "recommended_action": "先删除或重建这些特征；修复后重新运行 leakage gate，再使用模型结果。",
                }
            )

    warnings = warning_summary(rows)
    for row in warnings.itertuples(index=False):
        object_name = str(row.object)
        if object_name == "ed_los_hours" and str(row.prediction_time) == "inhospital_24h":
            reason = (
                "`ed_los_hours` 是急诊停留时长。入院时通常不能完整知道；在 24h 预测中通常已经可见，"
                "但前提是研究定义明确采用入院后 24 小时这个预测点，并且 ED 时间记录已完整。"
            )
            action = (
                "在 Methods 中写明 `ed_los_hours` 的取值窗口；同时建议做一个去掉 `ed_los_hours` 的敏感性分析，"
                "证明核心结论不依赖这个边界变量。"
            )
        else:
            reason = "该对象不是必须删除，但它的可用时间点不够绝对，需要人工确认。"
            action = "补充 feature_time_map 标注或 Methods 说明；如果不能解释清楚，就从对应 prediction time 的特征集中移除。"
        items.append(
            {
                "priority": "P1",
                "finding": f"{row.study} / {row.prediction_time} / {row.check_type} / {row.status}",
                "affected_objects": f"{object_name}; feature_sets={row.feature_sets}",
                "plain_language_reason": reason,
                "recommended_action": action,
            }
        )

    if not items:
        items.append(
            {
                "priority": "P2",
                "finding": "No warning or critical issue",
                "affected_objects": "None",
                "plain_language_reason": "当前规则没有发现需要处理的问题。",
                "recommended_action": "保持 feature_time_map 和 feature_window_specs 随新特征同步更新。",
            }
        )
    return pd.DataFrame(items, columns=columns)


def write_report(rows: pd.DataFrame, output: Path) -> None:
    issue_count = len(critical_issues(rows))
    warning_count = int(rows["severity"].eq(WARNING).sum()) if not rows.empty else 0
    status = "PASS" if issue_count == 0 else "BLOCKED"
    blocked = critical_issues(rows)
    blocked_md = markdown_table(blocked) if not blocked.empty else "No critical leakage-gate issues found."
    warnings = warning_summary(rows)
    actions = action_items(rows)
    text = f"""# Prediction-Time Leakage Gate Report

这个报告是 ChronoEHR-Agent 的建模前安全闸门。它检查四类常见风险：

1. 每个 feature file 的时间窗是否允许用于对应 prediction time。
2. 特征名是否包含明显的 outcome、follow-up 或未来信息 proxy。
3. 患者级 train/validation/test split 是否有同一患者跨集合重叠。
4. 显式列入模型的基础变量是否通过对应队列的 feature_time_map 审计。

- Status: `{status}`
- Total checks: {len(rows)}
- Critical issues: {issue_count}
- Warnings: {warning_count}

## Beginner Interpretation

{beginner_interpretation(rows)}

请注意：这个 gate 只审计 EHR 数据分析流程中的时间点错误和特征泄漏风险；它不是医学诊断、治疗或用药建议。

## Critical Issues

{blocked_md}

## Warning Summary

{markdown_warning_summary(warnings)}

## Action Items

{markdown_generic_table(actions)}

## All Checks

{markdown_table(rows)}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--window-spec", type=Path, default=DEFAULT_FEATURE_WINDOW_SPEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit_all(args.project_root, args.spec, args.window_spec)
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    rows.to_csv(tables / "prediction_time_leakage_gate.csv", index=False)
    action_items(rows).to_csv(tables / "prediction_time_leakage_gate_action_items.csv", index=False)
    write_report(rows, reports / "prediction_time_leakage_gate_report.md")
    issues = critical_issues(rows)
    print(f"Prediction-time leakage gate: critical={len(issues)} warnings={int(rows['severity'].eq(WARNING).sum()) if not rows.empty else 0}")
    if not issues.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
