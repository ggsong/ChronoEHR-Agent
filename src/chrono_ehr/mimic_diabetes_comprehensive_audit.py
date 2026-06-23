#!/usr/bin/env python3
"""Generate a consolidated audit report for the diabetes demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


def status_counts(df: pd.DataFrame, column: str) -> str:
    counts = df[column].value_counts(dropna=False).to_dict()
    return ", ".join(f"{key}: {value}" for key, value in counts.items())


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    lines = [
        "| " + " | ".join(columns) + " |",
        "|---" + "|---:" * (len(columns) - 1) + "|",
    ]
    for row in df[columns].itertuples(index=False):
        vals = []
        for col, val in zip(columns, row):
            if col in float_cols and pd.notna(val):
                vals.append(f"{float(val):.4f}")
            elif isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def read_validation_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8")
    if "Status: `PASS`" in text:
        return "PASS"
    if "Status: `FAIL`" in text:
        return "FAIL"
    return "unknown"


def generate(project: Path) -> str:
    tables = project / "outputs" / "tables"
    reports = project / "outputs" / "reports"
    feature_audit = pd.read_csv(tables / "mimic_diabetes_feature_time_audit.csv")
    split_summary = pd.read_csv(tables / "mimic_diabetes_split_summary.csv")
    leakage = pd.read_csv(tables / "mimic_diabetes_leakage_sensitivity.csv")
    prediction_time = pd.read_csv(tables / "mimic_diabetes_prediction_time_model_performance.csv")
    missingness = pd.read_csv(tables / "mimic_diabetes_feature_missingness.csv")
    validation_status = read_validation_status(reports / "study_config_validation_report.md")

    enabled_issues = feature_audit[
        feature_audit["feature_role"].str.startswith("enabled:")
        & feature_audit["audit_status"].isin(["conditional_review", "fail_leakage_risk", "needs_manual_review"])
    ]
    forbidden = feature_audit[feature_audit["feature_role"].eq("forbidden_or_high_risk")]
    test_prediction_time = prediction_time[prediction_time["split"].eq("test")].copy()
    leakage_display = leakage[["scenario", "AUROC", "AUPRC", "sensitivity", "specificity", "note"]].copy()
    top_missing = missingness.sort_values("missing_percent", ascending=False).head(6)

    split_table = split_summary[["split", "admissions", "subjects", "readmission_30d_count", "readmission_30d_rate"]]
    prediction_table = test_prediction_time[
        ["feature_set", "prediction_time", "AUROC", "AUPRC", "Brier_score", "ppv", "npv"]
    ]

    return f"""# ChronoEHR-Agent Comprehensive Audit Report

## Overall Status

- Study: `mimic_iv_3_1_diabetes_readmission`
- Config validation: `{validation_status}`
- Feature-time audit status counts: {status_counts(feature_audit, "audit_status")}
- Leakage risk counts: {status_counts(feature_audit, "leakage_risk")}

## Patient-Level Split Audit

训练、验证和测试集按患者级切分生成。下面的事件率接近，但并不完全相同，这是真实数据切分中常见情况。

{markdown_table(split_table, ["split", "admissions", "subjects", "readmission_30d_count", "readmission_30d_rate"], {"readmission_30d_rate"})}

## Feature-Time Audit

已启用特征中需要说明或复核的变量：

{chr(10).join(f"- `{row.variable_name}`: {row.audit_status}; risk={row.leakage_risk}; {row.reason}" for row in enabled_issues.itertuples(index=False)) or "- None"}

默认禁止或高风险变量：

{chr(10).join(f"- `{row.variable_name}`: risk={row.leakage_risk}; {row.reason}" for row in forbidden.itertuples(index=False))}

## Leakage Sensitivity

{markdown_table(leakage_display, ["scenario", "AUROC", "AUPRC", "sensitivity", "specificity", "note"], {"AUROC", "AUPRC", "sensitivity", "specificity"})}

结论：`days_to_next_admission` 让 AUROC/AUPRC 达到 1.0000，是直接答案泄漏。该变量必须保持在 forbidden/high-risk 列表中。

## Prediction-Time Model Audit

{markdown_table(prediction_table, ["feature_set", "prediction_time", "AUROC", "AUPRC", "Brier_score", "ppv", "npv"], {"AUROC", "AUPRC", "Brier_score", "ppv", "npv"})}

解释：入院后 24 小时特征略优于入院时特征；出院时特征进一步略升。这个结果支持 ChronoEHR-Agent 的核心设定：prediction time 会改变合法特征集合和模型表现。

## Missingness Check

{markdown_table(top_missing, ["variable", "missing_count", "missing_percent"], {"missing_percent"})}

## Audit Verdict

- 当前 demo 可以作为 ChronoEHR-Agent 的第一个完整 vertical slice。
- 合法模型、泄漏模型、不同预测时间点模型已经分开报告。
- 仍需在正式论文前细化 unplanned readmission、用药映射、以及外部/跨队列验证。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.project_root / "outputs" / "reports" / "mimic_diabetes_comprehensive_audit_report.md"
    output.write_text(generate(args.project_root), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
