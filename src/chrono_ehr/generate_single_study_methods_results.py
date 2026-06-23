#!/usr/bin/env python3
"""Generate Chinese single-study Methods/Results drafts for MIMIC chronic cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


STUDIES = {
    "heart_failure": {
        "prefix": "mimic_heart_failure",
        "title": "心衰",
        "english": "heart failure",
        "icd": "ICD-9 `428*` 或 ICD-10 `I50*`",
        "extra_labs": "albumin、NT-proBNP 和 troponin",
        "cohort_note": "允许当前住院出现心衰诊断，或本次住院前已有心衰诊断记录。",
    },
    "hypertension": {
        "prefix": "mimic_hypertension",
        "title": "高血压",
        "english": "hypertension",
        "icd": "ICD-9 `401*`-`405*` 或 ICD-10 `I10*`-`I15*`",
        "extra_labs": "glucose 和 albumin",
        "cohort_note": "允许当前住院出现高血压诊断，或本次住院前已有高血压诊断记录。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--study", choices=[*STUDIES, "all"], default="all")
    return parser.parse_args()


def metric_dict(path: Path) -> dict[str, str]:
    return pd.read_csv(path).set_index("metric")["value"].astype(str).to_dict()


def fmt_int(value: str | int | float) -> str:
    return f"{int(float(value)):,}"


def fmt_pct(value: str | float) -> str:
    return f"{float(value):.2%}"


def lab_table(df: pd.DataFrame, window: str) -> str:
    subset = df[df["window"].eq(window)]
    lines = [
        "| Lab | HADM with lab | Coverage | Total measurements | Median measurements |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in subset.itertuples(index=False):
        lines.append(
            f"| {row.lab} | {int(row.hadm_with_lab):,} | {row.hadm_with_lab_percent:.2%} | "
            f"{int(row.total_measurements):,} | {row.median_measurements_among_all:.1f} |"
        )
    return "\n".join(lines)


def model_table(perf: pd.DataFrame) -> str:
    tests = perf[perf["split"].eq("test")].copy()
    order = ["admission_safe_minimal", "inhospital_24h_lab_minimal", "discharge_lab_minimal"]
    tests["sort_order"] = tests["feature_set"].map({name: i for i, name in enumerate(order)}).fillna(99)
    tests = tests.sort_values(["sort_order", "feature_set"])
    lines = [
        "| Feature set | Prediction time | N | Events | Event rate | AUROC | AUPRC | Brier | Sensitivity | Specificity | PPV | NPV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tests.itertuples(index=False):
        lines.append(
            f"| {row.feature_set} | {row.prediction_time} | {int(row.n):,} | {int(row.events):,} | "
            f"{row.event_rate:.2%} | {row.AUROC:.4f} | {row.AUPRC:.4f} | {row.Brier_score:.4f} | "
            f"{row.sensitivity:.4f} | {row.specificity:.4f} | {row.ppv:.4f} | {row.npv:.4f} |"
        )
    return "\n".join(lines)


def outcome_table(df: pd.DataFrame) -> str:
    lines = ["| Outcome definition | Events | Event rate |", "|---|---:|---:|"]
    for row in df.itertuples(index=False):
        lines.append(f"| {row.outcome_definition} | {int(row.events):,} | {row.event_rate:.2%} |")
    return "\n".join(lines)


def leakage_table(df: pd.DataFrame) -> str:
    lines = ["| Scenario | AUROC | AUPRC | Sensitivity | Specificity |", "|---|---:|---:|---:|---:|"]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.AUROC:.4f} | {row.AUPRC:.4f} | "
            f"{row.sensitivity:.4f} | {row.specificity:.4f} |"
        )
    return "\n".join(lines)


def generate(project: Path, study_key: str) -> str:
    meta = STUDIES[study_key]
    prefix = meta["prefix"]
    summary = metric_dict(project / "outputs" / "tables" / f"{prefix}_cohort_summary.csv")
    labs = pd.read_csv(project / "outputs" / "tables" / f"{prefix}_lab_feature_availability.csv")
    perf = pd.read_csv(project / "outputs" / "tables" / f"{prefix}_prediction_time_model_performance.csv")
    outcome = pd.read_csv(project / "outputs" / "tables" / f"{prefix}_outcome_sensitivity.csv")
    leakage = pd.read_csv(project / "outputs" / "tables" / f"{prefix}_leakage_sensitivity.csv")
    test = perf[perf["split"].eq("test")].set_index("feature_set")
    admission = test.loc["admission_safe_minimal"]
    inhospital = test.loc["inhospital_24h_lab_minimal"]
    discharge = test.loc["discharge_lab_minimal"]
    all_cause = outcome[outcome["outcome_definition"].eq("all_cause_30d_readmission")].iloc[0]
    emergency = outcome[outcome["outcome_definition"].eq("emergency_urgent_30d_readmission")].iloc[0]
    valid = leakage[leakage["scenario"].str.startswith("valid_")].iloc[0]
    leaked = leakage[leakage["scenario"].eq("leaked_days_to_next_admission")].iloc[0]

    return f"""# MIMIC-IV {meta["title"]} 30 天再入院预测 Methods/Results 草稿

版本：自动生成草稿 v0.1  
边界：这是 EHR 数据分析研究草稿，用于 ChronoEHR-Agent 的慢病队列验证；不是医学诊疗建议。

## 研究目的

本 demo 评估{meta["title"]}住院患者 30 天再入院预测中，预测时间点改变对合法特征集合和模型表现的影响。它用于验证 ChronoEHR-Agent 是否能把同一套 time-aware 分析流程迁移到糖尿病、CKD 以外的慢病队列。

## Methods

### Data source

使用本地 MIMIC-IV v3.1 数据，路径为 `{summary["mimic_root"]}`。主要使用 `patients`、`admissions`、`diagnoses_icd`、`d_icd_diagnoses` 和 `labevents` 表。

### Cohort definition

{meta["title"]}通过 {meta["icd"]} 识别。纳入成人住院记录，{meta["cohort_note"]}排除 index admission 住院期间死亡，以及出院后 30 天内死亡但没有再入院的记录。

最终纳入 {fmt_int(summary["final_index_admissions"])} 次 index admissions，来自 {fmt_int(summary["final_subjects"])} 名患者。30 天再入院事件 {fmt_int(summary["readmission_30d_count"])} 次，事件率 {fmt_pct(summary["readmission_30d_rate"])}。

### Prediction times and features

比较三个预测时间点：

- 入院时：只使用入院时或入院前可知道的变量，例如人口学、入院类型、既往住院次数和既往{meta["title"]}诊断记录。
- 入院后 24 小时：在入院时变量基础上，加入 `available_time <= min(admittime + 24h, dischtime)` 的早期化验。
- 出院时：使用出院前可用变量和 `available_time <= dischtime` 的化验摘要。

通用化验包括 creatinine、BUN、potassium、sodium、bicarbonate 和 hemoglobin；本队列额外包括 {meta["extra_labs"]}。每类化验提取 count、mean、min、max、last、abnormal_count 和 has 指示变量。

### Model and metrics

模型为本地 `numpy/scipy` 实现的 L2 logistic regression。患者级 hash split 划分 train、validation 和 test，避免同一患者同时进入训练集和测试集。报告 AUROC、AUPRC、Brier score、sensitivity、specificity、PPV 和 NPV；不只报告 AUROC，因为再入院任务存在类别不平衡。

## Results

### Lab availability

First 24h:

{lab_table(labs, "first_24h")}

Admission to discharge:

{lab_table(labs, "admission_to_discharge")}

### Model performance

{model_table(perf)}

### Prediction-time effect

入院后 24 小时化验模型相较入院时模型，AUROC 提升 {inhospital["AUROC"] - admission["AUROC"]:.4f}，AUPRC 提升 {inhospital["AUPRC"] - admission["AUPRC"]:.4f}。出院时化验模型相较入院时模型，AUROC 提升 {discharge["AUROC"] - admission["AUROC"]:.4f}，AUPRC 提升 {discharge["AUPRC"] - admission["AUPRC"]:.4f}。

### Outcome-definition sensitivity

{outcome_table(outcome)}

把 all-cause 30 天再入院改为 emergency/urgent 代理定义后，事件率从 {all_cause["event_rate"]:.2%} 降至 {emergency["event_rate"]:.2%}。这说明再入院研究必须提前说明结局定义，否则不同研究之间很难比较。

### Leakage sensitivity

{leakage_table(leakage)}

合法出院前模型 AUROC 为 {valid["AUROC"]:.4f}，AUPRC 为 {valid["AUPRC"]:.4f}。错误加入 `days_to_next_admission` 后，AUROC/AUPRC 变为 {leaked["AUROC"]:.4f}/{leaked["AUPRC"]:.4f}，这是典型未来信息泄漏，不能作为真实模型特征。

### Leakage audit interpretation

入院模型不使用住院时长、出院前化验、下次入院时间或 30 天随访窗口内事件。24 小时模型只使用入院后 24 小时内可用的化验。出院模型只使用出院前可用信息，没有使用出院后的随访数据。需要注意的是，诊断编码可以用于回顾性 cohort 定义，但入院时预测特征不能把本次住院最终编码当作入院时已知变量。

## 当前局限

- 这是 baseline，不是最优模型。
- 本队列用于验证 ChronoEHR-Agent 的 time-aware 分析流程，不代表临床可部署模型。
- 暂未单独为本队列撰写完整英文 manuscript section；正式写作时应以跨慢病总稿为主，单队列草稿作为补充。
- eICU/CHARLS 尚未进入本队列外部验证，需等数据 readiness 通过后再设计对应任务。
"""


def main() -> None:
    args = parse_args()
    selected = STUDIES if args.study == "all" else {args.study: STUDIES[args.study]}
    for study_key, meta in selected.items():
        output = args.project_root / "outputs" / "reports" / f"{meta['prefix']}_methods_results_draft.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(generate(args.project_root, study_key), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
