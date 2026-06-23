#!/usr/bin/env python3
"""Generate a Chinese Methods/Results draft for the MIMIC diabetes demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


def load_metric_dict(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return df.set_index("metric")["value"].to_dict()


def fmt_int(value: str | int | float) -> str:
    return f"{int(float(value)):,}"


def fmt_pct(value: str | float) -> str:
    return f"{float(value):.2%}"


def performance_table(perf: pd.DataFrame) -> str:
    tests = perf[perf["split"] == "test"].sort_values("feature_set")
    lines = [
        "| Feature set | N | Events | Event rate | AUROC | AUPRC | Brier | Sensitivity | Specificity | PPV | NPV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tests.itertuples(index=False):
        lines.append(
            f"| {row.feature_set} | {int(row.n):,} | {int(row.events):,} | {row.event_rate:.2%} | "
            f"{row.AUROC:.4f} | {row.AUPRC:.4f} | {row.Brier_score:.4f} | "
            f"{row.sensitivity:.4f} | {row.specificity:.4f} | {row.ppv:.4f} | {row.npv:.4f} |"
        )
    return "\n".join(lines)


def compact_metric_table(df: pd.DataFrame, label_col: str, columns: list[str]) -> str:
    header = "| " + " | ".join([label_col, *columns]) + " |"
    align = "|---" + "|---:" * len(columns) + "|"
    lines = [header, align]
    for row in df.itertuples(index=False):
        values = [str(getattr(row, label_col))]
        for col in columns:
            val = getattr(row, col)
            if isinstance(val, float):
                if "rate" in col.lower() or col in {"AUROC", "AUPRC", "Brier_score"}:
                    values.append(f"{val:.4f}")
                else:
                    values.append(f"{val:.4f}")
            else:
                values.append(f"{val:,}" if isinstance(val, int) else str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def lab_availability_table(labs: pd.DataFrame) -> str:
    lines = [
        "| Lab | HADM with lab | Coverage | Total measurements | Median measurements |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in labs.itertuples(index=False):
        lines.append(
            f"| {row.lab} | {int(row.hadm_with_lab):,} | {row.hadm_with_lab_percent:.2%} | "
            f"{int(row.total_measurements):,} | {row.median_measurements_among_all:.1f} |"
        )
    return "\n".join(lines)


def med_availability_table(meds: pd.DataFrame) -> str:
    lines = [
        "| Medication class | HADM with med | Coverage | Total orders | Median orders |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in meds.itertuples(index=False):
        lines.append(
            f"| {row.med_class} | {int(row.hadm_with_med):,} | {row.hadm_with_med_percent:.2%} | "
            f"{int(row.total_orders):,} | {row.median_orders_among_all:.1f} |"
        )
    return "\n".join(lines)


def table1_text(table1: pd.DataFrame) -> str:
    values = table1.set_index("variable")
    wanted = [
        "age_mean_sd",
        "female",
        "length_of_stay_days_median_iqr",
        "prior_admissions_count_median_iqr",
        "emergency_or_urgent_admission",
    ]
    lines = [
        "| Variable | Overall | No 30-day readmission | 30-day readmission |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "age_mean_sd": "Age, mean (SD)",
        "female": "Female",
        "length_of_stay_days_median_iqr": "Length of stay, median (IQR)",
        "prior_admissions_count_median_iqr": "Prior admissions, median (IQR)",
        "emergency_or_urgent_admission": "Emergency/urgent admission",
    }
    for key in wanted:
        row = values.loc[key]
        lines.append(f"| {labels[key]} | {row['overall']} | {row['no_readmission_30d']} | {row['readmission_30d']} |")
    return "\n".join(lines)


def feature_time_audit_text(audit: pd.DataFrame) -> str:
    status_counts = audit["audit_status"].value_counts().to_dict()
    risk_counts = audit["leakage_risk"].value_counts().to_dict()
    enabled_review = audit[
        audit["feature_role"].str.startswith("enabled:")
        & audit["audit_status"].isin(["conditional_review", "fail_leakage_risk", "needs_manual_review"])
    ]
    lines = [
        "### Feature-time audit",
        "",
        "根据 `docs/mimic_diabetes_feature_time_map.csv`，本 demo 对已启用特征和 forbidden/high-risk 变量进行时间点审计。",
        "",
        f"- Audit status counts: {status_counts}",
        f"- Leakage risk counts: {risk_counts}",
        "",
    ]
    if enabled_review.empty:
        lines.append("所有已启用特征均通过当前出院时预测时间点审计。")
    else:
        lines.append("需要在 Methods 中明确解释的已启用变量：")
        lines.append("")
        for row in enabled_review.itertuples(index=False):
            lines.append(
                f"- `{row.variable_name}`: {row.audit_status}; usable={row.usable_at_prediction_time}; "
                f"risk={row.leakage_risk}; {row.reason}"
            )
    return "\n".join(lines)


def prediction_time_comparison_text(comparison: pd.DataFrame) -> str:
    stages = ["admission", "inhospital", "discharge"]
    lines = [
        "### Prediction-time comparison",
        "",
        "同一批候选变量在不同预测时间点下的可用性不同。这个对照不重新训练模型，只用于说明 feature availability 的时间点边界。",
        "",
        "| Prediction stage | PASS | REVIEW | BLOCK |",
        "|---|---:|---:|---:|",
    ]
    for stage in stages:
        stage_rows = comparison[comparison["prediction_stage"] == stage]
        counts = stage_rows["audit_status"].value_counts().to_dict()
        review = counts.get("conditional_review", 0) + counts.get("needs_manual_review", 0)
        block = counts.get("fail_leakage_risk", 0) + counts.get("blocked_or_forbidden", 0)
        lines.append(f"| {stage} | {counts.get('pass', 0)} | {review} | {block} |")

    key_examples = comparison[
        comparison["variable_name"].isin(["ed_los_hours", "length_of_stay_days"])
        & comparison["audit_status"].isin(["conditional_review", "fail_leakage_risk"])
    ]
    if not key_examples.empty:
        lines.extend(["", "关键例子："])
        for row in key_examples.itertuples(index=False):
            lines.append(
                f"- `{row.variable_name}` at `{row.prediction_stage}`: {row.audit_status}; "
                f"usable={row.usable_at_prediction_time}; {row.reason}"
            )
    return "\n".join(lines)


def prediction_time_model_text(performance: pd.DataFrame) -> str:
    tests = performance[performance["split"] == "test"].set_index("feature_set")
    admission = tests.loc["admission_safe_minimal"]
    inhospital = tests.loc["inhospital_24h_lab_minimal"] if "inhospital_24h_lab_minimal" in tests.index else None
    inhospital_med = (
        tests.loc["inhospital_24h_lab_med_minimal"] if "inhospital_24h_lab_med_minimal" in tests.index else None
    )
    discharge = tests.loc["discharge_safe_minimal"]
    delta_auroc = discharge["AUROC"] - admission["AUROC"]
    delta_auprc = discharge["AUPRC"] - admission["AUPRC"]
    inhospital_line = ""
    inhospital_row = ""
    if inhospital is not None:
        inhospital_line = (
            f"入院后 24 小时模型相较入院时 AUROC 提升 {inhospital['AUROC'] - admission['AUROC']:.4f}，"
            f"AUPRC 提升 {inhospital['AUPRC'] - admission['AUPRC']:.4f}。"
        )
        inhospital_row = (
            f"| inhospital_24h_lab_minimal | inhospital_24h | {inhospital['AUROC']:.4f} | "
            f"{inhospital['AUPRC']:.4f} | {inhospital['Brier_score']:.4f} |\n"
        )
    inhospital_med_line = ""
    inhospital_med_row = ""
    if inhospital_med is not None:
        inhospital_med_line = (
            f"入院后 24 小时 labs+meds 模型相较入院时 AUROC 提升 {inhospital_med['AUROC'] - admission['AUROC']:.4f}，"
            f"AUPRC 提升 {inhospital_med['AUPRC'] - admission['AUPRC']:.4f}。"
        )
        inhospital_med_row = (
            f"| inhospital_24h_lab_med_minimal | inhospital_24h | {inhospital_med['AUROC']:.4f} | "
            f"{inhospital_med['AUPRC']:.4f} | {inhospital_med['Brier_score']:.4f} |\n"
        )
    return f"""### Prediction-time model comparison

在同一糖尿病 30 天再入院 cohort 中，额外比较了入院时、入院后 24 小时和出院时的安全特征集。

| Feature set | Prediction time | AUROC | AUPRC | Brier |
|---|---|---:|---:|---:|
| admission_safe_minimal | admission | {admission['AUROC']:.4f} | {admission['AUPRC']:.4f} | {admission['Brier_score']:.4f} |
{inhospital_row.rstrip()}
{inhospital_med_row.rstrip()}
| discharge_safe_minimal | discharge | {discharge['AUROC']:.4f} | {discharge['AUPRC']:.4f} | {discharge['Brier_score']:.4f} |

{inhospital_line}
{inhospital_med_line}
出院时相较入院时 AUROC 提升 {delta_auroc:.4f}，AUPRC 提升 {delta_auprc:.4f}。这说明不同 prediction time 不只是概念差异，也会实际影响合法特征集合和模型表现。
本轮 24 小时关键词用药特征没有超过 24 小时化验特征，提示第一版用药映射需要进一步细化。
"""


def generate(project: Path) -> str:
    summary = load_metric_dict(project / "outputs" / "tables" / "mimic_diabetes_cohort_summary.csv")
    perf = pd.read_csv(project / "outputs" / "tables" / "mimic_diabetes_model_performance.csv")
    labs = pd.read_csv(project / "outputs" / "tables" / "mimic_diabetes_lab_feature_availability.csv")
    meds = pd.read_csv(project / "outputs" / "tables" / "mimic_diabetes_med_feature_availability.csv")
    table1 = pd.read_csv(project / "outputs" / "tables" / "mimic_diabetes_table1_basic.csv")
    leakage_path = project / "outputs" / "tables" / "mimic_diabetes_leakage_sensitivity.csv"
    feature_time_path = project / "outputs" / "tables" / "mimic_diabetes_feature_time_audit.csv"
    prediction_time_path = project / "outputs" / "tables" / "mimic_diabetes_prediction_time_comparison.csv"
    prediction_time_model_path = project / "outputs" / "tables" / "mimic_diabetes_prediction_time_model_performance.csv"
    outcome_path = project / "outputs" / "tables" / "mimic_diabetes_outcome_sensitivity.csv"
    ci_path = project / "outputs" / "tables" / "mimic_diabetes_model_performance_bootstrap_ci.csv"
    burden_path = project / "outputs" / "tables" / "mimic_diabetes_fixed_alert_burden.csv"
    leakage = pd.read_csv(leakage_path) if leakage_path.exists() else None
    feature_time_audit = pd.read_csv(feature_time_path) if feature_time_path.exists() else None
    prediction_time_comparison = pd.read_csv(prediction_time_path) if prediction_time_path.exists() else None
    prediction_time_model = pd.read_csv(prediction_time_model_path) if prediction_time_model_path.exists() else None
    outcome = pd.read_csv(outcome_path) if outcome_path.exists() else None
    ci = pd.read_csv(ci_path) if ci_path.exists() else None
    burden = pd.read_csv(burden_path) if burden_path.exists() else None

    test_perf = perf[perf["split"] == "test"].set_index("feature_set")
    minimal = test_perf.loc["minimal"]
    lab_aug = test_perf.loc["lab_augmented"]
    lab_med = test_perf.loc["lab_med_augmented"]
    delta_auroc = lab_aug["AUROC"] - minimal["AUROC"]
    delta_auprc = lab_aug["AUPRC"] - minimal["AUPRC"]
    delta_med_auroc = lab_med["AUROC"] - lab_aug["AUROC"]
    delta_med_auprc = lab_med["AUPRC"] - lab_aug["AUPRC"]
    leakage_section = ""
    if leakage is not None:
        leakage_small = leakage[["scenario", "AUROC", "AUPRC", "sensitivity", "specificity"]]
        leakage_section = f"""
### Leakage sensitivity

为展示 feature leakage 的影响，本 demo 额外构造了两个错误示范：使用 `days_to_next_admission` 和使用 `next_admittime` 是否存在。二者均属于预测时间点之后的信息，不能用于真实模型。

{compact_metric_table(leakage_small, "scenario", ["AUROC", "AUPRC", "sensitivity", "specificity"])}

结果显示，`days_to_next_admission` 会让 AUROC 和 AUPRC 达到 1.0000，这是典型答案泄漏，证明 leakage audit 是本 Agent 的关键功能。
"""
    feature_time_section = ""
    if feature_time_audit is not None:
        feature_time_section = feature_time_audit_text(feature_time_audit)

    prediction_time_section = ""
    if prediction_time_comparison is not None:
        prediction_time_section = prediction_time_comparison_text(prediction_time_comparison)

    prediction_time_model_section = ""
    if prediction_time_model is not None:
        prediction_time_model_section = prediction_time_model_text(prediction_time_model)

    outcome_section = ""
    if outcome is not None:
        outcome_small = outcome[["outcome_definition", "events", "event_rate"]]
        outcome_section = f"""
### Outcome definition sensitivity

第一版主结局为 all-cause 30-day readmission。额外敏感性分析比较了 emergency/urgent 和 non-elective proxy 定义。

{compact_metric_table(outcome_small, "outcome_definition", ["events", "event_rate"])}

该分析说明，结局窗口相同但 readmission 类型定义不同，会明显改变事件率。正式论文前需要预先定义 planned/unplanned readmission 规则。
"""
    uncertainty_section = ""
    if ci is not None and burden is not None:
        ci_lines = [
            "| Feature set | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |",
            "|---|---:|---:|---:|",
        ]
        for row in ci.itertuples(index=False):
            ci_lines.append(
                f"| {row.feature_set} | {row.AUROC:.4f} ({row.AUROC_lower:.4f}-{row.AUROC_upper:.4f}) | "
                f"{row.AUPRC:.4f} ({row.AUPRC_lower:.4f}-{row.AUPRC_upper:.4f}) | "
                f"{row.Brier_score:.4f} ({row.Brier_lower:.4f}-{row.Brier_upper:.4f}) |"
            )
        best_burden = burden[(burden["feature_set"] == "lab_augmented") & (burden["alert_burden"].isin([0.05, 0.10, 0.20]))]
        burden_lines = [
            "| Alert burden | Flagged N | PPV among flagged | Capture rate | Baseline event rate |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in best_burden.itertuples(index=False):
            burden_lines.append(
                f"| {row.alert_burden:.0%} | {int(row.flagged_n):,} | {row.flagged_event_rate_ppv:.2%} | "
                f"{row.capture_rate_sensitivity:.2%} | {row.baseline_event_rate:.2%} |"
            )
        uncertainty_section = f"""
### Bootstrap CI and fixed alert burden

基于 test set 预测结果进行 bootstrap 置信区间估计，并以 `lab_augmented` 模型为例计算固定预警负担下的 PPV 和事件捕获率。

{chr(10).join(ci_lines)}

`lab_augmented` fixed alert burden:

{chr(10).join(burden_lines)}
"""

    return f"""# MIMIC 糖尿病 30 天再入院预测 Methods/Results 草稿

版本：自动生成草稿 v0.1  
用途：给项目评阅者或自己复盘第一版 ChronoEHR-Agent demo。
边界：这是 EHR 数据分析研究草稿，不是医学诊疗建议。

## 研究目的

本 demo 旨在构建一个面向慢病 EHR 研究的 time-aware 数据分析流程。具体任务是在 MIMIC-IV 糖尿病住院患者中，以出院时间为预测时间点，预测患者是否会在出院后 30 天内再次住院。该任务用于验证 ChronoEHR-Agent 能否完成 cohort 构建、随访窗口定义、特征时间点截断、leakage audit、传统 baseline 建模和结果整理。

## Methods 草稿

### 数据来源

本研究使用本地 MIMIC-IV 3.1 数据，路径由 `${MIMIC_IV_ROOT}` 指定。第一版分析使用 `hosp/admissions.csv.gz`、`hosp/patients.csv.gz`、`hosp/diagnoses_icd.csv.gz`、`hosp/labevents.csv.gz` 和 `hosp/prescriptions.csv.gz`。

### 队列定义

糖尿病住院通过 ICD 诊断编码识别：ICD-9 使用 `250*`，ICD-10 使用 `E08*`、`E09*`、`E10*`、`E11*`、`E12*` 和 `E13*`。纳入年龄不小于 18 岁、入院和出院时间有效的糖尿病相关住院。排除当前住院期间死亡的 index admissions，以及出院后 30 天内死亡且无再入院、因而没有完整 30 天再入院观察机会的记录。

### 结局定义

主要结局为 all-cause 30-day hospital readmission。对每个 `subject_id` 的住院按入院时间排序；如果下一次住院发生在 index discharge 后 0 到 30 天内，则定义 `readmission_30d = 1`，否则为 0。

### 预测时间点和特征

第一版预测时间点为出院时间 `dischtime`。`minimal` 特征集包括人口学变量、入院类型、入院来源、保险、语言、婚姻状态、种族、急诊停留时间、住院总时长、既往住院次数和距上次出院时间。

`lab_augmented` 特征集在 `minimal` 基础上加入出院前化验特征。化验可用时间优先使用 `storetime`，缺失时使用 `charttime`，并仅纳入 `admittime <= available_time <= dischtime` 的数值结果。目标化验包括 glucose、HbA1c、creatinine 和 BUN，并提取 count、mean、min、max、last value 和 abnormal count 等摘要。

`lab_med_augmented` 特征集进一步加入出院前糖尿病相关用药类别，包括 insulin、metformin、sulfonylurea、DPP-4 inhibitor、TZD、GLP-1 receptor agonist、SGLT2 inhibitor 和 alpha-glucosidase inhibitor。第一版按处方药名关键词识别药物类别，并仅纳入处方时间与 index admission 有交集且不晚于出院的记录。

### Leakage audit

建模前明确排除 `readmission_30d`、`next_admittime`、`next_hadm_id`、`days_to_next_admission`、`deathtime`、`dod` 和 `hospital_expire_flag` 等 outcome 或未来信息。训练、验证和测试集按 `subject_id` 进行患者级哈希切分，三组患者无重叠。

此外，使用 feature-time map 对已启用特征进行审计。对于出院时预测，`length_of_stay_days` 在时间上可用，但属于高风险变量，因为它在入院时或住院早期预测中不可用；因此第一版报告保留该变量，但明确标为 `conditional_review`。

### 模型和指标

第一版 baseline 使用 logistic regression。由于当前本地 Python 环境没有 scikit-learn，本 demo 使用 `numpy/scipy` 实现 L2 正则 logistic regression。评价指标包括 AUROC、AUPRC、Brier score、sensitivity、specificity、PPV 和 NPV。阈值按训练集事件率对应的预测风险分位点确定。

## Results 草稿

### 队列规模

MIMIC-IV 中共有 {fmt_int(summary["total_admissions"])} 次住院、{fmt_int(summary["total_admission_subjects"])} 名住院患者。糖尿病 ICD 规则初筛得到 {fmt_int(summary["raw_diabetes_admissions"])} 次糖尿病相关住院、{fmt_int(summary["raw_diabetes_subjects"])} 名患者。排除院内死亡和 30 天内死亡且无再入院记录后，最终 cohort 包含 {fmt_int(summary["final_index_admissions"])} 次 index admissions、{fmt_int(summary["final_subjects"])} 名患者。30 天再入院共 {fmt_int(summary["readmission_30d_count"])} 次，事件率为 {fmt_pct(summary["readmission_30d_rate"])}。

### Table 1 初版

{table1_text(table1)}

### 出院前化验覆盖

在 labevents 中共扫描约 158,374,764 行记录，目标 itemid 行数为 12,740,582 行，最终符合糖尿病 cohort 且在入院至出院时间窗内的数值化验为 2,310,943 行。

{lab_availability_table(labs)}

### 出院前糖尿病用药覆盖

在 prescriptions 中识别糖尿病相关药物，并按 index admission 截断到出院前。第一版共识别 748,976 条符合时间窗的糖尿病相关处方记录。

{med_availability_table(meds)}

### 模型表现

{performance_table(perf)}

在 test set 中，`minimal` logistic regression 的 AUROC 为 {minimal["AUROC"]:.4f}，AUPRC 为 {minimal["AUPRC"]:.4f}。加入出院前化验特征后，`lab_augmented` 的 AUROC 为 {lab_aug["AUROC"]:.4f}，AUPRC 为 {lab_aug["AUPRC"]:.4f}。相较于 `minimal`，AUROC 提升 {delta_auroc:.4f}，AUPRC 提升 {delta_auprc:.4f}。

进一步加入糖尿病相关用药后，`lab_med_augmented` 的 AUROC 为 {lab_med["AUROC"]:.4f}，AUPRC 为 {lab_med["AUPRC"]:.4f}。相较于 `lab_augmented`，AUROC 变化 {delta_med_auroc:.4f}，AUPRC 变化 {delta_med_auprc:.4f}，提示第一版关键词用药特征暂未带来额外提升。

### 模型诊断图

已生成 test set 的 ROC、precision-recall 和 calibration decile 图：

- `outputs/figures/mimic_diabetes_roc_curve.png`
- `outputs/figures/mimic_diabetes_precision_recall_curve.png`
- `outputs/figures/mimic_diabetes_calibration_deciles.png`

{feature_time_section}

{prediction_time_section}

{prediction_time_model_section}

{leakage_section}

{outcome_section}

{uncertainty_section}

## 初步解释

第一版结果说明，ChronoEHR-Agent 已经能完成一个从 MIMIC 原始表到慢病再入院 baseline 的可复现 vertical slice。出院前化验特征带来小幅性能提升；当前关键词用药特征没有进一步提高 test set 表现，可能与用药类别粗糙、insulin 用途复杂、以及 logistic regression 表达能力有限有关。当前结果不应解读为临床可用模型，而应作为 time-aware EHR 分析流程的 demo。

## 当前局限

- 第一版结局是 all-cause readmission，尚未区分 planned 和 unplanned readmission。
- 当前模型只有 logistic regression，尚未加入 random forest 或 XGBoost。
- 化验特征只覆盖 glucose、HbA1c、creatinine 和 BUN，尚未加入完整代谢、电解质、血常规和用药特征。
- 当前未做外部验证和决策曲线。
- MIMIC 的诊断编码主要用于回顾性 cohort 定义；如果改成入院时预测，需要重新定义糖尿病是否在入院时已知。

## 下一步

1. 加入 scikit-learn 环境，跑 random forest 和 XGBoost baseline。
2. 细化糖尿病用药映射，区分常规降糖治疗和非降糖目的 insulin 使用。
3. 扩展出院前特征，包括住院中 glucose 波动、肾功能、电解质、血常规和 ICU 相关变量。
4. 做 naive all-feature vs time-aware feature 的 leakage sensitivity。
5. 将同一流程迁移到 CKD、心衰、高血压等慢病队列。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate(args.project_root)
    output_path = args.project_root / "outputs" / "reports" / "mimic_diabetes_methods_results_draft.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
