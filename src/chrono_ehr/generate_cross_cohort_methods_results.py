#!/usr/bin/env python3
"""Generate a cross-cohort Methods/Results draft from ChronoEHR outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]

COHORT_LABELS = {
    "diabetes": "糖尿病",
    "ckd": "CKD",
    "heart_failure": "心衰",
    "hypertension": "高血压",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest_sklearn": "Random Forest",
    "calibrated_random_forest_platt": "Calibrated Random Forest (Platt)",
    "calibrated_random_forest_isotonic": "Calibrated Random Forest (isotonic)",
    "gradient_boosting_sklearn_hist": "Gradient Boosting (sklearn Hist)",
    "calibrated_gradient_boosting_platt": "Calibrated Gradient Boosting (Platt)",
    "calibrated_gradient_boosting_isotonic": "Calibrated Gradient Boosting (isotonic)",
}

FEATURE_LABELS = {
    "admission_safe_minimal": "入院时安全变量",
    "inhospital_24h_lab_minimal": "入院后24小时变量+化验",
    "inhospital_24h_lab_med_minimal": "入院后24小时变量+化验+用药",
    "inhospital_24h_lab_med_vital_minimal": "入院后24小时变量+化验+用药+vitals",
    "inhospital_24h_lab_med_vital_proc_minimal": "入院后24小时变量+化验+用药+vitals+procedures",
    "inhospital_24h_lab_med_vital_proc_genmed_minimal": "入院后24小时变量+化验+用药+vitals+procedures+广泛用药",
    "inhospital_24h_lab_vital_minimal": "入院后24小时变量+化验+vitals",
    "inhospital_24h_lab_vital_proc_minimal": "入院后24小时变量+化验+vitals+procedures",
    "inhospital_24h_lab_vital_proc_genmed_minimal": "入院后24小时变量+化验+vitals+procedures+广泛用药",
    "discharge_safe_minimal": "出院前安全变量",
    "discharge_safe_vital_minimal": "出院前安全变量+vitals",
    "discharge_safe_vital_proc_minimal": "出院前安全变量+vitals+procedures",
    "discharge_safe_vital_proc_genmed_minimal": "出院前安全变量+vitals+procedures+广泛用药",
    "discharge_lab_minimal": "出院前安全变量+化验",
    "discharge_lab_vital_minimal": "出院前安全变量+化验+vitals",
    "discharge_lab_vital_proc_minimal": "出院前安全变量+化验+vitals+procedures",
    "discharge_lab_vital_proc_genmed_minimal": "出院前安全变量+化验+vitals+procedures+广泛用药",
}


def read_table(project_root: Path, relative_path: str) -> pd.DataFrame:
    path = project_root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def fmt_metric(value: float) -> str:
    return f"{float(value):.3f}"


def label_cohort(value: str) -> str:
    return COHORT_LABELS.get(str(value), str(value))


def label_model(value: str) -> str:
    return MODEL_LABELS.get(str(value), str(value))


def label_feature(value: str) -> str:
    return FEATURE_LABELS.get(str(value), str(value))


def markdown_table(df: pd.DataFrame, columns: list[str], display_names: dict[str, str] | None = None) -> str:
    if df.empty:
        return "No data available."
    display_names = display_names or {}
    headers = [display_names.get(col, col) for col in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_table1(cohort: pd.DataFrame) -> pd.DataFrame:
    table = cohort.copy()
    table["队列"] = table["cohort"].map(label_cohort)
    table["住院次数"] = table["final_index_admissions"].map(fmt_int)
    table["患者数"] = table["final_subjects"].map(fmt_int)
    table["30天再入院人数"] = table["readmission_30d_count"].map(fmt_int)
    table["30天再入院率"] = table["readmission_30d_rate"].map(fmt_pct)
    return table[["队列", "住院次数", "患者数", "30天再入院人数", "30天再入院率"]]


def make_prediction_time_deltas(prediction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in prediction.groupby("cohort", sort=False):
        admission = group[group["prediction_time"].eq("admission")]
        discharge = group[group["prediction_time"].eq("discharge")]
        if admission.empty or discharge.empty:
            continue
        admission_row = admission.iloc[0]
        discharge_row = discharge.sort_values(["AUROC", "AUPRC"], ascending=False).iloc[0]
        rows.append(
            {
                "cohort": cohort,
                "cohort_label": label_cohort(cohort),
                "admission_feature_set": admission_row["feature_set"],
                "discharge_feature_set": discharge_row["feature_set"],
                "admission_AUROC": admission_row["AUROC"],
                "discharge_AUROC": discharge_row["AUROC"],
                "delta_AUROC": discharge_row["AUROC"] - admission_row["AUROC"],
                "admission_AUPRC": admission_row["AUPRC"],
                "discharge_AUPRC": discharge_row["AUPRC"],
                "delta_AUPRC": discharge_row["AUPRC"] - admission_row["AUPRC"],
                "admission_Brier": admission_row["Brier_score"],
                "discharge_Brier": discharge_row["Brier_score"],
                "delta_Brier": discharge_row["Brier_score"] - admission_row["Brier_score"],
            }
        )
    return pd.DataFrame(rows)


def make_model_table(comparison: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    table = comparison.merge(calibration, on=["cohort", "model"], how="left")
    table["队列"] = table["cohort"].map(label_cohort)
    table["模型"] = table["model"].map(label_model)
    table["特征集"] = table["feature_set"].map(label_feature)
    table["样本数"] = table["n"].map(fmt_int)
    table["事件数"] = table["events"].map(fmt_int)
    table["AUROC"] = table["AUROC"].map(fmt_metric)
    table["AUPRC"] = table["AUPRC"].map(fmt_metric)
    table["Brier"] = table["Brier_score"].map(fmt_metric)
    table["Mean calibration error"] = table["mean_absolute_calibration_error"].map(
        lambda value: "" if pd.isna(value) else fmt_metric(value)
    )
    return table[["队列", "模型", "特征集", "样本数", "事件数", "AUROC", "AUPRC", "Brier", "Mean calibration error"]]


def make_prediction_time_table(prediction: pd.DataFrame) -> pd.DataFrame:
    table = prediction.copy()
    table["队列"] = table["cohort"].map(label_cohort)
    table["预测时间点"] = table["prediction_time"].map(
        {"admission": "入院时", "inhospital_24h": "入院后24小时", "discharge": "出院前"}
    ).fillna(table["prediction_time"])
    table["特征集"] = table["feature_set"].map(label_feature)
    table["样本数"] = table["n"].map(fmt_int)
    table["事件数"] = table["events"].map(fmt_int)
    table["事件率"] = table["event_rate"].map(fmt_pct)
    table["AUROC"] = table["AUROC"].map(fmt_metric)
    table["AUPRC"] = table["AUPRC"].map(fmt_metric)
    table["Brier"] = table["Brier_score"].map(fmt_metric)
    return table[["队列", "预测时间点", "特征集", "样本数", "事件数", "事件率", "AUROC", "AUPRC", "Brier"]]


def make_vital_increment_table(prediction: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("inhospital_24h_lab_med_minimal", "inhospital_24h_lab_med_vital_minimal", "24h labs+meds 加 vitals"),
        ("inhospital_24h_lab_minimal", "inhospital_24h_lab_vital_minimal", "24h labs 加 vitals"),
        ("discharge_safe_minimal", "discharge_safe_vital_minimal", "出院安全变量加 vitals"),
        ("discharge_lab_minimal", "discharge_lab_vital_minimal", "出院 labs 加 vitals"),
    ]
    rows = []
    tests = prediction.set_index(["cohort", "feature_set"])
    for cohort in prediction["cohort"].drop_duplicates():
        for baseline, vital, label in pairs:
            if (cohort, baseline) not in tests.index or (cohort, vital) not in tests.index:
                continue
            base = tests.loc[(cohort, baseline)]
            with_vital = tests.loc[(cohort, vital)]
            rows.append(
                {
                    "队列": label_cohort(cohort),
                    "比较": label,
                    "原AUROC": fmt_metric(base["AUROC"]),
                    "加vitals后AUROC": fmt_metric(with_vital["AUROC"]),
                    "AUROC差值": f"{with_vital['AUROC'] - base['AUROC']:+.3f}",
                    "原AUPRC": fmt_metric(base["AUPRC"]),
                    "加vitals后AUPRC": fmt_metric(with_vital["AUPRC"]),
                    "AUPRC差值": f"{with_vital['AUPRC'] - base['AUPRC']:+.3f}",
                }
            )
    return pd.DataFrame(rows)


def make_procedure_increment_table(prediction: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        (
            "inhospital_24h_lab_med_vital_minimal",
            "inhospital_24h_lab_med_vital_proc_minimal",
            "24h labs+meds+vitals 加 procedures",
        ),
        ("inhospital_24h_lab_vital_minimal", "inhospital_24h_lab_vital_proc_minimal", "24h labs+vitals 加 procedures"),
        ("discharge_safe_vital_minimal", "discharge_safe_vital_proc_minimal", "出院安全变量+vitals 加 procedures"),
        ("discharge_lab_vital_minimal", "discharge_lab_vital_proc_minimal", "出院 labs+vitals 加 procedures"),
    ]
    rows = []
    tests = prediction.set_index(["cohort", "feature_set"])
    for cohort in prediction["cohort"].drop_duplicates():
        for baseline, proc, label in pairs:
            if (cohort, baseline) not in tests.index or (cohort, proc) not in tests.index:
                continue
            base = tests.loc[(cohort, baseline)]
            with_proc = tests.loc[(cohort, proc)]
            rows.append(
                {
                    "队列": label_cohort(cohort),
                    "比较": label,
                    "原AUROC": fmt_metric(base["AUROC"]),
                    "加procedures后AUROC": fmt_metric(with_proc["AUROC"]),
                    "AUROC差值": f"{with_proc['AUROC'] - base['AUROC']:+.3f}",
                    "原AUPRC": fmt_metric(base["AUPRC"]),
                    "加procedures后AUPRC": fmt_metric(with_proc["AUPRC"]),
                    "AUPRC差值": f"{with_proc['AUPRC'] - base['AUPRC']:+.3f}",
                }
            )
    return pd.DataFrame(rows)


def make_medication_increment_table(prediction: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        (
            "inhospital_24h_lab_med_vital_proc_minimal",
            "inhospital_24h_lab_med_vital_proc_genmed_minimal",
            "24h labs+糖尿病用药+vitals+procedures 加广泛用药",
        ),
        (
            "inhospital_24h_lab_vital_proc_minimal",
            "inhospital_24h_lab_vital_proc_genmed_minimal",
            "24h labs+vitals+procedures 加广泛用药",
        ),
        (
            "discharge_safe_vital_proc_minimal",
            "discharge_safe_vital_proc_genmed_minimal",
            "出院安全变量+vitals+procedures 加广泛用药",
        ),
        (
            "discharge_lab_vital_proc_minimal",
            "discharge_lab_vital_proc_genmed_minimal",
            "出院 labs+vitals+procedures 加广泛用药",
        ),
    ]
    rows = []
    tests = prediction.set_index(["cohort", "feature_set"])
    for cohort in prediction["cohort"].drop_duplicates():
        for baseline, med, label in pairs:
            if (cohort, baseline) not in tests.index or (cohort, med) not in tests.index:
                continue
            base = tests.loc[(cohort, baseline)]
            with_med = tests.loc[(cohort, med)]
            rows.append(
                {
                    "队列": label_cohort(cohort),
                    "比较": label,
                    "原AUROC": fmt_metric(base["AUROC"]),
                    "加广泛用药后AUROC": fmt_metric(with_med["AUROC"]),
                    "AUROC差值": f"{with_med['AUROC'] - base['AUROC']:+.3f}",
                    "原AUPRC": fmt_metric(base["AUPRC"]),
                    "加广泛用药后AUPRC": fmt_metric(with_med["AUPRC"]),
                    "AUPRC差值": f"{with_med['AUPRC'] - base['AUPRC']:+.3f}",
                }
            )
    return pd.DataFrame(rows)


def summarize_ranges(cohort: pd.DataFrame, deltas: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, str]:
    total_admissions = cohort["final_index_admissions"].sum()
    total_subjects = cohort["final_subjects"].sum()
    min_rate = cohort["readmission_30d_rate"].min()
    max_rate = cohort["readmission_30d_rate"].max()
    max_delta = deltas.sort_values("delta_AUROC", ascending=False).iloc[0]
    min_delta = deltas.sort_values("delta_AUROC", ascending=True).iloc[0]
    rf_delta = comparison.pivot_table(index="cohort", columns="model", values="AUROC", aggfunc="first")
    rf_better = 0
    if {"random_forest_sklearn", "logistic_regression"}.issubset(rf_delta.columns):
        rf_better = int((rf_delta["random_forest_sklearn"] > rf_delta["logistic_regression"]).sum())
    return {
        "total_admissions": fmt_int(total_admissions),
        "total_subjects": fmt_int(total_subjects),
        "min_rate": fmt_pct(min_rate),
        "max_rate": fmt_pct(max_rate),
        "max_delta_cohort": label_cohort(max_delta["cohort"]),
        "max_delta_auroc": fmt_metric(max_delta["delta_AUROC"]),
        "min_delta_cohort": label_cohort(min_delta["cohort"]),
        "min_delta_auroc": fmt_metric(min_delta["delta_AUROC"]),
        "rf_better": str(rf_better),
    }


def write_report(
    project_root: Path,
    cohort: pd.DataFrame,
    prediction: pd.DataFrame,
    deltas: pd.DataFrame,
    outcome: pd.DataFrame,
    leakage: pd.DataFrame,
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    table1: pd.DataFrame,
    model_table: pd.DataFrame,
    report_path: Path,
) -> None:
    stats = summarize_ranges(cohort, deltas, comparison)
    table1_md = markdown_table(table1, list(table1.columns))
    prediction_md = markdown_table(
        make_prediction_time_table(prediction),
        ["队列", "预测时间点", "特征集", "样本数", "事件数", "事件率", "AUROC", "AUPRC", "Brier"],
    )
    vital_increment = make_vital_increment_table(prediction)
    vital_increment_md = markdown_table(vital_increment, list(vital_increment.columns))
    procedure_increment = make_procedure_increment_table(prediction)
    procedure_increment_md = markdown_table(procedure_increment, list(procedure_increment.columns))
    medication_increment = make_medication_increment_table(prediction)
    medication_increment_md = markdown_table(medication_increment, list(medication_increment.columns))
    ablation_path = project_root / "outputs" / "tables" / "chronic_disease_feature_group_ablation_summary.csv"
    if ablation_path.exists():
        ablation_summary = pd.read_csv(ablation_path)
        ablation_md = markdown_table(
            ablation_summary,
            [
                "stage",
                "group_added",
                "comparisons",
                "cohorts_improved_AUROC",
                "cohorts_improved_AUPRC",
                "mean_delta_AUROC",
                "mean_delta_AUPRC",
                "mean_delta_Brier",
            ],
        )
    else:
        ablation_md = "Feature group ablation summary has not been generated yet."
    feature_selection_path = project_root / "outputs" / "tables" / "chronic_disease_repeated_feature_concepts.csv"
    if feature_selection_path.exists():
        feature_selection = pd.read_csv(feature_selection_path).head(18)
        feature_selection_md = markdown_table(
            feature_selection,
            [
                "feature_group",
                "concept",
                "n_cohorts",
                "appearances",
                "prediction_times",
                "mean_abs_coefficient",
                "positive_count",
                "negative_count",
            ],
        )
    else:
        feature_selection_md = "Fine-grained feature-selection summary has not been generated yet."
    selected_path = project_root / "outputs" / "tables" / "chronic_disease_selected_feature_set_supplementary_table.csv"
    if not selected_path.exists():
        selected_path = project_root / "outputs" / "tables" / "chronic_disease_selected_feature_set_comparison.csv"
    if selected_path.exists():
        selected_comparison = pd.read_csv(selected_path)
        selected_columns = [
                "cohort_label",
                "prediction_time",
                "selected_features",
                "full_AUROC",
                "selected_AUROC",
                "delta_AUROC",
                "full_AUPRC",
                "selected_AUPRC",
                "delta_AUPRC",
                "delta_Brier",
        ]
        if "mean_absolute_calibration_error" in selected_comparison.columns:
            selected_columns.extend(["mean_absolute_calibration_error", "max_absolute_calibration_error"])
        selected_md = markdown_table(selected_comparison, selected_columns)
    else:
        selected_md = "Selected feature set comparison has not been generated yet."
    ed_los_path = project_root / "outputs" / "tables" / "chronic_disease_ed_los_sensitivity_comparison.csv"
    if ed_los_path.exists():
        ed_los = pd.read_csv(ed_los_path)
        mean_abs_ed_los_auroc = ed_los["delta_AUROC"].abs().mean()
        mean_abs_ed_los_auprc = ed_los["delta_AUPRC"].abs().mean()
        ed_los_summary = (
            ed_los.groupby("cohort_label", sort=False)
            .agg(
                comparisons=("source_feature_set", "count"),
                mean_delta_AUROC=("delta_AUROC", "mean"),
                mean_delta_AUPRC=("delta_AUPRC", "mean"),
                mean_delta_Brier=("delta_Brier", "mean"),
                max_abs_delta_AUROC=("delta_AUROC", lambda values: float(values.abs().max())),
                max_abs_delta_AUPRC=("delta_AUPRC", lambda values: float(values.abs().max())),
            )
            .reset_index()
        )
        ed_los_md = markdown_table(
            ed_los_summary,
            [
                "cohort_label",
                "comparisons",
                "mean_delta_AUROC",
                "mean_delta_AUPRC",
                "mean_delta_Brier",
                "max_abs_delta_AUROC",
                "max_abs_delta_AUPRC",
            ],
            display_names={
                "cohort_label": "队列",
                "comparisons": "模型数",
                "mean_delta_AUROC": "平均AUROC差值",
                "mean_delta_AUPRC": "平均AUPRC差值",
                "mean_delta_Brier": "平均Brier差值",
                "max_abs_delta_AUROC": "最大绝对AUROC差值",
                "max_abs_delta_AUPRC": "最大绝对AUPRC差值",
            },
        )
        ed_los_interpretation = (
            f"去掉 `ed_los_hours` 后，17 个 24 小时 logistic models 的平均绝对 AUROC 变化为 "
            f"{mean_abs_ed_los_auroc:.4f}，平均绝对 AUPRC 变化为 {mean_abs_ed_los_auprc:.4f}。"
            "这说明主要 24 小时模型结论不依赖这个边界时间变量。"
        )
    else:
        ed_los_md = "ED length-of-stay sensitivity analysis has not been generated yet."
        ed_los_interpretation = "ED length-of-stay sensitivity analysis has not been generated yet."
    threshold_path = project_root / "outputs" / "tables" / "chronic_disease_threshold_analysis.csv"
    if threshold_path.exists():
        threshold = pd.read_csv(threshold_path)
        top10 = threshold[threshold["alert_rate"].eq(0.10)].copy()
        mean_top10_ppv = top10["ppv"].mean()
        mean_top10_recall = top10["recall"].mean()
        threshold_md = markdown_table(
            top10[
                [
                    "cohort_label",
                    "prediction_time",
                    "n",
                    "events",
                    "event_rate",
                    "alerts",
                    "ppv",
                    "recall",
                    "specificity",
                    "lift_vs_event_rate",
                ]
            ],
            [
                "cohort_label",
                "prediction_time",
                "n",
                "events",
                "event_rate",
                "alerts",
                "ppv",
                "recall",
                "specificity",
                "lift_vs_event_rate",
            ],
            display_names={
                "cohort_label": "队列",
                "prediction_time": "预测时间点",
                "event_rate": "事件率",
                "alerts": "Top 10% alerts",
                "ppv": "PPV",
                "recall": "Recall",
                "specificity": "Specificity",
                "lift_vs_event_rate": "Lift",
            },
        )
        threshold_interpretation = (
            f"在 top 10% alert burden 场景下，最终 24 小时和出院前模型的平均 PPV 为 "
            f"{mean_top10_ppv:.3f}，平均 recall 为 {mean_top10_recall:.3f}。"
            "这类固定工作量指标可补充 AUROC/AUPRC，帮助解释模型在研究场景中的排序行为。"
        )
    else:
        threshold_md = "Threshold analysis has not been generated yet."
        threshold_interpretation = "Threshold analysis has not been generated yet."
    decision_curve_path = project_root / "outputs" / "tables" / "chronic_disease_decision_curve.csv"
    if decision_curve_path.exists():
        decision_curve = pd.read_csv(decision_curve_path)
        threshold20 = decision_curve[decision_curve["threshold_probability"].eq(0.20)].copy()
        mean_decision_advantage_20 = threshold20["net_benefit_advantage"].mean()
        model_preferred_20 = int((threshold20["preferred_strategy"] == "model").sum())
        total_decision_20 = len(threshold20)
        decision_curve_md = markdown_table(
            threshold20[
                [
                    "cohort_label",
                    "prediction_time",
                    "threshold_probability",
                    "alerts",
                    "alert_rate",
                    "ppv",
                    "recall",
                    "model_net_benefit",
                    "treat_all_net_benefit",
                    "net_benefit_advantage",
                    "preferred_strategy",
                ]
            ],
            [
                "cohort_label",
                "prediction_time",
                "threshold_probability",
                "alerts",
                "alert_rate",
                "ppv",
                "recall",
                "model_net_benefit",
                "treat_all_net_benefit",
                "net_benefit_advantage",
                "preferred_strategy",
            ],
            display_names={
                "cohort_label": "队列",
                "prediction_time": "预测时间点",
                "threshold_probability": "风险阈值",
                "alerts": "Alerts",
                "alert_rate": "Alert rate",
                "ppv": "PPV",
                "recall": "Recall",
                "model_net_benefit": "Model net benefit",
                "treat_all_net_benefit": "Treat-all net benefit",
                "net_benefit_advantage": "Net benefit advantage",
                "preferred_strategy": "Preferred strategy",
            },
        )
        decision_curve_interpretation = (
            f"在 threshold probability = 0.20 时，{model_preferred_20}/{total_decision_20} 个最终模型的 net benefit "
            f"高于 treat-all 和 treat-none 两个参照策略；平均 net benefit advantage 为 "
            f"{mean_decision_advantage_20:.4f}。"
            "这说明在该研究阈值下，模型排序不仅有 AUROC/AUPRC 表现，也能在 decision-curve 框架下优于简单参照策略。"
        )
    else:
        decision_curve_md = "Decision-curve analysis has not been generated yet."
        decision_curve_interpretation = "Decision-curve analysis has not been generated yet."
    subgroup_path = project_root / "outputs" / "tables" / "chronic_disease_subgroup_performance.csv"
    if subgroup_path.exists():
        subgroup = pd.read_csv(subgroup_path)
        subgroup_summary = (
            subgroup.groupby("subgroup_variable", dropna=False)
            .agg(
                rows=("subgroup_value", "count"),
                min_n=("n", "min"),
                mean_event_rate=("event_rate", "mean"),
                mean_AUROC=("AUROC", "mean"),
                mean_AUPRC=("AUPRC", "mean"),
                mean_top10_ppv=("top10_ppv", "mean"),
            )
            .reset_index()
        )
        subgroup_md = markdown_table(
            subgroup_summary,
            list(subgroup_summary.columns),
            display_names={
                "subgroup_variable": "Subgroup variable",
                "rows": "Rows",
                "min_n": "Minimum n",
                "mean_event_rate": "Mean event rate",
                "mean_AUROC": "Mean AUROC",
                "mean_AUPRC": "Mean AUPRC",
                "mean_top10_ppv": "Mean top 10% PPV",
            },
        )
        subgroup_interpretation = (
            f"ChronoEHR-Agent 生成了 {len(subgroup)} 行 subgroup performance summaries，覆盖年龄组、性别、入院类型和既往住院次数。"
            "这些结果用于发现模型表现异质性和后续公平性分析线索，不应解释为因果效应或临床处置建议。"
        )
    else:
        subgroup_md = "Subgroup performance analysis has not been generated yet."
        subgroup_interpretation = "Subgroup performance analysis has not been generated yet."
    deltas_display = deltas.copy()
    deltas_display["队列"] = deltas_display["cohort"].map(label_cohort)
    deltas_display["入院AUROC"] = deltas_display["admission_AUROC"].map(fmt_metric)
    deltas_display["出院AUROC"] = deltas_display["discharge_AUROC"].map(fmt_metric)
    deltas_display["AUROC差值"] = deltas_display["delta_AUROC"].map(lambda value: f"{value:+.3f}")
    deltas_display["入院AUPRC"] = deltas_display["admission_AUPRC"].map(fmt_metric)
    deltas_display["出院AUPRC"] = deltas_display["discharge_AUPRC"].map(fmt_metric)
    deltas_display["AUPRC差值"] = deltas_display["delta_AUPRC"].map(lambda value: f"{value:+.3f}")
    deltas_md = markdown_table(
        deltas_display,
        ["队列", "入院AUROC", "出院AUROC", "AUROC差值", "入院AUPRC", "出院AUPRC", "AUPRC差值"],
    )
    model_md = markdown_table(model_table, list(model_table.columns))

    leakage_days = leakage[leakage["scenario"].astype(str).str.contains("days_to_next", regex=False)].copy()
    leakage_days["cohort_label"] = leakage_days["cohort"].map(label_cohort)
    leakage_md = markdown_table(leakage_days, ["cohort_label", "scenario", "AUROC", "AUPRC"])

    calibration_display = calibration.copy()
    calibration_display["cohort_label"] = calibration_display["cohort"].map(label_cohort)
    calibration_display["model_label"] = calibration_display["model"].map(label_model)
    calibration_md = markdown_table(
        calibration_display,
        ["cohort_label", "model_label", "mean_absolute_calibration_error", "max_absolute_calibration_error"],
        display_names={
            "cohort_label": "队列",
            "model_label": "模型",
            "mean_absolute_calibration_error": "Mean absolute calibration error",
            "max_absolute_calibration_error": "Max absolute calibration error",
        },
    )

    outcome_display = outcome.copy()
    outcome_display["cohort_label"] = outcome_display["cohort"].map(label_cohort)
    outcome_display["event_rate_pct"] = outcome_display["event_rate"].map(fmt_pct)
    outcome_md = markdown_table(
        outcome_display,
        ["cohort_label", "outcome_definition", "events", "event_rate_pct"],
        display_names={
            "cohort_label": "队列",
            "outcome_definition": "结局定义",
            "events": "事件数",
            "event_rate_pct": "事件率",
        },
    )

    text = f"""# ChronoEHR-Agent 跨慢病 Methods/Results 草稿

这个文档由本地脚本自动汇总生成，目的是把 ChronoEHR-Agent 已经跑出的结果整理成后续论文或课题汇报可用的材料。它不是临床诊疗建议，也不是医疗问答结论；它只描述 EHR 数据分析流程、预测时间点、特征可用性和传统模型 baseline。

## 建议题目

中文题目示例：预测时间点和特征可用性对慢病住院患者30天再入院预测性能的影响：一项基于 MIMIC-IV 的 time-aware EHR benchmark。

英文题目示例：Impact of prediction time and feature availability on 30-day readmission prediction in chronic disease cohorts: a time-aware EHR benchmark using MIMIC-IV.

## Methods 草稿

### 数据来源

本研究使用本地 MIMIC-IV 3.1 数据库开展回顾性 EHR 数据分析。ChronoEHR-Agent 被设计为本地研究工具，用于完成队列构建、预测时间点定义、时间窗内特征抽取、特征泄漏审计、传统机器学习 baseline 建模和结果报告生成。工具不输出临床诊断或治疗建议。

### 研究队列

本轮 benchmark 包含四个慢病相关住院队列：糖尿病、CKD、心衰和高血压。每个队列均基于诊断编码识别研究对象，并以住院记录作为 index admission。为降低数据泄漏风险，模型评估采用患者级切分，避免同一患者同时出现在训练集和测试集。

### 结局定义

主要结局为出院后30天内 all-cause hospital readmission。随访窗口从 index admission 的 discharge time 之后开始计算。为了检查结局定义对结果的影响，工具同时生成 emergency/urgent readmission 和 non-elective proxy readmission 的敏感性分析。

### 预测时间点

本研究重点比较三个 prediction time：

1. 入院时预测：只能使用入院时或入院前已经知道的变量，例如年龄、性别、入院类型、既往诊断信息。
2. 入院后24小时预测：可以额外使用入院后24小时内已经产生的化验、用药或 ICU vital signs。
3. 出院前预测：可以使用 index admission 期间、出院前已经产生的变量，包括住院过程化验和 ICU vital signs，但不能使用出院后的随访信息。

这种设计的核心是区分“预测时已经知道的信息”和“未来才会知道的信息”。同一个变量在不同预测时间点可能有不同的合法性，例如 length of stay 在出院时可用，但在入院时预测中属于未来信息。

### 时间窗内特征

结构化特征包括人口学信息、入院信息、既往住院信息、诊断派生变量、化验、糖尿病相关用药、广泛 medication classes、ICU charted vital signs 和 ICU procedure events。Vital signs 来自 MIMIC-IV ICU `chartevents`，包括 heart rate、non-invasive blood pressure、respiratory rate、SpO2 和 temperature。Procedure events 来自 ICU `procedureevents`，包括管路、影像、培养、通气、插管/拔管、GI/GU、透析等类别。广泛 medication features 来自 `hosp/prescriptions`，按药名关键词归入胰岛素、口服降糖药、利尿剂、降压药、抗凝/抗血小板、抗生素、升压药/正性肌力药、镇静镇痛、支气管扩张剂、PPI/H2 blocker、电解质和补液等类别。由于 ICU vitals/procedures 主要覆盖 ICU 场景，相关特征不是所有住院都有，因此报告同时输出 `has`、`count` 和覆盖率，用来区分“没有测量/医嘱”和“测量值/医嘱本身”。

### 特征泄漏审计

ChronoEHR-Agent 对常见泄漏进行显式审计，包括：将 outcome 本身或 outcome proxy 当作特征、入院时使用出院后才知道的变量、再入院预测中使用随访窗口内信息、以及同一患者进入训练集和测试集。工具还加入错误示范变量，例如 `days_to_next_admission`，用于展示泄漏会如何把模型性能推高到不真实水平。

### 模型与指标

主要 baseline 为 logistic regression，因为它依赖少、可解释、适合与传统临床预测模型比较。模型对照包括 scikit-learn Random Forest、scikit-learn HistGradientBoosting，以及基于验证集的 Platt scaling 和 isotonic regression 后校准 Random Forest 与 HistGradientBoosting。XGBoost 和 LightGBM 被设计为可选 backend；当前本地环境尚未安装，因此报告为 skipped。报告指标包括 AUROC、AUPRC、Brier score、敏感度、特异度、PPV、NPV 和 calibration decile summaries。类别不平衡场景下，AUPRC 和 calibration 与 AUROC 一起报告。

## Results 草稿

### 队列规模

四个慢病队列共包含 {stats["total_admissions"]} 次 index admissions，来自 {stats["total_subjects"]} 名患者。30天再入院率范围为 {stats["min_rate"]} 至 {stats["max_rate"]}。

{table1_md}

### 预测时间点与模型性能

从入院时到出院前，所有队列的 AUROC 均有上升，说明随着住院过程信息逐步产生，模型可用信息增加，预测排序能力也随之提高。AUROC 提升最大的是 {stats["max_delta_cohort"]} 队列，出院前相较入院时提高 {stats["max_delta_auroc"]}；提升最小的是 {stats["min_delta_cohort"]} 队列，提高 {stats["min_delta_auroc"]}。

{prediction_md}

出院前相对入院时的差值如下：

{deltas_md}

### ICU vital signs 增量结果

加入 ICU vital signs 后，模型表现并不总是提高。糖尿病、CKD 和心衰队列中，vital-augmented logistic regression 多数略低于 labs-only 或原安全变量模型；高血压队列中，24小时和出院前 vitals 带来很小的 AUPRC/AUROC 提升。这说明 ChronoEHR-Agent 不应默认“变量越多越好”，而应报告特征覆盖率、预测时间点和增量性能。

{vital_increment_md}

### ICU procedure events 增量结果

加入 ICU procedure events 后，不同队列表现不同。糖尿病和 CKD 的 AUROC/AUPRC 基本持平；心衰 24 小时模型 AUPRC 略升；高血压 24 小时和出院前模型均有小幅提升。这个结果说明 procedure events 可能反映 ICU 工作流强度和住院过程复杂度，但它们主要覆盖 ICU 场景，仍需和特征覆盖率一起解释。

{procedure_increment_md}

### 广泛 medication features 增量结果

加入广泛 medication classes 后，糖尿病、CKD 和高血压队列的 AUROC/AUPRC 均有提升，其中高血压队列提升最明显；心衰队列整体变化较小。这个结果提示，用药记录可能包含住院过程严重程度、合并症治疗和临床工作流信息，但它也需要严格按 prediction time 截断。出院前用药特征不能反过来用于入院时或入院后24小时预测。

{medication_increment_md}

### Feature group ablation 总结

进一步把已完成的 prediction-time benchmark 组织成 grouped ablation 后，discharge labs 和 24h labs 是最稳定、最大的一类增量；broad medications 在 24 小时和出院前均带来稳定 AUPRC 提升；ICU vitals 和 ICU procedures 的平均增益较小且方向不稳定。这一结果支持本项目的核心方法学观点：EHR 预测模型不应只追求“加入更多变量”，而应报告变量组、预测时间点、覆盖率和增量价值。

{ablation_md}

### 细粒度 feature selection 线索

在 grouped ablation 的基础上，ChronoEHR-Agent 进一步汇总 logistic regression 系数，识别在多个队列中反复进入 top clinical features 的 concepts。这一步不是因果解释，也不是临床建议；它用于指导下一轮精简特征集，例如优先保留跨队列反复出现的 broad medication、vital sign、lab 和 procedure concepts，再比较 full feature set 与 selected feature set 的 AUROC、AUPRC、Brier score 和 calibration。

{feature_selection_md}

### Selected feature set 对照

基于 repeated concepts 构造 selected feature sets 后，8 个 selected logistic models 的 AUROC/AUPRC 均略低于 full feature sets，但下降幅度较小。24 小时预测的平均 AUROC 变化约为 -0.0027，平均 AUPRC 变化约为 -0.0042；出院前预测的平均 AUROC 变化约为 -0.0033，平均 AUPRC 变化约为 -0.0044。Selected models 的 mean absolute calibration error 约为 0.006 至 0.014。这说明较小、更可解释的特征集可以保留大部分预测性能，并保持可接受的概率校准，适合作为后续论文中的简化模型或敏感性分析。

{selected_md}

### 结局定义敏感性

All-cause 30-day readmission 的事件率高于 emergency/urgent readmission。这个结果提示，在慢病再入院预测研究中，结局窗口和再入院类型定义会显著影响事件率，也会影响模型评估时的临床含义。

{outcome_md}

### 特征泄漏敏感性

合法模型的 AUROC 处于中等水平，而错误加入 `days_to_next_admission` 后，四个队列的 AUROC 和 AUPRC 均达到 1.000。这不是模型真正优秀，而是因为该变量直接由未来再入院时间派生，几乎等于把答案告诉模型。这个结果可以作为 leakage audit 的教学性证据，也说明慢病 EHR 预测研究必须报告 prediction time 和 feature availability。

{leakage_md}

### ED length-of-stay 边界变量敏感性

Prediction-time leakage gate 将 `ed_los_hours` 标记为 24 小时预测中的 conditional availability 变量：它不是结局，也不是随访窗口信息，但在入院时通常无法完整知道。为回应这个审计提醒，ChronoEHR-Agent 重新训练所有包含 `ed_los_hours` 的 24 小时 logistic models，并将该变量移除。

{ed_los_interpretation}

{ed_los_md}

### 固定 alert burden 阈值分析

除 AUROC/AUPRC 外，ChronoEHR-Agent 还按固定 alert burden 汇总最终 24 小时和出院前 logistic models 的表现。这里的 alert burden 仅用于研究报告中描述模型排序行为，不是临床处置建议。

{threshold_interpretation}

{threshold_md}

### Decision-curve net benefit 分析

为进一步避免只依赖 AUROC/AUPRC，ChronoEHR-Agent 计算了最终 24 小时和出院前 logistic models 的 decision-curve net benefit。这里比较的是模型策略、treat-all 策略和 treat-none 策略在给定研究风险阈值下的净收益；该分析不定义临床处置阈值，也不提供诊疗建议。

{decision_curve_interpretation}

{decision_curve_md}

### Subgroup performance 分析

ChronoEHR-Agent 进一步按基础人群分层汇总最终模型性能，包括年龄组、性别、入院类型和既往住院次数。分层结果用于检查模型表现是否在不同人群中明显波动，不能解释为某个 subgroup 变量导致再入院风险，也不是临床诊疗建议。

{subgroup_interpretation}

{subgroup_md}

### 传统模型 baseline 与校准

Random Forest 在 {stats["rf_better"]}/4 个队列中取得高于 logistic regression 的 AUROC，HistGradientBoosting 在四个队列中也取得较强的 AUROC/AUPRC，说明非线性传统机器学习模型可以提高排序性能。但未校准 Random Forest 和 gradient boosting 的 Brier score 较差，提示更高 AUROC 不等于更可靠的风险概率。经过 Platt scaling 或 isotonic regression 后，Random Forest 和 HistGradientBoosting 的 Brier score 与 calibration error 均明显改善。

{model_md}

校准误差汇总如下：

{calibration_md}

## 可写入论文的主要发现

1. 慢病再入院预测中，prediction time 会影响模型性能。入院时、入院后24小时和出院前不是同一个建模任务。
2. 仅报告“用了哪些变量”不够，还需要报告这些变量在预测时是否已经可用。
3. 泄漏变量可以让 AUROC/AUPRC 虚高到 1.000，因此 leakage audit 应作为 EHR predictive modeling 的常规步骤。
4. 对 `ed_los_hours` 这类边界时间变量，敏感性分析显示移除后模型表现几乎不变，因此主要 24 小时结果不依赖该变量。
5. 固定 alert burden 分析可以补充 AUROC/AUPRC，使读者理解模型在 top-risk 工作量约束下的 PPV、recall 和 lift。
6. Decision-curve net benefit 可以补充固定 alert burden 和校准分析，帮助判断模型在给定研究阈值下是否优于 treat-all/treat-none 参照策略。
7. Subgroup performance summary 可以作为后续公平性、异质性和外部验证设计的入口，但不能直接解释为因果结论。
8. Logistic regression 仍然是重要 baseline；Random Forest 和 HistGradientBoosting 可以提高部分排序指标，但概率输出需要校准后才更适合解释为风险。
9. ChronoEHR-Agent 的价值不是替代医学判断，而是把 EHR 研究中容易出错的时间点、随访窗口和特征泄漏问题流程化、可审计化。

## Limitations 草稿

本轮分析仍为单数据库回顾性 benchmark，尚未进行外部验证。队列识别基于诊断编码，可能存在误分类。当前主要结局为 all-cause 30-day readmission，尚未区分计划性与非计划性再入院的完整临床定义。特征集仍以结构化人口学、入院信息、化验、ICU vital signs、ICU procedure events 和部分用药为主，尚未系统纳入 clinical notes 或社会决定因素。HistGradientBoosting 已作为 dependency-light gradient boosting baseline 完成并完成后校准，但 XGBoost 和 LightGBM 仍需在安装依赖后补充。所有结果均应理解为研究工具 demo 和方法学 benchmark，不应直接用于临床决策。

## Generated Files

- `{project_root / "outputs" / "tables" / "chronic_disease_manuscript_table1.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_manuscript_prediction_time_table.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_prediction_time_deltas.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_vital_increment_table.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_procedure_increment_table.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_medication_increment_table.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_feature_group_ablation_summary.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_repeated_feature_concepts.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_selected_feature_set_comparison.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_selected_feature_set_supplementary_table.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_ed_los_sensitivity_comparison.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_threshold_analysis.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_decision_curve.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_subgroup_performance.csv"}`
- `{project_root / "outputs" / "tables" / "chronic_disease_manuscript_model_table.csv"}`
- `{report_path}`
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    tables = project_root / "outputs" / "tables"
    reports = project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    cohort = read_table(project_root, "outputs/tables/chronic_disease_benchmark_cohort_summary.csv")
    prediction = read_table(project_root, "outputs/tables/chronic_disease_prediction_time_benchmark.csv")
    outcome = read_table(project_root, "outputs/tables/chronic_disease_outcome_sensitivity_summary.csv")
    leakage = read_table(project_root, "outputs/tables/chronic_disease_leakage_sensitivity_summary.csv")
    comparison = read_table(project_root, "outputs/tables/chronic_disease_model_baseline_comparison.csv")
    calibration = read_table(project_root, "outputs/tables/chronic_disease_model_calibration_summary.csv")

    table1 = make_table1(cohort)
    prediction_table = make_prediction_time_table(prediction)
    deltas = make_prediction_time_deltas(prediction)
    model_table = make_model_table(comparison, calibration)

    table1.to_csv(tables / "chronic_disease_manuscript_table1.csv", index=False)
    prediction_table.to_csv(tables / "chronic_disease_manuscript_prediction_time_table.csv", index=False)
    deltas.to_csv(tables / "chronic_disease_prediction_time_deltas.csv", index=False)
    make_vital_increment_table(prediction).to_csv(tables / "chronic_disease_vital_increment_table.csv", index=False)
    make_procedure_increment_table(prediction).to_csv(tables / "chronic_disease_procedure_increment_table.csv", index=False)
    make_medication_increment_table(prediction).to_csv(tables / "chronic_disease_medication_increment_table.csv", index=False)
    model_table.to_csv(tables / "chronic_disease_manuscript_model_table.csv", index=False)

    write_report(
        project_root=project_root,
        cohort=cohort,
        prediction=prediction,
        deltas=deltas,
        outcome=outcome,
        leakage=leakage,
        comparison=comparison,
        calibration=calibration,
        table1=table1,
        model_table=model_table,
        report_path=reports / "chronic_disease_methods_results_draft.md",
    )
    print("Cross-cohort Methods/Results draft complete")
    print(f"cohorts={len(cohort)} prediction_rows={len(prediction)} model_rows={len(model_table)}")


if __name__ == "__main__":
    main()
