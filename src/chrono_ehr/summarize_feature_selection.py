#!/usr/bin/env python3
"""Summarize fine-grained feature-selection signals from logistic coefficients."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


COEFFICIENT_FILES = {
    "diabetes": "outputs/tables/mimic_diabetes_prediction_time_logistic_coefficients.csv",
    "ckd": "outputs/tables/mimic_ckd_logistic_coefficients.csv",
    "heart_failure": "outputs/tables/mimic_heart_failure_logistic_coefficients.csv",
    "hypertension": "outputs/tables/mimic_hypertension_logistic_coefficients.csv",
}

COHORT_LABELS = {
    "diabetes": "糖尿病",
    "ckd": "CKD",
    "heart_failure": "心衰",
    "hypertension": "高血压",
}

FINAL_FEATURE_SETS = {
    "diabetes": {
        "inhospital_24h": "inhospital_24h_lab_med_vital_proc_genmed_minimal",
        "discharge": "discharge_safe_vital_proc_genmed_minimal",
    },
    "ckd": {
        "inhospital_24h": "inhospital_24h_lab_vital_proc_genmed_minimal",
        "discharge": "discharge_lab_vital_proc_genmed_minimal",
    },
    "heart_failure": {
        "inhospital_24h": "inhospital_24h_lab_vital_proc_genmed_minimal",
        "discharge": "discharge_lab_vital_proc_genmed_minimal",
    },
    "hypertension": {
        "inhospital_24h": "inhospital_24h_lab_vital_proc_genmed_minimal",
        "discharge": "discharge_lab_vital_proc_genmed_minimal",
    },
}

STAT_SUFFIXES = {
    "count",
    "has",
    "mean",
    "min",
    "max",
    "last",
    "abnormal_count",
    "warning_count",
    "minutes",
    "iv_count",
    "iv_has",
    "missing",
}


def read_coefficients(project_root: Path) -> pd.DataFrame:
    frames = []
    for cohort, relative_path in COEFFICIENT_FILES.items():
        path = project_root / relative_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["cohort"] = cohort
        df["cohort_label"] = COHORT_LABELS.get(cohort, cohort)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No logistic coefficient files found.")
    return pd.concat(frames, ignore_index=True)


def remove_missing_suffix(feature: str) -> str:
    return feature.removesuffix("_missing")


def base_feature(feature: str) -> str:
    if "=" in feature:
        return feature.split("=", 1)[0]
    return remove_missing_suffix(feature)


def feature_group(feature: str) -> str:
    base = base_feature(feature)
    if feature == "intercept":
        return "intercept"
    if "=" in feature:
        return "categorical_demographics"
    if base in {"anchor_age", "prior_admissions_count", "days_since_prior_discharge"}:
        return "admission_history"
    if base in {"ed_los_hours", "length_of_stay_days"}:
        return "encounter_timing"
    if base.startswith(("lab24h_", "ckdlab24h_", "hflab24h_", "htnlab24h_")):
        return "24h_labs"
    if base.startswith(("labdischarge_", "ckdlabdischarge_", "hflabdischarge_", "htnlabdischarge_")):
        return "discharge_labs"
    if base.startswith("med24h_"):
        return "diabetes_medications"
    if base.startswith(("genmed24h_", "genmeddischarge_")):
        return "broad_medications"
    if base.startswith(("vital24h_", "vitaldischarge_")):
        return "icu_vitals"
    if base.startswith(("proc24h_", "procdischarge_")):
        return "icu_procedures"
    if base.startswith(("current_", "known_", "prior_ckd", "prior_hf", "prior_hypertension")):
        return "cohort_definition"
    return "other"


def prediction_stage_from_feature(feature: str, prediction_time: str) -> str:
    base = base_feature(feature)
    if "24h" in base:
        return "24h"
    if "discharge" in base:
        return "discharge"
    if prediction_time == "admission":
        return "admission"
    return prediction_time


def strip_prefix(feature: str) -> str:
    base = base_feature(feature)
    prefixes = [
        "genmed24h_",
        "genmeddischarge_",
        "proc24h_",
        "procdischarge_",
        "vital24h_",
        "vitaldischarge_",
        "lab24h_",
        "labdischarge_",
        "ckdlab24h_",
        "ckdlabdischarge_",
        "hflab24h_",
        "hflabdischarge_",
        "htnlab24h_",
        "htnlabdischarge_",
        "med24h_",
    ]
    for prefix in prefixes:
        if base.startswith(prefix):
            return base[len(prefix) :]
    return base


def strip_stat_suffix(name: str) -> str:
    parts = name.split("_")
    for suffix_len in [2, 1]:
        if len(parts) <= suffix_len:
            continue
        suffix = "_".join(parts[-suffix_len:])
        if suffix in STAT_SUFFIXES:
            return "_".join(parts[:-suffix_len])
    return name


def concept_name(feature: str) -> str:
    if "=" in feature:
        return base_feature(feature)
    return strip_stat_suffix(strip_prefix(feature))


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feature"] = out["feature"].astype(str)
    out = out[out["feature"].ne("intercept")].copy()
    out["base_feature"] = out["feature"].map(base_feature)
    out["feature_group"] = out["feature"].map(feature_group)
    out["feature_stage"] = [prediction_stage_from_feature(f, p) for f, p in zip(out["feature"], out["prediction_time"], strict=False)]
    out["concept"] = out["feature"].map(concept_name)
    out["direction"] = out["coefficient"].map(lambda value: "positive" if value > 0 else "negative")
    out["is_missing_indicator"] = out["feature"].str.endswith("_missing")
    out["is_clinical_group"] = out["feature_group"].isin(
        {
            "24h_labs",
            "discharge_labs",
            "diabetes_medications",
            "broad_medications",
            "icu_vitals",
            "icu_procedures",
            "encounter_timing",
        }
    )
    return out


def keep_final_feature_sets(df: pd.DataFrame) -> pd.DataFrame:
    masks = []
    for cohort, stage_map in FINAL_FEATURE_SETS.items():
        for feature_set in stage_map.values():
            masks.append(df["cohort"].eq(cohort) & df["feature_set"].eq(feature_set))
    if not masks:
        return df.iloc[0:0].copy()
    mask = masks[0]
    for item in masks[1:]:
        mask = mask | item
    return df[mask].copy()


def top_features(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    clinical = keep_final_feature_sets(df)
    clinical = clinical[clinical["is_clinical_group"] & ~clinical["is_missing_indicator"]].copy()
    if clinical.empty:
        return clinical
    clinical = clinical.sort_values(["cohort", "prediction_time", "feature_group", "abs_coefficient"], ascending=[True, True, True, False])
    return clinical.groupby(["cohort", "prediction_time", "feature_group"], group_keys=False).head(top_n)


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    clinical = keep_final_feature_sets(df)
    clinical = clinical[clinical["is_clinical_group"]].copy()
    if clinical.empty:
        return clinical
    rows = []
    for keys, group in clinical.groupby(["cohort", "cohort_label", "prediction_time", "feature_set", "feature_group"], sort=False):
        top = group.sort_values("abs_coefficient", ascending=False).iloc[0]
        non_missing = group[~group["is_missing_indicator"]]
        rows.append(
            {
                "cohort": keys[0],
                "cohort_label": keys[1],
                "prediction_time": keys[2],
                "feature_set": keys[3],
                "feature_group": keys[4],
                "n_coefficients": int(len(group)),
                "n_non_missing_coefficients": int(len(non_missing)),
                "mean_abs_coefficient": float(group["abs_coefficient"].mean()),
                "median_abs_coefficient": float(group["abs_coefficient"].median()),
                "max_abs_coefficient": float(top["abs_coefficient"]),
                "top_feature": str(top["feature"]),
                "top_concept": str(top["concept"]),
                "top_direction": str(top["direction"]),
                "top_coefficient": float(top["coefficient"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["prediction_time", "cohort", "max_abs_coefficient"], ascending=[True, True, False])


def concept_stability(top: pd.DataFrame) -> pd.DataFrame:
    focus = top[top["feature_group"].isin({"broad_medications", "icu_procedures", "icu_vitals", "24h_labs", "discharge_labs"})].copy()
    if focus.empty:
        return focus
    summary = (
        focus.groupby(["feature_group", "concept"], sort=False)
        .agg(
            appearances=("feature", "count"),
            cohorts=("cohort", lambda values: ", ".join(sorted(set(map(str, values))))),
            n_cohorts=("cohort", lambda values: len(set(map(str, values)))),
            prediction_times=("prediction_time", lambda values: ", ".join(sorted(set(map(str, values))))),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
            positive_count=("direction", lambda values: int((pd.Series(values) == "positive").sum())),
            negative_count=("direction", lambda values: int((pd.Series(values) == "negative").sum())),
            example_features=("feature", lambda values: ", ".join(list(map(str, values))[:5])),
        )
        .reset_index()
    )
    return summary.sort_values(["n_cohorts", "appearances", "mean_abs_coefficient"], ascending=[False, False, False])


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "No data available."
    small = df.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in small[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(top: pd.DataFrame, groups: pd.DataFrame, stability: pd.DataFrame, output: Path) -> None:
    strongest_groups = groups.sort_values("max_abs_coefficient", ascending=False)
    strongest_features = top.sort_values("abs_coefficient", ascending=False)
    stable = stability[stability["n_cohorts"].ge(2)].copy()
    text = f"""# Fine-Grained Feature Selection Summary

这个报告使用已经训练好的 logistic regression 系数，整理哪些细粒度变量最值得后续关注。它不是因果分析，也不是临床建议；它只是帮助研究者理解当前 EHR 预测模型主要依赖哪些变量。

## How To Read This

- `abs_coefficient` 越大，说明该变量在当前 logistic 模型中的权重越大。
- 数值变量已做标准化，因此同一模型内的数值变量可以粗略比较。
- 类别变量和缺失指示变量可能反映数据采集模式，不一定是医学机制。
- 这些结果适合用来做 feature selection、误差分析和 Methods/Results 描述，不能解释为治疗效果。

## Strongest Clinical Feature Groups

{markdown_table(strongest_groups, ["cohort_label", "prediction_time", "feature_group", "n_coefficients", "max_abs_coefficient", "top_feature", "top_direction"], 18)}

## Top Clinical Features

{markdown_table(strongest_features, ["cohort_label", "prediction_time", "feature_group", "feature", "concept", "direction", "coefficient", "abs_coefficient"], 24)}

## Repeated Concepts Across Cohorts

{markdown_table(stable, ["feature_group", "concept", "n_cohorts", "appearances", "prediction_times", "mean_abs_coefficient", "positive_count", "negative_count"], 24)}

## Practical Interpretation

- Broad medications 中反复出现的变量更适合进入下一轮精简模型，而不是把所有药物类别都保留。
- ICU procedures 和 ICU vitals 的高权重变量需要同时检查覆盖率；如果只在 ICU 人群中出现，它们可能代表“是否进 ICU/是否被密集监测”。
- Labs 的 top concepts 可以帮助写 time-aware prediction 的结果解释：同一慢病任务中，24h labs 和 discharge labs 的贡献可能不同。
- 下一步可以把稳定出现的 concepts 做成 `selected_clinical_features` feature set，再和 full feature set 比较 AUROC、AUPRC、Brier 和校准。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    coefficients = annotate(read_coefficients(args.project_root))
    top = top_features(coefficients, args.top_n)
    groups = group_summary(coefficients)
    stability = concept_stability(top)

    coefficients.to_csv(tables / "chronic_disease_logistic_coefficients_annotated.csv", index=False)
    top.to_csv(tables / "chronic_disease_top_clinical_features.csv", index=False)
    groups.to_csv(tables / "chronic_disease_feature_selection_group_summary.csv", index=False)
    stability.to_csv(tables / "chronic_disease_repeated_feature_concepts.csv", index=False)
    write_report(top, groups, stability, reports / "chronic_disease_feature_selection_report.md")

    print("Fine-grained feature selection summary complete")
    print(f"annotated_coefficients={len(coefficients)} top_features={len(top)} repeated_concepts={len(stability)}")


if __name__ == "__main__":
    main()
