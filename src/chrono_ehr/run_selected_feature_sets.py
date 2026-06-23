#!/usr/bin/env python3
"""Train selected-concept logistic models and compare them with full feature sets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from prediction_time_model_tools import run_feature_set
from prediction_time_spec_loader import load_prediction_time_config
from summarize_feature_selection import COHORT_LABELS, FINAL_FEATURE_SETS, concept_name, feature_group


BENCHMARK_TABLE = "outputs/tables/chronic_disease_prediction_time_benchmark.csv"
REPEATED_CONCEPTS_TABLE = "outputs/tables/chronic_disease_repeated_feature_concepts.csv"

ALWAYS_KEEP_GROUPS = {"admission_history", "encounter_timing", "cohort_definition"}
SELECTABLE_GROUPS = {"24h_labs", "discharge_labs", "diabetes_medications", "broad_medications", "icu_vitals", "icu_procedures"}


def read_repeated_concepts(project_root: Path, min_cohorts: int, min_appearances: int) -> dict[str, set[str]]:
    path = project_root / REPEATED_CONCEPTS_TABLE
    if not path.exists():
        raise FileNotFoundError(f"Missing repeated concepts table: {path}. Run --feature-selection-summary first.")
    df = pd.read_csv(path)
    selected = df[(df["n_cohorts"].ge(min_cohorts)) & (df["appearances"].ge(min_appearances))].copy()
    concepts: dict[str, set[str]] = {}
    for row in selected.itertuples(index=False):
        concepts.setdefault(str(row.feature_group), set()).add(str(row.concept))
    return concepts


def keep_numeric_feature(feature: str, selected_concepts: dict[str, set[str]]) -> bool:
    group = feature_group(feature)
    if group in ALWAYS_KEEP_GROUPS:
        return True
    if group not in SELECTABLE_GROUPS:
        return False
    return concept_name(feature) in selected_concepts.get(group, set())


def selected_spec(full_spec: dict[str, Any], selected_concepts: dict[str, set[str]]) -> dict[str, Any]:
    spec = dict(full_spec)
    spec["source_feature_set"] = full_spec["feature_set"]
    spec["feature_set"] = f"{full_spec['prediction_time']}_selected_concepts"
    spec["numeric_features"] = [
        feature for feature in full_spec["numeric_features"] if keep_numeric_feature(str(feature), selected_concepts)
    ]
    spec["categorical_features"] = list(full_spec["categorical_features"])
    return spec


def selected_feature_rows(study: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for feature in spec["numeric_features"]:
        rows.append(
            {
                "cohort": study,
                "cohort_label": COHORT_LABELS.get(study, study),
                "prediction_time": spec["prediction_time"],
                "feature_set": spec["feature_set"],
                "source_feature_set": spec["source_feature_set"],
                "feature": feature,
                "feature_type": "numeric",
                "feature_group": feature_group(str(feature)),
                "concept": concept_name(str(feature)),
            }
        )
    for feature in spec["categorical_features"]:
        rows.append(
            {
                "cohort": study,
                "cohort_label": COHORT_LABELS.get(study, study),
                "prediction_time": spec["prediction_time"],
                "feature_set": spec["feature_set"],
                "source_feature_set": spec["source_feature_set"],
                "feature": feature,
                "feature_type": "categorical",
                "feature_group": "categorical_demographics",
                "concept": feature,
            }
        )
    return rows


def find_full_spec(config: dict[str, Any], feature_set: str) -> dict[str, Any]:
    for spec in config["specs"]:
        if spec["feature_set"] == feature_set:
            return spec
    raise KeyError(f"Feature set not found: {feature_set}")


def run_selected_models(project_root: Path, min_cohorts: int, min_appearances: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_concepts = read_repeated_concepts(project_root, min_cohorts, min_appearances)
    performance_parts = []
    coefficient_parts = []
    prediction_parts = []
    feature_rows = []

    for study, stage_map in FINAL_FEATURE_SETS.items():
        config = load_prediction_time_config(study)
        for full_feature_set in stage_map.values():
            full = find_full_spec(config, full_feature_set)
            spec = selected_spec(full, selected_concepts)
            feature_rows.extend(selected_feature_rows(study, spec))
            performance, coefficients, predictions = run_feature_set(project_root, config["cohort_path"], spec)
            for frame in [performance, coefficients, predictions]:
                frame.insert(0, "cohort", study)
                frame.insert(1, "source_feature_set", full_feature_set)
            performance_parts.append(performance)
            coefficient_parts.append(coefficients)
            prediction_parts.append(predictions)

    performance_all = pd.concat(performance_parts, ignore_index=True)
    coefficients_all = pd.concat(coefficient_parts, ignore_index=True)
    predictions_all = pd.concat(prediction_parts, ignore_index=True)
    selected_features = pd.DataFrame(feature_rows)
    return performance_all, coefficients_all, predictions_all, selected_features


def compare_with_full(project_root: Path, selected_performance: pd.DataFrame, selected_features: pd.DataFrame) -> pd.DataFrame:
    benchmark = pd.read_csv(project_root / BENCHMARK_TABLE)
    full = benchmark.set_index(["cohort", "feature_set"])
    rows = []
    selected_test = selected_performance[selected_performance["split"].eq("test")].copy()
    feature_counts = (
        selected_features.groupby(["cohort", "feature_set", "source_feature_set"])
        .agg(
            selected_features=("feature", "count"),
            selected_numeric_features=("feature_type", lambda values: int((pd.Series(values) == "numeric").sum())),
            selected_categorical_features=("feature_type", lambda values: int((pd.Series(values) == "categorical").sum())),
        )
        .reset_index()
    )
    feature_counts = feature_counts.set_index(["cohort", "feature_set", "source_feature_set"])
    for row in selected_test.itertuples(index=False):
        key = (row.cohort, row.source_feature_set)
        if key not in full.index:
            continue
        full_row = full.loc[key]
        count_key = (row.cohort, row.feature_set, row.source_feature_set)
        counts = feature_counts.loc[count_key]
        rows.append(
            {
                "cohort": row.cohort,
                "cohort_label": COHORT_LABELS.get(row.cohort, row.cohort),
                "prediction_time": row.prediction_time,
                "selected_feature_set": row.feature_set,
                "full_feature_set": row.source_feature_set,
                "selected_features": int(counts["selected_features"]),
                "selected_numeric_features": int(counts["selected_numeric_features"]),
                "selected_categorical_features": int(counts["selected_categorical_features"]),
                "n": int(row.n),
                "events": int(row.events),
                "full_AUROC": float(full_row["AUROC"]),
                "selected_AUROC": float(row.AUROC),
                "delta_AUROC": float(row.AUROC - full_row["AUROC"]),
                "full_AUPRC": float(full_row["AUPRC"]),
                "selected_AUPRC": float(row.AUPRC),
                "delta_AUPRC": float(row.AUPRC - full_row["AUPRC"]),
                "full_Brier": float(full_row["Brier_score"]),
                "selected_Brier": float(row.Brier_score),
                "delta_Brier": float(row.Brier_score - full_row["Brier_score"]),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No data available."
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(comparison: pd.DataFrame, selected_features: pd.DataFrame, output: Path) -> None:
    summary = (
        comparison.groupby("prediction_time", sort=False)
        .agg(
            comparisons=("cohort", "count"),
            mean_selected_features=("selected_features", "mean"),
            mean_delta_AUROC=("delta_AUROC", "mean"),
            mean_delta_AUPRC=("delta_AUPRC", "mean"),
            mean_delta_Brier=("delta_Brier", "mean"),
            selected_beats_full_AUROC=("delta_AUROC", lambda values: int((values > 0).sum())),
            selected_beats_full_AUPRC=("delta_AUPRC", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    concept_counts = (
        selected_features[selected_features["feature_type"].eq("numeric")]
        .groupby(["feature_group", "concept"], sort=False)
        .agg(
            feature_columns=("feature", "count"),
            cohorts=("cohort", lambda values: ", ".join(sorted(set(map(str, values))))),
            prediction_times=("prediction_time", lambda values: ", ".join(sorted(set(map(str, values))))),
        )
        .reset_index()
        .sort_values(["feature_columns", "feature_group", "concept"], ascending=[False, True, True])
        .head(30)
    )
    text = f"""# Selected Feature Set Comparison

这个报告把 fine-grained feature selection 中反复出现的 concepts 变成 selected feature sets，并重新训练 logistic regression，与 full feature sets 做对照。

## Selection Rule

- 人口学类别变量全部保留。
- 基础病史/入院史变量保留，例如 age、prior admissions、days since prior discharge。
- 临床扩展变量只保留 repeated concepts 表中跨队列反复出现的 concepts。
- 这一步用于研究建模和变量筛选，不是因果解释，也不是临床诊疗建议。

## Summary

{markdown_table(summary, ["prediction_time", "comparisons", "mean_selected_features", "mean_delta_AUROC", "mean_delta_AUPRC", "mean_delta_Brier", "selected_beats_full_AUROC", "selected_beats_full_AUPRC"])}

## Full Vs Selected

{markdown_table(comparison, ["cohort_label", "prediction_time", "selected_features", "full_AUROC", "selected_AUROC", "delta_AUROC", "full_AUPRC", "selected_AUPRC", "delta_AUPRC", "delta_Brier"])}

## Selected Concepts

{markdown_table(concept_counts, ["feature_group", "concept", "feature_columns", "cohorts", "prediction_times"])}

## Interpretation

- 如果 selected model 接近 full model，说明可以用更小、更可解释的特征集完成主要 benchmark。
- 如果 selected model 明显下降，说明 full model 中仍有许多低频或队列特异变量在提供增量，需要进一步做 cohort-specific selection。
- 本报告适合指导下一轮 `selected_clinical_features` 配置，但最终论文结果仍应同时报告 full model 和 selected model。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--min-cohorts", type=int, default=3)
    parser.add_argument("--min-appearances", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    performance, coefficients, predictions, selected_features = run_selected_models(
        args.project_root,
        min_cohorts=args.min_cohorts,
        min_appearances=args.min_appearances,
    )
    comparison = compare_with_full(args.project_root, performance, selected_features)

    performance.to_csv(tables / "chronic_disease_selected_feature_set_performance.csv", index=False)
    coefficients.to_csv(tables / "chronic_disease_selected_feature_set_coefficients.csv", index=False)
    predictions.to_csv(tables / "chronic_disease_selected_feature_set_predictions.csv", index=False)
    selected_features.to_csv(tables / "chronic_disease_selected_feature_set_features.csv", index=False)
    comparison.to_csv(tables / "chronic_disease_selected_feature_set_comparison.csv", index=False)
    write_report(comparison, selected_features, reports / "chronic_disease_selected_feature_set_report.md")

    print("Selected feature set comparison complete")
    print(f"models={len(performance[performance['split'].eq('test')])} comparison_rows={len(comparison)}")


if __name__ == "__main__":
    main()
