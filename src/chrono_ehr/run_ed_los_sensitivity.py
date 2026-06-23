#!/usr/bin/env python3
"""Run sensitivity models after removing ED length-of-stay features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from prediction_time_model_tools import run_feature_set
from prediction_time_spec_loader import load_prediction_time_config, load_raw_config


BENCHMARK_TABLE = "outputs/tables/chronic_disease_prediction_time_benchmark.csv"
COHORT_LABELS = {
    "diabetes": "糖尿病",
    "ckd": "CKD",
    "heart_failure": "心衰",
    "hypertension": "高血压",
}


def contains_ed_los(spec: dict[str, Any]) -> bool:
    return "ed_los_hours" in [str(feature) for feature in spec.get("numeric_features", [])]


def sensitivity_spec(spec: dict[str, Any]) -> dict[str, Any]:
    new_spec = dict(spec)
    new_spec["source_feature_set"] = spec["feature_set"]
    new_spec["feature_set"] = f"{spec['feature_set']}_no_ed_los"
    new_spec["numeric_features"] = [feature for feature in spec.get("numeric_features", []) if str(feature) != "ed_los_hours"]
    new_spec["categorical_features"] = list(spec.get("categorical_features", []))
    return new_spec


def candidate_specs(project_root: Path) -> list[tuple[str, str, dict[str, Any]]]:
    raw = load_raw_config()
    candidates = []
    for study in raw.get("studies", {}):
        config = load_prediction_time_config(study)
        for spec in config["specs"]:
            if spec.get("prediction_time") == "inhospital_24h" and contains_ed_los(spec):
                candidates.append((study, config["cohort_path"], sensitivity_spec(spec)))
    return candidates


def run_models(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_parts = []
    coefficient_parts = []
    prediction_parts = []
    for study, cohort_path, spec in candidate_specs(project_root):
        performance, coefficients, predictions = run_feature_set(project_root, cohort_path, spec)
        source_feature_set = spec["source_feature_set"]
        for frame in [performance, coefficients, predictions]:
            frame.insert(0, "cohort", study)
            frame.insert(1, "cohort_label", COHORT_LABELS.get(study, study))
            frame.insert(2, "source_feature_set", source_feature_set)
            frame.insert(3, "analysis_name", "remove_ed_los_hours")
        performance_parts.append(performance)
        coefficient_parts.append(coefficients)
        prediction_parts.append(predictions)

    if not performance_parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat(performance_parts, ignore_index=True),
        pd.concat(coefficient_parts, ignore_index=True),
        pd.concat(prediction_parts, ignore_index=True),
    )


def compare_with_original(project_root: Path, sensitivity_performance: pd.DataFrame) -> pd.DataFrame:
    if sensitivity_performance.empty:
        return pd.DataFrame()
    benchmark = pd.read_csv(project_root / BENCHMARK_TABLE)
    benchmark = benchmark.set_index(["cohort", "feature_set"])
    rows = []
    tests = sensitivity_performance[sensitivity_performance["split"].eq("test")].copy()
    for row in tests.itertuples(index=False):
        key = (row.cohort, row.source_feature_set)
        if key not in benchmark.index:
            continue
        original = benchmark.loc[key]
        rows.append(
            {
                "cohort": row.cohort,
                "cohort_label": row.cohort_label,
                "prediction_time": row.prediction_time,
                "source_feature_set": row.source_feature_set,
                "sensitivity_feature_set": row.feature_set,
                "n": int(row.n),
                "events": int(row.events),
                "original_AUROC": float(original["AUROC"]),
                "no_ed_los_AUROC": float(row.AUROC),
                "delta_AUROC": float(row.AUROC - original["AUROC"]),
                "original_AUPRC": float(original["AUPRC"]),
                "no_ed_los_AUPRC": float(row.AUPRC),
                "delta_AUPRC": float(row.AUPRC - original["AUPRC"]),
                "original_Brier": float(original["Brier_score"]),
                "no_ed_los_Brier": float(row.Brier_score),
                "delta_Brier": float(row.Brier_score - original["Brier_score"]),
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


def write_report(comparison: pd.DataFrame, output: Path) -> None:
    if comparison.empty:
        summary = pd.DataFrame()
        interpretation = "没有找到包含 `ed_los_hours` 的 24h feature set，因此未生成敏感性模型。"
    else:
        summary = (
            comparison.groupby("cohort_label", sort=False)
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
        mean_abs_auroc = comparison["delta_AUROC"].abs().mean()
        mean_abs_auprc = comparison["delta_AUPRC"].abs().mean()
        interpretation = (
            f"去掉 `ed_los_hours` 后，平均绝对 AUROC 变化为 {mean_abs_auroc:.4f}，"
            f"平均绝对 AUPRC 变化为 {mean_abs_auprc:.4f}。"
            "如果这些变化很小，可以在 Methods/Results 中说明 24h 模型结论不主要依赖这个边界时间变量。"
        )

    text = f"""# ED Length-of-Stay Sensitivity Analysis

这个报告回应 leakage gate 的 P1 action item：`ed_los_hours` 在 24h prediction 中属于 conditional availability。它不是 outcome，也不是随访信息，但入院时通常无法完整知道；因此本敏感性分析重新训练去掉 `ed_los_hours` 的 24h logistic models。

本报告只用于 EHR 数据研究质量控制，不提供医学诊疗建议。

## Interpretation

{interpretation}

## Cohort-Level Summary

{markdown_table(summary, ["cohort_label", "comparisons", "mean_delta_AUROC", "mean_delta_AUPRC", "mean_delta_Brier", "max_abs_delta_AUROC", "max_abs_delta_AUPRC"])}

## Model-Level Comparison

{markdown_table(comparison, ["cohort_label", "source_feature_set", "n", "events", "original_AUROC", "no_ed_los_AUROC", "delta_AUROC", "original_AUPRC", "no_ed_los_AUPRC", "delta_AUPRC", "delta_Brier"])}

## Suggested Methods Text

As a sensitivity analysis for prediction-time boundary variables, we retrained all 24-hour logistic regression models after removing emergency department length of stay (`ed_los_hours`). This analysis evaluates whether model discrimination materially depends on a variable that may require explicit timing assumptions at the 24-hour prediction point.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    performance, coefficients, predictions = run_models(args.project_root)
    comparison = compare_with_original(args.project_root, performance)
    performance.to_csv(tables / "chronic_disease_ed_los_sensitivity_performance.csv", index=False)
    coefficients.to_csv(tables / "chronic_disease_ed_los_sensitivity_coefficients.csv", index=False)
    predictions.to_csv(tables / "chronic_disease_ed_los_sensitivity_predictions.csv", index=False)
    comparison.to_csv(tables / "chronic_disease_ed_los_sensitivity_comparison.csv", index=False)
    write_report(comparison, reports / "chronic_disease_ed_los_sensitivity_report.md")
    print(f"ED LOS sensitivity models: {len(performance[performance['split'].eq('test')]) if not performance.empty else 0}")
    print(f"Wrote {reports / 'chronic_disease_ed_los_sensitivity_report.md'}")


if __name__ == "__main__":
    main()
