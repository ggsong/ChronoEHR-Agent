#!/usr/bin/env python3
"""Summarize registered chronic-disease demos into one benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT / "configs" / "study_registry.json"


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"metric", "value"}.issubset(df.columns):
        return {}
    return df.set_index("metric")["value"].astype(str).to_dict()


def as_int(metrics: dict[str, str], key: str) -> int | None:
    if key not in metrics:
        return None
    return int(float(metrics[key]))


def as_float(metrics: dict[str, str], key: str) -> float | None:
    if key not in metrics:
        return None
    return float(metrics[key])


def prefix_for(study: dict[str, Any]) -> str:
    cohort = study.get("cohort")
    if not cohort:
        return ""
    return f"mimic_{cohort}"


def collect_cohort_summary(project: Path, studies: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for study in studies:
        prefix = prefix_for(study)
        if not prefix:
            continue
        metrics = metric_dict(project / "outputs" / "tables" / f"{prefix}_cohort_summary.csv")
        if not metrics:
            continue
        rows.append(
            {
                "study_id": study["id"],
                "status": study.get("status", "NA"),
                "cohort": study.get("cohort", "NA"),
                "outcome": study.get("outcome", "NA"),
                "final_index_admissions": as_int(metrics, "final_index_admissions"),
                "final_subjects": as_int(metrics, "final_subjects"),
                "readmission_30d_count": as_int(metrics, "readmission_30d_count"),
                "readmission_30d_rate": as_float(metrics, "readmission_30d_rate"),
            }
        )
    return pd.DataFrame(rows)


def collect_prediction_time(project: Path, studies: list[dict[str, Any]]) -> pd.DataFrame:
    parts = []
    for study in studies:
        prefix = prefix_for(study)
        path = project / "outputs" / "tables" / f"{prefix}_prediction_time_model_performance.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "split" in df.columns:
            df = df[df["split"].eq("test")].copy()
        df.insert(0, "study_id", study["id"])
        df.insert(1, "cohort", study.get("cohort", "NA"))
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    keep = [
        "study_id",
        "cohort",
        "feature_set",
        "prediction_time",
        "n",
        "events",
        "event_rate",
        "AUROC",
        "AUPRC",
        "Brier_score",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
    ]
    combined = pd.concat(parts, ignore_index=True)
    return combined[[col for col in keep if col in combined.columns]]


def collect_outcome_sensitivity(project: Path, studies: list[dict[str, Any]]) -> pd.DataFrame:
    parts = []
    for study in studies:
        prefix = prefix_for(study)
        path = project / "outputs" / "tables" / f"{prefix}_outcome_sensitivity.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "study_id", study["id"])
        df.insert(1, "cohort", study.get("cohort", "NA"))
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_leakage_sensitivity(project: Path, studies: list[dict[str, Any]]) -> pd.DataFrame:
    parts = []
    for study in studies:
        prefix = prefix_for(study)
        path = project / "outputs" / "tables" / f"{prefix}_leakage_sensitivity.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "study_id", study["id"])
        df.insert(1, "cohort", study.get("cohort", "NA"))
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def markdown_table(df: pd.DataFrame, columns: list[str], pct_cols: set[str] | None = None) -> str:
    pct_cols = pct_cols or set()
    if df.empty:
        return "No data found."
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in df[columns].itertuples(index=False):
        values = []
        for col, value in zip(columns, row):
            if pd.isna(value):
                values.append("")
            elif col in pct_cols:
                values.append(f"{float(value):.2%}")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_prediction_time(df: pd.DataFrame, metric: str, path: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    plot_df = df.copy()
    feature_labels = {
        "admission_safe_minimal": "Admission",
        "inhospital_24h_lab_minimal": "24h labs",
        "inhospital_24h_lab_med_minimal": "24h labs+meds",
        "inhospital_24h_lab_med_vital_minimal": "24h labs+meds+vitals",
        "inhospital_24h_lab_med_vital_proc_minimal": "24h labs+meds+vitals+procedures",
        "inhospital_24h_lab_med_vital_proc_genmed_minimal": "24h labs+meds+vitals+procedures+general meds",
        "inhospital_24h_lab_vital_minimal": "24h labs+vitals",
        "inhospital_24h_lab_vital_proc_minimal": "24h labs+vitals+procedures",
        "inhospital_24h_lab_vital_proc_genmed_minimal": "24h labs+vitals+procedures+general meds",
        "discharge_safe_minimal": "Discharge",
        "discharge_safe_vital_minimal": "Discharge+vitals",
        "discharge_safe_vital_proc_minimal": "Discharge+vitals+procedures",
        "discharge_safe_vital_proc_genmed_minimal": "Discharge+vitals+procedures+general meds",
        "discharge_lab_minimal": "Discharge labs",
        "discharge_lab_vital_minimal": "Discharge labs+vitals",
        "discharge_lab_vital_proc_minimal": "Discharge labs+vitals+procedures",
        "discharge_lab_vital_proc_genmed_minimal": "Discharge labs+vitals+procedures+general meds",
    }
    plot_df["feature_label"] = plot_df["feature_set"].map(feature_labels).fillna(plot_df["feature_set"])
    plot_df["label"] = (
        plot_df["cohort"].astype(str)
        + " | "
        + plot_df["feature_label"].astype(str)
    )
    stage_order = {"admission": 0, "inhospital_24h": 1, "discharge": 2}
    plot_df["stage_order"] = plot_df["prediction_time"].map(stage_order).fillna(99)
    plot_df = plot_df.sort_values(["cohort", "stage_order", "feature_set"], ascending=[True, True, True])
    colors = plot_df["cohort"].map(
        {"diabetes": "#4C78A8", "ckd": "#59A14F", "heart_failure": "#E15759", "hypertension": "#F28E2B"}
    ).fillna("#BAB0AC")

    plt.figure(figsize=(10, max(4.8, 0.55 * len(plot_df) + 1.5)))
    bars = plt.barh(plot_df["label"], plot_df[metric], color=colors)
    plt.xlim(max(0, plot_df[metric].min() - 0.03), min(1, plot_df[metric].max() + 0.04))
    plt.xlabel(metric)
    plt.title(f"Chronic disease prediction-time benchmark: {metric}")
    for bar, value in zip(bars, plot_df[metric]):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_report(
    project: Path,
    cohort: pd.DataFrame,
    prediction: pd.DataFrame,
    outcome: pd.DataFrame,
    leakage: pd.DataFrame,
    report_path: Path,
) -> None:
    cohort_cols = [
        "cohort",
        "status",
        "final_index_admissions",
        "final_subjects",
        "readmission_30d_count",
        "readmission_30d_rate",
    ]
    pred_cols = ["cohort", "feature_set", "prediction_time", "AUROC", "AUPRC", "Brier_score"]
    outcome_cols = ["cohort", "outcome_definition", "events", "event_rate"]
    leakage_cols = ["cohort", "scenario", "AUROC", "AUPRC", "sensitivity", "specificity"]

    text = f"""# ChronoEHR-Agent Chronic Disease Benchmark

这个报告把当前 registry 里的慢病 demo 合并成一个总览。它不是新的医学结论，而是 ChronoEHR-Agent 的工具化产物：同一套报告逻辑可以跨糖尿病、CKD、心衰，后续也可以扩展到高血压和其他慢病队列。

## Cohort Overview

{markdown_table(cohort, [col for col in cohort_cols if col in cohort.columns], pct_cols={"readmission_30d_rate"})}

## Prediction-Time Model Benchmark

{markdown_table(prediction, [col for col in pred_cols if col in prediction.columns])}

## Outcome Sensitivity

{markdown_table(outcome, [col for col in outcome_cols if col in outcome.columns], pct_cols={"event_rate"})}

## Leakage Sensitivity

{markdown_table(leakage, [col for col in leakage_cols if col in leakage.columns])}

## Generated Files

- `outputs/tables/chronic_disease_benchmark_cohort_summary.csv`
- `outputs/tables/chronic_disease_prediction_time_benchmark.csv`
- `outputs/tables/chronic_disease_outcome_sensitivity_summary.csv`
- `outputs/tables/chronic_disease_leakage_sensitivity_summary.csv`
- `outputs/figures/chronic_disease_prediction_time_auroc.png`
- `outputs/figures/chronic_disease_prediction_time_auprc.png`
"""
    report_path.write_text(text, encoding="utf-8")


def write_brief_report(
    cohort: pd.DataFrame,
    prediction: pd.DataFrame,
    outcome: pd.DataFrame,
    leakage: pd.DataFrame,
    brief_path: Path,
) -> None:
    cohort_lines = []
    for row in cohort.itertuples(index=False):
        cohort_lines.append(
            f"- {row.cohort}: {int(row.final_index_admissions):,} admissions, "
            f"{int(row.final_subjects):,} subjects, 30-day readmission {row.readmission_30d_rate:.2%}."
        )

    pred_lines = []
    for cohort_name, group in prediction.groupby("cohort", sort=False):
        tests = group.set_index("feature_set")
        if "admission_safe_minimal" not in tests.index:
            continue
        admission = tests.loc["admission_safe_minimal"]
        discharge_candidates = tests[tests["prediction_time"].eq("discharge")]
        best = group.sort_values("AUROC", ascending=False).iloc[0]
        pred_lines.append(
            f"- {cohort_name}: admission AUROC {admission['AUROC']:.4f}/AUPRC {admission['AUPRC']:.4f}; "
            f"best tested set `{best['feature_set']}` AUROC {best['AUROC']:.4f}/AUPRC {best['AUPRC']:.4f}."
        )
        if not discharge_candidates.empty:
            discharge = discharge_candidates.iloc[0]
            pred_lines.append(
                f"  Discharge-time minus admission AUROC {discharge['AUROC'] - admission['AUROC']:.4f}, "
                f"AUPRC {discharge['AUPRC'] - admission['AUPRC']:.4f}."
            )

    leakage_lines = []
    leaked = leakage[leakage["scenario"].eq("leaked_days_to_next_admission")]
    for row in leaked.itertuples(index=False):
        leakage_lines.append(
            f"- {row.cohort}: adding `days_to_next_admission` gives AUROC {row.AUROC:.4f} and AUPRC {row.AUPRC:.4f}."
        )

    outcome_lines = []
    for cohort_name, group in outcome.groupby("cohort", sort=False):
        all_cause = group[group["outcome_definition"].eq("all_cause_30d_readmission")]
        emerg = group[group["outcome_definition"].eq("emergency_urgent_30d_readmission")]
        if all_cause.empty or emerg.empty:
            continue
        outcome_lines.append(
            f"- {cohort_name}: all-cause {all_cause.iloc[0]['event_rate']:.2%}; "
            f"emergency/urgent proxy {emerg.iloc[0]['event_rate']:.2%}."
        )

    text = f"""# ChronoEHR-Agent Benchmark Brief

## Current Scope

{chr(10).join(cohort_lines)}

## Main Signal

{chr(10).join(pred_lines)}

## Leakage Demonstration

{chr(10).join(leakage_lines)}

## Outcome Sensitivity

{chr(10).join(outcome_lines)}

## Interpretation

- The project is no longer a single-cohort script: the registered chronic-disease cohorts now share reporting, leakage, outcome-sensitivity, and prediction-time modeling infrastructure.
- The strongest demo message remains simple and teachable: prediction time changes which variables are legal, and future information can make a model look unrealistically perfect.
- Next engineering step: add calibrated Random Forest or gradient-boosting baselines, then abstract diagnosis-code cohort building and feature-window extraction into reusable configuration.
"""
    brief_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    studies = registry.get("studies", [])

    tables = args.project_root / "outputs" / "tables"
    figures = args.project_root / "outputs" / "figures"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    cohort = collect_cohort_summary(args.project_root, studies)
    prediction = collect_prediction_time(args.project_root, studies)
    outcome = collect_outcome_sensitivity(args.project_root, studies)
    leakage = collect_leakage_sensitivity(args.project_root, studies)

    cohort.to_csv(tables / "chronic_disease_benchmark_cohort_summary.csv", index=False)
    prediction.to_csv(tables / "chronic_disease_prediction_time_benchmark.csv", index=False)
    outcome.to_csv(tables / "chronic_disease_outcome_sensitivity_summary.csv", index=False)
    leakage.to_csv(tables / "chronic_disease_leakage_sensitivity_summary.csv", index=False)

    plot_prediction_time(prediction, "AUROC", figures / "chronic_disease_prediction_time_auroc.png")
    plot_prediction_time(prediction, "AUPRC", figures / "chronic_disease_prediction_time_auprc.png")
    write_report(cohort=cohort, prediction=prediction, outcome=outcome, leakage=leakage, report_path=reports / "chronic_disease_benchmark_report.md", project=args.project_root)
    write_brief_report(
        cohort=cohort,
        prediction=prediction,
        outcome=outcome,
        leakage=leakage,
        brief_path=reports / "chronic_disease_benchmark_brief.md",
    )

    print("Chronic disease benchmark summary complete")
    print(f"studies={len(studies)} cohorts_with_summary={len(cohort)} prediction_rows={len(prediction)}")


if __name__ == "__main__":
    main()
