#!/usr/bin/env python3
"""Build unified external benchmark hard-metric summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


MODEL_COMPARISON_MODELS = {"random_forest_balanced", "hist_gradient_boosting_weighted"}
CALIBRATED_METHODS = {"intercept_validation", "platt_validation", "isotonic_validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def metric_value(summary: pd.DataFrame, metric: str) -> float | None:
    if summary.empty or not {"metric", "value"}.issubset(summary.columns):
        return None
    row = summary[summary["metric"].astype(str).eq(metric)]
    return float(row["value"].iloc[0]) if not row.empty else None


def normalize_method(value: object, dataset: str) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "raw_traditional" if dataset == "CDSL" else "raw"
    return str(value)


def dataset_task(dataset: str) -> str:
    return {
        "CDSL": "COVID hospitalization mortality prediction",
        "eICU": "ICU first-24h hospital mortality prediction",
        "CHARLS": "2011 baseline prediction of incident diabetes by 2013/2015",
    }.get(dataset, "")


def dataset_role(dataset: str) -> str:
    return {
        "CDSL": "external structured EHR temporal-method benchmark",
        "eICU": "external multicenter ICU EHR benchmark",
        "CHARLS": "external longitudinal chronic-disease benchmark",
    }.get(dataset, "")


def feature_window(row: pd.Series) -> str:
    dataset = str(row.get("dataset", ""))
    feature_set = str(row.get("feature_set", ""))
    if dataset == "eICU":
        return "ICU admission to 24h"
    if dataset == "CHARLS":
        return "CHARLS 2011 wave baseline"
    return feature_set


def leakage_status(project_root: Path, dataset: str) -> tuple[str, int | None]:
    if dataset == "CDSL":
        audit = read_csv(project_root / "outputs" / "tables" / "cdsl_leakage_audit.csv")
        if audit.empty:
            return "UNKNOWN", None
        critical_failures = int(
            (audit["severity"].astype(str).str.lower().eq("critical") & audit["status"].astype(str).str.upper().ne("PASS")).sum()
        ) if {"severity", "status"}.issubset(audit.columns) else 0
        warning_count = int(audit["status"].astype(str).str.upper().eq("WARNING").sum()) if "status" in audit else 0
        status = "PASS" if critical_failures == 0 and warning_count == 0 else ("PASS_WITH_WARNINGS" if critical_failures == 0 else "REVIEW")
        return status, critical_failures
    if dataset == "eICU":
        leakage = read_csv(project_root / "outputs" / "tables" / "eicu_leakage_gate.csv")
        blocked = int((leakage["status"].astype(str).str.upper() == "BLOCK").sum()) if not leakage.empty and "status" in leakage else 0
        return ("PASS" if blocked == 0 else "REVIEW"), blocked
    if dataset == "CHARLS":
        leakage = read_csv(project_root / "outputs" / "tables" / "charls_leakage_gate.csv")
        blocked = int(leakage["status"].astype(str).str.lower().eq("blocked").sum()) if not leakage.empty and "status" in leakage else 0
        return ("PASS" if blocked == 0 else "REVIEW"), blocked
    return "UNKNOWN", None


def subject_counts(project_root: Path, dataset: str) -> tuple[int | None, int | None]:
    if dataset == "CDSL":
        summary = read_csv(project_root / "outputs" / "tables" / "cdsl_external_validation_summary.csv")
        patients = int(summary["patients"].iloc[0]) if not summary.empty and "patients" in summary else None
        rows = int(summary["formatted_rows"].iloc[0]) if not summary.empty and "formatted_rows" in summary else None
        return patients, rows
    if dataset == "eICU":
        summary = read_csv(project_root / "outputs" / "tables" / "eicu_temporal_mortality_cohort_summary.csv")
        patients = metric_value(summary, "patients")
        stays = metric_value(summary, "first_24h_eligible_stays")
        return int(patients) if patients is not None else None, int(stays) if stays is not None else None
    if dataset == "CHARLS":
        summary = read_csv(project_root / "outputs" / "tables" / "charls_incident_diabetes_cohort_summary.csv")
        if summary.empty:
            return None, None
        candidates = [col for col in ["persons", "participants", "n", "eligible_baseline_persons"] if col in summary.columns]
        value = int(summary[candidates[0]].iloc[0]) if candidates else None
        return value, value
    return None, None


def feature_counts(project_root: Path) -> pd.DataFrame:
    frames = []
    cdsl = read_csv(project_root / "outputs" / "tables" / "cdsl_traditional_baselines_metrics.csv")
    if not cdsl.empty and {"feature_set", "model", "split", "feature_count"}.issubset(cdsl.columns):
        part = cdsl[cdsl["split"].astype(str).eq("test")][["feature_set", "model", "feature_count"]].copy()
        part["dataset"] = "CDSL"
        frames.append(part)
    eicu = read_csv(project_root / "outputs" / "tables" / "eicu_first24h_model_comparison_metrics.csv")
    if not eicu.empty and {"feature_set", "model", "split", "feature_count"}.issubset(eicu.columns):
        part = eicu[eicu["split"].astype(str).eq("test")][["feature_set", "model", "feature_count"]].copy()
        part["dataset"] = "eICU"
        frames.append(part)
    charls = read_csv(project_root / "outputs" / "tables" / "charls_incident_diabetes_model_comparison_metrics.csv")
    if not charls.empty and {"feature_set", "model", "split", "feature_count"}.issubset(charls.columns):
        part = charls[charls["split"].astype(str).eq("test")][["feature_set", "model", "feature_count"]].copy()
        part["dataset"] = "CHARLS"
        frames.append(part)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def calibration_lookup(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    cdsl = read_csv(project_root / "outputs" / "tables" / "cdsl_calibration_summary.csv")
    if not cdsl.empty:
        for item in cdsl[cdsl["split"].astype(str).eq("test")].to_dict("records"):
            rows.append(
                {
                    "dataset": "CDSL",
                    "feature_set": item.get("feature_set", ""),
                    "model": item.get("model", ""),
                    "calibration_method": "raw_traditional",
                    "mean_absolute_calibration_error": item.get("mean_absolute_calibration_error", np.nan),
                    "max_absolute_calibration_error": item.get("max_absolute_calibration_error", np.nan),
                    "calibration_deciles": item.get("deciles", np.nan),
                }
            )

    for dataset, path in [
        ("eICU", project_root / "outputs" / "tables" / "eicu_probability_recalibration_summary.csv"),
        ("CHARLS", project_root / "outputs" / "tables" / "charls_probability_recalibration_summary.csv"),
    ]:
        df = read_csv(path)
        if not df.empty:
            for item in df[df["split"].astype(str).eq("test")].to_dict("records"):
                rows.append(
                    {
                        "dataset": dataset,
                        "feature_set": "first24h_lab_vital" if dataset == "eICU" else "charls_2011_baseline",
                        "model": "logistic_regression_balanced",
                        "calibration_method": item.get("calibration_method", ""),
                        "mean_absolute_calibration_error": item.get("mean_absolute_calibration_error", np.nan),
                        "max_absolute_calibration_error": item.get("max_absolute_calibration_error", np.nan),
                        "calibration_deciles": item.get("deciles", np.nan),
                    }
                )

    comparison = read_csv(project_root / "outputs" / "tables" / "external_model_comparison_recalibration_summary.csv")
    if not comparison.empty:
        for item in comparison[comparison["split"].astype(str).eq("test")].to_dict("records"):
            rows.append(
                {
                    "dataset": item.get("dataset", ""),
                    "feature_set": "first24h_lab_vital" if item.get("dataset", "") == "eICU" else "charls_2011_baseline",
                    "model": item.get("model", ""),
                    "calibration_method": "raw_model_comparison" if item.get("calibration_method") == "raw" else item.get("calibration_method", ""),
                    "mean_absolute_calibration_error": item.get("mean_absolute_calibration_error", np.nan),
                    "max_absolute_calibration_error": item.get("max_absolute_calibration_error", np.nan),
                    "calibration_deciles": item.get("deciles", np.nan),
                }
            )
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def subgroup_summary(project_root: Path) -> pd.DataFrame:
    table = read_csv(project_root / "outputs" / "tables" / "external_subgroup_performance.csv")
    if table.empty:
        return pd.DataFrame()
    table = table.copy()
    table["calibration_method"] = [
        normalize_method(value, dataset) for value, dataset in zip(table.get("calibration_method", ""), table["dataset"].astype(str))
    ]
    rows = []
    group_cols = ["dataset", "feature_set", "model", "calibration_method"]
    for key, group in table.groupby(group_cols, dropna=False, sort=True):
        dataset, feature_set, model, method = key
        rows.append(
            {
                "dataset": dataset,
                "feature_set": feature_set,
                "model": model,
                "calibration_method": method,
                "subgroup_rows": int(len(group)),
                "subgroup_ok_rows": int(group["status"].astype(str).eq("OK").sum()) if "status" in group else 0,
                "subgroup_small_or_single_class_rows": int(group["status"].astype(str).ne("OK").sum()) if "status" in group else 0,
                "subgroup_types": ",".join(sorted(set(group["subgroup_type"].dropna().astype(str)))) if "subgroup_type" in group else "",
            }
        )
    return pd.DataFrame(rows)


def build_hard_metrics(project_root: Path) -> pd.DataFrame:
    ci = read_csv(project_root / "outputs" / "tables" / "external_model_bootstrap_ci.csv")
    if ci.empty:
        raise FileNotFoundError("Missing external_model_bootstrap_ci.csv")
    table = ci.copy()
    table["calibration_method"] = [normalize_method(value, dataset) for value, dataset in zip(table.get("calibration_method", ""), table["dataset"].astype(str))]
    table["prediction_time"] = table["prediction_time"].fillna(table["feature_set"])
    table["task"] = table["dataset"].map(dataset_task)
    table["role"] = table["dataset"].map(dataset_role)
    table["feature_window"] = table.apply(feature_window, axis=1)

    counts = {dataset: subject_counts(project_root, dataset) for dataset in table["dataset"].dropna().astype(str).unique()}
    statuses = {dataset: leakage_status(project_root, dataset) for dataset in table["dataset"].dropna().astype(str).unique()}
    table["subjects_or_patients"] = table["dataset"].map(lambda dataset: counts.get(str(dataset), (None, None))[0])
    table["records_or_stays"] = table["dataset"].map(lambda dataset: counts.get(str(dataset), (None, None))[1])
    table["leakage_gate_status"] = table["dataset"].map(lambda dataset: statuses.get(str(dataset), ("UNKNOWN", None))[0])
    table["blocked_leakage_checks"] = table["dataset"].map(lambda dataset: statuses.get(str(dataset), ("UNKNOWN", None))[1])

    features = feature_counts(project_root)
    if not features.empty:
        table = table.merge(features, on=["dataset", "feature_set", "model"], how="left")
    else:
        table["feature_count"] = np.nan

    calibration = calibration_lookup(project_root)
    if not calibration.empty:
        table = table.merge(calibration, on=["dataset", "feature_set", "model", "calibration_method"], how="left")
    else:
        table["mean_absolute_calibration_error"] = np.nan
        table["max_absolute_calibration_error"] = np.nan
        table["calibration_deciles"] = np.nan

    subgroups = subgroup_summary(project_root)
    if not subgroups.empty:
        table = table.merge(subgroups, on=["dataset", "feature_set", "model", "calibration_method"], how="left")
    else:
        table["subgroup_rows"] = np.nan
        table["subgroup_ok_rows"] = np.nan
        table["subgroup_small_or_single_class_rows"] = np.nan
        table["subgroup_types"] = ""

    table["is_prediction_time_valid"] = ~(
        table["dataset"].astype(str).eq("CDSL") & table["feature_set"].astype(str).eq("full_stay_naive_reference")
    )
    table["interpretation_note"] = table.apply(interpretation_note, axis=1)
    columns = [
        "dataset",
        "role",
        "task",
        "prediction_time",
        "feature_window",
        "feature_set",
        "model",
        "calibration_method",
        "n",
        "events",
        "event_rate",
        "AUROC",
        "AUROC_lower",
        "AUROC_upper",
        "AUPRC",
        "AUPRC_lower",
        "AUPRC_upper",
        "Brier",
        "Brier_lower",
        "Brier_upper",
        "mean_absolute_calibration_error",
        "max_absolute_calibration_error",
        "calibration_deciles",
        "subgroup_rows",
        "subgroup_ok_rows",
        "subgroup_small_or_single_class_rows",
        "subgroup_types",
        "feature_count",
        "subjects_or_patients",
        "records_or_stays",
        "leakage_gate_status",
        "blocked_leakage_checks",
        "is_prediction_time_valid",
        "interpretation_note",
    ]
    return table[columns].sort_values(["dataset", "is_prediction_time_valid", "feature_set", "model", "calibration_method"], ascending=[True, False, True, True, True])


def interpretation_note(row: pd.Series) -> str:
    dataset = str(row.get("dataset", ""))
    method = str(row.get("calibration_method", ""))
    if dataset == "CDSL" and str(row.get("feature_set", "")) == "full_stay_naive_reference":
        return "Naive upper-reference only; should not be interpreted as early prediction performance."
    if dataset == "CDSL":
        return "Prediction-time comparison row; excludes full-stay future information."
    if dataset == "eICU":
        base = "External ICU benchmark; not a chronic readmission external validation cohort."
    elif dataset == "CHARLS":
        base = "External longitudinal chronic-disease benchmark using CHARLS 2011 baseline features."
    else:
        base = "External research benchmark row."
    if method in CALIBRATED_METHODS:
        return base + " Validation-set calibrated probability row."
    if method == "raw_model_comparison":
        return base + " Raw RF/HGB comparison probability row."
    return base


def selected_rows(hard: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def pick(frame: pd.DataFrame, label: str, reason: str) -> None:
        if frame.empty:
            return
        selected = frame.sort_values(["AUPRC", "AUROC", "Brier"], ascending=[False, False, True]).head(1).copy()
        selected["benchmark_row"] = label
        selected["selection_reason"] = reason
        rows.append(selected)

    pick(
        hard[hard["dataset"].eq("CDSL") & hard["is_prediction_time_valid"].astype(bool)],
        "CDSL early-window best",
        "Best test AUPRC among prediction-time CDSL rows.",
    )
    pick(
        hard[hard["dataset"].eq("CDSL") & ~hard["is_prediction_time_valid"].astype(bool)],
        "CDSL full-stay naive reference",
        "Best test AUPRC among full-stay rows; upper reference only.",
    )
    pick(
        hard[
            hard["dataset"].eq("eICU")
            & hard["model"].eq("logistic_regression_balanced")
            & hard["calibration_method"].eq("platt_validation")
        ],
        "eICU calibrated logistic reference",
        "Primary calibrated logistic reference row.",
    )
    pick(
        hard[
            hard["dataset"].eq("eICU")
            & hard["model"].isin(MODEL_COMPARISON_MODELS)
            & hard["calibration_method"].isin(CALIBRATED_METHODS)
        ],
        "eICU best calibrated RF/HGB",
        "Best test AUPRC among calibrated RF/HGB rows.",
    )
    pick(
        hard[
            hard["dataset"].eq("CHARLS")
            & hard["model"].eq("logistic_regression_balanced")
            & hard["calibration_method"].eq("platt_validation")
        ],
        "CHARLS calibrated logistic reference",
        "Primary calibrated logistic reference row.",
    )
    pick(
        hard[
            hard["dataset"].eq("CHARLS")
            & hard["model"].isin(MODEL_COMPARISON_MODELS)
            & hard["calibration_method"].isin(CALIBRATED_METHODS)
        ],
        "CHARLS best calibrated RF/HGB",
        "Best test AUPRC among calibrated RF/HGB rows.",
    )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for column in ["event_rate", "AUROC", "AUROC_lower", "AUROC_upper", "AUPRC", "AUPRC_lower", "AUPRC_upper", "Brier", "Brier_lower", "Brier_upper", "mean_absolute_calibration_error"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = [
        "dataset",
        "benchmark_row",
        "task",
        "model",
        "calibration_method",
        "n",
        "events",
        "AUROC",
        "AUROC_lower",
        "AUROC_upper",
        "AUPRC",
        "AUPRC_lower",
        "AUPRC_upper",
        "Brier",
        "mean_absolute_calibration_error",
        "subgroup_ok_rows",
        "selection_reason",
    ]
    existing = [col for col in columns if col in display.columns]
    display = display[existing].astype(object).where(pd.notna(display[existing]), "")
    lines = ["| " + " | ".join(existing) + " |", "|" + "|".join("---" for _ in existing) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, summary: pd.DataFrame, hard: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_benchmark_summary_table.md"
    text = f"""# External Benchmark Hard-Metric Summary

This table summarizes completed external method benchmarks for ChronoEHR-Agent. It is a research-method summary, not a clinical decision-support report.

## Selected Rows

{markdown_table(summary)}

## Coverage

- Hard-metric rows: {len(hard)}
- Datasets: {", ".join(sorted(hard["dataset"].dropna().astype(str).unique()))}
- Bootstrap CI: included for AUROC, AUPRC, and Brier.
- Calibration: included when decile summaries are available.
- Subgroups: included as row counts and evaluable subgroup counts.
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    hard = build_hard_metrics(args.project_root)
    summary = selected_rows(hard)
    if summary.empty:
        raise SystemExit("No external benchmark summary rows found. Run external benchmark commands first.")
    tables = args.project_root / "outputs" / "tables"
    supplement = tables / "supplementary_appendix"
    tables.mkdir(parents=True, exist_ok=True)
    supplement.mkdir(parents=True, exist_ok=True)
    table_path = tables / "external_benchmark_summary_table.csv"
    hard_path = tables / "external_benchmark_hard_metrics_table.csv"
    supp_path = supplement / "table_s13_external_benchmark_summary.csv"
    hard_supp_path = supplement / "table_s14_external_benchmark_hard_metrics.csv"
    summary.to_csv(table_path, index=False)
    hard.to_csv(hard_path, index=False)
    summary.to_csv(supp_path, index=False)
    hard.to_csv(hard_supp_path, index=False)
    report = write_report(args.project_root, summary, hard)
    print(f"Wrote {report}")
    print(f"Wrote {table_path}")
    print(f"Wrote {hard_path}")
    print(f"Wrote {supp_path}")
    print(f"Wrote {hard_supp_path}")


if __name__ == "__main__":
    main()
