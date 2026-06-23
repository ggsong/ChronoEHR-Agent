#!/usr/bin/env python3
"""Build a unified external calibration and decision-curve comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


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


def calibration_rows(project_root: Path) -> pd.DataFrame:
    tables = project_root / "outputs" / "tables"
    frames: list[pd.DataFrame] = []

    cdsl = read_csv(tables / "cdsl_calibration_summary.csv")
    if not cdsl.empty:
        part = cdsl[cdsl["split"].astype(str).eq("test")].copy()
        part["dataset"] = "CDSL"
        part["calibration_method"] = "raw_traditional"
        frames.append(part)

    eicu = read_csv(tables / "eicu_probability_recalibration_summary.csv")
    if not eicu.empty:
        part = eicu[eicu["split"].astype(str).eq("test")].copy()
        part["dataset"] = "eICU"
        part["feature_set"] = "first24h_lab_vital"
        part["model"] = "logistic_regression_balanced"
        frames.append(part)

    charls = read_csv(tables / "charls_probability_recalibration_summary.csv")
    if not charls.empty:
        part = charls[charls["split"].astype(str).eq("test")].copy()
        part["dataset"] = "CHARLS"
        part["feature_set"] = "charls_2011_baseline"
        part["model"] = "logistic_regression_balanced"
        frames.append(part)

    comparison = read_csv(tables / "external_model_comparison_recalibration_summary.csv")
    if not comparison.empty:
        part = comparison[comparison["split"].astype(str).eq("test")].copy()
        part["feature_set"] = np.where(part["dataset"].astype(str).eq("eICU"), "first24h_lab_vital", "charls_2011_baseline")
        part["calibration_method"] = part["calibration_method"].replace({"raw": "raw_model_comparison"})
        frames.append(part)

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True, sort=False)
    table["calibration_method"] = [
        normalize_method(method, dataset) for method, dataset in zip(table["calibration_method"], table["dataset"].astype(str))
    ]
    keep = [
        "dataset",
        "feature_set",
        "model",
        "calibration_method",
        "split",
        "mean_absolute_calibration_error",
        "max_absolute_calibration_error",
        "deciles",
        "n",
        "events",
    ]
    return table[[col for col in keep if col in table.columns]].copy()


def decision_curve_rows(project_root: Path) -> pd.DataFrame:
    tables = project_root / "outputs" / "tables"
    frames: list[pd.DataFrame] = []

    cdsl = read_csv(tables / "cdsl_decision_curve.csv")
    if not cdsl.empty:
        part = cdsl[cdsl["split"].astype(str).eq("test")].copy()
        part["dataset"] = "CDSL"
        part["calibration_method"] = "raw_traditional"
        frames.append(part)

    eicu = read_csv(tables / "eicu_probability_recalibration_decision_curve.csv")
    if not eicu.empty:
        part = eicu[eicu["split"].astype(str).eq("test")].copy()
        part["dataset"] = "eICU"
        part["feature_set"] = "first24h_lab_vital"
        part["model"] = "logistic_regression_balanced"
        frames.append(part)

    charls = read_csv(tables / "charls_probability_recalibration_decision_curve.csv")
    if not charls.empty:
        part = charls[charls["split"].astype(str).eq("test")].copy()
        part["dataset"] = "CHARLS"
        part["feature_set"] = "charls_2011_baseline"
        part["model"] = "logistic_regression_balanced"
        frames.append(part)

    comparison = read_csv(tables / "external_model_comparison_recalibration_decision_curve.csv")
    if not comparison.empty:
        part = comparison[comparison["split"].astype(str).eq("test")].copy()
        part["feature_set"] = np.where(part["dataset"].astype(str).eq("eICU"), "first24h_lab_vital", "charls_2011_baseline")
        part["calibration_method"] = part["calibration_method"].replace({"raw": "raw_model_comparison"})
        frames.append(part)

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True, sort=False)
    table["calibration_method"] = [
        normalize_method(method, dataset) for method, dataset in zip(table["calibration_method"], table["dataset"].astype(str))
    ]
    return table


def summarize_decision_curves(decision: pd.DataFrame) -> pd.DataFrame:
    if decision.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "feature_set", "model", "calibration_method"]
    for key, group in decision.groupby(group_cols, dropna=False, sort=True):
        dataset, feature_set, model, method = key
        positive = group[group["net_benefit_advantage"].fillna(0).astype(float).gt(0)].copy()
        preferred = group[group["preferred_strategy"].astype(str).eq("model")].copy()
        best = group.sort_values(["net_benefit_advantage", "threshold_probability"], ascending=[False, True]).head(1)
        rows.append(
            {
                "dataset": dataset,
                "feature_set": feature_set,
                "model": model,
                "calibration_method": method,
                "decision_thresholds": int(len(group)),
                "decision_model_preferred_thresholds": int(len(preferred)),
                "decision_positive_advantage_thresholds": int(len(positive)),
                "decision_best_threshold": float(best["threshold_probability"].iloc[0]) if not best.empty else np.nan,
                "decision_best_net_benefit_advantage": float(best["net_benefit_advantage"].iloc[0]) if not best.empty else np.nan,
                "decision_best_preferred_strategy": str(best["preferred_strategy"].iloc[0]) if not best.empty else "",
                "decision_best_alert_rate": float(best["alert_rate"].iloc[0]) if not best.empty and "alert_rate" in best else np.nan,
                "decision_best_ppv": float(best["ppv"].iloc[0]) if not best.empty and "ppv" in best else np.nan,
                "decision_best_recall": float(best["recall"].iloc[0]) if not best.empty and "recall" in best else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_raw_comparison(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    raw_by_group: dict[tuple[str, str, str], float] = {}
    for key, group in table.groupby(["dataset", "feature_set", "model"], dropna=False, sort=True):
        dataset = str(key[0])
        raw_method = "raw_traditional" if dataset == "CDSL" else "raw_model_comparison"
        if dataset in {"eICU", "CHARLS"} and str(key[2]) == "logistic_regression_balanced":
            raw_method = "raw"
        raw = group[group["calibration_method"].astype(str).eq(raw_method)]
        raw_value = float(raw["mean_absolute_calibration_error"].iloc[0]) if not raw.empty else np.nan
        raw_by_group[key] = raw_value

    raw_values = []
    deltas = []
    for item in table.itertuples(index=False):
        key = (str(item.dataset), str(item.feature_set), str(item.model))
        raw_value = raw_by_group.get(key, np.nan)
        current = float(getattr(item, "mean_absolute_calibration_error"))
        raw_values.append(raw_value)
        deltas.append(current - raw_value if pd.notna(raw_value) else np.nan)
    table["raw_mean_absolute_calibration_error"] = raw_values
    table["calibration_mae_delta_vs_raw"] = deltas
    table["calibration_mae_improved_vs_raw"] = table["calibration_mae_delta_vs_raw"].lt(0)
    table["calibration_rank_within_model"] = (
        table.groupby(["dataset", "feature_set", "model"], dropna=False)["mean_absolute_calibration_error"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return table


def selected_row_lookup(project_root: Path) -> pd.DataFrame:
    selected = read_csv(project_root / "outputs" / "tables" / "external_benchmark_summary_table.csv")
    if selected.empty:
        return pd.DataFrame()
    selected = selected.copy()
    selected["calibration_method"] = [
        normalize_method(method, dataset) for method, dataset in zip(selected["calibration_method"], selected["dataset"].astype(str))
    ]
    selected["is_selected_technical_summary_row"] = True
    keep = ["dataset", "feature_set", "model", "calibration_method", "benchmark_row", "is_selected_technical_summary_row"]
    return selected[[col for col in keep if col in selected.columns]]


def interpretation_note(row: pd.Series) -> str:
    dataset = str(row.get("dataset", ""))
    method = str(row.get("calibration_method", ""))
    if dataset == "CDSL":
        if str(row.get("feature_set", "")) == "full_stay_naive_reference":
            return "CDSL full-stay row is a naive upper-reference, not early prediction performance."
        return "CDSL row uses raw traditional-model probabilities for a temporal-method benchmark."
    base = (
        "External ICU mortality benchmark, not chronic readmission validation."
        if dataset == "eICU"
        else "External longitudinal chronic-disease benchmark using CHARLS baseline features."
    )
    if method in {"intercept_validation", "platt_validation", "isotonic_validation"}:
        return base + " Calibration fitted on validation split."
    return base + " Raw model probability reference."


def build_summary(project_root: Path) -> pd.DataFrame:
    calibration = calibration_rows(project_root)
    if calibration.empty:
        raise FileNotFoundError("No calibration summary rows found.")
    decision = summarize_decision_curves(decision_curve_rows(project_root))
    table = calibration.merge(decision, on=["dataset", "feature_set", "model", "calibration_method"], how="left")
    table = add_raw_comparison(table)
    selected = selected_row_lookup(project_root)
    if not selected.empty:
        table = table.merge(selected, on=["dataset", "feature_set", "model", "calibration_method"], how="left")
    else:
        table["benchmark_row"] = ""
        table["is_selected_technical_summary_row"] = False
    table["benchmark_row"] = table["benchmark_row"].fillna("")
    table["is_selected_technical_summary_row"] = table["is_selected_technical_summary_row"].fillna(False).astype(bool)
    table["task"] = table["dataset"].map(dataset_task)
    table["event_rate"] = table["events"].astype(float) / table["n"].astype(float)
    table["interpretation_note"] = table.apply(interpretation_note, axis=1)
    columns = [
        "dataset",
        "task",
        "feature_set",
        "model",
        "calibration_method",
        "split",
        "n",
        "events",
        "event_rate",
        "mean_absolute_calibration_error",
        "max_absolute_calibration_error",
        "raw_mean_absolute_calibration_error",
        "calibration_mae_delta_vs_raw",
        "calibration_mae_improved_vs_raw",
        "calibration_rank_within_model",
        "deciles",
        "decision_thresholds",
        "decision_model_preferred_thresholds",
        "decision_positive_advantage_thresholds",
        "decision_best_threshold",
        "decision_best_net_benefit_advantage",
        "decision_best_preferred_strategy",
        "decision_best_alert_rate",
        "decision_best_ppv",
        "decision_best_recall",
        "is_selected_technical_summary_row",
        "benchmark_row",
        "interpretation_note",
    ]
    for column in columns:
        if column not in table:
            table[column] = np.nan
    return table[columns].sort_values(["dataset", "feature_set", "model", "calibration_rank_within_model", "calibration_method"])


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = [
        "dataset",
        "feature_set",
        "model",
        "calibration_method",
        "mean_absolute_calibration_error",
        "calibration_mae_delta_vs_raw",
        "calibration_rank_within_model",
        "decision_model_preferred_thresholds",
        "decision_thresholds",
        "decision_best_net_benefit_advantage",
        "benchmark_row",
    ]
    display = display[[col for col in columns if col in display.columns]].astype(object).where(pd.notna(display), "")
    lines = ["| " + " | ".join(display.columns) + " |", "|" + "|".join("---" for _ in display.columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, table: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_calibration_decision_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    selected = table[table["is_selected_technical_summary_row"].astype(bool)].copy()
    best_calibrated = table[table["calibration_rank_within_model"].eq(1)].copy()
    report.write_text(
        f"""# External Calibration and Decision-Curve Summary

- Boundary: research model evaluation only; no medical QA, diagnosis, treatment recommendation, or clinical deployment guidance.
- Scope: test-split calibration and decision-curve summaries for completed CDSL, eICU, and CHARLS external benchmark rows.
- Decision-curve fields summarize threshold-level net benefit and do not define clinical action thresholds.

## Selected Technical Rows

{markdown_table(selected)}

## Best Calibration Per Dataset / Feature Set / Model

{markdown_table(best_calibrated)}

## Coverage

- Calibration-decision rows: {len(table)}
- Datasets: {", ".join(sorted(table["dataset"].dropna().astype(str).unique()))}
- Selected technical summary rows represented: {int(table["is_selected_technical_summary_row"].sum())}
- Rows with decision-curve summaries: {int(table["decision_thresholds"].fillna(0).gt(0).sum())}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    table = build_summary(args.project_root)
    tables = args.project_root / "outputs" / "tables"
    supplement = tables / "supplementary_appendix"
    tables.mkdir(parents=True, exist_ok=True)
    supplement.mkdir(parents=True, exist_ok=True)
    table_path = tables / "external_calibration_decision_summary.csv"
    supp_path = supplement / "table_s16_external_calibration_decision_summary.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(supp_path, index=False)
    report = write_report(args.project_root, table)
    print(f"Wrote {report}")
    print(f"Wrote {table_path}")
    print(f"Wrote {supp_path}")


if __name__ == "__main__":
    main()
