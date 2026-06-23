#!/usr/bin/env python3
"""Build a mentor-facing technical summary from external hard metrics."""

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
    value_str = str(value)
    return "raw_model_comparison" if value_str == "raw" and dataset in {"eICU", "CHARLS"} else value_str


def metric_ci(value: object, lower: object, upper: object) -> str:
    if pd.isna(value):
        return ""
    if pd.isna(lower) or pd.isna(upper):
        return f"{float(value):.4f}"
    return f"{float(value):.4f} ({float(lower):.4f}-{float(upper):.4f})"


def row_key(df: pd.DataFrame) -> pd.Series:
    normalized = df.copy()
    for col in ["dataset", "feature_set", "model", "calibration_method"]:
        if col not in normalized:
            normalized[col] = ""
        normalized[col] = normalized[col].fillna("").astype(str)
    return normalized[["dataset", "feature_set", "model", "calibration_method"]].agg("||".join, axis=1)


def subgroup_ci_summary(project_root: Path) -> pd.DataFrame:
    table = read_csv(project_root / "outputs" / "tables" / "external_subgroup_bootstrap_ci.csv")
    if table.empty:
        return pd.DataFrame()
    table = table.copy()
    table["calibration_method"] = [
        normalize_method(method, dataset) for method, dataset in zip(table.get("calibration_method", ""), table["dataset"].astype(str))
    ]
    rows = []
    for key, group in table.groupby(["dataset", "feature_set", "model", "calibration_method"], dropna=False, sort=True):
        dataset, feature_set, model, method = key
        ok = group[group["status"].astype(str).eq("OK")].copy()
        small = group[~group["status"].astype(str).eq("OK")].copy()
        rows.append(
            {
                "dataset": dataset,
                "feature_set": feature_set,
                "model": model,
                "calibration_method": method,
                "subgroup_ci_rows": int(len(group)),
                "subgroup_ci_ok_rows": int(len(ok)),
                "subgroup_ci_small_or_single_class_rows": int(len(small)),
                "subgroup_ci_types": ",".join(sorted(set(group["subgroup_type"].dropna().astype(str)))) if "subgroup_type" in group else "",
                "subgroup_ci_min_auroc": float(ok["AUROC"].min()) if not ok.empty and "AUROC" in ok else np.nan,
                "subgroup_ci_min_auroc_lower": float(ok["AUROC_lower"].min()) if not ok.empty and "AUROC_lower" in ok else np.nan,
                "subgroup_ci_min_auprc": float(ok["AUPRC"].min()) if not ok.empty and "AUPRC" in ok else np.nan,
                "subgroup_ci_min_auprc_lower": float(ok["AUPRC_lower"].min()) if not ok.empty and "AUPRC_lower" in ok else np.nan,
                "subgroup_ci_min_replicates": int(ok["bootstrap_replicates"].min()) if not ok.empty and "bootstrap_replicates" in ok else 0,
            }
        )
    return pd.DataFrame(rows)


def decision_curve_frames(project_root: Path) -> pd.DataFrame:
    tables = project_root / "outputs" / "tables"
    frames = []

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
        part["calibration_method"] = [
            normalize_method(method, dataset) for method, dataset in zip(part["calibration_method"], part["dataset"].astype(str))
        ]
        frames.append(part)

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table["calibration_method"] = [
        normalize_method(method, dataset) for method, dataset in zip(table["calibration_method"], table["dataset"].astype(str))
    ]
    return table


def decision_curve_summary(project_root: Path) -> pd.DataFrame:
    table = decision_curve_frames(project_root)
    if table.empty:
        return pd.DataFrame()
    rows = []
    for key, group in table.groupby(["dataset", "feature_set", "model", "calibration_method"], dropna=False, sort=True):
        dataset, feature_set, model, method = key
        model_preferred = group[group["preferred_strategy"].astype(str).eq("model")].copy()
        positive = group[group["net_benefit_advantage"].fillna(0).astype(float).gt(0)].copy()
        best = group.sort_values(["net_benefit_advantage", "threshold_probability"], ascending=[False, True]).head(1)
        rows.append(
            {
                "dataset": dataset,
                "feature_set": feature_set,
                "model": model,
                "calibration_method": method,
                "decision_thresholds": int(len(group)),
                "decision_model_preferred_thresholds": int(len(model_preferred)),
                "decision_positive_advantage_thresholds": int(len(positive)),
                "decision_best_threshold": float(best["threshold_probability"].iloc[0]) if not best.empty else np.nan,
                "decision_best_net_benefit_advantage": float(best["net_benefit_advantage"].iloc[0]) if not best.empty else np.nan,
                "decision_best_preferred_strategy": str(best["preferred_strategy"].iloc[0]) if not best.empty else "",
            }
        )
    return pd.DataFrame(rows)


def comparison_notes(summary: pd.DataFrame) -> dict[str, str]:
    notes: dict[str, str] = {}

    def get(label: str) -> pd.Series | None:
        rows = summary[summary["benchmark_row"].astype(str).eq(label)]
        return rows.iloc[0] if not rows.empty else None

    eicu_logistic = get("eICU calibrated logistic reference")
    eicu_best = get("eICU best calibrated RF/HGB")
    if eicu_logistic is not None and eicu_best is not None:
        notes["eICU best calibrated RF/HGB"] = (
            f"Compared with calibrated logistic, AUPRC delta={float(eicu_best['AUPRC']) - float(eicu_logistic['AUPRC']):.4f}; "
            f"Brier delta={float(eicu_best['Brier']) - float(eicu_logistic['Brier']):.4f}."
        )
        notes["eICU calibrated logistic reference"] = "Reference calibrated logistic row for eICU RF/HGB comparison."

    charls_logistic = get("CHARLS calibrated logistic reference")
    charls_best = get("CHARLS best calibrated RF/HGB")
    if charls_logistic is not None and charls_best is not None:
        notes["CHARLS best calibrated RF/HGB"] = (
            f"Compared with calibrated logistic, AUPRC delta={float(charls_best['AUPRC']) - float(charls_logistic['AUPRC']):.4f}; "
            f"Brier delta={float(charls_best['Brier']) - float(charls_logistic['Brier']):.4f}."
        )
        notes["CHARLS calibrated logistic reference"] = "Reference calibrated logistic row for CHARLS RF/HGB comparison."

    cdsl_early = get("CDSL early-window best")
    cdsl_full = get("CDSL full-stay naive reference")
    if cdsl_early is not None and cdsl_full is not None:
        notes["CDSL early-window best"] = "Best prediction-time CDSL row; excludes full-stay future information."
        notes["CDSL full-stay naive reference"] = (
            f"Naive upper reference; AUPRC exceeds early-window best by {float(cdsl_full['AUPRC']) - float(cdsl_early['AUPRC']):.4f}."
        )
    return notes


def build_summary(project_root: Path) -> pd.DataFrame:
    summary = read_csv(project_root / "outputs" / "tables" / "external_benchmark_summary_table.csv")
    if summary.empty:
        raise FileNotFoundError("Missing external_benchmark_summary_table.csv")
    summary = summary.copy()
    summary["calibration_method"] = [
        normalize_method(method, dataset) for method, dataset in zip(summary["calibration_method"], summary["dataset"].astype(str))
    ]
    subgroup_ci = subgroup_ci_summary(project_root)
    decision = decision_curve_summary(project_root)

    table = summary.copy()
    if not subgroup_ci.empty:
        table["_key"] = row_key(table)
        subgroup_ci["_key"] = row_key(subgroup_ci)
        table = table.merge(subgroup_ci.drop(columns=["dataset", "feature_set", "model", "calibration_method"]), on="_key", how="left")
    if not decision.empty:
        table["_key"] = row_key(table)
        decision["_key"] = row_key(decision)
        table = table.merge(decision.drop(columns=["dataset", "feature_set", "model", "calibration_method"]), on="_key", how="left")
    table = table.drop(columns=[col for col in ["_key"] if col in table.columns])

    notes = comparison_notes(summary)
    table["model_selection_note"] = table["benchmark_row"].map(notes).fillna(table["selection_reason"])
    table["auroc_ci"] = table.apply(lambda row: metric_ci(row["AUROC"], row["AUROC_lower"], row["AUROC_upper"]), axis=1)
    table["auprc_ci"] = table.apply(lambda row: metric_ci(row["AUPRC"], row["AUPRC_lower"], row["AUPRC_upper"]), axis=1)
    table["brier_ci"] = table.apply(lambda row: metric_ci(row["Brier"], row["Brier_lower"], row["Brier_upper"]), axis=1)
    table["subgroup_ci_note"] = table.apply(
        lambda row: (
            f"{int(row.get('subgroup_ci_ok_rows', 0) or 0)} evaluable subgroup CI rows"
            + (
                f"; {int(row.get('subgroup_ci_small_or_single_class_rows', 0) or 0)} small/single-class rows"
                if int(row.get("subgroup_ci_small_or_single_class_rows", 0) or 0) > 0
                else ""
            )
        ),
        axis=1,
    )
    table["decision_curve_note"] = table.apply(
        lambda row: (
            f"{int(row.get('decision_model_preferred_thresholds', 0) or 0)}/{int(row.get('decision_thresholds', 0) or 0)} test thresholds prefer model; "
            f"best advantage {float(row.get('decision_best_net_benefit_advantage', np.nan)):.4f} at threshold {float(row.get('decision_best_threshold', np.nan)):.2f}"
        )
        if pd.notna(row.get("decision_thresholds", np.nan))
        else "decision curve not available",
        axis=1,
    )

    columns = [
        "benchmark_row",
        "dataset",
        "role",
        "task",
        "feature_window",
        "model",
        "calibration_method",
        "n",
        "events",
        "event_rate",
        "auroc_ci",
        "auprc_ci",
        "brier_ci",
        "mean_absolute_calibration_error",
        "subgroup_ci_rows",
        "subgroup_ci_ok_rows",
        "subgroup_ci_small_or_single_class_rows",
        "subgroup_ci_min_auroc",
        "subgroup_ci_min_auroc_lower",
        "subgroup_ci_min_auprc",
        "subgroup_ci_min_auprc_lower",
        "subgroup_ci_min_replicates",
        "decision_thresholds",
        "decision_model_preferred_thresholds",
        "decision_positive_advantage_thresholds",
        "decision_best_threshold",
        "decision_best_net_benefit_advantage",
        "leakage_gate_status",
        "is_prediction_time_valid",
        "model_selection_note",
        "subgroup_ci_note",
        "decision_curve_note",
        "interpretation_note",
    ]
    for column in columns:
        if column not in table:
            table[column] = np.nan
    return table[columns]


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for column in display.select_dtypes(include=[float]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = [
        "benchmark_row",
        "dataset",
        "model",
        "calibration_method",
        "n",
        "events",
        "auroc_ci",
        "auprc_ci",
        "brier_ci",
        "mean_absolute_calibration_error",
        "subgroup_ci_note",
        "decision_curve_note",
        "model_selection_note",
    ]
    existing = [col for col in columns if col in display.columns]
    display = display[existing].astype(object).where(pd.notna(display[existing]), "")
    lines = ["| " + " | ".join(existing) + " |", "|" + "|".join("---" for _ in existing) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, table: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_technical_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# External Technical Summary

- Boundary: research model evaluation only; not medical QA, diagnosis, treatment recommendation, or clinical deployment guidance.
- Scope: completed CDSL, eICU, and CHARLS external benchmark artifacts.
- Inputs: hard-metric summary, bootstrap CI, subgroup bootstrap CI, calibration summaries, and test-split decision curves.
- Interpretation guardrails: CDSL full-stay rows are naive upper references; eICU is an ICU mortality benchmark, not chronic readmission external validation; CHARLS is a longitudinal chronic-disease cohort extension.

## Technical Summary Table

{markdown_table(table)}

## Coverage

- Benchmark rows: {len(table)}
- Datasets: {", ".join(sorted(table["dataset"].dropna().astype(str).unique()))}
- Rows with subgroup bootstrap CI: {int(table["subgroup_ci_ok_rows"].fillna(0).gt(0).sum())}
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
    table_path = tables / "external_technical_summary_table.csv"
    supp_path = supplement / "table_s15_external_technical_summary.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(supp_path, index=False)
    report = write_report(args.project_root, table)
    print(f"Wrote {report}")
    print(f"Wrote {table_path}")
    print(f"Wrote {supp_path}")


if __name__ == "__main__":
    main()
