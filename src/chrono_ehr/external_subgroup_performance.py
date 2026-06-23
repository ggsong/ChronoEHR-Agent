#!/usr/bin/env python3
"""Summarize external benchmark subgroup performance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def safe_metrics(y: pd.Series, score: pd.Series) -> dict[str, float | int | str]:
    y_int = y.astype(int)
    score_float = score.astype(float)
    has_two_classes = y_int.nunique() == 2
    return {
        "n": int(len(y_int)),
        "events": int(y_int.sum()),
        "event_rate": float(y_int.mean()) if len(y_int) else np.nan,
        "AUROC": float(roc_auc_score(y_int, score_float)) if has_two_classes else np.nan,
        "AUPRC": float(average_precision_score(y_int, score_float)) if has_two_classes else np.nan,
        "Brier": float(brier_score_loss(y_int, score_float)) if len(y_int) else np.nan,
        "status": "OK" if len(y_int) >= 30 and has_two_classes else "SMALL_OR_SINGLE_CLASS",
    }


def age_group(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True, right=False).astype(str)


def rows_for_groups(
    dataset: str,
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
    model_cols: list[str],
    subgroup_cols: list[tuple[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_keys, model_group in frame.groupby(model_cols, sort=True):
        key_values = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        model_context = {col: value for col, value in zip(model_cols, key_values)}
        for subgroup_type, subgroup_col in subgroup_cols:
            for subgroup_value, part in model_group.groupby(subgroup_col, dropna=False, sort=True):
                if str(subgroup_value) in {"nan", "None", "<NA>"}:
                    continue
                row = {"dataset": dataset, "subgroup_type": subgroup_type, "subgroup": str(subgroup_value), **model_context}
                row.update(safe_metrics(part[label_col], part[score_col]))
                rows.append(row)
    return rows


def cdsl_rows(project_root: Path) -> list[dict[str, object]]:
    pred_path = project_root / "outputs" / "tables" / "cdsl_traditional_baselines_predictions.csv"
    demo_path = project_root / "data" / "processed" / "cdsl_temporal_benchmark" / "admission_demographics.csv"
    if not pred_path.exists() or not demo_path.exists():
        return []
    predictions = pd.read_csv(pred_path)
    demographics = pd.read_csv(demo_path, usecols=["PatientID", "Sex", "Age"])
    frame = predictions[predictions["split"].astype(str).eq("test")].merge(demographics, on="PatientID", how="left")
    frame["sex_group"] = frame["Sex"].map({0.0: "sex_0", 1.0: "sex_1"}).fillna("sex_unknown")
    frame["age_group"] = age_group(frame["Age"], [0, 60, 75, 200], ["age_lt_60", "age_60_74", "age_ge_75"])
    return rows_for_groups(
        "CDSL",
        frame,
        "outcome",
        "predicted_risk",
        ["feature_set", "model"],
        [("sex", "sex_group"), ("age", "age_group")],
    )


def eicu_rows(project_root: Path) -> list[dict[str, object]]:
    pred_path = project_root / "outputs" / "tables" / "eicu_probability_recalibration_predictions.csv"
    model_comparison_path = project_root / "outputs" / "tables" / "eicu_first24h_model_comparison_predictions.csv"
    comparison_recalibration_path = project_root / "outputs" / "tables" / "external_model_comparison_recalibration_predictions.csv"
    cohort_path = project_root / "data" / "processed" / "eicu_temporal_mortality_cohort.csv"
    if not cohort_path.exists():
        return []
    cohort = pd.read_csv(cohort_path, usecols=["stay_id", "age_years", "gender"])
    cohort["row_id"] = cohort["stay_id"].astype(str)
    rows: list[dict[str, object]] = []

    if pred_path.exists():
        predictions = pd.read_csv(pred_path)
        predictions = predictions[
            predictions["split"].astype(str).eq("test")
            & predictions["calibration_method"].astype(str).eq("platt_validation")
        ].copy()
        frame = predictions.merge(cohort, on="stay_id", how="left")
        frame["gender_group"] = frame["gender"].astype(str).str.lower().replace({"nan": "unknown", "": "unknown"})
        frame["age_group"] = age_group(frame["age_years"], [18, 50, 65, 80, 200], ["age_18_49", "age_50_64", "age_65_79", "age_ge_80"])
        rows.extend(
            rows_for_groups(
                "eICU",
                frame,
                "hospital_mortality",
                "calibrated_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("gender", "gender_group"), ("age", "age_group")],
            )
        )

    if model_comparison_path.exists():
        predictions = pd.read_csv(model_comparison_path)
        predictions = predictions[
            predictions["split"].astype(str).eq("test")
            & predictions["model"].astype(str).ne("logistic_regression_balanced")
        ].copy()
        frame = predictions.merge(cohort, on="stay_id", how="left")
        frame["calibration_method"] = "raw_model_comparison"
        frame["gender_group"] = frame["gender"].astype(str).str.lower().replace({"nan": "unknown", "": "unknown"})
        frame["age_group"] = age_group(frame["age_years"], [18, 50, 65, 80, 200], ["age_18_49", "age_50_64", "age_65_79", "age_ge_80"])
        rows.extend(
            rows_for_groups(
                "eICU",
                frame,
                "hospital_mortality",
                "predicted_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("gender", "gender_group"), ("age", "age_group")],
            )
        )
    if comparison_recalibration_path.exists():
        predictions = pd.read_csv(comparison_recalibration_path, low_memory=False)
        predictions = predictions[
            predictions["dataset"].astype(str).eq("eICU")
            & predictions["split"].astype(str).eq("test")
            & predictions["calibration_method"].astype(str).ne("raw")
        ].copy()
        predictions["row_id"] = predictions["row_id"].astype(str)
        frame = predictions.merge(cohort[["row_id", "age_years", "gender"]], on="row_id", how="left")
        frame["gender_group"] = frame["gender"].astype(str).str.lower().replace({"nan": "unknown", "": "unknown"})
        frame["age_group"] = age_group(frame["age_years"], [18, 50, 65, 80, 200], ["age_18_49", "age_50_64", "age_65_79", "age_ge_80"])
        rows.extend(
            rows_for_groups(
                "eICU",
                frame,
                "label",
                "calibrated_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("gender", "gender_group"), ("age", "age_group")],
            )
        )
    return rows


def charls_rows(project_root: Path) -> list[dict[str, object]]:
    pred_path = project_root / "outputs" / "tables" / "charls_probability_recalibration_predictions.csv"
    model_comparison_path = project_root / "outputs" / "tables" / "charls_incident_diabetes_model_comparison_predictions.csv"
    comparison_recalibration_path = project_root / "outputs" / "tables" / "external_model_comparison_recalibration_predictions.csv"
    feature_path = project_root / "data" / "processed" / "charls_incident_diabetes_baseline_features.csv"
    if not feature_path.exists():
        return []
    features = pd.read_csv(feature_path, usecols=["person_id", "charls_baseline_age_years", "charls_baseline_sex_code"])
    features["row_id"] = features["person_id"].astype(str)
    rows: list[dict[str, object]] = []

    if pred_path.exists():
        predictions = pd.read_csv(pred_path)
        predictions = predictions[
            predictions["split"].astype(str).eq("test")
            & predictions["calibration_method"].astype(str).eq("platt_validation")
        ].copy()
        frame = predictions.merge(features, on="person_id", how="left")
        frame["model"] = frame["source_model"]
        frame["sex_group"] = frame["charls_baseline_sex_code"].map({1.0: "sex_1", 2.0: "sex_2"}).fillna("sex_unknown")
        frame["age_group"] = age_group(frame["charls_baseline_age_years"], [0, 50, 65, 200], ["age_lt_50", "age_50_64", "age_ge_65"])
        rows.extend(
            rows_for_groups(
                "CHARLS",
                frame,
                "incident_diabetes_2013_or_2015",
                "calibrated_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("sex", "sex_group"), ("age", "age_group")],
            )
        )

    if model_comparison_path.exists():
        predictions = pd.read_csv(model_comparison_path)
        predictions = predictions[
            predictions["split"].astype(str).eq("test")
            & predictions["model"].astype(str).ne("logistic_regression_balanced")
        ].copy()
        frame = predictions.merge(features, on="person_id", how="left")
        frame["calibration_method"] = "raw_model_comparison"
        frame["sex_group"] = frame["charls_baseline_sex_code"].map({1.0: "sex_1", 2.0: "sex_2"}).fillna("sex_unknown")
        frame["age_group"] = age_group(frame["charls_baseline_age_years"], [0, 50, 65, 200], ["age_lt_50", "age_50_64", "age_ge_65"])
        rows.extend(
            rows_for_groups(
                "CHARLS",
                frame,
                "incident_diabetes_2013_or_2015",
                "predicted_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("sex", "sex_group"), ("age", "age_group")],
            )
        )
    if comparison_recalibration_path.exists():
        predictions = pd.read_csv(comparison_recalibration_path, low_memory=False)
        predictions = predictions[
            predictions["dataset"].astype(str).eq("CHARLS")
            & predictions["split"].astype(str).eq("test")
            & predictions["calibration_method"].astype(str).ne("raw")
        ].copy()
        predictions["row_id"] = predictions["row_id"].astype(str)
        frame = predictions.merge(features[["row_id", "charls_baseline_age_years", "charls_baseline_sex_code"]], on="row_id", how="left")
        frame["sex_group"] = frame["charls_baseline_sex_code"].map({1.0: "sex_1", 2.0: "sex_2"}).fillna("sex_unknown")
        frame["age_group"] = age_group(frame["charls_baseline_age_years"], [0, 50, 65, 200], ["age_lt_50", "age_50_64", "age_ge_65"])
        rows.extend(
            rows_for_groups(
                "CHARLS",
                frame,
                "label",
                "calibrated_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("sex", "sex_group"), ("age", "age_group")],
            )
        )
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, table: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_subgroup_performance.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    focus = table[table["split_note"].astype(str).eq("test")].copy() if "split_note" in table.columns else table.copy()
    compact_cols = [
        "dataset",
        "feature_set",
        "model",
        "calibration_method",
        "subgroup_type",
        "subgroup",
        "n",
        "events",
        "event_rate",
        "AUROC",
        "AUPRC",
        "Brier",
        "status",
    ]
    cols = [col for col in compact_cols if col in focus.columns]
    report.write_text(
        f"""# External Subgroup Performance

- Boundary: research model evaluation only; no medical QA, diagnosis, or treatment recommendation.
- Scope: test split subgroup performance for completed external benchmarks.
- CDSL includes all traditional model/window rows; eICU and CHARLS include Platt-calibrated logistic probabilities plus raw and validation-calibrated RF/HGB model-comparison probabilities.

## Subgroup Table

{markdown_table(focus[cols])}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    rows = cdsl_rows(args.project_root) + eicu_rows(args.project_root) + charls_rows(args.project_root)
    table = pd.DataFrame(rows)
    if not table.empty:
        table["split_note"] = "test"
    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "external_subgroup_performance.csv", index=False)
    report = write_report(args.project_root, table)
    print(f"Wrote {report}")
    print(table.groupby(["dataset", "subgroup_type"]).size().reset_index(name="rows").to_string(index=False))


if __name__ == "__main__":
    main()
