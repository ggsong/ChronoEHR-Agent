"""Shared prediction-time logistic modeling utilities for ChronoEHR studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from leakage_gate import enforce_spec_gate
from mimic_diabetes_baseline import (
    choose_threshold_by_train_prevalence,
    evaluate_split,
    fit_preprocessor,
    train_logistic_regression,
    transform,
    validate_no_forbidden_features,
)


def load_analysis_data(project_root: Path, cohort_path: str, extra_feature_files: list[str] | None = None) -> pd.DataFrame:
    cohort = pd.read_csv(project_root / cohort_path, low_memory=False)
    for relative_path in extra_feature_files or []:
        feature_path = project_root / relative_path
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        features = pd.read_csv(feature_path, low_memory=False)
        cohort = cohort.merge(features, on="hadm_id", how="left")
    return cohort


def run_feature_set(project_root: Path, cohort_path: str, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enforce_spec_gate(project_root, str(spec.get("study_key", "unknown")), spec)
    cohort = load_analysis_data(project_root, cohort_path, spec.get("extra_feature_files"))
    feature_cols = spec["numeric_features"] + spec["categorical_features"]
    missing = [col for col in feature_cols if col not in cohort.columns]
    if missing:
        raise ValueError(f"Missing columns for feature set {spec['feature_set']}: {missing}")

    train = cohort[cohort["split"].eq("train")].copy()
    validation = cohort[cohort["split"].eq("validation")].copy()
    test = cohort[cohort["split"].eq("test")].copy()

    preprocessor = fit_preprocessor(train, spec["numeric_features"], spec["categorical_features"])
    x_train, feature_names = transform(train, preprocessor)
    validate_no_forbidden_features(feature_names)
    y_train = train["readmission_30d"].astype(int).to_numpy()

    weights = train_logistic_regression(x_train, y_train)
    train_scores = expit(x_train @ weights)
    threshold = choose_threshold_by_train_prevalence(train_scores, y_train)

    performance = pd.DataFrame(
        [
            evaluate_split(spec["feature_set"], "train", train, weights, preprocessor, threshold),
            evaluate_split(spec["feature_set"], "validation", validation, weights, preprocessor, threshold),
            evaluate_split(spec["feature_set"], "test", test, weights, preprocessor, threshold),
        ]
    )
    performance["prediction_time"] = spec["prediction_time"]

    coefficients = pd.DataFrame(
        {
            "feature_set": spec["feature_set"],
            "prediction_time": spec["prediction_time"],
            "feature": feature_names,
            "coefficient": weights,
            "abs_coefficient": np.abs(weights),
        }
    ).sort_values(["feature_set", "abs_coefficient"], ascending=[True, False])

    x_test, _ = transform(test, preprocessor)
    predictions = test[["subject_id", "hadm_id", "readmission_30d"]].copy()
    predictions["feature_set"] = spec["feature_set"]
    predictions["prediction_time"] = spec["prediction_time"]
    predictions["predicted_risk"] = expit(x_test @ weights)
    return performance, coefficients, predictions


def format_test_rows(performance: pd.DataFrame, ordered_feature_sets: list[str]) -> str:
    tests = performance[performance["split"].eq("test")].set_index("feature_set")
    rows = []
    for name in ordered_feature_sets:
        if name not in tests.index:
            continue
        row = tests.loc[name]
        rows.append(
            f"| {name} | {row['prediction_time']} | {int(row['n']):,} | {int(row['events']):,} | "
            f"{row['event_rate']:.2%} | {row['AUROC']:.4f} | {row['AUPRC']:.4f} | "
            f"{row['Brier_score']:.4f} | {row['sensitivity']:.4f} | {row['specificity']:.4f} | "
            f"{row['ppv']:.4f} | {row['npv']:.4f} |"
        )
    return "\n".join(rows)


def format_differences(performance: pd.DataFrame, difference_pairs: list[tuple[str, str, str]]) -> str:
    tests = performance[performance["split"].eq("test")].set_index("feature_set")
    lines = []
    for baseline, comparator, label in difference_pairs:
        if baseline not in tests.index or comparator not in tests.index:
            continue
        base = tests.loc[baseline]
        comp = tests.loc[comparator]
        lines.append(f"- {label} AUROC: {comp['AUROC'] - base['AUROC']:.4f}")
        lines.append(f"- {label} AUPRC: {comp['AUPRC'] - base['AUPRC']:.4f}")
    return "\n".join(lines)


def write_prediction_time_report(
    performance: pd.DataFrame,
    report_path: Path,
    title: str,
    intro: str,
    feature_descriptions: list[str],
    ordered_feature_sets: list[str],
    difference_pairs: list[tuple[str, str, str]],
    leakage_notes: list[str],
) -> None:
    text = f"""# {title}

{intro}

## Feature Sets

{chr(10).join(f"- {line}" for line in feature_descriptions)}

## Test Set Results

| Feature set | Prediction time | N | Events | Event rate | AUROC | AUPRC | Brier | Sensitivity | Specificity | PPV | NPV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{format_test_rows(performance, ordered_feature_sets)}

## Prediction-Time Differences

{format_differences(performance, difference_pairs)}

## Leakage Notes

{chr(10).join(f"- {line}" for line in leakage_notes)}
"""
    report_path.write_text(text, encoding="utf-8")


def run_prediction_time_models(
    project_root: Path,
    cohort_path: str,
    specs: list[dict],
    performance_path: Path,
    coefficient_path: Path,
    prediction_path: Path,
    report_path: Path,
    report_options: dict,
) -> pd.DataFrame:
    performance_parts = []
    coefficient_parts = []
    prediction_parts = []
    for spec in specs:
        performance, coefficients, predictions = run_feature_set(project_root, cohort_path, spec)
        performance_parts.append(performance)
        coefficient_parts.append(coefficients)
        prediction_parts.append(predictions)

    performance_all = pd.concat(performance_parts, ignore_index=True)
    coefficients_all = pd.concat(coefficient_parts, ignore_index=True)
    predictions_all = pd.concat(prediction_parts, ignore_index=True)

    performance_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    performance_all.to_csv(performance_path, index=False)
    coefficients_all.to_csv(coefficient_path, index=False)
    predictions_all.to_csv(prediction_path, index=False)
    write_prediction_time_report(performance_all, report_path, **report_options)
    return performance_all
