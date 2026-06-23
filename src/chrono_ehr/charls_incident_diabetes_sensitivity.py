#!/usr/bin/env python3
"""Run CHARLS incident diabetes baseline sensitivity analyses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mimic_diabetes_baseline import DEFAULT_PROJECT


PRIMARY_LABEL = "incident_diabetes_2013_or_2015"
FEATURE_PREFIX = "charls_baseline_"
MODEL_NAME = "logistic_regression_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]
    )


def choose_threshold(train_scores: np.ndarray, y_train: pd.Series) -> float:
    event_rate = float(y_train.mean())
    if event_rate <= 0 or event_rate >= 1:
        return 0.5
    return float(np.quantile(train_scores, 1 - event_rate))


def binary_metrics(y: pd.Series, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = score >= threshold
    y_bool = y.astype(bool).to_numpy()
    tp = int(np.logical_and(pred, y_bool).sum())
    fp = int(np.logical_and(pred, ~y_bool).sum())
    tn = int(np.logical_and(~pred, ~y_bool).sum())
    fn = int(np.logical_and(~pred, y_bool).sum())
    return {
        "n": len(y),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "AUROC": float(roc_auc_score(y, score)) if y.nunique() == 2 else np.nan,
        "AUPRC": float(average_precision_score(y, score)) if y.nunique() == 2 else np.nan,
        "Brier_score": float(brier_score_loss(y, score)),
        "threshold": threshold,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
    }


def feature_columns(matrix: pd.DataFrame, scenario: str) -> list[str]:
    cols = [column for column in matrix.columns if column.startswith(FEATURE_PREFIX)]
    if scenario == "no_bmi":
        cols = [column for column in cols if "baseline_bmi" not in column]
    return cols


def scenario_frame(matrix: pd.DataFrame, cohort: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, str, str]:
    frame = matrix.copy()
    if scenario == "primary":
        return frame, PRIMARY_LABEL, "Primary 2013/2015 incident diabetes label."
    merged = frame.merge(
        cohort[
            [
                "person_id",
                "baseline_age_years",
                "followup_2013_diabetes_known",
                "followup_2015_diabetes_known",
                "incident_diabetes_2013",
                "incident_diabetes_2015",
            ]
        ],
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    if scenario == "no_bmi":
        return frame, PRIMARY_LABEL, "Primary label, excluding BMI and BMI missingness features."
    if scenario == "outcome_2013_only":
        out = merged[merged["followup_2013_diabetes_known"].astype(bool)].copy()
        out["scenario_label"] = out["incident_diabetes_2013"].astype(int)
        return out, "scenario_label", "Only respondents with known 2013 diabetes status; label is 2013 incident diabetes."
    if scenario == "outcome_2015_only":
        out = merged[merged["followup_2015_diabetes_known"].astype(bool)].copy()
        out["scenario_label"] = out["incident_diabetes_2015"].astype(int)
        return out, "scenario_label", "Only respondents with known 2015 diabetes status; label is 2015 incident diabetes."
    if scenario == "age_ge_50":
        out = merged[merged["baseline_age_years"].ge(50)].copy()
        return out, PRIMARY_LABEL, "Primary label restricted to baseline age >=50."
    if scenario == "age_ge_60":
        out = merged[merged["baseline_age_years"].ge(60)].copy()
        return out, PRIMARY_LABEL, "Primary label restricted to baseline age >=60."
    raise ValueError(f"Unknown sensitivity scenario: {scenario}")


def evaluate_scenario(matrix: pd.DataFrame, cohort: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, label_col, note = scenario_frame(matrix, cohort, scenario)
    cols = feature_columns(frame, scenario)
    train = frame[frame["split"].eq("train")].copy()
    validation = frame[frame["split"].eq("validation")].copy()
    test = frame[frame["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"{scenario} is missing train/validation/test rows.")
    if train[label_col].nunique() < 2 or test[label_col].nunique() < 2:
        raise ValueError(f"{scenario} lacks both label classes in train or test.")

    model = make_model()
    y_train = train[label_col].astype(int)
    model.fit(train[cols], y_train)
    threshold = choose_threshold(model.predict_proba(train[cols])[:, 1], y_train)

    metric_rows = []
    pred_rows = []
    for split, part in [("train", train), ("validation", validation), ("test", test)]:
        y = part[label_col].astype(int)
        score = model.predict_proba(part[cols])[:, 1]
        metrics = binary_metrics(y, score, threshold)
        metrics.update(
            {
                "scenario": scenario,
                "split": split,
                "model": MODEL_NAME,
                "feature_count": len(cols),
                "label_column": label_col,
                "note": note,
            }
        )
        metric_rows.append(metrics)
        pred = part[["person_id", "split"]].copy()
        pred["scenario"] = scenario
        pred["label"] = y.to_numpy()
        pred["predicted_risk"] = score
        pred_rows.append(pred)

    coefficients = pd.DataFrame(
        {
            "scenario": scenario,
            "model": MODEL_NAME,
            "feature": cols,
            "coefficient": model.named_steps["model"].coef_[0],
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True), coefficients


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, metrics: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "charls_incident_diabetes_sensitivity_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    test = metrics[metrics["split"].eq("test")].copy()
    report.write_text(
        f"""# CHARLS Incident Diabetes Sensitivity Analysis

- Boundary: research robustness analysis only; no medical QA, diagnosis, or treatment recommendation.
- Base task: 2011 baseline prediction of incident diabetes by 2013/2015.
- Model: balanced logistic regression retrained within each scenario.

## Test Metrics

{markdown_table(test[["scenario", "n", "events", "event_rate", "AUROC", "AUPRC", "Brier_score", "feature_count", "note"]])}

## Interpretation

These scenarios check whether the baseline result is sensitive to BMI availability, follow-up wave choice, or baseline age restriction. They are not final clinical claims.
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    matrix = pd.read_csv(args.project_root / "data" / "processed" / "charls_incident_diabetes_baseline_features.csv", low_memory=False)
    cohort = pd.read_csv(args.project_root / "data" / "processed" / "charls_incident_diabetes_cohort.csv", low_memory=False)
    scenarios = ["primary", "no_bmi", "outcome_2013_only", "outcome_2015_only", "age_ge_50", "age_ge_60"]
    metrics_parts = []
    prediction_parts = []
    coefficient_parts = []
    for scenario in scenarios:
        metrics, predictions, coefficients = evaluate_scenario(matrix, cohort, scenario)
        metrics_parts.append(metrics)
        prediction_parts.append(predictions)
        coefficient_parts.append(coefficients)

    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    coefficients_df = pd.concat(coefficient_parts, ignore_index=True).sort_values(["scenario", "abs_coefficient"], ascending=[True, False])
    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(tables / "charls_incident_diabetes_sensitivity_metrics.csv", index=False)
    predictions_df.to_csv(tables / "charls_incident_diabetes_sensitivity_predictions.csv", index=False)
    coefficients_df.to_csv(tables / "charls_incident_diabetes_sensitivity_coefficients.csv", index=False)
    report = write_report(args.project_root, metrics_df)
    print(f"Wrote {report}")
    print(metrics_df[metrics_df["split"].eq("test")][["scenario", "n", "events", "event_rate", "AUROC", "AUPRC", "Brier_score", "feature_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
