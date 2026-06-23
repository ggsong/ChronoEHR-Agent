#!/usr/bin/env python3
"""Run a lightweight CHARLS incident diabetes logistic baseline."""

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


ID_COLUMNS = {"person_id", "household_id", "community_id", "split", "incident_diabetes_2013_or_2015"}
LABEL_COLUMN = "incident_diabetes_2013_or_2015"
MODEL_NAME = "logistic_regression_balanced"
FEATURE_SET = "charls_2011_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def load_matrix(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "charls_incident_diabetes_baseline_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CHARLS baseline feature matrix: {path}")
    return pd.read_csv(path, low_memory=False)


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    cols = [column for column in matrix.columns if column not in ID_COLUMNS]
    bad = [column for column in cols if not column.startswith("charls_baseline_")]
    if bad:
        raise ValueError(f"Unexpected non-baseline feature columns: {bad[:20]}")
    return cols


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
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


def evaluate(model: Pipeline, frame: pd.DataFrame, cols: list[str], split: str, threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    y = frame[LABEL_COLUMN].astype(int)
    score = model.predict_proba(frame[cols])[:, 1]
    metrics = binary_metrics(y, score, threshold)
    metrics.update(
        {
            "study": "charls_incident_diabetes",
            "feature_set": FEATURE_SET,
            "prediction_time": "2011_wave1_baseline",
            "model": MODEL_NAME,
            "split": split,
            "feature_count": len(cols),
        }
    )
    predictions = frame[["person_id", "household_id", "community_id", "split", LABEL_COLUMN]].copy()
    predictions["prediction_time"] = "2011_wave1_baseline"
    predictions["feature_set"] = FEATURE_SET
    predictions["model"] = MODEL_NAME
    predictions["predicted_risk"] = score
    return metrics, predictions


def coefficient_table(model: Pipeline, cols: list[str]) -> pd.DataFrame:
    fitted = model.named_steps["model"]
    coefficients = fitted.coef_[0]
    return pd.DataFrame(
        {
            "study": "charls_incident_diabetes",
            "prediction_time": "2011_wave1_baseline",
            "feature_set": FEATURE_SET,
            "model": MODEL_NAME,
            "feature": cols,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, metrics: pd.DataFrame, coefficients: pd.DataFrame) -> Path:
    report_path = project_root / "outputs" / "reports" / "charls_incident_diabetes_logistic_baseline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    test = metrics[metrics["split"].eq("test")]
    top = coefficients.head(20)
    report_path.write_text(
        f"""# CHARLS Incident Diabetes Logistic Baseline

- Boundary: research benchmark only; no medical QA, diagnosis, or treatment recommendation.
- Task: 2011 baseline prediction of incident diabetes by 2013/2015.
- Model: balanced logistic regression with median imputation and standard scaling.
- Leakage gate: run before modeling; 2013/2015 follow-up variables and raw outcome components are blocked.

## Split Metrics

{markdown_table(metrics[["split", "n", "events", "event_rate", "AUROC", "AUPRC", "Brier_score", "sensitivity", "specificity", "ppv", "npv", "feature_count"]])}

## Test Result

{markdown_table(test[["model", "feature_set", "AUROC", "AUPRC", "Brier_score", "event_rate", "feature_count"]])}

## Top Coefficients By Absolute Value

{markdown_table(top[["feature", "coefficient", "abs_coefficient"]])}

## Interpretation

This is a minimal traditional baseline for the CHARLS external longitudinal slice. It should be treated as a validation target for the workflow, not as a clinical decision tool.
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    matrix = load_matrix(args.project_root)
    cols = feature_columns(matrix)
    train = matrix[matrix["split"].eq("train")].copy()
    validation = matrix[matrix["split"].eq("validation")].copy()
    test = matrix[matrix["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise SystemExit("Missing train/validation/test split rows in CHARLS feature matrix.")

    model = make_model()
    y_train = train[LABEL_COLUMN].astype(int)
    model.fit(train[cols], y_train)
    train_scores = model.predict_proba(train[cols])[:, 1]
    threshold = choose_threshold(train_scores, y_train)

    metric_rows = []
    prediction_parts = []
    for split_name, frame in [("train", train), ("validation", validation), ("test", test)]:
        metrics, predictions = evaluate(model, frame, cols, split_name, threshold)
        metric_rows.append(metrics)
        prediction_parts.append(predictions)
    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    coefficients_df = coefficient_table(model, cols)

    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(tables / "charls_incident_diabetes_logistic_baseline_metrics.csv", index=False)
    predictions_df.to_csv(tables / "charls_incident_diabetes_logistic_baseline_predictions.csv", index=False)
    coefficients_df.to_csv(tables / "charls_incident_diabetes_logistic_baseline_coefficients.csv", index=False)
    report = write_report(args.project_root, metrics_df, coefficients_df)
    print(f"Wrote {report}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
