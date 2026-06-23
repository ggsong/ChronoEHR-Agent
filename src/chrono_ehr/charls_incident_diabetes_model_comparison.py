#!/usr/bin/env python3
"""Run CHARLS incident diabetes traditional model comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from mimic_diabetes_baseline import DEFAULT_PROJECT


ID_COLUMNS = {"person_id", "household_id", "community_id", "split", "incident_diabetes_2013_or_2015"}
LABEL_COLUMN = "incident_diabetes_2013_or_2015"
FEATURE_SET = "charls_2011_baseline"
PREDICTION_TIME = "2011_wave1_baseline"
STUDY = "charls_incident_diabetes"


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


def make_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression_balanced": Pipeline(
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
        ),
        "random_forest_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=8,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=2026,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_weighted": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.04,
                        l2_regularization=0.05,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        random_state=2026,
                    ),
                ),
            ]
        ),
    }


def fit_model(model_name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    if model_name == "hist_gradient_boosting_weighted":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def choose_threshold(scores: np.ndarray, y: pd.Series) -> float:
    event_rate = float(y.mean())
    if event_rate <= 0 or event_rate >= 1:
        return 0.5
    return float(np.quantile(scores, 1 - event_rate))


def binary_metrics(y: pd.Series, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = score >= threshold
    y_bool = y.astype(bool).to_numpy()
    tp = int(np.logical_and(pred, y_bool).sum())
    fp = int(np.logical_and(pred, ~y_bool).sum())
    tn = int(np.logical_and(~pred, ~y_bool).sum())
    fn = int(np.logical_and(~pred, y_bool).sum())
    return {
        "n": int(len(y)),
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


def predict_split(
    model: Pipeline,
    frame: pd.DataFrame,
    cols: list[str],
    split: str,
    model_name: str,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = frame[LABEL_COLUMN].astype(int)
    score = model.predict_proba(frame[cols])[:, 1]
    metrics = binary_metrics(y, score, threshold)
    metrics.update(
        {
            "study": STUDY,
            "feature_set": FEATURE_SET,
            "prediction_time": PREDICTION_TIME,
            "model": model_name,
            "split": split,
            "feature_count": len(cols),
        }
    )
    predictions = frame[["person_id", "household_id", "community_id", "split", LABEL_COLUMN]].copy()
    predictions["prediction_time"] = PREDICTION_TIME
    predictions["feature_set"] = FEATURE_SET
    predictions["model"] = model_name
    predictions["predicted_risk"] = score
    return metrics, predictions


def model_importance(model: Pipeline, cols: list[str], model_name: str) -> pd.DataFrame:
    fitted = model.named_steps["model"]
    values: np.ndarray | None = None
    importance_type = ""
    if hasattr(fitted, "coef_"):
        values = np.asarray(fitted.coef_[0], dtype=float)
        importance_type = "coefficient"
    elif hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float)
        importance_type = "feature_importance"
    if values is None:
        return pd.DataFrame(
            {
                "study": [STUDY],
                "prediction_time": [PREDICTION_TIME],
                "feature_set": [FEATURE_SET],
                "model": [model_name],
                "feature": ["not_available"],
                "importance_type": ["not_available"],
                "importance": [np.nan],
                "abs_importance": [np.nan],
            }
        )
    return pd.DataFrame(
        {
            "study": STUDY,
            "prediction_time": PREDICTION_TIME,
            "feature_set": FEATURE_SET,
            "model": model_name,
            "feature": cols,
            "importance_type": importance_type,
            "importance": values,
            "abs_importance": np.abs(values),
        }
    ).sort_values(["model", "abs_importance"], ascending=[True, False])


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, metrics: pd.DataFrame, importances: pd.DataFrame) -> Path:
    report_path = project_root / "outputs" / "reports" / "charls_incident_diabetes_model_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    test = metrics[metrics["split"].eq("test")].sort_values(["AUPRC", "AUROC"], ascending=False)
    top = importances[importances["feature"].ne("not_available")].groupby("model", group_keys=False).head(10)
    report_path.write_text(
        f"""# CHARLS Incident Diabetes Model Comparison

- Boundary: research benchmark only; no medical QA, diagnosis, or treatment recommendation.
- Task: 2011 baseline prediction of incident diabetes by 2013/2015.
- Models: balanced logistic regression, balanced random forest, weighted HistGradientBoosting.
- Split policy: reuse the existing CHARLS train/validation/test split.
- Leakage gate: run before modeling; follow-up and outcome-derived variables remain blocked.

## Test Metrics

{markdown_table(test[["model", "n", "events", "event_rate", "AUROC", "AUPRC", "Brier_score", "sensitivity", "specificity", "ppv", "npv", "feature_count"]])}

## Split Metrics

{markdown_table(metrics[["model", "split", "n", "events", "event_rate", "AUROC", "AUPRC", "Brier_score", "feature_count"]])}

## Top Available Feature Signals

{markdown_table(top[["model", "feature", "importance_type", "importance", "abs_importance"]])}

## Interpretation

This comparison tests whether nonlinear traditional models add discrimination over the logistic baseline on the same leakage-gated CHARLS slice. Absolute probabilities still require calibration checks before interpretation.
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    matrix = load_matrix(args.project_root)
    cols = feature_columns(matrix)
    split_frames = {split: matrix[matrix["split"].eq(split)].copy() for split in ["train", "validation", "test"]}
    if any(frame.empty for frame in split_frames.values()):
        raise SystemExit("Missing train/validation/test split rows in CHARLS feature matrix.")

    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    for model_name, model in make_models().items():
        train = split_frames["train"]
        y_train = train[LABEL_COLUMN].astype(int)
        fitted = fit_model(model_name, model, train[cols], y_train)
        train_scores = fitted.predict_proba(train[cols])[:, 1]
        threshold = choose_threshold(train_scores, y_train)
        importance_parts.append(model_importance(fitted, cols, model_name))
        for split, frame in split_frames.items():
            metrics, predictions = predict_split(fitted, frame, cols, split, model_name, threshold)
            metric_rows.append(metrics)
            prediction_parts.append(predictions)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    importances_df = pd.concat(importance_parts, ignore_index=True)

    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(tables / "charls_incident_diabetes_model_comparison_metrics.csv", index=False)
    predictions_df.to_csv(tables / "charls_incident_diabetes_model_comparison_predictions.csv", index=False)
    importances_df.to_csv(tables / "charls_incident_diabetes_model_comparison_importances.csv", index=False)
    report = write_report(args.project_root, metrics_df, importances_df)
    print(f"Wrote {report}")
    print(metrics_df[metrics_df["split"].eq("test")].sort_values(["AUPRC", "AUROC"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
