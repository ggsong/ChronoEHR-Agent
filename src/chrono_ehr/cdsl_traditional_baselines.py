#!/usr/bin/env python3
"""Run traditional baselines for the CDSL temporal benchmark."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from cdsl_external_validation_readiness import CDSL_CANDIDATE_ROOTS, choose_cdsl_root
from cdsl_temporal_benchmark import (
    ID_COL,
    WINDOWS,
    build_feature_matrix,
    load_fold_pids,
    load_formatted,
    patient_labels,
)
from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--cdsl-root", type=Path, help="Optional explicit CDSL root.")
    parser.add_argument("--min-coverage", type=float, default=0.05)
    return parser.parse_args()


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
                        n_estimators=250,
                        min_samples_leaf=10,
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
                        max_iter=200,
                        learning_rate=0.05,
                        l2_regularization=0.01,
                        random_state=2026,
                    ),
                ),
            ]
        ),
    }


def predict_frame(
    model: Pipeline,
    split_df: pd.DataFrame,
    feature_cols: list[str],
    split: str,
    feature_set: str,
    model_name: str,
) -> pd.DataFrame:
    proba = model.predict_proba(split_df[feature_cols])[:, 1]
    predictions = split_df[[ID_COL, "outcome"]].copy()
    predictions["split"] = split
    predictions["feature_set"] = feature_set
    predictions["model"] = model_name
    predictions["predicted_risk"] = proba
    return predictions


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, split: str) -> dict[str, Any]:
    proba = model.predict_proba(X)[:, 1]
    return {
        "split": split,
        "n": len(y),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "AUROC": float(roc_auc_score(y, proba)) if y.nunique() == 2 else np.nan,
        "AUPRC": float(average_precision_score(y, proba)) if y.nunique() == 2 else np.nan,
        "Brier": float(brier_score_loss(y, proba)),
    }


def fit_model(model_name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    if model_name == "hist_gradient_boosting_weighted":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def run_baselines(features: pd.DataFrame, pids: dict[str, set[str]], feature_set: str) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    feature_cols = [col for col in features.columns if col not in {ID_COL, "outcome"}]
    train = features[features[ID_COL].isin(pids["train"])].copy()
    val = features[features[ID_COL].isin(pids["val"])].copy()
    test = features[features[ID_COL].isin(pids["test"])].copy()

    rows = []
    predictions = []
    for model_name, model in make_models().items():
        fitted = fit_model(model_name, model, train[feature_cols], train["outcome"])
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            row = evaluate(fitted, split_df[feature_cols], split_df["outcome"], split_name)
            row["feature_set"] = feature_set
            row["model"] = model_name
            row["feature_count"] = len(feature_cols)
            rows.append(row)
            predictions.append(predict_frame(fitted, split_df, feature_cols, split_name, feature_set, model_name))
    return rows, predictions


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, root: Path, metrics: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "cdsl_traditional_baselines_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    test = metrics[metrics["split"].eq("test")].sort_values(["feature_set", "model"])
    best = test.sort_values(["AUPRC", "AUROC"], ascending=False).head(8)
    lines = [
        "# CDSL Traditional Baselines Report",
        "",
        f"- CDSL root: `{root}`",
        "- Task: CDSL in-hospital mortality prediction",
        "- Models: logistic regression, random forest, HistGradientBoosting",
        "- Purpose: method benchmark and leakage-aware baseline comparison, not chronic readmission external validation.",
        "",
        "## Test Metrics",
        "",
        markdown_table(test[["feature_set", "model", "n", "events", "event_rate", "AUROC", "AUPRC", "Brier", "feature_count"]]),
        "",
        "## Best Rows By AUPRC",
        "",
        markdown_table(best[["feature_set", "model", "AUROC", "AUPRC", "Brier"]]),
        "",
        "## 解释",
        "",
        "这些传统模型用于对照，不替代临床判断。`full_stay_naive_reference` 使用全住院窗口，不能被解释为入院时预测性能。",
        "如果它明显优于 24h/48h 模型，说明时间窗口越晚、可用信息越多，模型表现会被抬高。",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    args = parse_args()
    root, _ = choose_cdsl_root(args.cdsl_root)
    if root is None:
        raise SystemExit(f"No usable CDSL root found. Checked: {', '.join(str(path) for path in CDSL_CANDIDATE_ROOTS)}")

    df = load_formatted(root)
    labels = patient_labels(df)
    pids = load_fold_pids(root, fold=0)

    feature_dir = args.project_root / "data" / "processed" / "cdsl_temporal_benchmark"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for window in WINDOWS:
        features, summary = build_feature_matrix(df, labels, window, pids["train"], args.min_coverage)
        features.to_csv(feature_dir / f"{window['feature_set']}.csv", index=False)
        summary_rows.append(summary)
        rows, predictions = run_baselines(features, pids, window["feature_set"])
        metric_rows.extend(rows)
        prediction_parts.extend(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    window_summary = pd.DataFrame(summary_rows)

    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "cdsl_traditional_baselines_metrics.csv", index=False)
    predictions.to_csv(tables / "cdsl_traditional_baselines_predictions.csv", index=False)
    window_summary.to_csv(tables / "cdsl_traditional_baselines_window_audit.csv", index=False)
    report = write_report(args.project_root, root, metrics)
    print(f"Wrote {report}")
    print(metrics[metrics["split"].eq("test")].sort_values(["feature_set", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
