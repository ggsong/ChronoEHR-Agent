#!/usr/bin/env python3
"""Recalibrate eICU/CHARLS RF and HGB model-comparison probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_MODELS = {"random_forest_balanced", "hist_gradient_boosting_weighted"}
EXPECTED_METHODS = ["raw", "intercept_validation", "platt_validation", "isotonic_validation"]
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def inv_logit(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def load_eicu(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "tables" / "eicu_first24h_model_comparison_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"stay_id", "patient_id", "split", "hospital_mortality", "prediction_time", "feature_set", "model", "predicted_risk"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"eICU model-comparison predictions missing columns: {missing}")
    df = df[df["model"].astype(str).isin(EXPECTED_MODELS)].copy()
    df["dataset"] = "eICU"
    df["row_id"] = df["stay_id"].astype(str)
    df["label"] = df["hospital_mortality"].astype(int)
    return df[["dataset", "row_id", "patient_id", "split", "label", "prediction_time", "feature_set", "model", "predicted_risk"]]


def load_charls(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "tables" / "charls_incident_diabetes_model_comparison_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {
        "person_id",
        "household_id",
        "community_id",
        "split",
        "incident_diabetes_2013_or_2015",
        "prediction_time",
        "feature_set",
        "model",
        "predicted_risk",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CHARLS model-comparison predictions missing columns: {missing}")
    df = df[df["model"].astype(str).isin(EXPECTED_MODELS)].copy()
    df["dataset"] = "CHARLS"
    df["row_id"] = df["person_id"].astype(str)
    df["patient_id"] = df["person_id"].astype(str)
    df["label"] = df["incident_diabetes_2013_or_2015"].astype(int)
    return df[["dataset", "row_id", "patient_id", "split", "label", "prediction_time", "feature_set", "model", "predicted_risk"]]


def load_predictions(project_root: Path) -> pd.DataFrame:
    frames = [load_eicu(project_root), load_charls(project_root)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError("Missing eICU/CHARLS model-comparison prediction files.")
    return pd.concat(frames, ignore_index=True)


def fit_calibrators(group: pd.DataFrame) -> dict[str, object]:
    validation = group[group["split"].astype(str).eq("validation")].copy()
    if validation.empty:
        raise ValueError("Validation split is required for model-comparison recalibration.")
    y = validation["label"].astype(int)
    scores = validation["predicted_risk"].astype(float)
    if y.nunique() != 2:
        raise ValueError("Validation split must contain both outcome classes.")

    raw_logit = logit(scores)
    observed = float(y.mean())
    predicted = float(scores.mean())
    intercept_shift = float(logit(np.array([observed]))[0] - logit(np.array([predicted]))[0])

    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(raw_logit.reshape(-1, 1), y)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(scores, y)

    return {
        "intercept_shift": intercept_shift,
        "platt": platt,
        "isotonic": isotonic,
        "validation_observed_event_rate": observed,
        "validation_mean_raw_prediction": predicted,
    }


def apply_calibration(group: pd.DataFrame, calibrators: dict[str, object]) -> pd.DataFrame:
    raw = group["predicted_risk"].astype(float)
    raw_logit = logit(raw)
    calibrated = {
        "raw": raw.to_numpy(),
        "intercept_validation": inv_logit(raw_logit + float(calibrators["intercept_shift"])),
        "platt_validation": calibrators["platt"].predict_proba(raw_logit.reshape(-1, 1))[:, 1],
        "isotonic_validation": calibrators["isotonic"].predict(raw),
    }
    frames = []
    for method, values in calibrated.items():
        part = group[["dataset", "row_id", "patient_id", "split", "label", "prediction_time", "feature_set", "model"]].copy()
        part["calibration_method"] = method
        part["raw_predicted_risk"] = raw.to_numpy()
        part["calibrated_risk"] = np.clip(values, 0.0, 1.0)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def build_predictions(raw_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts = []
    calibrator_rows = []
    for (dataset, model), group in raw_predictions.groupby(["dataset", "model"], sort=True):
        calibrators = fit_calibrators(group)
        prediction_parts.append(apply_calibration(group, calibrators))
        calibrator_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "validation_observed_event_rate": calibrators["validation_observed_event_rate"],
                "validation_mean_raw_prediction": calibrators["validation_mean_raw_prediction"],
                "intercept_shift": calibrators["intercept_shift"],
            }
        )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(calibrator_rows)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, method, split), group in predictions.groupby(["dataset", "model", "calibration_method", "split"], sort=False):
        y = group["label"].astype(int)
        score = group["calibrated_risk"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "calibration_method": method,
                "split": split,
                "n": int(len(group)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "mean_predicted_risk": float(score.mean()),
                "absolute_mean_error": abs(float(score.mean()) - float(y.mean())),
                "AUROC": float(roc_auc_score(y, score)) if y.nunique() == 2 else np.nan,
                "AUPRC": float(average_precision_score(y, score)) if y.nunique() == 2 else np.nan,
                "Brier_score": float(brier_score_loss(y, score)),
            }
        )
    return pd.DataFrame(rows)


def calibration_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, method, split), group in predictions.groupby(["dataset", "model", "calibration_method", "split"], sort=False):
        ranked = group.sort_values("calibrated_risk").copy()
        ranked["decile"] = pd.qcut(ranked["calibrated_risk"].rank(method="first"), 10, labels=False) + 1
        for decile, part in ranked.groupby("decile", sort=True):
            mean_pred = float(part["calibrated_risk"].mean())
            observed = float(part["label"].mean())
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "calibration_method": method,
                    "split": split,
                    "decile": int(decile),
                    "n": int(len(part)),
                    "events": int(part["label"].sum()),
                    "mean_predicted_risk": mean_pred,
                    "observed_event_rate": observed,
                    "absolute_calibration_error": abs(mean_pred - observed),
                }
            )
    return pd.DataFrame(rows)


def calibration_summary(deciles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, method, split), group in deciles.groupby(["dataset", "model", "calibration_method", "split"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "calibration_method": method,
                "split": split,
                "mean_absolute_calibration_error": float((group["absolute_calibration_error"] * group["n"]).sum() / group["n"].sum()),
                "max_absolute_calibration_error": float(group["absolute_calibration_error"].max()),
                "deciles": int(group["decile"].nunique()),
                "n": int(group["n"].sum()),
                "events": int(group["events"].sum()),
            }
        )
    return pd.DataFrame(rows)


def decision_curve_metrics(df: pd.DataFrame, threshold: float) -> dict[str, float | int | str]:
    n = len(df)
    events = int(df["label"].sum())
    non_events = n - events
    event_rate = events / n if n else 0.0
    weight = threshold / (1 - threshold)
    flagged = df[df["calibrated_risk"].ge(threshold)]
    alerts = len(flagged)
    tp = int(flagged["label"].sum())
    fp = alerts - tp
    model_net_benefit = (tp / n) - (fp / n) * weight
    treat_all_net_benefit = event_rate - (non_events / n) * weight
    treat_none_net_benefit = 0.0
    best_reference = max(treat_all_net_benefit, treat_none_net_benefit)
    return {
        "threshold_probability": threshold,
        "n": n,
        "events": events,
        "event_rate": event_rate,
        "alerts": alerts,
        "alert_rate": alerts / n if n else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "ppv": tp / alerts if alerts else 0.0,
        "recall": tp / events if events else 0.0,
        "model_net_benefit": model_net_benefit,
        "treat_all_net_benefit": treat_all_net_benefit,
        "treat_none_net_benefit": treat_none_net_benefit,
        "net_benefit_advantage": model_net_benefit - best_reference,
        "preferred_strategy": "model" if model_net_benefit > best_reference else ("treat_all" if treat_all_net_benefit > 0 else "treat_none"),
    }


def decision_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, method, split), group in predictions.groupby(["dataset", "model", "calibration_method", "split"], sort=False):
        for threshold in THRESHOLDS:
            rows.append({"dataset": dataset, "model": model, "calibration_method": method, "split": split, **decision_curve_metrics(group, threshold)})
    return pd.DataFrame(rows)


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


def write_report(project_root: Path, metrics: pd.DataFrame, summary: pd.DataFrame, decision: pd.DataFrame, calibrators: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_model_comparison_recalibration.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = metrics[metrics["split"].astype(str).eq("test")].copy()
    test_summary = summary[summary["split"].astype(str).eq("test")].copy()
    test_decision = decision[decision["split"].astype(str).eq("test")].copy()
    report.write_text(
        f"""# External Model-Comparison Recalibration

- Boundary: research model evaluation only; no medical QA, diagnosis, or treatment recommendation.
- Scope: validation-set recalibration of eICU/CHARLS RF and HGB model-comparison probabilities.
- Calibration methods: raw, validation intercept shift, Platt scaling, isotonic regression.

## Calibration Fit Summary

{markdown_table(calibrators)}

## Test Metrics

{markdown_table(test_metrics[["dataset", "model", "calibration_method", "n", "events", "event_rate", "mean_predicted_risk", "absolute_mean_error", "AUROC", "AUPRC", "Brier_score"]])}

## Test Calibration Summary

{markdown_table(test_summary[["dataset", "model", "calibration_method", "mean_absolute_calibration_error", "max_absolute_calibration_error", "deciles", "n", "events"]])}

## Test Decision Curve

{markdown_table(test_decision[["dataset", "model", "calibration_method", "threshold_probability", "alerts", "alert_rate", "ppv", "recall", "model_net_benefit", "treat_all_net_benefit", "net_benefit_advantage", "preferred_strategy"]])}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    raw = load_predictions(args.project_root)
    predictions, calibrators = build_predictions(raw)
    metrics = metric_rows(predictions)
    deciles = calibration_deciles(predictions)
    summary = calibration_summary(deciles)
    decision = decision_curve(predictions)

    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(tables / "external_model_comparison_recalibration_predictions.csv", index=False)
    calibrators.to_csv(tables / "external_model_comparison_recalibration_calibrators.csv", index=False)
    metrics.to_csv(tables / "external_model_comparison_recalibration_metrics.csv", index=False)
    deciles.to_csv(tables / "external_model_comparison_recalibration_deciles.csv", index=False)
    summary.to_csv(tables / "external_model_comparison_recalibration_summary.csv", index=False)
    decision.to_csv(tables / "external_model_comparison_recalibration_decision_curve.csv", index=False)
    report = write_report(args.project_root, metrics, summary, decision, calibrators)
    print(f"Wrote {report}")
    print(metrics[metrics["split"].astype(str).eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
