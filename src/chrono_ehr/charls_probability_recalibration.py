#!/usr/bin/env python3
"""Recalibrate CHARLS incident diabetes baseline probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mimic_diabetes_baseline import DEFAULT_PROJECT


LABEL_COLUMN = "incident_diabetes_2013_or_2015"
RAW_MODEL = "logistic_regression_balanced"
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
EXPECTED_METHODS = ["raw", "intercept_validation", "platt_validation", "isotonic_validation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def inv_logit(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def load_predictions(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "tables" / "charls_incident_diabetes_logistic_baseline_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CHARLS baseline predictions: {path}")
    df = pd.read_csv(path)
    required = {"person_id", "household_id", "community_id", "split", LABEL_COLUMN, "predicted_risk"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"baseline prediction file missing columns: {missing}")
    return df


def fit_calibrators(predictions: pd.DataFrame) -> dict[str, object]:
    validation = predictions[predictions["split"].astype(str).eq("validation")].copy()
    if validation.empty:
        raise ValueError("Validation split is required for probability recalibration.")
    y = validation[LABEL_COLUMN].astype(int)
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


def apply_calibration(predictions: pd.DataFrame, calibrators: dict[str, object]) -> pd.DataFrame:
    frames = []
    raw = predictions["predicted_risk"].astype(float)
    raw_logit = logit(raw)
    calibrated = {
        "raw": raw.to_numpy(),
        "intercept_validation": inv_logit(raw_logit + float(calibrators["intercept_shift"])),
        "platt_validation": calibrators["platt"].predict_proba(raw_logit.reshape(-1, 1))[:, 1],
        "isotonic_validation": calibrators["isotonic"].predict(raw),
    }
    for method, values in calibrated.items():
        part = predictions[["person_id", "household_id", "community_id", "split", LABEL_COLUMN]].copy()
        part["prediction_time"] = "2011_wave1_baseline"
        part["feature_set"] = "charls_2011_baseline"
        part["source_model"] = RAW_MODEL
        part["calibration_method"] = method
        part["raw_predicted_risk"] = raw.to_numpy()
        part["calibrated_risk"] = np.clip(values, 0.0, 1.0)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, split), group in predictions.groupby(["calibration_method", "split"], sort=False):
        y = group[LABEL_COLUMN].astype(int)
        score = group["calibrated_risk"].astype(float)
        rows.append(
            {
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
    for (method, split), group in predictions.groupby(["calibration_method", "split"], sort=False):
        ranked = group.sort_values("calibrated_risk").copy()
        ranked["decile"] = pd.qcut(ranked["calibrated_risk"].rank(method="first"), 10, labels=False) + 1
        for decile, part in ranked.groupby("decile", sort=True):
            mean_pred = float(part["calibrated_risk"].mean())
            observed = float(part[LABEL_COLUMN].mean())
            rows.append(
                {
                    "calibration_method": method,
                    "split": split,
                    "decile": int(decile),
                    "n": int(len(part)),
                    "events": int(part[LABEL_COLUMN].sum()),
                    "mean_predicted_risk": mean_pred,
                    "observed_event_rate": observed,
                    "absolute_calibration_error": abs(mean_pred - observed),
                }
            )
    return pd.DataFrame(rows)


def calibration_summary(deciles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, split), group in deciles.groupby(["calibration_method", "split"], sort=False):
        weighted = float((group["absolute_calibration_error"] * group["n"]).sum() / group["n"].sum())
        rows.append(
            {
                "calibration_method": method,
                "split": split,
                "mean_absolute_calibration_error": weighted,
                "max_absolute_calibration_error": float(group["absolute_calibration_error"].max()),
                "deciles": int(group["decile"].nunique()),
                "n": int(group["n"].sum()),
                "events": int(group["events"].sum()),
            }
        )
    return pd.DataFrame(rows)


def decision_curve_metrics(df: pd.DataFrame, threshold: float) -> dict[str, float | int | str]:
    n = len(df)
    events = int(df[LABEL_COLUMN].sum())
    non_events = n - events
    event_rate = events / n if n else 0.0
    weight = threshold / (1 - threshold)
    flagged = df[df["calibrated_risk"].ge(threshold)]
    alerts = len(flagged)
    tp = int(flagged[LABEL_COLUMN].sum())
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
    for (method, split), group in predictions.groupby(["calibration_method", "split"], sort=False):
        for threshold in THRESHOLDS:
            rows.append({"calibration_method": method, "split": split, **decision_curve_metrics(group, threshold)})
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, metrics: pd.DataFrame, summary: pd.DataFrame, decision: pd.DataFrame, calibrators: dict[str, object]) -> Path:
    report = project_root / "outputs" / "reports" / "charls_probability_recalibration_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = metrics[metrics["split"].astype(str).eq("test")].copy()
    test_summary = summary[summary["split"].astype(str).eq("test")].copy()
    test_decision = decision[decision["split"].astype(str).eq("test")].copy()
    report.write_text(
        f"""# CHARLS Probability Recalibration

- Boundary: research model evaluation only; no medical QA, diagnosis, or treatment recommendation.
- Source model: balanced logistic regression trained on CHARLS 2011 baseline features.
- Calibration fit split: validation.
- Validation observed event rate: {float(calibrators["validation_observed_event_rate"]):.4f}.
- Validation mean raw prediction: {float(calibrators["validation_mean_raw_prediction"]):.4f}.

## Test Metrics

{markdown_table(test_metrics[["calibration_method", "n", "events", "event_rate", "mean_predicted_risk", "absolute_mean_error", "AUROC", "AUPRC", "Brier_score"]])}

## Test Calibration Summary

{markdown_table(test_summary[["calibration_method", "mean_absolute_calibration_error", "max_absolute_calibration_error", "deciles", "n", "events"]])}

## Test Decision Curve

{markdown_table(test_decision[["calibration_method", "threshold_probability", "alerts", "alert_rate", "ppv", "recall", "model_net_benefit", "treat_all_net_benefit", "net_benefit_advantage", "preferred_strategy"]])}

## Interpretation

Recalibration is used here to test whether the balanced logistic baseline can produce usable absolute probabilities after validation-set correction. AUROC/AUPRC mostly measure ranking, while Brier score and calibration error measure probability quality.
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    baseline = load_predictions(args.project_root)
    calibrators = fit_calibrators(baseline)
    predictions = apply_calibration(baseline, calibrators)
    metrics = metric_rows(predictions)
    deciles = calibration_deciles(predictions)
    summary = calibration_summary(deciles)
    decision = decision_curve(predictions)

    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(tables / "charls_probability_recalibration_predictions.csv", index=False)
    metrics.to_csv(tables / "charls_probability_recalibration_metrics.csv", index=False)
    deciles.to_csv(tables / "charls_probability_recalibration_deciles.csv", index=False)
    summary.to_csv(tables / "charls_probability_recalibration_summary.csv", index=False)
    decision.to_csv(tables / "charls_probability_recalibration_decision_curve.csv", index=False)
    report = write_report(args.project_root, metrics, summary, decision, calibrators)
    print(f"Wrote {report}")
    print(metrics[metrics["split"].astype(str).eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
