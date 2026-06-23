#!/usr/bin/env python3
"""Summarize CDSL calibration and decision-curve net benefit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


LABEL_COLUMN = "outcome"
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def load_predictions(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "tables" / "cdsl_traditional_baselines_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CDSL baseline predictions: {path}")
    df = pd.read_csv(path)
    required = {"PatientID", "split", "feature_set", "model", LABEL_COLUMN, "predicted_risk"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CDSL prediction file missing columns: {missing}")
    return df


def calibration_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, model, split), group in predictions.groupby(["feature_set", "model", "split"], sort=True):
        ranked = group.sort_values("predicted_risk").copy()
        ranked["decile"] = pd.qcut(ranked["predicted_risk"].rank(method="first"), 10, labels=False) + 1
        for decile, part in ranked.groupby("decile", sort=True):
            mean_pred = float(part["predicted_risk"].mean())
            observed = float(part[LABEL_COLUMN].mean())
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
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
    for (feature_set, model, split), group in deciles.groupby(["feature_set", "model", "split"], sort=True):
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
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
    events = int(df[LABEL_COLUMN].sum())
    non_events = n - events
    event_rate = events / n if n else 0.0
    weight = threshold / (1 - threshold)
    flagged = df[df["predicted_risk"].ge(threshold)]
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
    for (feature_set, model, split), group in predictions.groupby(["feature_set", "model", "split"], sort=True):
        for threshold in THRESHOLDS:
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "split": split,
                    **decision_curve_metrics(group, threshold),
                }
            )
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


def write_report(project_root: Path, summary: pd.DataFrame, decision: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "cdsl_calibration_decision_curve_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    test_summary = summary[summary["split"].astype(str).eq("test")].sort_values(
        ["mean_absolute_calibration_error", "feature_set", "model"]
    )
    test_decision = decision[decision["split"].astype(str).eq("test")].copy()
    high_value = test_decision[test_decision["net_benefit_advantage"].gt(0)].sort_values(
        ["threshold_probability", "net_benefit_advantage"], ascending=[True, False]
    )
    report.write_text(
        f"""# CDSL Calibration and Decision-Curve Summary

- Boundary: research model evaluation only; no medical QA, diagnosis, or treatment recommendation.
- Task: CDSL in-hospital mortality prediction.
- Scope: traditional model predictions across admission, 24h, 48h, and full-stay reference windows.
- Note: full-stay rows are naive upper-reference rows and should not be interpreted as early prediction performance.

## Test Calibration Summary

{markdown_table(test_summary[["feature_set", "model", "n", "events", "mean_absolute_calibration_error", "max_absolute_calibration_error"]])}

## Test Decision Rows With Positive Model Advantage

{markdown_table(high_value[["feature_set", "model", "threshold_probability", "alerts", "alert_rate", "ppv", "recall", "model_net_benefit", "treat_all_net_benefit", "net_benefit_advantage", "preferred_strategy"]])}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.project_root)
    deciles = calibration_deciles(predictions)
    summary = calibration_summary(deciles)
    decision = decision_curve(predictions)
    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    deciles.to_csv(tables / "cdsl_calibration_deciles.csv", index=False)
    summary.to_csv(tables / "cdsl_calibration_summary.csv", index=False)
    decision.to_csv(tables / "cdsl_decision_curve.csv", index=False)
    report = write_report(args.project_root, summary, decision)
    print(f"Wrote {report}")
    print(summary[summary["split"].astype(str).eq("test")].sort_values("mean_absolute_calibration_error").to_string(index=False))


if __name__ == "__main__":
    main()
