#!/usr/bin/env python3
"""Generate eICU first-24h baseline ROC/PR/calibration figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT
from mimic_diabetes_figures import pr_curve_points, roc_curve_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def calibration_deciles(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for (model, feature_set), group in predictions.groupby(["model", "feature_set"], sort=False):
        ranked = group.sort_values("predicted_risk").copy()
        ranked["decile"] = pd.qcut(ranked["predicted_risk"].rank(method="first"), q=n_bins, labels=False) + 1
        for decile, part in ranked.groupby("decile", sort=True):
            mean_pred = float(part["predicted_risk"].mean())
            obs_rate = float(part["hospital_mortality"].mean())
            rows.append(
                {
                    "study": "eicu_temporal_mortality",
                    "model": model,
                    "feature_set": feature_set,
                    "prediction_time": "first_24h",
                    "decile": int(decile),
                    "n": int(len(part)),
                    "mean_predicted_risk": mean_pred,
                    "observed_event_rate": obs_rate,
                    "absolute_calibration_error": abs(mean_pred - obs_rate),
                }
            )
    return pd.DataFrame(rows)


def calibration_summary(deciles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature_set), group in deciles.groupby(["model", "feature_set"], sort=False):
        weighted = float((group["absolute_calibration_error"] * group["n"]).sum() / group["n"].sum())
        rows.append(
            {
                "study": "eicu_temporal_mortality",
                "model": model,
                "feature_set": feature_set,
                "prediction_time": "first_24h",
                "mean_absolute_calibration_error": weighted,
                "max_absolute_calibration_error": float(group["absolute_calibration_error"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_roc(predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    test_metrics = metrics[metrics["split"].eq("test")].iloc[0]
    y = predictions["hospital_mortality"].astype(int).to_numpy()
    score = predictions["predicted_risk"].to_numpy()
    fpr, tpr = roc_curve_points(y, score)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot(fpr, tpr, linewidth=2.2, color="#2F6F8F", label=f"Logistic (AUROC {test_metrics['AUROC']:.3f})")
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("eICU First-24h Hospital Mortality ROC")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_pr(predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    test_metrics = metrics[metrics["split"].eq("test")].iloc[0]
    y = predictions["hospital_mortality"].astype(int).to_numpy()
    score = predictions["predicted_risk"].to_numpy()
    recall, precision = pr_curve_points(y, score)
    baseline = float(predictions["hospital_mortality"].mean())
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot(recall, precision, linewidth=2.2, color="#B85C38", label=f"Logistic (AUPRC {test_metrics['AUPRC']:.3f})")
    ax.axhline(baseline, color="#777777", linestyle="--", linewidth=1, label=f"prevalence {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("eICU First-24h Hospital Mortality Precision-Recall")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_calibration(deciles: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot(
        deciles["mean_predicted_risk"],
        deciles["observed_event_rate"],
        marker="o",
        linewidth=2.2,
        color="#3E7C59",
        label="Logistic deciles",
    )
    lo = min(float(deciles["mean_predicted_risk"].min()), float(deciles["observed_event_rate"].min()), 0.0)
    hi = max(float(deciles["mean_predicted_risk"].max()), float(deciles["observed_event_rate"].max()), 1.0)
    ax.plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("eICU First-24h Calibration By Decile")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, metrics: pd.DataFrame, summary: pd.DataFrame, outputs: dict[str, Path]) -> Path:
    report_path = project_root / "outputs" / "reports" / "eicu_baseline_figures_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    test = metrics[metrics["split"].eq("test")]
    report_path.write_text(
        f"""# eICU Baseline Figures And Calibration

- Boundary: research benchmark visualization only; no clinical recommendation.
- Task: eICU first-24h hospital mortality prediction.
- Model: balanced logistic regression.

## Test Metrics

{markdown_table(test[["n", "events", "event_rate", "AUROC", "AUPRC", "Brier_score", "feature_count"]])}

## Calibration Summary

{markdown_table(summary)}

## Outputs

- ROC: `{outputs["roc"].relative_to(project_root)}`
- Precision-recall: `{outputs["pr"].relative_to(project_root)}`
- Calibration: `{outputs["calibration"].relative_to(project_root)}`
- Calibration deciles: `outputs/tables/eicu_first24h_calibration_deciles.csv`
- Calibration summary: `outputs/tables/eicu_first24h_calibration_summary.csv`
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    tables = args.project_root / "outputs" / "tables"
    figures = args.project_root / "outputs" / "figures"
    metrics = pd.read_csv(tables / "eicu_first24h_logistic_baseline_metrics.csv")
    predictions = pd.read_csv(tables / "eicu_first24h_logistic_baseline_predictions.csv")
    test_predictions = predictions[predictions["split"].eq("test")].copy()
    deciles = calibration_deciles(test_predictions)
    summary = calibration_summary(deciles)
    outputs = {
        "roc": figures / "eicu_first24h_logistic_roc.png",
        "pr": figures / "eicu_first24h_logistic_precision_recall.png",
        "calibration": figures / "eicu_first24h_logistic_calibration_deciles.png",
    }
    plot_roc(test_predictions, metrics, outputs["roc"])
    plot_pr(test_predictions, metrics, outputs["pr"])
    plot_calibration(deciles, outputs["calibration"])
    deciles.to_csv(tables / "eicu_first24h_calibration_deciles.csv", index=False)
    summary.to_csv(tables / "eicu_first24h_calibration_summary.csv", index=False)
    report = write_report(args.project_root, metrics, summary, outputs)
    print(f"Wrote {report}")
    for output in outputs.values():
        print(output)


if __name__ == "__main__":
    main()
