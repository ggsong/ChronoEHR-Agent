#!/usr/bin/env python3
"""Generate ROC, PR, and calibration figures for the diabetes demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


def roc_curve_points(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y = y_true[order].astype(int)
    pos = y.sum()
    neg = len(y) - pos
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = np.r_[0, tps / pos if pos else np.zeros_like(tps), 1]
    fpr = np.r_[0, fps / neg if neg else np.zeros_like(fps), 1]
    return fpr, tpr


def pr_curve_points(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y = y_true[order].astype(int)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(y.sum(), 1)
    return np.r_[0, recall], np.r_[1, precision]


def plot_roc(predictions: pd.DataFrame, performance: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for feature_set, part in predictions.groupby("feature_set", sort=True):
        y = part["readmission_30d"].to_numpy()
        score = part["predicted_risk"].to_numpy()
        fpr, tpr = roc_curve_points(y, score)
        auc = performance[(performance["feature_set"] == feature_set) & (performance["split"] == "test")]["AUROC"].iloc[0]
        plt.plot(fpr, tpr, linewidth=2, label=f"{feature_set} (AUROC {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve: 30-day readmission")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pr(predictions: pd.DataFrame, performance: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    baseline = predictions["readmission_30d"].mean()
    for feature_set, part in predictions.groupby("feature_set", sort=True):
        y = part["readmission_30d"].to_numpy()
        score = part["predicted_risk"].to_numpy()
        recall, precision = pr_curve_points(y, score)
        ap = performance[(performance["feature_set"] == feature_set) & (performance["split"] == "test")]["AUPRC"].iloc[0]
        plt.plot(recall, precision, linewidth=2, label=f"{feature_set} (AUPRC {ap:.3f})")
    plt.axhline(baseline, color="0.6", linestyle="--", linewidth=1, label=f"prevalence {baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve: 30-day readmission")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def calibration_table(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for feature_set, part in predictions.groupby("feature_set", sort=True):
        part = part.sort_values("predicted_risk").copy()
        part["bin"] = pd.qcut(part["predicted_risk"].rank(method="first"), q=n_bins, labels=False)
        grouped = part.groupby("bin", sort=True)
        for bin_id, bin_df in grouped:
            rows.append(
                {
                    "feature_set": feature_set,
                    "bin": int(bin_id) + 1,
                    "n": int(len(bin_df)),
                    "mean_predicted_risk": float(bin_df["predicted_risk"].mean()),
                    "observed_event_rate": float(bin_df["readmission_30d"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_calibration(calib: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for feature_set, part in calib.groupby("feature_set", sort=True):
        plt.plot(
            part["mean_predicted_risk"],
            part["observed_event_rate"],
            marker="o",
            linewidth=2,
            label=feature_set,
        )
    lo = min(calib["mean_predicted_risk"].min(), calib["observed_event_rate"].min())
    hi = max(calib["mean_predicted_risk"].max(), calib["observed_event_rate"].max())
    pad = 0.02
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed event rate")
    plt.title("Calibration by risk decile")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_report(paths: dict[str, Path], report_path: Path) -> None:
    text = f"""# MIMIC 糖尿病模型图表报告

已生成：

- ROC curve：`{paths["roc"]}`
- Precision-recall curve：`{paths["pr"]}`
- Calibration deciles：`{paths["calibration"]}`
- Calibration table：`{paths["calibration_table"]}`

这些图基于 test set 预测概率生成。图用于研究汇报和模型诊断，不代表模型已经可用于临床决策。
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables_dir = args.project_root / "outputs" / "tables"
    figures_dir = args.project_root / "outputs" / "figures"
    reports_dir = args.project_root / "outputs" / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(tables_dir / "mimic_diabetes_test_predictions.csv")
    performance = pd.read_csv(tables_dir / "mimic_diabetes_model_performance.csv")
    calib = calibration_table(predictions)

    paths = {
        "roc": figures_dir / "mimic_diabetes_roc_curve.png",
        "pr": figures_dir / "mimic_diabetes_precision_recall_curve.png",
        "calibration": figures_dir / "mimic_diabetes_calibration_deciles.png",
        "calibration_table": tables_dir / "mimic_diabetes_calibration_deciles.csv",
    }
    plot_roc(predictions, performance, paths["roc"])
    plot_pr(predictions, performance, paths["pr"])
    plot_calibration(calib, paths["calibration"])
    calib.to_csv(paths["calibration_table"], index=False)
    write_report(paths, reports_dir / "mimic_diabetes_figure_report.md")

    print("MIMIC diabetes figures generated")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()

