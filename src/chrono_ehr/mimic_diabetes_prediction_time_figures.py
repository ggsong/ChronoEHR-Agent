#!/usr/bin/env python3
"""Generate figures for prediction-time model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


ORDER = [
    "admission_safe_minimal",
    "inhospital_24h_lab_minimal",
    "inhospital_24h_lab_med_minimal",
    "discharge_safe_minimal",
]

LABELS = {
    "admission_safe_minimal": "Admission",
    "inhospital_24h_lab_minimal": "24h labs",
    "inhospital_24h_lab_med_minimal": "24h labs+meds",
    "discharge_safe_minimal": "Discharge",
}


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    tests = df[df["split"].eq("test")].copy()
    tests["sort_order"] = tests["feature_set"].map({name: i for i, name in enumerate(ORDER)}).fillna(99)
    tests = tests.sort_values(["sort_order", "feature_set"])
    labels = [LABELS.get(name, name) for name in tests["feature_set"]]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, tests[metric], color=["#4C78A8", "#59A14F", "#F28E2B", "#E15759"])
    plt.ylim(max(0, tests[metric].min() - 0.02), min(1, tests[metric].max() + 0.02))
    plt.ylabel(metric)
    plt.title(f"Prediction-time comparison: {metric}")
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, tests[metric]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_report(paths: dict[str, Path], output: Path) -> None:
    text = f"""# Prediction-Time Figure Report

已生成 prediction-time model comparison 图：

- AUROC bar chart: `{paths["auroc"]}`
- AUPRC bar chart: `{paths["auprc"]}`

这些图用于展示 admission、inhospital 24h 和 discharge 三个预测时间点下的模型表现差异。
"""
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.project_root / "outputs" / "tables"
    figures = args.project_root / "outputs" / "figures"
    reports = args.project_root / "outputs" / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tables / "mimic_diabetes_prediction_time_model_performance.csv")
    paths = {
        "auroc": figures / "mimic_diabetes_prediction_time_auroc.png",
        "auprc": figures / "mimic_diabetes_prediction_time_auprc.png",
    }
    plot_metric(df, "AUROC", paths["auroc"])
    plot_metric(df, "AUPRC", paths["auprc"])
    write_report(paths, reports / "mimic_diabetes_prediction_time_figure_report.md")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
