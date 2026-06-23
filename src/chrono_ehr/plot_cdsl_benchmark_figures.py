#!/usr/bin/env python3
"""Plot CDSL temporal benchmark summary figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


ORDER = [
    "admission_demographics",
    "first_24h_vitals_labs",
    "first_48h_vitals_labs",
    "full_stay_naive_reference",
]
LABELS = {
    "admission_demographics": "Admission\nage+sex",
    "first_24h_vitals_labs": "First 24h\nvitals+labs",
    "first_48h_vitals_labs": "First 48h\nvitals+labs",
    "full_stay_naive_reference": "Full stay\nreference",
}
MODEL_LABELS = {
    "logistic_regression_balanced": "Logistic",
    "random_forest_balanced": "Random Forest",
    "hist_gradient_boosting_weighted": "HistGB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def plot_metric(test: pd.DataFrame, metric: str, output: Path) -> None:
    pivot = test.pivot(index="feature_set", columns="model", values=metric).reindex(ORDER)
    pivot = pivot.rename(index=LABELS, columns=MODEL_LABELS)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = ["#2F6F8F", "#B85C38", "#3E7C59"]
    pivot.plot(kind="bar", ax=ax, width=0.78, color=colors[: len(pivot.columns)], edgecolor="#2f2f2f", linewidth=0.4)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_ylim(0, max(1.06, float(pivot.max().max()) + 0.08))
    ax.set_title(f"CDSL Temporal Benchmark: Test {metric}", pad=14)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))

    for idx, patch in enumerate(ax.patches):
        height = patch.get_height()
        if pd.notna(height):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axvspan(2.5, 3.5, color="#f1f1f1", alpha=0.65, zorder=0)
    ax.axvline(2.5, color="#5c5c5c", linestyle="--", linewidth=1)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_report(project_root: Path, outputs: list[Path]) -> Path:
    report = project_root / "outputs" / "reports" / "cdsl_summary_figures_report.md"
    lines = [
        "# CDSL Summary Figures Report",
        "",
        "Generated figures for the CDSL external time-aware benchmark. These figures are for method validation, not for clinical decision support.",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path.relative_to(project_root)}`")
    lines.extend(
        [
            "",
            "解释：虚线右侧的 `full_stay_naive_reference` 使用全住院窗口，不能解释为入院时或 24 小时早期预测性能。",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    metrics_path = args.project_root / "outputs" / "tables" / "cdsl_traditional_baselines_metrics.csv"
    if not metrics_path.exists():
        raise SystemExit("Missing CDSL baseline metrics. Run --cdsl-traditional-baselines first.")
    metrics = pd.read_csv(metrics_path)
    test = metrics[metrics["split"].eq("test")].copy()
    figures = args.project_root / "outputs" / "figures"
    outputs = [
        figures / "cdsl_temporal_benchmark_auroc.png",
        figures / "cdsl_temporal_benchmark_auprc.png",
    ]
    plot_metric(test, "AUROC", outputs[0])
    plot_metric(test, "AUPRC", outputs[1])
    report = write_report(args.project_root, outputs)
    print(f"Wrote {report}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
