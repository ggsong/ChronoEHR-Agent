#!/usr/bin/env python3
"""Generate manuscript-oriented summary figures from completed ChronoEHR tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


COHORT_EN = {
    "糖尿病": "Diabetes",
    "CKD": "CKD",
    "心衰": "Heart failure",
    "高血压": "Hypertension",
}

STAGE_EN = {
    "inhospital_24h": "24h",
    "discharge": "Discharge",
}


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E8EC", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_decision_curve(project_root: Path, figures_dir: Path) -> Path:
    path = project_root / "outputs" / "tables" / "chronic_disease_decision_curve.csv"
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    colors = {
        "Diabetes": "#2F6FB0",
        "CKD": "#5A9E6F",
        "Heart failure": "#B45A5A",
        "Hypertension": "#7B6AB8",
    }
    linestyles = {"24h": "-", "Discharge": "--"}
    for (cohort_label, prediction_time), group in df.groupby(["cohort_label", "prediction_time"], sort=False):
        cohort = COHORT_EN.get(str(cohort_label), str(cohort_label))
        stage = STAGE_EN.get(str(prediction_time), str(prediction_time))
        label = f"{cohort} {stage}"
        ax.plot(
            group["threshold_probability"],
            group["model_net_benefit"],
            label=label,
            color=colors.get(cohort, "#333333"),
            linestyle=linestyles.get(stage, "-"),
            linewidth=1.8,
        )
    treat_all = df.groupby("threshold_probability", as_index=False)["treat_all_net_benefit"].mean()
    ax.plot(
        treat_all["threshold_probability"],
        treat_all["treat_all_net_benefit"],
        label="Treat all (mean)",
        color="#777777",
        linestyle=":",
        linewidth=2.0,
    )
    ax.axhline(0, color="#333333", linewidth=1.0, label="Treat none")
    ax.set_title("Decision-curve net benefit for final models")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.05, 0.50)
    style_axes(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    output = figures_dir / "chronic_disease_decision_curve_net_benefit.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_subgroup_auroc(project_root: Path, figures_dir: Path) -> Path:
    path = project_root / "outputs" / "tables" / "chronic_disease_subgroup_performance.csv"
    df = pd.read_csv(path)
    summary = (
        df.groupby("subgroup_variable", as_index=False)
        .agg(mean_AUROC=("AUROC", "mean"), mean_AUPRC=("AUPRC", "mean"), rows=("subgroup_value", "count"))
        .sort_values("mean_AUROC", ascending=True)
    )
    labels = summary["subgroup_variable"].str.replace("_", " ").tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(labels, summary["mean_AUROC"], color="#3E7CB1")
    for y, (_, row) in enumerate(summary.iterrows()):
        ax.text(row["mean_AUROC"] + 0.004, y, f"AUROC {row['mean_AUROC']:.3f} / AUPRC {row['mean_AUPRC']:.3f}", va="center", fontsize=8)
    ax.set_title("Mean subgroup performance by subgroup variable")
    ax.set_xlabel("Mean AUROC")
    ax.set_xlim(0.55, max(0.75, float(summary["mean_AUROC"].max()) + 0.05))
    style_axes(ax)
    fig.tight_layout()
    output = figures_dir / "chronic_disease_subgroup_mean_auroc.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_subgroup_event_ppv(project_root: Path, figures_dir: Path) -> Path:
    path = project_root / "outputs" / "tables" / "chronic_disease_subgroup_performance.csv"
    df = pd.read_csv(path)
    variables = sorted(df["subgroup_variable"].unique())
    colors = {
        "admission_type_group": "#8767A6",
        "age_group": "#3E7CB1",
        "gender_group": "#5A9E6F",
        "prior_admission_group": "#B45A5A",
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for variable in variables:
        group = df[df["subgroup_variable"].eq(variable)]
        ax.scatter(
            group["event_rate"],
            group["top10_ppv"],
            s=(group["n"] / group["n"].max()).clip(lower=0.10) * 120,
            alpha=0.72,
            label=variable.replace("_", " "),
            color=colors.get(variable, "#555555"),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.plot([0.10, 0.40], [0.10, 0.40], color="#777777", linestyle=":", linewidth=1.2, label="PPV = event rate")
    ax.set_title("Subgroup event rate versus top 10% PPV")
    ax.set_xlabel("Subgroup event rate")
    ax.set_ylabel("Top 10% PPV")
    ax.set_xlim(0.10, 0.40)
    ax.set_ylim(0.15, 0.70)
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    output = figures_dir / "chronic_disease_subgroup_event_rate_top10_ppv.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def write_report(outputs: list[Path], report: Path) -> None:
    lines = [
        "# Summary Figure Export",
        "",
        "这些图由已完成的 ChronoEHR summary tables 生成，不重新训练模型，也不读取原始 EHR 大表。",
        "",
        "## Figures",
        "",
    ]
    for output in outputs:
        lines.append(f"- `{output}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Decision-curve 图用于补充 AUROC/AUPRC，展示不同研究阈值下的 net benefit。",
            "- Subgroup 图用于查看模型表现异质性，不代表因果解释，也不是临床诊疗建议。",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_decision_curve(args.project_root, figures_dir),
        plot_subgroup_auroc(args.project_root, figures_dir),
        plot_subgroup_event_ppv(args.project_root, figures_dir),
    ]
    report = args.project_root / "outputs" / "reports" / "chronic_disease_summary_figures_report.md"
    write_report(outputs, report)
    print(f"Wrote {len(outputs)} figures")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
