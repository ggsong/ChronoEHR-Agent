#!/usr/bin/env python3
"""Bootstrap confidence intervals for external benchmark test predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from mimic_diabetes_baseline import DEFAULT_PROJECT


RNG_SEED = 20260622
N_BOOTSTRAP = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score)
    sorted_y = y_true[order]
    cum_pos = np.cumsum(sorted_y)
    precision = cum_pos / (np.arange(len(sorted_y)) + 1)
    return float((precision * sorted_y).sum() / n_pos)


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    has_two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()) if len(y) else float("nan"),
        "AUROC": float(roc_auc_score(y, score)) if has_two_classes else float("nan"),
        "AUPRC": float(average_precision_score(y, score)) if has_two_classes else float("nan"),
        "Brier": float(np.mean((score - y) ** 2)),
    }


def bootstrap_ci(y: np.ndarray, score: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> dict[str, float | int]:
    n = len(y)
    aurocs: list[float] = []
    auprcs: list[float] = []
    briers: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        score_b = score[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        aurocs.append(auroc(y_b, score_b))
        auprcs.append(average_precision(y_b, score_b))
        briers.append(float(np.mean((score_b - y_b) ** 2)))

    def lower(values: list[float]) -> float:
        return float(np.nanquantile(values, 0.025)) if values else float("nan")

    def upper(values: list[float]) -> float:
        return float(np.nanquantile(values, 0.975)) if values else float("nan")

    return {
        "AUROC_lower": lower(aurocs),
        "AUROC_upper": upper(aurocs),
        "AUPRC_lower": lower(auprcs),
        "AUPRC_upper": upper(auprcs),
        "Brier_lower": lower(briers),
        "Brier_upper": upper(briers),
        "bootstrap_replicates": int(len(aurocs)),
    }


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def external_prediction_frames(project_root: Path) -> list[tuple[str, pd.DataFrame, str, str, list[str]]]:
    tables = project_root / "outputs" / "tables"
    frames: list[tuple[str, pd.DataFrame, str, str, list[str]]] = []

    cdsl = read_optional(tables / "cdsl_traditional_baselines_predictions.csv")
    if not cdsl.empty:
        frames.append(("CDSL", cdsl, "outcome", "predicted_risk", ["feature_set", "model"]))

    eicu_recalibrated = read_optional(tables / "eicu_probability_recalibration_predictions.csv")
    if not eicu_recalibrated.empty:
        frames.append(("eICU", eicu_recalibrated, "hospital_mortality", "calibrated_risk", ["prediction_time", "feature_set", "model", "calibration_method"]))
    else:
        eicu = read_optional(tables / "eicu_first24h_logistic_baseline_predictions.csv")
        if not eicu.empty:
            frames.append(("eICU", eicu, "hospital_mortality", "predicted_risk", ["prediction_time", "feature_set", "model"]))
    eicu_model_comparison = read_optional(tables / "eicu_first24h_model_comparison_predictions.csv")
    if not eicu_model_comparison.empty:
        eicu_model_comparison = eicu_model_comparison[
            eicu_model_comparison["model"].astype(str).ne("logistic_regression_balanced")
        ].copy()
        eicu_model_comparison["calibration_method"] = "raw_model_comparison"
        frames.append(("eICU", eicu_model_comparison, "hospital_mortality", "predicted_risk", ["prediction_time", "feature_set", "model", "calibration_method"]))
    comparison_recalibrated = read_optional(tables / "external_model_comparison_recalibration_predictions.csv")
    if not comparison_recalibrated.empty:
        comparison_recalibrated = comparison_recalibrated[
            comparison_recalibrated["calibration_method"].astype(str).ne("raw")
        ].copy()
        for dataset, subset in comparison_recalibrated.groupby("dataset", sort=True):
            frames.append((str(dataset), subset, "label", "calibrated_risk", ["prediction_time", "feature_set", "model", "calibration_method"]))

    charls = read_optional(tables / "charls_probability_recalibration_predictions.csv")
    if not charls.empty:
        frames.append(
            (
                "CHARLS",
                charls,
                "incident_diabetes_2013_or_2015",
                "calibrated_risk",
                ["prediction_time", "feature_set", "source_model", "calibration_method"],
            )
        )
    charls_model_comparison = read_optional(tables / "charls_incident_diabetes_model_comparison_predictions.csv")
    if not charls_model_comparison.empty:
        charls_model_comparison = charls_model_comparison[
            charls_model_comparison["model"].astype(str).ne("logistic_regression_balanced")
        ].copy()
        charls_model_comparison["calibration_method"] = "raw_model_comparison"
        frames.append(
            (
                "CHARLS",
                charls_model_comparison,
                "incident_diabetes_2013_or_2015",
                "predicted_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
            )
        )
    return frames


def build_ci(project_root: Path, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for dataset, frame, label_col, score_col, group_cols in external_prediction_frames(project_root):
        missing = [col for col in [label_col, score_col, "split", *group_cols] if col not in frame.columns]
        if missing:
            raise ValueError(f"{dataset} prediction table missing columns: {missing}")
        test = frame[frame["split"].astype(str).eq("test")].copy()
        for key, group in test.groupby(group_cols, sort=True):
            key_values = key if isinstance(key, tuple) else (key,)
            y = group[label_col].astype(int).to_numpy()
            score = group[score_col].astype(float).to_numpy()
            row = {"dataset": dataset}
            row.update({col: value for col, value in zip(group_cols, key_values)})
            if dataset == "CHARLS" and "source_model" in row:
                row["model"] = row.get("source_model", "")
            row.update(point_metrics(y, score))
            row.update(bootstrap_ci(y, score, rng, n_bootstrap))
            rows.append(row)
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


def write_report(project_root: Path, ci: pd.DataFrame, n_bootstrap: int) -> Path:
    report = project_root / "outputs" / "reports" / "external_model_bootstrap_ci.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    compact_cols = [
        "dataset",
        "feature_set",
        "model",
        "calibration_method",
        "n",
        "events",
        "AUROC",
        "AUROC_lower",
        "AUROC_upper",
        "AUPRC",
        "AUPRC_lower",
        "AUPRC_upper",
        "Brier",
        "Brier_lower",
        "Brier_upper",
        "bootstrap_replicates",
    ]
    display_cols = [col for col in compact_cols if col in ci.columns]
    report.write_text(
        f"""# External Model Bootstrap CI

- Boundary: research model evaluation only; no medical QA, diagnosis, or treatment recommendation.
- Bootstrap replicates requested: {n_bootstrap}.
- Scope: test-set uncertainty for existing external benchmark predictions; models are not refit inside bootstrap.

## Bootstrap CI

{markdown_table(ci[display_cols])}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    ci = build_ci(args.project_root, args.n_bootstrap)
    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ci.to_csv(tables / "external_model_bootstrap_ci.csv", index=False)
    report = write_report(args.project_root, ci, args.n_bootstrap)
    print(f"Wrote {report}")
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
