#!/usr/bin/env python3
"""Generate and score a tiny synthetic ChronoEHR-Agent demo cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--n", type=int, default=240, help="Number of synthetic admissions to generate.")
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


def make_cohort(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(35, 91, size=n)
    prior_admissions = rng.poisson(1.2, size=n)
    length_of_stay = np.clip(rng.gamma(2.2, 2.0, size=n), 0.5, 21.0)
    abnormal_labs = rng.binomial(6, np.clip((age - 30) / 100, 0.05, 0.7))
    medication_changes = rng.poisson(np.clip(prior_admissions + 1, 1, 6) / 2)
    logits = -3.2 + 0.025 * age + 0.33 * prior_admissions + 0.08 * length_of_stay + 0.18 * abnormal_labs
    probability = 1 / (1 + np.exp(-logits))
    outcome = rng.binomial(1, np.clip(probability, 0.02, 0.85))
    split = np.where(rng.random(n) < 0.75, "train", "test")
    return pd.DataFrame(
        {
            "synthetic_admission_id": np.arange(1, n + 1),
            "prediction_time": "discharge",
            "age": age,
            "prior_admissions_count": prior_admissions,
            "length_of_stay_days": np.round(length_of_stay, 2),
            "abnormal_lab_count": abnormal_labs,
            "medication_change_count": medication_changes,
            "readmission_30d": outcome,
            "split": split,
        }
    )


def score_rows(cohort: pd.DataFrame) -> pd.DataFrame:
    score = (
        -2.8
        + 0.022 * cohort["age"]
        + 0.29 * cohort["prior_admissions_count"]
        + 0.07 * cohort["length_of_stay_days"]
        + 0.16 * cohort["abnormal_lab_count"]
    )
    out = cohort.copy()
    out["synthetic_risk_score"] = 1 / (1 + np.exp(-score))
    return out


def metric_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", sort=True):
        predicted = (group["synthetic_risk_score"] >= 0.5).astype(int)
        actual = group["readmission_30d"].astype(int)
        accuracy = float((predicted == actual).mean())
        event_rate = float(actual.mean())
        rows.append(
            {
                "split": split,
                "n": int(len(group)),
                "event_rate": round(event_rate, 4),
                "mean_risk_score": round(float(group["synthetic_risk_score"].mean()), 4),
                "threshold_0_50_accuracy": round(accuracy, 4),
            }
        )
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, report_path: Path) -> None:
    table_lines = ["| split | n | event_rate | mean_risk_score | threshold_0_50_accuracy |", "|---|---:|---:|---:|---:|"]
    for row in metrics.to_dict(orient="records"):
        table_lines.append(
            f"| {row['split']} | {row['n']} | {row['event_rate']} | {row['mean_risk_score']} | {row['threshold_0_50_accuracy']} |"
        )
    lines = [
        "# Synthetic ChronoEHR Demo",
        "",
        "This report was generated from artificial data only.",
        "",
        *table_lines,
        "",
        "The demo verifies that the public repository can generate, score, and validate a temporal prediction artifact without controlled clinical data.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.project_root / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = make_cohort(args.n, args.seed)
    scored = score_rows(cohort)
    metrics = metric_rows(scored)
    scored.to_csv(out_dir / "synthetic_cohort.csv", index=False)
    metrics.to_csv(out_dir / "synthetic_demo_metrics.csv", index=False)
    write_report(metrics, out_dir / "synthetic_demo_report.md")
    print(f"Wrote synthetic demo artifacts to {out_dir}")


if __name__ == "__main__":
    main()
