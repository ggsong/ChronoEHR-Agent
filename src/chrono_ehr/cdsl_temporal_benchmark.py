#!/usr/bin/env python3
"""Run a lightweight CDSL temporal benchmark for ChronoEHR-Agent."""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cdsl_external_validation_readiness import CDSL_CANDIDATE_ROOTS, choose_cdsl_root
from mimic_diabetes_baseline import DEFAULT_PROJECT


ID_COL = "PatientID"
TIME_COL = "RecordTime"
ADMIT_COL = "AdmissionTime"
DISCHARGE_COL = "DischargeTime"
OUTCOME_COL = "Outcome"
LOS_COL = "LOS"
BASELINE_COLS = ["Sex", "Age"]
FORBIDDEN_FEATURES = {ID_COL, TIME_COL, ADMIT_COL, DISCHARGE_COL, OUTCOME_COL, LOS_COL}


WINDOWS = [
    {
        "feature_set": "admission_demographics",
        "hours_after_admission": 0,
        "use_timevarying": False,
        "interpretation": "Only age and sex; safe for admission-time prediction.",
    },
    {
        "feature_set": "first_24h_vitals_labs",
        "hours_after_admission": 24,
        "use_timevarying": True,
        "interpretation": "Use records from admission through first 24 hours.",
    },
    {
        "feature_set": "first_48h_vitals_labs",
        "hours_after_admission": 48,
        "use_timevarying": True,
        "interpretation": "Use records from admission through first 48 hours.",
    },
    {
        "feature_set": "full_stay_naive_reference",
        "hours_after_admission": None,
        "use_timevarying": True,
        "interpretation": "Uses all in-stay records before discharge; useful as a naive reference, not an admission-time model.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--cdsl-root",
        type=Path,
        help="Optional explicit CDSL root. Defaults to the first usable known local CDSL path.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.05, help="Minimum train-set coverage for time-varying features.")
    return parser.parse_args()


def load_fold_pids(root: Path, fold: int = 0) -> dict[str, set[str]]:
    fold_dir = root / "processed" / f"fold_{fold}"
    result = {}
    for split in ["train", "val", "test"]:
        with (fold_dir / f"{split}_pid.pkl").open("rb") as handle:
            result[split] = {str(value) for value in pickle.load(handle)}
    return result


def load_formatted(root: Path) -> pd.DataFrame:
    path = root / "processed" / "cdsl_dataset_formatted.csv"
    df = pd.read_csv(path, low_memory=False)
    df[ID_COL] = df[ID_COL].astype(str)
    for col in [TIME_COL, ADMIT_COL, DISCHARGE_COL]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors="coerce").fillna(0).astype(int)
    return df


def patient_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = (
        df.groupby(ID_COL)
        .agg(
            outcome=(OUTCOME_COL, "max"),
            admission_time=(ADMIT_COL, "min"),
            discharge_time=(DISCHARGE_COL, "max"),
            n_records=(TIME_COL, "size"),
        )
        .reset_index()
    )
    return labels


def make_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df.sort_values([ID_COL, TIME_COL]).groupby(ID_COL)[BASELINE_COLS].first().reset_index()
    for col in BASELINE_COLS:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
    return baseline


def aggregate_timevarying(df: pd.DataFrame, labels: pd.DataFrame, window: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = [col for col in df.columns if col not in FORBIDDEN_FEATURES and col not in BASELINE_COLS]
    joined = df[[ID_COL, TIME_COL, ADMIT_COL, DISCHARGE_COL] + feature_cols].copy()
    hours = window["hours_after_admission"]
    if hours is None:
        mask = joined[TIME_COL].notna() & joined[DISCHARGE_COL].notna() & (joined[TIME_COL] <= joined[DISCHARGE_COL])
        prediction_label = "before discharge"
    else:
        cutoff = joined[ADMIT_COL] + pd.to_timedelta(hours, unit="h")
        mask = joined[TIME_COL].notna() & joined[ADMIT_COL].notna() & (joined[TIME_COL] >= joined[ADMIT_COL]) & (joined[TIME_COL] <= cutoff)
        prediction_label = f"admission + {hours}h"
    eligible = joined.loc[mask, [ID_COL, TIME_COL] + feature_cols].sort_values([ID_COL, TIME_COL])

    numeric = eligible[[ID_COL] + feature_cols].copy()
    for col in feature_cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    means = numeric.groupby(ID_COL)[feature_cols].mean().add_prefix("mean_")
    lasts = numeric.groupby(ID_COL)[feature_cols].last().add_prefix("last_")
    counts = numeric.groupby(ID_COL)[feature_cols].count().add_prefix("count_")
    features = pd.concat([means, lasts, counts], axis=1).reset_index()

    excluded_future = int((~mask & joined[TIME_COL].notna()).sum())
    rows_before_admission = int((joined[TIME_COL].notna() & joined[ADMIT_COL].notna() & (joined[TIME_COL] < joined[ADMIT_COL])).sum())
    summary = {
        "prediction_label": prediction_label,
        "eligible_records": len(eligible),
        "patients_with_window_records": int(eligible[ID_COL].nunique()),
        "excluded_records_outside_window": excluded_future,
        "records_before_admission": rows_before_admission,
        "raw_timevarying_columns": len(feature_cols),
    }
    return features, summary


def build_feature_matrix(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    window: dict[str, Any],
    train_ids: set[str],
    min_coverage: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = make_baseline_features(df)
    features = labels[[ID_COL, "outcome"]].merge(baseline, on=ID_COL, how="left")
    summary: dict[str, Any] = {
        "feature_set": window["feature_set"],
        "interpretation": window["interpretation"],
        "selected_feature_count": len(BASELINE_COLS),
    }

    if window["use_timevarying"]:
        timevarying, window_summary = aggregate_timevarying(df, labels, window)
        features = features.merge(timevarying, on=ID_COL, how="left")
        summary.update(window_summary)

        candidate_cols = [col for col in features.columns if col not in {ID_COL, "outcome"}]
        train_mask = features[ID_COL].isin(train_ids)
        coverage = features.loc[train_mask, candidate_cols].notna().mean()
        keep_cols = [col for col in candidate_cols if coverage.get(col, 0.0) >= min_coverage or col in BASELINE_COLS]
        features = features[[ID_COL, "outcome"] + keep_cols]
        summary["selected_feature_count"] = len(keep_cols)
        summary["dropped_low_coverage_features"] = len(candidate_cols) - len(keep_cols)
    else:
        summary.update(
            {
                "prediction_label": "admission",
                "eligible_records": None,
                "patients_with_window_records": None,
                "excluded_records_outside_window": None,
                "records_before_admission": int(
                    (df[TIME_COL].notna() & df[ADMIT_COL].notna() & (df[TIME_COL] < df[ADMIT_COL])).sum()
                ),
                "raw_timevarying_columns": 0,
                "dropped_low_coverage_features": 0,
            }
        )
    return features, summary


def evaluate_split(model: Pipeline, X: pd.DataFrame, y: pd.Series, split: str) -> dict[str, Any]:
    proba = model.predict_proba(X)[:, 1]
    return {
        "split": split,
        "n": len(y),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "AUROC": float(roc_auc_score(y, proba)) if y.nunique() == 2 else np.nan,
        "AUPRC": float(average_precision_score(y, proba)) if y.nunique() == 2 else np.nan,
        "Brier": float(brier_score_loss(y, proba)),
    }


def run_window(features: pd.DataFrame, pids: dict[str, set[str]], feature_set: str) -> list[dict[str, Any]]:
    rows = []
    feature_cols = [col for col in features.columns if col not in {ID_COL, "outcome"}]
    train = features[features[ID_COL].isin(pids["train"])].copy()
    val = features[features[ID_COL].isin(pids["val"])].copy()
    test = features[features[ID_COL].isin(pids["test"])].copy()

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train[feature_cols], train["outcome"])
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        row = evaluate_split(model, split_df[feature_cols], split_df["outcome"], split_name)
        row["feature_set"] = feature_set
        row["model"] = "logistic_regression_balanced"
        row["feature_count"] = len(feature_cols)
        rows.append(row)
    return rows


def write_report(project_root: Path, root: Path, metrics: pd.DataFrame, window_summary: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "cdsl_temporal_benchmark_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    test = metrics[metrics["split"].eq("test")].copy()
    lines = [
        "# CDSL Temporal Benchmark Report",
        "",
        f"- CDSL root: `{root}`",
        "- Task: CDSL in-hospital mortality prediction",
        "- Model: class-balanced logistic regression",
        "- Purpose: external EHR time-aware benchmark, not chronic readmission validation.",
        "",
        "## 关键解释",
        "",
        "这个结果用来验证 ChronoEHR-Agent 的时间点处理能力：不同预测时间点只能使用对应窗口内已经发生的记录。"
        "它不回答糖尿病 30 天再入院问题，也不提供临床诊疗建议。",
        "",
        "## Test Metrics",
        "",
        markdown_table(test[["feature_set", "n", "events", "event_rate", "AUROC", "AUPRC", "Brier", "feature_count"]]),
        "",
        "## Window Audit",
        "",
        markdown_table(window_summary),
        "",
        "## 读法",
        "",
        "- `admission_demographics` 是最保守的入院时 baseline，只用年龄和性别。",
        "- `first_24h_vitals_labs` 和 `first_48h_vitals_labs` 展示住院早期信息增加后性能如何变化。",
        "- `full_stay_naive_reference` 使用全住院记录，不能作为入院时模型，只能作为 naive reference；如果性能明显更高，需要在报告里解释这是未来信息窗口带来的优势。",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    args = parse_args()
    root, _ = choose_cdsl_root(args.cdsl_root)
    if root is None:
        raise SystemExit(f"No usable CDSL root found. Checked: {', '.join(str(p) for p in CDSL_CANDIDATE_ROOTS)}")

    df = load_formatted(root)
    labels = patient_labels(df)
    pids = load_fold_pids(root, fold=0)

    metric_rows = []
    summary_rows = []
    feature_dir = args.project_root / "data" / "processed" / "cdsl_temporal_benchmark"
    feature_dir.mkdir(parents=True, exist_ok=True)

    for window in WINDOWS:
        features, summary = build_feature_matrix(df, labels, window, pids["train"], args.min_coverage)
        feature_path = feature_dir / f"{window['feature_set']}.csv"
        features.to_csv(feature_path, index=False)
        summary["feature_file"] = str(feature_path.relative_to(args.project_root))
        summary_rows.append(summary)
        metric_rows.extend(run_window(features, pids, window["feature_set"]))

    metrics = pd.DataFrame(metric_rows)
    window_summary = pd.DataFrame(summary_rows)

    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "cdsl_temporal_benchmark_metrics.csv", index=False)
    window_summary.to_csv(tables / "cdsl_temporal_benchmark_window_audit.csv", index=False)
    report = write_report(args.project_root, root, metrics, window_summary)
    print(f"Wrote {report}")
    print(metrics[metrics["split"].eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
