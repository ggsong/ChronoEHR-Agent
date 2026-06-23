#!/usr/bin/env python3
"""Run dependency-light logistic baselines for the diabetes demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]

BASE_NUMERIC_FEATURES = [
    "anchor_age",
    "ed_los_hours",
    "length_of_stay_days",
    "prior_admissions_count",
    "days_since_prior_discharge",
]

LAB_NUMERIC_FEATURES = [
    "lab_glucose_count",
    "lab_glucose_mean",
    "lab_glucose_min",
    "lab_glucose_max",
    "lab_glucose_last",
    "lab_glucose_abnormal_count",
    "lab_glucose_has",
    "lab_hba1c_count",
    "lab_hba1c_mean",
    "lab_hba1c_min",
    "lab_hba1c_max",
    "lab_hba1c_last",
    "lab_hba1c_abnormal_count",
    "lab_hba1c_has",
    "lab_hba1c_absolute_count",
    "lab_hba1c_absolute_mean",
    "lab_hba1c_absolute_min",
    "lab_hba1c_absolute_max",
    "lab_hba1c_absolute_last",
    "lab_hba1c_absolute_abnormal_count",
    "lab_hba1c_absolute_has",
    "lab_creatinine_count",
    "lab_creatinine_mean",
    "lab_creatinine_min",
    "lab_creatinine_max",
    "lab_creatinine_last",
    "lab_creatinine_abnormal_count",
    "lab_creatinine_has",
    "lab_bun_count",
    "lab_bun_mean",
    "lab_bun_min",
    "lab_bun_max",
    "lab_bun_last",
    "lab_bun_abnormal_count",
    "lab_bun_has",
]

MED_NUMERIC_FEATURES = [
    "med_insulin_count",
    "med_insulin_has",
    "med_metformin_count",
    "med_metformin_has",
    "med_sulfonylurea_count",
    "med_sulfonylurea_has",
    "med_dpp4_count",
    "med_dpp4_has",
    "med_tzd_count",
    "med_tzd_has",
    "med_glp1_count",
    "med_glp1_has",
    "med_sglt2_count",
    "med_sglt2_has",
    "med_alpha_glucosidase_count",
    "med_alpha_glucosidase_has",
    "med_any_diabetes_count",
    "med_any_diabetes_has",
]

CATEGORICAL_FEATURES = [
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "marital_status",
    "race",
]

FORBIDDEN_COLUMNS = {
    "readmission_30d",
    "next_admittime",
    "next_hadm_id",
    "days_to_next_admission",
    "postdischarge_death_within_30d",
}


def load_cohort(
    project_root: Path,
    include_labs: bool,
    include_meds: bool,
    extra_feature_files: list[str] | None = None,
) -> pd.DataFrame:
    cohort_path = project_root / "data" / "processed" / "mimic_diabetes_readmission_cohort.csv"
    cohort = pd.read_csv(cohort_path, low_memory=False)

    if include_labs:
        lab_path = project_root / "data" / "processed" / "mimic_diabetes_lab_features.csv"
        if not lab_path.exists():
            raise FileNotFoundError(
                f"Lab feature file not found: {lab_path}. Run src/chrono_ehr/mimic_diabetes_lab_features.py first."
            )
        labs = pd.read_csv(lab_path, low_memory=False)
        cohort = cohort.merge(labs, on="hadm_id", how="left")

    if include_meds:
        med_path = project_root / "data" / "processed" / "mimic_diabetes_med_features.csv"
        if not med_path.exists():
            raise FileNotFoundError(
                f"Medication feature file not found: {med_path}. Run src/chrono_ehr/mimic_diabetes_med_features.py first."
            )
        meds = pd.read_csv(med_path, low_memory=False)
        cohort = cohort.merge(meds, on="hadm_id", how="left")

    for relative_path in extra_feature_files or []:
        feature_path = project_root / relative_path
        if not feature_path.exists():
            raise FileNotFoundError(f"Extra feature file not found: {feature_path}")
        features = pd.read_csv(feature_path, low_memory=False)
        cohort = cohort.merge(features, on="hadm_id", how="left")

    return cohort


def available_feature_sets(project_root: Path) -> list[dict]:
    lab_path = project_root / "data" / "processed" / "mimic_diabetes_lab_features.csv"
    med_path = project_root / "data" / "processed" / "mimic_diabetes_med_features.csv"
    sets = [
        {
            "feature_set": "minimal",
            "include_labs": False,
            "include_meds": False,
            "numeric_features": BASE_NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }
    ]
    if lab_path.exists():
        sets.append(
            {
                "feature_set": "lab_augmented",
                "include_labs": True,
                "include_meds": False,
                "numeric_features": BASE_NUMERIC_FEATURES + LAB_NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
            }
        )
    if lab_path.exists() and med_path.exists():
        sets.append(
            {
                "feature_set": "lab_med_augmented",
                "include_labs": True,
                "include_meds": True,
                "numeric_features": BASE_NUMERIC_FEATURES + LAB_NUMERIC_FEATURES + MED_NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
            }
        )
    return sets


def fit_preprocessor(
    train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    min_category_count: int = 50,
) -> dict:
    numeric_stats = {}
    for col in numeric_features:
        values = pd.to_numeric(train[col], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        mean = float(filled.mean())
        std = float(filled.std())
        if not np.isfinite(std) or std == 0:
            std = 1.0
        numeric_stats[col] = {"median": median, "mean": mean, "std": std}

    categorical_levels = {}
    for col in categorical_features:
        values = train[col].astype("string").fillna("MISSING")
        counts = values.value_counts(dropna=False)
        levels = [str(idx) for idx, count in counts.items() if count >= min_category_count]
        if "MISSING" not in levels:
            levels.append("MISSING")
        levels.append("OTHER")
        categorical_levels[col] = levels

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_stats": numeric_stats,
        "categorical_levels": categorical_levels,
        "min_category_count": min_category_count,
    }


def transform(df: pd.DataFrame, preprocessor: dict) -> tuple[np.ndarray, list[str]]:
    parts = []
    names = []

    for col in preprocessor["numeric_features"]:
        stats = preprocessor["numeric_stats"][col]
        values = pd.to_numeric(df[col], errors="coerce")
        missing = values.isna().astype(float).to_numpy()
        filled = values.fillna(stats["median"])
        scaled = ((filled - stats["mean"]) / stats["std"]).astype(float).to_numpy()
        parts.append(scaled.reshape(-1, 1))
        names.append(col)
        parts.append(missing.reshape(-1, 1))
        names.append(f"{col}_missing")

    for col in preprocessor["categorical_features"]:
        levels = preprocessor["categorical_levels"][col]
        allowed = set(levels)
        values = df[col].astype("string").fillna("MISSING").astype(str)
        values = values.where(values.isin(allowed), "OTHER")
        for level in levels:
            parts.append(values.eq(level).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"{col}={level}")

    x = np.hstack(parts).astype(float)
    intercept = np.ones((x.shape[0], 1), dtype=float)
    return np.hstack([intercept, x]), ["intercept", *names]


def train_logistic_regression(x: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> np.ndarray:
    n_features = x.shape[1]

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ weights
        probs = expit(logits)
        eps = 1e-12
        nll = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        penalty = 0.5 * l2 * np.sum(weights[1:] ** 2)
        grad = (x.T @ (probs - y)) / len(y)
        grad[1:] += l2 * weights[1:]
        return nll + penalty, grad

    result = minimize(
        fun=lambda w: objective(w)[0],
        x0=np.zeros(n_features, dtype=float),
        jac=lambda w: objective(w)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-8},
    )
    if not result.success:
        print(f"WARNING: optimizer ended with status: {result.message}")
    return result.x


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    sorted_y = y_true[order]
    cum_pos = np.cumsum(sorted_y)
    precision = cum_pos / (np.arange(len(sorted_y)) + 1)
    return float((precision * sorted_y).sum() / n_pos)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    y_true = y_true.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {
        "threshold": float(threshold),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def evaluate_split(
    feature_set: str,
    split_name: str,
    df: pd.DataFrame,
    weights: np.ndarray,
    preprocessor: dict,
    threshold: float,
) -> dict:
    x, _ = transform(df, preprocessor)
    y = df["readmission_30d"].astype(int).to_numpy()
    score = expit(x @ weights)
    metrics = {
        "feature_set": feature_set,
        "model": "logistic_regression_scipy_l2",
        "split": split_name,
        "n": int(len(df)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "AUROC": auroc(y, score),
        "AUPRC": average_precision(y, score),
        "Brier_score": float(np.mean((score - y) ** 2)),
        "mean_predicted_risk": float(score.mean()),
    }
    metrics.update(binary_metrics(y, score, threshold))
    return metrics


def choose_threshold_by_train_prevalence(train_scores: np.ndarray, train_y: np.ndarray) -> float:
    event_rate = float(train_y.mean())
    return float(np.quantile(train_scores, 1 - event_rate))


def validate_no_forbidden_features(feature_names: list[str]) -> None:
    offenders = [name for name in feature_names if name in FORBIDDEN_COLUMNS]
    if offenders:
        raise ValueError(f"Forbidden leakage columns found in features: {offenders}")


def run_feature_set(project: Path, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = load_cohort(
        project,
        include_labs=spec["include_labs"],
        include_meds=spec["include_meds"],
        extra_feature_files=spec.get("extra_feature_files"),
    )
    missing = [col for col in spec["numeric_features"] + spec["categorical_features"] if col not in cohort.columns]
    if missing:
        raise ValueError(f"Missing columns for feature set {spec['feature_set']}: {missing}")

    train = cohort[cohort["split"] == "train"].copy()
    validation = cohort[cohort["split"] == "validation"].copy()
    test = cohort[cohort["split"] == "test"].copy()

    preprocessor = fit_preprocessor(train, spec["numeric_features"], spec["categorical_features"])
    x_train, feature_names = transform(train, preprocessor)
    validate_no_forbidden_features(feature_names)
    y_train = train["readmission_30d"].astype(int).to_numpy()

    weights = train_logistic_regression(x_train, y_train)
    train_scores = expit(x_train @ weights)
    threshold = choose_threshold_by_train_prevalence(train_scores, y_train)

    performance = pd.DataFrame(
        [
            evaluate_split(spec["feature_set"], "train", train, weights, preprocessor, threshold),
            evaluate_split(spec["feature_set"], "validation", validation, weights, preprocessor, threshold),
            evaluate_split(spec["feature_set"], "test", test, weights, preprocessor, threshold),
        ]
    )
    coefficients = pd.DataFrame(
        {
            "feature_set": spec["feature_set"],
            "feature": feature_names,
            "coefficient": weights,
            "abs_coefficient": np.abs(weights),
        }
    ).sort_values(["feature_set", "abs_coefficient"], ascending=[True, False])

    x_test, _ = transform(test, preprocessor)
    test_scores = expit(x_test @ weights)
    predictions = test[["subject_id", "hadm_id", "readmission_30d"]].copy()
    predictions["feature_set"] = spec["feature_set"]
    predictions["predicted_risk"] = test_scores
    return performance, coefficients, predictions


def write_report(performance: pd.DataFrame, report_path: Path) -> None:
    tests = performance[performance["split"] == "test"].sort_values("feature_set")
    metric_lines = []
    for row in tests.itertuples(index=False):
        metric_lines.append(
            f"| {row.feature_set} | {int(row.n)} | {int(row.events)} | {row.event_rate:.2%} | "
            f"{row.AUROC:.4f} | {row.AUPRC:.4f} | {row.Brier_score:.4f} | "
            f"{row.sensitivity:.4f} | {row.specificity:.4f} | {row.ppv:.4f} | {row.npv:.4f} |"
        )
    table = "\n".join(metric_lines)
    text = f"""# MIMIC 糖尿病 Logistic Baseline Report

## 模型

- 模型：logistic regression
- 实现：`numpy/scipy` 本地实现，L2 正则
- 预测时间点：出院时
- 标签：30 天再入院
- 特征集：
  - `minimal`：人口学、入院信息、住院时长、既往住院
  - `lab_augmented`：`minimal` + 出院前 glucose/HbA1c/creatinine/BUN 摘要
  - `lab_med_augmented`：`lab_augmented` + 出院前糖尿病相关用药类别

## Test Set 结果

| Feature set | N | Events | Event rate | AUROC | AUPRC | Brier | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## 解释

这是第一版传统 baseline，用来确认数据管道、患者级切分、出院前化验截断和指标输出能跑通。它不是最终模型。下一步可以加入 random forest、XGBoost、用药特征，以及更完整的校准图和 PR/ROC 曲线。
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project_root
    tables_dir = project / "outputs" / "tables"
    reports_dir = project / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    performance_parts = []
    coefficient_parts = []
    prediction_parts = []
    for spec in available_feature_sets(project):
        performance, coefficients, predictions = run_feature_set(project, spec)
        performance_parts.append(performance)
        coefficient_parts.append(coefficients)
        prediction_parts.append(predictions)

    performance_all = pd.concat(performance_parts, ignore_index=True)
    coefficients_all = pd.concat(coefficient_parts, ignore_index=True)
    predictions_all = pd.concat(prediction_parts, ignore_index=True)
    performance_all.to_csv(tables_dir / "mimic_diabetes_model_performance.csv", index=False)
    coefficients_all.to_csv(tables_dir / "mimic_diabetes_logistic_coefficients.csv", index=False)
    predictions_all.to_csv(tables_dir / "mimic_diabetes_test_predictions.csv", index=False)
    write_report(performance_all, reports_dir / "mimic_diabetes_logistic_baseline_report.md")

    print("MIMIC diabetes logistic baselines complete")
    for row in performance_all[performance_all["split"] == "test"].itertuples(index=False):
        print(
            f"{row.feature_set}: test_AUROC={row.AUROC:.4f} "
            f"test_AUPRC={row.AUPRC:.4f} test_Brier={row.Brier_score:.4f}"
        )


if __name__ == "__main__":
    main()
