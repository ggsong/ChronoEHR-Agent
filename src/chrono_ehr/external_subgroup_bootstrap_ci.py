#!/usr/bin/env python3
"""Bootstrap confidence intervals for external benchmark subgroup performance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mimic_diabetes_baseline import DEFAULT_PROJECT


RNG_SEED = 20260622
N_BOOTSTRAP = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def read_optional(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def age_group(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True, right=False).astype(str)


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


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int | str]:
    has_two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()) if len(y) else float("nan"),
        "AUROC": float(roc_auc_score(y, score)) if has_two_classes else float("nan"),
        "AUPRC": float(average_precision_score(y, score)) if has_two_classes else float("nan"),
        "Brier": float(brier_score_loss(y, score)) if len(y) else float("nan"),
        "status": "OK" if len(y) >= 30 and has_two_classes else "SMALL_OR_SINGLE_CLASS",
    }


def bootstrap_ci(y: np.ndarray, score: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> dict[str, float | int]:
    if len(y) < 30 or len(np.unique(y)) < 2:
        return {
            "AUROC_lower": float("nan"),
            "AUROC_upper": float("nan"),
            "AUPRC_lower": float("nan"),
            "AUPRC_upper": float("nan"),
            "Brier_lower": float("nan"),
            "Brier_upper": float("nan"),
            "bootstrap_replicates": 0,
        }
    aurocs: list[float] = []
    auprcs: list[float] = []
    briers: list[float] = []
    n = len(y)
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


def rows_for_groups(
    dataset: str,
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
    model_cols: list[str],
    subgroup_cols: list[tuple[str, str]],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_keys, model_group in frame.groupby(model_cols, sort=True):
        key_values = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        model_context = {col: value for col, value in zip(model_cols, key_values)}
        for subgroup_type, subgroup_col in subgroup_cols:
            for subgroup_value, part in model_group.groupby(subgroup_col, dropna=False, sort=True):
                if str(subgroup_value) in {"nan", "None", "<NA>"}:
                    continue
                y = part[label_col].astype(int).to_numpy()
                score = part[score_col].astype(float).to_numpy()
                row: dict[str, object] = {"dataset": dataset, "subgroup_type": subgroup_type, "subgroup": str(subgroup_value), **model_context}
                if dataset == "CHARLS" and "source_model" in row:
                    row["model"] = row.get("source_model", "")
                row.update(point_metrics(y, score))
                row.update(bootstrap_ci(y, score, rng, n_bootstrap))
                rows.append(row)
    return rows


def cdsl_rows(project_root: Path, rng: np.random.Generator, n_bootstrap: int) -> list[dict[str, object]]:
    pred_path = project_root / "outputs" / "tables" / "cdsl_traditional_baselines_predictions.csv"
    demo_path = project_root / "data" / "processed" / "cdsl_temporal_benchmark" / "admission_demographics.csv"
    if not pred_path.exists() or not demo_path.exists():
        return []
    predictions = pd.read_csv(pred_path)
    demographics = pd.read_csv(demo_path, usecols=["PatientID", "Sex", "Age"])
    frame = predictions[predictions["split"].astype(str).eq("test")].merge(demographics, on="PatientID", how="left")
    frame["sex_group"] = frame["Sex"].map({0.0: "sex_0", 1.0: "sex_1"}).fillna("sex_unknown")
    frame["age_group"] = age_group(frame["Age"], [0, 60, 75, 200], ["age_lt_60", "age_60_74", "age_ge_75"])
    return rows_for_groups(
        "CDSL",
        frame,
        "outcome",
        "predicted_risk",
        ["feature_set", "model"],
        [("sex", "sex_group"), ("age", "age_group")],
        rng,
        n_bootstrap,
    )


def eicu_rows(project_root: Path, rng: np.random.Generator, n_bootstrap: int) -> list[dict[str, object]]:
    tables = project_root / "outputs" / "tables"
    cohort_path = project_root / "data" / "processed" / "eicu_temporal_mortality_cohort.csv"
    if not cohort_path.exists():
        return []
    cohort = pd.read_csv(cohort_path, usecols=["stay_id", "age_years", "gender"])
    cohort["row_id"] = cohort["stay_id"].astype(str)
    rows: list[dict[str, object]] = []

    predictions = read_optional(tables / "eicu_probability_recalibration_predictions.csv")
    if not predictions.empty:
        predictions = predictions[
            predictions["split"].astype(str).eq("test")
            & predictions["calibration_method"].astype(str).eq("platt_validation")
        ].copy()
        frame = predictions.merge(cohort, on="stay_id", how="left")
        frame["gender_group"] = frame["gender"].astype(str).str.lower().replace({"nan": "unknown", "": "unknown"})
        frame["age_group"] = age_group(frame["age_years"], [18, 50, 65, 80, 200], ["age_18_49", "age_50_64", "age_65_79", "age_ge_80"])
        rows.extend(
            rows_for_groups(
                "eICU",
                frame,
                "hospital_mortality",
                "calibrated_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("gender", "gender_group"), ("age", "age_group")],
                rng,
                n_bootstrap,
            )
        )

    predictions = read_optional(tables / "eicu_first24h_model_comparison_predictions.csv")
    if not predictions.empty:
        predictions = predictions[
            predictions["split"].astype(str).eq("test")
            & predictions["model"].astype(str).ne("logistic_regression_balanced")
        ].copy()
        frame = predictions.merge(cohort, on="stay_id", how="left")
        frame["calibration_method"] = "raw_model_comparison"
        frame["gender_group"] = frame["gender"].astype(str).str.lower().replace({"nan": "unknown", "": "unknown"})
        frame["age_group"] = age_group(frame["age_years"], [18, 50, 65, 80, 200], ["age_18_49", "age_50_64", "age_65_79", "age_ge_80"])
        rows.extend(
            rows_for_groups(
                "eICU",
                frame,
                "hospital_mortality",
                "predicted_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("gender", "gender_group"), ("age", "age_group")],
                rng,
                n_bootstrap,
            )
        )

    predictions = read_optional(tables / "external_model_comparison_recalibration_predictions.csv", low_memory=False)
    if not predictions.empty:
        predictions = predictions[
            predictions["dataset"].astype(str).eq("eICU")
            & predictions["split"].astype(str).eq("test")
            & predictions["calibration_method"].astype(str).ne("raw")
        ].copy()
        predictions["row_id"] = predictions["row_id"].astype(str)
        frame = predictions.merge(cohort[["row_id", "age_years", "gender"]], on="row_id", how="left")
        frame["gender_group"] = frame["gender"].astype(str).str.lower().replace({"nan": "unknown", "": "unknown"})
        frame["age_group"] = age_group(frame["age_years"], [18, 50, 65, 80, 200], ["age_18_49", "age_50_64", "age_65_79", "age_ge_80"])
        rows.extend(
            rows_for_groups(
                "eICU",
                frame,
                "label",
                "calibrated_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("gender", "gender_group"), ("age", "age_group")],
                rng,
                n_bootstrap,
            )
        )
    return rows


def charls_rows(project_root: Path, rng: np.random.Generator, n_bootstrap: int) -> list[dict[str, object]]:
    tables = project_root / "outputs" / "tables"
    feature_path = project_root / "data" / "processed" / "charls_incident_diabetes_baseline_features.csv"
    if not feature_path.exists():
        return []
    features = pd.read_csv(feature_path, usecols=["person_id", "charls_baseline_age_years", "charls_baseline_sex_code"])
    features["row_id"] = features["person_id"].astype(str)
    rows: list[dict[str, object]] = []

    predictions = read_optional(tables / "charls_probability_recalibration_predictions.csv")
    if not predictions.empty:
        predictions = predictions[
            predictions["split"].astype(str).eq("test")
            & predictions["calibration_method"].astype(str).eq("platt_validation")
        ].copy()
        frame = predictions.merge(features, on="person_id", how="left")
        frame["model"] = frame["source_model"]
        frame["sex_group"] = frame["charls_baseline_sex_code"].map({1.0: "sex_1", 2.0: "sex_2"}).fillna("sex_unknown")
        frame["age_group"] = age_group(frame["charls_baseline_age_years"], [0, 50, 65, 200], ["age_lt_50", "age_50_64", "age_ge_65"])
        rows.extend(
            rows_for_groups(
                "CHARLS",
                frame,
                "incident_diabetes_2013_or_2015",
                "calibrated_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("sex", "sex_group"), ("age", "age_group")],
                rng,
                n_bootstrap,
            )
        )

    predictions = read_optional(tables / "charls_incident_diabetes_model_comparison_predictions.csv")
    if not predictions.empty:
        predictions = predictions[
            predictions["split"].astype(str).eq("test")
            & predictions["model"].astype(str).ne("logistic_regression_balanced")
        ].copy()
        frame = predictions.merge(features, on="person_id", how="left")
        frame["calibration_method"] = "raw_model_comparison"
        frame["sex_group"] = frame["charls_baseline_sex_code"].map({1.0: "sex_1", 2.0: "sex_2"}).fillna("sex_unknown")
        frame["age_group"] = age_group(frame["charls_baseline_age_years"], [0, 50, 65, 200], ["age_lt_50", "age_50_64", "age_ge_65"])
        rows.extend(
            rows_for_groups(
                "CHARLS",
                frame,
                "incident_diabetes_2013_or_2015",
                "predicted_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("sex", "sex_group"), ("age", "age_group")],
                rng,
                n_bootstrap,
            )
        )

    predictions = read_optional(tables / "external_model_comparison_recalibration_predictions.csv", low_memory=False)
    if not predictions.empty:
        predictions = predictions[
            predictions["dataset"].astype(str).eq("CHARLS")
            & predictions["split"].astype(str).eq("test")
            & predictions["calibration_method"].astype(str).ne("raw")
        ].copy()
        predictions["row_id"] = predictions["row_id"].astype(str)
        frame = predictions.merge(features[["row_id", "charls_baseline_age_years", "charls_baseline_sex_code"]], on="row_id", how="left")
        frame["sex_group"] = frame["charls_baseline_sex_code"].map({1.0: "sex_1", 2.0: "sex_2"}).fillna("sex_unknown")
        frame["age_group"] = age_group(frame["charls_baseline_age_years"], [0, 50, 65, 200], ["age_lt_50", "age_50_64", "age_ge_65"])
        rows.extend(
            rows_for_groups(
                "CHARLS",
                frame,
                "label",
                "calibrated_risk",
                ["prediction_time", "feature_set", "model", "calibration_method"],
                [("sex", "sex_group"), ("age", "age_group")],
                rng,
                n_bootstrap,
            )
        )
    return rows


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


def write_report(project_root: Path, table: pd.DataFrame, n_bootstrap: int) -> Path:
    report = project_root / "outputs" / "reports" / "external_subgroup_bootstrap_ci.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    compact_cols = [
        "dataset",
        "feature_set",
        "model",
        "calibration_method",
        "subgroup_type",
        "subgroup",
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
        "status",
    ]
    display_cols = [col for col in compact_cols if col in table.columns]
    report.write_text(
        f"""# External Subgroup Bootstrap CI

- Boundary: research model evaluation only; no medical QA, diagnosis, or treatment recommendation.
- Bootstrap replicates requested: {n_bootstrap}.
- Scope: test-split subgroup uncertainty for completed CDSL/eICU/CHARLS external benchmark predictions; models are not refit inside bootstrap.
- Rows marked `SMALL_OR_SINGLE_CLASS` retain point estimates where valid but do not receive bootstrap intervals.

## Subgroup Bootstrap CI

{markdown_table(table[display_cols])}
""",
        encoding="utf-8",
    )
    return report


def build_ci(project_root: Path, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = cdsl_rows(project_root, rng, n_bootstrap)
    rows.extend(eicu_rows(project_root, rng, n_bootstrap))
    rows.extend(charls_rows(project_root, rng, n_bootstrap))
    table = pd.DataFrame(rows)
    if not table.empty:
        table["split_note"] = "test"
    return table


def main() -> None:
    args = parse_args()
    table = build_ci(args.project_root, args.n_bootstrap)
    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "external_subgroup_bootstrap_ci.csv", index=False)
    report = write_report(args.project_root, table, args.n_bootstrap)
    print(f"Wrote {report}")
    if not table.empty:
        summary = table.groupby(["dataset", "subgroup_type", "status"]).size().reset_index(name="rows")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
