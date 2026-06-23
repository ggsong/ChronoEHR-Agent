#!/usr/bin/env python3
"""Optional gradient boosting baselines for chronic disease cohorts.

The script always supports a scikit-learn HistGradientBoosting fallback. XGBoost
and LightGBM are treated as optional backends: if a package is missing, a skipped
status row is written instead of breaking the pipeline.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd

from leakage_gate import enforce_spec_gate
from mimic_diabetes_baseline import DEFAULT_PROJECT
from random_forest_baseline import DEFAULT_FEATURE_SETS


BACKENDS = ["sklearn_hist", "xgboost", "lightgbm"]


def optional_import(import_name: str):
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - user-facing status
        return None, f"{type(exc).__name__}: {exc}"
    return module, str(getattr(module, "__version__", "unknown"))


def make_preprocessor(numeric: list[str], categorical: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


def build_model(backend: str, args: argparse.Namespace):
    if backend == "sklearn_hist":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            class_weight="balanced",
            early_stopping=True,
            random_state=20260620,
        ), "sklearn"

    if backend == "xgboost":
        xgboost, detail = optional_import("xgboost")
        if xgboost is None:
            return None, detail
        return xgboost.XGBClassifier(
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            random_state=20260620,
            n_jobs=-1,
        ), detail

    if backend == "lightgbm":
        lightgbm, detail = optional_import("lightgbm")
        if lightgbm is None:
            return None, detail
        return lightgbm.LGBMClassifier(
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=args.max_leaf_nodes,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=20260620,
            n_jobs=-1,
            verbose=-1,
        ), detail

    raise ValueError(f"Unknown backend: {backend}")


def evaluate_split(study_key: str, spec: dict, backend: str, split_name: str, split_df: pd.DataFrame, score) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = split_df["readmission_30d"].astype(int)
    return {
        "study": study_key,
        "cohort": study_key,
        "feature_set": spec["feature_set"],
        "model": f"gradient_boosting_{backend}",
        "backend": backend,
        "split": split_name,
        "n": int(len(split_df)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, score)),
        "Brier_score": float(brier_score_loss(y, score)),
    }


def probability_scores(model, x) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def run_backend(project_root: Path, backend: str, studies: list[str], args: argparse.Namespace) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    from sklearn.pipeline import Pipeline

    from prediction_time_model_tools import load_analysis_data
    from prediction_time_spec_loader import load_prediction_time_config

    metric_rows = []
    prediction_parts = []
    status_rows = []
    model, detail = build_model(backend, args)
    if model is None:
        for study_key in studies:
            status_rows.append({"backend": backend, "study": study_key, "status": "skipped", "reason": detail})
        return metric_rows, prediction_parts, status_rows

    for study_key in studies:
        config = load_prediction_time_config(study_key)
        spec = next(item for item in config["specs"] if item["feature_set"] == DEFAULT_FEATURE_SETS[study_key])
        enforce_spec_gate(project_root, study_key, spec)
        df = load_analysis_data(project_root, config["cohort_path"], spec.get("extra_feature_files"))
        train = df[df["split"].eq("train")].copy()
        validation = df[df["split"].eq("validation")].copy()
        test = df[df["split"].eq("test")].copy()

        numeric = spec["numeric_features"]
        categorical = spec["categorical_features"]
        pipeline = Pipeline(steps=[("preprocess", make_preprocessor(numeric, categorical)), ("model", model)])
        pipeline.fit(train[numeric + categorical], train["readmission_30d"].astype(int))

        for split_name, split_df in [("validation", validation), ("test", test)]:
            score = probability_scores(pipeline, split_df[numeric + categorical])
            row = evaluate_split(study_key, spec, backend, split_name, split_df, score)
            row["backend_detail"] = detail
            row["max_iter"] = args.max_iter
            row["learning_rate"] = args.learning_rate
            row["max_leaf_nodes"] = args.max_leaf_nodes
            metric_rows.append(row)

            pred = split_df[["subject_id", "hadm_id", "readmission_30d"]].copy()
            pred["study"] = study_key
            pred["feature_set"] = spec["feature_set"]
            pred["model"] = f"gradient_boosting_{backend}"
            pred["backend"] = backend
            pred["split"] = split_name
            pred["predicted_risk"] = score
            prediction_parts.append(pred)

        status_rows.append({"backend": backend, "study": study_key, "status": "complete", "reason": detail})
    return metric_rows, prediction_parts, status_rows


def write_report(performance: pd.DataFrame, status: pd.DataFrame, output: Path) -> None:
    test = performance[performance["split"].eq("test")].copy() if not performance.empty else pd.DataFrame()
    lines = [
        "# Gradient Boosting Baseline Report",
        "",
        "## Backend Status",
        "",
        "| Backend | Study | Status | Detail |",
        "|---|---|---|---|",
    ]
    for row in status.itertuples(index=False):
        lines.append(f"| {row.backend} | {row.study} | {row.status} | {row.reason} |")

    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| Study | Backend | Feature set | N | Events | AUROC | AUPRC | Brier |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in test.itertuples(index=False):
        lines.append(
            f"| {row.study} | {row.backend} | {row.feature_set} | {int(row.n):,} | {int(row.events):,} | "
            f"{row.AUROC:.4f} | {row.AUPRC:.4f} | {row.Brier_score:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `sklearn_hist` is a dependency-light gradient boosting baseline.",
            "- XGBoost and LightGBM rows are skipped when packages are not installed.",
            "- These baselines use the same primary discharge-time feature sets as the Random Forest comparison.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--study", choices=["diabetes", "ckd", "heart_failure", "hypertension", "all"], default="all")
    parser.add_argument("--backend", choices=[*BACKENDS, "all"], default="sklearn_hist")
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    studies = ["diabetes", "ckd", "heart_failure", "hypertension"] if args.study == "all" else [args.study]
    backends = BACKENDS if args.backend == "all" else [args.backend]
    metric_rows = []
    prediction_parts = []
    status_rows = []
    for backend in backends:
        rows, preds, statuses = run_backend(args.project_root, backend, studies, args)
        metric_rows.extend(rows)
        prediction_parts.extend(preds)
        status_rows.extend(statuses)

    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    performance = pd.DataFrame(metric_rows)
    status = pd.DataFrame(status_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    performance.to_csv(tables / "gradient_boosting_baseline_performance.csv", index=False)
    predictions.to_csv(tables / "gradient_boosting_baseline_predictions.csv", index=False)
    status.to_csv(tables / "gradient_boosting_baseline_status.csv", index=False)
    write_report(performance, status, reports / "gradient_boosting_baseline_report.md")

    print("Gradient boosting baseline complete")
    if performance.empty:
        print(status.to_string(index=False))
    else:
        print(performance[performance["split"].eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
