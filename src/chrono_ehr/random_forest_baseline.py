#!/usr/bin/env python3
"""Optional Random Forest baselines driven by prediction-time specs.

If scikit-learn is unavailable, this script writes a skipped status report instead
of failing the whole project pipeline.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


DEFAULT_FEATURE_SETS = {
    "diabetes": "discharge_safe_minimal",
    "ckd": "discharge_lab_minimal",
    "heart_failure": "discharge_lab_minimal",
    "hypertension": "discharge_lab_minimal",
}


def sklearn_available() -> tuple[bool, str]:
    try:
        sklearn = importlib.import_module("sklearn")
    except Exception as exc:  # noqa: BLE001 - user-facing status
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(sklearn, "__version__", "unknown"))


def write_skipped(project_root: Path, reason: str) -> None:
    reports = project_root / "outputs" / "reports"
    tables = project_root / "outputs" / "tables"
    reports.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": "random_forest",
            "status": "skipped",
            "reason": reason,
        }
    ]
    pd.DataFrame(rows).to_csv(tables / "random_forest_baseline_status.csv", index=False)
    text = f"""# Random Forest Baseline Status

- Status: `skipped`
- Reason: {reason}

当前 logistic regression baseline、prediction-time comparison、leakage audit 和报告生成不依赖 scikit-learn。安装 scikit-learn 后可以重新运行本脚本来生成 Random Forest 对照。
"""
    (reports / "random_forest_baseline_status.md").write_text(text, encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--study", choices=["diabetes", "ckd", "heart_failure", "hypertension", "all"], default="all")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available, detail = sklearn_available()
    if not available:
        write_skipped(args.project_root, f"scikit-learn is not installed ({detail})")
        return

    # Imports stay inside this branch so the script remains runnable without sklearn.
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from prediction_time_model_tools import load_analysis_data
    from prediction_time_spec_loader import load_prediction_time_config

    studies = ["diabetes", "ckd", "heart_failure", "hypertension"] if args.study == "all" else [args.study]
    rows = []
    prediction_rows = []
    for study_key in studies:
        config = load_prediction_time_config(study_key)
        spec = next(item for item in config["specs"] if item["feature_set"] == DEFAULT_FEATURE_SETS[study_key])
        df = load_analysis_data(args.project_root, config["cohort_path"], spec.get("extra_feature_files"))
        train = df[df["split"].eq("train")].copy()
        validation = df[df["split"].eq("validation")].copy()
        test = df[df["split"].eq("test")].copy()

        numeric = spec["numeric_features"]
        categorical = spec["categorical_features"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
                        ]
                    ),
                    categorical,
                ),
            ]
        )
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        min_samples_leaf=50,
                        random_state=20260619,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        )
        model.fit(train[numeric + categorical], train["readmission_30d"].astype(int))
        for split_name, split_df in [("validation", validation), ("test", test)]:
            y = split_df["readmission_30d"].astype(int)
            score = model.predict_proba(split_df[numeric + categorical])[:, 1]
            rows.append(
                {
                    "study": study_key,
                    "feature_set": spec["feature_set"],
                    "model": "random_forest_sklearn",
                    "split": split_name,
                    "n": int(len(split_df)),
                    "events": int(y.sum()),
                    "event_rate": float(y.mean()),
                    "AUROC": float(roc_auc_score(y, score)),
                    "AUPRC": float(average_precision_score(y, score)),
                    "Brier_score": float(brier_score_loss(y, score)),
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "sklearn_version": detail,
                }
            )
            prediction_part = split_df[["subject_id", "hadm_id", "readmission_30d"]].copy()
            prediction_part["study"] = study_key
            prediction_part["feature_set"] = spec["feature_set"]
            prediction_part["model"] = "random_forest_sklearn"
            prediction_part["split"] = split_name
            prediction_part["predicted_risk"] = score
            prediction_rows.append(prediction_part)

    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(tables / "random_forest_baseline_performance.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(tables / "random_forest_baseline_predictions.csv", index=False)

    lines = [
        "# Random Forest Baseline Report",
        "",
        f"- scikit-learn version: `{detail}`",
        "- Predictions: `outputs/tables/random_forest_baseline_predictions.csv`",
        "",
        "| Study | Feature set | Split | N | Events | AUROC | AUPRC | Brier |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in out.itertuples(index=False):
        lines.append(
            f"| {row.study} | {row.feature_set} | {row.split} | {int(row.n):,} | {int(row.events):,} | "
            f"{row.AUROC:.4f} | {row.AUPRC:.4f} | {row.Brier_score:.4f} |"
        )
    (reports / "random_forest_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
