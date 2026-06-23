#!/usr/bin/env python3
"""Post-hoc calibration for HistGradientBoosting baseline predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


GB_PREDICTIONS = "outputs/tables/gradient_boosting_baseline_predictions.csv"


def sklearn_available() -> tuple[bool, str]:
    try:
        import sklearn  # type: ignore
    except Exception as exc:  # noqa: BLE001 - user-facing status
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(sklearn, "__version__", "unknown"))


def load_predictions(project_root: Path) -> pd.DataFrame:
    path = project_root / GB_PREDICTIONS
    if not path.exists():
        raise FileNotFoundError(f"Missing gradient boosting predictions: {path}")
    return pd.read_csv(path)


def evaluate(y, score, metrics) -> dict[str, float | int]:
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "AUROC": float(metrics.roc_auc_score(y, score)),
        "AUPRC": float(metrics.average_precision_score(y, score)),
        "Brier_score": float(metrics.brier_score_loss(y, score)),
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data found."
    columns = ["study", "calibration_method", "split", "n", "events", "AUROC", "AUPRC", "Brier_score"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(performance: pd.DataFrame, report_path: Path, sklearn_version: str) -> None:
    test = performance[performance["split"].eq("test")].copy()
    lines = [
        "# Calibrated Gradient Boosting Baseline Report",
        "",
        f"- scikit-learn version: `{sklearn_version}`",
        "- Base model: `gradient_boosting_sklearn_hist`.",
        "- Calibration data: validation split.",
        "- Reported comparison data: test split.",
        "",
        "## Test And Validation Metrics",
        "",
        markdown_table(performance),
        "",
        "## Interpretation",
        "",
    ]
    if not test.empty:
        for method, part in test.groupby("calibration_method", sort=False):
            brier_mean = float(part["Brier_score"].mean())
            lines.append(f"- `{method}` mean test Brier score across cohorts: {brier_mean:.4f}.")
    lines.extend(
        [
            "- Calibration should be judged mainly by Brier score and calibration deciles, not AUROC alone.",
            "- Platt calibration is monotonic, so AUROC/AUPRC should stay close to the uncalibrated model.",
            "- Isotonic calibration can improve calibration more flexibly but may overfit validation data.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available, detail = sklearn_available()
    if not available:
        raise SystemExit(f"scikit-learn is required for calibration: {detail}")

    from sklearn import metrics
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    base = load_predictions(args.project_root)
    rows = []
    prediction_parts = []
    for study, group in base.groupby("study", sort=False):
        validation = group[group["split"].eq("validation")].copy()
        test = group[group["split"].eq("test")].copy()
        if validation.empty or test.empty:
            continue

        x_val = validation["predicted_risk"].to_numpy().reshape(-1, 1)
        y_val = validation["readmission_30d"].astype(int).to_numpy()
        calibrators = {
            "platt": LogisticRegression(solver="lbfgs", max_iter=1000).fit(x_val, y_val),
            "isotonic": IsotonicRegression(out_of_bounds="clip").fit(validation["predicted_risk"], y_val),
        }

        for method, calibrator in calibrators.items():
            for split_name, split_df in [("validation", validation), ("test", test)]:
                y = split_df["readmission_30d"].astype(int)
                if method == "platt":
                    score = calibrator.predict_proba(split_df["predicted_risk"].to_numpy().reshape(-1, 1))[:, 1]
                else:
                    score = calibrator.predict(split_df["predicted_risk"].to_numpy())
                metric_row = evaluate(y, score, metrics)
                metric_row.update(
                    {
                        "study": study,
                        "cohort": study,
                        "feature_set": split_df["feature_set"].iloc[0],
                        "model": f"calibrated_gradient_boosting_{method}",
                        "calibration_method": method,
                        "split": split_name,
                        "sklearn_version": detail,
                    }
                )
                rows.append(metric_row)

                pred = split_df[["subject_id", "hadm_id", "readmission_30d", "study", "feature_set", "split"]].copy()
                pred["model"] = f"calibrated_gradient_boosting_{method}"
                pred["calibration_method"] = method
                pred["predicted_risk"] = score
                prediction_parts.append(pred)

    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    performance = pd.DataFrame(rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    performance.to_csv(tables / "calibrated_gradient_boosting_performance.csv", index=False)
    predictions.to_csv(tables / "calibrated_gradient_boosting_predictions.csv", index=False)
    write_report(performance, reports / "calibrated_gradient_boosting_report.md", detail)

    print("Calibrated gradient boosting baseline complete")
    print(performance[performance["split"].eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
