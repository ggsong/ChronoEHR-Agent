#!/usr/bin/env python3
"""Audit the model-selection rationale for external benchmark summary rows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


CALIBRATED_METHODS = {"intercept_validation", "platt_validation", "isotonic_validation"}
MODEL_COMPARISON_MODELS = {"random_forest_balanced", "hist_gradient_boosting_weighted"}


@dataclass(frozen=True)
class SelectionRule:
    benchmark_row: str
    dataset: str
    candidate_description: str
    ranking_rule: str
    boundary_note: str


RULES = [
    SelectionRule(
        "CDSL early-window best",
        "CDSL",
        "CDSL prediction-time rows where is_prediction_time_valid is true.",
        "Sort by AUPRC descending, AUROC descending, Brier ascending.",
        "Excludes full-stay future information.",
    ),
    SelectionRule(
        "CDSL full-stay naive reference",
        "CDSL",
        "CDSL full-stay rows where is_prediction_time_valid is false.",
        "Sort by AUPRC descending, AUROC descending, Brier ascending.",
        "Naive upper-reference only; not early prediction performance.",
    ),
    SelectionRule(
        "eICU calibrated logistic reference",
        "eICU",
        "eICU logistic_regression_balanced validation-calibrated rows.",
        "Sort validation-calibrated logistic rows by AUPRC descending, AUROC descending, Brier ascending.",
        "External ICU mortality benchmark; not chronic readmission validation.",
    ),
    SelectionRule(
        "eICU best calibrated RF/HGB",
        "eICU",
        "eICU random_forest_balanced and hist_gradient_boosting_weighted validation-calibrated rows.",
        "Sort calibrated RF/HGB rows by AUPRC descending, AUROC descending, Brier ascending.",
        "External ICU mortality benchmark; not chronic readmission validation.",
    ),
    SelectionRule(
        "CHARLS calibrated logistic reference",
        "CHARLS",
        "CHARLS logistic_regression_balanced validation-calibrated rows.",
        "Sort validation-calibrated logistic rows by AUPRC descending, AUROC descending, Brier ascending.",
        "External longitudinal chronic-disease cohort extension.",
    ),
    SelectionRule(
        "CHARLS best calibrated RF/HGB",
        "CHARLS",
        "CHARLS random_forest_balanced and hist_gradient_boosting_weighted validation-calibrated rows.",
        "Sort calibrated RF/HGB rows by AUPRC descending, AUROC descending, Brier ascending.",
        "External longitudinal chronic-disease cohort extension.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def candidates_for_rule(hard: pd.DataFrame, rule: SelectionRule) -> pd.DataFrame:
    if rule.benchmark_row == "CDSL early-window best":
        return hard[hard["dataset"].eq("CDSL") & hard["is_prediction_time_valid"].astype(bool)].copy()
    if rule.benchmark_row == "CDSL full-stay naive reference":
        return hard[hard["dataset"].eq("CDSL") & ~hard["is_prediction_time_valid"].astype(bool)].copy()
    if rule.benchmark_row in {"eICU calibrated logistic reference", "CHARLS calibrated logistic reference"}:
        return hard[
            hard["dataset"].eq(rule.dataset)
            & hard["model"].eq("logistic_regression_balanced")
            & hard["calibration_method"].isin(CALIBRATED_METHODS)
        ].copy()
    return hard[
        hard["dataset"].eq(rule.dataset)
        & hard["model"].isin(MODEL_COMPARISON_MODELS)
        & hard["calibration_method"].isin(CALIBRATED_METHODS)
    ].copy()


def sort_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates.sort_values(
        ["AUPRC", "AUROC", "Brier", "mean_absolute_calibration_error"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def row_identity(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("feature_set", "")),
        str(row.get("model", "")),
        str(row.get("calibration_method", "")),
    )


def metric(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row or pd.isna(row[column]):
        return float("nan")
    return float(row[column])


def build_rationale(project_root: Path) -> pd.DataFrame:
    hard = read_csv(project_root / "outputs" / "tables" / "external_benchmark_hard_metrics_table.csv")
    summary = read_csv(project_root / "outputs" / "tables" / "external_benchmark_summary_table.csv")
    if hard.empty or summary.empty:
        raise FileNotFoundError("Missing external benchmark hard metrics or summary table.")
    summary_by_label = {str(row["benchmark_row"]): row for _, row in summary.iterrows()}
    rows: list[dict[str, object]] = []
    for rule in RULES:
        candidates = sort_candidates(candidates_for_rule(hard, rule))
        selected = candidates.iloc[0] if not candidates.empty else None
        runner_up = candidates.iloc[1] if len(candidates) > 1 else None
        summary_row = summary_by_label.get(rule.benchmark_row)
        selected_matches_summary = bool(
            selected is not None
            and summary_row is not None
            and row_identity(selected) == row_identity(summary_row)
        )
        selected_auprc = metric(selected, "AUPRC")
        runner_auprc = metric(runner_up, "AUPRC")
        selected_auroc = metric(selected, "AUROC")
        runner_auroc = metric(runner_up, "AUROC")
        selected_brier = metric(selected, "Brier")
        runner_brier = metric(runner_up, "Brier")
        selected_calibration = metric(selected, "mean_absolute_calibration_error")
        runner_calibration = metric(runner_up, "mean_absolute_calibration_error")
        rows.append(
            {
                "benchmark_row": rule.benchmark_row,
                "dataset": rule.dataset,
                "candidate_description": rule.candidate_description,
                "ranking_rule": rule.ranking_rule,
                "candidate_count": int(len(candidates)),
                "selected_feature_set": str(selected.get("feature_set", "")) if selected is not None else "",
                "selected_model": str(selected.get("model", "")) if selected is not None else "",
                "selected_calibration_method": str(selected.get("calibration_method", "")) if selected is not None else "",
                "selected_AUROC": selected_auroc,
                "selected_AUPRC": selected_auprc,
                "selected_Brier": selected_brier,
                "selected_mean_absolute_calibration_error": selected_calibration,
                "selected_subgroup_ok_rows": metric(selected, "subgroup_ok_rows"),
                "runner_up_feature_set": str(runner_up.get("feature_set", "")) if runner_up is not None else "",
                "runner_up_model": str(runner_up.get("model", "")) if runner_up is not None else "",
                "runner_up_calibration_method": str(runner_up.get("calibration_method", "")) if runner_up is not None else "",
                "runner_up_AUROC": runner_auroc,
                "runner_up_AUPRC": runner_auprc,
                "runner_up_Brier": runner_brier,
                "runner_up_mean_absolute_calibration_error": runner_calibration,
                "delta_AUPRC_vs_runner_up": selected_auprc - runner_auprc if pd.notna(runner_auprc) else np.nan,
                "delta_AUROC_vs_runner_up": selected_auroc - runner_auroc if pd.notna(runner_auroc) else np.nan,
                "delta_Brier_vs_runner_up": selected_brier - runner_brier if pd.notna(runner_brier) else np.nan,
                "delta_calibration_mae_vs_runner_up": selected_calibration - runner_calibration if pd.notna(runner_calibration) else np.nan,
                "selected_matches_summary": selected_matches_summary,
                "selection_status": "PASS" if selected_matches_summary and len(candidates) > 0 else "FAIL",
                "boundary_note": rule.boundary_note,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = [
        "benchmark_row",
        "candidate_count",
        "selected_model",
        "selected_calibration_method",
        "selected_AUPRC",
        "selected_Brier",
        "runner_up_model",
        "runner_up_calibration_method",
        "delta_AUPRC_vs_runner_up",
        "delta_Brier_vs_runner_up",
        "selected_matches_summary",
        "boundary_note",
    ]
    display = display[[col for col in columns if col in display.columns]].astype(object).where(pd.notna(display), "")
    lines = ["| " + " | ".join(display.columns) + " |", "|" + "|".join("---" for _ in display.columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, table: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_model_selection_rationale.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# External Model-Selection Rationale

- Boundary: research model evaluation only; no medical QA, diagnosis, treatment recommendation, or clinical deployment guidance.
- Scope: six selected external benchmark rows used by the external benchmark summary and technical summary.
- Selection rules are deterministic and auditable against `external_benchmark_hard_metrics_table.csv`.

## Rationale Table

{markdown_table(table)}

## Coverage

- Selection rules: {len(table)}
- Passing selections: {int(table["selection_status"].astype(str).eq("PASS").sum())}
- Datasets: {", ".join(sorted(table["dataset"].dropna().astype(str).unique()))}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    table = build_rationale(args.project_root)
    tables = args.project_root / "outputs" / "tables"
    supplement = tables / "supplementary_appendix"
    tables.mkdir(parents=True, exist_ok=True)
    supplement.mkdir(parents=True, exist_ok=True)
    table_path = tables / "external_model_selection_rationale.csv"
    supp_path = supplement / "table_s17_external_model_selection_rationale.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(supp_path, index=False)
    report = write_report(args.project_root, table)
    print(f"Wrote {report}")
    print(f"Wrote {table_path}")
    print(f"Wrote {supp_path}")


if __name__ == "__main__":
    main()
