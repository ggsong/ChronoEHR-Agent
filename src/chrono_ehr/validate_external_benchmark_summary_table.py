#!/usr/bin/env python3
"""Validate the external benchmark hard-metric summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_DATASETS = {"CDSL", "eICU", "CHARLS"}
EXPECTED_SELECTED_ROWS = {
    "CDSL early-window best",
    "CDSL full-stay naive reference",
    "eICU calibrated logistic reference",
    "eICU best calibrated RF/HGB",
    "CHARLS calibrated logistic reference",
    "CHARLS best calibrated RF/HGB",
}
EXPECTED_MODEL_METHODS = {"raw_model_comparison", "intercept_validation", "platt_validation", "isotonic_validation"}
MODEL_COMPARISON_MODELS = {"random_forest_balanced", "hist_gradient_boosting_weighted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def validate(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "external_benchmark_summary_table.csv"
    hard_path = project_root / "outputs" / "tables" / "external_benchmark_hard_metrics_table.csv"
    supp_path = project_root / "outputs" / "tables" / "supplementary_appendix" / "table_s13_external_benchmark_summary.csv"
    hard_supp_path = project_root / "outputs" / "tables" / "supplementary_appendix" / "table_s14_external_benchmark_hard_metrics.csv"
    report_path = project_root / "outputs" / "reports" / "external_benchmark_summary_table.md"
    summary = read_csv(table_path)
    hard = read_csv(hard_path)
    rows = [
        row("summary_table_exists", "PASS" if exists(table_path) else "FAIL", str(table_path), f"rows={len(summary)}"),
        row("hard_metrics_table_exists", "PASS" if exists(hard_path) else "FAIL", str(hard_path), f"rows={len(hard)}"),
        row("supplement_table_exists", "PASS" if exists(supp_path) else "FAIL", str(supp_path), f"size={supp_path.stat().st_size if supp_path.exists() else 0}"),
        row("hard_supplement_table_exists", "PASS" if exists(hard_supp_path) else "FAIL", str(hard_supp_path), f"size={hard_supp_path.stat().st_size if hard_supp_path.exists() else 0}"),
        row("report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    required_columns = {
        "dataset",
        "role",
        "task",
        "prediction_time",
        "feature_window",
        "feature_set",
        "model",
        "calibration_method",
        "n",
        "events",
        "event_rate",
        "AUROC",
        "AUROC_lower",
        "AUROC_upper",
        "AUPRC",
        "AUPRC_lower",
        "AUPRC_upper",
        "Brier",
        "Brier_lower",
        "Brier_upper",
        "mean_absolute_calibration_error",
        "subgroup_ok_rows",
        "leakage_gate_status",
        "interpretation_note",
    }
    missing = sorted(required_columns - set(hard.columns)) if not hard.empty else sorted(required_columns)
    rows.append(row("hard_required_columns", "PASS" if not missing else "FAIL", str(hard_path), "missing=" + ",".join(missing)))

    datasets = set(hard["dataset"].astype(str)) if not hard.empty and "dataset" in hard else set()
    rows.append(row("contains_all_external_datasets", "PASS" if EXPECTED_DATASETS <= datasets else "FAIL", str(hard_path), "datasets=" + ",".join(sorted(datasets))))
    selected = set(summary["benchmark_row"].astype(str)) if not summary.empty and "benchmark_row" in summary else set()
    rows.append(row("selected_rows_complete", "PASS" if EXPECTED_SELECTED_ROWS <= selected else "FAIL", str(table_path), "rows=" + ",".join(sorted(selected))))
    rows.append(row("selected_rows_are_six", "PASS" if len(summary) == 6 else "FAIL", str(table_path), f"rows={len(summary)}"))

    if not hard.empty:
        metric_cols = ["event_rate", "AUROC", "AUROC_lower", "AUROC_upper", "AUPRC", "AUPRC_lower", "AUPRC_upper", "Brier", "Brier_lower", "Brier_upper"]
        metric_ok = all(hard[col].dropna().between(0, 1).all() for col in metric_cols if col in hard.columns)
        rows.append(row("metrics_and_ci_in_unit_interval", "PASS" if metric_ok else "FAIL", str(hard_path), "metrics and CI bounds within [0,1]"))
        ci_ok = (
            hard["AUROC"].between(hard["AUROC_lower"], hard["AUROC_upper"]).all()
            and hard["AUPRC"].between(hard["AUPRC_lower"], hard["AUPRC_upper"]).all()
            and hard["Brier"].between(hard["Brier_lower"], hard["Brier_upper"]).all()
        )
        rows.append(row("point_estimates_inside_ci", "PASS" if ci_ok else "FAIL", str(hard_path), "AUROC/AUPRC/Brier inside CI"))

        missing_method_groups = []
        for dataset in ["eICU", "CHARLS"]:
            for model in MODEL_COMPARISON_MODELS:
                methods = set(
                    hard[
                        hard["dataset"].astype(str).eq(dataset)
                        & hard["model"].astype(str).eq(model)
                    ]["calibration_method"].dropna().astype(str)
                )
                if not EXPECTED_MODEL_METHODS <= methods:
                    missing_method_groups.append(f"{dataset}/{model}:{','.join(sorted(methods))}")
        rows.append(row("model_comparison_methods_complete", "PASS" if not missing_method_groups else "FAIL", str(hard_path), "bad=" + ";".join(missing_method_groups)))

        selected_keys = set(
            zip(
                summary["dataset"].astype(str),
                summary["model"].astype(str),
                summary["calibration_method"].astype(str),
            )
        ) if not summary.empty else set()
        subgroup_hard = hard[
            hard.apply(
                lambda item: (
                    (str(item["dataset"]), str(item["model"]), str(item["calibration_method"])) in selected_keys
                    or (
                        str(item["dataset"]) in {"eICU", "CHARLS"}
                        and str(item["model"]) in MODEL_COMPARISON_MODELS
                        and str(item["calibration_method"]) in EXPECTED_MODEL_METHODS
                    )
                ),
                axis=1,
            )
        ]
        subgroup_ok = subgroup_hard["subgroup_ok_rows"].fillna(0).astype(float).gt(0).all()
        rows.append(row("selected_and_model_rows_have_subgroups", "PASS" if subgroup_ok else "FAIL", str(hard_path), "selected rows and RF/HGB method rows have subgroup_ok_rows > 0"))
        text = " ".join(hard["interpretation_note"].astype(str))
        rows.append(row("eicu_not_chronic_validation", "PASS" if "not a chronic readmission external validation" in text else "FAIL", str(hard_path), "eICU boundary note required"))
        rows.append(row("cdsl_full_stay_marked_naive", "PASS" if "Naive upper-reference" in text else "FAIL", str(hard_path), "CDSL full-stay row must be marked naive"))

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = validate(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "external_benchmark_summary_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_benchmark_summary_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Benchmark Summary Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research benchmark summary outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External benchmark summary checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
