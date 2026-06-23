#!/usr/bin/env python3
"""Validate the unified external calibration and decision-curve summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_DATASETS = {"CDSL", "eICU", "CHARLS"}
EXPECTED_MODELS = {"logistic_regression_balanced", "random_forest_balanced", "hist_gradient_boosting_weighted"}
EXPECTED_VALIDATION_METHODS = {"intercept_validation", "platt_validation", "isotonic_validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "external_calibration_decision_summary.csv"
    supp_path = project_root / "outputs" / "tables" / "supplementary_appendix" / "table_s16_external_calibration_decision_summary.csv"
    report_path = project_root / "outputs" / "reports" / "external_calibration_decision_summary.md"
    table = read_csv(table_path)
    rows = [
        row("summary_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("supplement_table_exists", "PASS" if exists(supp_path) else "FAIL", str(supp_path), f"size={supp_path.stat().st_size if supp_path.exists() else 0}"),
        row("report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)

    required = {
        "dataset",
        "task",
        "feature_set",
        "model",
        "calibration_method",
        "n",
        "events",
        "event_rate",
        "mean_absolute_calibration_error",
        "max_absolute_calibration_error",
        "raw_mean_absolute_calibration_error",
        "calibration_mae_delta_vs_raw",
        "calibration_mae_improved_vs_raw",
        "calibration_rank_within_model",
        "decision_thresholds",
        "decision_model_preferred_thresholds",
        "decision_positive_advantage_thresholds",
        "decision_best_threshold",
        "decision_best_net_benefit_advantage",
        "is_selected_technical_summary_row",
        "benchmark_row",
        "interpretation_note",
    }
    missing = sorted(required - set(table.columns))
    rows.append(row("required_columns_present", "PASS" if not missing else "FAIL", str(table_path), "missing=" + ",".join(missing)))
    datasets = set(table["dataset"].astype(str))
    rows.append(row("expected_datasets", "PASS" if EXPECTED_DATASETS <= datasets else "FAIL", str(table_path), "datasets=" + ",".join(sorted(datasets))))
    rows.append(row("minimum_rows", "PASS" if len(table) >= 36 else "FAIL", str(table_path), f"rows={len(table)}"))

    counts = table.groupby("dataset").size().to_dict()
    bad_counts = [dataset for dataset in ["CDSL", "eICU", "CHARLS"] if int(counts.get(dataset, 0)) < 12]
    rows.append(row("minimum_rows_per_dataset", "PASS" if not bad_counts else "FAIL", str(table_path), "bad=" + ",".join(bad_counts)))

    cdsl = table[table["dataset"].astype(str).eq("CDSL")]
    cdsl_raw_ok = not cdsl.empty and set(cdsl["calibration_method"].astype(str)) == {"raw_traditional"}
    rows.append(row("cdsl_raw_traditional_rows", "PASS" if cdsl_raw_ok else "FAIL", str(table_path), "methods=" + ",".join(sorted(set(cdsl["calibration_method"].astype(str))))))

    missing_method_groups = []
    for dataset in ["eICU", "CHARLS"]:
        dataset_rows = table[table["dataset"].astype(str).eq(dataset)]
        for model in EXPECTED_MODELS:
            model_rows = dataset_rows[dataset_rows["model"].astype(str).eq(model)]
            methods = set(model_rows["calibration_method"].dropna().astype(str))
            raw = "raw" if model == "logistic_regression_balanced" else "raw_model_comparison"
            expected = {raw, *EXPECTED_VALIDATION_METHODS}
            if not expected <= methods:
                missing_method_groups.append(f"{dataset}/{model}:{','.join(sorted(methods))}")
    rows.append(row("eicu_charls_methods_complete", "PASS" if not missing_method_groups else "FAIL", str(table_path), "bad=" + ";".join(missing_method_groups)))

    metric_ok = (
        table["event_rate"].dropna().between(0, 1).all()
        and table["mean_absolute_calibration_error"].dropna().between(0, 1).all()
        and table["max_absolute_calibration_error"].dropna().between(0, 1).all()
    )
    rows.append(row("calibration_metrics_in_unit_interval", "PASS" if metric_ok else "FAIL", str(table_path), "calibration metrics within [0,1]"))
    rank_ok = table["calibration_rank_within_model"].fillna(0).astype(float).ge(1).all()
    rows.append(row("calibration_ranks_present", "PASS" if rank_ok else "FAIL", str(table_path), "rank >= 1"))
    decision_ok = table["decision_thresholds"].fillna(0).astype(float).ge(8).all()
    rows.append(row("decision_curve_thresholds_present", "PASS" if decision_ok else "FAIL", str(table_path), f"min={float(table['decision_thresholds'].min())}"))
    best_threshold_ok = table["decision_best_threshold"].dropna().between(0, 1).all()
    rows.append(row("decision_best_threshold_range", "PASS" if best_threshold_ok else "FAIL", str(table_path), "best threshold within [0,1]"))

    selected_count = int(table["is_selected_technical_summary_row"].fillna(False).astype(bool).sum())
    rows.append(row("selected_technical_rows_represented", "PASS" if selected_count == 6 else "FAIL", str(table_path), f"selected={selected_count}"))
    selected_names = set(table[table["is_selected_technical_summary_row"].fillna(False).astype(bool)]["benchmark_row"].dropna().astype(str))
    rows.append(row("selected_rows_named", "PASS" if len(selected_names) == 6 else "FAIL", str(table_path), "rows=" + ",".join(sorted(selected_names))))

    improved = table[
        table["dataset"].astype(str).isin(["eICU", "CHARLS"])
        & table["calibration_method"].astype(str).isin(EXPECTED_VALIDATION_METHODS)
    ]
    improvement_ok = not improved.empty and improved["calibration_mae_delta_vs_raw"].lt(0).any()
    rows.append(row("validation_calibration_improves_some_rows", "PASS" if improvement_ok else "FAIL", str(table_path), "expects at least one negative MAE delta vs raw"))

    notes = " ".join(table["interpretation_note"].astype(str))
    rows.append(row("boundary_notes_present", "PASS" if "not chronic readmission validation" in notes and "naive upper-reference" in notes else "FAIL", str(table_path), "expects eICU and CDSL boundary notes"))
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rows.append(row("report_declares_no_clinical_thresholds", "PASS" if "do not define clinical action thresholds" in report_text else "FAIL", str(report_path), "decision-curve boundary required"))
    rows.append(row("report_avoids_treatment_recommendations", "PASS" if "recommended treatment" not in report_text.lower() else "FAIL", str(report_path), "no treatment recommendation wording"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_calibration_decision_summary_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_calibration_decision_summary_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Calibration and Decision-Curve Summary Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research calibration and decision-curve summary outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External calibration/decision summary validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
