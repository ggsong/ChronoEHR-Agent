#!/usr/bin/env python3
"""Validate the external technical summary artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_ROWS = {
    "CDSL early-window best",
    "CDSL full-stay naive reference",
    "eICU calibrated logistic reference",
    "eICU best calibrated RF/HGB",
    "CHARLS calibrated logistic reference",
    "CHARLS best calibrated RF/HGB",
}
EXPECTED_DATASETS = {"CDSL", "eICU", "CHARLS"}


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


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def validate(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "external_technical_summary_table.csv"
    supp_path = project_root / "outputs" / "tables" / "supplementary_appendix" / "table_s15_external_technical_summary.csv"
    report_path = project_root / "outputs" / "reports" / "external_technical_summary.md"
    table = read_csv(table_path)
    rows = [
        row("summary_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("supplement_table_exists", "PASS" if exists(supp_path) else "FAIL", str(supp_path), f"size={supp_path.stat().st_size if supp_path.exists() else 0}"),
        row("report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)

    required_columns = {
        "benchmark_row",
        "dataset",
        "role",
        "task",
        "feature_window",
        "model",
        "calibration_method",
        "n",
        "events",
        "event_rate",
        "auroc_ci",
        "auprc_ci",
        "brier_ci",
        "mean_absolute_calibration_error",
        "subgroup_ci_ok_rows",
        "subgroup_ci_min_replicates",
        "decision_thresholds",
        "decision_model_preferred_thresholds",
        "leakage_gate_status",
        "is_prediction_time_valid",
        "model_selection_note",
        "subgroup_ci_note",
        "decision_curve_note",
        "interpretation_note",
    }
    missing = sorted(required_columns - set(table.columns))
    rows.append(row("required_columns_present", "PASS" if not missing else "FAIL", str(table_path), "missing=" + ",".join(missing)))

    labels = set(table["benchmark_row"].astype(str))
    rows.append(row("expected_benchmark_rows", "PASS" if EXPECTED_ROWS <= labels else "FAIL", str(table_path), "rows=" + ",".join(sorted(labels))))
    rows.append(row("exactly_six_rows", "PASS" if len(table) == 6 else "FAIL", str(table_path), f"rows={len(table)}"))
    datasets = set(table["dataset"].astype(str))
    rows.append(row("expected_datasets", "PASS" if EXPECTED_DATASETS <= datasets else "FAIL", str(table_path), "datasets=" + ",".join(sorted(datasets))))

    metric_strings_ok = table["auroc_ci"].astype(str).str.contains(r"\(", regex=True).all() and table["auprc_ci"].astype(str).str.contains(r"\(", regex=True).all()
    rows.append(row("metric_ci_strings_present", "PASS" if metric_strings_ok else "FAIL", str(table_path), "AUROC/AUPRC display strings include 95% CI"))
    brier_strings_ok = table["brier_ci"].astype(str).str.len().gt(0).all()
    rows.append(row("brier_summary_present", "PASS" if brier_strings_ok else "FAIL", str(table_path), "Brier display strings present"))

    subgroup_ok = table["subgroup_ci_ok_rows"].fillna(0).astype(float).gt(0).all()
    rows.append(row("all_rows_have_subgroup_ci", "PASS" if subgroup_ok else "FAIL", str(table_path), "subgroup_ci_ok_rows > 0 for every selected row"))
    replicate_ok = table["subgroup_ci_min_replicates"].fillna(0).astype(float).ge(500).all()
    rows.append(row("subgroup_ci_replicates_complete", "PASS" if replicate_ok else "FAIL", str(table_path), f"min={float(table['subgroup_ci_min_replicates'].min())}"))
    decision_ok = table["decision_thresholds"].fillna(0).astype(float).gt(0).all()
    rows.append(row("all_rows_have_decision_curve_summary", "PASS" if decision_ok else "FAIL", str(table_path), "decision_thresholds > 0 for every selected row"))

    notes = " ".join(table["interpretation_note"].astype(str).tolist() + table["model_selection_note"].astype(str).tolist())
    rows.append(row("cdsl_full_stay_boundary_present", "PASS" if "Naive upper" in notes or "naive upper" in notes else "FAIL", str(table_path), "expects CDSL full-stay upper-reference note"))
    rows.append(row("eicu_boundary_present", "PASS" if "not a chronic readmission external validation" in notes else "FAIL", str(table_path), "expects eICU boundary note"))
    rows.append(row("rf_hgb_comparison_notes_present", "PASS" if "AUPRC delta" in notes and "Brier delta" in notes else "FAIL", str(table_path), "expects RF/HGB comparison deltas"))

    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    forbidden = ["ready for clinical deployment", "recommended treatment", "diagnosis system performance"]
    rows.append(row("report_declares_boundary", "PASS" if "research model evaluation only" in report_text and "not medical QA" in report_text else "FAIL", str(report_path), "expects explicit research boundary"))
    rows.append(row("report_avoids_clinical_claims", "PASS" if not any(token in report_text.lower() for token in forbidden) else "FAIL", str(report_path), "forbidden clinical-system wording absent"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_technical_summary_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_technical_summary_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Technical Summary Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research technical summary outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External technical summary validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
