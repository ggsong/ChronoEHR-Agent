#!/usr/bin/env python3
"""Validate the external model-selection rationale table."""

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


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "external_model_selection_rationale.csv"
    supp_path = project_root / "outputs" / "tables" / "supplementary_appendix" / "table_s17_external_model_selection_rationale.csv"
    report_path = project_root / "outputs" / "reports" / "external_model_selection_rationale.md"
    table = read_csv(table_path)
    rows = [
        row("rationale_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("supplement_table_exists", "PASS" if exists(supp_path) else "FAIL", str(supp_path), f"size={supp_path.stat().st_size if supp_path.exists() else 0}"),
        row("report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)

    required = {
        "benchmark_row",
        "dataset",
        "candidate_description",
        "ranking_rule",
        "candidate_count",
        "selected_feature_set",
        "selected_model",
        "selected_calibration_method",
        "selected_AUROC",
        "selected_AUPRC",
        "selected_Brier",
        "runner_up_model",
        "delta_AUPRC_vs_runner_up",
        "delta_Brier_vs_runner_up",
        "selected_matches_summary",
        "selection_status",
        "boundary_note",
    }
    missing = sorted(required - set(table.columns))
    rows.append(row("required_columns_present", "PASS" if not missing else "FAIL", str(table_path), "missing=" + ",".join(missing)))
    labels = set(table["benchmark_row"].astype(str))
    rows.append(row("expected_rows_present", "PASS" if EXPECTED_ROWS <= labels else "FAIL", str(table_path), "rows=" + ",".join(sorted(labels))))
    rows.append(row("exactly_six_rows", "PASS" if len(table) == 6 else "FAIL", str(table_path), f"rows={len(table)}"))
    datasets = set(table["dataset"].astype(str))
    rows.append(row("expected_datasets", "PASS" if EXPECTED_DATASETS <= datasets else "FAIL", str(table_path), "datasets=" + ",".join(sorted(datasets))))
    rows.append(row("candidate_counts_positive", "PASS" if table["candidate_count"].astype(int).gt(0).all() else "FAIL", str(table_path), "candidate_count > 0"))
    rows.append(row("all_selected_match_summary", "PASS" if table["selected_matches_summary"].astype(bool).all() else "FAIL", str(table_path), "selected identity matches external summary"))
    rows.append(row("selection_status_pass", "PASS" if table["selection_status"].astype(str).eq("PASS").all() else "FAIL", str(table_path), "selection_status all PASS"))

    metric_ok = (
        table["selected_AUROC"].dropna().between(0, 1).all()
        and table["selected_AUPRC"].dropna().between(0, 1).all()
        and table["selected_Brier"].dropna().between(0, 1).all()
    )
    rows.append(row("selected_metrics_in_unit_interval", "PASS" if metric_ok else "FAIL", str(table_path), "selected metrics in [0,1]"))
    runner_ok = table["runner_up_model"].fillna("").astype(str).str.len().gt(0).all()
    rows.append(row("runner_up_present", "PASS" if runner_ok else "FAIL", str(table_path), "runner-up model present for every rule"))

    selected = dict(zip(table["benchmark_row"].astype(str), table["selected_model"].astype(str)))
    expected_models = {
        "CDSL early-window best": "logistic_regression_balanced",
        "CDSL full-stay naive reference": "logistic_regression_balanced",
        "eICU calibrated logistic reference": "logistic_regression_balanced",
        "eICU best calibrated RF/HGB": "hist_gradient_boosting_weighted",
        "CHARLS calibrated logistic reference": "logistic_regression_balanced",
        "CHARLS best calibrated RF/HGB": "random_forest_balanced",
    }
    bad_models = [label for label, model in expected_models.items() if selected.get(label) != model]
    rows.append(row("expected_selected_models", "PASS" if not bad_models else "FAIL", str(table_path), "bad=" + ",".join(bad_models)))

    methods = dict(zip(table["benchmark_row"].astype(str), table["selected_calibration_method"].astype(str)))
    expected_methods = {
        "CDSL early-window best": "raw_traditional",
        "CDSL full-stay naive reference": "raw_traditional",
        "eICU calibrated logistic reference": "platt_validation",
        "eICU best calibrated RF/HGB": "platt_validation",
        "CHARLS calibrated logistic reference": "platt_validation",
        "CHARLS best calibrated RF/HGB": "platt_validation",
    }
    bad_methods = [label for label, method in expected_methods.items() if methods.get(label) != method]
    rows.append(row("expected_selected_methods", "PASS" if not bad_methods else "FAIL", str(table_path), "bad=" + ",".join(bad_methods)))

    notes = " ".join(table["boundary_note"].astype(str))
    rows.append(row("boundary_notes_present", "PASS" if "Naive upper-reference" in notes and "not chronic readmission validation" in notes else "FAIL", str(table_path), "expects CDSL/eICU boundary notes"))
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rows.append(row("report_declares_research_boundary", "PASS" if "research model evaluation only" in report_text else "FAIL", str(report_path), "research-only boundary"))
    rows.append(row("report_has_no_clinical_recommendation", "PASS" if "recommended treatment" not in report_text.lower() else "FAIL", str(report_path), "no treatment recommendation wording"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_model_selection_rationale_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_model_selection_rationale_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Model-Selection Rationale Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research model-selection rationale outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External model-selection rationale validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
