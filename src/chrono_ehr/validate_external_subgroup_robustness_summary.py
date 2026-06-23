#!/usr/bin/env python3
"""Validate the external subgroup robustness summary."""

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
    table_path = project_root / "outputs" / "tables" / "external_subgroup_robustness_summary.csv"
    supp_path = project_root / "outputs" / "tables" / "supplementary_appendix" / "table_s18_external_subgroup_robustness_summary.csv"
    report_path = project_root / "outputs" / "reports" / "external_subgroup_robustness_summary.md"
    table = read_csv(table_path)
    rows = [
        row("robustness_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("supplement_table_exists", "PASS" if exists(supp_path) else "FAIL", str(supp_path), f"size={supp_path.stat().st_size if supp_path.exists() else 0}"),
        row("report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)

    required_columns = {
        "benchmark_row",
        "dataset",
        "feature_set",
        "model",
        "calibration_method",
        "subgroup_ci_rows",
        "subgroup_ci_ok_rows",
        "subgroup_ci_small_or_single_class_rows",
        "subgroup_types",
        "min_bootstrap_replicates",
        "weakest_auroc_subgroup",
        "weakest_auroc_n",
        "weakest_auroc_events",
        "weakest_auroc",
        "weakest_auroc_lower",
        "weakest_auprc_subgroup",
        "weakest_auprc_n",
        "weakest_auprc_events",
        "weakest_auprc",
        "weakest_auprc_lower",
        "highest_brier_upper_subgroup",
        "highest_brier_upper",
        "robustness_status",
        "robustness_note",
    }
    missing = sorted(required_columns - set(table.columns))
    rows.append(row("required_columns_present", "PASS" if not missing else "FAIL", str(table_path), "missing=" + ",".join(missing)))
    labels = set(table["benchmark_row"].astype(str))
    rows.append(row("expected_rows_present", "PASS" if EXPECTED_ROWS <= labels else "FAIL", str(table_path), "rows=" + ",".join(sorted(labels))))
    rows.append(row("exactly_six_rows", "PASS" if len(table) == 6 else "FAIL", str(table_path), f"rows={len(table)}"))
    datasets = set(table["dataset"].astype(str))
    rows.append(row("expected_datasets", "PASS" if EXPECTED_DATASETS <= datasets else "FAIL", str(table_path), "datasets=" + ",".join(sorted(datasets))))
    rows.append(row("all_rows_have_evaluable_subgroups", "PASS" if table["subgroup_ci_ok_rows"].astype(int).gt(0).all() else "FAIL", str(table_path), "subgroup_ci_ok_rows > 0"))
    rows.append(row("all_rows_have_bootstrap_support", "PASS" if table["min_bootstrap_replicates"].astype(int).ge(450).all() else "FAIL", str(table_path), f"min={int(table['min_bootstrap_replicates'].min())}"))
    rows.append(row("subgroup_counts_consistent", "PASS" if (table["subgroup_ci_rows"].astype(int) == table["subgroup_ci_ok_rows"].astype(int) + table["subgroup_ci_small_or_single_class_rows"].astype(int)).all() else "FAIL", str(table_path), "rows = OK + small/single-class"))

    metric_ok = True
    for col in ["weakest_auroc", "weakest_auroc_lower", "weakest_auroc_upper", "weakest_auprc", "weakest_auprc_lower", "weakest_auprc_upper", "highest_brier_upper"]:
        if col in table:
            metric_ok = metric_ok and table[col].dropna().between(0, 1).all()
    rows.append(row("metrics_in_unit_interval", "PASS" if metric_ok else "FAIL", str(table_path), "metrics in [0,1]"))
    statuses = set(table["robustness_status"].astype(str))
    rows.append(row("status_values_known", "PASS" if statuses <= {"SUPPORTED", "CAUTION", "INSUFFICIENT"} else "FAIL", str(table_path), "statuses=" + ",".join(sorted(statuses))))
    rows.append(row("no_insufficient_rows", "PASS" if "INSUFFICIENT" not in statuses else "FAIL", str(table_path), "no selected row should lack subgroup support"))
    rows.append(row("weakest_subgroups_named", "PASS" if table["weakest_auroc_subgroup"].astype(str).str.len().gt(0).all() and table["weakest_auprc_subgroup"].astype(str).str.len().gt(0).all() else "FAIL", str(table_path), "weakest subgroup descriptors present"))

    notes = " ".join(table["robustness_note"].astype(str))
    rows.append(row("boundary_notes_present", "PASS" if "naive upper-reference" in notes and "not chronic readmission" in notes and "longitudinal cohort extension" in notes else "FAIL", str(table_path), "expects CDSL/eICU/CHARLS boundaries"))
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rows.append(row("report_declares_research_boundary", "PASS" if "research subgroup robustness summary only" in report_text else "FAIL", str(report_path), "research-only boundary"))
    forbidden = ["recommended treatment", "ready for clinical deployment", "clinical action threshold recommendation"]
    rows.append(row("report_avoids_clinical_claims", "PASS" if not any(token in report_text.lower() for token in forbidden) else "FAIL", str(report_path), "forbidden clinical wording absent"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_subgroup_robustness_summary_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_subgroup_robustness_summary_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Subgroup Robustness Summary Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research subgroup robustness outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External subgroup robustness validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
