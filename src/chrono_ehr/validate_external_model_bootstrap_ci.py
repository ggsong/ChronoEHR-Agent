#!/usr/bin/env python3
"""Validate external model bootstrap confidence interval outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_DATASETS = {"CDSL", "eICU", "CHARLS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "external_model_bootstrap_ci.csv"
    report_path = project_root / "outputs" / "reports" / "external_model_bootstrap_ci.md"
    ci = read_csv(table_path)
    rows = [
        row("ci_table_exists", "PASS" if not ci.empty else "FAIL", str(table_path), f"rows={len(ci)}"),
        row("ci_report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
    ]
    if ci.empty:
        return pd.DataFrame(rows)

    datasets = set(ci["dataset"].astype(str))
    rows.append(row("expected_datasets", "PASS" if EXPECTED_DATASETS <= datasets else "FAIL", str(table_path), "datasets=" + ",".join(sorted(datasets))))
    min_rows = {"CDSL": 12, "eICU": 12, "CHARLS": 12}
    counts = ci.groupby("dataset").size().to_dict()
    bad_counts = [dataset for dataset, minimum in min_rows.items() if int(counts.get(dataset, 0)) < minimum]
    rows.append(row("minimum_rows_per_dataset", "PASS" if not bad_counts else "FAIL", str(table_path), "bad=" + ",".join(bad_counts)))
    eicu_models = set(ci[ci["dataset"].astype(str).eq("eICU")]["model"].dropna().astype(str))
    charls_models = set(ci[ci["dataset"].astype(str).eq("CHARLS")]["model"].dropna().astype(str))
    expected_comparison_models = {"random_forest_balanced", "hist_gradient_boosting_weighted"}
    rows.append(row("eicu_model_comparison_rows", "PASS" if expected_comparison_models <= eicu_models else "FAIL", str(table_path), "models=" + ",".join(sorted(eicu_models))))
    rows.append(row("charls_model_comparison_rows", "PASS" if expected_comparison_models <= charls_models else "FAIL", str(table_path), "models=" + ",".join(sorted(charls_models))))
    expected_model_methods = {"raw_model_comparison", "intercept_validation", "platt_validation", "isotonic_validation"}
    missing_method_groups = []
    for dataset in ["eICU", "CHARLS"]:
        for model in expected_comparison_models:
            methods = set(
                ci[
                    ci["dataset"].astype(str).eq(dataset)
                    & ci["model"].astype(str).eq(model)
                ]["calibration_method"].dropna().astype(str)
            )
            if not expected_model_methods <= methods:
                missing_method_groups.append(f"{dataset}/{model}:{','.join(sorted(methods))}")
    rows.append(row("model_comparison_calibrated_methods", "PASS" if not missing_method_groups else "FAIL", str(table_path), "bad=" + ";".join(missing_method_groups)))

    metric_cols = ["AUROC", "AUPRC", "Brier", "AUROC_lower", "AUROC_upper", "AUPRC_lower", "AUPRC_upper", "Brier_lower", "Brier_upper"]
    range_ok = all(ci[col].between(0, 1).all() for col in metric_cols if col in ci.columns)
    rows.append(row("metric_ranges", "PASS" if range_ok else "FAIL", str(table_path), "metrics and CI bounds within [0,1]"))
    interval_ok = (
        ci["AUROC"].between(ci["AUROC_lower"], ci["AUROC_upper"]).all()
        and ci["AUPRC"].between(ci["AUPRC_lower"], ci["AUPRC_upper"]).all()
        and ci["Brier"].between(ci["Brier_lower"], ci["Brier_upper"]).all()
    )
    rows.append(row("point_estimates_inside_intervals", "PASS" if interval_ok else "FAIL", str(table_path), "AUROC/AUPRC/Brier inside 95% CI"))
    enough_replicates = ci["bootstrap_replicates"].ge(450).all()
    rows.append(row("enough_bootstrap_replicates", "PASS" if enough_replicates else "FAIL", str(table_path), f"min={int(ci['bootstrap_replicates'].min())}"))
    positive_sample_sizes = ci["n"].gt(0).all() and ci["events"].gt(0).all() and ci["events"].lt(ci["n"]).all()
    rows.append(row("valid_test_sample_sizes", "PASS" if positive_sample_sizes else "FAIL", str(table_path), "0 < events < n for all rows"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_model_bootstrap_ci_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_model_bootstrap_ci_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Model Bootstrap CI Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates external benchmark uncertainty outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External bootstrap CI validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
