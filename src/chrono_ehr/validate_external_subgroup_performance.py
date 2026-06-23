#!/usr/bin/env python3
"""Validate external subgroup performance outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_DATASETS = {"CDSL", "eICU", "CHARLS"}
EXPECTED_SUBGROUP_TYPES = {"age", "sex", "gender"}


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
    table_path = project_root / "outputs" / "tables" / "external_subgroup_performance.csv"
    report_path = project_root / "outputs" / "reports" / "external_subgroup_performance.md"
    table = read_csv(table_path)
    rows = [
        row("table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
    ]
    if table.empty:
        return pd.DataFrame(rows)

    datasets = set(table["dataset"].astype(str))
    rows.append(row("expected_datasets", "PASS" if EXPECTED_DATASETS <= datasets else "FAIL", str(table_path), "datasets=" + ",".join(sorted(datasets))))
    subgroup_types = set(table["subgroup_type"].astype(str))
    rows.append(row("expected_subgroup_types", "PASS" if {"age"} <= subgroup_types and subgroup_types <= EXPECTED_SUBGROUP_TYPES else "FAIL", str(table_path), "types=" + ",".join(sorted(subgroup_types))))
    valid_sizes = table["n"].gt(0).all() and table["events"].ge(0).all() and table["events"].le(table["n"]).all()
    rows.append(row("valid_sample_sizes", "PASS" if valid_sizes else "FAIL", str(table_path), "0 <= events <= n"))
    metric_cols = ["event_rate", "Brier"]
    range_ok = all(table[col].dropna().between(0, 1).all() for col in metric_cols)
    for col in ["AUROC", "AUPRC"]:
        range_ok = range_ok and table[col].dropna().between(0, 1).all()
    rows.append(row("metric_ranges", "PASS" if range_ok else "FAIL", str(table_path), "metrics within [0,1] when defined"))
    ok_rows = int(table["status"].astype(str).eq("OK").sum()) if "status" in table else 0
    rows.append(row("has_evaluable_rows", "PASS" if ok_rows > 0 else "FAIL", str(table_path), f"OK rows={ok_rows}"))
    per_dataset_ok = table.groupby("dataset")["status"].apply(lambda values: values.astype(str).eq("OK").sum()).to_dict()
    bad = [dataset for dataset in EXPECTED_DATASETS if int(per_dataset_ok.get(dataset, 0)) == 0]
    rows.append(row("each_dataset_has_evaluable_rows", "PASS" if not bad else "FAIL", str(table_path), "bad=" + ",".join(bad)))
    expected_comparison_models = {"random_forest_balanced", "hist_gradient_boosting_weighted"}
    eicu_models = set(table[table["dataset"].astype(str).eq("eICU")]["model"].dropna().astype(str)) if "model" in table else set()
    charls_models = set(table[table["dataset"].astype(str).eq("CHARLS")]["model"].dropna().astype(str)) if "model" in table else set()
    rows.append(row("eicu_model_comparison_subgroups", "PASS" if expected_comparison_models <= eicu_models else "FAIL", str(table_path), "models=" + ",".join(sorted(eicu_models))))
    rows.append(row("charls_model_comparison_subgroups", "PASS" if expected_comparison_models <= charls_models else "FAIL", str(table_path), "models=" + ",".join(sorted(charls_models))))
    expected_model_methods = {"raw_model_comparison", "intercept_validation", "platt_validation", "isotonic_validation"}
    missing_method_groups = []
    for dataset in ["eICU", "CHARLS"]:
        for model in expected_comparison_models:
            methods = set(
                table[
                    table["dataset"].astype(str).eq(dataset)
                    & table["model"].astype(str).eq(model)
                ]["calibration_method"].dropna().astype(str)
            )
            if not expected_model_methods <= methods:
                missing_method_groups.append(f"{dataset}/{model}:{','.join(sorted(methods))}")
    rows.append(row("model_comparison_calibrated_subgroups", "PASS" if not missing_method_groups else "FAIL", str(table_path), "bad=" + ";".join(missing_method_groups)))
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
    table_path = args.project_root / "outputs" / "tables" / "external_subgroup_performance_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_subgroup_performance_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Subgroup Performance Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research subgroup outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External subgroup validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
