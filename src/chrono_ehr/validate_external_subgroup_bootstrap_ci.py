#!/usr/bin/env python3
"""Validate external subgroup bootstrap confidence interval outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_DATASETS = {"CDSL", "eICU", "CHARLS"}
EXPECTED_SUBGROUP_TYPES = {"age", "sex", "gender"}
KEY_COLS = [
    "dataset",
    "feature_set",
    "model",
    "calibration_method",
    "subgroup_type",
    "subgroup",
]


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


def row_key(frame: pd.DataFrame) -> pd.Series:
    normalized = frame.copy()
    for col in KEY_COLS:
        if col not in normalized:
            normalized[col] = ""
        normalized[col] = normalized[col].fillna("").astype(str)
    return normalized[KEY_COLS].agg("||".join, axis=1)


def validate(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "external_subgroup_bootstrap_ci.csv"
    report_path = project_root / "outputs" / "reports" / "external_subgroup_bootstrap_ci.md"
    point_path = project_root / "outputs" / "tables" / "external_subgroup_performance.csv"
    ci = read_csv(table_path)
    point = read_csv(point_path)
    rows = [
        row("ci_table_exists", "PASS" if not ci.empty else "FAIL", str(table_path), f"rows={len(ci)}"),
        row("ci_report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "markdown report"),
        row("point_table_exists", "PASS" if not point.empty else "FAIL", str(point_path), f"rows={len(point)}"),
    ]
    if ci.empty:
        return pd.DataFrame(rows)

    datasets = set(ci["dataset"].astype(str))
    rows.append(row("expected_datasets", "PASS" if EXPECTED_DATASETS <= datasets else "FAIL", str(table_path), "datasets=" + ",".join(sorted(datasets))))
    subgroup_types = set(ci["subgroup_type"].astype(str))
    rows.append(row("expected_subgroup_types", "PASS" if {"age"} <= subgroup_types and subgroup_types <= EXPECTED_SUBGROUP_TYPES else "FAIL", str(table_path), "types=" + ",".join(sorted(subgroup_types))))

    if not point.empty:
        missing_ci = sorted(set(row_key(point)) - set(row_key(ci)))
        rows.append(row("matches_subgroup_point_rows", "PASS" if not missing_ci else "FAIL", str(table_path), f"missing={len(missing_ci)}"))

    valid_sizes = ci["n"].gt(0).all() and ci["events"].ge(0).all() and ci["events"].le(ci["n"]).all()
    rows.append(row("valid_sample_sizes", "PASS" if valid_sizes else "FAIL", str(table_path), "0 <= events <= n"))
    metric_cols = ["event_rate", "Brier", "Brier_lower", "Brier_upper"]
    range_ok = all(ci[col].dropna().between(0, 1).all() for col in metric_cols if col in ci.columns)
    for col in ["AUROC", "AUPRC", "AUROC_lower", "AUROC_upper", "AUPRC_lower", "AUPRC_upper"]:
        range_ok = range_ok and ci[col].dropna().between(0, 1).all()
    rows.append(row("metric_ranges", "PASS" if range_ok else "FAIL", str(table_path), "metrics and CI bounds within [0,1] when defined"))

    ok = ci[ci["status"].astype(str).eq("OK")].copy()
    small = ci[ci["status"].astype(str).ne("OK")].copy()
    rows.append(row("has_evaluable_rows", "PASS" if not ok.empty else "FAIL", str(table_path), f"OK rows={len(ok)}"))
    per_dataset_ok = ok.groupby("dataset").size().to_dict() if not ok.empty else {}
    bad = [dataset for dataset in EXPECTED_DATASETS if int(per_dataset_ok.get(dataset, 0)) == 0]
    rows.append(row("each_dataset_has_evaluable_rows", "PASS" if not bad else "FAIL", str(table_path), "bad=" + ",".join(bad)))

    interval_ok = True
    if not ok.empty:
        interval_ok = (
            ok["AUROC"].between(ok["AUROC_lower"], ok["AUROC_upper"]).all()
            and ok["AUPRC"].between(ok["AUPRC_lower"], ok["AUPRC_upper"]).all()
            and ok["Brier"].between(ok["Brier_lower"], ok["Brier_upper"]).all()
        )
    rows.append(row("ok_point_estimates_inside_intervals", "PASS" if interval_ok else "FAIL", str(table_path), "OK rows have AUROC/AUPRC/Brier inside 95% CI"))
    enough_replicates = ok["bootstrap_replicates"].ge(450).all() if not ok.empty else False
    rows.append(row("ok_rows_have_enough_replicates", "PASS" if enough_replicates else "FAIL", str(table_path), f"min={int(ok['bootstrap_replicates'].min()) if not ok.empty else 0}"))
    small_replicates_ok = small["bootstrap_replicates"].fillna(0).eq(0).all() if not small.empty else True
    rows.append(row("small_rows_do_not_fake_intervals", "PASS" if small_replicates_ok else "FAIL", str(table_path), f"small_rows={len(small)}"))

    expected_comparison_models = {"random_forest_balanced", "hist_gradient_boosting_weighted"}
    eicu_models = set(ci[ci["dataset"].astype(str).eq("eICU")]["model"].dropna().astype(str)) if "model" in ci else set()
    charls_models = set(ci[ci["dataset"].astype(str).eq("CHARLS")]["model"].dropna().astype(str)) if "model" in ci else set()
    rows.append(row("eicu_model_comparison_subgroup_ci", "PASS" if expected_comparison_models <= eicu_models else "FAIL", str(table_path), "models=" + ",".join(sorted(eicu_models))))
    rows.append(row("charls_model_comparison_subgroup_ci", "PASS" if expected_comparison_models <= charls_models else "FAIL", str(table_path), "models=" + ",".join(sorted(charls_models))))
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
    rows.append(row("model_comparison_calibrated_subgroup_ci", "PASS" if not missing_method_groups else "FAIL", str(table_path), "bad=" + ";".join(missing_method_groups)))
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
    table_path = args.project_root / "outputs" / "tables" / "external_subgroup_bootstrap_ci_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_subgroup_bootstrap_ci_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Subgroup Bootstrap CI Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates external benchmark subgroup uncertainty outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External subgroup bootstrap CI validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
