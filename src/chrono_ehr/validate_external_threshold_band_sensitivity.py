#!/usr/bin/env python3
"""Validate external threshold-band sensitivity outputs."""

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
EXPECTED_BANDS = {"very_low_0.02_0.05", "low_mid_0.10_0.20", "high_0.30_0.50"}


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
    table_path = project_root / "outputs" / "tables" / "external_threshold_band_sensitivity.csv"
    supp_path = project_root / "outputs" / "tables" / "supplementary_appendix" / "table_s19_external_threshold_band_sensitivity.csv"
    report_path = project_root / "outputs" / "reports" / "external_threshold_band_sensitivity.md"
    table = read_csv(table_path)
    rows = [
        row("table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("supplement_exists", "PASS" if exists(supp_path) else "FAIL", str(supp_path), f"size={supp_path.stat().st_size if supp_path.exists() else 0}"),
        row("report_exists", "PASS" if exists(report_path) else "FAIL", str(report_path), f"size={report_path.stat().st_size if report_path.exists() else 0}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)
    required = {
        "benchmark_row",
        "dataset",
        "feature_set",
        "model",
        "calibration_method",
        "threshold_band",
        "threshold_lower",
        "threshold_upper",
        "threshold_count",
        "model_preferred_thresholds",
        "positive_advantage_thresholds",
        "min_net_benefit_advantage",
        "mean_net_benefit_advantage",
        "max_net_benefit_advantage",
        "best_threshold",
        "best_alert_rate",
        "best_ppv",
        "best_recall",
        "band_status",
        "boundary_note",
    }
    missing = sorted(required - set(table.columns))
    rows.append(row("required_columns_present", "PASS" if not missing else "FAIL", str(table_path), "missing=" + ",".join(missing)))
    labels = set(table["benchmark_row"].astype(str))
    bands = set(table["threshold_band"].astype(str))
    rows.append(row("expected_rows_present", "PASS" if EXPECTED_ROWS <= labels else "FAIL", str(table_path), "rows=" + ",".join(sorted(labels))))
    rows.append(row("expected_bands_present", "PASS" if EXPECTED_BANDS <= bands else "FAIL", str(table_path), "bands=" + ",".join(sorted(bands))))
    rows.append(row("exactly_eighteen_rows", "PASS" if len(table) == 18 else "FAIL", str(table_path), f"rows={len(table)}"))
    grid_counts = table.groupby("benchmark_row")["threshold_band"].nunique().to_dict()
    rows.append(row("three_bands_per_selected_row", "PASS" if all(int(grid_counts.get(label, 0)) == 3 for label in EXPECTED_ROWS) else "FAIL", str(table_path), str(grid_counts)))
    rows.append(row("threshold_counts_positive", "PASS" if table["threshold_count"].astype(int).gt(0).all() else "FAIL", str(table_path), "threshold_count > 0"))
    range_ok = True
    for col in ["threshold_lower", "threshold_upper", "best_threshold", "best_alert_rate", "best_ppv", "best_recall"]:
        range_ok = range_ok and table[col].dropna().between(0, 1).all()
    rows.append(row("probability_fields_in_unit_interval", "PASS" if range_ok else "FAIL", str(table_path), "probability fields in [0,1]"))
    status_values = set(table["band_status"].astype(str))
    rows.append(row("status_values_known", "PASS" if status_values <= {"CONSISTENT_MODEL_ADVANTAGE", "MIXED", "NO_MODEL_ADVANTAGE", "NO_DATA"} else "FAIL", str(table_path), "statuses=" + ",".join(sorted(status_values))))
    rows.append(row("some_band_has_model_advantage", "PASS" if table["positive_advantage_thresholds"].astype(int).gt(0).any() else "FAIL", str(table_path), "positive advantage present"))
    notes = " ".join(table["boundary_note"].astype(str))
    rows.append(row("boundary_notes_present", "PASS" if "not chronic readmission" in notes and "naive upper-reference" in notes and "longitudinal cohort extension" in notes else "FAIL", str(table_path), "expects dataset boundary notes"))
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rows.append(row("report_declares_research_boundary", "PASS" if "research decision-curve sensitivity only" in report_text else "FAIL", str(report_path), "research boundary"))
    forbidden = ["recommended treatment", "ready for clinical deployment"]
    rows.append(row("report_avoids_clinical_claims", "PASS" if not any(token in report_text.lower() for token in forbidden) else "FAIL", str(report_path), "forbidden wording absent"))
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
    table_path = args.project_root / "outputs" / "tables" / "external_threshold_band_sensitivity_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "external_threshold_band_sensitivity_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# External Threshold-Band Sensitivity Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research threshold-band outputs only.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"External threshold-band validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
