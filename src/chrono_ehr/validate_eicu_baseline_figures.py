#!/usr/bin/env python3
"""Validate eICU baseline figures and calibration outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


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


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def validate(project_root: Path) -> pd.DataFrame:
    figures = project_root / "outputs" / "figures"
    tables = project_root / "outputs" / "tables"
    reports = project_root / "outputs" / "reports"
    paths = {
        "roc": figures / "eicu_first24h_logistic_roc.png",
        "pr": figures / "eicu_first24h_logistic_precision_recall.png",
        "calibration": figures / "eicu_first24h_logistic_calibration_deciles.png",
        "deciles": tables / "eicu_first24h_calibration_deciles.csv",
        "summary": tables / "eicu_first24h_calibration_summary.csv",
        "report": reports / "eicu_baseline_figures_report.md",
    }
    rows = [row(f"{name}_exists", "PASS" if exists(path) else "FAIL", str(path), f"size={path.stat().st_size if path.exists() else 0}") for name, path in paths.items()]
    deciles = read_csv(paths["deciles"])
    summary = read_csv(paths["summary"])
    rows.append(row("deciles_have_10_bins", "PASS" if not deciles.empty and deciles["decile"].nunique() == 10 else "FAIL", str(paths["deciles"]), f"bins={deciles['decile'].nunique() if not deciles.empty and 'decile' in deciles else 0}"))
    if not summary.empty:
        required = {"mean_absolute_calibration_error", "max_absolute_calibration_error"}
        missing = sorted(required - set(summary.columns))
        rows.append(row("summary_columns", "PASS" if not missing else "FAIL", str(paths["summary"]), "missing=" + ",".join(missing)))
        if not missing:
            values_ok = summary[list(required)].notna().all().all()
            rows.append(row("summary_values_nonmissing", "PASS" if values_ok else "FAIL", str(paths["summary"]), summary[list(required)].to_string(index=False)))
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
    table_path = args.project_root / "outputs" / "tables" / "eicu_baseline_figures_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "eicu_baseline_figures_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# eICU Baseline Figures Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates research figures/calibration outputs only; no clinical recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"eICU figure validation checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
