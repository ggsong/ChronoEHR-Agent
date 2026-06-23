#!/usr/bin/env python3
"""Validate the ChronoEHR-Agent progress score artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


REQUIRED_COMPONENTS = {
    "mainline_mvp_gate",
    "primary_diabetes_demo",
    "mimic_replication_cohorts",
    "agent_control_layer",
    "external_benchmark_readiness",
    "documentation_reproducibility",
    "TOTAL",
}


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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path) -> pd.DataFrame:
    table_path = project_root / "outputs" / "tables" / "agent_progress_score.csv"
    report_path = project_root / "outputs" / "reports" / "agent_progress_score.md"
    table = read_csv(table_path)
    report = read_text(report_path)
    rows = [
        row("progress_score_table_exists", "PASS" if not table.empty else "FAIL", str(table_path), f"rows={len(table)}"),
        row("progress_score_report_exists", "PASS" if bool(report) else "FAIL", str(report_path), f"chars={len(report)}"),
    ]
    if table.empty:
        return pd.DataFrame(rows)
    required_columns = {"component", "weight", "fraction_complete", "weighted_points", "status", "detail"}
    missing_columns = sorted(required_columns - set(table.columns))
    rows.append(row("required_columns", "PASS" if not missing_columns else "FAIL", str(table_path), "missing=" + ",".join(missing_columns)))

    components = set(table["component"].astype(str)) if "component" in table else set()
    missing_components = sorted(REQUIRED_COMPONENTS - components)
    rows.append(row("required_components_present", "PASS" if not missing_components else "FAIL", str(table_path), "missing=" + ",".join(missing_components)))

    component_rows = table[table["component"].astype(str).ne("TOTAL")]
    total_rows = table[table["component"].astype(str).eq("TOTAL")]
    weight_sum = float(pd.to_numeric(component_rows["weight"], errors="coerce").fillna(0).sum()) if "weight" in component_rows else 0.0
    rows.append(row("component_weights_sum_to_100", "PASS" if abs(weight_sum - 100.0) < 0.01 else "FAIL", str(table_path), f"weight_sum={weight_sum:.3f}"))
    if not total_rows.empty:
        total = float(pd.to_numeric(total_rows.iloc[0]["weighted_points"], errors="coerce"))
        recomputed = float(pd.to_numeric(component_rows["weighted_points"], errors="coerce").fillna(0).sum())
        rows.append(row("total_matches_component_sum", "PASS" if abs(total - recomputed) < 0.01 else "FAIL", str(table_path), f"total={total:.3f}; recomputed={recomputed:.3f}"))
        rows.append(row("score_at_least_mvp_threshold", "PASS" if total >= 95.0 else "FAIL", str(table_path), f"total={total:.3f}"))
    else:
        rows.append(row("total_row_present", "FAIL", str(table_path), "missing TOTAL row"))
    rows.append(row("report_declares_nonclinical_boundary", "PASS" if "no medical QA" in report and "no medical" in report else "FAIL", str(report_path), "expects boundary text"))
    rows.append(row("report_mentions_charls_gap", "PASS" if "CHARLS" in report else "FAIL", str(report_path), "expects CHARLS pending interpretation"))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_progress_score_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_progress_score_validation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Progress Score Validation

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: validates local project-progress scoring only; no medical QA, diagnosis, or treatment recommendation.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent progress-score checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
