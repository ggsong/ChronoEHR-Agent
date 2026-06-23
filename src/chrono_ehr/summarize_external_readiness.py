#!/usr/bin/env python3
"""Summarize external benchmark readiness across CDSL, eICU, and CHARLS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def cdsl_status(project_root: Path) -> dict[str, Any]:
    summary = read_csv(project_root / "outputs" / "tables" / "cdsl_external_validation_summary.csv")
    metrics = read_csv(project_root / "outputs" / "tables" / "cdsl_traditional_baselines_metrics.csv")
    audit = read_csv(project_root / "outputs" / "tables" / "cdsl_leakage_audit.csv")
    ready = not summary.empty and not metrics.empty and not audit.empty
    best = metrics[metrics["split"].eq("test")].sort_values(["AUPRC", "AUROC"], ascending=False).head(1)
    early = metrics[
        metrics["split"].eq("test") & metrics["feature_set"].ne("full_stay_naive_reference")
    ].sort_values(["AUPRC", "AUROC"], ascending=False).head(1)
    return {
        "dataset": "CDSL",
        "role": "external structured EHR time-series method benchmark",
        "local_status": "READY" if ready else "NOT_READY",
        "recommended_first_task": "COVID hospitalization mortality prediction",
        "current_outputs": "readiness, temporal benchmark, leakage audit, traditional baselines, figures, supplementary S11/S12"
        if ready
        else "missing one or more CDSL benchmark outputs",
        "n_subjects_or_patients": int(summary["patients"].iloc[0]) if not summary.empty and "patients" in summary else None,
        "n_rows_or_records": int(summary["formatted_rows"].iloc[0]) if not summary.empty and "formatted_rows" in summary else None,
        "best_test_AUROC": float(best["AUROC"].iloc[0]) if not best.empty else None,
        "best_test_AUPRC": float(best["AUPRC"].iloc[0]) if not best.empty else None,
        "best_early_test_AUROC": float(early["AUROC"].iloc[0]) if not early.empty else None,
        "best_early_test_AUPRC": float(early["AUPRC"].iloc[0]) if not early.empty else None,
        "critical_blocker": "",
        "next_action": "Keep as supplementary external method validation; do not label as chronic readmission validation.",
    }


def eicu_status(project_root: Path) -> dict[str, Any]:
    tables = read_csv(project_root / "outputs" / "tables" / "eicu_data_readiness_expected_tables.csv")
    required = tables[tables["required_for_first_demo"].eq(True)] if not tables.empty else pd.DataFrame()
    required_present = bool(not required.empty and required["present"].all())
    missing = required[required["present"].eq(False)]["table"].tolist() if not required.empty else ["patient", "lab", "vitalPeriodic"]
    cohort_path = project_root / "data" / "processed" / "eicu_temporal_mortality_cohort.csv"
    cohort_ready = cohort_path.exists() and cohort_path.stat().st_size > 0
    feature_path = project_root / "data" / "processed" / "eicu_first24h_feature_matrix_skeleton.csv"
    leakage_path = project_root / "outputs" / "tables" / "eicu_leakage_gate.csv"
    feature_ready = feature_path.exists() and feature_path.stat().st_size > 0 and leakage_path.exists() and leakage_path.stat().st_size > 0
    baseline = read_csv(project_root / "outputs" / "tables" / "eicu_first24h_logistic_baseline_metrics.csv")
    test = baseline[baseline["split"].eq("test")] if not baseline.empty and "split" in baseline else pd.DataFrame()
    baseline_ready = not test.empty
    figures_ready = all(
        (project_root / relative).exists() and (project_root / relative).stat().st_size > 0
        for relative in [
            "outputs/figures/eicu_first24h_logistic_roc.png",
            "outputs/figures/eicu_first24h_logistic_precision_recall.png",
            "outputs/figures/eicu_first24h_logistic_calibration_deciles.png",
            "outputs/tables/eicu_first24h_calibration_summary.csv",
        ]
    )
    summary_table_ready = (project_root / "outputs" / "tables" / "external_benchmark_summary_table.csv").exists()
    status = "BASELINE_READY" if baseline_ready else ("FEATURE_READY" if feature_ready else ("COHORT_READY" if cohort_ready else ("READY_FOR_COHORT_CODE" if required_present else "DATA_PENDING")))
    outputs = "readiness report, protocol, config template, feature-time map, leakage checklist"
    if cohort_ready:
        outputs += ", first-24h mortality cohort skeleton, cohort validation"
    if feature_ready:
        outputs += ", first-24h lab/vital feature skeleton, eICU leakage gate"
    if baseline_ready:
        outputs += ", first-24h logistic baseline"
    if figures_ready:
        outputs += ", ROC/PR/calibration figures, calibration summary"
    best_auroc = float(test["AUROC"].iloc[0]) if baseline_ready and "AUROC" in test else None
    best_auprc = float(test["AUPRC"].iloc[0]) if baseline_ready and "AUPRC" in test else None
    return {
        "dataset": "eICU",
        "role": "planned external multicenter ICU EHR benchmark",
        "local_status": status,
        "recommended_first_task": "ICU first-24h hospital mortality prediction",
        "current_outputs": outputs,
        "n_subjects_or_patients": None,
        "n_rows_or_records": None,
        "best_test_AUROC": best_auroc,
        "best_test_AUPRC": best_auprc,
        "best_early_test_AUROC": best_auroc,
        "best_early_test_AUPRC": best_auprc,
        "critical_blocker": "Missing required raw tables: " + ", ".join(missing) if missing else "",
        "next_action": "Keep the CDSL/eICU external benchmark summary table current; use it as the concise external-method evidence table." if summary_table_ready else ("Build a concise external benchmark summary table combining CDSL and eICU." if figures_ready else ("Summarize eICU baseline and add calibration/figures." if baseline_ready else ("Start eICU first-24h baseline specs and lightweight traditional baseline." if feature_ready else ("Set EICU_ROOT to the complete eICU CSV directory, then rerun --eicu-readiness.")))),
    }


def charls_status(project_root: Path) -> dict[str, Any]:
    waves = read_csv(project_root / "outputs" / "tables" / "charls_wave_detection.csv")
    detected = set(waves[waves["detected"].eq(True)]["wave"]) if not waves.empty else set()
    required = {"2011_wave1", "2013_wave2"}
    ready = required.issubset(detected)
    missing = sorted(required - detected)
    return {
        "dataset": "CHARLS",
        "role": "planned longitudinal chronic-disease cohort extension",
        "local_status": "READY_FOR_PROTOCOL_CODE" if ready else "DATA_PENDING",
        "recommended_first_task": "2011 baseline prediction of incident diabetes by 2013/2015",
        "current_outputs": "readiness report, incident diabetes protocol, config template, wave map, leakage checklist",
        "n_subjects_or_patients": None,
        "n_rows_or_records": None,
        "best_test_AUROC": None,
        "best_test_AUPRC": None,
        "best_early_test_AUROC": None,
        "best_early_test_AUPRC": None,
        "critical_blocker": "Missing required waves: " + ", ".join(missing) if missing else "",
        "next_action": "After CHARLS approval, set CHARLS_ROOT to the downloaded waves and rerun --charls-readiness.",
    }


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in ["best_test_AUROC", "best_test_AUPRC", "best_early_test_AUROC", "best_early_test_AUPRC"]:
        if col in display:
            display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    display = display.astype(object).where(pd.notna(display), "")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, summary: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "external_benchmark_readiness_summary.md"
    ready_count = int(summary["local_status"].isin(["READY", "READY_FOR_COHORT_CODE", "COHORT_READY", "FEATURE_READY", "BASELINE_READY", "READY_FOR_PROTOCOL_CODE"]).sum())
    text = f"""# External Benchmark Readiness Summary

- Datasets tracked: {len(summary)}
- Ready or partly ready: {ready_count}
- Boundary: these are research-method benchmarks and extensions, not clinical decision-support systems.

## Summary Table

{markdown_table(summary)}

## Interpretation

- CDSL is already usable as an external structured EHR time-series method benchmark.
- CDSL reports both a full-stay reference metric and an early-window metric. The early-window metric is the appropriate number for prediction-time comparisons; the full-stay metric is kept only as a naive upper-reference.
- eICU is planned as an external multicenter ICU benchmark. If its status is `BASELINE_READY`, the first-24h cohort, lab/vital feature skeleton, leakage gate, and lightweight logistic baseline already exist.
- CHARLS is planned as a longitudinal chronic-disease cohort extension, but approved wave files are not yet local.
- eICU and CHARLS should enter the pipeline only after readiness is rerun and status changes from `DATA_PENDING`.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    rows = [cdsl_status(args.project_root), eicu_status(args.project_root), charls_status(args.project_root)]
    summary = pd.DataFrame(rows)
    table_path = args.project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    report = write_report(args.project_root, summary)
    print(f"Wrote {report}")
    print(summary[["dataset", "local_status", "recommended_first_task"]].to_string(index=False))


if __name__ == "__main__":
    main()
