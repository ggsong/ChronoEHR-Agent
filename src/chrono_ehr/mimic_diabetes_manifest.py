#!/usr/bin/env python3
"""Create a machine-readable and human-readable manifest for the diabetes demo."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]

REQUIRED_OUTPUTS = {
    "cohort": "data/processed/mimic_diabetes_readmission_cohort.csv",
    "lab_features": "data/processed/mimic_diabetes_lab_features.csv",
    "lab24h_features": "data/processed/mimic_diabetes_lab_features_24h.csv",
    "med_features": "data/processed/mimic_diabetes_med_features.csv",
    "med24h_features": "data/processed/mimic_diabetes_med_features_24h.csv",
    "cohort_summary": "outputs/tables/mimic_diabetes_cohort_summary.csv",
    "model_performance": "outputs/tables/mimic_diabetes_model_performance.csv",
    "bootstrap_ci": "outputs/tables/mimic_diabetes_model_performance_bootstrap_ci.csv",
    "fixed_alert_burden": "outputs/tables/mimic_diabetes_fixed_alert_burden.csv",
    "feature_time_audit": "outputs/tables/mimic_diabetes_feature_time_audit.csv",
    "lab24h_feature_availability": "outputs/tables/mimic_diabetes_lab24h_feature_availability.csv",
    "med24h_feature_availability": "outputs/tables/mimic_diabetes_med24h_feature_availability.csv",
    "prediction_time_comparison": "outputs/tables/mimic_diabetes_prediction_time_comparison.csv",
    "prediction_time_model_performance": "outputs/tables/mimic_diabetes_prediction_time_model_performance.csv",
    "prediction_time_auroc_figure": "outputs/figures/mimic_diabetes_prediction_time_auroc.png",
    "prediction_time_auprc_figure": "outputs/figures/mimic_diabetes_prediction_time_auprc.png",
    "leakage_sensitivity": "outputs/tables/mimic_diabetes_leakage_sensitivity.csv",
    "outcome_sensitivity": "outputs/tables/mimic_diabetes_outcome_sensitivity.csv",
    "roc_curve": "outputs/figures/mimic_diabetes_roc_curve.png",
    "pr_curve": "outputs/figures/mimic_diabetes_precision_recall_curve.png",
    "calibration_curve": "outputs/figures/mimic_diabetes_calibration_deciles.png",
    "methods_results": "outputs/reports/mimic_diabetes_methods_results_draft.md",
    "feature_time_audit_report": "outputs/reports/mimic_diabetes_feature_time_audit_report.md",
    "lab24h_feature_report": "outputs/reports/mimic_diabetes_lab24h_feature_report.md",
    "med24h_feature_report": "outputs/reports/mimic_diabetes_med24h_feature_report.md",
    "prediction_time_comparison_report": "outputs/reports/mimic_diabetes_prediction_time_comparison_report.md",
    "prediction_time_model_report": "outputs/reports/mimic_diabetes_prediction_time_model_report.md",
    "prediction_time_figure_report": "outputs/reports/mimic_diabetes_prediction_time_figure_report.md",
    "comprehensive_audit_report": "outputs/reports/mimic_diabetes_comprehensive_audit_report.md",
    "leakage_report": "outputs/reports/mimic_diabetes_leakage_sensitivity_report.md",
    "outcome_report": "outputs/reports/mimic_diabetes_outcome_sensitivity_report.md",
    "uncertainty_report": "outputs/reports/mimic_diabetes_model_uncertainty_report.md",
}


def file_info(relative_path: str) -> dict[str, Any]:
    path = PROJECT / relative_path
    if not path.exists():
        return {"path": relative_path, "exists": False}
    return {
        "path": relative_path,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def metric_dict(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return df.set_index("metric")["value"].astype(str).to_dict()


def read_core_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    summary_path = PROJECT / "outputs" / "tables" / "mimic_diabetes_cohort_summary.csv"
    if summary_path.exists():
        summary = metric_dict(summary_path)
        metrics["cohort"] = {
            "final_index_admissions": int(float(summary["final_index_admissions"])),
            "final_subjects": int(float(summary["final_subjects"])),
            "readmission_30d_count": int(float(summary["readmission_30d_count"])),
            "readmission_30d_rate": float(summary["readmission_30d_rate"]),
        }

    perf_path = PROJECT / "outputs" / "tables" / "mimic_diabetes_model_performance.csv"
    if perf_path.exists():
        perf = pd.read_csv(perf_path)
        test = perf[perf["split"].eq("test")].copy()
        metrics["test_performance"] = [
            {
                "feature_set": row.feature_set,
                "AUROC": float(row.AUROC),
                "AUPRC": float(row.AUPRC),
                "Brier_score": float(row.Brier_score),
                "sensitivity": float(row.sensitivity),
                "specificity": float(row.specificity),
                "PPV": float(row.ppv),
                "NPV": float(row.npv),
            }
            for row in test.itertuples(index=False)
        ]

    leakage_path = PROJECT / "outputs" / "tables" / "mimic_diabetes_leakage_sensitivity.csv"
    if leakage_path.exists():
        leakage = pd.read_csv(leakage_path)
        metrics["leakage_sensitivity"] = [
            {
                "scenario": row.scenario,
                "AUROC": float(row.AUROC),
                "AUPRC": float(row.AUPRC),
            }
            for row in leakage.itertuples(index=False)
        ]

    outcome_path = PROJECT / "outputs" / "tables" / "mimic_diabetes_outcome_sensitivity.csv"
    if outcome_path.exists():
        outcome = pd.read_csv(outcome_path)
        metrics["outcome_sensitivity"] = [
            {
                "outcome_definition": row.outcome_definition,
                "events": int(row.events),
                "event_rate": float(row.event_rate),
            }
            for row in outcome.itertuples(index=False)
        ]

    return metrics


def build_manifest() -> dict[str, Any]:
    files = {name: file_info(path) for name, path in REQUIRED_OUTPUTS.items()}
    missing = [name for name, info in files.items() if not info["exists"]]
    return {
        "project": "ChronoEHR-Agent",
        "demo": "mimic_iv_diabetes_30d_readmission",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT),
        "mimic_root": "${MIMIC_IV_ROOT}",
        "status": "complete" if not missing else "incomplete",
        "missing_outputs": missing,
        "files": files,
        "metrics": read_core_metrics(),
        "recommended_next_steps": [
            "Add first-24h vital sign features and compare them with first-24h labs/medications.",
            "Add scikit-learn Random Forest baseline.",
            "Add XGBoost or LightGBM baseline.",
            "Extend to CKD and heart failure cohorts.",
        ],
    }


def markdown_table_test_performance(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Feature set | AUROC | AUPRC | Brier | Sensitivity | Specificity | PPV | NPV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature_set']} | {row['AUROC']:.4f} | {row['AUPRC']:.4f} | {row['Brier_score']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['PPV']:.4f} | {row['NPV']:.4f} |"
        )
    return "\n".join(lines)


def write_status_markdown(manifest: dict[str, Any], path: Path) -> None:
    metrics = manifest["metrics"]
    cohort = metrics.get("cohort", {})
    perf_rows = metrics.get("test_performance", [])
    missing = manifest["missing_outputs"]
    missing_text = "None" if not missing else ", ".join(missing)

    leakage_lines = []
    for row in metrics.get("leakage_sensitivity", []):
        leakage_lines.append(f"- {row['scenario']}: AUROC {row['AUROC']:.4f}, AUPRC {row['AUPRC']:.4f}")
    outcome_lines = []
    for row in metrics.get("outcome_sensitivity", []):
        outcome_lines.append(f"- {row['outcome_definition']}: {row['events']:,} events ({row['event_rate']:.2%})")

    text = f"""# ChronoEHR-Agent Pipeline Status

Generated at: {manifest["generated_at"]}

## Status

- Demo: `{manifest["demo"]}`
- Status: `{manifest["status"]}`
- Missing outputs: {missing_text}

## Cohort

- Final index admissions: {cohort.get("final_index_admissions", "NA"):,}
- Final subjects: {cohort.get("final_subjects", "NA"):,}
- 30-day readmission count: {cohort.get("readmission_30d_count", "NA"):,}
- 30-day readmission rate: {cohort.get("readmission_30d_rate", 0):.2%}

## Test Performance

{markdown_table_test_performance(perf_rows) if perf_rows else "No model performance table found."}

## Leakage Sensitivity

{chr(10).join(leakage_lines) if leakage_lines else "No leakage sensitivity table found."}

## Outcome Sensitivity

{chr(10).join(outcome_lines) if outcome_lines else "No outcome sensitivity table found."}

## Most Important Files

- `README.md`
- `docs/resume_state.md`
- `docs/tool_development_roadmap.md`
- `outputs/reports/mimic_diabetes_prediction_time_comparison_report.md`
- `outputs/reports/mimic_diabetes_feature_time_audit_report.md`
- `outputs/reports/mimic_diabetes_prediction_time_model_report.md`
- `outputs/reports/mimic_diabetes_lab24h_feature_report.md`
- `outputs/reports/mimic_diabetes_med24h_feature_report.md`
- `outputs/reports/mimic_diabetes_methods_results_draft.md`
- `outputs/reports/mimic_diabetes_model_uncertainty_report.md`

## Recommended Next Steps

{chr(10).join(f"- {step}" for step in manifest["recommended_next_steps"])}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    outputs_dir = PROJECT / "outputs"
    reports_dir = outputs_dir / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    manifest_path = outputs_dir / "pipeline_manifest.json"
    status_path = reports_dir / "pipeline_status.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_status_markdown(manifest, status_path)

    print(f"Wrote {manifest_path}")
    print(f"Wrote {status_path}")
    print(f"status={manifest['status']}")


if __name__ == "__main__":
    main()
