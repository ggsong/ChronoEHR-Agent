#!/usr/bin/env python3
"""Build a formal external-summary asset manifest for mentor handoff."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


ASSETS = [
    {
        "asset_id": "start_here_external_technical_summary",
        "package_section": "01_start_here",
        "audience_role": "mentor_entry_point",
        "formal_role": "main_summary_table",
        "table_number": "Package T1",
        "path": "outputs/tables/external_technical_summary_table.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-technical-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-technical-summary",
        "boundary_note": "Research model evaluation summary; no clinical deployment or treatment recommendation.",
    },
    {
        "asset_id": "start_here_external_technical_report",
        "package_section": "01_start_here",
        "audience_role": "mentor_entry_point",
        "formal_role": "summary_report",
        "table_number": "",
        "path": "outputs/reports/external_technical_summary.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-technical-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-technical-summary",
        "boundary_note": "Research technical summary only; not medical QA.",
    },
    {
        "asset_id": "external_benchmark_summary",
        "package_section": "02_main_tables",
        "audience_role": "mentor_entry_point",
        "formal_role": "main_summary_table",
        "table_number": "Package T2",
        "path": "outputs/tables/external_benchmark_summary_table.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-benchmark-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary",
        "boundary_note": "Selected external benchmark rows with CDSL/eICU/CHARLS interpretation boundaries.",
    },
    {
        "asset_id": "external_calibration_decision_summary",
        "package_section": "02_main_tables",
        "audience_role": "mentor_detail",
        "formal_role": "main_summary_table",
        "table_number": "Package T3",
        "path": "outputs/tables/external_calibration_decision_summary.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-calibration-decision-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-calibration-decision-summary",
        "boundary_note": "Decision-curve comparison is threshold-level research utility only; no clinical action threshold.",
    },
    {
        "asset_id": "external_model_selection_rationale",
        "package_section": "02_main_tables",
        "audience_role": "mentor_detail",
        "formal_role": "main_summary_table",
        "table_number": "Package T4",
        "path": "outputs/tables/external_model_selection_rationale.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-model-selection-rationale",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-model-selection-rationale",
        "boundary_note": "Deterministic model-selection rules for the six selected external rows.",
    },
    {
        "asset_id": "external_subgroup_robustness_summary",
        "package_section": "02_main_tables",
        "audience_role": "mentor_detail",
        "formal_role": "main_summary_table",
        "table_number": "Package T5",
        "path": "outputs/tables/external_subgroup_robustness_summary.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-subgroup-robustness-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-subgroup-robustness-summary",
        "boundary_note": "Selected-row subgroup robustness summary; reports caution where subgroup evidence is sparse or lower CI bounds are weak.",
    },
    {
        "asset_id": "external_threshold_band_sensitivity",
        "package_section": "02_main_tables",
        "audience_role": "mentor_detail",
        "formal_role": "main_summary_table",
        "table_number": "Package T6",
        "path": "outputs/tables/external_threshold_band_sensitivity.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-threshold-band-sensitivity",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-threshold-band-sensitivity",
        "boundary_note": "Selected-row threshold-band decision-curve sensitivity; threshold grid points are research evaluation values only.",
    },
    {
        "asset_id": "external_calibration_method_rationale",
        "package_section": "02_main_tables",
        "audience_role": "mentor_detail",
        "formal_role": "main_summary_table",
        "table_number": "Package T7",
        "path": "outputs/tables/external_calibration_method_rationale.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-calibration-method-rationale",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-calibration-method-rationale",
        "boundary_note": "Selected-row calibration-method rationale across raw/intercept/Platt/isotonic candidates.",
    },
    {
        "asset_id": "table_s13_external_benchmark_summary",
        "package_section": "03_supplementary_tables",
        "audience_role": "supplement",
        "formal_role": "supplementary_table",
        "table_number": "Table S13",
        "path": "outputs/tables/supplementary_appendix/table_s13_external_benchmark_summary.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-benchmark-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary",
        "boundary_note": "Supplementary external benchmark summary.",
    },
    {
        "asset_id": "table_s14_external_benchmark_hard_metrics",
        "package_section": "03_supplementary_tables",
        "audience_role": "supplement",
        "formal_role": "supplementary_table",
        "table_number": "Table S14",
        "path": "outputs/tables/supplementary_appendix/table_s14_external_benchmark_hard_metrics.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-benchmark-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary",
        "boundary_note": "Full hard-metric rows used by selected summaries.",
    },
    {
        "asset_id": "table_s15_external_technical_summary",
        "package_section": "03_supplementary_tables",
        "audience_role": "supplement",
        "formal_role": "supplementary_table",
        "table_number": "Table S15",
        "path": "outputs/tables/supplementary_appendix/table_s15_external_technical_summary.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-technical-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-technical-summary",
        "boundary_note": "Supplementary technical summary with CI, subgroup, calibration, and decision-curve fields.",
    },
    {
        "asset_id": "table_s16_external_calibration_decision_summary",
        "package_section": "03_supplementary_tables",
        "audience_role": "supplement",
        "formal_role": "supplementary_table",
        "table_number": "Table S16",
        "path": "outputs/tables/supplementary_appendix/table_s16_external_calibration_decision_summary.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-calibration-decision-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-calibration-decision-summary",
        "boundary_note": "Supplementary calibration and decision-curve comparison.",
    },
    {
        "asset_id": "table_s17_external_model_selection_rationale",
        "package_section": "03_supplementary_tables",
        "audience_role": "supplement",
        "formal_role": "supplementary_table",
        "table_number": "Table S17",
        "path": "outputs/tables/supplementary_appendix/table_s17_external_model_selection_rationale.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-model-selection-rationale",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-model-selection-rationale",
        "boundary_note": "Supplementary deterministic model-selection rationale.",
    },
    {
        "asset_id": "table_s18_external_subgroup_robustness_summary",
        "package_section": "03_supplementary_tables",
        "audience_role": "supplement",
        "formal_role": "supplementary_table",
        "table_number": "Table S18",
        "path": "outputs/tables/supplementary_appendix/table_s18_external_subgroup_robustness_summary.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-subgroup-robustness-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-subgroup-robustness-summary",
        "boundary_note": "Supplementary selected-row subgroup robustness summary.",
    },
    {
        "asset_id": "table_s19_external_threshold_band_sensitivity",
        "package_section": "03_supplementary_tables",
        "audience_role": "supplement",
        "formal_role": "supplementary_table",
        "table_number": "Table S19",
        "path": "outputs/tables/supplementary_appendix/table_s19_external_threshold_band_sensitivity.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-threshold-band-sensitivity",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-threshold-band-sensitivity",
        "boundary_note": "Supplementary threshold-band sensitivity table.",
    },
    {
        "asset_id": "table_s20_external_calibration_method_rationale",
        "package_section": "03_supplementary_tables",
        "audience_role": "supplement",
        "formal_role": "supplementary_table",
        "table_number": "Table S20",
        "path": "outputs/tables/supplementary_appendix/table_s20_external_calibration_method_rationale.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-calibration-method-rationale",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-calibration-method-rationale",
        "boundary_note": "Supplementary calibration-method rationale table.",
    },
    {
        "asset_id": "external_metric_consistency_audit",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "cross_table_audit",
        "table_number": "",
        "path": "outputs/tables/external_metric_consistency_audit.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-metric-consistency-audit",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-metric-consistency-audit",
        "boundary_note": "Checks that repeated external metrics match across tables.",
    },
    {
        "asset_id": "external_metric_consistency_audit_report",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "cross_table_audit",
        "table_number": "",
        "path": "outputs/reports/external_metric_consistency_audit.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-metric-consistency-audit",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-metric-consistency-audit",
        "boundary_note": "Readable cross-table consistency audit.",
    },
    {
        "asset_id": "external_benchmark_summary_validation",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "validation_report",
        "table_number": "",
        "path": "outputs/reports/external_benchmark_summary_validation.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-benchmark-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-benchmark-summary",
        "boundary_note": "Validation for external benchmark summary and boundaries.",
    },
    {
        "asset_id": "external_technical_summary_validation",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "validation_report",
        "table_number": "",
        "path": "outputs/reports/external_technical_summary_validation.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-technical-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-technical-summary",
        "boundary_note": "Validation for the technical summary package entry point.",
    },
    {
        "asset_id": "external_calibration_decision_summary_validation",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "validation_report",
        "table_number": "",
        "path": "outputs/reports/external_calibration_decision_summary_validation.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-calibration-decision-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-calibration-decision-summary",
        "boundary_note": "Validation for calibration and decision-curve summary.",
    },
    {
        "asset_id": "external_model_selection_rationale_validation",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "validation_report",
        "table_number": "",
        "path": "outputs/reports/external_model_selection_rationale_validation.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-model-selection-rationale",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-model-selection-rationale",
        "boundary_note": "Validation for model-selection rationale.",
    },
    {
        "asset_id": "external_subgroup_robustness_summary_validation",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "validation_report",
        "table_number": "",
        "path": "outputs/reports/external_subgroup_robustness_summary_validation.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-subgroup-robustness-summary",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-subgroup-robustness-summary",
        "boundary_note": "Validation for selected-row subgroup robustness summary.",
    },
    {
        "asset_id": "external_threshold_band_sensitivity_validation",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "validation_report",
        "table_number": "",
        "path": "outputs/reports/external_threshold_band_sensitivity_validation.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-threshold-band-sensitivity",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-threshold-band-sensitivity",
        "boundary_note": "Validation for threshold-band sensitivity.",
    },
    {
        "asset_id": "external_calibration_method_rationale_validation",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "validation_report",
        "table_number": "",
        "path": "outputs/reports/external_calibration_method_rationale_validation.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-calibration-method-rationale",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-calibration-method-rationale",
        "boundary_note": "Validation for calibration-method rationale.",
    },
    {
        "asset_id": "external_metric_consistency_audit_validation",
        "package_section": "04_validation_evidence",
        "audience_role": "validation_evidence",
        "formal_role": "validation_report",
        "table_number": "",
        "path": "outputs/reports/external_metric_consistency_audit_validation.md",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-metric-consistency-audit",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-metric-consistency-audit",
        "boundary_note": "Validation for cross-table consistency audit.",
    },
    {
        "asset_id": "external_model_bootstrap_ci",
        "package_section": "05_internal_audit_sources",
        "audience_role": "internal_audit",
        "formal_role": "source_table",
        "table_number": "",
        "path": "outputs/tables/external_model_bootstrap_ci.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-bootstrap-ci",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-bootstrap-ci",
        "boundary_note": "Bootstrap CI source for selected external metrics.",
    },
    {
        "asset_id": "external_subgroup_bootstrap_ci",
        "package_section": "05_internal_audit_sources",
        "audience_role": "internal_audit",
        "formal_role": "source_table",
        "table_number": "",
        "path": "outputs/tables/external_subgroup_bootstrap_ci.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-subgroup-bootstrap-ci",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-subgroup-bootstrap-ci",
        "boundary_note": "Subgroup bootstrap CI source for technical summary.",
    },
    {
        "asset_id": "external_subgroup_performance",
        "package_section": "05_internal_audit_sources",
        "audience_role": "internal_audit",
        "formal_role": "source_table",
        "table_number": "",
        "path": "outputs/tables/external_subgroup_performance.csv",
        "source_command": "python3 src/chrono_ehr/run_study.py --external-subgroup-performance",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-subgroup-performance",
        "boundary_note": "Subgroup performance source table.",
    },
    {
        "asset_id": "cdsl_auroc_figure",
        "package_section": "06_figures",
        "audience_role": "optional_visual",
        "formal_role": "figure",
        "table_number": "",
        "path": "outputs/figures/cdsl_temporal_benchmark_auroc.png",
        "source_command": "python3 src/chrono_ehr/run_study.py --cdsl-summary-figures",
        "validation_command": "",
        "boundary_note": "Optional CDSL temporal benchmark AUROC figure.",
    },
    {
        "asset_id": "cdsl_auprc_figure",
        "package_section": "06_figures",
        "audience_role": "optional_visual",
        "formal_role": "figure",
        "table_number": "",
        "path": "outputs/figures/cdsl_temporal_benchmark_auprc.png",
        "source_command": "python3 src/chrono_ehr/run_study.py --cdsl-summary-figures",
        "validation_command": "",
        "boundary_note": "Optional CDSL temporal benchmark AUPRC figure.",
    },
    {
        "asset_id": "eicu_roc_figure",
        "package_section": "06_figures",
        "audience_role": "optional_visual",
        "formal_role": "figure",
        "table_number": "",
        "path": "outputs/figures/eicu_first24h_logistic_roc.png",
        "source_command": "python3 src/chrono_ehr/run_study.py --eicu-baseline-figures",
        "validation_command": "python3 src/chrono_ehr/run_study.py --validate-eicu-baseline-figures",
        "boundary_note": "Optional eICU logistic ROC figure.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(project_root: Path) -> pd.DataFrame:
    rows = []
    for asset in ASSETS:
        path = project_root / asset["path"]
        exists = path.exists() and path.stat().st_size > 0
        rows.append(
            {
                **asset,
                "exists": exists,
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "sha256": sha256(path) if exists else "",
                "status": "PASS" if exists else "FAIL",
            }
        )
    boundary_rows = [
        {
            "asset_id": "boundary_cdsl_full_stay",
            "package_section": "07_boundary_statements",
            "audience_role": "boundary_statement",
            "formal_role": "boundary_statement",
            "table_number": "",
            "path": "outputs/reports/external_summary_asset_manifest.md",
            "source_command": "python3 src/chrono_ehr/run_study.py --external-summary-asset-manifest",
            "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-summary-asset-manifest",
            "boundary_note": "CDSL full-stay row is a naive upper-reference, not early prediction performance.",
            "exists": True,
            "size_bytes": 0,
            "sha256": "",
            "status": "PASS",
        },
        {
            "asset_id": "boundary_eicu_task_scope",
            "package_section": "07_boundary_statements",
            "audience_role": "boundary_statement",
            "formal_role": "boundary_statement",
            "table_number": "",
            "path": "outputs/reports/external_summary_asset_manifest.md",
            "source_command": "python3 src/chrono_ehr/run_study.py --external-summary-asset-manifest",
            "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-summary-asset-manifest",
            "boundary_note": "eICU is an ICU mortality benchmark, not chronic readmission external validation.",
            "exists": True,
            "size_bytes": 0,
            "sha256": "",
            "status": "PASS",
        },
        {
            "asset_id": "boundary_charls_scope",
            "package_section": "07_boundary_statements",
            "audience_role": "boundary_statement",
            "formal_role": "boundary_statement",
            "table_number": "",
            "path": "outputs/reports/external_summary_asset_manifest.md",
            "source_command": "python3 src/chrono_ehr/run_study.py --external-summary-asset-manifest",
            "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-summary-asset-manifest",
            "boundary_note": "CHARLS is a longitudinal cohort extension for incident diabetes.",
            "exists": True,
            "size_bytes": 0,
            "sha256": "",
            "status": "PASS",
        },
        {
            "asset_id": "boundary_decision_curve_scope",
            "package_section": "07_boundary_statements",
            "audience_role": "boundary_statement",
            "formal_role": "boundary_statement",
            "table_number": "",
            "path": "outputs/reports/external_summary_asset_manifest.md",
            "source_command": "python3 src/chrono_ehr/run_study.py --external-summary-asset-manifest",
            "validation_command": "python3 src/chrono_ehr/run_study.py --validate-external-summary-asset-manifest",
            "boundary_note": "Decision-curve outputs do not define a clinical action threshold.",
            "exists": True,
            "size_bytes": 0,
            "sha256": "",
            "status": "PASS",
        },
    ]
    return pd.DataFrame(rows + boundary_rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "package_section",
        "asset_id",
        "audience_role",
        "formal_role",
        "table_number",
        "path",
        "status",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, manifest: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_summary_asset_manifest.md"
    failures = manifest[manifest["status"].ne("PASS")]
    by_section = manifest.groupby("package_section")["asset_id"].count().to_dict()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# External Summary Asset Manifest

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Assets: {len(manifest)}
- Failures: {len(failures)}
- Sections: {", ".join(f"{section}={count}" for section, count in sorted(by_section.items()))}
- Boundary: research model evaluation handoff package only; no diagnosis, treatment, deployment, or clinical action threshold recommendation.

## Asset Table

{markdown_table(manifest)}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "external_summary_asset_manifest.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(table_path, index=False)
    report = write_report(args.project_root, manifest)
    failures = int(manifest["status"].ne("PASS").sum())
    print(f"External summary assets: {len(manifest)}")
    print(f"Failures: {failures}")
    print(f"Wrote {report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
