#!/usr/bin/env python3
"""Generate an English Methods/Results brief from completed ChronoEHR outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


DEFAULT_TEMPLATE = DEFAULT_PROJECT / "configs" / "report_text_templates_english_brief.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing English brief template: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(project_root: Path, relative: str) -> pd.DataFrame:
    path = project_root / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def fmt_metric(value: float) -> str:
    return f"{float(value):.3f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def cohort_table(cohort: pd.DataFrame) -> pd.DataFrame:
    table = cohort.copy()
    table["Cohort"] = table["cohort"].map(
        {
            "diabetes": "Diabetes",
            "ckd": "Chronic kidney disease",
            "heart_failure": "Heart failure",
            "hypertension": "Hypertension",
        }
    )
    table["Index admissions"] = table["final_index_admissions"].map(fmt_int)
    table["Patients"] = table["final_subjects"].map(fmt_int)
    table["30-day readmissions"] = table["readmission_30d_count"].map(fmt_int)
    table["Event rate"] = table["readmission_30d_rate"].map(fmt_pct)
    return table[["Cohort", "Index admissions", "Patients", "30-day readmissions", "Event rate"]]


def delta_table(deltas: pd.DataFrame) -> pd.DataFrame:
    table = deltas.copy()
    table["Cohort"] = table["cohort"].map(
        {
            "diabetes": "Diabetes",
            "ckd": "Chronic kidney disease",
            "heart_failure": "Heart failure",
            "hypertension": "Hypertension",
        }
    )
    for col in ["admission_AUROC", "discharge_AUROC", "delta_AUROC", "admission_AUPRC", "discharge_AUPRC", "delta_AUPRC"]:
        table[col] = table[col].map(lambda value: f"{value:+.3f}" if col.startswith("delta") else fmt_metric(value))
    return table[
        [
            "Cohort",
            "admission_AUROC",
            "discharge_AUROC",
            "delta_AUROC",
            "admission_AUPRC",
            "discharge_AUPRC",
            "delta_AUPRC",
        ]
    ].rename(
        columns={
            "admission_AUROC": "Admission AUROC",
            "discharge_AUROC": "Discharge AUROC",
            "delta_AUROC": "Delta AUROC",
            "admission_AUPRC": "Admission AUPRC",
            "discharge_AUPRC": "Discharge AUPRC",
            "delta_AUPRC": "Delta AUPRC",
        }
    )


def best_model_table(models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = {
        "diabetes": "Diabetes",
        "ckd": "Chronic kidney disease",
        "heart_failure": "Heart failure",
        "hypertension": "Hypertension",
    }
    model_labels = {
        "calibrated_gradient_boosting_platt": "Calibrated gradient boosting",
        "calibrated_gradient_boosting_isotonic": "Calibrated gradient boosting (isotonic)",
        "gradient_boosting_sklearn_hist": "Gradient boosting",
        "calibrated_random_forest_platt": "Calibrated random forest",
        "calibrated_random_forest_isotonic": "Calibrated random forest (isotonic)",
        "random_forest_sklearn": "Random forest",
        "logistic_regression": "Logistic regression",
    }
    feature_labels = {
        "discharge_lab_minimal": "Discharge labs",
        "discharge_safe_minimal": "Discharge-safe",
        "discharge_lab_vital_proc_genmed_minimal": "Discharge labs + vitals + procedures + meds",
        "discharge_safe_vital_proc_genmed_minimal": "Discharge-safe + vitals + procedures + meds",
    }
    for cohort, group in models.groupby("cohort", sort=False):
        best = group.sort_values(["AUPRC", "AUROC"], ascending=False).iloc[0]
        rows.append(
            {
                "Cohort": labels.get(cohort, cohort),
                "Best model by AUPRC": model_labels.get(best["model"], best["model"]),
                "Feature set": feature_labels.get(best["feature_set"], best["feature_set"]),
                "N": fmt_int(best["n"]),
                "Events": fmt_int(best["events"]),
                "AUROC": fmt_metric(best["AUROC"]),
                "AUPRC": fmt_metric(best["AUPRC"]),
                "Brier": fmt_metric(best["Brier_score"]),
            }
        )
    return pd.DataFrame(rows)


def threshold_summary(threshold: pd.DataFrame) -> pd.DataFrame:
    subset = threshold[threshold["alert_rate"].round(2).eq(0.10)].copy()
    if subset.empty:
        return pd.DataFrame()
    subset = subset.sort_values(["cohort", "prediction_time", "ppv"], ascending=[True, True, False])
    rows = []
    labels = {
        "diabetes": "Diabetes",
        "ckd": "Chronic kidney disease",
        "heart_failure": "Heart failure",
        "hypertension": "Hypertension",
    }
    for cohort, group in subset.groupby("cohort", sort=False):
        best = group.iloc[0]
        rows.append(
            {
                "Cohort": labels.get(cohort, cohort),
                "Prediction time": best["prediction_time"],
                "Alert burden": fmt_pct(best["alert_rate"]),
                "PPV": fmt_pct(best["ppv"]),
                "Recall": fmt_pct(best["recall"]),
                "Lift vs event rate": fmt_metric(best["lift_vs_event_rate"]),
            }
        )
    return pd.DataFrame(rows)


def decision_curve_summary(decision: pd.DataFrame) -> pd.DataFrame:
    subset = decision[decision["threshold_probability"].round(2).eq(0.20)].copy()
    if subset.empty:
        return pd.DataFrame()
    labels = {
        "diabetes": "Diabetes",
        "ckd": "Chronic kidney disease",
        "heart_failure": "Heart failure",
        "hypertension": "Hypertension",
    }
    rows = []
    for cohort, group in subset.groupby("cohort", sort=False):
        best = group.sort_values("model_net_benefit", ascending=False).iloc[0]
        rows.append(
            {
                "Cohort": labels.get(cohort, cohort),
                "Prediction time": best["prediction_time"],
                "Model net benefit": fmt_metric(best["model_net_benefit"]),
                "Treat-all net benefit": fmt_metric(best["treat_all_net_benefit"]),
                "Preferred strategy": best["preferred_strategy"],
            }
        )
    return pd.DataFrame(rows)


def leakage_summary(ed_los: pd.DataFrame) -> tuple[float, float, int]:
    return float(ed_los["delta_AUROC"].abs().mean()), float(ed_los["delta_AUPRC"].abs().mean()), int(len(ed_los))


def generate(project_root: Path, template: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    cohort = read_csv(project_root, "outputs/tables/chronic_disease_benchmark_cohort_summary.csv")
    deltas = read_csv(project_root, "outputs/tables/chronic_disease_prediction_time_deltas.csv")
    models = read_csv(project_root, "outputs/tables/chronic_disease_model_baseline_comparison.csv")
    ed_los = read_csv(project_root, "outputs/tables/chronic_disease_ed_los_sensitivity_comparison.csv")
    threshold = read_csv(project_root, "outputs/tables/chronic_disease_threshold_analysis.csv")
    decision = read_csv(project_root, "outputs/tables/chronic_disease_decision_curve.csv")

    total_admissions = int(cohort["final_index_admissions"].sum())
    total_subjects = int(cohort["final_subjects"].sum())
    min_event = float(cohort["readmission_30d_rate"].min())
    max_event = float(cohort["readmission_30d_rate"].max())
    mean_delta_auroc = float(deltas["delta_AUROC"].mean())
    mean_delta_auprc = float(deltas["delta_AUPRC"].mean())
    ed_auroc, ed_auprc, ed_rows = leakage_summary(ed_los)

    key_results = pd.DataFrame(
        [
            {"metric": "total_index_admissions", "value": total_admissions},
            {"metric": "total_subjects", "value": total_subjects},
            {"metric": "min_30d_readmission_rate", "value": min_event},
            {"metric": "max_30d_readmission_rate", "value": max_event},
            {"metric": "mean_discharge_minus_admission_AUROC", "value": mean_delta_auroc},
            {"metric": "mean_discharge_minus_admission_AUPRC", "value": mean_delta_auprc},
            {"metric": "mean_abs_ED_LOS_delta_AUROC", "value": ed_auroc},
            {"metric": "mean_abs_ED_LOS_delta_AUPRC", "value": ed_auprc},
        ]
    )

    abstract = template["abstract"]
    methods = template["methods"]
    limitations = "\n".join(f"- {item}" for item in template.get("limitations", []))
    text = f"""# {template["title"]}

{template["subtitle"]}

**Boundary.** {template["boundary_statement"]}

## Structured Abstract

**Background.** {abstract["background"]}

**Objective.** {abstract["objective"]}

**Methods.** {abstract["methods"]}

**Results.** The four MIMIC-IV cohorts included {fmt_int(total_admissions)} index admissions from {fmt_int(total_subjects)} patients. The 30-day readmission rate ranged from {fmt_pct(min_event)} to {fmt_pct(max_event)}. Across cohorts, discharge-time models improved over admission-time models by a mean AUROC difference of {mean_delta_auroc:+.3f} and a mean AUPRC difference of {mean_delta_auprc:+.3f}. ED length-of-stay sensitivity checks changed AUROC by a mean absolute difference of {ed_auroc:.4f} across {ed_rows} comparisons.

**Conclusion.** {abstract["conclusion"]}

## Methods Brief

### Data Source

{methods["data_source"]}

### Cohort Design

{methods["cohort_design"]}

### Prediction Times And Feature Windows

{methods["prediction_times"]}

### Leakage Audit

{methods["leakage_audit"]}

### Models And Metrics

{methods["models"]}

## Results Brief

### Cohort Summary

{markdown_table(cohort_table(cohort))}

### Prediction-Time Effect

Discharge-time models consistently improved over admission-time models, but the gain was modest rather than dramatic. This supports the core ChronoEHR-Agent argument: prediction time changes both feature availability and apparent model performance.

{markdown_table(delta_table(deltas))}

### Traditional Baseline Models

The table below reports the best model within each cohort according to test-set AUPRC among the completed traditional baselines. These models are research baselines, not clinical deployment candidates.

{markdown_table(best_model_table(models))}

### Fixed Alert-Burden Summary

At an alert burden of approximately 10%, PPV and recall differed by cohort. This is included as a practical model-behavior summary rather than a clinical alert recommendation.

{markdown_table(threshold_summary(threshold))}

### Decision-Curve Summary

Decision-curve summaries are reported to avoid relying only on AUROC. Net benefit depends on the threshold probability and should be interpreted as a research evaluation measure.

{markdown_table(decision_curve_summary(decision))}

### Leakage And Sensitivity Interpretation

ChronoEHR-Agent explicitly tracks variables whose availability depends on time. The ED length-of-stay sensitivity analysis showed small average absolute changes in AUROC ({ed_auroc:.4f}) and AUPRC ({ed_auprc:.4f}), suggesting that the documented high-risk process variable did not drive the main benchmark results. Separately, deliberate leakage demonstrations using future readmission-derived variables remain excluded from valid models.

## Limitations

{limitations}

## Next Writing Tasks

- Review the English Markdown and DOCX draft wording before promoting it into the main manuscript exporter.
- Add journal-specific presets only after the Methods and Results wording is stable.
- Keep eICU and CHARLS as data-pending extensions until local readiness changes to READY.
"""
    return text, key_results


def main() -> None:
    args = parse_args()
    template = read_json(args.template if args.template.is_absolute() else args.project_root / args.template)
    text, key_results = generate(args.project_root, template)
    report_path = args.project_root / "outputs" / "reports" / "chronic_disease_methods_results_english_brief.md"
    table_path = args.project_root / "outputs" / "tables" / "english_brief_key_results.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    key_results.to_csv(table_path, index=False)
    print(f"Wrote {report_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
