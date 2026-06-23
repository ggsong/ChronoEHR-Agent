#!/usr/bin/env python3
"""Audit consistency across external benchmark summary tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


NUMERIC_FIELDS = [
    "n",
    "events",
    "event_rate",
    "AUROC",
    "AUROC_lower",
    "AUROC_upper",
    "AUPRC",
    "AUPRC_lower",
    "AUPRC_upper",
    "Brier",
    "Brier_lower",
    "Brier_upper",
    "mean_absolute_calibration_error",
]

KEY_FIELDS = ["dataset", "feature_set", "model", "calibration_method"]
CI_PATTERN = re.compile(r"^\s*([0-9.]+)\s*\(([0-9.]+)-([0-9.]+)\)\s*$")


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


def canonical_method(method: object, dataset: object) -> str:
    text = "" if pd.isna(method) else str(method)
    dataset_text = "" if pd.isna(dataset) else str(dataset)
    if text == "":
        return "raw_traditional" if dataset_text == "CDSL" else "raw"
    if text == "raw" and dataset_text in {"eICU", "CHARLS"}:
        return "raw_model_comparison"
    return text


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "feature_set" not in normalized and "feature_window" in normalized:
        normalized["feature_set"] = normalized["feature_window"]
    for col in KEY_FIELDS:
        if col not in normalized:
            normalized[col] = ""
    normalized["calibration_method"] = [
        canonical_method(method, dataset)
        for method, dataset in zip(normalized["calibration_method"], normalized["dataset"])
    ]
    normalized["_key"] = normalized[KEY_FIELDS].fillna("").astype(str).agg("||".join, axis=1)
    return normalized


def row(
    check_group: str,
    benchmark_row: str,
    compared_artifacts: str,
    field: str,
    source_value: object,
    target_value: object,
    tolerance: float,
    detail: str = "",
) -> dict[str, object]:
    source_num = pd.to_numeric(pd.Series([source_value]), errors="coerce").iloc[0]
    target_num = pd.to_numeric(pd.Series([target_value]), errors="coerce").iloc[0]
    if pd.isna(source_num) and pd.isna(target_num):
        delta = 0.0
        status = "PASS"
    elif pd.isna(source_num) or pd.isna(target_num):
        delta = float("nan")
        status = "FAIL"
    else:
        delta = abs(float(source_num) - float(target_num))
        status = "PASS" if delta <= tolerance else "FAIL"
    return {
        "check_group": check_group,
        "benchmark_row": benchmark_row,
        "compared_artifacts": compared_artifacts,
        "field": field,
        "source_value": source_value,
        "target_value": target_value,
        "abs_delta": delta,
        "tolerance": tolerance,
        "status": status,
        "detail": detail,
    }


def status_row(check_group: str, field: str, status: str, detail: str) -> dict[str, object]:
    return {
        "check_group": check_group,
        "benchmark_row": "",
        "compared_artifacts": "",
        "field": field,
        "source_value": "",
        "target_value": "",
        "abs_delta": "",
        "tolerance": "",
        "status": status,
        "detail": detail,
    }


def parse_ci(text: object) -> tuple[float, float, float]:
    if pd.isna(text):
        return (float("nan"), float("nan"), float("nan"))
    match = CI_PATTERN.match(str(text))
    if not match:
        return (float("nan"), float("nan"), float("nan"))
    return tuple(float(item) for item in match.groups())  # type: ignore[return-value]


def compare_by_key(
    summary: pd.DataFrame,
    target: pd.DataFrame,
    target_name: str,
    fields: Iterable[str],
    tolerance: float,
    check_group: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_by_key = {str(item["_key"]): item for _, item in target.iterrows()}
    for _, summary_row in summary.iterrows():
        benchmark_row = str(summary_row["benchmark_row"])
        key = str(summary_row["_key"])
        target_row = target_by_key.get(key)
        if target_row is None:
            rows.append(status_row(check_group, benchmark_row, "FAIL", f"missing key in {target_name}: {key}"))
            continue
        for field in fields:
            if field in summary_row and field in target_row:
                rows.append(
                    row(
                        check_group,
                        benchmark_row,
                        f"external_benchmark_summary_table.csv vs {target_name}",
                        field,
                        summary_row[field],
                        target_row[field],
                        tolerance,
                    )
                )
    return rows


def compare_technical(summary: pd.DataFrame, technical: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    technical_by_label = {str(item["benchmark_row"]): item for _, item in technical.iterrows()}
    ci_map = {
        "AUROC": "auroc_ci",
        "AUPRC": "auprc_ci",
        "Brier": "brier_ci",
    }
    for _, summary_row in summary.iterrows():
        label = str(summary_row["benchmark_row"])
        technical_row = technical_by_label.get(label)
        if technical_row is None:
            rows.append(status_row("summary_vs_technical", label, "FAIL", "missing technical summary row"))
            continue
        for field in ["n", "events", "event_rate", "mean_absolute_calibration_error"]:
            rows.append(
                row(
                    "summary_vs_technical",
                    label,
                    "external_benchmark_summary_table.csv vs external_technical_summary_table.csv",
                    field,
                    summary_row[field],
                    technical_row[field],
                    1e-9,
                )
            )
        for metric, display_col in ci_map.items():
            value, lower, upper = parse_ci(technical_row[display_col])
            for source_col, parsed_value in [
                (metric, value),
                (f"{metric}_lower", lower),
                (f"{metric}_upper", upper),
            ]:
                rows.append(
                    row(
                        "summary_vs_technical",
                        label,
                        "external_benchmark_summary_table.csv vs external_technical_summary_table.csv",
                        source_col,
                        summary_row[source_col],
                        parsed_value,
                        5e-4,
                        "technical summary rounds displayed CI values to four decimals",
                    )
                )
    return rows


def compare_selection_rationale(summary: pd.DataFrame, rationale: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rationale_by_label = {str(item["benchmark_row"]): item for _, item in rationale.iterrows()}
    field_map = {
        "feature_set": "selected_feature_set",
        "model": "selected_model",
        "calibration_method": "selected_calibration_method",
        "AUROC": "selected_AUROC",
        "AUPRC": "selected_AUPRC",
        "Brier": "selected_Brier",
        "mean_absolute_calibration_error": "selected_mean_absolute_calibration_error",
    }
    for _, summary_row in summary.iterrows():
        label = str(summary_row["benchmark_row"])
        rationale_row = rationale_by_label.get(label)
        if rationale_row is None:
            rows.append(status_row("summary_vs_selection_rationale", label, "FAIL", "missing model-selection rationale row"))
            continue
        for source_col, target_col in field_map.items():
            if source_col in KEY_FIELDS:
                status = "PASS" if str(summary_row[source_col]) == str(rationale_row[target_col]) else "FAIL"
                rows.append(
                    status_row(
                        "summary_vs_selection_rationale",
                        source_col,
                        status,
                        f"{label}: summary={summary_row[source_col]}; rationale={rationale_row[target_col]}",
                    )
                )
            else:
                rows.append(
                    row(
                        "summary_vs_selection_rationale",
                        label,
                        "external_benchmark_summary_table.csv vs external_model_selection_rationale.csv",
                        source_col,
                        summary_row[source_col],
                        rationale_row[target_col],
                        1e-9,
                    )
                )
    return rows


def compare_calibration_decision(summary: pd.DataFrame, technical: pd.DataFrame, calibration: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected = calibration[calibration["is_selected_technical_summary_row"].astype(bool)].copy()
    selected_by_label = {str(item["benchmark_row"]): item for _, item in selected.iterrows()}
    technical_by_label = {str(item["benchmark_row"]): item for _, item in technical.iterrows()}
    fields = [
        "mean_absolute_calibration_error",
        "decision_thresholds",
        "decision_model_preferred_thresholds",
        "decision_positive_advantage_thresholds",
        "decision_best_threshold",
        "decision_best_net_benefit_advantage",
    ]
    for _, summary_row in summary.iterrows():
        label = str(summary_row["benchmark_row"])
        selected_row = selected_by_label.get(label)
        technical_row = technical_by_label.get(label)
        if selected_row is None or technical_row is None:
            rows.append(status_row("technical_vs_calibration_decision", label, "FAIL", "missing selected calibration/technical row"))
            continue
        for field in fields:
            rows.append(
                row(
                    "technical_vs_calibration_decision",
                    label,
                    "external_technical_summary_table.csv vs external_calibration_decision_summary.csv",
                    field,
                    technical_row[field],
                    selected_row[field],
                    1e-9,
                )
            )
    return rows


def boundary_rows(summary: pd.DataFrame, technical: pd.DataFrame, calibration: pd.DataFrame, rationale: pd.DataFrame) -> list[dict[str, object]]:
    combined = " ".join(
        [
            summary.astype(str).to_string(),
            technical.astype(str).to_string(),
            calibration.astype(str).to_string(),
            rationale.astype(str).to_string(),
        ]
    )
    checks = {
        "cdsl_naive_upper_reference": ["Naive upper", "full-stay"],
        "eicu_not_chronic_readmission": ["not chronic readmission", "ICU mortality"],
        "charls_longitudinal_extension": ["longitudinal", "CHARLS"],
    }
    rows = []
    for check, tokens in checks.items():
        status = "PASS" if all(token.lower() in combined.lower() for token in tokens) else "FAIL"
        rows.append(status_row("boundary_declarations", check, status, "requires centralized boundary wording across external tables"))
    forbidden = ["recommended treatment", "ready for clinical deployment", "clinical action threshold recommendation"]
    decision_status = "PASS" if "decision" in combined.lower() and not any(token in combined.lower() for token in forbidden) else "FAIL"
    rows.append(
        status_row(
            "boundary_declarations",
            "decision_curve_no_clinical_threshold",
            decision_status,
            "decision-curve outputs are present without clinical-threshold recommendation wording",
        )
    )
    return rows


def build_audit(project_root: Path) -> pd.DataFrame:
    tables = project_root / "outputs" / "tables"
    summary = normalize_frame(read_csv(tables / "external_benchmark_summary_table.csv"))
    hard = normalize_frame(read_csv(tables / "external_benchmark_hard_metrics_table.csv"))
    technical = normalize_frame(read_csv(tables / "external_technical_summary_table.csv"))
    calibration = normalize_frame(read_csv(tables / "external_calibration_decision_summary.csv"))
    rationale = read_csv(tables / "external_model_selection_rationale.csv")
    bootstrap = normalize_frame(read_csv(tables / "external_model_bootstrap_ci.csv"))

    if summary.empty:
        raise FileNotFoundError("Missing external_benchmark_summary_table.csv")

    rows: list[dict[str, object]] = []
    rows.extend(compare_by_key(summary, hard, "external_benchmark_hard_metrics_table.csv", NUMERIC_FIELDS, 1e-9, "summary_vs_hard_metrics"))
    rows.extend(compare_technical(summary, technical))
    rows.extend(compare_selection_rationale(summary, rationale))
    rows.extend(compare_by_key(summary, bootstrap, "external_model_bootstrap_ci.csv", [field for field in NUMERIC_FIELDS if field != "mean_absolute_calibration_error"], 1e-9, "summary_vs_bootstrap_ci"))
    rows.extend(compare_calibration_decision(summary, technical, calibration))
    rows.extend(boundary_rows(summary, technical, calibration, rationale))
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    if "abs_delta" in display:
        display["abs_delta"] = display["abs_delta"].map(lambda value: f"{float(value):.6g}" if pd.notna(value) and value != "" else "")
    columns = ["check_group", "benchmark_row", "field", "status", "abs_delta", "tolerance", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, audit: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_metric_consistency_audit.md"
    failures = audit[audit["status"].ne("PASS")]
    groups = audit.groupby("check_group")["status"].apply(lambda series: int(series.ne("PASS").sum())).to_dict()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# External Metric Consistency Audit

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(audit)}
- Failures: {len(failures)}
- Boundary: research model evaluation consistency audit only; it does not define clinical action thresholds.
- Check groups: {", ".join(f"{group} failures={count}" for group, count in sorted(groups.items()))}

## Check Table

{markdown_table(audit)}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    audit = build_audit(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "external_metric_consistency_audit.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(table_path, index=False)
    report = write_report(args.project_root, audit)
    failures = int(audit["status"].ne("PASS").sum())
    print(f"External metric consistency checks: {len(audit)}")
    print(f"Failures: {failures}")
    print(f"Wrote {report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
