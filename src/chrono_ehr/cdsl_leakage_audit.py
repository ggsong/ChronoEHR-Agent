#!/usr/bin/env python3
"""Audit leakage risks in the CDSL temporal benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from cdsl_external_validation_readiness import CDSL_CANDIDATE_ROOTS, choose_cdsl_root
from mimic_diabetes_baseline import DEFAULT_PROJECT


ID_COL = "PatientID"
TIME_COL = "RecordTime"
ADMIT_COL = "AdmissionTime"
DISCHARGE_COL = "DischargeTime"
OUTCOME_COL = "Outcome"
LOS_COL = "LOS"

LABEL_OR_TIME_ANCHOR_COLUMNS = {OUTCOME_COL, LOS_COL, DISCHARGE_COL}
NON_PREDICTOR_COLUMNS = {ID_COL, "outcome"}
FEATURE_DIR = Path("data/processed/cdsl_temporal_benchmark")
WINDOW_SPECS = {
    "admission_demographics": {
        "prediction_time": "admission",
        "allowed_window": "baseline demographics only",
        "severity_if_misused": "critical",
    },
    "first_24h_vitals_labs": {
        "prediction_time": "admission + 24h",
        "allowed_window": "AdmissionTime <= RecordTime <= AdmissionTime + 24h",
        "severity_if_misused": "critical",
    },
    "first_48h_vitals_labs": {
        "prediction_time": "admission + 48h",
        "allowed_window": "AdmissionTime <= RecordTime <= AdmissionTime + 48h",
        "severity_if_misused": "critical",
    },
    "full_stay_naive_reference": {
        "prediction_time": "before discharge",
        "allowed_window": "RecordTime <= DischargeTime",
        "severity_if_misused": "warning",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--cdsl-root", type=Path, help="Optional explicit CDSL root.")
    return parser.parse_args()


def load_formatted(root: Path) -> pd.DataFrame:
    path = root / "processed" / "cdsl_dataset_formatted.csv"
    df = pd.read_csv(path, low_memory=False)
    for col in [TIME_COL, ADMIT_COL, DISCHARGE_COL]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def row(
    check: str,
    status: str,
    severity: str,
    evidence: str,
    beginner_interpretation: str,
    recommended_action: str,
) -> dict[str, str]:
    return {
        "check": check,
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "beginner_interpretation": beginner_interpretation,
        "recommended_action": recommended_action,
    }


def audit_raw_time_windows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    records_before_admission = int((df[TIME_COL].notna() & df[ADMIT_COL].notna() & (df[TIME_COL] < df[ADMIT_COL])).sum())
    records_after_discharge = int((df[TIME_COL].notna() & df[DISCHARGE_COL].notna() & (df[TIME_COL] > df[DISCHARGE_COL])).sum())
    total_records = len(df)

    rows.append(
        row(
            "raw_records_before_admission",
            "WARNING" if records_before_admission else "PASS",
            "warning",
            f"{records_before_admission} / {total_records} records have RecordTime < AdmissionTime.",
            "少量记录时间早于入院时间，可能来自急诊、登记时间差或预处理边界。它们不一定是错误，但不能静默进入入院后窗口。",
            "在 CDSL benchmark 中保留统计说明；入院后 24h/48h 模型继续要求 RecordTime >= AdmissionTime。",
        )
    )
    rows.append(
        row(
            "raw_records_after_discharge",
            "WARNING" if records_after_discharge else "PASS",
            "warning",
            f"{records_after_discharge} / {total_records} records have RecordTime > DischargeTime.",
            "出院后的记录如果进入模型，会变成典型未来信息。",
            "full-stay reference 也只能允许 RecordTime <= DischargeTime；出院后记录应排除或单独解释。",
        )
    )
    return rows


def audit_feature_file(path: Path, feature_set: str) -> list[dict[str, str]]:
    if not path.exists():
        return [
            row(
                f"{feature_set}_feature_file_exists",
                "FAIL",
                "critical",
                f"Missing file: {path}",
                "模型特征文件不存在，无法判断这个 prediction time 是否真的可复现。",
                "先运行 python3 src/chrono_ehr/run_study.py --cdsl-temporal-benchmark。",
            )
        ]
    df = pd.read_csv(path, nrows=0)
    predictor_cols = [col for col in df.columns if col not in NON_PREDICTOR_COLUMNS]
    leaked = sorted(set(predictor_cols) & LABEL_OR_TIME_ANCHOR_COLUMNS)
    raw_time_cols = sorted(set(predictor_cols) & {TIME_COL, ADMIT_COL})
    status = "PASS" if not leaked and not raw_time_cols else "FAIL"
    rows = [
        row(
            f"{feature_set}_forbidden_predictor_columns",
            status,
            WINDOW_SPECS[feature_set]["severity_if_misused"],
            f"Forbidden predictors found: {', '.join(leaked + raw_time_cols) if leaked or raw_time_cols else 'none'}; predictors={len(predictor_cols)}.",
            "`Outcome`、`LOS`、`DischargeTime` 或原始时间戳如果被当作普通特征，模型会学到标签或未来信息。",
            "保持这些字段只作为标签、时间锚点或审计字段；不要放入 X 矩阵。",
        )
    ]
    if feature_set == "full_stay_naive_reference":
        rows.append(
            row(
                "full_stay_naive_reference_interpretation",
                "WARNING",
                "warning",
                "This feature set uses all in-stay records before discharge.",
                "它可以作为 naive reference，但不能当作入院时或 24 小时模型。性能高并不等于早期预测能力强。",
                "报告中必须明确标注为全住院窗口参考模型，不与入院时模型混淆。",
            )
        )
    return rows


def audit_window_report(project_root: Path) -> list[dict[str, str]]:
    path = project_root / "outputs" / "tables" / "cdsl_temporal_benchmark_window_audit.csv"
    if not path.exists():
        return [
            row(
                "cdsl_window_audit_table_exists",
                "FAIL",
                "critical",
                f"Missing file: {path}",
                "没有窗口审计表，就不能证明模型只用了指定时间点之前的信息。",
                "先运行 CDSL temporal benchmark。",
            )
        ]
    audit = pd.read_csv(path)
    rows = []
    for _, item in audit.iterrows():
        feature_set = str(item["feature_set"])
        if feature_set not in WINDOW_SPECS:
            continue
        rows.append(
            row(
                f"{feature_set}_window_counts_documented",
                "PASS",
                "info",
                (
                    f"eligible_records={item.get('eligible_records')}; "
                    f"excluded_records_outside_window={item.get('excluded_records_outside_window')}; "
                    f"records_before_admission={item.get('records_before_admission')}"
                ),
                "窗口内记录数和窗口外排除记录数已经被保存，可用于 Methods/Results 或补充材料说明。",
                "保留该表；后续可把它并入 supplementary appendix。",
            )
        )
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "check",
        "status",
        "severity",
        "evidence",
        "beginner_interpretation",
        "recommended_action",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        values = [str(value).replace("|", "/").replace("\n", " ") for value in item]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, cdsl_root: Path | None, summary: pd.DataFrame) -> Path:
    reports = project_root / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "cdsl_leakage_audit_report.md"
    critical_failures = summary[(summary["status"].eq("FAIL")) & (summary["severity"].eq("critical"))]
    warnings = summary[summary["status"].eq("WARNING")]
    overall = "PASS" if critical_failures.empty else "FAIL"
    text = f"""# CDSL Leakage Audit Report

- CDSL root: `{cdsl_root}`
- Overall status: `{overall}`
- Critical failures: {len(critical_failures)}
- Warnings: {len(warnings)}

## 结论

这个审计检查 CDSL temporal benchmark 是否把标签、LOS、出院时间或预测时间之后的信息误当作模型特征。它是研究工具审计，不是临床诊疗建议。

当前最重要的解释是：`full_stay_naive_reference` 是全住院窗口参考模型，性能高不能被解释成入院时预测能力；144 条 `RecordTime < AdmissionTime` 的边界记录需要在后续报告中说明。

## Audit Table

{markdown_table(summary)}
"""
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    cdsl_root, _ = choose_cdsl_root(args.cdsl_root)
    if cdsl_root is None:
        raise SystemExit(f"No usable CDSL root found. Checked: {', '.join(str(path) for path in CDSL_CANDIDATE_ROOTS)}")

    rows: list[dict[str, str]] = []
    df = load_formatted(cdsl_root)
    rows.extend(audit_raw_time_windows(df))
    rows.extend(audit_window_report(args.project_root))
    for feature_set in WINDOW_SPECS:
        rows.extend(audit_feature_file(args.project_root / FEATURE_DIR / f"{feature_set}.csv", feature_set))

    summary = pd.DataFrame(rows)
    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "cdsl_leakage_audit.csv", index=False)
    report = write_report(args.project_root, cdsl_root, summary)
    print(f"Wrote {report}")
    print(summary["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
