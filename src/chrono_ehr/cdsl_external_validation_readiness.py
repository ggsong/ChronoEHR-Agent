#!/usr/bin/env python3
"""Audit CDSL readiness as an external validation or benchmark dataset."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


CDSL_CANDIDATE_ROOTS = [
    Path(os.environ.get("CDSL_ROOT", "~/cdsl")).expanduser(),
    Path("~/datasets/cdsl").expanduser(),
    DEFAULT_PROJECT / "data" / "raw" / "cdsl",
]

RAW_REQUIREMENTS = {
    "COVID_DSL_01.CSV": "demographics/admission/discharge/outcome",
    "COVID_DSL_02.CSV": "vital signs",
    "COVID_DSL_04.CSV": "medication source in original notebook; not required by current formatted table",
    "COVID_DSL_06_v2.CSV": "laboratory tests",
}

FORMATTED_REQUIRED_COLUMNS = {
    "PatientID": "patient or admission identifier",
    "RecordTime": "feature measurement time",
    "AdmissionTime": "hospital admission time",
    "DischargeTime": "hospital discharge time",
    "Outcome": "in-hospital mortality label in CDSL preprocessing",
    "LOS": "length-of-stay label",
    "Sex": "demographic feature",
    "Age": "demographic feature",
}

FOLD_REQUIRED_FILES = [
    "train_x.pkl",
    "train_y.pkl",
    "train_pid.pkl",
    "val_x.pkl",
    "val_y.pkl",
    "val_pid.pkl",
    "test_x.pkl",
    "test_y.pkl",
    "test_pid.pkl",
    "los_info.pkl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--cdsl-root",
        type=Path,
        help="Optional explicit CDSL root. Defaults to the first usable known local CDSL path.",
    )
    return parser.parse_args()


def choose_cdsl_root(explicit: Path | None) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates = [explicit] if explicit else CDSL_CANDIDATE_ROOTS
    rows = []
    selected: Path | None = None
    for root in candidates:
        if root is None:
            continue
        formatted = root / "processed" / "cdsl_dataset_formatted.csv"
        raw_dir = root / "raw" / "19_04_2021"
        fold0 = root / "processed" / "fold_0"
        score = sum([formatted.exists(), raw_dir.exists(), fold0.exists()])
        rows.append(
            {
                "candidate_root": str(root),
                "exists": root.exists(),
                "has_formatted_csv": formatted.exists(),
                "has_raw_dir": raw_dir.exists(),
                "has_fold_0": fold0.exists(),
                "selection_score": score,
            }
        )
        if selected is None and score >= 2:
            selected = root
    return selected, rows


def safe_csv_header(path: Path) -> list[str]:
    try:
        return pd.read_csv(path, nrows=0, low_memory=False).columns.tolist()
    except Exception:
        for sep in ["|", ";", ","]:
            try:
                return pd.read_csv(path, nrows=0, sep=sep, encoding="ISO-8859-1", low_memory=False).columns.tolist()
            except Exception:
                continue
    return []


def count_rows_fast(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() not in {".csv"}:
        return None
    try:
        with path.open("rb") as handle:
            lines = sum(1 for _ in handle)
        return max(lines - 1, 0)
    except Exception:
        return None


def inventory_files(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        row: dict[str, Any] = {
            "relative_path": str(relative),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
            "rows": None,
            "columns": None,
            "column_preview": "",
        }
        if path.suffix.lower() == ".csv":
            columns = safe_csv_header(path)
            row["rows"] = count_rows_fast(path)
            row["columns"] = len(columns) if columns else None
            row["column_preview"] = ", ".join(columns[:12])
        rows.append(row)
    return pd.DataFrame(rows)


def load_pickle_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "object_type": "", "length": None, "shape": "", "first_shape": ""}
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - report should continue after one bad file
        return {"exists": True, "object_type": "unreadable", "length": None, "shape": "", "first_shape": str(exc)}

    shape = getattr(obj, "shape", "")
    length = len(obj) if hasattr(obj, "__len__") else None
    first_shape = ""
    if isinstance(obj, list) and obj:
        first_shape = str(getattr(obj[0], "shape", type(obj[0]).__name__))
    elif isinstance(obj, dict):
        first_shape = ", ".join(sorted(obj.keys()))
    return {
        "exists": True,
        "object_type": type(obj).__name__,
        "length": length,
        "shape": str(shape) if shape != "" else "",
        "first_shape": first_shape,
    }


def audit_folds(root: Path) -> pd.DataFrame:
    rows = []
    processed = root / "processed"
    for fold_idx in range(12):
        fold = processed / f"fold_{fold_idx}"
        missing = [name for name in FOLD_REQUIRED_FILES if not (fold / name).exists()]
        train_pid = load_pickle_summary(fold / "train_pid.pkl")
        val_pid = load_pickle_summary(fold / "val_pid.pkl")
        test_pid = load_pickle_summary(fold / "test_pid.pkl")
        train_x = load_pickle_summary(fold / "train_x.pkl")
        train_y = load_pickle_summary(fold / "train_y.pkl")
        los_info = load_pickle_summary(fold / "los_info.pkl")

        overlap_status = "not_checked"
        try:
            with (fold / "train_pid.pkl").open("rb") as handle:
                train_ids = set(pickle.load(handle))
            with (fold / "val_pid.pkl").open("rb") as handle:
                val_ids = set(pickle.load(handle))
            with (fold / "test_pid.pkl").open("rb") as handle:
                test_ids = set(pickle.load(handle))
            overlaps = {
                "train_val": len(train_ids & val_ids),
                "train_test": len(train_ids & test_ids),
                "val_test": len(val_ids & test_ids),
            }
            overlap_status = "PASS" if all(value == 0 for value in overlaps.values()) else json.dumps(overlaps)
        except Exception:
            pass

        rows.append(
            {
                "fold": fold_idx,
                "fold_path": str(fold),
                "status": "PASS" if fold.exists() and not missing else "FAIL",
                "missing_files": "; ".join(missing),
                "train_n": train_pid["length"],
                "val_n": val_pid["length"],
                "test_n": test_pid["length"],
                "train_x_object": train_x["object_type"],
                "train_x_first_shape": train_x["first_shape"],
                "train_y_first_shape": train_y["first_shape"],
                "los_info_keys": los_info["first_shape"],
                "patient_overlap_status": overlap_status,
            }
        )
    return pd.DataFrame(rows)


def audit_schema(root: Path) -> pd.DataFrame:
    formatted = root / "processed" / "cdsl_dataset_formatted.csv"
    columns = safe_csv_header(formatted) if formatted.exists() else []
    rows = []
    for column, meaning in FORMATTED_REQUIRED_COLUMNS.items():
        rows.append(
            {
                "field": column,
                "meaning": meaning,
                "present": column in columns,
                "chrono_ehr_role": {
                    "PatientID": "patient_id / encounter_id",
                    "RecordTime": "feature_time",
                    "AdmissionTime": "admission_time",
                    "DischargeTime": "discharge_time",
                    "Outcome": "outcome",
                    "LOS": "secondary_outcome_or_forbidden_feature",
                    "Sex": "baseline_feature",
                    "Age": "baseline_feature",
                }[column],
                "leakage_note": leakage_note(column),
            }
        )

    for name, purpose in RAW_REQUIREMENTS.items():
        rows.append(
            {
                "field": name,
                "meaning": purpose,
                "present": (root / "raw" / "19_04_2021" / name).exists(),
                "chrono_ehr_role": "raw_source_file",
                "leakage_note": "原始数据源；进入模型前仍需按 prediction time 截断。",
            }
        )
    return pd.DataFrame(rows)


def leakage_note(column: str) -> str:
    if column in {"Outcome", "LOS", "DischargeTime"}:
        return "作为死亡/LOS 预测任务时不能把该字段当作普通特征；应作为标签、随访定义或时间锚点。"
    if column == "RecordTime":
        return "每条记录必须满足 RecordTime <= prediction_time。"
    if column == "AdmissionTime":
        return "可作为时间锚点；不要让 admission 后未来测量混入 admission-time 预测。"
    return "通常可作为候选特征，但仍需按具体 prediction time 审计。"


def summarize_formatted(root: Path) -> dict[str, Any]:
    formatted = root / "processed" / "cdsl_dataset_formatted.csv"
    if not formatted.exists():
        return {}
    df = pd.read_csv(formatted, low_memory=False)
    feature_columns = [
        col
        for col in df.columns
        if col not in {"PatientID", "RecordTime", "AdmissionTime", "DischargeTime", "Outcome", "LOS"}
    ]
    summary: dict[str, Any] = {
        "formatted_rows": len(df),
        "formatted_columns": len(df.columns),
        "patients": int(df["PatientID"].nunique()) if "PatientID" in df else None,
        "outcome_rate": float(df.groupby("PatientID")["Outcome"].max().mean()) if {"PatientID", "Outcome"}.issubset(df.columns) else None,
        "feature_columns": len(feature_columns),
    }
    if "RecordTime" in df and "AdmissionTime" in df:
        record_time = pd.to_datetime(df["RecordTime"], errors="coerce")
        admission_time = pd.to_datetime(df["AdmissionTime"], errors="coerce")
        hours_from_admission = (record_time - admission_time).dt.total_seconds() / 3600
        summary.update(
            {
                "median_record_hour_from_admission": float(hours_from_admission.median()),
                "max_record_hour_from_admission": float(hours_from_admission.max()),
                "records_before_admission": int((hours_from_admission < 0).sum()),
            }
        )
    return summary


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    table = df.head(max_rows).copy()
    columns = table.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in table.itertuples(index=False):
        values = [str(value).replace("|", "/").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only the first {max_rows} of {len(df)} rows are shown._")
    return "\n".join(lines)


def write_report(
    project_root: Path,
    selected_root: Path | None,
    candidate_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    schema_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    formatted_summary: dict[str, Any],
) -> Path:
    reports = project_root / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "cdsl_external_validation_readiness_report.md"

    raw_ready = bool(selected_root) and all((selected_root / "raw" / "19_04_2021" / name).exists() for name in RAW_REQUIREMENTS)
    formatted_ready = bool(selected_root) and (selected_root / "processed" / "cdsl_dataset_formatted.csv").exists()
    folds_ready = not fold_df.empty and fold_df.loc[fold_df["fold"].lt(10), "status"].eq("PASS").all()
    schema_ready = not schema_df.empty and schema_df[schema_df["chrono_ehr_role"].ne("raw_source_file")]["present"].all()

    method_status = "READY" if raw_ready and formatted_ready and folds_ready and schema_ready else "NOT_READY"
    chronic_status = "NOT_DIRECTLY_COMPARABLE"

    lines = [
        "# CDSL External Validation Readiness Report",
        "",
        f"- Selected CDSL root: `{selected_root}`" if selected_root else "- Selected CDSL root: `none`",
        f"- Method benchmark readiness: `{method_status}`",
        f"- Direct chronic readmission external validation: `{chronic_status}`",
        "",
        "## 结论",
        "",
        "CDSL 当前可以作为 ChronoEHR-Agent 的 EHR 时间序列 prediction-time benchmark 补充数据：它有入院时间、记录时间、出院时间、死亡结局、LOS、生命体征和化验特征。"
        "但它不是 MIMIC 糖尿病 30 天再入院模型的直接外部验证集，因为 CDSL 是 COVID 住院数据，主要标签是住院死亡和 LOS，不是慢病再入院。",
        "",
        "适合先做的外部验证形式是：用 CDSL 测试 Agent 的时间点审计、prediction-time 截断、标签/特征分离和传统 baseline 报告生成能力。"
        "不建议把它包装成糖尿病再入院外部验证。",
        "",
        "## Formatted Data Summary",
        "",
    ]
    if formatted_summary:
        for key, value in formatted_summary.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- No formatted CDSL table was found.")

    lines.extend(
        [
            "",
            "## Candidate Paths",
            "",
            markdown_table(candidate_df),
            "",
            "## Schema Readiness",
            "",
            markdown_table(schema_df),
            "",
            "## Fold Audit",
            "",
            markdown_table(fold_df),
            "",
            "## 下一步建议",
            "",
            "1. 先把 CDSL 定位为 external benchmark，不作为糖尿病再入院 direct validation。",
            "2. 新增 CDSL temporal task adapter：从 `cdsl_dataset_formatted.csv` 生成 admission-time、24h、discharge-before-outcome 三套特征。",
            "3. 只允许 `RecordTime <= prediction_time` 的特征进入模型，把 `Outcome`、`LOS`、`DischargeTime` 标成标签/时间锚点/禁用特征。",
            "4. 用 logistic regression、Random Forest、HistGradientBoosting 跑 CDSL 死亡预测 baseline，并生成独立 leakage audit report。",
            "",
            "## File Inventory Preview",
            "",
            markdown_table(inventory_df[["relative_path", "suffix", "size_mb", "rows", "columns", "column_preview"]], max_rows=60),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    selected_root, candidate_rows = choose_cdsl_root(args.cdsl_root)

    tables = args.project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.DataFrame(candidate_rows)
    if selected_root is None:
        inventory_df = pd.DataFrame()
        schema_df = pd.DataFrame()
        fold_df = pd.DataFrame()
        formatted_summary: dict[str, Any] = {}
    else:
        inventory_df = inventory_files(selected_root)
        schema_df = audit_schema(selected_root)
        fold_df = audit_folds(selected_root)
        formatted_summary = summarize_formatted(selected_root)

    candidate_df.to_csv(tables / "cdsl_external_validation_candidate_paths.csv", index=False)
    inventory_df.to_csv(tables / "cdsl_external_validation_file_inventory.csv", index=False)
    schema_df.to_csv(tables / "cdsl_external_validation_schema_candidates.csv", index=False)
    fold_df.to_csv(tables / "cdsl_external_validation_fold_audit.csv", index=False)
    pd.DataFrame([formatted_summary]).to_csv(tables / "cdsl_external_validation_summary.csv", index=False)

    report_path = write_report(
        args.project_root,
        selected_root,
        candidate_df,
        inventory_df,
        schema_df,
        fold_df,
        formatted_summary,
    )
    print(f"Wrote {report_path}")
    if formatted_summary:
        print(pd.DataFrame([formatted_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
