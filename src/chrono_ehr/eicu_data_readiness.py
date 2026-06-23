#!/usr/bin/env python3
"""Audit local eICU data readiness for ChronoEHR-Agent."""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


EICU_CANDIDATE_ROOTS = [
    Path(os.environ.get("EICU_ROOT", "~/eicu-2.0")).expanduser(),
    Path("~/eicu-collaborative-research-database-2.0").expanduser(),
    Path("~/eicu-crd").expanduser(),
    DEFAULT_PROJECT / "data" / "raw" / "eicu",
]

EXPECTED_TABLES = {
    "patient": {
        "filenames": ["patient.csv", "patient.csv.gz"],
        "role": "patient/stay identifiers, ICU and hospital outcomes, admission/discharge offsets",
        "required": True,
    },
    "lab": {
        "filenames": ["lab.csv", "lab.csv.gz"],
        "role": "time-stamped laboratory measurements",
        "required": True,
    },
    "vitalPeriodic": {
        "filenames": ["vitalPeriodic.csv", "vitalPeriodic.csv.gz", "vitalperiodic.csv", "vitalperiodic.csv.gz"],
        "role": "regular vital-sign time series",
        "required": True,
    },
    "apachePatientResult": {
        "filenames": ["apachePatientResult.csv", "apachePatientResult.csv.gz", "apachepatientresult.csv", "apachepatientresult.csv.gz"],
        "role": "severity scores and mortality prediction fields",
        "required": False,
    },
    "diagnosis": {
        "filenames": ["diagnosis.csv", "diagnosis.csv.gz"],
        "role": "diagnosis table for cohort and baseline severity",
        "required": False,
    },
    "medication": {
        "filenames": ["medication.csv", "medication.csv.gz"],
        "role": "medication orders with offsets",
        "required": False,
    },
    "treatment": {
        "filenames": ["treatment.csv", "treatment.csv.gz"],
        "role": "treatment categories and procedures",
        "required": False,
    },
    "infusionDrug": {
        "filenames": ["infusionDrug.csv", "infusionDrug.csv.gz", "infusiondrug.csv", "infusiondrug.csv.gz"],
        "role": "infusion drug events",
        "required": False,
    },
    "nurseCharting": {
        "filenames": ["nurseCharting.csv", "nurseCharting.csv.gz", "nursecharting.csv", "nursecharting.csv.gz"],
        "role": "nursing charted observations",
        "required": False,
    },
}

SCHEMA_EXPECTATIONS = [
    {
        "chrono_ehr_concept": "patient_id",
        "candidate_eicu_fields": "patientunitstayid; uniquepid",
        "needed_for": "patient/stay split and identifier mapping",
        "leakage_note": "Use stay-level and patient-level IDs to prevent the same patient crossing splits.",
    },
    {
        "chrono_ehr_concept": "prediction_time",
        "candidate_eicu_fields": "hospitaladmitoffset; unitadmitoffset; observation offsets",
        "needed_for": "ICU admission and first-24h prediction windows",
        "leakage_note": "Only events with offset <= prediction offset should enter early models.",
    },
    {
        "chrono_ehr_concept": "discharge_time",
        "candidate_eicu_fields": "unitdischargeoffset; hospitaldischargeoffset",
        "needed_for": "full-stay reference and LOS labels",
        "leakage_note": "Discharge offsets are anchors/labels, not admission-time features.",
    },
    {
        "chrono_ehr_concept": "outcome",
        "candidate_eicu_fields": "unitdischargestatus; hospitaldischargestatus",
        "needed_for": "ICU or hospital mortality prediction",
        "leakage_note": "Outcome status must never be included in feature matrices.",
    },
    {
        "chrono_ehr_concept": "labs",
        "candidate_eicu_fields": "labresultoffset; labname; labresult",
        "needed_for": "first-24h lab features",
        "leakage_note": "Drop or flag lab rows after the prediction window.",
    },
    {
        "chrono_ehr_concept": "vitals",
        "candidate_eicu_fields": "observationoffset; heartrate; respiration; sao2; temperature; systemicsystolic",
        "needed_for": "first-24h vital features",
        "leakage_note": "Use offset windows, not full-stay summaries, for early prediction.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--eicu-root", type=Path, help="Optional explicit eICU root.")
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    try:
        compression = "gzip" if path.suffix == ".gz" else None
        return pd.read_csv(path, nrows=0, compression=compression, low_memory=False).columns.tolist()
    except Exception:
        return []


def count_rows_sample(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return None


def choose_root(explicit: Path | None) -> tuple[Path | None, pd.DataFrame]:
    candidates = [explicit] if explicit else EICU_CANDIDATE_ROOTS
    rows = []
    selected: Path | None = None
    for root in candidates:
        if root is None:
            continue
        found_required = 0
        found_any = 0
        for spec in EXPECTED_TABLES.values():
            located = find_first(root, spec["filenames"])
            found_any += int(located is not None)
            found_required += int(spec["required"] and located is not None)
        score = found_required * 10 + found_any
        rows.append(
            {
                "candidate_root": str(root),
                "exists": root.exists(),
                "found_required_tables": found_required,
                "required_tables": sum(int(spec["required"]) for spec in EXPECTED_TABLES.values()),
                "found_any_expected_tables": found_any,
                "selection_score": score,
            }
        )
        if selected is None and found_required >= 2:
            selected = root
    return selected, pd.DataFrame(rows)


def find_first(root: Path, filenames: list[str]) -> Path | None:
    if not root.exists():
        return None
    for name in filenames:
        direct = root / name
        if direct.exists():
            return direct
    lowered = {name.lower() for name in filenames}
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in lowered:
            return path
    return None


def table_readiness(root: Path | None) -> pd.DataFrame:
    rows = []
    for table, spec in EXPECTED_TABLES.items():
        path = find_first(root, spec["filenames"]) if root is not None else None
        columns = read_header(path) if path is not None else []
        rows.append(
            {
                "table": table,
                "required_for_first_demo": spec["required"],
                "present": path is not None,
                "path": str(path) if path is not None else "",
                "role": spec["role"],
                "columns": len(columns) if columns else None,
                "column_preview": ", ".join(columns[:16]),
                "rows": count_rows_sample(path) if path is not None and path.stat().st_size < 2_000_000_000 else None,
            }
        )
    return pd.DataFrame(rows)


def inventory(root: Path | None, max_files: int = 300) -> pd.DataFrame:
    if root is None or not root.exists():
        return pd.DataFrame(columns=["relative_path", "suffix", "size_mb", "columns", "column_preview"])
    rows = []
    for path in sorted(root.rglob("*"))[:max_files]:
        if not path.is_file():
            continue
        columns = read_header(path) if path.name.lower().endswith((".csv", ".csv.gz")) else []
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "suffix": "".join(path.suffixes),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                "columns": len(columns) if columns else None,
                "column_preview": ", ".join(columns[:12]),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    project_root: Path,
    selected_root: Path | None,
    candidates: pd.DataFrame,
    tables: pd.DataFrame,
    schema: pd.DataFrame,
    files: pd.DataFrame,
) -> Path:
    reports = project_root / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "eicu_data_readiness_report.md"
    required = tables[tables["required_for_first_demo"].eq(True)]
    ready = not required.empty and required["present"].all()
    status = "READY" if ready else "NOT_READY"
    text = f"""# eICU Data Readiness Report

- Selected eICU root: `{selected_root}`
- Status for first demo: `{status}`
- Recommended first task: ICU first-24h hospital mortality prediction.

## 结论

eICU 适合作为多中心 ICU 外部 EHR benchmark。它不应被包装成 MIMIC 慢病 30 天再入院模型的直接外部验证。

如果当前状态是 `NOT_READY`，通常只是说明完整 eICU CSV 尚未下载到本机固定路径。建议下载到：

`${EICU_ROOT}`

## Candidate Roots

{markdown_table(candidates)}

## Expected Table Readiness

{markdown_table(tables)}

## ChronoEHR Schema Mapping Draft

{markdown_table(schema)}

## First Demo Proposal

1. Cohort: adult ICU stays with valid hospital mortality label.
2. Prediction times: ICU admission, first 24h, full-stay reference.
3. Features: demographics, admission severity fields, first-24h labs/vitals/treatments.
4. Outcomes: hospital mortality first; prolonged ICU LOS second.
5. Audit: outcome labels, discharge offsets, and events after prediction time must be excluded from early models.

## File Inventory Preview

{markdown_table(files.head(80))}
"""
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    selected, candidates = choose_root(args.eicu_root)
    tables = table_readiness(selected)
    schema = pd.DataFrame(SCHEMA_EXPECTATIONS)
    files = inventory(selected)

    out = args.project_root / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / "eicu_data_readiness_candidate_paths.csv", index=False)
    tables.to_csv(out / "eicu_data_readiness_expected_tables.csv", index=False)
    schema.to_csv(out / "eicu_schema_mapping_draft.csv", index=False)
    files.to_csv(out / "eicu_data_readiness_file_inventory.csv", index=False)
    report = write_report(args.project_root, selected, candidates, tables, schema, files)
    print(f"Wrote {report}")
    status = "READY" if not tables.empty and tables[tables["required_for_first_demo"].eq(True)]["present"].all() else "NOT_READY"
    print(f"eICU readiness: {status}")


if __name__ == "__main__":
    main()
