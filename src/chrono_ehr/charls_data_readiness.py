#!/usr/bin/env python3
"""Audit local CHARLS data readiness for longitudinal chronic-disease tasks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


CHARLS_CANDIDATE_ROOTS = [
    Path(os.environ.get("CHARLS_ROOT", "~/CHARLS")).expanduser(),
    Path("~/charls").expanduser(),
    DEFAULT_PROJECT / "data" / "raw" / "charls",
]

WAVES = [
    {"wave": "2008_pilot", "year": 2008, "patterns": ["2008", "pilot"]},
    {"wave": "2011_wave1", "year": 2011, "patterns": ["2011", "wave1", "wave_1", "w1"]},
    {"wave": "2012_pilot_wave2", "year": 2012, "patterns": ["2012", "pilot", "wave2", "wave_2"]},
    {"wave": "2013_wave2", "year": 2013, "patterns": ["2013", "wave2", "wave_2", "w2"]},
    {"wave": "2014_life_history", "year": 2014, "patterns": ["2014", "life", "history"]},
    {"wave": "2015_wave3", "year": 2015, "patterns": ["2015", "wave3", "wave_3", "w3"]},
    {"wave": "2018_wave4", "year": 2018, "patterns": ["2018", "wave4", "wave_4", "w4"]},
    {"wave": "2020_wave5", "year": 2020, "patterns": ["2020", "wave5", "wave_5", "w5"]},
    {"wave": "harmonized_charls", "year": None, "patterns": ["harmonized", "harmonised"]},
]

COMMON_SUFFIXES = {".dta", ".sav", ".csv", ".xlsx", ".xls", ".zip", ".rar", ".7z"}

SCHEMA_EXPECTATIONS = [
    {
        "chrono_concept": "person_id",
        "candidate_fields": "ID; householdID; communityID; individual id fields vary by release",
        "needed_for": "cross-wave linkage",
        "leakage_note": "The same person must be linked across waves before defining incident outcomes.",
    },
    {
        "chrono_concept": "baseline_wave",
        "candidate_fields": "2011 Wave1 variables or Harmonized CHARLS baseline fields",
        "needed_for": "prediction time",
        "leakage_note": "Only baseline-wave variables should enter baseline prediction models.",
    },
    {
        "chrono_concept": "future_wave_outcome",
        "candidate_fields": "2013/2015 disease status, biomarker, ADL/IADL, death/follow-up fields",
        "needed_for": "incident chronic disease or functional decline labels",
        "leakage_note": "Future-wave outcome fields must not enter baseline features.",
    },
    {
        "chrono_concept": "diabetes_status",
        "candidate_fields": "self-reported doctor diagnosis; glucose/HbA1c if available; medication",
        "needed_for": "incident diabetes task",
        "leakage_note": "Exclude baseline diabetes before predicting incident diabetes.",
    },
    {
        "chrono_concept": "hypertension_status",
        "candidate_fields": "self-reported diagnosis; blood pressure measurements; medication",
        "needed_for": "incident hypertension task",
        "leakage_note": "Outcome wave blood pressure cannot be used as baseline feature.",
    },
    {
        "chrono_concept": "attrition",
        "candidate_fields": "death; lost to follow-up; missing wave indicators",
        "needed_for": "sensitivity analysis",
        "leakage_note": "Loss to follow-up can bias longitudinal prediction if ignored.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--charls-root", type=Path, help="Optional explicit CHARLS root.")
    return parser.parse_args()


def choose_root(explicit: Path | None) -> tuple[Path | None, pd.DataFrame]:
    candidates = [explicit] if explicit else CHARLS_CANDIDATE_ROOTS
    rows = []
    selected: Path | None = None
    for root in candidates:
        if root is None:
            continue
        files = list_candidate_files(root)
        wave_hits = detect_waves(files)
        score = len(wave_hits[wave_hits["candidate_files"].ne("")])
        rows.append(
            {
                "candidate_root": str(root),
                "exists": root.exists(),
                "candidate_files": len(files),
                "waves_detected": score,
                "selection_score": score,
            }
        )
        if selected is None and score >= 2:
            selected = root
    return selected, pd.DataFrame(rows)


def list_candidate_files(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix.lower() in COMMON_SUFFIXES or any(token in path.name.lower() for token in ["charls", "wave", "harmon"])):
            files.append(path)
    return sorted(files)


def detect_waves(files: list[Path]) -> pd.DataFrame:
    rows = []
    lowered = [(path, str(path).lower()) for path in files]
    for wave in WAVES:
        matches = []
        for path, text in lowered:
            if all(pattern in text for pattern in wave["patterns"][:2]) or any(pattern in text for pattern in wave["patterns"]):
                matches.append(path)
        rows.append(
            {
                "wave": wave["wave"],
                "year": wave["year"],
                "detected": bool(matches),
                "candidate_files": "; ".join(str(path) for path in matches[:8]),
                "n_candidate_files": len(matches),
            }
        )
    return pd.DataFrame(rows)


def inventory(root: Path | None, files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files[:400]:
        rows.append(
            {
                "relative_path": str(path.relative_to(root)) if root is not None and root.exists() else str(path),
                "suffix": path.suffix.lower(),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                "likely_role": infer_role(path.name.lower()),
            }
        )
    return pd.DataFrame(rows, columns=["relative_path", "suffix", "size_mb", "likely_role"])


def infer_role(name: str) -> str:
    if "harmon" in name:
        return "harmonized cross-wave file"
    if "tracker" in name or "tracking" in name:
        return "tracking / follow-up status"
    if "biomarker" in name or "blood" in name:
        return "biomarker / lab-like measurements"
    if "health" in name:
        return "health status and chronic disease variables"
    if "demo" in name or "demographic" in name:
        return "demographics"
    if "weight" in name:
        return "survey weights"
    return "unknown / inspect data dictionary"


def proposed_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_id": "charls_incident_diabetes_2011_to_2013_2015",
                "baseline": "2011 Wave1",
                "outcome_window": "2013 Wave2 or 2015 Wave3",
                "cohort_rule": "Exclude baseline diabetes; require follow-up disease status or biomarker definition.",
                "leakage_risk": "Future wave diagnosis, biomarkers, medication, or follow-up status accidentally used as baseline features.",
                "priority": 1,
            },
            {
                "task_id": "charls_incident_hypertension_2011_to_2013_2015",
                "baseline": "2011 Wave1",
                "outcome_window": "2013 Wave2 or 2015 Wave3",
                "cohort_rule": "Exclude baseline hypertension; define outcome from diagnosis/BP/medication.",
                "leakage_risk": "Outcome-wave blood pressure or medication status leaks into baseline features.",
                "priority": 2,
            },
            {
                "task_id": "charls_multimorbidity_progression",
                "baseline": "2011 or 2013",
                "outcome_window": "next wave or multi-wave follow-up",
                "cohort_rule": "Baseline chronic disease count below threshold; outcome is progression to multimorbidity.",
                "leakage_risk": "Future chronic disease count used as feature.",
                "priority": 3,
            },
        ]
    )


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
    wave_map: pd.DataFrame,
    schema: pd.DataFrame,
    tasks: pd.DataFrame,
    files: pd.DataFrame,
) -> Path:
    reports = project_root / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "charls_data_readiness_report.md"
    required = {"2011_wave1", "2013_wave2"}
    detected = set(wave_map[wave_map["detected"].eq(True)]["wave"])
    status = "READY_FOR_PROTOCOL_DRAFT" if required.issubset(detected) else "NOT_READY"
    text = f"""# CHARLS Data Readiness Report

- Selected CHARLS root: `{selected_root}`
- Status for first longitudinal demo: `{status}`
- Recommended first task: 2011 baseline prediction of incident diabetes by 2013/2015.

## 结论

CHARLS 是中国慢病纵向随访数据，不是 EHR。它适合验证 ChronoEHR-Agent 的 wave-based prediction time、follow-up window 和未来 wave leakage audit 思路。

如果当前状态是 `NOT_READY`，通常只是说明 CHARLS 数据仍在申请或尚未下载到本机固定路径。建议下载到：

`${CHARLS_ROOT}`

## Candidate Roots

{markdown_table(candidates)}

## Wave Detection

{markdown_table(wave_map)}

## ChronoEHR Longitudinal Schema Mapping Draft

{markdown_table(schema)}

## Proposed First Tasks

{markdown_table(tasks)}

## File Inventory Preview

{markdown_table(files.head(80))}
"""
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    selected, candidates = choose_root(args.charls_root)
    files = list_candidate_files(selected)
    wave_map = detect_waves(files)
    schema = pd.DataFrame(SCHEMA_EXPECTATIONS)
    tasks = proposed_tasks()
    file_inventory = inventory(selected, files)

    out = args.project_root / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / "charls_data_readiness_candidate_paths.csv", index=False)
    wave_map.to_csv(out / "charls_wave_detection.csv", index=False)
    schema.to_csv(out / "charls_schema_mapping_draft.csv", index=False)
    tasks.to_csv(out / "charls_proposed_tasks.csv", index=False)
    file_inventory.to_csv(out / "charls_data_readiness_file_inventory.csv", index=False)
    report = write_report(args.project_root, selected, candidates, wave_map, schema, tasks, file_inventory)
    print(f"Wrote {report}")
    required = {"2011_wave1", "2013_wave2"}
    detected = set(wave_map[wave_map["detected"].eq(True)]["wave"])
    print(f"CHARLS readiness: {'READY_FOR_PROTOCOL_DRAFT' if required.issubset(detected) else 'NOT_READY'}")


if __name__ == "__main__":
    main()
