#!/usr/bin/env python3
"""Audit local SHARE data readiness for longitudinal ageing-survey tasks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


SHARE_CANDIDATE_ROOTS = [
    Path(os.environ.get("SHARE_ROOT", "~/SHARE")).expanduser(),
    Path("~/share").expanduser(),
    DEFAULT_PROJECT / "data" / "raw" / "share",
]

COMMON_SUFFIXES = {".dta", ".sav", ".csv", ".xlsx", ".xls", ".zip", ".rar", ".7z", ".pdf", ".do"}

WAVES = [
    {"wave": "wave1", "label": "Wave 1 baseline", "participation_column": "inw1"},
    {"wave": "wave2", "label": "Wave 2 follow-up", "participation_column": "inw2"},
    {"wave": "wave3_life_history", "label": "Wave 3 life history", "participation_column": "inw3lh"},
    {"wave": "wave4", "label": "Wave 4 follow-up", "participation_column": "inw4"},
    {"wave": "wave5", "label": "Wave 5 follow-up", "participation_column": "inw5"},
    {"wave": "wave6", "label": "Wave 6 follow-up", "participation_column": "inw6"},
    {"wave": "wave7", "label": "Wave 7 follow-up", "participation_column": "inw7"},
    {"wave": "wave8", "label": "Wave 8 follow-up", "participation_column": "inw8"},
]

SCHEMA_EXPECTATIONS = [
    ("person_id", "mergeid", "cross-wave linkage", "allowed"),
    ("household_id", "hhid", "household linkage", "allowed"),
    ("country", "country", "country-level stratification", "allowed"),
    ("baseline_participation", "inw1", "baseline cohort entry", "allowed"),
    ("followup_participation", "inw2/inw4", "follow-up availability", "do_not_use_as_baseline_feature"),
    ("baseline_diabetes", "r1diabe/r1rxdiab", "baseline exclusion", "label_boundary_only"),
    ("followup_diabetes", "r2diabe/r2rxdiab/r4diabe/r4rxdiab", "incident diabetes outcome", "forbidden_as_baseline_feature"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--share-root", type=Path, help="Optional explicit SHARE root.")
    return parser.parse_args()


def list_candidate_files(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix.lower() in COMMON_SUFFIXES or "share" in path.name.lower()):
            files.append(path)
    return sorted(files)


def find_harmonized_file(root: Path | None) -> Path | None:
    if root is None or not root.exists():
        return None
    candidates = []
    for path in root.rglob("*.dta"):
        name = path.name.lower()
        text = str(path).lower()
        if name.startswith("h_share") or "harmonized_share" in text or "harmonised_share" in text:
            candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def stata_metadata(path: Path | None) -> tuple[list[str], dict[str, str]]:
    if path is None or not path.exists():
        return [], {}
    reader = pd.io.stata.StataReader(path)
    labels = reader.variable_labels()
    return list(labels), labels


def detect_waves(files: list[Path], columns: list[str]) -> pd.DataFrame:
    lower_paths = [(path, str(path).lower()) for path in files]
    rows = []
    for wave in WAVES:
        participation_column = wave["participation_column"]
        path_hits = [
            path
            for path, text in lower_paths
            if wave["wave"].replace("_life_history", "") in text
            or participation_column in text
            or f"sharew{participation_column[-1]}" in text
        ]
        rows.append(
            {
                "wave": wave["wave"],
                "label": wave["label"],
                "participation_column": participation_column,
                "column_present": participation_column in columns,
                "candidate_files": "; ".join(str(path) for path in path_hits[:8]),
                "n_candidate_files": len(path_hits),
            }
        )
    return pd.DataFrame(rows)


def infer_role(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("h_share") or "harmonized" in str(path).lower() or "harmonised" in str(path).lower():
        return "harmonized cross-wave file"
    if "condition" in name or "use" in name or "statement" in name:
        return "data-use documentation"
    if "leaflet" in name or "pdf" in name:
        return "documentation"
    if "do" == path.suffix.lower().lstrip("."):
        return "Stata conversion/helper script"
    return "raw/support file"


def inventory(root: Path | None, files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files[:400]:
        rows.append(
            {
                "relative_path": str(path.relative_to(root)) if root is not None and root.exists() else str(path),
                "suffix": path.suffix.lower(),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                "likely_role": infer_role(path),
            }
        )
    return pd.DataFrame(rows, columns=["relative_path", "suffix", "size_mb", "likely_role"])


def schema_expectations(columns: list[str], labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for concept, variable_text, needed_for, leakage_status in SCHEMA_EXPECTATIONS:
        variables = variable_text.split("/")
        present = [variable for variable in variables if variable in columns]
        rows.append(
            {
                "chrono_concept": concept,
                "candidate_fields": variable_text,
                "present_fields": "; ".join(present),
                "n_present": len(present),
                "needed_for": needed_for,
                "leakage_status": leakage_status,
                "label_preview": "; ".join(labels.get(variable, "") for variable in present[:3]),
            }
        )
    return pd.DataFrame(rows)


def proposed_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_id": "share_incident_diabetes_wave1_to_wave2_wave4",
                "baseline": "wave 1",
                "outcome_window": "wave 2 or wave 4",
                "cohort_rule": "Exclude baseline diabetes; require follow-up diabetes status.",
                "leakage_risk": "Future-wave diabetes or medication variables enter baseline features.",
                "priority": 1,
            },
            {
                "task_id": "share_multimorbidity_progression",
                "baseline": "wave 1 or later baseline wave",
                "outcome_window": "next observed wave",
                "cohort_rule": "Baseline multimorbidity below threshold; outcome is progression.",
                "leakage_risk": "Future chronic disease count used as a baseline feature.",
                "priority": 2,
            },
            {
                "task_id": "share_function_decline",
                "baseline": "wave 1",
                "outcome_window": "wave 2 or wave 4",
                "cohort_rule": "Baseline functional status measured; outcome is new ADL/IADL limitation.",
                "leakage_risk": "Future ADL/IADL or attrition variables used as baseline features.",
                "priority": 3,
            },
        ]
    )


def choose_root(explicit: Path | None) -> tuple[Path | None, pd.DataFrame]:
    candidates = [explicit] if explicit else SHARE_CANDIDATE_ROOTS
    rows: list[dict[str, Any]] = []
    selected: Path | None = None
    for root in candidates:
        if root is None:
            continue
        files = list_candidate_files(root)
        harmonized = find_harmonized_file(root)
        columns, _ = stata_metadata(harmonized)
        wave_hits = detect_waves(files, columns)
        score = int(wave_hits["column_present"].sum()) + (2 if harmonized else 0)
        rows.append(
            {
                "candidate_root": str(root),
                "exists": root.exists(),
                "candidate_files": len(files),
                "harmonized_file": str(harmonized) if harmonized else "",
                "waves_detected": int(wave_hits["column_present"].sum()),
                "selection_score": score,
            }
        )
        if selected is None and score >= 4:
            selected = root
    return selected, pd.DataFrame(rows)


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
    harmonized: Path | None,
    candidates: pd.DataFrame,
    wave_map: pd.DataFrame,
    schema: pd.DataFrame,
    tasks: pd.DataFrame,
    files: pd.DataFrame,
) -> Path:
    reports = project_root / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "share_data_readiness_report.md"
    required = {"inw1", "inw2"}
    present = set(wave_map.loc[wave_map["column_present"].eq(True), "participation_column"])
    status = "READY_FOR_COHORT_SKELETON" if harmonized and required.issubset(present) else "NOT_READY"
    text = f"""# SHARE Data Readiness Report

- Selected SHARE root: `{selected_root}`
- Harmonized SHARE file: `{harmonized}`
- Status for first longitudinal demo: `{status}`
- Recommended first task: wave 1 baseline prediction of incident diabetes by wave 2 or wave 4.

## Boundary

SHARE is a longitudinal ageing survey rather than an EHR database. This readiness report tests whether ChronoEHR-Agent can connect a new longitudinal database, identify wave-based prediction times, and separate baseline features from follow-up outcomes. It does not train a model and does not provide clinical advice.

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
    selected_root, candidates = choose_root(args.share_root)
    if selected_root is None:
        selected_root = args.share_root or next((root for root in SHARE_CANDIDATE_ROOTS if root.exists()), None)
    files = list_candidate_files(selected_root)
    harmonized = find_harmonized_file(selected_root)
    columns, labels = stata_metadata(harmonized)
    waves = detect_waves(files, columns)
    schema = schema_expectations(columns, labels)
    tasks = proposed_tasks()
    inv = inventory(selected_root, files)

    table_dir = args.project_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(table_dir / "share_candidate_roots.csv", index=False)
    waves.to_csv(table_dir / "share_wave_detection.csv", index=False)
    schema.to_csv(table_dir / "share_schema_expectations.csv", index=False)
    tasks.to_csv(table_dir / "share_task_plan.csv", index=False)
    inv.to_csv(table_dir / "share_file_inventory.csv", index=False)
    report = write_report(args.project_root, selected_root, harmonized, candidates, waves, schema, tasks, inv)
    print(f"SHARE candidate roots: {len(candidates)}")
    print(f"Selected SHARE root: {selected_root}")
    print(f"Harmonized SHARE file: {harmonized}")
    print(f"Detected wave columns: {int(waves['column_present'].sum())}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
