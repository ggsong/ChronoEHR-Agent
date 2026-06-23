#!/usr/bin/env python3
"""Generate a prioritized backlog for migrating hard-coded runner logic to config."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]

AREA_PRIORITY = {
    "cohort_code_rules": ("P1", 1, "High reuse value across CKD, heart failure, hypertension, and future ICD cohorts."),
    "feature_sets": ("P1", 2, "Directly affects time-aware model interpretation and cross-cohort comparability."),
    "outcome_window": ("P2", 3, "Important for leakage boundaries; current diabetes runner still owns parts of the logic."),
    "inclusion_exclusion": ("P2", 4, "Useful for reproducibility, but less urgent while v0.1 outputs are stable."),
    "dataset_tables": ("P3", 5, "Mostly maintainability; migrate after cohort logic is stable."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def target_module(study_id: str, area: str) -> str:
    if area == "cohort_code_rules":
        if study_id == "mimic_iv_3_1_diabetes_readmission":
            return "src/chrono_ehr/mimic_diabetes_cohort.py"
        return "src/chrono_ehr/mimic_diagnosis_cohort_builder.py"
    if area == "feature_sets":
        return "src/chrono_ehr/prediction_time_model_tools.py and configs/prediction_time_model_specs.json"
    if area in {"outcome_window", "inclusion_exclusion"}:
        return "src/chrono_ehr/mimic_diabetes_cohort.py and src/chrono_ehr/mimic_diagnosis_cohort_builder.py"
    if area == "dataset_tables":
        return "cohort readers and study YAML dataset.tables"
    return "src/chrono_ehr"


def acceptance_criteria(study_id: str, area: str) -> str:
    if area == "cohort_code_rules":
        return "ICD prefixes are loaded from study config and cohort counts match existing outputs within exact equality."
    if area == "feature_sets":
        return "Prediction-time model feature lists are loaded from config/specs and validation still reports zero issues."
    if area == "outcome_window":
        return "30-day follow-up window is loaded from config and leakage gate still reports zero critical issues."
    if area == "inclusion_exclusion":
        return "Adult, valid-time, in-hospital death, and post-discharge death rules are traceable to config notes or fields."
    if area == "dataset_tables":
        return "Raw table paths come from dataset.tables and diabetes one-click demo remains PASS."
    return "Agent self-check and delivery readiness remain PASS."


def build_backlog(project_root: Path) -> pd.DataFrame:
    columns = [
        "priority",
        "rank",
        "study_id",
        "cohort",
        "area",
        "target_module",
        "config_path",
        "rationale",
        "recommended_action",
        "acceptance_criteria",
    ]
    coverage = read_csv(project_root / "outputs" / "tables" / "config_coverage_audit.csv")
    if coverage.empty:
        return pd.DataFrame(columns=columns)
    gaps = coverage[coverage["status"].astype(str).eq("CONFIGURED_BUT_RUNNER_PARTLY_HARDCODED")].copy()
    rows = []
    for item in gaps.to_dict(orient="records"):
        area = str(item.get("area", ""))
        priority, rank, rationale = AREA_PRIORITY.get(area, ("P3", 99, "Maintainability migration."))
        study_id = str(item.get("study_id", ""))
        rows.append(
            {
                "priority": priority,
                "rank": rank,
                "study_id": study_id,
                "cohort": item.get("cohort", ""),
                "area": area,
                "target_module": target_module(study_id, area),
                "config_path": item.get("config_path", ""),
                "rationale": rationale,
                "recommended_action": item.get("action", ""),
                "acceptance_criteria": acceptance_criteria(study_id, area),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["rank", "study_id", "area"]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["priority", "study_id", "area", "target_module", "acceptance_criteria"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, backlog: pd.DataFrame) -> Path:
    table_path = project_root / "outputs" / "tables" / "config_migration_backlog.csv"
    report_path = project_root / "outputs" / "reports" / "config_migration_backlog.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    backlog.to_csv(table_path, index=False)
    p1 = int((backlog["priority"] == "P1").sum()) if not backlog.empty else 0
    suggested_order = (
        "No migration items remain for the checked mainline study definitions."
        if backlog.empty
        else """1. Migrate ICD cohort code rules into config-driven loading.
2. Migrate model feature-set selection into shared specs/configs.
3. Then migrate diabetes outcome/inclusion/table-path details."""
    )
    report_path.write_text(
        f"""# Config Migration Backlog

- Backlog items: {len(backlog)}
- P1 items: {p1}
- Boundary: engineering maintainability backlog only; no medical QA, diagnosis, or treatment recommendation.

## Prioritized Items

{markdown_table(backlog) if not backlog.empty else "No migration items found."}

## Suggested Order

{suggested_order}
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    backlog = build_backlog(args.project_root)
    report = write_outputs(args.project_root, backlog)
    print(f"Wrote {report}")
    print(f"Config migration backlog items: {len(backlog)}")
    if not backlog.empty:
        print(backlog[["priority", "study_id", "area", "target_module"]].to_string(index=False))


if __name__ == "__main__":
    main()
