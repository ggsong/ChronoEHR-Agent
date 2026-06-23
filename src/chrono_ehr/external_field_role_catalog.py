#!/usr/bin/env python3
"""Build a unified field-role catalog for planned external datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


ROLE_ORDER = {
    "id": 1,
    "time_anchor": 2,
    "prediction_time": 3,
    "outcome": 4,
    "feature_source": 5,
    "forbidden_or_leakage": 6,
    "censoring_or_attrition": 7,
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


def split_fields(value: str) -> list[str]:
    parts = []
    for sep in [";", ","]:
        if sep in value:
            parts = [item.strip() for item in value.split(sep)]
            break
    if not parts:
        parts = [value.strip()]
    return [item for item in parts if item]


def eicu_rows(project_root: Path) -> list[dict[str, str]]:
    schema = read_csv(project_root / "outputs" / "tables" / "eicu_schema_mapping_draft.csv")
    tables = read_csv(project_root / "outputs" / "tables" / "eicu_data_readiness_expected_tables.csv")
    present_by_table = {}
    columns_by_table = {}
    if not tables.empty:
        for item in tables.to_dict(orient="records"):
            table = str(item.get("table", ""))
            present_by_table[table] = str(item.get("present", "False")) == "True" or item.get("present") is True
            columns_by_table[table] = str(item.get("column_preview", ""))
    rows = []
    for item in schema.to_dict(orient="records"):
        concept = str(item.get("chrono_ehr_concept", ""))
        candidates = str(item.get("candidate_eicu_fields", ""))
        role = eicu_role(concept)
        for field in split_fields(candidates):
            source_table = infer_eicu_table(field)
            rows.append(
                {
                    "dataset": "eICU",
                    "study_id": "eicu_temporal_mortality",
                    "chrono_concept": concept,
                    "field_role": role,
                    "source_table_or_wave": source_table,
                    "candidate_field": field,
                    "time_available": eicu_time_available(role, field),
                    "prediction_time_use": eicu_prediction_use(role, field),
                    "leakage_risk": eicu_leakage_risk(role, field),
                    "leakage_policy": str(item.get("leakage_note", "")),
                    "raw_data_status": "present" if present_by_table.get(source_table, False) else "data_pending",
                    "local_column_evidence": columns_by_table.get(source_table, ""),
                    "notes": str(item.get("needed_for", "")),
                }
            )
    return rows


def eicu_role(concept: str) -> str:
    if concept in {"patient_id"}:
        return "id"
    if concept in {"prediction_time"}:
        return "prediction_time"
    if concept in {"discharge_time"}:
        return "time_anchor"
    if concept in {"outcome"}:
        return "outcome"
    if concept in {"labs", "vitals"}:
        return "feature_source"
    return "feature_source"


def infer_eicu_table(field: str) -> str:
    field_lower = field.lower()
    if field_lower in {"patientunitstayid", "uniquepid", "hospitaladmitoffset", "unitadmitoffset", "unitdischargeoffset", "hospitaldischargeoffset", "unitdischargestatus", "hospitaldischargestatus"}:
        return "patient"
    if field_lower in {"labresultoffset", "labname", "labresult"}:
        return "lab"
    if field_lower in {"observationoffset", "heartrate", "respiration", "sao2", "temperature", "systemicsystolic"}:
        return "vitalPeriodic"
    return "unknown"


def eicu_time_available(role: str, field: str) -> str:
    field_lower = field.lower()
    if field_lower in {"unitdischargestatus", "hospitaldischargestatus", "unitdischargeoffset", "hospitaldischargeoffset"}:
        return "after_discharge"
    if role in {"id", "prediction_time"}:
        return "at_or_before_icu_admission"
    if role == "feature_source":
        return "event_offset_dependent"
    return "availability_must_be_verified"


def eicu_prediction_use(role: str, field: str) -> str:
    field_lower = field.lower()
    if role == "outcome" or field_lower.endswith("dischargestatus"):
        return "forbidden_as_feature"
    if "dischargeoffset" in field_lower:
        return "full_stay_reference_only"
    if role == "feature_source":
        return "allowed_only_with_offset_window"
    return "allowed_as_anchor_or_merge_key_not_model_feature"


def eicu_leakage_risk(role: str, field: str) -> str:
    field_lower = field.lower()
    if role == "outcome" or field_lower.endswith("dischargestatus"):
        return "critical"
    if "dischargeoffset" in field_lower:
        return "high"
    if role in {"id", "prediction_time", "time_anchor"}:
        return "medium"
    return "medium"


def charls_rows(project_root: Path) -> list[dict[str, str]]:
    schema = read_csv(project_root / "outputs" / "tables" / "charls_schema_mapping_draft.csv")
    template = read_csv(project_root / "docs" / "charls_wave_map_template.csv")
    waves = read_csv(project_root / "outputs" / "tables" / "charls_wave_detection.csv")
    detected = {}
    if not waves.empty:
        detected = {str(item.get("wave", "")): bool(item.get("detected", False)) for item in waves.to_dict(orient="records")}
    rows = []
    for item in schema.to_dict(orient="records"):
        concept = str(item.get("chrono_concept", ""))
        candidates = str(item.get("candidate_fields", ""))
        role = charls_role(concept)
        for field in split_fields(candidates):
            wave = infer_charls_wave(concept, field)
            rows.append(
                {
                    "dataset": "CHARLS",
                    "study_id": "charls_incident_diabetes",
                    "chrono_concept": concept,
                    "field_role": role,
                    "source_table_or_wave": wave,
                    "candidate_field": field,
                    "time_available": charls_time_available(role, wave),
                    "prediction_time_use": charls_prediction_use(role, wave),
                    "leakage_risk": charls_leakage_risk(role, wave),
                    "leakage_policy": str(item.get("leakage_note", "")),
                    "raw_data_status": "present" if detected.get(wave, False) else "data_pending",
                    "local_column_evidence": "",
                    "notes": str(item.get("needed_for", "")),
                }
            )
    for item in template.to_dict(orient="records"):
        variable = str(item.get("variable_name", ""))
        if not variable or variable == "nan":
            continue
        wave = template_wave(str(item.get("wave", "")))
        role = charls_template_role(str(item.get("variable_group", "")), variable)
        rows.append(
            {
                "dataset": "CHARLS",
                "study_id": "charls_incident_diabetes",
                "chrono_concept": variable,
                "field_role": role,
                "source_table_or_wave": wave,
                "candidate_field": variable,
                "time_available": str(item.get("time_available", "")) if "time_available" in item else charls_time_available(role, wave),
                "prediction_time_use": charls_template_prediction_use(item, role, wave),
                "leakage_risk": str(item.get("leakage_risk", "")) or charls_leakage_risk(role, wave),
                "leakage_policy": str(item.get("reason", "")) or "Use wave-aware feature eligibility before modeling.",
                "raw_data_status": "present" if detected.get(wave, False) else "data_pending",
                "local_column_evidence": "",
                "notes": str(item.get("meaning", "")),
            }
        )
    return rows


def charls_role(concept: str) -> str:
    if concept == "person_id":
        return "id"
    if concept == "baseline_wave":
        return "prediction_time"
    if concept in {"future_wave_outcome", "diabetes_status", "hypertension_status"}:
        return "outcome"
    if concept == "attrition":
        return "censoring_or_attrition"
    return "feature_source"


def infer_charls_wave(concept: str, field: str) -> str:
    text = f"{concept} {field}".lower()
    if "2015" in text:
        return "2015_wave3"
    if "2013" in text:
        return "2013_wave2"
    if "2011" in text or "baseline" in text:
        return "2011_wave1"
    if "harmonized" in text:
        return "harmonized_charls"
    return "all_waves"


def template_wave(value: str) -> str:
    text = value.lower()
    if "2015" in text:
        return "2015_wave3"
    if "2013" in text:
        return "2013_wave2"
    if "2011" in text:
        return "2011_wave1"
    if "followup" in text:
        return "all_waves"
    if "all" in text:
        return "all_waves"
    return text or "to_be_confirmed"


def charls_template_role(variable_group: str, variable: str) -> str:
    group = variable_group.lower()
    name = variable.lower()
    if group == "id":
        return "id"
    if group in {"outcome", "outcome_or_future_biomarker"} or name.endswith("_2013") or name.endswith("_2015"):
        return "outcome"
    if group in {"censoring"}:
        return "censoring_or_attrition"
    if group in {"cohort_exclusion"}:
        return "forbidden_or_leakage"
    return "feature_source"


def charls_template_prediction_use(item: dict, role: str, wave: str) -> str:
    if role == "id":
        return "merge_split_only_not_model_feature"
    if role == "outcome":
        return "forbidden_as_baseline_feature"
    if role == "censoring_or_attrition":
        return "sensitivity_or_censoring_only"
    if role == "forbidden_or_leakage":
        return "cohort_definition_only_not_ordinary_feature"
    usable = str(item.get("usable_for_2011_baseline_prediction", "")).lower()
    if usable in {"true", "conditional"} and wave == "2011_wave1":
        return "allowed_if_baseline_wave"
    return "verify_before_modeling"


def charls_time_available(role: str, wave: str) -> str:
    if role in {"outcome", "censoring_or_attrition"} and wave != "2011_wave1":
        return "future_wave"
    if role in {"id", "prediction_time"}:
        return "at_baseline_or_linkage"
    return "baseline_wave_if_confirmed"


def charls_prediction_use(role: str, wave: str) -> str:
    if role == "outcome" and wave != "2011_wave1":
        return "forbidden_as_baseline_feature"
    if role == "censoring_or_attrition":
        return "sensitivity_or_censoring_only"
    if role == "id":
        return "merge_split_only_not_model_feature"
    return "allowed_if_baseline_wave"


def charls_leakage_risk(role: str, wave: str) -> str:
    if role == "outcome" and wave != "2011_wave1":
        return "critical"
    if role == "censoring_or_attrition":
        return "high"
    if role in {"id", "prediction_time"}:
        return "medium"
    return "low"


def build_catalog(project_root: Path) -> pd.DataFrame:
    rows = eicu_rows(project_root) + charls_rows(project_root)
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog
    catalog["role_rank"] = catalog["field_role"].map(ROLE_ORDER).fillna(99).astype(int)
    return catalog.sort_values(["dataset", "role_rank", "chrono_concept", "candidate_field"]).drop(columns=["role_rank"])


def summarize(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame(columns=["dataset", "fields", "roles", "critical_leakage_fields", "data_pending_fields"])
    return (
        catalog.groupby("dataset")
        .agg(
            fields=("candidate_field", "count"),
            roles=("field_role", lambda values: ", ".join(sorted(set(map(str, values))))),
            critical_leakage_fields=("leakage_risk", lambda values: int((pd.Series(values).astype(str) == "critical").sum())),
            data_pending_fields=("raw_data_status", lambda values: int((pd.Series(values).astype(str) == "data_pending").sum())),
        )
        .reset_index()
    )


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).astype(object).where(pd.notna(df.head(max_rows)), "")
    lines = ["| " + " | ".join(display.columns) + " |", "|" + "|".join("---" for _ in display.columns) + "|"]
    for item in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_outputs(project_root: Path, catalog: pd.DataFrame, summary: pd.DataFrame) -> Path:
    tables = project_root / "outputs" / "tables"
    reports = project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(tables / "external_field_role_catalog.csv", index=False)
    summary.to_csv(tables / "external_field_role_summary.csv", index=False)
    report = reports / "external_field_role_catalog.md"
    report.write_text(
        f"""# External Field Role Catalog

- Datasets: {catalog["dataset"].nunique() if not catalog.empty else 0}
- Candidate fields: {len(catalog)}
- Boundary: research workflow field mapping only; no medical QA, diagnosis, or treatment recommendation.

## Summary

{markdown_table(summary)}

## Field Role Catalog

{markdown_table(catalog)}

## How To Use

1. After eICU or CHARLS data lands locally, rerun the dataset readiness command.
2. Compare actual columns against `candidate_field`.
3. Treat `outcome`, `forbidden_or_leakage`, future-wave, and after-discharge fields as blocked from early prediction features.
4. Use this catalog before writing cohort extraction or model code.
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    catalog = build_catalog(args.project_root)
    summary = summarize(catalog)
    report = write_outputs(args.project_root, catalog, summary)
    print(f"Wrote {report}")
    print(f"External field-role rows: {len(catalog)}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
