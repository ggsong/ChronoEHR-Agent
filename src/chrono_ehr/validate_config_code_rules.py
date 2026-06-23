#!/usr/bin/env python3
"""Validate config-driven ICD code rules for chronic disease cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_cohort import (
    DEFAULT_PROJECT,
    DIABETES_ICD10_PREFIXES,
    DIABETES_ICD9_PREFIXES,
    diabetes_code_prefixes,
)
from mimic_diagnosis_cohort_builder import CKD_SPEC, HEART_FAILURE_SPEC, HYPERTENSION_SPEC, spec_code_prefixes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def row(
    cohort: str,
    rule_key: str,
    config_path: str,
    loaded_icd9: tuple[str, ...],
    loaded_icd10: tuple[str, ...],
    fallback_icd9: tuple[str, ...],
    fallback_icd10: tuple[str, ...],
) -> dict[str, str]:
    status = "PASS" if loaded_icd9 == fallback_icd9 and loaded_icd10 == fallback_icd10 else "FAIL"
    return {
        "cohort": cohort,
        "rule_key": rule_key,
        "config_path": config_path,
        "loaded_icd9_prefixes": ", ".join(loaded_icd9),
        "loaded_icd10_prefixes": ", ".join(loaded_icd10),
        "fallback_icd9_prefixes": ", ".join(fallback_icd9),
        "fallback_icd10_prefixes": ", ".join(fallback_icd10),
        "status": status,
    }


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["cohort", "rule_key", "loaded_icd9_prefixes", "loaded_icd10_prefixes", "status"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in item) + " |")
    return "\n".join(lines)


def write_report(rows: pd.DataFrame, project_root: Path) -> Path:
    failures = int(rows["status"].eq("FAIL").sum())
    status = "PASS" if failures == 0 else "FAIL"
    report = project_root / "outputs" / "reports" / "config_code_rules_validation.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# Config Code Rules Validation

- Status: `{status}`
- Checked cohorts: {len(rows)}
- Failures: {failures}
- Boundary: local cohort-definition validation only; no diagnosis, treatment, or medical QA.

## Results

{markdown_table(rows)}

## Meaning

This check confirms that the ICD prefix rules used by the cohort builders can be loaded from study configs. It keeps the first Agent MVP reproducible while moving runner logic away from hidden script constants.
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    diabetes_icd9, diabetes_icd10 = diabetes_code_prefixes(args.project_root / "configs" / "diabetes_mimic_readmission.yaml")
    rows = [
        row(
            "diabetes",
            "diabetes_code_rules",
            "configs/diabetes_mimic_readmission.yaml",
            diabetes_icd9,
            diabetes_icd10,
            DIABETES_ICD9_PREFIXES,
            DIABETES_ICD10_PREFIXES,
        )
    ]
    for spec in [CKD_SPEC, HEART_FAILURE_SPEC, HYPERTENSION_SPEC]:
        loaded_icd9, loaded_icd10 = spec_code_prefixes(spec, args.project_root)
        rows.append(
            row(
                spec.cohort_key,
                spec.code_rule_key or "",
                spec.config_path or "",
                loaded_icd9,
                loaded_icd10,
                spec.icd9_prefixes,
                spec.icd10_prefixes,
            )
        )
    df = pd.DataFrame(rows)
    table = args.project_root / "outputs" / "tables" / "config_code_rules_validation.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table, index=False)
    report = write_report(df, args.project_root)
    failures = int(df["status"].eq("FAIL").sum())
    print(f"Config code rules validation: {len(df) - failures}/{len(df)} PASS")
    print(f"Wrote {report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
