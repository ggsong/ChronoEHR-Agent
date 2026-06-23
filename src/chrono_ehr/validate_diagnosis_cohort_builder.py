#!/usr/bin/env python3
"""Validate the generic diagnosis cohort builder against existing cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_cohort import DEFAULT_PROJECT, DEFAULT_ROOT
from mimic_diagnosis_cohort_builder import PRESETS, build_diagnosis_readmission_cohort


EXPECTED = {
    "ckd": "data/processed/mimic_ckd_readmission_cohort.csv",
    "heart_failure": "data/processed/mimic_heart_failure_readmission_cohort.csv",
}


def compare_cohort(existing: pd.DataFrame, generated: pd.DataFrame, key: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    existing_keys = set(zip(existing["subject_id"], existing["hadm_id"]))
    generated_keys = set(zip(generated["subject_id"], generated["hadm_id"]))
    missing = existing_keys - generated_keys
    extra = generated_keys - existing_keys
    if missing:
        issues.append({"cohort": key, "severity": "error", "message": f"generated cohort missing {len(missing)} existing rows"})
    if extra:
        issues.append({"cohort": key, "severity": "error", "message": f"generated cohort has {len(extra)} extra rows"})

    checks = {
        "row_count": (len(existing), len(generated)),
        "subject_count": (existing["subject_id"].nunique(), generated["subject_id"].nunique()),
        "readmission_count": (int(existing["readmission_30d"].sum()), int(generated["readmission_30d"].sum())),
    }
    for name, (left, right) in checks.items():
        if left != right:
            issues.append({"cohort": key, "severity": "error", "message": f"{name} differs: existing={left}, generated={right}"})
    return issues


def write_report(rows: list[dict[str, str]], output: Path, checked: list[str]) -> None:
    if rows:
        issue_lines = "\n".join(f"- `{row['cohort']}` {row['severity']}: {row['message']}" for row in rows)
        status = "FAIL"
    else:
        issue_lines = "- No differences found in row keys or key counts."
        status = "PASS"
    text = f"""# Generic Diagnosis Cohort Builder Validation

- Status: `{status}`
- Checked cohorts: {", ".join(checked)}

## Issues

{issue_lines}

## Meaning

This validation confirms that `mimic_diagnosis_cohort_builder.py` can reproduce the existing CKD and heart failure cohort membership and key outcome counts. It allows future ICD-prefix cohorts, such as hypertension, to reuse one builder instead of copying a full cohort script.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cohorts", nargs="*", choices=sorted(EXPECTED), default=sorted(EXPECTED))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for key in args.cohorts:
        existing = pd.read_csv(args.project_root / EXPECTED[key], low_memory=False)
        generated, _ = build_diagnosis_readmission_cohort(args.mimic_root, PRESETS[key])
        rows.extend(compare_cohort(existing, generated, key))

    table_path = args.project_root / "outputs" / "tables" / "diagnosis_cohort_builder_validation.csv"
    report_path = args.project_root / "outputs" / "reports" / "diagnosis_cohort_builder_validation_report.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows or [{"cohort": ",".join(args.cohorts), "severity": "pass", "message": "no differences"}]).to_csv(table_path, index=False)
    write_report(rows, report_path, args.cohorts)
    print(f"Diagnosis cohort builder validation issues={len(rows)}")
    print(f"Wrote {report_path}")
    if rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
