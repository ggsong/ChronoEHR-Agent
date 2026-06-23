#!/usr/bin/env python3
"""Audit the generated English brief for required sections and safety boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def audit(project_root: Path) -> pd.DataFrame:
    report_path = project_root / "outputs" / "reports" / "chronic_disease_methods_results_english_brief.md"
    key_path = project_root / "outputs" / "tables" / "english_brief_key_results.csv"
    template_path = project_root / "configs" / "report_text_templates_english_brief.json"
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rows: list[dict[str, str]] = []

    rows.append(row("report_exists", "PASS" if report_path.exists() and report_path.stat().st_size > 0 else "FAIL", str(report_path), "English brief markdown must exist."))
    rows.append(row("template_exists", "PASS" if template_path.exists() and template_path.stat().st_size > 0 else "FAIL", str(template_path), "English text template must exist."))
    rows.append(row("key_results_exist", "PASS" if key_path.exists() and key_path.stat().st_size > 0 else "FAIL", str(key_path), "Key English brief metrics table must exist."))

    required_phrases = [
        "not a medical question-answering system",
        "does not provide diagnosis or treatment recommendations",
        "not as a deployable clinical model",
        "research baselines, not clinical deployment candidates",
        "rather than a clinical alert recommendation",
    ]
    for phrase in required_phrases:
        rows.append(row(f"boundary_phrase:{phrase[:30]}", "PASS" if phrase in text else "FAIL", str(report_path), phrase))

    required_sections = [
        "## Structured Abstract",
        "## Methods Brief",
        "## Results Brief",
        "### Cohort Summary",
        "### Prediction-Time Effect",
        "### Traditional Baseline Models",
        "### Leakage And Sensitivity Interpretation",
        "## Limitations",
    ]
    for section in required_sections:
        rows.append(row(f"required_section:{section}", "PASS" if section in text else "FAIL", str(report_path), section))

    risky_phrases = [
        "should be used clinically",
        "recommended treatment",
        "diagnose patients",
        "clinical decision should",
        "is a deployable clinical model",
    ]
    for phrase in risky_phrases:
        rows.append(row(f"risky_phrase_absent:{phrase}", "PASS" if phrase not in text else "FAIL", str(report_path), f"Phrase should be absent: {phrase}"))

    if key_path.exists():
        key = pd.read_csv(key_path)
        metrics = set(key["metric"].astype(str)) if "metric" in key else set()
        expected = {
            "total_index_admissions",
            "total_subjects",
            "mean_discharge_minus_admission_AUROC",
            "mean_discharge_minus_admission_AUPRC",
            "mean_abs_ED_LOS_delta_AUROC",
            "mean_abs_ED_LOS_delta_AUPRC",
        }
        missing = sorted(expected - metrics)
        rows.append(row("key_metrics_complete", "PASS" if not missing else "FAIL", str(key_path), "missing=" + ", ".join(missing)))

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, checks: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "english_brief_quality_audit.md"
    failures = checks[checks["status"].ne("PASS")]
    text = f"""# English Brief Quality Audit

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Checks: {len(checks)}
- Failures: {len(failures)}

This audit checks whether the generated English brief contains required research-tool boundaries, required sections, key metrics, and no obvious clinical-deployment wording.

## Check Table

{markdown_table(checks)}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "english_brief_quality_audit.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report = write_report(args.project_root, checks)
    failures = int((checks["status"] != "PASS").sum())
    print(f"Wrote {report}")
    print(f"English brief audit checks: {len(checks)}")
    print(f"Failures: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
