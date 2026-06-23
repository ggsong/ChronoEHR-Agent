#!/usr/bin/env python3
"""Audit ChronoEHR-Agent project boundaries against medical-QA/clinical-advice drift."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimic_diabetes_baseline import DEFAULT_PROJECT


KEY_BOUNDARY_FILES = [
    "docs/agent_design.md",
    "docs/mainline_mvp_definition.md",
    "docs/quickstart_usage.md",
    "docs/resume_state.md",
    "configs/agent_action_catalog.json",
    "configs/agent_demo_workflows.json",
    "outputs/reports/agent_progress_report.md",
    "outputs/reports/agent_self_check.md",
    "outputs/reports/agent_command_lint.md",
    "outputs/reports/agent_state_validation.md",
    "outputs/reports/mainline_mvp_validation.md",
]

BOUNDARY_PHRASES = [
    "not medical QA",
    "not a medical QA",
    "not a clinical",
    "no medical QA",
    "no diagnosis",
    "no medical",
    "不是医疗问答",
    "不做诊断建议",
    "不做治疗建议",
    "不是临床诊疗建议",
    "不涉及医学诊疗建议",
]

RISKY_POSITIVE_PHRASES = [
    "should be used clinically",
    "recommended treatment",
    "recommend treatment",
    "diagnose patients",
    "diagnostic advice",
    "treatment advice",
    "is a deployable clinical model",
    "clinical decision should",
    "用于临床诊疗",
    "指导治疗",
    "推荐治疗方案",
    "诊断患者",
    "临床决策应",
]

SCAN_ROOTS = ["docs", "configs", "outputs/reports"]
SCAN_SUFFIXES = {".md", ".json", ".yaml", ".yml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def row(check: str, status: str, evidence: str, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "evidence": evidence, "detail": detail}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def has_boundary_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in BOUNDARY_PHRASES)


def iter_scannable_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        base = project_root / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in SCAN_SUFFIXES:
                if path.suffix.lower() == ".md" and any(token in path.name for token in ["audit", "validation"]):
                    continue
                files.append(path)
    return sorted(files)


def audit(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for relative in KEY_BOUNDARY_FILES:
        path = project_root / relative
        text = read_text(path)
        rows.append(
            row(
                f"boundary_phrase_present:{relative}",
                "PASS" if path.exists() and has_boundary_phrase(text) else "FAIL",
                relative,
                "expects explicit not-medical-QA / no diagnosis-treatment boundary",
            )
        )

    for path in iter_scannable_files(project_root):
        text = read_text(path)
        lowered = text.lower()
        found = [phrase for phrase in RISKY_POSITIVE_PHRASES if phrase.lower() in lowered]
        relative = str(path.relative_to(project_root))
        rows.append(
            row(
                f"risky_positive_absent:{relative}",
                "PASS" if not found else "FAIL",
                relative,
                "found=" + ", ".join(found),
            )
        )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["check", "status", "evidence", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checks = audit(args.project_root)
    failures = checks[checks["status"].ne("PASS")]
    table_path = args.project_root / "outputs" / "tables" / "agent_boundary_audit.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_boundary_audit.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Boundary Audit

- Overall status: `{"PASS" if failures.empty else "FAIL"}`
- Key boundary files: {len(KEY_BOUNDARY_FILES)}
- Files scanned for risky positive clinical-advice wording: {len(iter_scannable_files(args.project_root))}
- Checks: {len(checks)}
- Failures: {len(failures)}
- Boundary: this audit protects the project scope as a local EHR research workflow tool, not medical QA, diagnosis, treatment advice, or clinical decision support.

## Check Table

{markdown_table(checks)}
""",
        encoding="utf-8",
    )
    print(f"Agent boundary audit checks: {len(checks)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote {report_path}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
