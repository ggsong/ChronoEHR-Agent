#!/usr/bin/env python3
"""Compute a concise ChronoEHR-Agent local MVP progress score."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


COMPONENTS = [
    ("mainline_mvp_gate", 20.0),
    ("primary_diabetes_demo", 20.0),
    ("mimic_replication_cohorts", 15.0),
    ("agent_control_layer", 25.0),
    ("external_benchmark_readiness", 10.0),
    ("documentation_reproducibility", 10.0),
]


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


def pass_rate(df: pd.DataFrame) -> float:
    if df.empty or "status" not in df.columns:
        return 0.0
    return float((df["status"].astype(str) == "PASS").mean())


def capability_completion(project_root: Path, study_ids: list[str]) -> float:
    summary = read_csv(project_root / "outputs" / "tables" / "study_capability_summary.csv")
    if summary.empty or not {"study_id", "completion_percent"}.issubset(summary.columns):
        return 0.0
    subset = summary[summary["study_id"].astype(str).isin(study_ids)]
    if subset.empty:
        return 0.0
    return float(pd.to_numeric(subset["completion_percent"], errors="coerce").fillna(0).mean() / 100.0)


def external_score(project_root: Path) -> tuple[float, str]:
    external = read_csv(project_root / "outputs" / "tables" / "external_benchmark_readiness_summary.csv")
    if external.empty or not {"dataset", "local_status"}.issubset(external.columns):
        return 0.0, "external readiness summary missing"
    statuses = dict(zip(external["dataset"].astype(str), external["local_status"].astype(str)))
    score_by_dataset = {
        "CDSL": 1.0 if statuses.get("CDSL") == "READY" else 0.0,
        "eICU": 1.0 if statuses.get("eICU") == "BASELINE_READY" else 0.5 if statuses.get("eICU") else 0.0,
        "CHARLS": 0.6 if statuses.get("CHARLS") == "DATA_PENDING" else 1.0 if statuses.get("CHARLS") in {"READY", "READY_FOR_PROTOCOL_CODE"} else 0.0,
    }
    return sum(score_by_dataset.values()) / 3.0, ", ".join(f"{key}={statuses.get(key, '')}" for key in ["CDSL", "eICU", "CHARLS"])


def component_rows(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weights = dict(COMPONENTS)

    mvp = read_csv(project_root / "outputs" / "tables" / "mainline_mvp_validation.csv")
    mvp_fraction = pass_rate(mvp)
    rows.append(
        {
            "component": "mainline_mvp_gate",
            "weight": weights["mainline_mvp_gate"],
            "fraction_complete": mvp_fraction,
            "weighted_points": weights["mainline_mvp_gate"] * mvp_fraction,
            "status": "PASS" if mvp_fraction == 1.0 else "ATTENTION",
            "evidence": "outputs/tables/mainline_mvp_validation.csv",
            "detail": f"{int((mvp.get('status', pd.Series(dtype=str)).astype(str) == 'PASS').sum()) if not mvp.empty and 'status' in mvp else 0}/{len(mvp)} checks PASS",
        }
    )

    diabetes_fraction = capability_completion(project_root, ["mimic_iv_3_1_diabetes_readmission"])
    rows.append(
        {
            "component": "primary_diabetes_demo",
            "weight": weights["primary_diabetes_demo"],
            "fraction_complete": diabetes_fraction,
            "weighted_points": weights["primary_diabetes_demo"] * diabetes_fraction,
            "status": "PASS" if diabetes_fraction >= 1.0 else "ATTENTION",
            "evidence": "outputs/tables/study_capability_summary.csv",
            "detail": "MIMIC-IV diabetes vertical slice capability completion",
        }
    )

    replication_fraction = capability_completion(
        project_root,
        ["mimic_iv_ckd_readmission", "mimic_iv_heart_failure_readmission", "mimic_iv_hypertension_readmission"],
    )
    rows.append(
        {
            "component": "mimic_replication_cohorts",
            "weight": weights["mimic_replication_cohorts"],
            "fraction_complete": replication_fraction,
            "weighted_points": weights["mimic_replication_cohorts"] * replication_fraction,
            "status": "PASS" if replication_fraction >= 1.0 else "ATTENTION",
            "evidence": "outputs/tables/study_capability_summary.csv",
            "detail": "CKD, heart failure, and hypertension replication cohorts",
        }
    )

    control_sources = [
        "outputs/tables/agent_self_check.csv",
        "outputs/tables/agent_doctor.csv",
        "outputs/tables/agent_command_lint.csv",
        "outputs/tables/agent_control_consistency_audit.csv",
        "outputs/tables/agent_dependency_audit.csv",
        "outputs/tables/agent_artifact_freshness.csv",
        "outputs/tables/agent_status_card_validation.csv",
    ]
    control_rates = [pass_rate(read_csv(project_root / source)) for source in control_sources]
    control_fraction = sum(control_rates) / len(control_rates)
    rows.append(
        {
            "component": "agent_control_layer",
            "weight": weights["agent_control_layer"],
            "fraction_complete": control_fraction,
            "weighted_points": weights["agent_control_layer"] * control_fraction,
            "status": "PASS" if control_fraction == 1.0 else "ATTENTION",
            "evidence": "; ".join(control_sources),
            "detail": "self-check, doctor, command lint, consistency, dependency, freshness, status-card validation",
        }
    )

    ext_fraction, ext_detail = external_score(project_root)
    rows.append(
        {
            "component": "external_benchmark_readiness",
            "weight": weights["external_benchmark_readiness"],
            "fraction_complete": ext_fraction,
            "weighted_points": weights["external_benchmark_readiness"] * ext_fraction,
            "status": "PASS" if ext_fraction >= 0.85 else "PARTIAL",
            "evidence": "outputs/tables/external_benchmark_readiness_summary.csv",
            "detail": ext_detail,
        }
    )

    doc_sources = [
        "outputs/tables/agent_doc_command_audit.csv",
        "outputs/tables/agent_boundary_audit.csv",
        "outputs/tables/delivery_readiness_audit.csv",
    ]
    doc_rates = [pass_rate(read_csv(project_root / source)) for source in doc_sources]
    doc_fraction = sum(doc_rates) / len(doc_rates)
    rows.append(
        {
            "component": "documentation_reproducibility",
            "weight": weights["documentation_reproducibility"],
            "fraction_complete": doc_fraction,
            "weighted_points": weights["documentation_reproducibility"] * doc_fraction,
            "status": "PASS" if doc_fraction == 1.0 else "ATTENTION",
            "evidence": "; ".join(doc_sources),
            "detail": "doc command audit, boundary audit, delivery readiness",
        }
    )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["component", "weight", "fraction_complete", "weighted_points", "status", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in df[columns].itertuples(index=False):
        values = list(item)
        values[2] = f"{float(values[2]) * 100:.1f}%"
        values[3] = f"{float(values[3]):.1f}"
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = component_rows(args.project_root)
    total = float(rows["weighted_points"].sum()) if not rows.empty else 0.0
    overall_status = "PASS" if total >= 95.0 else "PARTIAL" if total >= 80.0 else "ATTENTION"
    table_path = args.project_root / "outputs" / "tables" / "agent_progress_score.csv"
    report_path = args.project_root / "outputs" / "reports" / "agent_progress_score.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output = rows.copy()
    output.loc[len(output)] = {
        "component": "TOTAL",
        "weight": 100.0,
        "fraction_complete": total / 100.0,
        "weighted_points": total,
        "status": overall_status,
        "evidence": "weighted component score",
        "detail": "Local MVP progress score; remaining gap is mainly CHARLS data availability and future expansion.",
    }
    output.to_csv(table_path, index=False)
    report_path.write_text(
        f"""# Agent Progress Score

- Overall score: `{total:.1f}/100`
- Overall status: `{overall_status}`
- Boundary: local research-tool progress estimate only; no medical QA, diagnosis, or treatment recommendation.

## Component Scores

{markdown_table(rows)}

## Interpretation

- `95+` means v0.1 is effectively usable as a stable local research-tool MVP.
- Scores below 100 can still be acceptable when gaps are due to external data pending, such as CHARLS wave approval/download.
- This score is a project-management summary, not a model performance metric.
""",
        encoding="utf-8",
    )
    print(f"Agent progress score: {total:.1f}/100 ({overall_status})")
    print(f"Wrote {report_path}")
    if overall_status == "ATTENTION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
