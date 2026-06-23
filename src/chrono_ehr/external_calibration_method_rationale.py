#!/usr/bin/env python3
"""Build calibration-method rationale rows for selected external benchmark rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def normalize_method(value: object, dataset: object) -> str:
    dataset_text = "" if pd.isna(dataset) else str(dataset)
    if pd.isna(value) or str(value).strip() == "":
        return "raw_traditional" if dataset_text == "CDSL" else "raw"
    text = str(value)
    if text == "raw" and dataset_text in {"eICU", "CHARLS"}:
        return "raw_model_comparison"
    return text


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["dataset", "feature_set", "model", "calibration_method"]:
        if col not in out:
            out[col] = ""
    out["calibration_method"] = [
        normalize_method(method, dataset) for method, dataset in zip(out["calibration_method"], out["dataset"])
    ]
    out["_key"] = out[["dataset", "feature_set", "model", "calibration_method"]].fillna("").astype(str).agg("||".join, axis=1)
    return out


def selected_reason(selected: pd.Series, candidates: pd.DataFrame, selected_rank: int) -> str:
    dataset = str(selected["dataset"])
    method = str(selected["calibration_method"])
    if dataset == "CDSL":
        return "CDSL traditional temporal benchmark uses raw probabilities; no validation-set recalibration method is defined for these rows."
    if method == "platt_validation":
        return f"Selected Platt calibration; calibration MAE rank within model={selected_rank}, with validation-set calibration and selected benchmark performance."
    return f"Selected {method}; calibration MAE rank within model={selected_rank}."


def build_rationale(project_root: Path) -> pd.DataFrame:
    tables = project_root / "outputs" / "tables"
    selected = normalize(read_csv(tables / "external_benchmark_summary_table.csv"))
    calibration = normalize(read_csv(tables / "external_calibration_decision_summary.csv"))
    if selected.empty or calibration.empty:
        raise FileNotFoundError("Missing external benchmark summary or calibration decision summary.")

    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        group = calibration[
            calibration["dataset"].astype(str).eq(str(selected_row["dataset"]))
            & calibration["feature_set"].astype(str).eq(str(selected_row["feature_set"]))
            & calibration["model"].astype(str).eq(str(selected_row["model"]))
        ].copy()
        group = group.sort_values(["mean_absolute_calibration_error", "decision_best_net_benefit_advantage"], ascending=[True, False])
        selected_match = group[group["calibration_method"].astype(str).eq(str(selected_row["calibration_method"]))]
        selected_candidate = selected_match.iloc[0] if not selected_match.empty else None
        best_calibration = group.iloc[0] if not group.empty else None
        raw = group[
            group["calibration_method"].astype(str).isin(["raw", "raw_model_comparison", "raw_traditional"])
        ]
        raw_candidate = raw.iloc[0] if not raw.empty else None
        selected_rank = int(selected_candidate["calibration_rank_within_model"]) if selected_candidate is not None else 0
        best_method = str(best_calibration["calibration_method"]) if best_calibration is not None else ""
        rows.append(
            {
                "benchmark_row": selected_row["benchmark_row"],
                "dataset": selected_row["dataset"],
                "feature_set": selected_row["feature_set"],
                "model": selected_row["model"],
                "selected_calibration_method": selected_row["calibration_method"],
                "candidate_methods": ",".join(group["calibration_method"].dropna().astype(str).tolist()),
                "candidate_method_count": int(len(group)),
                "selected_calibration_rank_within_model": selected_rank,
                "best_calibration_method_by_mae": best_method,
                "selected_is_best_calibration_mae": bool(str(selected_row["calibration_method"]) == best_method),
                "selected_mean_absolute_calibration_error": float(selected_candidate["mean_absolute_calibration_error"]) if selected_candidate is not None else np.nan,
                "best_mean_absolute_calibration_error": float(best_calibration["mean_absolute_calibration_error"]) if best_calibration is not None else np.nan,
                "raw_mean_absolute_calibration_error": float(raw_candidate["mean_absolute_calibration_error"]) if raw_candidate is not None else np.nan,
                "selected_mae_delta_vs_raw": float(selected_candidate["mean_absolute_calibration_error"] - raw_candidate["mean_absolute_calibration_error"]) if selected_candidate is not None and raw_candidate is not None else np.nan,
                "selected_decision_positive_advantage_thresholds": int(selected_candidate["decision_positive_advantage_thresholds"]) if selected_candidate is not None else 0,
                "selected_decision_best_threshold": float(selected_candidate["decision_best_threshold"]) if selected_candidate is not None else np.nan,
                "selected_decision_best_net_benefit_advantage": float(selected_candidate["decision_best_net_benefit_advantage"]) if selected_candidate is not None else np.nan,
                "rationale_status": "PASS" if selected_candidate is not None and len(group) > 0 else "FAIL",
                "rationale_note": selected_reason(selected_row, group, selected_rank),
                "boundary_note": boundary_note(str(selected_row["benchmark_row"]), str(selected_row["dataset"])),
            }
        )
    return pd.DataFrame(rows)


def boundary_note(benchmark_row: str, dataset: str) -> str:
    note = "Research calibration-method rationale only."
    if benchmark_row == "CDSL full-stay naive reference":
        note += " CDSL full-stay is a naive upper-reference."
    if dataset == "eICU":
        note += " eICU is ICU mortality, not chronic readmission external validation."
    if dataset == "CHARLS":
        note += " CHARLS is a longitudinal cohort extension."
    return note


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = [
        "benchmark_row",
        "selected_calibration_method",
        "candidate_method_count",
        "selected_calibration_rank_within_model",
        "best_calibration_method_by_mae",
        "selected_is_best_calibration_mae",
        "selected_mae_delta_vs_raw",
        "rationale_status",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, table: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_calibration_method_rationale.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# External Calibration-Method Rationale

- Boundary: research calibration-method rationale only; no diagnosis, treatment, deployment guidance, or care-threshold advice.
- Scope: selected external benchmark rows and their within-model calibration candidates.
- Rows: {len(table)}

## Rationale Table

{markdown_table(table)}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    table = build_rationale(args.project_root)
    tables = args.project_root / "outputs" / "tables"
    supplement = tables / "supplementary_appendix"
    tables.mkdir(parents=True, exist_ok=True)
    supplement.mkdir(parents=True, exist_ok=True)
    table_path = tables / "external_calibration_method_rationale.csv"
    supp_path = supplement / "table_s20_external_calibration_method_rationale.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(supp_path, index=False)
    report = write_report(args.project_root, table)
    print(f"External calibration-method rationale rows: {len(table)}")
    print(f"Wrote {table_path}")
    print(f"Wrote {supp_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
