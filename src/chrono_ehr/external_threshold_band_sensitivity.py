#!/usr/bin/env python3
"""Summarize decision-curve sensitivity by threshold bands for selected external rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


BANDS = [
    ("very_low_0.02_0.05", 0.02, 0.05),
    ("low_mid_0.10_0.20", 0.10, 0.20),
    ("high_0.30_0.50", 0.30, 0.50),
]


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
    if "feature_set" not in out and "feature_window" in out:
        out["feature_set"] = out["feature_window"]
    for col in ["dataset", "feature_set", "model", "calibration_method"]:
        if col not in out:
            out[col] = ""
    out["calibration_method"] = [
        normalize_method(method, dataset) for method, dataset in zip(out["calibration_method"], out["dataset"])
    ]
    out["_key"] = out[["dataset", "feature_set", "model", "calibration_method"]].fillna("").astype(str).agg("||".join, axis=1)
    return out


def decision_curve_rows(project_root: Path) -> pd.DataFrame:
    tables = project_root / "outputs" / "tables"
    frames: list[pd.DataFrame] = []

    cdsl = read_csv(tables / "cdsl_decision_curve.csv")
    if not cdsl.empty:
        part = cdsl[cdsl["split"].astype(str).eq("test")].copy()
        part["dataset"] = "CDSL"
        part["calibration_method"] = "raw_traditional"
        frames.append(part)

    eicu = read_csv(tables / "eicu_probability_recalibration_decision_curve.csv")
    if not eicu.empty:
        part = eicu[eicu["split"].astype(str).eq("test")].copy()
        part["dataset"] = "eICU"
        part["feature_set"] = "first24h_lab_vital"
        part["model"] = "logistic_regression_balanced"
        frames.append(part)

    charls = read_csv(tables / "charls_probability_recalibration_decision_curve.csv")
    if not charls.empty:
        part = charls[charls["split"].astype(str).eq("test")].copy()
        part["dataset"] = "CHARLS"
        part["feature_set"] = "charls_2011_baseline"
        part["model"] = "logistic_regression_balanced"
        frames.append(part)

    comparison = read_csv(tables / "external_model_comparison_recalibration_decision_curve.csv")
    if not comparison.empty:
        part = comparison[comparison["split"].astype(str).eq("test")].copy()
        part["feature_set"] = np.where(part["dataset"].astype(str).eq("eICU"), "first24h_lab_vital", "charls_2011_baseline")
        part["calibration_method"] = part["calibration_method"].replace({"raw": "raw_model_comparison"})
        frames.append(part)

    if not frames:
        return pd.DataFrame()
    return normalize(pd.concat(frames, ignore_index=True, sort=False))


def band_status(group: pd.DataFrame) -> str:
    if group.empty:
        return "NO_DATA"
    advantages = group["net_benefit_advantage"].fillna(0).astype(float)
    preferred = group["preferred_strategy"].astype(str)
    if advantages.gt(0).all() and preferred.eq("model").all():
        return "CONSISTENT_MODEL_ADVANTAGE"
    if advantages.gt(0).any() or preferred.eq("model").any():
        return "MIXED"
    return "NO_MODEL_ADVANTAGE"


def build_summary(project_root: Path) -> pd.DataFrame:
    selected = normalize(read_csv(project_root / "outputs" / "tables" / "external_benchmark_summary_table.csv"))
    decision = decision_curve_rows(project_root)
    if selected.empty or decision.empty:
        raise FileNotFoundError("Missing external benchmark summary or decision-curve rows.")
    decision_by_key = {key: frame.copy() for key, frame in decision.groupby("_key", sort=True)}
    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        curves = decision_by_key.get(str(selected_row["_key"]), pd.DataFrame())
        for band_name, low, high in BANDS:
            band = curves[
                curves["threshold_probability"].astype(float).between(low, high, inclusive="both")
            ].copy()
            best = band.sort_values(["net_benefit_advantage", "threshold_probability"], ascending=[False, True]).head(1)
            rows.append(
                {
                    "benchmark_row": selected_row["benchmark_row"],
                    "dataset": selected_row["dataset"],
                    "feature_set": selected_row["feature_set"],
                    "model": selected_row["model"],
                    "calibration_method": selected_row["calibration_method"],
                    "threshold_band": band_name,
                    "threshold_lower": low,
                    "threshold_upper": high,
                    "threshold_count": int(len(band)),
                    "model_preferred_thresholds": int(band["preferred_strategy"].astype(str).eq("model").sum()) if not band.empty else 0,
                    "positive_advantage_thresholds": int(band["net_benefit_advantage"].fillna(0).astype(float).gt(0).sum()) if not band.empty else 0,
                    "min_net_benefit_advantage": float(band["net_benefit_advantage"].min()) if not band.empty else np.nan,
                    "mean_net_benefit_advantage": float(band["net_benefit_advantage"].mean()) if not band.empty else np.nan,
                    "max_net_benefit_advantage": float(band["net_benefit_advantage"].max()) if not band.empty else np.nan,
                    "best_threshold": float(best["threshold_probability"].iloc[0]) if not best.empty else np.nan,
                    "best_alert_rate": float(best["alert_rate"].iloc[0]) if not best.empty else np.nan,
                    "best_ppv": float(best["ppv"].iloc[0]) if not best.empty else np.nan,
                    "best_recall": float(best["recall"].iloc[0]) if not best.empty else np.nan,
                    "band_status": band_status(band),
                    "boundary_note": boundary_note(str(selected_row["benchmark_row"]), str(selected_row["dataset"])),
                }
            )
    return pd.DataFrame(rows)


def boundary_note(benchmark_row: str, dataset: str) -> str:
    note = "Research threshold-band sensitivity only; thresholds are evaluation grid points, not care advice."
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
        "threshold_band",
        "threshold_count",
        "model_preferred_thresholds",
        "positive_advantage_thresholds",
        "mean_net_benefit_advantage",
        "best_threshold",
        "band_status",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, table: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_threshold_band_sensitivity.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# External Threshold-Band Sensitivity

- Boundary: research decision-curve sensitivity only; threshold grid points are not care advice or deployment guidance.
- Scope: selected external benchmark rows and three fixed threshold bands.
- Rows: {len(table)}

## Threshold-Band Table

{markdown_table(table)}
""",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    table = build_summary(args.project_root)
    tables = args.project_root / "outputs" / "tables"
    supplement = tables / "supplementary_appendix"
    tables.mkdir(parents=True, exist_ok=True)
    supplement.mkdir(parents=True, exist_ok=True)
    table_path = tables / "external_threshold_band_sensitivity.csv"
    supp_path = supplement / "table_s19_external_threshold_band_sensitivity.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(supp_path, index=False)
    report = write_report(args.project_root, table)
    print(f"External threshold-band rows: {len(table)}")
    print(f"Wrote {table_path}")
    print(f"Wrote {supp_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
