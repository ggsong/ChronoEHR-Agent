#!/usr/bin/env python3
"""Summarize subgroup robustness for selected external benchmark rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from mimic_diabetes_baseline import DEFAULT_PROJECT


EXPECTED_ROWS = [
    "CDSL early-window best",
    "CDSL full-stay naive reference",
    "eICU calibrated logistic reference",
    "eICU best calibrated RF/HGB",
    "CHARLS calibrated logistic reference",
    "CHARLS best calibrated RF/HGB",
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


def fmt_metric(value: object) -> str:
    return "" if pd.isna(value) else f"{float(value):.4f}"


def subgroup_descriptor(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return f"{row['subgroup_type']}={row['subgroup']}"


def robustness_status(ok_rows: int, small_rows: int, min_replicates: int, min_auroc_lower: float, min_auprc_lower: float) -> str:
    if ok_rows == 0 or min_replicates < 450:
        return "INSUFFICIENT"
    if small_rows > 0 or min_auroc_lower < 0.5 or min_auprc_lower < 0.05:
        return "CAUTION"
    return "SUPPORTED"


def robustness_note(dataset: str, benchmark_row: str, status: str, small_rows: int, weakest_auroc: str, weakest_auprc: str) -> str:
    parts = []
    if status == "SUPPORTED":
        parts.append("All evaluable subgroup CI rows have adequate bootstrap support.")
    elif status == "CAUTION":
        parts.append("Subgroup evidence is usable but should be read with caution.")
    else:
        parts.append("Subgroup evidence is insufficient for robustness interpretation.")
    if small_rows:
        parts.append(f"{small_rows} subgroup rows are small or single-class.")
    parts.append(f"Weakest AUROC-lower subgroup: {weakest_auroc}.")
    parts.append(f"Weakest AUPRC-lower subgroup: {weakest_auprc}.")
    if benchmark_row == "CDSL full-stay naive reference":
        parts.append("This remains a naive upper-reference rather than early prediction performance.")
    if dataset == "eICU":
        parts.append("This is an ICU mortality benchmark, not chronic readmission external validation.")
    if dataset == "CHARLS":
        parts.append("This is a longitudinal cohort extension with low-event subgroup caution.")
    return " ".join(parts)


def build_summary(project_root: Path) -> pd.DataFrame:
    tables = project_root / "outputs" / "tables"
    summary = normalize(read_csv(tables / "external_benchmark_summary_table.csv"))
    subgroup = normalize(read_csv(tables / "external_subgroup_bootstrap_ci.csv"))
    if summary.empty or subgroup.empty:
        raise FileNotFoundError("Missing external benchmark summary or subgroup bootstrap CI table.")

    rows: list[dict[str, object]] = []
    subgroup_by_key = {key: frame.copy() for key, frame in subgroup.groupby("_key", sort=True)}
    for _, selected in summary.sort_values("benchmark_row").iterrows():
        selected_group = subgroup_by_key.get(str(selected["_key"]), pd.DataFrame())
        ok = selected_group[selected_group["status"].astype(str).eq("OK")].copy()
        small = selected_group[~selected_group["status"].astype(str).eq("OK")].copy()
        weakest_auroc = ok.sort_values(["AUROC_lower", "AUPRC_lower", "n"], ascending=[True, True, True]).head(1)
        weakest_auprc = ok.sort_values(["AUPRC_lower", "AUROC_lower", "n"], ascending=[True, True, True]).head(1)
        weakest_brier = ok.sort_values(["Brier_upper", "n"], ascending=[False, True]).head(1)
        auroc_row = weakest_auroc.iloc[0] if not weakest_auroc.empty else None
        auprc_row = weakest_auprc.iloc[0] if not weakest_auprc.empty else None
        brier_row = weakest_brier.iloc[0] if not weakest_brier.empty else None
        min_replicates = int(ok["bootstrap_replicates"].min()) if not ok.empty else 0
        min_auroc_lower = float(ok["AUROC_lower"].min()) if not ok.empty else float("nan")
        min_auprc_lower = float(ok["AUPRC_lower"].min()) if not ok.empty else float("nan")
        status = robustness_status(
            ok_rows=len(ok),
            small_rows=len(small),
            min_replicates=min_replicates,
            min_auroc_lower=min_auroc_lower,
            min_auprc_lower=min_auprc_lower,
        )
        rows.append(
            {
                "benchmark_row": selected["benchmark_row"],
                "dataset": selected["dataset"],
                "feature_set": selected["feature_set"],
                "model": selected["model"],
                "calibration_method": selected["calibration_method"],
                "subgroup_ci_rows": int(len(selected_group)),
                "subgroup_ci_ok_rows": int(len(ok)),
                "subgroup_ci_small_or_single_class_rows": int(len(small)),
                "subgroup_types": ",".join(sorted(selected_group["subgroup_type"].dropna().astype(str).unique())) if not selected_group.empty else "",
                "min_bootstrap_replicates": min_replicates,
                "weakest_auroc_subgroup": subgroup_descriptor(auroc_row),
                "weakest_auroc_n": int(auroc_row["n"]) if auroc_row is not None else 0,
                "weakest_auroc_events": int(auroc_row["events"]) if auroc_row is not None else 0,
                "weakest_auroc": float(auroc_row["AUROC"]) if auroc_row is not None else float("nan"),
                "weakest_auroc_lower": float(auroc_row["AUROC_lower"]) if auroc_row is not None else float("nan"),
                "weakest_auroc_upper": float(auroc_row["AUROC_upper"]) if auroc_row is not None else float("nan"),
                "weakest_auprc_subgroup": subgroup_descriptor(auprc_row),
                "weakest_auprc_n": int(auprc_row["n"]) if auprc_row is not None else 0,
                "weakest_auprc_events": int(auprc_row["events"]) if auprc_row is not None else 0,
                "weakest_auprc": float(auprc_row["AUPRC"]) if auprc_row is not None else float("nan"),
                "weakest_auprc_lower": float(auprc_row["AUPRC_lower"]) if auprc_row is not None else float("nan"),
                "weakest_auprc_upper": float(auprc_row["AUPRC_upper"]) if auprc_row is not None else float("nan"),
                "highest_brier_upper_subgroup": subgroup_descriptor(brier_row),
                "highest_brier_upper": float(brier_row["Brier_upper"]) if brier_row is not None else float("nan"),
                "robustness_status": status,
                "robustness_note": robustness_note(
                    str(selected["dataset"]),
                    str(selected["benchmark_row"]),
                    status,
                    len(small),
                    subgroup_descriptor(auroc_row),
                    subgroup_descriptor(auprc_row),
                ),
            }
        )
    table = pd.DataFrame(rows)
    expected_order = {label: i for i, label in enumerate(EXPECTED_ROWS)}
    table["_order"] = table["benchmark_row"].map(expected_order).fillna(999).astype(int)
    return table.sort_values(["_order", "benchmark_row"]).drop(columns=["_order"]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[float]).columns:
        display[col] = display[col].map(lambda value: fmt_metric(value))
    columns = [
        "benchmark_row",
        "subgroup_ci_ok_rows",
        "subgroup_ci_small_or_single_class_rows",
        "weakest_auroc_subgroup",
        "weakest_auroc_lower",
        "weakest_auprc_subgroup",
        "weakest_auprc_lower",
        "robustness_status",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for item in display[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in item) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, table: pd.DataFrame) -> Path:
    report = project_root / "outputs" / "reports" / "external_subgroup_robustness_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"""# External Subgroup Robustness Summary

- Boundary: research subgroup robustness summary only; no diagnosis, treatment, deployment guidance, or care-threshold advice.
- Scope: selected external benchmark rows already used by the external technical summary.
- Rows: {len(table)}
- Datasets: {", ".join(sorted(table["dataset"].dropna().astype(str).unique()))}

## Robustness Table

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
    table_path = tables / "external_subgroup_robustness_summary.csv"
    supp_path = supplement / "table_s18_external_subgroup_robustness_summary.csv"
    table.to_csv(table_path, index=False)
    table.to_csv(supp_path, index=False)
    report = write_report(args.project_root, table)
    print(f"External subgroup robustness rows: {len(table)}")
    print(f"Wrote {table_path}")
    print(f"Wrote {supp_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
