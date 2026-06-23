"""Shared leakage and outcome sensitivity utilities for ChronoEHR studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score)
    sorted_y = y_true[order]
    cum_pos = np.cumsum(sorted_y)
    precision = cum_pos / (np.arange(len(sorted_y)) + 1)
    return float((precision * sorted_y).sum() / n_pos)


def binary_metrics(name: str, y: np.ndarray, score: np.ndarray, note: str) -> dict:
    pred = (score >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "scenario": name,
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "AUROC": auroc(y, score),
        "AUPRC": average_precision(y, score),
        "accuracy": float((pred == y).mean()),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "note": note,
    }


def is_emergency_or_urgent(admission_type: object) -> bool:
    text = str(admission_type).upper()
    return any(term in text for term in ["EMER", "URGENT", "EW"])


def fmt_float(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.4f}"


def run_leakage_sensitivity(
    cohort_path: Path,
    performance_path: Path,
    output_table: Path,
    output_report: Path,
    valid_feature_set: str,
    valid_scenario: str,
    report_title: str,
    intro: str,
    valid_note: str,
) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path, low_memory=False)
    test = cohort[cohort["split"].eq("test")].copy()
    y = test["readmission_30d"].astype(int).to_numpy()

    perf = pd.read_csv(performance_path)
    valid = perf[(perf["split"].eq("test")) & (perf["feature_set"].eq(valid_feature_set))].iloc[0]

    future_days_score = test["days_to_next_admission"].between(0, 30, inclusive="both").astype(float).to_numpy()
    future_next_admit_score = test["next_admittime"].notna().astype(float).to_numpy()

    rows = [
        {
            "scenario": valid_scenario,
            "n": int(valid["n"]),
            "events": int(valid["events"]),
            "event_rate": float(valid["event_rate"]),
            "AUROC": float(valid["AUROC"]),
            "AUPRC": float(valid["AUPRC"]),
            "accuracy": np.nan,
            "sensitivity": float(valid["sensitivity"]),
            "specificity": float(valid["specificity"]),
            "note": valid_note,
        },
        binary_metrics(
            "leaked_days_to_next_admission",
            y,
            future_days_score,
            "错误示范：days_to_next_admission 由未来住院时间直接派生，几乎等于答案。",
        ),
        binary_metrics(
            "leaked_next_admittime_available",
            y,
            future_next_admit_score,
            "错误示范：是否存在下一次入院来自随访之后，不能作为预测特征。",
        ),
    ]

    output_table.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(output_table, index=False)

    lines = [
        f"# {report_title}",
        "",
        intro,
        "",
        "| Scenario | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['AUROC']:.4f} | {row['AUPRC']:.4f} | "
            f"{fmt_float(row['accuracy'])} | {row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "结论：`days_to_next_admission` 和 `next_admittime` 都是在预测时间点之后才知道的信息。",
            "如果把它们放进模型，AUROC/AUPRC 会被人为抬高，不能作为真实研究结果报告。",
        ]
    )
    output_report.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_outcome_sensitivity(
    cohort_path: Path,
    output_table: Path,
    output_type_table: Path,
    output_report: Path,
    report_title: str,
) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path, low_memory=False)
    next_type = cohort["next_admission_type"].astype("string").fillna("").str.upper()
    cohort["readmission_30d_emergency_urgent"] = (
        cohort["readmission_30d"].eq(1) & cohort["next_admission_type"].map(is_emergency_or_urgent)
    ).astype(int)
    cohort["readmission_30d_non_elective_proxy"] = (
        cohort["readmission_30d"].eq(1) & ~next_type.str.contains("ELECTIVE|SURGICAL SAME DAY", regex=True)
    ).astype(int)

    definitions = [
        (
            "all_cause_30d_readmission",
            "Any next hospital admission within 30 days after discharge.",
            "readmission_30d",
        ),
        (
            "emergency_urgent_30d_readmission",
            "Next admission within 30 days where next_admission_type contains EMER, URGENT, or EW.",
            "readmission_30d_emergency_urgent",
        ),
        (
            "non_elective_proxy_30d_readmission",
            "Next admission within 30 days excluding ELECTIVE and SURGICAL SAME DAY admission types.",
            "readmission_30d_non_elective_proxy",
        ),
    ]
    rows = [
        {
            "outcome_definition": name,
            "definition": definition,
            "n": int(len(cohort)),
            "events": int(cohort[col].sum()),
            "event_rate": float(cohort[col].mean()),
        }
        for name, definition, col in definitions
    ]

    by_type = (
        cohort[cohort["readmission_30d"].eq(1)]["next_admission_type"]
        .fillna("MISSING")
        .value_counts()
        .rename_axis("next_admission_type")
        .reset_index(name="count")
    )
    by_type["percent_among_30d_readmissions"] = by_type["count"] / by_type["count"].sum()

    output_table.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(output_table, index=False)
    by_type.to_csv(output_type_table, index=False)

    lines = [
        f"# {report_title}",
        "",
        "第一版主结局使用 all-cause 30-day readmission。为了给后续 unplanned readmission 做准备，这里比较几个更保守的代理定义。",
        "",
        "| Outcome definition | Events | Event rate | Definition |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['outcome_definition']} | {row['events']:,} | {row['event_rate']:.2%} | {row['definition']} |"
        )
    lines.extend(
        [
            "",
            "## 30 天再入院的 next_admission_type 分布",
            "",
            "| Next admission type | Count | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in by_type.itertuples(index=False):
        lines.append(f"| {row.next_admission_type} | {int(row.count):,} | {row.percent_among_30d_readmissions:.2%} |")
    lines.extend(
        [
            "",
            "解释：正式论文中最好预先定义 planned/unplanned readmission 规则。当前 emergency/urgent 和 non-elective proxy 只是敏感性分析，不替代最终临床定义。",
        ]
    )
    output_report.write_text("\n".join(lines), encoding="utf-8")
    return out
