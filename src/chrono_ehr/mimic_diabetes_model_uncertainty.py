#!/usr/bin/env python3
"""Bootstrap confidence intervals and fixed-burden summaries for test predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROJECT = Path(__file__).resolve().parents[2]
RNG_SEED = 20260619
N_BOOTSTRAP = 500
ALERT_BURDENS = [0.05, 0.10, 0.20]


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


def bootstrap_ci(y: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    n = len(y)
    aurocs = []
    auprcs = []
    briers = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        score_b = score[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        aurocs.append(auroc(y_b, score_b))
        auprcs.append(average_precision(y_b, score_b))
        briers.append(float(np.mean((score_b - y_b) ** 2)))

    def interval(values: list[float], q: float) -> float:
        return float(np.nanquantile(values, q))

    return {
        "AUROC_lower": interval(aurocs, 0.025),
        "AUROC_upper": interval(aurocs, 0.975),
        "AUPRC_lower": interval(auprcs, 0.025),
        "AUPRC_upper": interval(auprcs, 0.975),
        "Brier_lower": interval(briers, 0.025),
        "Brier_upper": interval(briers, 0.975),
        "bootstrap_replicates": len(aurocs),
    }


def fixed_burden_rows(feature_set: str, part: pd.DataFrame) -> list[dict]:
    rows = []
    sorted_part = part.sort_values("predicted_risk", ascending=False).reset_index(drop=True)
    total_events = int(sorted_part["readmission_30d"].sum())
    n = len(sorted_part)
    for burden in ALERT_BURDENS:
        k = max(1, int(round(n * burden)))
        flagged = sorted_part.iloc[:k]
        events_flagged = int(flagged["readmission_30d"].sum())
        rows.append(
            {
                "feature_set": feature_set,
                "alert_burden": burden,
                "flagged_n": k,
                "flagged_event_count": events_flagged,
                "flagged_event_rate_ppv": events_flagged / k,
                "capture_rate_sensitivity": events_flagged / total_events if total_events else float("nan"),
                "baseline_event_rate": total_events / n if n else float("nan"),
            }
        )
    return rows


def write_report(ci: pd.DataFrame, burden: pd.DataFrame, report_path: Path) -> None:
    ci_lines = [
        "| Feature set | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for row in ci.itertuples(index=False):
        ci_lines.append(
            f"| {row.feature_set} | {row.AUROC:.4f} ({row.AUROC_lower:.4f}-{row.AUROC_upper:.4f}) | "
            f"{row.AUPRC:.4f} ({row.AUPRC_lower:.4f}-{row.AUPRC_upper:.4f}) | "
            f"{row.Brier_score:.4f} ({row.Brier_lower:.4f}-{row.Brier_upper:.4f}) |"
        )

    burden_lines = [
        "| Feature set | Alert burden | Flagged N | PPV among flagged | Capture rate | Baseline event rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in burden.itertuples(index=False):
        burden_lines.append(
            f"| {row.feature_set} | {row.alert_burden:.0%} | {int(row.flagged_n)} | "
            f"{row.flagged_event_rate_ppv:.2%} | {row.capture_rate_sensitivity:.2%} | {row.baseline_event_rate:.2%} |"
        )

    text = f"""# MIMIC 糖尿病模型不确定性与固定预警负担分析

## Bootstrap 95% CI

基于 test set 预测结果进行 {N_BOOTSTRAP} 次 bootstrap。该分析不重新训练模型，只评估 test set 性能估计的不确定性。

{chr(10).join(ci_lines)}

## Fixed Alert Burden

按预测风险从高到低排序，分别查看 top 5%、10%、20% 住院作为高风险提醒时的事件比例和捕获率。

{chr(10).join(burden_lines)}

## 解释

固定预警负担表适合向项目评阅者解释模型在真实工作流中的含义：如果只关注预测风险最高的一小部分出院患者，能捕获多少 30 天再入院事件，以及这部分人群中的再入院比例是多少。
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    tables_dir = PROJECT / "outputs" / "tables"
    reports_dir = PROJECT / "outputs" / "reports"
    predictions = pd.read_csv(tables_dir / "mimic_diabetes_test_predictions.csv")
    performance = pd.read_csv(tables_dir / "mimic_diabetes_model_performance.csv")
    test_perf = performance[performance["split"].eq("test")].set_index("feature_set")
    rng = np.random.default_rng(RNG_SEED)

    ci_rows = []
    burden_rows = []
    for feature_set, part in predictions.groupby("feature_set", sort=True):
        y = part["readmission_30d"].astype(int).to_numpy()
        score = part["predicted_risk"].to_numpy()
        base = test_perf.loc[feature_set]
        row = {
            "feature_set": feature_set,
            "n": int(len(part)),
            "events": int(y.sum()),
            "AUROC": float(base["AUROC"]),
            "AUPRC": float(base["AUPRC"]),
            "Brier_score": float(base["Brier_score"]),
        }
        row.update(bootstrap_ci(y, score, rng))
        ci_rows.append(row)
        burden_rows.extend(fixed_burden_rows(feature_set, part))

    ci = pd.DataFrame(ci_rows)
    burden = pd.DataFrame(burden_rows)
    ci.to_csv(tables_dir / "mimic_diabetes_model_performance_bootstrap_ci.csv", index=False)
    burden.to_csv(tables_dir / "mimic_diabetes_fixed_alert_burden.csv", index=False)
    write_report(ci, burden, reports_dir / "mimic_diabetes_model_uncertainty_report.md")

    print(ci[["feature_set", "AUROC", "AUROC_lower", "AUROC_upper", "AUPRC", "AUPRC_lower", "AUPRC_upper"]].to_string(index=False))
    print()
    print(burden.to_string(index=False))


if __name__ == "__main__":
    main()

