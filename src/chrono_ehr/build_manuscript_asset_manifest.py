#!/usr/bin/env python3
"""Build a manifest of report, table, figure, and Word assets for writing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


ASSETS = [
    {
        "asset_type": "report",
        "role": "主文草稿",
        "file": "outputs/reports/chronic_disease_methods_results_draft.md",
        "use_case": "中文 Methods/Results 主草稿，覆盖四个 MIMIC 慢病队列。",
    },
    {
        "asset_type": "report",
        "role": "英文简版草稿",
        "file": "outputs/reports/chronic_disease_methods_results_english_brief.md",
        "use_case": "英文 brief，可作为后续英文论文正文的初稿材料。",
    },
    {
        "asset_type": "word",
        "role": "英文审阅版",
        "file": "outputs/reports/ChronoEHR_Methods_Results_English_Brief.docx",
        "use_case": "可直接打开审阅的英文 brief DOCX，不是最终投稿格式。",
    },
    {
        "asset_type": "word",
        "role": "中文完整 Word",
        "file": "outputs/reports/ChronoEHR_Methods_Results_Draft.docx",
        "use_case": "配置驱动的中文完整主文 Word 草稿。",
    },
    {
        "asset_type": "word",
        "role": "中文简版 Word",
        "file": "outputs/reports/ChronoEHR_Methods_Results_Brief.docx",
        "use_case": "适合快速给自己或合作者看的中文简版。",
    },
    {
        "asset_type": "supplement",
        "role": "补充材料草稿",
        "file": "outputs/reports/chronic_disease_supplementary_appendix.md",
        "use_case": "补充表 S1-S12 的文字入口。",
    },
    {
        "asset_type": "word",
        "role": "补充材料 Word",
        "file": "outputs/reports/ChronoEHR_Supplementary_Appendix.docx",
        "use_case": "可审阅的补充材料 Word 草稿。",
    },
    {
        "asset_type": "table",
        "role": "Table 1",
        "file": "outputs/tables/chronic_disease_manuscript_table1.csv",
        "use_case": "四个慢病队列的基线/队列描述主表。",
    },
    {
        "asset_type": "table",
        "role": "预测时间主表",
        "file": "outputs/tables/chronic_disease_manuscript_prediction_time_table.csv",
        "use_case": "admission、24h、discharge prediction-time 的主结果表。",
    },
    {
        "asset_type": "table",
        "role": "模型对照主表",
        "file": "outputs/tables/chronic_disease_manuscript_model_table.csv",
        "use_case": "logistic/RF/gradient boosting 等传统 baseline 对照。",
    },
    {
        "asset_type": "table",
        "role": "阈值/alert burden",
        "file": "outputs/tables/chronic_disease_threshold_analysis.csv",
        "use_case": "固定提醒比例下的 sensitivity、PPV、NPV 等。",
    },
    {
        "asset_type": "table",
        "role": "Decision curve",
        "file": "outputs/tables/chronic_disease_decision_curve.csv",
        "use_case": "不同阈值下的 net benefit。",
    },
    {
        "asset_type": "table",
        "role": "亚组分析",
        "file": "outputs/tables/chronic_disease_subgroup_performance.csv",
        "use_case": "年龄、性别等亚组中的模型表现检查。",
    },
    {
        "asset_type": "table",
        "role": "leakage 审计",
        "file": "outputs/tables/prediction_time_leakage_gate.csv",
        "use_case": "时间点和未来信息风险的规则化审计表。",
    },
    {
        "asset_type": "table",
        "role": "leakage action items",
        "file": "outputs/tables/prediction_time_leakage_gate_action_items.csv",
        "use_case": "需要在 Methods/Limitations 中说明的 leakage 风险处理。",
    },
    {
        "asset_type": "figure",
        "role": "预测时间 AUROC 图",
        "file": "outputs/figures/chronic_disease_prediction_time_auroc.png",
        "use_case": "展示 prediction-time 对 AUROC 的影响。",
    },
    {
        "asset_type": "figure",
        "role": "预测时间 AUPRC 图",
        "file": "outputs/figures/chronic_disease_prediction_time_auprc.png",
        "use_case": "展示 prediction-time 对 AUPRC 的影响。",
    },
    {
        "asset_type": "figure",
        "role": "Decision curve 图",
        "file": "outputs/figures/chronic_disease_decision_curve_net_benefit.png",
        "use_case": "展示不同阈值下的 net benefit。",
    },
    {
        "asset_type": "figure",
        "role": "亚组 AUROC 图",
        "file": "outputs/figures/chronic_disease_subgroup_mean_auroc.png",
        "use_case": "展示亚组间平均 AUROC 差异。",
    },
    {
        "asset_type": "external",
        "role": "CDSL 外部方法验证",
        "file": "outputs/reports/cdsl_temporal_benchmark_report.md",
        "use_case": "说明 CDSL 是 temporal benchmark 补充，不是 MIMIC 慢病再入院外部验证。",
    },
    {
        "asset_type": "external",
        "role": "外部数据 readiness",
        "file": "outputs/reports/external_benchmark_readiness_summary.md",
        "use_case": "汇总 CDSL/eICU/CHARLS 当前可用性。",
    },
    {
        "asset_type": "audit",
        "role": "总交付审计",
        "file": "outputs/reports/delivery_readiness_audit.md",
        "use_case": "一键确认当前 demo 交付材料是否齐全。",
    },
    {
        "asset_type": "audit",
        "role": "英文 brief 文本审计",
        "file": "outputs/reports/english_brief_quality_audit.md",
        "use_case": "检查英文 brief 是否包含研究工具边界和必要章节。",
    },
    {
        "asset_type": "audit",
        "role": "英文 DOCX 审计",
        "file": "outputs/reports/english_brief_docx_audit.md",
        "use_case": "检查英文 DOCX 和渲染页是否可审阅。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def build_manifest(project_root: Path) -> pd.DataFrame:
    rows = []
    for asset in ASSETS:
        path = project_root / asset["file"]
        rows.append(
            {
                **asset,
                "exists": path.exists() and path.stat().st_size > 0,
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = ["asset_type", "role", "exists", "file", "use_case"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def write_report(project_root: Path, manifest: pd.DataFrame) -> Path:
    output = project_root / "outputs" / "reports" / "manuscript_asset_manifest.md"
    missing = manifest[~manifest["exists"]]
    by_type = manifest.groupby("asset_type")["exists"].agg(["count", "sum"]).reset_index()
    summary_lines = [
        f"- {row['asset_type']}: {int(row['sum'])}/{int(row['count'])} files present"
        for _, row in by_type.iterrows()
    ]
    text = f"""# Manuscript Asset Manifest

- Present assets: {int(manifest["exists"].sum())}/{len(manifest)}
- Missing assets: {len(missing)}
- Boundary: these assets support research writing and reproducibility review, not medical advice or clinical deployment.

## By Type

{chr(10).join(summary_lines)}

## Asset Table

{markdown_table(manifest)}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.project_root)
    table_path = args.project_root / "outputs" / "tables" / "manuscript_asset_manifest.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(table_path, index=False)
    report = write_report(args.project_root, manifest)
    print(f"Wrote {report}")
    print(f"Assets present: {int(manifest['exists'].sum())}/{len(manifest)}")


if __name__ == "__main__":
    main()
