#!/usr/bin/env python3
"""Create CKD-related lab itemid mapping from MIMIC-IV d_labitems."""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MIMIC_ROOT = Path(os.environ.get("MIMIC_IV_ROOT", "~/mimic-iv-3.1")).expanduser()

TARGET_LABS = {
    "creatinine": ["creatinine"],
    "bun": ["urea nitrogen"],
    "potassium": ["potassium"],
    "sodium": ["sodium"],
    "bicarbonate": ["bicarbonate"],
    "hemoglobin": ["hemoglobin"],
    "chloride": ["chloride"],
    "phosphate": ["phosphate"],
    "magnesium": ["magnesium"],
    "albumin": ["albumin"],
}

FIRST_PASS_ITEMIDS = {
    "albumin": {50862, 52022, 53085},
    "bicarbonate": {50803, 50882},
    "bun": {51006, 52647},
    "chloride": {50806, 50902, 52434, 52535},
    "creatinine": {50912, 52024, 52546},
    "hemoglobin": {50811, 51222, 51640},
    "magnesium": {50960},
    "phosphate": {50970},
    "potassium": {50822, 50971, 52452, 52610},
    "sodium": {50824, 50983, 52455, 52623},
}


def classify_lab(label: str) -> str | None:
    text = str(label).lower()
    for lab, terms in TARGET_LABS.items():
        if any(term in text for term in terms):
            return lab
    return None


def load_mapping(mimic_root: Path) -> pd.DataFrame:
    path = mimic_root / "hosp" / "d_labitems.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    df["target_lab"] = df["label"].map(classify_lab)
    df = df[df["target_lab"].notna()].copy()

    df["recommended_first_pass"] = [
        int(itemid) in FIRST_PASS_ITEMIDS.get(str(target_lab), set())
        for itemid, target_lab in zip(df["itemid"], df["target_lab"])
    ]
    return df[["target_lab", "itemid", "label", "fluid", "category", "recommended_first_pass"]].sort_values(
        ["target_lab", "recommended_first_pass", "itemid"],
        ascending=[True, False, True],
    )


def write_report(mapping: pd.DataFrame, output: Path) -> None:
    lines = ["| Target lab | Recommended itemids | Notes |", "|---|---|---|"]
    for lab, part in mapping.groupby("target_lab", sort=True):
        recommended = part[part["recommended_first_pass"]]
        itemids = ", ".join(str(int(itemid)) for itemid in recommended["itemid"])
        if not itemids:
            itemids = "needs manual review"
        labels = "; ".join(
            f"{int(row.itemid)} {row.label} ({row.fluid}, {row.category})"
            for row in recommended.itertuples(index=False)
        )
        lines.append(f"| {lab} | {itemids} | {labels or 'No first-pass blood item selected.'} |")

    text = f"""# CKD Lab Itemid Mapping

这个文件从本地 MIMIC-IV `hosp/d_labitems.csv.gz` 自动筛选 CKD 相关候选化验 itemid。

## First-Pass Mapping

{chr(10).join(lines)}

## 使用原则

- 第一版使用脚本内 `FIRST_PASS_ITEMIDS` 白名单，避免把 HbA1c、methemoglobin、尿液或体液项目误当作 CKD 首轮特征。
- 尿液、胸腹水、关节液、CSF、stool 等项目保留在 CSV 中，但默认不进入第一版 CKD readmission demo。
- CKD demo 首轮建议优先使用 creatinine、BUN、potassium、sodium、bicarbonate、hemoglobin。
- 正式论文前需要人工复核 itemid 和单位。
"""
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = args.project_root / "outputs" / "tables"
    reports = args.project_root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mimic_root)
    table_path = tables / "mimic_ckd_lab_itemid_mapping.csv"
    report_path = reports / "mimic_ckd_lab_itemid_mapping_report.md"
    mapping.to_csv(table_path, index=False)
    write_report(mapping, report_path)
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
