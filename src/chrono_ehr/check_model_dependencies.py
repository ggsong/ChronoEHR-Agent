#!/usr/bin/env python3
"""Check optional model dependencies for ChronoEHR-Agent."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from mimic_diabetes_baseline import DEFAULT_PROJECT


DEPENDENCIES = [
    ("scikit-learn", "sklearn", "Random forest baseline"),
    ("xgboost", "xgboost", "Gradient boosting baseline"),
    ("lightgbm", "lightgbm", "Gradient boosting baseline"),
]


def check_dependency(import_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - user-facing report
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__version__", "unknown"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for package, import_name, use_case in DEPENDENCIES:
        installed, detail = check_dependency(import_name)
        rows.append(
            {
                "package": package,
                "import_name": import_name,
                "installed": installed,
                "detail": detail,
                "use_case": use_case,
            }
        )

    reports = args.project_root / "outputs" / "reports"
    tables = args.project_root / "outputs" / "tables"
    reports.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(tables / "model_dependency_status.csv", index=False)

    lines = [
        "# Model Dependency Status",
        "",
        "| Package | Import | Installed | Detail | Use case |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['package']} | `{row['import_name']}` | {str(row['installed'])} | {row['detail']} | {row['use_case']} |"
        )
    lines.extend(
        [
            "",
            "解释：ChronoEHR-Agent 当前不依赖这些包即可完成 logistic baseline、leakage audit 和报告生成。"
            "如果以后要运行 Random Forest、XGBoost 或 LightGBM baseline，需要先在当前 Python 环境安装对应包。",
        ]
    )
    (reports / "model_dependency_status.md").write_text("\n".join(lines), encoding="utf-8")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
