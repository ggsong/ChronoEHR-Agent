#!/usr/bin/env python3
"""Run feature-time audit for the heart-failure demo config."""

from __future__ import annotations

import argparse
from pathlib import Path

from audit_feature_time_map import audit, read_feature_time_map, write_csv, write_report
from validate_study_config import PROJECT, load_yaml_with_ruby


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "heart_failure_mimic_readmission.yaml")
    parser.add_argument("--feature-map", type=Path, default=PROJECT / "docs" / "mimic_heart_failure_feature_time_map.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_with_ruby(args.config)
    feature_map = read_feature_time_map(args.feature_map)
    rows, summary = audit(config, feature_map)

    table_path = args.project_root / "outputs" / "tables" / "mimic_heart_failure_feature_time_audit.csv"
    report_path = args.project_root / "outputs" / "reports" / "mimic_heart_failure_feature_time_audit_report.md"
    write_csv(rows, table_path)
    write_report(rows, summary, report_path)

    failing_enabled = [
        row
        for row in rows
        if row["audit_status"] in {"fail_leakage_risk", "needs_manual_review"}
        and not row["feature_role"].startswith("forbidden")
    ]
    print(f"Heart failure feature time audit checked {summary['variables_checked']} variables")
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")
    if failing_enabled:
        print(f"Found {len(failing_enabled)} enabled feature problems")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
