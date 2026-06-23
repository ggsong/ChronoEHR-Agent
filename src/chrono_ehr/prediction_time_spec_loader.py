"""Load prediction-time model specs from JSON config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "configs" / "prediction_time_model_specs.json"


def expand_feature_group(group: Any) -> list[str]:
    if isinstance(group, list):
        return [str(item) for item in group]
    if isinstance(group, dict):
        prefix = group["prefix"]
        names = group["names"]
        stats = group["stats"]
        return [f"{prefix}_{name}_{stat}" for name in names for stat in stats]
    raise TypeError(f"Unsupported feature group: {group!r}")


def load_raw_config(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_groups(feature_groups: dict[str, Any], group_names: list[str]) -> list[str]:
    columns: list[str] = []
    for group_name in group_names:
        if group_name not in feature_groups:
            raise KeyError(f"Unknown feature group `{group_name}`")
        columns.extend(expand_feature_group(feature_groups[group_name]))
    return columns


def load_prediction_time_config(study_key: str, spec_path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    raw = load_raw_config(spec_path)
    studies = raw.get("studies", {})
    if study_key not in studies:
        available = ", ".join(studies)
        raise KeyError(f"Unknown prediction-time study `{study_key}`. Available: {available}")

    study = studies[study_key]
    feature_groups = study.get("feature_groups", {})
    specs = []
    for raw_spec in study.get("specs", []):
        spec = dict(raw_spec)
        spec["study_key"] = study_key
        spec["numeric_features"] = resolve_groups(feature_groups, spec.pop("numeric_groups", [])) + [
            str(item) for item in spec.get("numeric_features", [])
        ]
        spec["categorical_features"] = resolve_groups(feature_groups, spec.pop("categorical_groups", [])) + [
            str(item) for item in spec.get("categorical_features", [])
        ]
        specs.append(spec)

    return {
        "cohort_path": study["cohort_path"],
        "outputs": study["outputs"],
        "specs": specs,
        "report": study["report"],
    }
