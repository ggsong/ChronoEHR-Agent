"""Load feature-window specs for ChronoEHR-Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FEATURE_WINDOW_SPEC = Path(__file__).resolve().parents[2] / "configs" / "feature_window_specs.json"


def load_feature_window_spec(path: Path = DEFAULT_FEATURE_WINDOW_SPEC) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_window(spec: dict[str, Any], window_name: str) -> dict[str, Any]:
    windows = spec.get("windows", {})
    if window_name not in windows:
        available = ", ".join(windows)
        raise KeyError(f"Unknown feature window `{window_name}`. Available: {available}")
    return windows[window_name]


def add_window_end(
    df: pd.DataFrame,
    window_name: str,
    output_col: str = "window_end",
    spec: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach a concrete window-end timestamp using the configured window name."""
    spec = spec or load_feature_window_spec()
    get_window(spec, window_name)
    out = df.copy()
    if window_name == "admission_baseline":
        out[output_col] = out["admittime"]
    elif window_name == "first_24h":
        end = out["admittime"] + pd.Timedelta(hours=24)
        out[output_col] = pd.concat([end, out["dischtime"]], axis=1).min(axis=1)
    elif window_name == "admission_to_discharge":
        out[output_col] = out["dischtime"]
    elif window_name == "followup_30d":
        out[output_col] = out["dischtime"] + pd.Timedelta(days=30)
    else:
        raise KeyError(f"Window `{window_name}` is defined but has no concrete implementation yet")
    return out


def available_time_from_source(chunk: pd.DataFrame, source: dict[str, Any]) -> pd.Series:
    available = source.get("available_time", {})
    primary = available.get("primary")
    fallback = available.get("fallback")
    if not primary:
        raise KeyError("Feature source missing available_time.primary")
    primary_values = pd.to_datetime(chunk[primary], errors="coerce")
    if fallback and fallback in chunk.columns:
        fallback_values = pd.to_datetime(chunk[fallback], errors="coerce")
        return primary_values.where(primary_values.notna(), fallback_values)
    return primary_values


def source_spec(spec: dict[str, Any], source_name: str) -> dict[str, Any]:
    sources = spec.get("feature_sources", {})
    if source_name not in sources:
        available = ", ".join(sources)
        raise KeyError(f"Unknown feature source `{source_name}`. Available: {available}")
    return sources[source_name]


def vital_itemid_map(spec: dict[str, Any]) -> dict[int, str]:
    source = spec["feature_sources"]["vital_signs"]
    return {int(itemid): str(name) for itemid, name in source.get("itemids", {}).items()}


def vital_names(spec: dict[str, Any]) -> list[str]:
    names = []
    for name in vital_itemid_map(spec).values():
        if name not in names:
            names.append(name)
    return names


def vital_plausible_ranges(spec: dict[str, Any]) -> dict[str, tuple[float, float]]:
    ranges = {}
    for name, bounds in spec["feature_sources"]["vital_signs"].get("plausible_ranges", {}).items():
        if len(bounds) != 2:
            continue
        ranges[str(name)] = (float(bounds[0]), float(bounds[1]))
    return ranges
