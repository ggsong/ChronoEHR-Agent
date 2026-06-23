#!/usr/bin/env python3
"""Small config helpers used by local ChronoEHR runners.

The project intentionally avoids adding a YAML dependency for the first local
MVP. These helpers parse only the tiny subset we need for stable runner logic:
cohort code-rule prefix lists such as `diabetes_code_rules.icd9_prefixes`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _clean_value(value: str) -> str:
    return value.strip().strip("\"'")


def _parse_inline_list(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value:
        return ()
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected inline list, got: {value}")
    return tuple(str(item).strip() for item in parsed)


def load_cohort_code_rules(
    config_path: Path,
    rule_key: str,
    fallback_icd9: tuple[str, ...],
    fallback_icd10: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load ICD prefix rules from a study YAML config with fallback constants."""
    if not config_path.exists():
        return fallback_icd9, fallback_icd10

    lines = config_path.read_text(encoding="utf-8").splitlines()
    start_index = None
    block_indent = 0
    rule_pattern = re.compile(rf"^(\s*){re.escape(rule_key)}\s*:\s*(?:#.*)?$")
    for index, line in enumerate(lines):
        if rule_pattern.match(line):
            start_index = index + 1
            block_indent = _indent(line)
            break
    if start_index is None:
        return fallback_icd9, fallback_icd10

    values: dict[str, list[str]] = {"icd9_prefixes": [], "icd10_prefixes": []}
    current_key: str | None = None
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent(line)
        if indent <= block_indent:
            break
        key_match = re.match(r"^(icd9_prefixes|icd10_prefixes)\s*:\s*(.*)$", stripped)
        if key_match:
            current_key = key_match.group(1)
            remainder = key_match.group(2).strip()
            if remainder:
                values[current_key].extend(_parse_inline_list(remainder))
            continue
        if current_key and stripped.startswith("-"):
            values[current_key].append(_clean_value(stripped[1:]))

    icd9 = tuple(value for value in values["icd9_prefixes"] if value) or fallback_icd9
    icd10 = tuple(value for value in values["icd10_prefixes"] if value) or fallback_icd10
    return icd9, icd10
