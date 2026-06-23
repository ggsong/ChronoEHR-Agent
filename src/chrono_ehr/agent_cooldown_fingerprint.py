#!/usr/bin/env python3
"""Build stable fingerprints for safe-auto Agent cooldown decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/agent_cooldown_fingerprint.json")
DEFAULT_COOLDOWN_INPUTS = [
    "src/chrono_ehr/agent_cooldown_fingerprint.py",
    "src/chrono_ehr/validate_agent_cooldown_fingerprint.py",
    "src/chrono_ehr/agent_next_task_planner.py",
    "src/chrono_ehr/agent_task_queue.py",
    "src/chrono_ehr/agent_task_queue_runner.py",
    "src/chrono_ehr/agent_task_router.py",
    "src/chrono_ehr/build_agent_state.py",
    "src/chrono_ehr/validate_agent_state.py",
    "src/chrono_ehr/agent_status_card.py",
    "src/chrono_ehr/validate_agent_status_card.py",
    "src/chrono_ehr/validate_agent_next_tasks.py",
    "src/chrono_ehr/validate_agent_task_queue.py",
    "src/chrono_ehr/validate_agent_task_queue_execution.py",
    "configs/agent_action_catalog.json",
    "configs/agent_entrypoints.json",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cooldown_input_paths(project_root: Path) -> list[str]:
    config_path = project_root / DEFAULT_CONFIG
    config = read_json(config_path)
    inputs = config.get("fingerprint_inputs", DEFAULT_COOLDOWN_INPUTS)
    if not isinstance(inputs, list) or not inputs:
        return DEFAULT_COOLDOWN_INPUTS
    return [str(item) for item in inputs]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cooldown_fingerprint(project_root: Path, scenario_id: str, task: str, command: str) -> dict[str, str]:
    digest = hashlib.sha256()
    digest.update(f"scenario_id={scenario_id}\n".encode("utf-8"))
    digest.update(f"task={task}\n".encode("utf-8"))
    digest.update(f"command={command}\n".encode("utf-8"))
    config_path = project_root / DEFAULT_CONFIG
    if config_path.exists():
        digest.update(f"config={DEFAULT_CONFIG}:{file_digest(config_path)}\n".encode("utf-8"))
    else:
        digest.update(f"missing_config={DEFAULT_CONFIG}\n".encode("utf-8"))
    present = []
    missing = []
    for relative in cooldown_input_paths(project_root):
        path = project_root / relative
        if path.exists() and path.is_file():
            file_hash = file_digest(path)
            digest.update(f"file={relative}:{file_hash}\n".encode("utf-8"))
            present.append(relative)
        else:
            digest.update(f"missing={relative}\n".encode("utf-8"))
            missing.append(relative)
    return {
        "cooldown_fingerprint": digest.hexdigest(),
        "cooldown_fingerprint_config": str(DEFAULT_CONFIG),
        "cooldown_fingerprint_inputs": ",".join(present),
        "cooldown_missing_inputs": ",".join(missing),
    }
