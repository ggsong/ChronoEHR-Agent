#!/usr/bin/env python3
"""Audit the public ChronoEHR-Agent release surface."""

from __future__ import annotations

import argparse
from pathlib import Path


PUBLIC_DIRS = ["configs", "src", "scripts", ".github"]
PUBLIC_FILES = [
    "README.md",
    ".env.example",
    ".gitignore",
    "requirements.txt",
    "docs/INSTALL.md",
    "docs/SYNTHETIC_DEMO.md",
    "docs/RELATED_WORK.md",
    "docs/GITHUB_RELEASE_AUDIT.md",
    "docs/RELEASE_STRUCTURE.md",
]
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".toml"}
FORBIDDEN_SNIPPETS = [
    "/Users/" + "gg",
    "codex-runtimes/" + "codex-primary-runtime",
]
REQUIRED_GITIGNORE_PATTERNS = ["data/", "outputs/", "references/", ".env"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def iter_public_text_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in PUBLIC_FILES:
        path = project_root / relative
        if path.exists():
            files.append(path)
    for dirname in PUBLIC_DIRS:
        root = project_root / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return sorted(set(files))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    failures: list[str] = []

    gitignore = project_root / ".gitignore"
    if not gitignore.exists():
        failures.append("Missing .gitignore")
    else:
        gitignore_text = gitignore.read_text(encoding="utf-8")
        for pattern in REQUIRED_GITIGNORE_PATTERNS:
            if pattern not in gitignore_text:
                failures.append(f".gitignore missing required pattern: {pattern}")

    for path in iter_public_text_files(project_root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                failures.append(f"Forbidden local snippet `{snippet}` in {path.relative_to(project_root)}")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        raise SystemExit(f"Release audit failed with {len(failures)} issue(s).")

    print("Release audit PASS: public files avoid local absolute paths and controlled-data directories are ignored.")


if __name__ == "__main__":
    main()
