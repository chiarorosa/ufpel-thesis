"""Shared CLI helpers for thesis orchestration scripts."""

from __future__ import annotations

from pathlib import Path


def repo_root_from_script(script_path: Path) -> Path:
    return script_path.resolve().parents[2]


def parse_sequences(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    flattened: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                flattened.append(token)
    return flattened or None
