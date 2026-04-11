#!/usr/bin/env python3
"""Validate fresh-start cleanup safety constraints."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.runtime import validate_cleanup_safety


def main() -> None:
    stats = validate_cleanup_safety(repo_root=REPO_ROOT, include_caches=True)
    print("Cleanup safety validation: PASS")
    print(f"Planned paths: {stats['planned_paths']}")
    print(f"Protected skips: {stats['protected_skips']}")
    print(f"Violations: {stats['violations']}")


if __name__ == "__main__":
    main()
