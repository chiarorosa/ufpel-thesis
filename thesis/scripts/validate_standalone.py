#!/usr/bin/env python3
"""Validate standalone thesis reference guards."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.runtime import validate_standalone_reference_contract


def main() -> None:
    result = validate_standalone_reference_contract(REPO_ROOT)
    print("Standalone reference guard: PASS")
    print(f"Legacy reference matches: {result['legacy_reference_matches']}")
    print(f"Numbered surface matches: {result['numbered_surface_matches']}")


if __name__ == "__main__":
    main()
