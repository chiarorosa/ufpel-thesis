#!/usr/bin/env python3
"""Clean generated thesis runtime artifacts for a fresh start."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.runtime import cleanup_thesis_outputs
from thesis.scripts._common import repo_root_from_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean generated thesis artifacts (runs/logs/cleanup reports/caches) "
            "without touching source code, docs, or raw inputs."
        )
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional specific run under thesis/runs to clean (default: all runs).",
    )
    parser.add_argument(
        "--no-caches",
        action="store_true",
        help="Skip __pycache__ and *.pyc cleanup.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform deletion. Default is dry-run preview.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest output path (relative to repo root).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from_script(Path(__file__))
    manifest_path = (repo_root / args.manifest).resolve() if args.manifest else None

    report = cleanup_thesis_outputs(
        repo_root=repo_root,
        run_name=args.run_name,
        include_caches=not args.no_caches,
        execute=args.execute,
        manifest_path=manifest_path,
    )

    print(f"Mode: {report['mode']}")
    print(f"Planned paths: {len(report['planned_paths'])}")
    print(f"Removed paths: {len(report['removed_paths'])}")
    print(f"Skipped paths: {len(report['skipped_paths'])}")
    print(f"Missing paths: {len(report['missing_paths'])}")
    print(f"Manifest: {report['manifest_path']}")


if __name__ == "__main__":
    main()
