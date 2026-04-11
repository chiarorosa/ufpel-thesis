#!/usr/bin/env python3
"""Canonical thesis evaluate-pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.runtime import CANONICAL_RUNTIME_FAMILY, evaluate_thesis_run
from thesis.scripts._common import repo_root_from_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical thesis hierarchical pipeline."
    )
    parser.add_argument("--run-name", required=True, help="Run identifier under thesis/runs")
    parser.add_argument(
        "--block-size",
        default="16",
        choices=["8", "16", "32", "64"],
        help="Block size for canonical flow.",
    )
    parser.add_argument(
        "--runtime-family",
        default=CANONICAL_RUNTIME_FAMILY,
        help="Canonical runtime selector.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda", help="Device for evaluation script.")
    parser.add_argument("--python", default=None, help="Python executable for child script.")
    parser.add_argument("--dry-run", action="store_true", help="Plan command without executing.")
    parser.add_argument(
        "--no-docs-gate",
        action="store_true",
        help="Disable documentation consistency gate.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from_script(Path(__file__))

    paths = evaluate_thesis_run(
        repo_root=repo_root,
        run_name=args.run_name,
        block_size=args.block_size,
        runtime_family=args.runtime_family,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        python_executable=args.python,
        dry_run=args.dry_run,
        docs_gate=not args.no_docs_gate,
    )
    print(f"Evaluation completed for run: {paths.run_dir}")


if __name__ == "__main__":
    main()
