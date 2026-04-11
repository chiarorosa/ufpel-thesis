#!/usr/bin/env python3
"""Canonical thesis train-pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.runtime import CANONICAL_RUNTIME_FAMILY, train_thesis_run
from thesis.scripts._common import repo_root_from_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train canonical thesis Stage1/2/3 models."
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
    parser.add_argument("--device", default="cuda", help="Device for training scripts.")
    parser.add_argument("--epochs-stage2", type=int, default=100)
    parser.add_argument("--epochs-stage3-rect", type=int, default=30)
    parser.add_argument("--epochs-stage3-ab-binary", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=None, help="Python executable for child scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Plan commands without executing.")
    parser.add_argument(
        "--no-docs-gate",
        action="store_true",
        help="Disable documentation consistency gate.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from_script(Path(__file__))

    paths = train_thesis_run(
        repo_root=repo_root,
        run_name=args.run_name,
        block_size=args.block_size,
        runtime_family=args.runtime_family,
        device=args.device,
        epochs_stage2=args.epochs_stage2,
        epochs_stage3_rect=args.epochs_stage3_rect,
        epochs_stage3_ab_binary=args.epochs_stage3_ab_binary,
        batch_size=args.batch_size,
        seed=args.seed,
        python_executable=args.python,
        dry_run=args.dry_run,
        docs_gate=not args.no_docs_gate,
    )
    print(f"Train completed for run: {paths.run_dir}")


if __name__ == "__main__":
    main()
