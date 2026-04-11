#!/usr/bin/env python3
"""Compatibility forwarder from legacy thesis commands to canonical entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    print("[forward]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward legacy thesis-relevant command patterns."
    )
    parser.add_argument(
        "legacy_command",
        choices=["prepare", "prepare_stage3", "train", "evaluate"],
        help="Legacy command identifier.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--block-size", default="16", choices=["8", "16", "32", "64"])
    parser.add_argument("--raw-root", default="thesis/uvg")
    parser.add_argument("--legacy-base-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.legacy_command in {"prepare", "prepare_stage3"}:
        forward = [
            "thesis/scripts/prepare_data.py",
            "--run-name",
            args.run_name,
            "--block-size",
            args.block_size,
            "--raw-root",
            args.raw_root,
            "--require-intra-raw-blocks",
        ]
        if args.legacy_base_path:
            forward.extend(["--legacy-base-path", args.legacy_base_path])
        else:
            forward.append("--skip-legacy-generation")
        if args.dry_run:
            forward.append("--dry-run")
        _run(forward)
        return

    if args.legacy_command == "train":
        forward = [
            "thesis/scripts/train_pipeline.py",
            "--run-name",
            args.run_name,
            "--block-size",
            args.block_size,
            "--device",
            args.device,
        ]
        if args.dry_run:
            forward.append("--dry-run")
        _run(forward)
        return

    forward = [
        "thesis/scripts/evaluate_pipeline.py",
        "--run-name",
        args.run_name,
        "--block-size",
        args.block_size,
        "--split",
        args.split,
        "--device",
        args.device,
    ]
    if args.dry_run:
        forward.append("--dry-run")
    _run(forward)


if __name__ == "__main__":
    main()
