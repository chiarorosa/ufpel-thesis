#!/usr/bin/env python3
"""Validate expected thesis raw-data flow contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.runtime import validate_expected_raw_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate expected flow: partition_frame -> intra_raw_blocks -> labels/qps."
        )
    )
    parser.add_argument(
        "--raw-root",
        default="thesis/uvg",
        help="Path containing <sequence>/partition_frame_0.txt",
    )
    parser.add_argument(
        "--legacy-base-path",
        required=True,
        help="Path containing intra_raw_blocks/labels/qps",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON report path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    raw_root = (REPO_ROOT / args.raw_root).resolve()
    legacy_base_path = (REPO_ROOT / args.legacy_base_path).resolve()
    output_json = (
        (REPO_ROOT / args.output_json).resolve() if args.output_json else None
    )

    report = validate_expected_raw_flow(
        raw_root=raw_root,
        legacy_base_path=legacy_base_path,
        output_json=output_json,
    )

    print(f"Flow validation ok: {report['ok']}")
    print(f"Active sequences: {len(report['active_sequences'])}")
    print(f"Warnings: {len(report['warnings'])}")
    print(f"Errors: {len(report['errors'])}")

    if report["warnings"]:
        print("Warning details:")
        for warning in report["warnings"]:
            print(f"- {warning}")

    if report["errors"]:
        print("Error details:")
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
