#!/usr/bin/env python3
"""Export thesis partition_frame text into legacy XLSX layout for primary_method.

The generated XLSX is compatible with primary_method/training/rearrange_video.py:
- sheets: 64, 32, 16 (optional 8)
- column B: col (in 4x4 units)
- column C: partition_mode
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


BLOCK_INDEX_TO_SHEET = {
    12: "64",
    9: "32",
    6: "16",
    3: "8",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-file",
        required=True,
        help="Path to partition_frame_<n>.txt",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the legacy XLSX will be written",
    )
    parser.add_argument(
        "--video-base",
        default=None,
        help=(
            "Video basename (without extension) used in output filename. "
            "Default: parent directory name of partition file"
        ),
    )
    parser.add_argument(
        "--include-8x8",
        action="store_true",
        help="Also write sheet '8' from block_size index 3.",
    )
    return parser.parse_args()


def _infer_frame_number(partition_file: Path) -> int:
    match = re.match(r"partition_frame_(\d+)\.txt$", partition_file.name)
    if not match:
        raise ValueError(
            f"partition file must match partition_frame_<n>.txt, got {partition_file.name}"
        )
    return int(match.group(1))


def _load_rows(partition_file: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with open(partition_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) != 7:
                continue
            try:
                order_hint, frame_type, block_size, row_u4, col_u4, mode, qp = (
                    int(value) for value in parts
                )
            except ValueError:
                continue
            if frame_type != 0:
                continue
            rows.append(
                {
                    "order_hint": order_hint,
                    "frame_type": frame_type,
                    "block_size": block_size,
                    "row_u4": row_u4,
                    "col_u4": col_u4,
                    "partition_mode": mode,
                    "qp": qp,
                }
            )
    if not rows:
        raise RuntimeError(f"No valid intra rows found in {partition_file}")
    return rows


def _export_xlsx(
    rows: list[dict[str, int]],
    *,
    output_xlsx: Path,
    include_8x8: bool,
) -> dict[str, int]:
    sheet_counts: dict[str, int] = {}
    selected_sheets = {"64", "32", "16"}
    if include_8x8:
        selected_sheets.add("8")

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for block_idx, sheet_name in BLOCK_INDEX_TO_SHEET.items():
            if sheet_name not in selected_sheets:
                continue
            subset = [row for row in rows if row["block_size"] == block_idx]
            subset.sort(key=lambda item: (item["row_u4"], item["col_u4"], item["order_hint"]))

            frame = pd.DataFrame(
                {
                    "row_u4": [item["row_u4"] for item in subset],
                    "col_u4": [item["col_u4"] for item in subset],
                    "partition_mode": [item["partition_mode"] for item in subset],
                    "qp": [item["qp"] for item in subset],
                    "order_hint": [item["order_hint"] for item in subset],
                }
            )
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_counts[sheet_name] = len(frame)
    return sheet_counts


def main() -> None:
    args = _parse_args()
    partition_file = Path(args.partition_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not partition_file.exists():
        raise FileNotFoundError(f"partition file not found: {partition_file}")

    rows = _load_rows(partition_file)
    qp_values = sorted({row["qp"] for row in rows})
    if len(qp_values) != 1:
        raise RuntimeError(
            "Expected a single QP value in partition file for legacy filename, "
            f"got {qp_values}"
        )
    qp_value = qp_values[0]

    frame_number = _infer_frame_number(partition_file)
    video_base = args.video_base or partition_file.parent.name
    output_xlsx = output_dir / f"{video_base}-{qp_value}-{frame_number}.xlsx"

    sheet_counts = _export_xlsx(
        rows,
        output_xlsx=output_xlsx,
        include_8x8=args.include_8x8,
    )

    print(f"Wrote: {output_xlsx}")
    for sheet_name in sorted(sheet_counts.keys(), key=lambda value: int(value)):
        print(f"  sheet {sheet_name}: {sheet_counts[sheet_name]} rows")


if __name__ == "__main__":
    main()
