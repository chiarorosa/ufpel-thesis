#!/usr/bin/env python3
"""Render visual sample images from primary_method rearrange raw outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARTITION_ID_TO_NAME = {
    0: "PARTITION_NONE",
    1: "PARTITION_HORZ",
    2: "PARTITION_VERT",
    3: "PARTITION_SPLIT",
    4: "PARTITION_HORZ_A",
    5: "PARTITION_HORZ_B",
    6: "PARTITION_VERT_A",
    7: "PARTITION_VERT_B",
    8: "PARTITION_HORZ_4",
    9: "PARTITION_VERT_4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", required=True, help="Legacy xlsx path")
    parser.add_argument("--raw-dir", required=True, help="Dir containing *_raw_*.txt files")
    parser.add_argument("--output-dir", required=True, help="Destination dir for images")
    parser.add_argument("--max-per-label", type=int, default=10)
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg", "jpeg", "tif", "tiff"],
        default="png",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=10,
        help="Source bit depth used in rearrange raw files (default: 10).",
    )
    return parser.parse_args()


def _select_center_indices(indices: list[int], rows: np.ndarray, cols: np.ndarray, k: int) -> list[int]:
    if len(indices) <= k:
        return indices
    center_row = (rows.min() + rows.max()) / 2.0
    center_col = (cols.min() + cols.max()) / 2.0
    ranked = sorted(
        indices,
        key=lambda idx: ((rows[idx] - center_row) ** 2 + (cols[idx] - center_col) ** 2, idx),
    )
    return ranked[:k]


def _to_display_u8(blocks_f64: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth <= 0:
        raise ValueError(f"bit_depth must be positive, got {bit_depth}")
    max_value = float((1 << bit_depth) - 1)
    clipped = np.clip(blocks_f64, 0.0, max_value)
    scaled = np.rint((clipped * 255.0) / max_value).astype(np.uint8)
    return scaled


def main() -> None:
    args = parse_args()
    xlsx = Path(args.xlsx).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    base = xlsx.stem

    for block_size in (64, 32, 16):
        sheet = str(block_size)
        frame = pd.read_excel(xlsx, sheet_name=sheet, usecols="A:C")
        # Expected by export script: A=row_u4, B=col_u4, C=partition_mode
        rows = frame.iloc[:, 0].to_numpy(dtype=np.int64)
        cols = frame.iloc[:, 1].to_numpy(dtype=np.int64)
        labels = frame.iloc[:, 2].to_numpy(dtype=np.int64)

        raw_path = raw_dir / f"{base}_raw_{block_size}.txt"
        raw = np.fromfile(raw_path, dtype=np.float64)
        blocks_f64 = raw.reshape(-1, block_size, block_size)
        blocks = _to_display_u8(blocks_f64, bit_depth=args.bit_depth)

        if len(blocks) != len(labels):
            raise RuntimeError(
                f"Count mismatch for block {block_size}: blocks={len(blocks)} labels={len(labels)}"
            )

        out_dir = out_root / f"block_{block_size}"
        out_dir.mkdir(parents=True, exist_ok=True)

        unique_labels = sorted(set(int(value) for value in labels.tolist()))
        generated = 0
        for label_id in unique_labels:
            candidates = [i for i, value in enumerate(labels.tolist()) if int(value) == label_id]
            chosen = _select_center_indices(candidates, rows=rows, cols=cols, k=args.max_per_label)
            class_name = PARTITION_ID_TO_NAME.get(label_id, f"PARTITION_UNKNOWN_{label_id}")
            for rank, idx in enumerate(chosen, start=1):
                suffix = "" if rank == 1 else f"_{rank}"
                name = f"{base}_{block_size}_{class_name}{suffix}.{args.image_format}"
                target = out_dir / name
                plt.imsave(target, blocks[idx], cmap="gray", vmin=0, vmax=255, format=args.image_format)
                generated += 1

        print(f"block {block_size}: generated {generated} images -> {out_dir}")


if __name__ == "__main__":
    main()
