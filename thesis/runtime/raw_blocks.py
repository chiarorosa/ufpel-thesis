"""Generate intra_raw_blocks using primary rearrange-compatible selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import math
import os
import re

import numpy as np


BLOCK_SIZE_TO_INDEX = {
    8: 3,
    16: 6,
    32: 9,
    64: 12,
}


@dataclass(frozen=True)
class _PartitionEntry:
    order_hint: int
    frame_type: int
    bsize: int
    row_u4: int
    col_u4: int
    partition_mode: int
    qp: int


@dataclass(frozen=True)
class RawBlockGenerationStats:
    sequence: str
    frame_number: int
    block_size: int
    partition_entries: int
    output_file: str
    output_bytes: int


def _parse_frame_number(frame_file_name: str) -> int:
    match = re.match(r"^partition_frame_(\d+)\.txt$", frame_file_name)
    if not match:
        return 0
    return int(match.group(1))


def _read_partition_rows(partition_file: Path) -> list[_PartitionEntry]:
    rows: list[_PartitionEntry] = []
    with open(partition_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                order_hint, frame_type, bsize, row_u4, col_u4, partition_mode, qp = (
                    int(value) for value in parts
                )
            except ValueError:
                continue
            rows.append(
                _PartitionEntry(
                    order_hint=order_hint,
                    frame_type=frame_type,
                    bsize=bsize,
                    row_u4=row_u4,
                    col_u4=col_u4,
                    partition_mode=partition_mode,
                    qp=qp,
                )
            )
    return rows


def _yuv_frame_sizes(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            "YUV 4:2:0 requires even width and height, got "
            f"{width}x{height}"
        )
    y_plane_bytes = width * height * 2
    uv_plane_bytes = (width // 2) * (height // 2) * 2
    frame_bytes = y_plane_bytes + uv_plane_bytes + uv_plane_bytes
    return y_plane_bytes, frame_bytes


def _load_y_plane(
    *,
    yuv_path: Path,
    frame_number: int,
    width: int,
    height: int,
) -> np.ndarray:
    y_plane_bytes, frame_bytes = _yuv_frame_sizes(width, height)
    file_size = os.path.getsize(yuv_path)
    if file_size % frame_bytes != 0:
        raise ValueError(
            f"Invalid YUV size for {yuv_path}: size={file_size}, frame_bytes={frame_bytes}"
        )
    total_frames = file_size // frame_bytes
    if frame_number >= total_frames:
        raise ValueError(
            f"Frame {frame_number} is out of range for {yuv_path}; total_frames={total_frames}"
        )

    offset = frame_number * frame_bytes
    with open(yuv_path, "rb") as handle:
        handle.seek(offset, 0)
        y_buffer = handle.read(y_plane_bytes)
    if len(y_buffer) != y_plane_bytes:
        raise IOError(
            f"Failed to read Y plane from {yuv_path}; expected {y_plane_bytes} bytes, got {len(y_buffer)}"
        )

    return np.frombuffer(y_buffer, dtype="<u2").reshape(height, width)


def _sorted_entries_for_block(
    rows: Sequence[_PartitionEntry],
    block_size: int,
) -> list[_PartitionEntry]:
    block_index = BLOCK_SIZE_TO_INDEX[block_size]
    selected = [
        item
        for item in rows
        if item.frame_type == 0 and item.bsize == block_index
    ]
    selected.sort(key=lambda item: (item.row_u4, item.col_u4, item.order_hint))
    return selected


def _pad_to_grid(y_plane: np.ndarray, block_size: int) -> tuple[np.ndarray, int, int]:
    height, width = y_plane.shape
    rows = math.ceil(height / block_size)
    cols = math.ceil(width / block_size)
    valid_height = rows * block_size
    valid_width = cols * block_size

    if valid_height == height and valid_width == width:
        return y_plane, rows, cols

    padded = np.zeros((valid_height, valid_width), dtype=np.uint16)
    padded[:height, :width] = y_plane
    return padded, rows, cols


def _build_row_major_blocks(
    y_plane: np.ndarray,
    block_size: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    blocks = np.zeros((rows * cols, block_size, block_size), dtype=np.uint16)
    index = 0
    ystart = 0
    while ystart < rows * block_size:
        xstart = 0
        while xstart < cols * block_size:
            blocks[index] = y_plane[ystart : ystart + block_size, xstart : xstart + block_size]
            index += 1
            xstart += block_size
        ystart += block_size
    return blocks


def _required_lcols(
    entries: Sequence[_PartitionEntry],
    block_size: int,
    coord_scale: int,
) -> list[int]:
    # Mirrors primary_method/rearrange_video.py behavior:
    # lcols = (col_u4 / image_size) * 4
    # generalized as (col_u4 / block_size) * coord_scale.
    values = [int((item.col_u4 / block_size) * coord_scale) for item in entries]
    return values


def _legacy_rearrange_select_indices(rows: int, cols: int, lcols: Sequence[int]) -> list[int]:
    candidates = list(range(rows * cols))
    num_label = len(lcols)
    if num_label == 0:
        return []

    index = 0
    for _row in range(rows):
        for col in range(cols):
            if lcols[index] != col:
                if index >= len(candidates):
                    raise IndexError(
                        "Legacy rearrange selection index out of range while deleting "
                        f"(index={index}, candidates={len(candidates)})"
                    )
                del candidates[index]
            elif index == num_label - 1:
                break
            else:
                index += 1

    if len(candidates) < num_label:
        raise RuntimeError(
            "Legacy rearrange selection produced fewer candidates than labels: "
            f"candidates={len(candidates)}, labels={num_label}"
        )
    return candidates[:num_label]


def generate_intra_raw_blocks_from_partition(
    *,
    raw_root: Path,
    videos_root: Path,
    output_root: Path,
    sequences: Sequence[str],
    width: int = 3840,
    height: int = 2160,
    video_ext: str = "yuv",
    frame_file_name: str = "partition_frame_0.txt",
    partition_coord_scale: int = 4,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[RawBlockGenerationStats]:
    """Generate `<sequence>_sample_<block>.txt` with rearrange-compatible strategy."""
    if not videos_root.exists():
        raise FileNotFoundError(f"videos_root does not exist: {videos_root}")
    if partition_coord_scale <= 0:
        raise ValueError(
            f"partition_coord_scale must be >= 1, got {partition_coord_scale}"
        )

    frame_number = _parse_frame_number(frame_file_name)
    output_root.mkdir(parents=True, exist_ok=True)
    stats: list[RawBlockGenerationStats] = []

    for sequence in sequences:
        partition_file = raw_root / sequence / frame_file_name
        if not partition_file.exists():
            raise FileNotFoundError(
                f"Missing partition file for sequence {sequence}: {partition_file}"
            )

        video_file = videos_root / f"{sequence}.{video_ext}"
        if not video_file.exists():
            raise FileNotFoundError(
                f"Missing YUV file for sequence {sequence}: {video_file}"
            )

        rows = _read_partition_rows(partition_file)
        if not rows:
            raise ValueError(f"No valid partition rows found in {partition_file}")

        block_entries = {
            block_size: _sorted_entries_for_block(rows, block_size)
            for block_size in (8, 16, 32, 64)
        }
        for block_size, entries in block_entries.items():
            if not entries:
                raise ValueError(
                    f"No partition entries found for block {block_size} in {partition_file}"
                )

        output_targets = {
            block_size: output_root / f"{sequence}_sample_{block_size}.txt"
            for block_size in (8, 16, 32, 64)
        }
        if not overwrite:
            conflicts = [target for target in output_targets.values() if target.exists()]
            if conflicts:
                conflict_paths = "\n".join(str(path) for path in conflicts)
                raise FileExistsError(
                    "Refusing to overwrite existing intra_raw_blocks files:\n"
                    + conflict_paths
                )

        y_plane = _load_y_plane(
            yuv_path=video_file,
            frame_number=frame_number,
            width=width,
            height=height,
        )

        for block_size, entries in block_entries.items():
            padded, grid_rows, grid_cols = _pad_to_grid(y_plane, block_size)
            candidates = _build_row_major_blocks(padded, block_size, grid_rows, grid_cols)
            lcols = _required_lcols(entries, block_size, partition_coord_scale)
            selected_indices = _legacy_rearrange_select_indices(grid_rows, grid_cols, lcols)
            selected = candidates[selected_indices]

            if len(selected) != len(entries):
                raise RuntimeError(
                    "Rearrange-compatible extraction count mismatch for "
                    f"{sequence} block {block_size}: selected={len(selected)} "
                    f"entries={len(entries)}"
                )

            target = output_targets[block_size]
            output_bytes = len(entries) * block_size * block_size * 2
            if not dry_run:
                with open(target, "wb") as handle:
                    selected.astype("<u2").tofile(handle)
                actual_size = os.path.getsize(target)
                if actual_size != output_bytes:
                    raise IOError(
                        f"Unexpected output size for {target}: expected={output_bytes}, got={actual_size}"
                    )

            stats.append(
                RawBlockGenerationStats(
                    sequence=sequence,
                    frame_number=frame_number,
                    block_size=block_size,
                    partition_entries=len(entries),
                    output_file=str(target),
                    output_bytes=output_bytes,
                )
            )

    return stats
