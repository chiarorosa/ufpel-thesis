"""Bootstrap legacy labels/qps contract from thesis partition files.

This implementation is aligned with rearrange-compatible sample selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import math
import os


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
class LegacyBootstrapStats:
    sequence: str
    block_size: int
    labels_written: int
    qps_written: int
    sample_blocks: int
    alignment_mode: str
    dropped_entries: int
    trimmed_entries: int


def _read_partition_rows(partition_file: Path) -> list[_PartitionEntry]:
    rows: list[_PartitionEntry] = []
    with open(partition_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
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


def _sorted_entries_for_block(rows: Sequence[_PartitionEntry], block_size: int) -> list[_PartitionEntry]:
    block_index = BLOCK_SIZE_TO_INDEX[block_size]
    selected = [
        item
        for item in rows
        if item.frame_type == 0 and item.bsize == block_index
    ]
    selected.sort(key=lambda item: (item.row_u4, item.col_u4, item.order_hint))
    return selected


def _sample_block_count(sample_file: Path, block_size: int) -> int:
    block_bytes = 2 * block_size * block_size
    size_bytes = os.path.getsize(sample_file)
    remainder = size_bytes % block_bytes
    if remainder != 0:
        raise ValueError(
            f"Sample file has invalid byte size for block {block_size}: "
            f"{sample_file} (remainder {remainder})"
        )
    return size_bytes // block_bytes


def _legacy_required_lcols(
    entries: Sequence[_PartitionEntry],
    block_size: int,
    coord_scale: int,
) -> list[int]:
    return [int((entry.col_u4 / block_size) * coord_scale) for entry in entries]


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
                        "Legacy selection index out of range while deleting "
                        f"(index={index}, candidates={len(candidates)})"
                    )
                del candidates[index]
            elif index == num_label - 1:
                break
            else:
                index += 1

    if len(candidates) < num_label:
        raise RuntimeError(
            "Legacy selection produced fewer candidates than labels: "
            f"candidates={len(candidates)}, labels={num_label}"
        )
    return candidates[:num_label]


def _extract_labels_qps_rearrange_order(
    entries: Sequence[_PartitionEntry],
    block_size: int,
    frame_width: int,
    frame_height: int,
    partition_coord_scale: int,
) -> tuple[list[int], list[int]]:
    rows = math.ceil(frame_height / block_size)
    cols = math.ceil(frame_width / block_size)
    lcols = _legacy_required_lcols(entries, block_size, partition_coord_scale)
    selected_indices = _legacy_rearrange_select_indices(rows, cols, lcols)
    if len(selected_indices) != len(entries):
        raise RuntimeError(
            "Rearrange-exact selection produced unexpected count: "
            f"selected={len(selected_indices)} entries={len(entries)}"
        )
    # Primary flow reads labels directly from XLSX sheet order, so labels/qps
    # follow the sorted entry order rather than candidate index order.
    labels = [entry.partition_mode for entry in entries]
    qps = [entry.qp for entry in entries]
    return labels, qps


def _write_numeric_vector(path: Path, values: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if values:
            handle.write(" ".join(str(v) for v in values))
        handle.write("\n")


def bootstrap_legacy_labels_qps_from_partition(
    *,
    uvg_root: Path,
    contract_root: Path,
    sequences: Sequence[str],
    block_sizes: Sequence[int] = (8, 16, 32, 64),
    frame_file_name: str = "partition_frame_0.txt",
    drop_first_entry: bool = False,
    align_to_samples: bool = True,
    alignment_mode: str = "rearrange_exact",
    frame_width: int = 3840,
    frame_height: int = 2160,
    partition_coord_scale: int = 4,
) -> list[LegacyBootstrapStats]:
    """Build legacy labels/qps with rearrange-compatible sample ordering."""
    intra_raw_blocks = contract_root / "intra_raw_blocks"
    labels_root = contract_root / "labels"
    qps_root = contract_root / "qps"

    if not intra_raw_blocks.exists():
        raise FileNotFoundError(
            f"Missing intra_raw_blocks for legacy contract: {intra_raw_blocks}"
        )

    if alignment_mode != "rearrange_exact":
        raise ValueError(
            "Only 'rearrange_exact' alignment_mode is supported in the new thesis flow"
        )
    if drop_first_entry:
        raise ValueError("drop_first_entry is not supported in rearrange_exact flow")

    stats: list[LegacyBootstrapStats] = []

    for sequence in sequences:
        partition_file = uvg_root / sequence / frame_file_name
        if not partition_file.exists():
            raise FileNotFoundError(
                f"Missing partition file for sequence {sequence}: {partition_file}"
            )
        rows = _read_partition_rows(partition_file)

        for block_size in block_sizes:
            sample_file = intra_raw_blocks / f"{sequence}_sample_{block_size}.txt"
            if not sample_file.exists():
                raise FileNotFoundError(
                    "Missing sample file required for legacy compatibility: "
                    f"{sample_file}"
                )

            entries = _sorted_entries_for_block(rows, block_size)
            labels, qps = _extract_labels_qps_rearrange_order(
                entries,
                block_size,
                frame_width,
                frame_height,
                partition_coord_scale,
            )

            sample_blocks = _sample_block_count(sample_file, block_size)
            if align_to_samples and len(labels) != sample_blocks:
                raise ValueError(
                    "Rearrange-exact labels/sample mismatch for "
                    f"{sequence} block {block_size}: labels={len(labels)}, samples={sample_blocks}"
                )

            labels_path = labels_root / f"{sequence}_labels_{block_size}_intra.txt"
            qps_path = qps_root / f"{sequence}_qps_{block_size}_intra.txt"

            _write_numeric_vector(labels_path, labels)
            _write_numeric_vector(qps_path, qps)

            stats.append(
                LegacyBootstrapStats(
                    sequence=sequence,
                    block_size=block_size,
                    labels_written=len(labels),
                    qps_written=len(qps),
                    sample_blocks=sample_blocks,
                    alignment_mode=alignment_mode,
                    dropped_entries=0,
                    trimmed_entries=0,
                )
            )

    return stats
