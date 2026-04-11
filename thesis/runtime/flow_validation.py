"""Validation utilities for the expected thesis raw-data flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json
import os
import re


BLOCK_SIZES = (8, 16, 32, 64)
BLOCK_SIZE_TO_INDEX = {8: 3, 16: 6, 32: 9, 64: 12}


@dataclass(frozen=True)
class BlockFlowStats:
    sequence: str
    block_size: int
    sample_blocks: int
    labels_count: int
    qps_count: int
    partition_entries: int
    partition_alignment: str


def _read_int_tokens(path: Path) -> list[int]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    values: list[int] = []
    for token in text.split():
        values.append(int(token))
    return values


def _count_sample_blocks(sample_file: Path, block_size: int) -> int:
    block_bytes = 2 * block_size * block_size
    total_bytes = os.path.getsize(sample_file)
    remainder = total_bytes % block_bytes
    if remainder != 0:
        raise ValueError(
            f"Invalid raw sample size for block {block_size}: {sample_file} "
            f"(remainder={remainder})"
        )
    return total_bytes // block_bytes


def _count_partition_entries(partition_file: Path, block_size: int) -> int:
    block_index = BLOCK_SIZE_TO_INDEX[block_size]
    count = 0
    with open(partition_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                frame_type = int(parts[1])
                bsize = int(parts[2])
            except ValueError:
                continue
            if frame_type == 0 and bsize == block_index:
                count += 1
    return count


def _discover_intra_sequences(intra_raw_blocks_root: Path) -> dict[str, set[int]]:
    pattern = re.compile(r"^(?P<seq>.+)_sample_(?P<block>\d+)\.txt$")
    discovered: dict[str, set[int]] = {}
    for path in sorted(intra_raw_blocks_root.iterdir()):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        seq = match.group("seq")
        block = int(match.group("block"))
        if block not in BLOCK_SIZES:
            continue
        discovered.setdefault(seq, set()).add(block)
    return discovered


def validate_expected_raw_flow(
    *,
    raw_root: Path,
    legacy_base_path: Path,
    output_json: Path | None = None,
) -> dict:
    """
    Validate expected flow:
    partition_frame -> intra_raw_blocks -> labels/qps.

    The active sequence set is inferred from intra_raw_blocks.
    """
    intra_raw_blocks_root = legacy_base_path / "intra_raw_blocks"
    labels_root = legacy_base_path / "labels"
    qps_root = legacy_base_path / "qps"

    if not intra_raw_blocks_root.exists():
        raise FileNotFoundError(f"Missing intra_raw_blocks: {intra_raw_blocks_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Missing labels: {labels_root}")
    if not qps_root.exists():
        raise FileNotFoundError(f"Missing qps: {qps_root}")

    discovered = _discover_intra_sequences(intra_raw_blocks_root)
    if not discovered:
        raise FileNotFoundError(
            f"No *_sample_<block>.txt files found in {intra_raw_blocks_root}"
        )

    errors: list[str] = []
    warnings: list[str] = []
    stats: list[BlockFlowStats] = []

    for sequence, blocks in sorted(discovered.items()):
        missing_blocks = sorted(set(BLOCK_SIZES) - blocks)
        if missing_blocks:
            errors.append(
                f"{sequence}: missing sample blocks {missing_blocks} in intra_raw_blocks"
            )
            continue

        partition_file = raw_root / sequence / "partition_frame_0.txt"
        if not partition_file.exists():
            errors.append(f"{sequence}: missing partition file {partition_file}")
            continue

        for block_size in BLOCK_SIZES:
            sample_file = intra_raw_blocks_root / f"{sequence}_sample_{block_size}.txt"
            labels_file = labels_root / f"{sequence}_labels_{block_size}_intra.txt"
            qps_file = qps_root / f"{sequence}_qps_{block_size}_intra.txt"

            if not labels_file.exists():
                errors.append(f"{sequence} block {block_size}: missing labels file")
                continue
            if not qps_file.exists():
                errors.append(f"{sequence} block {block_size}: missing qps file")
                continue

            sample_blocks = _count_sample_blocks(sample_file, block_size)
            labels_values = _read_int_tokens(labels_file)
            qps_values = _read_int_tokens(qps_file)
            partition_entries = _count_partition_entries(partition_file, block_size)

            if len(labels_values) != sample_blocks:
                errors.append(
                    f"{sequence} block {block_size}: labels_count={len(labels_values)} "
                    f"!= sample_blocks={sample_blocks}"
                )
            if len(qps_values) != sample_blocks:
                errors.append(
                    f"{sequence} block {block_size}: qps_count={len(qps_values)} "
                    f"!= sample_blocks={sample_blocks}"
                )

            if partition_entries == sample_blocks:
                alignment = "rearrange_exact"
            else:
                alignment = "mismatch"
                errors.append(
                    f"{sequence} block {block_size}: partition_entries={partition_entries} "
                    f"not aligned with sample_blocks={sample_blocks} under rearrange_exact flow"
                )

            stats.append(
                BlockFlowStats(
                    sequence=sequence,
                    block_size=block_size,
                    sample_blocks=sample_blocks,
                    labels_count=len(labels_values),
                    qps_count=len(qps_values),
                    partition_entries=partition_entries,
                    partition_alignment=alignment,
                )
            )

    result = {
        "ok": len(errors) == 0,
        "raw_root": str(raw_root),
        "legacy_base_path": str(legacy_base_path),
        "active_sequences": sorted(discovered.keys()),
        "errors": errors,
        "warnings": warnings,
        "stats": [asdict(item) for item in stats],
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    return result
