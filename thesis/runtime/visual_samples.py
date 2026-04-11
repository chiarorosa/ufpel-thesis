"""Generate visual JPEG samples from legacy block contract binaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


PARTITION_ID_TO_NAME: dict[int, str] = {
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


@dataclass(frozen=True)
class VisualSampleStats:
    sequence: str
    block_size: int
    generated_images: int
    unique_label_qp_pairs: int
    output_dir: str


IMAGE_FORMAT_TO_EXT: dict[str, str] = {
    "jpg": "jpg",
    "jpeg": "jpg",
    "png": "png",
    "tif": "tif",
    "tiff": "tiff",
}

BLOCK_SIZE_TO_INDEX: dict[int, int] = {
    8: 3,
    16: 6,
    32: 9,
    64: 12,
}


def _normalize_image_format(image_format: str) -> tuple[str, str]:
    fmt = image_format.strip().lower()
    ext = IMAGE_FORMAT_TO_EXT.get(fmt)
    if ext is None:
        allowed = ", ".join(sorted(IMAGE_FORMAT_TO_EXT.keys()))
        raise ValueError(f"Unsupported image format '{image_format}'. Allowed: {allowed}")
    # matplotlib accepts 'jpeg' but not always 'jpg' as format string.
    mpl_format = "jpeg" if fmt in {"jpg", "jpeg"} else fmt
    return ext, mpl_format


def _read_tokens(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.int64, sep=" ")
    return values.reshape(-1)


def _read_samples(path: Path, block_size: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    pixels_per_block = block_size * block_size
    remainder = raw.size % pixels_per_block
    if remainder != 0:
        raise ValueError(
            f"Invalid sample size in {path} for block {block_size}; "
            f"remainder={remainder}"
        )
    return raw.reshape(-1, block_size, block_size)


def _read_partition_coords(
    *,
    raw_root: Path,
    sequence: str,
    block_size: int,
    frame_file_name: str,
) -> list[tuple[int, int]] | None:
    block_index = BLOCK_SIZE_TO_INDEX.get(block_size)
    if block_index is None:
        return None
    partition_file = raw_root / sequence / frame_file_name
    if not partition_file.exists():
        return None

    coords: list[tuple[int, int]] = []
    with open(partition_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) != 7:
                continue
            try:
                _, frame_type, bsize, row, col, _, _ = (int(value) for value in parts)
            except ValueError:
                continue
            if frame_type != 0:
                continue
            if bsize != block_index:
                continue
            coords.append((row, col))
    return coords


def _align_coords_to_samples(
    coords: list[tuple[int, int]] | None,
    sample_count: int,
) -> list[tuple[int, int]] | None:
    if coords is None:
        return None
    if len(coords) == sample_count:
        return coords
    if len(coords) - 1 == sample_count:
        return coords[1:]
    if len(coords) > sample_count:
        trim_start = len(coords) - sample_count
        return coords[trim_start:]
    return None


def _select_evenly_spaced_indices(indices: Sequence[int], k: int) -> list[int]:
    n = len(indices)
    if n <= k:
        return list(indices)
    if k == 1:
        return [indices[n // 2]]

    selected: list[int] = []
    used: set[int] = set()
    for rank in range(k):
        pos = round(rank * (n - 1) / (k - 1))
        idx = indices[pos]
        if idx in used:
            continue
        selected.append(idx)
        used.add(idx)

    if len(selected) < k:
        for idx in indices:
            if idx in used:
                continue
            selected.append(idx)
            used.add(idx)
            if len(selected) == k:
                break
    return selected


def _select_center_roi_indices(
    *,
    candidate_indices: Sequence[int],
    max_samples: int,
    aligned_coords: Sequence[tuple[int, int]] | None,
) -> list[int]:
    if len(candidate_indices) <= max_samples:
        return list(candidate_indices)
    if aligned_coords is None:
        return _select_evenly_spaced_indices(candidate_indices, max_samples)

    rows = [row for row, _ in aligned_coords]
    cols = [col for _, col in aligned_coords]
    center_row = (min(rows) + max(rows)) / 2.0
    center_col = (min(cols) + max(cols)) / 2.0

    ranked = sorted(
        candidate_indices,
        key=lambda idx: (
            (aligned_coords[idx][0] - center_row) ** 2
            + (aligned_coords[idx][1] - center_col) ** 2,
            idx,
        ),
    )
    return ranked[:max_samples]


def _to_display_uint8(sample: np.ndarray, scale: int) -> np.ndarray:
    clipped = np.clip(sample, 0, 1023)
    display = np.rint((clipped.astype(np.float32) * 255.0) / 1023.0).astype(np.uint8)
    if scale > 1:
        display = np.repeat(display, scale, axis=0)
        display = np.repeat(display, scale, axis=1)
    return display


def _label_to_name(label_id: int) -> str:
    return PARTITION_ID_TO_NAME.get(label_id, "PARTITION_UNKNOWN")


def generate_visual_samples_from_legacy_contract(
    *,
    legacy_base_path: Path,
    output_root: Path,
    sequences: Sequence[str],
    block_sizes: Sequence[int] = (8, 16, 32, 64),
    max_per_label_qp: int = 1,
    scale: int = 8,
    image_format: str = "jpg",
    raw_root: Path | None = None,
    partition_frame_file_name: str = "partition_frame_0.txt",
    dry_run: bool = False,
) -> list[VisualSampleStats]:
    """Generate JPEG previews with `<sequence>_<block>_<label>_<qp>.jpg` naming."""
    if max_per_label_qp <= 0:
        raise ValueError(f"max_per_label_qp must be >= 1, got {max_per_label_qp}")
    if scale <= 0:
        raise ValueError(f"scale must be >= 1, got {scale}")
    ext, mpl_format = _normalize_image_format(image_format)

    samples_root = legacy_base_path / "intra_raw_blocks"
    labels_root = legacy_base_path / "labels"
    qps_root = legacy_base_path / "qps"
    for root in (samples_root, labels_root, qps_root):
        if not root.exists():
            raise FileNotFoundError(f"Missing legacy contract directory: {root}")

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    stats: list[VisualSampleStats] = []
    for sequence in sequences:
        for block_size in block_sizes:
            sample_file = samples_root / f"{sequence}_sample_{block_size}.txt"
            label_file = labels_root / f"{sequence}_labels_{block_size}_intra.txt"
            qps_file = qps_root / f"{sequence}_qps_{block_size}_intra.txt"

            if not sample_file.exists():
                raise FileNotFoundError(f"Missing sample file: {sample_file}")
            if not label_file.exists():
                raise FileNotFoundError(f"Missing labels file: {label_file}")
            if not qps_file.exists():
                raise FileNotFoundError(f"Missing qps file: {qps_file}")

            samples = _read_samples(sample_file, block_size)
            labels = _read_tokens(label_file)
            qps = _read_tokens(qps_file)

            if len(samples) != len(labels) or len(samples) != len(qps):
                raise ValueError(
                    "Visual sample generation requires aligned sample/label/qp counts: "
                    f"{sequence} block {block_size} -> "
                    f"samples={len(samples)}, labels={len(labels)}, qps={len(qps)}"
                )

            output_dir = output_root / sequence / f"block_{block_size}"
            if not dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)

            aligned_coords: list[tuple[int, int]] | None = None
            if raw_root is not None:
                raw_coords = _read_partition_coords(
                    raw_root=raw_root,
                    sequence=sequence,
                    block_size=block_size,
                    frame_file_name=partition_frame_file_name,
                )
                aligned_coords = _align_coords_to_samples(raw_coords, len(samples))

            pair_to_indices: dict[tuple[int, int], list[int]] = {}
            for idx, (label, qp) in enumerate(zip(labels, qps)):
                key = (int(label), int(qp))
                pair_to_indices.setdefault(key, []).append(idx)

            generated = 0
            for key in sorted(pair_to_indices.keys()):
                candidate_indices = pair_to_indices[key]
                selected_indices = _select_center_roi_indices(
                    candidate_indices=candidate_indices,
                    max_samples=max_per_label_qp,
                    aligned_coords=aligned_coords,
                )
                label_name = _label_to_name(key[0])
                for rank, sample_idx in enumerate(selected_indices, start=1):
                    suffix = "" if rank == 1 else f"_{rank}"
                    filename = (
                        f"{sequence}_{block_size}_{label_name}_{key[1]}{suffix}.{ext}"
                    )
                    target = output_dir / filename

                    if not dry_run:
                        image = _to_display_uint8(samples[sample_idx], scale)
                        plt.imsave(
                            target,
                            image,
                            cmap="gray",
                            vmin=0,
                            vmax=255,
                            format=mpl_format,
                        )
                    generated += 1

            stats.append(
                VisualSampleStats(
                    sequence=sequence,
                    block_size=block_size,
                    generated_images=generated,
                    unique_label_qp_pairs=len(pair_to_indices),
                    output_dir=str(output_dir),
                )
            )

    return stats
