"""Generate full-frame block-partition overlays using a leaf-tiling view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


BLOCK_INDEX_TO_SIZE: dict[int, int] = {
    3: 8,
    6: 16,
    9: 32,
    12: 64,
    15: 128,
}

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

PARTITION_COLORS: dict[int, tuple[int, int, int]] = {
    0: (70, 70, 70),
    1: (220, 30, 30),
    2: (30, 140, 230),
    3: (255, 170, 0),
    4: (160, 70, 220),
    5: (220, 90, 170),
    6: (0, 180, 160),
    7: (150, 200, 40),
    8: (240, 120, 40),
    9: (70, 210, 210),
    99: (20, 20, 20),
}

PARTITION_GAP_FILL_ID = 99
PARTITION_GAP_FILL_NAME = "PARTITION_GAP_FILL"

IMAGE_FORMAT_TO_EXT: dict[str, str] = {
    "jpg": "jpg",
    "jpeg": "jpg",
    "png": "png",
    "tif": "tif",
    "tiff": "tiff",
}


@dataclass(frozen=True)
class FrameOverlayStats:
    sequence: str
    frame_number: int
    output_file: str
    total_regions: int
    class_counts: dict[str, int]
    block_size_counts: dict[str, int]
    coverage_ratio: float


@dataclass(frozen=True)
class _Candidate:
    order_idx: int
    row_u: int
    col_u: int
    block_px: int
    block_u: int
    label_id: int


@dataclass(frozen=True)
class _LeafTile:
    row_u: int
    col_u: int
    h_u: int
    w_u: int
    block_px: int
    label_id: int


def _normalize_image_format(image_format: str) -> tuple[str, str]:
    fmt = image_format.strip().lower()
    ext = IMAGE_FORMAT_TO_EXT.get(fmt)
    if ext is None:
        allowed = ", ".join(sorted(IMAGE_FORMAT_TO_EXT.keys()))
        raise ValueError(f"Unsupported image format '{image_format}'. Allowed: {allowed}")
    mpl_format = "jpeg" if fmt in {"jpg", "jpeg"} else fmt
    return ext, mpl_format


def _read_partition_rows(
    partition_file: Path,
) -> list[tuple[int, int, int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int, int, int]] = []
    with open(partition_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                row = tuple(int(value) for value in parts)
            except ValueError:
                continue
            rows.append(row)  # type: ignore[arg-type]
    return rows


def _yuv_frame_sizes(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            "YUV 4:2:0 requires even width and height, got " f"{width}x{height}"
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

    y_values = np.frombuffer(y_buffer, dtype="<u2")
    return y_values.reshape(height, width)


def _to_u8_plane(y_plane: np.ndarray) -> np.ndarray:
    clipped = np.clip(y_plane, 0, 1023)
    return np.rint((clipped.astype(np.float32) * 255.0) / 1023.0).astype(np.uint8)


def _extract_candidates(
    rows: Iterable[tuple[int, int, int, int, int, int, int]],
    *,
    coord_scale: int,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for order_idx, (_, frame_type, bsize, row, col, partition_mode, _) in enumerate(rows):
        if frame_type != 0:
            continue
        block_px = BLOCK_INDEX_TO_SIZE.get(bsize)
        if block_px is None:
            continue
        if block_px % coord_scale != 0:
            raise ValueError(
                f"Block size {block_px} is not divisible by coord_scale={coord_scale}"
            )
        block_u = block_px // coord_scale
        candidates.append(
            _Candidate(
                order_idx=order_idx,
                row_u=row,
                col_u=col,
                block_px=block_px,
                block_u=block_u,
                label_id=partition_mode,
            )
        )
    if not candidates:
        raise ValueError("No valid intra candidates found in partition rows")
    return candidates


def _build_leaf_tiling(
    candidates: Sequence[_Candidate],
    *,
    frame_h_units: int,
    frame_w_units: int,
) -> tuple[list[_LeafTile], float]:
    assigned = np.zeros((frame_h_units, frame_w_units), dtype=bool)
    tiles: list[_LeafTile] = []

    # Leaf-tiling approximation: use non-SPLIT nodes only, prefer larger blocks first.
    non_split = [item for item in candidates if item.label_id != 3]
    ordered = sorted(non_split, key=lambda item: (-item.block_u, item.order_idx))

    for item in ordered:
        if item.row_u < 0 or item.col_u < 0:
            continue
        r0 = item.row_u
        c0 = item.col_u
        if r0 >= frame_h_units or c0 >= frame_w_units:
            continue
        r1 = min(frame_h_units, r0 + item.block_u)
        c1 = min(frame_w_units, c0 + item.block_u)
        if r1 <= r0 or c1 <= c0:
            continue
        region = assigned[r0:r1, c0:c1]
        if region.any():
            continue
        assigned[r0:r1, c0:c1] = True
        tiles.append(
            _LeafTile(
                row_u=r0,
                col_u=c0,
                h_u=r1 - r0,
                w_u=c1 - c0,
                block_px=item.block_px,
                label_id=item.label_id,
            )
        )

    # Fill residual holes with 1x1 unit tiles to guarantee full frame coverage.
    hole_rows, hole_cols = np.where(~assigned)
    for row_u, col_u in zip(hole_rows.tolist(), hole_cols.tolist()):
        tiles.append(
            _LeafTile(
                row_u=row_u,
                col_u=col_u,
                h_u=1,
                w_u=1,
                block_px=4,
                label_id=PARTITION_GAP_FILL_ID,
            )
        )

    assigned[:, :] = True
    coverage_ratio = float(assigned.mean())
    return tiles, coverage_ratio


def _render_leaf_overlay(
    *,
    y_plane_u8: np.ndarray,
    tiles: Sequence[_LeafTile],
    coord_scale: int,
    output_file: Path,
    title: str,
    class_counts: dict[str, int],
) -> None:
    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)
    ax.imshow(y_plane_u8, cmap="gray", vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=12)

    for tile in tiles:
        x = tile.col_u * coord_scale
        y = tile.row_u * coord_scale
        w = tile.w_u * coord_scale
        h = tile.h_u * coord_scale
        color = np.array(PARTITION_COLORS.get(tile.label_id, (255, 255, 255)), dtype=np.float32) / 255.0
        patch = mpatches.Rectangle(
            (x, y),
            w,
            h,
            facecolor=color,
            edgecolor=color,
            linewidth=0.20,
            alpha=0.35,
        )
        ax.add_patch(patch)

    legend_patches: list[mpatches.Patch] = []
    for class_name, count in class_counts.items():
        label_id = next(
            (idx for idx, name in PARTITION_ID_TO_NAME.items() if name == class_name),
            PARTITION_GAP_FILL_ID if class_name == PARTITION_GAP_FILL_NAME else -1,
        )
        color = np.array(PARTITION_COLORS.get(label_id, (255, 255, 255)), dtype=np.float32) / 255.0
        legend_patches.append(mpatches.Patch(color=color, label=f"{class_name}: {count}"))

    legend = ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title=f"Leaf tiles (total: {sum(class_counts.values())})",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    legend.get_frame().set_alpha(0.95)

    fig.tight_layout(rect=[0, 0, 0.82, 1.0])
    fig.savefig(output_file)
    plt.close(fig)


def generate_frame_overlay_images(
    *,
    raw_root: Path,
    videos_root: Path,
    output_root: Path,
    sequences: Sequence[str],
    frame_number: int = 0,
    width: int = 3840,
    height: int = 2160,
    video_ext: str = "yuv",
    partition_coord_scale: int = 4,
    image_format: str = "jpg",
    dry_run: bool = False,
) -> list[FrameOverlayStats]:
    """Generate one full-frame leaf-tiling overlay per sequence."""
    if frame_number < 0:
        raise ValueError(f"frame_number must be >= 0, got {frame_number}")
    if partition_coord_scale <= 0:
        raise ValueError(
            f"partition_coord_scale must be >= 1, got {partition_coord_scale}"
        )
    if not videos_root.exists():
        raise FileNotFoundError(f"videos_root does not exist: {videos_root}")

    ext, _ = _normalize_image_format(image_format)
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    frame_h_units = math.ceil(height / partition_coord_scale)
    frame_w_units = math.ceil(width / partition_coord_scale)

    stats: list[FrameOverlayStats] = []
    for sequence in sequences:
        partition_file = raw_root / sequence / f"partition_frame_{frame_number}.txt"
        if not partition_file.exists():
            raise FileNotFoundError(f"Missing partition file: {partition_file}")

        video_file = videos_root / f"{sequence}.{video_ext}"
        if not video_file.exists():
            raise FileNotFoundError(f"Missing YUV file: {video_file}")

        rows = _read_partition_rows(partition_file)
        candidates = _extract_candidates(rows, coord_scale=partition_coord_scale)
        leaf_tiles, coverage_ratio = _build_leaf_tiling(
            candidates,
            frame_h_units=frame_h_units,
            frame_w_units=frame_w_units,
        )

        class_counts_raw: dict[int, int] = {}
        block_size_counts_raw: dict[int, int] = {}
        for tile in leaf_tiles:
            class_counts_raw[tile.label_id] = class_counts_raw.get(tile.label_id, 0) + 1
            block_size_counts_raw[tile.block_px] = block_size_counts_raw.get(tile.block_px, 0) + 1

        class_counts: dict[str, int] = {}
        for label_id, count in sorted(class_counts_raw.items(), key=lambda item: item[0]):
            if label_id == PARTITION_GAP_FILL_ID:
                class_name = PARTITION_GAP_FILL_NAME
            else:
                class_name = PARTITION_ID_TO_NAME.get(label_id, f"PARTITION_UNKNOWN_{label_id}")
            class_counts[class_name] = count

        block_size_counts = {str(size): count for size, count in sorted(block_size_counts_raw.items())}

        sequence_root = output_root / sequence
        output_file = sequence_root / f"{sequence}_frame{frame_number}_overlay_all_blocks.{ext}"

        if not dry_run:
            sequence_root.mkdir(parents=True, exist_ok=True)
            y_plane = _load_y_plane(
                yuv_path=video_file,
                frame_number=frame_number,
                width=width,
                height=height,
            )
            _render_leaf_overlay(
                y_plane_u8=_to_u8_plane(y_plane),
                tiles=leaf_tiles,
                coord_scale=partition_coord_scale,
                output_file=output_file,
                title=f"{sequence} frame {frame_number} - leaf tiling partition view",
                class_counts=class_counts,
            )

        stats.append(
            FrameOverlayStats(
                sequence=sequence,
                frame_number=frame_number,
                output_file=str(output_file),
                total_regions=len(leaf_tiles),
                class_counts=class_counts,
                block_size_counts=block_size_counts,
                coverage_ratio=coverage_ratio,
            )
        )

    return stats
