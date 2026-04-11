#!/usr/bin/env python3
"""Run canonical thesis pipeline end-to-end flow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.runtime import CANONICAL_RUNTIME_FAMILY, run_end_to_end_thesis_flow
from thesis.scripts._common import parse_sequences, repo_root_from_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical thesis prepare -> train -> evaluate flow."
    )
    parser.add_argument("--run-name", required=True, help="Run identifier under thesis/runs")
    parser.add_argument(
        "--block-size",
        default="16",
        choices=["8", "16", "32", "64"],
        help="Block size for canonical flow.",
    )
    parser.add_argument(
        "--raw-root",
        default="thesis/uvg",
        help="Raw thesis UVG root containing <sequence>/partition_frame_0.txt",
    )
    parser.add_argument(
        "--legacy-base-path",
        default=None,
        help=(
            "Legacy-compatible dataset root (with intra_raw_blocks/labels/qps). "
            "Required unless --skip-legacy-generation is set."
        ),
    )
    parser.add_argument(
        "--runtime-family",
        default=CANONICAL_RUNTIME_FAMILY,
        help="Canonical runtime selector.",
    )
    parser.add_argument(
        "--auto-bootstrap-legacy-contract",
        action="store_true",
        help=(
            "Build temporary legacy contract (intra_raw_blocks/labels/qps) from "
            "thesis partition data when --legacy-base-path is not provided."
        ),
    )
    parser.add_argument(
        "--auto-generate-intra-raw-blocks",
        action="store_true",
        help=(
            "Generate thesis/uvg/intra_raw_blocks from partition_frame_0.txt + YUV "
            "before prepare/bootstrap. Existing files are never overwritten."
        ),
    )
    parser.add_argument(
        "--videos-root",
        default="videoset/uvg",
        help="Root containing <sequence>.yuv source videos.",
    )
    parser.add_argument(
        "--video-ext",
        default="yuv",
        help="Extension for source videos under --videos-root.",
    )
    parser.add_argument("--width", type=int, default=3840, help="Frame width (default: 3840).")
    parser.add_argument("--height", type=int, default=2160, help="Frame height (default: 2160).")
    parser.add_argument(
        "--partition-coord-scale",
        type=int,
        default=4,
        help=(
            "Scale factor from partition row/col units to pixel coordinates. "
            "Default: 4 (AV1 4x4 units)."
        ),
    )
    parser.add_argument(
        "--generate-visual-samples",
        action="store_true",
        help=(
            "Generate JPEG visual samples from legacy contract binaries after "
            "intra_raw_blocks/labels/qps are available."
        ),
    )
    parser.add_argument(
        "--visual-output-root",
        default=None,
        help=(
            "Optional output root for JPEG visual samples. "
            "Default: thesis/runs/<run-name>/artifacts/visual_samples"
        ),
    )
    parser.add_argument(
        "--visual-max-per-label-qp",
        type=int,
        default=1,
        help="Maximum images generated for each (label, qp) pair per sequence/block.",
    )
    parser.add_argument(
        "--visual-scale",
        type=int,
        default=8,
        help="Nearest-neighbor upscaling factor for visualization output.",
    )
    parser.add_argument(
        "--visual-image-format",
        default="jpg",
        choices=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Image format for visual samples (png/tiff are near-lossless).",
    )
    parser.add_argument(
        "--generate-frame-overlay",
        action="store_true",
        help=(
            "Generate one consolidated full-frame overlay image with extraction "
            "regions and class-count legend."
        ),
    )
    parser.add_argument(
        "--overlay-frame-number",
        type=int,
        default=0,
        help="Frame number used for consolidated frame overlay (default: 0).",
    )
    parser.add_argument(
        "--overlay-output-root",
        default=None,
        help=(
            "Optional output root for frame overlays. "
            "Default: thesis/runs/<run-name>/artifacts/frame_overlays"
        ),
    )
    parser.add_argument(
        "--overlay-image-format",
        default="jpg",
        choices=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Image format for full-frame overlay (png/tiff are near-lossless).",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs-stage2", type=int, default=100)
    parser.add_argument("--epochs-stage3-rect", type=int, default=30)
    parser.add_argument("--epochs-stage3-ab-binary", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Optional required sequence(s), repeat or comma-separate.",
    )
    parser.add_argument(
        "--skip-legacy-generation",
        action="store_true",
        help="Only validate contracts/manifests, skip running dataset generation scripts.",
    )
    parser.add_argument(
        "--require-intra-raw-blocks",
        action="store_true",
        help="Require thesis/uvg/intra_raw_blocks with 8/16/32/64 coverage.",
    )
    parser.add_argument(
        "--intra-raw-blocks-root",
        default=None,
        help="Override root for intra_raw_blocks validation.",
    )
    parser.add_argument(
        "--min-intra-raw-sequences",
        type=int,
        default=2,
        help="Minimum number of sequences required in intra_raw_blocks.",
    )
    parser.add_argument("--python", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-docs-gate",
        action="store_true",
        help="Disable documentation consistency gate.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from_script(Path(__file__))

    raw_root = (repo_root / args.raw_root).resolve()
    legacy_base_path = (
        (repo_root / args.legacy_base_path).resolve()
        if args.legacy_base_path
        else None
    )
    intra_raw_blocks_root = (
        (repo_root / args.intra_raw_blocks_root).resolve()
        if args.intra_raw_blocks_root
        else None
    )
    visual_output_root = (
        (repo_root / args.visual_output_root).resolve() if args.visual_output_root else None
    )
    overlay_output_root = (
        (repo_root / args.overlay_output_root).resolve() if args.overlay_output_root else None
    )
    videos_root = (repo_root / args.videos_root).resolve()
    sequences = parse_sequences(args.sequence)

    paths = run_end_to_end_thesis_flow(
        repo_root=repo_root,
        run_name=args.run_name,
        block_size=args.block_size,
        raw_root=raw_root,
        runtime_family=args.runtime_family,
        legacy_base_path=legacy_base_path,
        split=args.split,
        device=args.device,
        test_ratio=args.test_ratio,
        epochs_stage2=args.epochs_stage2,
        epochs_stage3_rect=args.epochs_stage3_rect,
        epochs_stage3_ab_binary=args.epochs_stage3_ab_binary,
        batch_size=args.batch_size,
        seed=args.seed,
        python_executable=args.python,
        skip_legacy_generation=args.skip_legacy_generation,
        dry_run=args.dry_run,
        required_sequences=sequences,
        intra_raw_blocks_root=intra_raw_blocks_root,
        require_intra_raw_blocks=args.require_intra_raw_blocks,
        min_intra_raw_sequences=args.min_intra_raw_sequences,
        auto_bootstrap_legacy_contract=args.auto_bootstrap_legacy_contract,
        auto_generate_intra_raw_blocks=args.auto_generate_intra_raw_blocks,
        videos_root=videos_root,
        video_ext=args.video_ext,
        frame_width=args.width,
        frame_height=args.height,
        partition_coord_scale=args.partition_coord_scale,
        generate_visual_samples=args.generate_visual_samples,
        visual_output_root=visual_output_root,
        visual_max_per_label_qp=args.visual_max_per_label_qp,
        visual_scale=args.visual_scale,
        visual_image_format=args.visual_image_format,
        generate_frame_overlay=args.generate_frame_overlay,
        overlay_frame_number=args.overlay_frame_number,
        overlay_output_root=overlay_output_root,
        overlay_image_format=args.overlay_image_format,
        docs_gate=not args.no_docs_gate,
    )
    print(f"End-to-end flow completed for run: {paths.run_dir}")


if __name__ == "__main__":
    main()
