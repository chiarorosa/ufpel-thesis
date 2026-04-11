"""Canonical orchestration for thesis prepare/train/evaluate workflows."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence
import sys

from .contracts import (
    CANONICAL_RUNTIME_FAMILY,
    RunPaths,
    assert_canonical_runtime_family,
    discover_sequences,
    ensure_dir,
    resolve_run_paths,
    validate_conv_adapter_frozen_backbone_contract,
    validate_documents_structure,
    validate_intra_raw_blocks_contract,
    validate_prepare_handoff,
    validate_raw_input_contract,
    validate_standalone_reference_contract,
    validate_train_handoff,
    write_json,
)
from .legacy_contract import bootstrap_legacy_labels_qps_from_partition
from .raw_blocks import generate_intra_raw_blocks_from_partition
from .visual_samples import generate_visual_samples_from_legacy_contract
from .frame_overlays import generate_frame_overlay_images
from .flow_validation import validate_expected_raw_flow
from .runner import run_command


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _default_python(python_executable: str | None) -> str:
    return python_executable or sys.executable


def _resolve_python_executable(repo_root: Path, python_executable: str | None) -> str:
    candidate = _default_python(python_executable)
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return str(candidate_path)
    if any(sep in candidate for sep in ("/", "\\")):
        return str((repo_root / candidate_path).absolute())
    return candidate


def _write_phase_manifest(paths: RunPaths, phase: str, payload: dict) -> Path:
    manifest = dict(payload)
    manifest["phase"] = phase
    manifest["timestamp"] = _now_iso()
    manifest["run_paths"] = {k: str(v) for k, v in asdict(paths).items()}
    target = paths.manifests_dir / f"{phase}.json"
    write_json(target, manifest)
    return target


def _validate_docs_gate(repo_root: Path) -> None:
    validate_documents_structure(repo_root / "thesis" / "documents")
    validate_standalone_reference_contract(repo_root)


def prepare_thesis_run(
    *,
    repo_root: Path,
    run_name: str,
    block_size: str,
    raw_root: Path,
    runtime_family: str = CANONICAL_RUNTIME_FAMILY,
    legacy_base_path: Path | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    python_executable: str | None = None,
    skip_legacy_generation: bool = False,
    dry_run: bool = False,
    required_sequences: Sequence[str] | None = None,
    intra_raw_blocks_root: Path | None = None,
    require_intra_raw_blocks: bool = False,
    min_intra_raw_sequences: int = 2,
    auto_bootstrap_legacy_contract: bool = False,
    auto_generate_intra_raw_blocks: bool = False,
    videos_root: Path | None = None,
    video_ext: str = "yuv",
    frame_width: int = 3840,
    frame_height: int = 2160,
    partition_coord_scale: int = 4,
    generate_visual_samples: bool = False,
    visual_output_root: Path | None = None,
    visual_max_per_label_qp: int = 1,
    visual_scale: int = 8,
    visual_image_format: str = "jpg",
    generate_frame_overlay: bool = False,
    overlay_frame_number: int = 0,
    overlay_output_root: Path | None = None,
    overlay_image_format: str = "jpg",
    docs_gate: bool = True,
) -> RunPaths:
    """Prepare canonical thesis datasets and manifests."""
    assert_canonical_runtime_family(runtime_family)
    if docs_gate:
        _validate_docs_gate(repo_root)

    run_root = repo_root / "thesis" / "runs"
    paths = resolve_run_paths(run_root=run_root, run_name=run_name, block_size=block_size)
    ensure_dir(paths.run_dir)
    ensure_dir(paths.manifests_dir)

    if required_sequences is None:
        required_sequences = discover_sequences(raw_root)
    validated_inputs = validate_raw_input_contract(
        uvg_root=raw_root,
        required_sequences=required_sequences,
    )

    raw_blocks_generation: list[dict[str, object]] | None = None
    if auto_generate_intra_raw_blocks:
        if intra_raw_blocks_root is None:
            intra_raw_blocks_root = raw_root / "intra_raw_blocks"
        resolved_videos_root = videos_root or (repo_root / "videoset" / "uvg")
        generated = generate_intra_raw_blocks_from_partition(
            raw_root=raw_root,
            videos_root=resolved_videos_root,
            output_root=intra_raw_blocks_root,
            sequences=required_sequences,
            width=frame_width,
            height=frame_height,
            video_ext=video_ext,
            partition_coord_scale=partition_coord_scale,
            overwrite=False,
            dry_run=dry_run,
        )
        raw_blocks_generation = [
            {
                "sequence": item.sequence,
                "frame_number": item.frame_number,
                "block_size": item.block_size,
                "partition_entries": item.partition_entries,
                "output_file": item.output_file,
                "output_bytes": item.output_bytes,
            }
            for item in generated
        ]

    intra_raw_blocks_summary: dict[str, list[str]] | None = None
    if require_intra_raw_blocks:
        root = intra_raw_blocks_root or (raw_root / "intra_raw_blocks")
        intra_raw_blocks_summary = validate_intra_raw_blocks_contract(
            intra_raw_blocks_root=root,
            min_sequences=min_intra_raw_sequences,
        )

    summary = {
        "runtime_family": runtime_family,
        "block_size": block_size,
        "raw_root": str(raw_root),
        "legacy_base_path": str(legacy_base_path) if legacy_base_path else None,
        "skip_legacy_generation": skip_legacy_generation,
        "dry_run": dry_run,
        "required_sequences": list(required_sequences),
        "validated_inputs": [str(path) for path in validated_inputs],
        "require_intra_raw_blocks": require_intra_raw_blocks,
        "intra_raw_blocks_root": (
            str(intra_raw_blocks_root) if intra_raw_blocks_root else str(raw_root / "intra_raw_blocks")
        ),
        "intra_raw_blocks_sequences": intra_raw_blocks_summary,
        "auto_generate_intra_raw_blocks": auto_generate_intra_raw_blocks,
        "videos_root": str(videos_root) if videos_root else str(repo_root / "videoset" / "uvg"),
        "video_ext": video_ext,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "partition_coord_scale": partition_coord_scale,
        "raw_blocks_generation": raw_blocks_generation,
        "generate_visual_samples": generate_visual_samples,
        "visual_output_root": (
            str(visual_output_root)
            if visual_output_root is not None
            else str(paths.run_dir / "artifacts" / "visual_samples")
        ),
        "visual_max_per_label_qp": visual_max_per_label_qp,
        "visual_scale": visual_scale,
        "visual_image_format": visual_image_format,
        "generate_frame_overlay": generate_frame_overlay,
        "overlay_frame_number": overlay_frame_number,
        "overlay_output_root": (
            str(overlay_output_root)
            if overlay_output_root is not None
            else str(paths.run_dir / "artifacts" / "frame_overlays")
        ),
        "overlay_image_format": overlay_image_format,
    }

    if generate_frame_overlay:
        resolved_videos_root = videos_root or (repo_root / "videoset" / "uvg")
        overlay_root = overlay_output_root or (paths.run_dir / "artifacts" / "frame_overlays")
        overlay_stats = generate_frame_overlay_images(
            raw_root=raw_root,
            videos_root=resolved_videos_root,
            output_root=overlay_root,
            sequences=required_sequences,
            frame_number=overlay_frame_number,
            width=frame_width,
            height=frame_height,
            video_ext=video_ext,
            partition_coord_scale=partition_coord_scale,
            image_format=overlay_image_format,
            dry_run=dry_run,
        )
        summary["frame_overlay"] = {
            "enabled": True,
            "output_root": str(overlay_root),
            "frame_number": overlay_frame_number,
            "stats": [
                {
                    "sequence": item.sequence,
                    "frame_number": item.frame_number,
                    "output_file": item.output_file,
                    "total_regions": item.total_regions,
                    "class_counts": item.class_counts,
                    "block_size_counts": item.block_size_counts,
                    "coverage_ratio": item.coverage_ratio,
                }
                for item in overlay_stats
            ],
        }
    _write_phase_manifest(paths, "prepare_raw_contract", summary)

    if skip_legacy_generation:
        _write_phase_manifest(
            paths,
            "prepare",
            {
                **summary,
                "status": "raw contract validated; legacy generation skipped",
            },
        )
        return paths

    if legacy_base_path is None:
        if auto_bootstrap_legacy_contract:
            if intra_raw_blocks_summary is None:
                root = intra_raw_blocks_root or (raw_root / "intra_raw_blocks")
                intra_raw_blocks_summary = validate_intra_raw_blocks_contract(
                    intra_raw_blocks_root=root,
                    min_sequences=min_intra_raw_sequences,
                )

            bootstrap_sequences = sorted(intra_raw_blocks_summary.keys())
            legacy_base_path = paths.datasets_root / "legacy_bootstrap"
            ensure_dir(legacy_base_path)
            ensure_dir(legacy_base_path / "intra_raw_blocks")

            source_intra = intra_raw_blocks_root or (raw_root / "intra_raw_blocks")
            for sequence in bootstrap_sequences:
                for block in (8, 16, 32, 64):
                    src = source_intra / f"{sequence}_sample_{block}.txt"
                    dst = legacy_base_path / "intra_raw_blocks" / src.name
                    if not src.exists():
                        raise FileNotFoundError(f"Missing source sample for bootstrap: {src}")
                    if not dst.exists():
                        dst.write_bytes(src.read_bytes())

            bootstrap_stats = bootstrap_legacy_labels_qps_from_partition(
                uvg_root=raw_root,
                contract_root=legacy_base_path,
                sequences=bootstrap_sequences,
                drop_first_entry=False,
                align_to_samples=True,
                alignment_mode="rearrange_exact",
                frame_width=frame_width,
                frame_height=frame_height,
                partition_coord_scale=partition_coord_scale,
            )

            summary["legacy_bootstrap"] = {
                "enabled": True,
                "legacy_base_path": str(legacy_base_path),
                "stats": [
                    {
                        "sequence": item.sequence,
                        "block_size": item.block_size,
                        "labels_written": item.labels_written,
                        "qps_written": item.qps_written,
                        "sample_blocks": item.sample_blocks,
                        "alignment_mode": item.alignment_mode,
                        "dropped_entries": item.dropped_entries,
                        "trimmed_entries": item.trimmed_entries,
                    }
                    for item in bootstrap_stats
                ],
            }
        else:
            raise ValueError(
                "legacy_base_path is required unless --skip-legacy-generation is used "
                "or auto_bootstrap_legacy_contract is enabled."
            )

    visual_stats: list[dict[str, object]] | None = None
    if generate_visual_samples:
        visual_root = visual_output_root or (paths.run_dir / "artifacts" / "visual_samples")
        visual_results = generate_visual_samples_from_legacy_contract(
            legacy_base_path=legacy_base_path,
            output_root=visual_root,
            sequences=required_sequences,
            max_per_label_qp=visual_max_per_label_qp,
            scale=visual_scale,
            image_format=visual_image_format,
            raw_root=raw_root,
            partition_frame_file_name="partition_frame_0.txt",
            dry_run=dry_run,
        )
        visual_stats = [
            {
                "sequence": item.sequence,
                "block_size": item.block_size,
                "generated_images": item.generated_images,
                "unique_label_qp_pairs": item.unique_label_qp_pairs,
                "output_dir": item.output_dir,
            }
            for item in visual_results
        ]
        summary["visual_samples"] = {
            "enabled": True,
            "output_root": str(visual_root),
            "max_per_label_qp": visual_max_per_label_qp,
            "scale": visual_scale,
            "image_format": visual_image_format,
            "stats": visual_stats,
        }

    python_bin = _resolve_python_executable(repo_root, python_executable)
    prepare_cmd = [
        python_bin,
        str(repo_root / "thesis" / "scripts" / "prepare_dataset.py"),
        "--base-path",
        str(legacy_base_path),
        "--block-size",
        block_size,
        "--output-dir",
        str(paths.v7_dataset_dir),
        "--test-ratio",
        str(test_ratio),
        "--seed",
        str(seed),
    ]
    prepare_stage3_cmd = [
        python_bin,
        str(repo_root / "thesis" / "scripts" / "prepare_stage3_datasets.py"),
        "--base-path",
        str(legacy_base_path),
        "--block-size",
        block_size,
        "--output-base",
        str(paths.v7_stage3_root),
        "--test-ratio",
        str(test_ratio),
        "--seed",
        str(seed),
    ]

    if not dry_run:
        run_command(prepare_cmd, cwd=repo_root)
        run_command(prepare_stage3_cmd, cwd=repo_root)
        flow_validation = validate_expected_raw_flow(
            raw_root=raw_root,
            legacy_base_path=legacy_base_path,
            output_json=paths.manifests_dir / "flow_validation.json",
        )
        if not flow_validation["ok"]:
            joined = "\n".join(flow_validation["errors"])
            raise RuntimeError(
                "Expected raw flow validation failed after prepare:\n" + joined
            )
        validate_prepare_handoff(paths)

    _write_phase_manifest(
        paths,
        "prepare",
        {
            **summary,
            "status": "ok" if not dry_run else "dry-run",
            "commands": [prepare_cmd, prepare_stage3_cmd],
        },
    )
    return paths


def train_thesis_run(
    *,
    repo_root: Path,
    run_name: str,
    block_size: str,
    runtime_family: str = CANONICAL_RUNTIME_FAMILY,
    device: str = "cuda",
    epochs_stage2: int = 100,
    epochs_stage3_rect: int = 30,
    epochs_stage3_ab_binary: int = 50,
    batch_size: int = 128,
    seed: int = 42,
    python_executable: str | None = None,
    dry_run: bool = False,
    docs_gate: bool = True,
) -> RunPaths:
    """Train canonical thesis models (Stage 1-3)."""
    assert_canonical_runtime_family(runtime_family)
    if docs_gate:
        _validate_docs_gate(repo_root)
    validate_conv_adapter_frozen_backbone_contract(repo_root)

    run_root = repo_root / "thesis" / "runs"
    paths = resolve_run_paths(run_root=run_root, run_name=run_name, block_size=block_size)
    ensure_dir(paths.run_dir)
    ensure_dir(paths.manifests_dir)
    ensure_dir(paths.training_root)

    validate_prepare_handoff(paths)
    python_bin = _resolve_python_executable(repo_root, python_executable)

    stage2_cmd = [
        python_bin,
        str(repo_root / "thesis" / "scripts" / "train_adapter_solution.py"),
        "--dataset-dir",
        str(paths.v7_dataset_dir),
        "--output-dir",
        str(paths.solution1_training_dir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs_stage2),
        "--seed",
        str(seed),
    ]

    stage3_rect_cmd = [
        python_bin,
        str(repo_root / "thesis" / "scripts" / "train_stage3_rect.py"),
        "--dataset-dir",
        str(paths.v7_stage3_rect_dir),
        "--stage2-checkpoint",
        str(paths.stage2_checkpoint),
        "--output",
        str(paths.training_root / "stage3_rect"),
        "--epochs",
        str(epochs_stage3_rect),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--seed",
        str(seed),
        "--fix-batchnorm",
    ]

    stage3_ab_binary_cmd = [
        python_bin,
        str(repo_root / "thesis" / "scripts" / "train_stage3_ab_binary.py"),
        "--dataset-dir",
        str(paths.v7_dataset_dir),
        "--stage2-checkpoint",
        str(paths.stage2_checkpoint),
        "--output-dir",
        str(paths.training_root / "stage3_ab_binary"),
        "--epochs",
        str(epochs_stage3_ab_binary),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]

    if not dry_run:
        run_command(stage2_cmd, cwd=repo_root)
        run_command(stage3_rect_cmd, cwd=repo_root)
        run_command(stage3_ab_binary_cmd, cwd=repo_root)
        validate_train_handoff(paths)

    _write_phase_manifest(
        paths,
        "train",
        {
            "runtime_family": runtime_family,
            "block_size": block_size,
            "dry_run": dry_run,
            "device": device,
            "commands": [stage2_cmd, stage3_rect_cmd, stage3_ab_binary_cmd],
            "status": "ok" if not dry_run else "dry-run",
        },
    )
    return paths


def evaluate_thesis_run(
    *,
    repo_root: Path,
    run_name: str,
    block_size: str,
    runtime_family: str = CANONICAL_RUNTIME_FAMILY,
    split: str = "val",
    batch_size: int = 128,
    device: str = "cuda",
    python_executable: str | None = None,
    dry_run: bool = False,
    docs_gate: bool = True,
) -> RunPaths:
    """Evaluate canonical thesis hierarchical pipeline."""
    assert_canonical_runtime_family(runtime_family)
    if docs_gate:
        _validate_docs_gate(repo_root)
    validate_conv_adapter_frozen_backbone_contract(repo_root)

    run_root = repo_root / "thesis" / "runs"
    paths = resolve_run_paths(run_root=run_root, run_name=run_name, block_size=block_size)
    ensure_dir(paths.run_dir)
    ensure_dir(paths.manifests_dir)
    ensure_dir(paths.evaluation_pipeline_dir)

    validate_train_handoff(paths)
    python_bin = _resolve_python_executable(repo_root, python_executable)

    eval_cmd = [
        python_bin,
        str(repo_root / "thesis" / "scripts" / "evaluate_pipeline_ab_binary.py"),
        "--stage1-checkpoint",
        str(paths.stage1_checkpoint),
        "--stage2-checkpoint",
        str(paths.stage2_checkpoint),
        "--stage3-rect-checkpoint",
        str(paths.stage3_rect_checkpoint),
        "--stage3-ab-checkpoint",
        str(paths.stage3_ab_binary_checkpoint),
        "--dataset-dir",
        str(paths.v7_dataset_dir),
        "--split",
        split,
        "--output",
        str(paths.evaluation_pipeline_dir),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]

    if not dry_run:
        run_command(eval_cmd, cwd=repo_root)

    _write_phase_manifest(
        paths,
        "evaluate",
        {
            "runtime_family": runtime_family,
            "block_size": block_size,
            "split": split,
            "dry_run": dry_run,
            "device": device,
            "command": eval_cmd,
            "status": "ok" if not dry_run else "dry-run",
        },
    )
    return paths


def run_end_to_end_thesis_flow(
    *,
    repo_root: Path,
    run_name: str,
    block_size: str,
    raw_root: Path,
    runtime_family: str = CANONICAL_RUNTIME_FAMILY,
    legacy_base_path: Path | None = None,
    split: str = "val",
    device: str = "cuda",
    test_ratio: float = 0.2,
    epochs_stage2: int = 100,
    epochs_stage3_rect: int = 30,
    epochs_stage3_ab_binary: int = 50,
    batch_size: int = 128,
    seed: int = 42,
    python_executable: str | None = None,
    skip_legacy_generation: bool = False,
    dry_run: bool = False,
    required_sequences: Sequence[str] | None = None,
    intra_raw_blocks_root: Path | None = None,
    require_intra_raw_blocks: bool = False,
    min_intra_raw_sequences: int = 2,
    auto_bootstrap_legacy_contract: bool = False,
    auto_generate_intra_raw_blocks: bool = False,
    videos_root: Path | None = None,
    video_ext: str = "yuv",
    frame_width: int = 3840,
    frame_height: int = 2160,
    partition_coord_scale: int = 4,
    generate_visual_samples: bool = False,
    visual_output_root: Path | None = None,
    visual_max_per_label_qp: int = 1,
    visual_scale: int = 8,
    visual_image_format: str = "jpg",
    generate_frame_overlay: bool = False,
    overlay_frame_number: int = 0,
    overlay_output_root: Path | None = None,
    overlay_image_format: str = "jpg",
    docs_gate: bool = True,
) -> RunPaths:
    """Run canonical thesis prepare -> train -> evaluate sequence."""
    paths = prepare_thesis_run(
        repo_root=repo_root,
        run_name=run_name,
        block_size=block_size,
        raw_root=raw_root,
        runtime_family=runtime_family,
        legacy_base_path=legacy_base_path,
        test_ratio=test_ratio,
        seed=seed,
        python_executable=python_executable,
        skip_legacy_generation=skip_legacy_generation,
        dry_run=dry_run,
        required_sequences=required_sequences,
        intra_raw_blocks_root=intra_raw_blocks_root,
        require_intra_raw_blocks=require_intra_raw_blocks,
        min_intra_raw_sequences=min_intra_raw_sequences,
        auto_bootstrap_legacy_contract=auto_bootstrap_legacy_contract,
        auto_generate_intra_raw_blocks=auto_generate_intra_raw_blocks,
        videos_root=videos_root,
        video_ext=video_ext,
        frame_width=frame_width,
        frame_height=frame_height,
        partition_coord_scale=partition_coord_scale,
        generate_visual_samples=generate_visual_samples,
        visual_output_root=visual_output_root,
        visual_max_per_label_qp=visual_max_per_label_qp,
        visual_scale=visual_scale,
        visual_image_format=visual_image_format,
        generate_frame_overlay=generate_frame_overlay,
        overlay_frame_number=overlay_frame_number,
        overlay_output_root=overlay_output_root,
        overlay_image_format=overlay_image_format,
        docs_gate=docs_gate,
    )

    train_thesis_run(
        repo_root=repo_root,
        run_name=run_name,
        block_size=block_size,
        runtime_family=runtime_family,
        device=device,
        epochs_stage2=epochs_stage2,
        epochs_stage3_rect=epochs_stage3_rect,
        epochs_stage3_ab_binary=epochs_stage3_ab_binary,
        batch_size=batch_size,
        seed=seed,
        python_executable=python_executable,
        dry_run=dry_run,
        docs_gate=docs_gate,
    )

    evaluate_thesis_run(
        repo_root=repo_root,
        run_name=run_name,
        block_size=block_size,
        runtime_family=runtime_family,
        split=split,
        batch_size=batch_size,
        device=device,
        python_executable=python_executable,
        dry_run=dry_run,
        docs_gate=docs_gate,
    )

    _write_phase_manifest(
        paths,
        "end_to_end",
        {
            "runtime_family": runtime_family,
            "block_size": block_size,
            "split": split,
            "dry_run": dry_run,
            "device": device,
            "status": "ok" if not dry_run else "dry-run",
        },
    )
    return paths
