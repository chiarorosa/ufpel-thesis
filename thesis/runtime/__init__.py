"""Canonical thesis runtime helpers."""

from .contracts import (
    CANONICAL_RUNTIME_DOC_VERSION,
    CANONICAL_RUNTIME_FAMILY,
    RunPaths,
    assert_canonical_runtime_family,
    discover_sequences,
    discover_intra_raw_block_sequences,
    ensure_dir,
    resolve_run_paths,
    validate_standalone_reference_contract,
    validate_documents_structure,
    validate_conv_adapter_frozen_backbone_contract,
    validate_intra_raw_blocks_contract,
    validate_prepare_handoff,
    validate_raw_input_contract,
    validate_train_handoff,
    write_json,
)
from .canonical import (
    evaluate_thesis_run,
    prepare_thesis_run,
    run_end_to_end_thesis_flow,
    train_thesis_run,
)
from .legacy_contract import (
    BLOCK_SIZE_TO_INDEX,
    LegacyBootstrapStats,
    bootstrap_legacy_labels_qps_from_partition,
)
from .raw_blocks import (
    RawBlockGenerationStats,
    generate_intra_raw_blocks_from_partition,
)
from .visual_samples import (
    VisualSampleStats,
    generate_visual_samples_from_legacy_contract,
)
from .frame_overlays import (
    FrameOverlayStats,
    generate_frame_overlay_images,
)
from .flow_validation import (
    BLOCK_SIZES,
    BlockFlowStats,
    validate_expected_raw_flow,
)
from .cleanup import (
    PROTECTED_ROOTS,
    cleanup_thesis_outputs,
    collect_cleanup_candidates,
    validate_cleanup_safety,
)
from .drive_bootstrap import (
    DEFAULT_UVG_DRIVE_URL,
    bootstrap_uvg_from_drive,
)
from .runner import run_command

__all__ = [
    "CANONICAL_RUNTIME_FAMILY",
    "BLOCK_SIZE_TO_INDEX",
    "BLOCK_SIZES",
    "CANONICAL_RUNTIME_DOC_VERSION",
    "RunPaths",
    "assert_canonical_runtime_family",
    "discover_sequences",
    "discover_intra_raw_block_sequences",
    "ensure_dir",
    "evaluate_thesis_run",
    "prepare_thesis_run",
    "resolve_run_paths",
    "validate_standalone_reference_contract",
    "run_end_to_end_thesis_flow",
    "run_command",
    "bootstrap_legacy_labels_qps_from_partition",
    "generate_intra_raw_blocks_from_partition",
    "generate_visual_samples_from_legacy_contract",
    "generate_frame_overlay_images",
    "train_thesis_run",
    "LegacyBootstrapStats",
    "RawBlockGenerationStats",
    "VisualSampleStats",
    "FrameOverlayStats",
    "BlockFlowStats",
    "DEFAULT_UVG_DRIVE_URL",
    "PROTECTED_ROOTS",
    "bootstrap_uvg_from_drive",
    "cleanup_thesis_outputs",
    "collect_cleanup_candidates",
    "validate_cleanup_safety",
    "validate_conv_adapter_frozen_backbone_contract",
    "validate_expected_raw_flow",
    "validate_documents_structure",
    "validate_intra_raw_blocks_contract",
    "validate_prepare_handoff",
    "validate_raw_input_contract",
    "validate_train_handoff",
    "write_json",
]
