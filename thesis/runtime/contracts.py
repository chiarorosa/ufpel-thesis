"""Contracts and path helpers for the canonical thesis runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import re


CANONICAL_RUNTIME_FAMILY = "conv_adapter_frozen_backbone"
CANONICAL_RUNTIME_DOC_VERSION = "thesis-runtime-v1"
LEGACY_REFERENCE_PATTERN = re.compile(r"pesquisa_v\d+", re.IGNORECASE)
NUMBERED_SCRIPT_PATTERN = re.compile(r"(?:thesis/scripts/)?\d{3}_[A-Za-z0-9_]+\.py")


@dataclass(frozen=True)
class RunPaths:
    """Resolved output paths for a canonical thesis run."""

    run_root: Path
    run_dir: Path
    manifests_dir: Path
    datasets_root: Path
    v7_dataset_dir: Path
    v7_stage3_root: Path
    v7_stage3_rect_dir: Path
    v7_stage3_ab_dir: Path
    training_root: Path
    solution1_training_dir: Path
    stage1_checkpoint: Path
    stage2_checkpoint: Path
    stage3_rect_checkpoint: Path
    stage3_ab_binary_checkpoint: Path
    evaluation_root: Path
    evaluation_pipeline_dir: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def assert_canonical_runtime_family(runtime_family: str) -> None:
    if runtime_family != CANONICAL_RUNTIME_FAMILY:
        raise ValueError(
            "Only canonical runtime family is supported: "
            f"{CANONICAL_RUNTIME_FAMILY}. Got: {runtime_family}"
        )


def discover_sequences(uvg_root: Path) -> list[str]:
    """Discover sequences that satisfy the thesis raw input contract."""
    if not uvg_root.exists():
        raise FileNotFoundError(f"UVG root does not exist: {uvg_root}")

    sequences: list[str] = []
    for child in sorted(uvg_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "partition_frame_0.txt").exists():
            sequences.append(child.name)
    return sequences


def discover_intra_raw_block_sequences(
    intra_raw_blocks_root: Path,
    block_sizes: Sequence[str] = ("8", "16", "32", "64"),
) -> dict[str, set[str]]:
    """Discover binary raw block files grouped by sequence and block size."""
    if not intra_raw_blocks_root.exists():
        raise FileNotFoundError(
            f"intra_raw_blocks root does not exist: {intra_raw_blocks_root}"
        )

    pattern = re.compile(r"^(?P<seq>.+)_sample_(?P<block>\d+)\.txt$")
    discovered: dict[str, set[str]] = {}
    allowed_blocks = set(block_sizes)

    for path in sorted(intra_raw_blocks_root.iterdir()):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        block = match.group("block")
        if block not in allowed_blocks:
            continue
        sequence = match.group("seq")
        discovered.setdefault(sequence, set()).add(block)

    return discovered


def validate_raw_input_contract(
    uvg_root: Path,
    required_sequences: Sequence[str] | None = None,
) -> list[Path]:
    """Validate `thesis/uvg/<sequence>/partition_frame_0.txt` contract."""
    if not uvg_root.exists():
        raise FileNotFoundError(f"UVG root does not exist: {uvg_root}")

    if required_sequences is None:
        required_sequences = discover_sequences(uvg_root)

    if not required_sequences:
        raise FileNotFoundError(
            "No valid sequences found under thesis/uvg. "
            "Expected at least one `<sequence>/partition_frame_0.txt`."
        )

    validated: list[Path] = []
    missing: list[Path] = []
    for sequence in required_sequences:
        partition_file = uvg_root / sequence / "partition_frame_0.txt"
        if partition_file.exists():
            validated.append(partition_file)
        else:
            missing.append(partition_file)

    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing required raw partition files:\n"
            f"{joined}\n"
            "Expected contract: thesis/uvg/<sequence>/partition_frame_0.txt"
        )

    return validated


def validate_intra_raw_blocks_contract(
    intra_raw_blocks_root: Path,
    block_sizes: Sequence[str] = ("8", "16", "32", "64"),
    min_sequences: int = 1,
) -> dict[str, list[str]]:
    """Validate binary raw block contract in `thesis/uvg/intra_raw_blocks`."""
    discovered = discover_intra_raw_block_sequences(
        intra_raw_blocks_root=intra_raw_blocks_root,
        block_sizes=block_sizes,
    )
    if len(discovered) < min_sequences:
        raise FileNotFoundError(
            "Not enough sequences in thesis/uvg/intra_raw_blocks. "
            f"Found {len(discovered)}, required at least {min_sequences}."
        )

    required = set(block_sizes)
    missing_by_sequence: dict[str, list[str]] = {}
    for sequence, found_blocks in discovered.items():
        missing = sorted(required - found_blocks)
        if missing:
            missing_by_sequence[sequence] = missing

    if missing_by_sequence:
        entries = []
        for seq, missing in sorted(missing_by_sequence.items()):
            entries.append(f"- {seq}: missing {', '.join(missing)}")
        raise FileNotFoundError(
            "intra_raw_blocks contract failed. Missing block sizes:\n" + "\n".join(entries)
        )

    return {seq: sorted(blocks) for seq, blocks in sorted(discovered.items())}


def resolve_run_paths(run_root: Path, run_name: str, block_size: str) -> RunPaths:
    run_root = run_root.resolve()
    run_dir = run_root / run_name

    manifests_dir = run_dir / "manifests"

    datasets_root = run_dir / "datasets"
    v7_dataset_dir = datasets_root / "v7_dataset" / f"block_{block_size}"
    v7_stage3_root = datasets_root / "v7_dataset_stage3"
    v7_stage3_rect_dir = v7_stage3_root / "RECT" / f"block_{block_size}"
    v7_stage3_ab_dir = v7_stage3_root / "AB" / f"block_{block_size}"

    training_root = run_dir / "training"
    solution1_training_dir = training_root / "solution1_adapter"
    stage1_checkpoint = solution1_training_dir / "stage1" / "stage1_model_best.pt"
    stage2_checkpoint = (
        solution1_training_dir / "stage2_adapter" / "stage2_adapter_model_best.pt"
    )
    stage3_rect_checkpoint = training_root / "stage3_rect" / "model_best.pt"
    stage3_ab_binary_checkpoint = training_root / "stage3_ab_binary" / "model_best.pt"

    evaluation_root = run_dir / "evaluation"
    evaluation_pipeline_dir = evaluation_root / "pipeline"

    return RunPaths(
        run_root=run_root,
        run_dir=run_dir,
        manifests_dir=manifests_dir,
        datasets_root=datasets_root,
        v7_dataset_dir=v7_dataset_dir,
        v7_stage3_root=v7_stage3_root,
        v7_stage3_rect_dir=v7_stage3_rect_dir,
        v7_stage3_ab_dir=v7_stage3_ab_dir,
        training_root=training_root,
        solution1_training_dir=solution1_training_dir,
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint,
        stage3_rect_checkpoint=stage3_rect_checkpoint,
        stage3_ab_binary_checkpoint=stage3_ab_binary_checkpoint,
        evaluation_root=evaluation_root,
        evaluation_pipeline_dir=evaluation_pipeline_dir,
    )


def validate_prepare_handoff(paths: RunPaths) -> None:
    required = [
        paths.v7_dataset_dir / "train.pt",
        paths.v7_dataset_dir / "val.pt",
        paths.v7_dataset_dir / "metadata.json",
        paths.v7_stage3_rect_dir / "train.pt",
        paths.v7_stage3_rect_dir / "val.pt",
        paths.v7_stage3_rect_dir / "metadata.json",
        paths.v7_stage3_ab_dir / "train_v1.pt",
        paths.v7_stage3_ab_dir / "train_v2.pt",
        paths.v7_stage3_ab_dir / "train_v3.pt",
        paths.v7_stage3_ab_dir / "val.pt",
        paths.v7_stage3_ab_dir / "metadata.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Prepare phase handoff validation failed. Missing artifacts:\n"
            f"{missing_text}"
        )


def validate_train_handoff(paths: RunPaths, include_stage1: bool = True) -> None:
    required = [
        paths.stage2_checkpoint,
        paths.stage3_rect_checkpoint,
        paths.stage3_ab_binary_checkpoint,
    ]
    if include_stage1:
        required.insert(0, paths.stage1_checkpoint)

    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Train phase handoff validation failed. Missing artifacts:\n"
            f"{missing_text}"
        )


def validate_documents_structure(documents_root: Path) -> list[Path]:
    required = [
        documents_root / "README.md",
        documents_root / "runtime-contract.json",
        documents_root / "baseline-inventory.md",
        documents_root / "keep-archive-remove-matrix.md",
        documents_root / "regression-criteria.md",
        documents_root / "raw-input-contract.md",
        documents_root / "architecture.md",
        documents_root / "workflow.md",
        documents_root / "migration-guide.md",
        documents_root / "regression-matrix.md",
        documents_root / "runtime-checklist.md",
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Documentation structure validation failed. Missing files:\n"
            f"{missing_text}"
        )

    contract_file = documents_root / "runtime-contract.json"
    with open(contract_file, "r", encoding="utf-8") as handle:
        contract = json.load(handle)

    version = contract.get("doc_version")
    if version != CANONICAL_RUNTIME_DOC_VERSION:
        raise ValueError(
            "Documentation contract version mismatch. "
            f"Expected {CANONICAL_RUNTIME_DOC_VERSION}, got {version}."
        )

    runtime_family = contract.get("runtime_family")
    if runtime_family != CANONICAL_RUNTIME_FAMILY:
        raise ValueError(
            "Documentation runtime family mismatch. "
            f"Expected {CANONICAL_RUNTIME_FAMILY}, got {runtime_family}."
        )

    expected_entrypoints = {
        "thesis/scripts/clean.py",
        "thesis/scripts/prepare_data.py",
        "thesis/scripts/train_pipeline.py",
        "thesis/scripts/evaluate_pipeline.py",
        "thesis/scripts/run_pipeline_end_to_end.py",
    }
    documented_entrypoints = set(contract.get("canonical_entrypoints", []))
    missing_entrypoints = sorted(expected_entrypoints - documented_entrypoints)
    if missing_entrypoints:
        joined = ", ".join(missing_entrypoints)
        raise ValueError(
            "Documentation contract missing canonical entrypoints: "
            f"{joined}"
        )

    return required


def _scan_file_patterns(path: Path, patterns: Sequence[tuple[str, re.Pattern[str]]]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    findings: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for label, pattern in patterns:
            match = pattern.search(line)
            if match:
                findings.append(f"{path}:{line_no} [{label}] {match.group(0)}")
    return findings


def validate_standalone_reference_contract(repo_root: Path) -> dict[str, int]:
    """Ensure thesis-critical runtime/docs remain standalone and non-numbered."""
    thesis_root = (repo_root / "thesis").resolve()
    if not thesis_root.exists():
        raise FileNotFoundError(f"Missing thesis root: {thesis_root}")

    legacy_scan_roots = [
        thesis_root / "runtime",
        thesis_root / "scripts",
        thesis_root / "pipeline",
        thesis_root / "documents",
    ]
    findings: list[str] = []

    for root in legacy_scan_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".md", ".json"}:
                continue
            findings.extend(
                _scan_file_patterns(path, [("legacy-reference", LEGACY_REFERENCE_PATTERN)])
            )

    canonical_surface_files = [
        thesis_root / "runtime" / "canonical.py",
        thesis_root / "runtime" / "contracts.py",
        thesis_root / "documents" / "runtime-contract.json",
        thesis_root / "documents" / "workflow.md",
        thesis_root / "documents" / "runtime-checklist.md",
        thesis_root / "documents" / "migration-guide.md",
        thesis_root / "documents" / "baseline-inventory.md",
    ]
    for path in canonical_surface_files:
        if not path.exists() or not path.is_file():
            continue
        findings.extend(
            _scan_file_patterns(path, [("numbered-script-surface", NUMBERED_SCRIPT_PATTERN)])
        )

    if findings:
        joined = "\n".join(findings)
        raise RuntimeError(
            "Standalone thesis reference guard failed. Forbidden references found:\n"
            f"{joined}"
        )

    legacy_count = sum(
        len(_scan_file_patterns(path, [("legacy-reference", LEGACY_REFERENCE_PATTERN)]))
        for root in legacy_scan_roots
        if root.exists()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md", ".json"}
    )
    numbered_count = sum(
        len(_scan_file_patterns(path, [("numbered-script-surface", NUMBERED_SCRIPT_PATTERN)]))
        for path in canonical_surface_files
        if path.exists() and path.is_file()
    )

    return {
        "legacy_reference_matches": legacy_count,
        "numbered_surface_matches": numbered_count,
    }


def validate_conv_adapter_frozen_backbone_contract(repo_root: Path) -> dict[str, bool]:
    """Static guard to ensure canonical training scripts keep thesis constraints."""
    checks: dict[str, bool] = {}

    stage2_script = repo_root / "thesis" / "scripts" / "train_adapter_solution.py"
    conv_adapter_module = repo_root / "thesis" / "pipeline" / "conv_adapter.py"
    stage3_rect_script = repo_root / "thesis" / "scripts" / "train_stage3_rect.py"
    stage3_ab_script = repo_root / "thesis" / "scripts" / "train_stage3_ab_binary.py"

    stage2_text = stage2_script.read_text(encoding="utf-8")
    conv_adapter_text = conv_adapter_module.read_text(encoding="utf-8")
    stage3_rect_text = stage3_rect_script.read_text(encoding="utf-8")
    stage3_ab_text = stage3_ab_script.read_text(encoding="utf-8")

    checks["stage2_uses_adapter_backbone"] = "AdapterBackbone" in stage2_text
    checks["stage2_freezes_backbone"] = "param.requires_grad = False" in conv_adapter_text
    checks["stage3_rect_freezes_backbone"] = "freeze_backbone=True" in stage3_rect_text
    checks["stage3_ab_freezes_backbone"] = "freeze_backbone=True" in stage3_ab_text

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        joined = ", ".join(failed)
        raise RuntimeError(
            "Conv-Adapter/frozen-backbone contract validation failed: "
            f"{joined}"
        )

    return checks
