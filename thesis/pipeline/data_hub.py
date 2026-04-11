"""
V6 Pipeline - Data Hub (standalone version)
Key changes:
- Stage 2 only has 3 classes: SPLIT, RECT, AB (removed NONE and 1TO4)
- Enhanced sampling strategies for AB classes
- Support for ensemble datasets
- No dependencies on v5 pipeline (standalone)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Constants (copied from v5, now standalone)
# ---------------------------------------------------------------------------
PARTITION_ID_TO_NAME: Dict[int, str] = {
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

PARTITION_NAME_TO_ID = {name: idx for idx, name in PARTITION_ID_TO_NAME.items()}

# Flatten architecture: 7 classes (NONE removed, HORZ_4/VERT_4 don't exist in dataset)
FLATTEN_ID_TO_NAME: Dict[int, str] = {
    0: "PARTITION_HORZ",
    1: "PARTITION_VERT",
    2: "PARTITION_SPLIT",
    3: "PARTITION_HORZ_A",
    4: "PARTITION_HORZ_B",
    5: "PARTITION_VERT_A",
    6: "PARTITION_VERT_B",
}

FLATTEN_NAME_TO_ID = {name: idx for idx, name in FLATTEN_ID_TO_NAME.items()}

BLOCK_SIZES = ("8", "16", "32", "64")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BlockRecord:
    """Holds raw numpy arrays for a single block size."""
    samples: np.ndarray  # (N, block_size, block_size, C)
    labels: np.ndarray   # (N,)
    qps: np.ndarray      # (N, 1)

    @property
    def block_size(self) -> int:
        return self.samples.shape[1]

    def to_torch(self) -> "TorchBlockRecord":
        # Normalize 10-bit data (0-1023) to [0, 1] range
        torchvision_order = np.transpose(self.samples, (0, 3, 1, 2)).astype(np.float32) / 1023.0
        return TorchBlockRecord(
            samples=torch.from_numpy(torchvision_order),
            labels=torch.from_numpy(self.labels.astype(np.int64)),
            qps=torch.from_numpy(self.qps.squeeze(-1).astype(np.float32)),
        )


@dataclass
class TorchBlockRecord:
    samples: torch.Tensor  # (N, C, H, W)
    labels: torch.Tensor   # int64
    qps: torch.Tensor      # float32

# ---------------------------------------------------------------------------
# File discovery and loading
# ---------------------------------------------------------------------------

def index_sequences(base_path: Path) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    """Enumerate sample/label/QP triplets per sequence and block size."""
    base_path = Path(base_path).expanduser().resolve()
    dirs = {
        "samples": base_path / "intra_raw_blocks",
        "labels": base_path / "labels",
        "qps": base_path / "qps",
    }
    for name, folder in dirs.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"Required directory missing: {folder} ({name})")

    inventory: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    sample_files = sorted(p for p in dirs["samples"].iterdir() if p.suffix == ".txt")

    sequence_names = {
        path.name.replace(".txt", "").split("_sample_")[0]
        for path in sample_files
        if "_sample_" in path.name
    }

    dir_map = {
        "sample": dirs["samples"],
        "label": dirs["labels"],
        "qps": dirs["qps"],
    }

    for seq_name in sorted(sequence_names):
        inventory[seq_name] = {}
        for block in BLOCK_SIZES:
            entry = {
                "sample": f"{seq_name}_sample_{block}.txt",
                "label": f"{seq_name}_labels_{block}_intra.txt",
                "qps": f"{seq_name}_qps_{block}_intra.txt",
            }
            resolved = {}
            for key, file in entry.items():
                folder = dir_map[key]
                resolved[key] = file if (folder / file).exists() else None
            inventory[seq_name][block] = resolved
    return inventory


def load_block_records(base_path: Path, block_size: str) -> BlockRecord:
    """Load every sample/label/qp tuple for a given block size into memory."""
    if block_size not in BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {BLOCK_SIZES}, got {block_size}")

    base_path = Path(base_path)
    index = index_sequences(base_path)

    samples: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    qps: List[np.ndarray] = []

    for seq_name, blocks in index.items():
        entry = blocks.get(block_size)
        if not entry:
            continue

        sample_file = entry.get("sample")
        label_file = entry.get("label")
        qp_file = entry.get("qps")
        if not (sample_file and label_file and qp_file):
            continue

        with open(base_path / "intra_raw_blocks" / sample_file, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint16)
        block = int(block_size)
        sample_array = raw.reshape(-1, block, block, 1)
        samples.append(sample_array)

        label_array = np.fromfile(base_path / "labels" / label_file, dtype=np.uint8, sep=" ")
        labels.append(label_array.reshape(-1))

        qp_array = np.fromfile(base_path / "qps" / qp_file, dtype=np.uint8, sep=" ")
        qps.append(qp_array.reshape(-1, 1))

    if not samples:
        raise RuntimeError(f"No samples found for block size {block_size}")

    stacked_samples = np.concatenate(samples, axis=0)
    stacked_labels = np.concatenate(labels, axis=0)
    stacked_qps = np.concatenate(qps, axis=0)

    return BlockRecord(
        samples=stacked_samples,
        labels=stacked_labels,
        qps=stacked_qps,
    )


def train_test_split(record: BlockRecord, test_ratio: float = 0.2, seed: int = 42) -> Tuple[BlockRecord, BlockRecord]:
    """Shuffle and split in-memory arrays into train/test partitions."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    total = record.samples.shape[0]
    indices = rng.permutation(total)
    split_point = int(total * (1 - test_ratio))

    def subset(idxs: np.ndarray) -> BlockRecord:
        return BlockRecord(
            samples=record.samples[idxs],
            labels=record.labels[idxs],
            qps=record.qps[idxs],
        )

    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return subset(train_idx), subset(test_idx)

# ---------------------------------------------------------------------------
# V6 Hierarchical groupings (MODIFIED)
# ---------------------------------------------------------------------------

# Stage 2 in V6: Only 3 classes (NONE filtered in Stage 1)
STAGE2_GROUPS_V6: Dict[str, Tuple[str, ...]] = {
    "SPLIT": ("PARTITION_SPLIT",),
    "RECT": ("PARTITION_HORZ", "PARTITION_VERT"),
    "AB": (
        "PARTITION_HORZ_A",
        "PARTITION_HORZ_B",
        "PARTITION_VERT_A",
        "PARTITION_VERT_B",
    ),
}

# Stage 3 groups (same as v5)
STAGE3_GROUPS_V6: Dict[str, Tuple[str, ...]] = {
    "RECT": ("PARTITION_HORZ", "PARTITION_VERT"),
    "AB": (
        "PARTITION_HORZ_A",
        "PARTITION_HORZ_B",
        "PARTITION_VERT_A",
        "PARTITION_VERT_B",
    ),
}

STAGE2_NAME_TO_ID_V6 = {name: i for i, name in enumerate(STAGE2_GROUPS_V6.keys())}

STAGE3_NAME_TO_ID_V6 = {
    head: {label: i for i, label in enumerate(group)} 
    for head, group in STAGE3_GROUPS_V6.items()
}


# ---------------------------------------------------------------------------
# V6 Label mapping (MODIFIED for 3-way Stage 2)
# ---------------------------------------------------------------------------

def map_to_stage1_v6(label_ids: np.ndarray) -> np.ndarray:
    """Binary: NONE (0) vs PARTITION (1)"""
    names = np.vectorize(PARTITION_ID_TO_NAME.get)(label_ids)
    return (names != "PARTITION_NONE").astype(np.uint8)


def map_to_stage2_v6(label_ids: np.ndarray) -> np.ndarray:
    """3-way: SPLIT (0), RECT (1), AB (2)"""
    names = np.vectorize(PARTITION_ID_TO_NAME.get)(label_ids)
    mapped = np.full(label_ids.shape, fill_value=-1, dtype=np.int16)  # Use int16 for -1
    
    for group_name, members in STAGE2_GROUPS_V6.items():
        mask = np.isin(names, members)
        mapped[mask] = STAGE2_NAME_TO_ID_V6[group_name]
    
    # Exclude NONE samples (they shouldn't reach Stage 2)
    # Also exclude 1TO4 (HORZ_4, VERT_4) as they're not in our groups
    valid_mask = mapped != -1
    return mapped, valid_mask


def map_to_stage3_v6(label_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """Stage 3 specialist heads"""
    names = np.vectorize(PARTITION_ID_TO_NAME.get)(label_ids)
    result: Dict[str, np.ndarray] = {}
    
    for head, members in STAGE3_GROUPS_V6.items():
        head_labels = np.full(label_ids.shape, fill_value=-1, dtype=np.int64)
        for idx, member in enumerate(members):
            head_labels[names == member] = idx
        result[head] = head_labels
    
    return result


# ---------------------------------------------------------------------------
# V6 Dataset with augmentation support
# ---------------------------------------------------------------------------

class HierarchicalBlockDatasetV6(Dataset):
    """V6 Dataset with support for stage-specific augmentation"""
    
    def __init__(
        self,
        record: TorchBlockRecord,
        stage1_labels: torch.Tensor,
        stage2_labels: torch.Tensor,
        stage3_labels: Dict[str, torch.Tensor],
        augmentation=None,
        stage: str = 'stage1'
    ):
        self.samples = record.samples
        self.labels_stage0 = record.labels
        self.qps = record.qps
        self.labels_stage1 = stage1_labels
        self.labels_stage2 = stage2_labels
        self.labels_stage3 = stage3_labels
        self.augmentation = augmentation
        self.stage = stage
    
    def __len__(self) -> int:
        return self.samples.shape[0]
    
    def __getitem__(self, idx: int):
        image = self.samples[idx]
        
        # Apply augmentation
        if self.augmentation:
            if self.stage == 'stage3_ab':
                # AB stage needs label-aware augmentation
                label_ab = self.labels_stage3['AB'][idx].item()
                image, label_ab = self.augmentation(image, label_ab)
                # Update label
                item_label_stage3_AB = torch.tensor(label_ab, dtype=torch.int64)
            else:
                image = self.augmentation(image)
                item_label_stage3_AB = self.labels_stage3['AB'][idx]
        else:
            item_label_stage3_AB = self.labels_stage3['AB'][idx]
        
        item = {
            "image": image,
            "qp": self.qps[idx],
            "label_stage0": self.labels_stage0[idx],
            "label_stage1": self.labels_stage1[idx],
            "label_stage2": self.labels_stage2[idx],
            "label_stage3_RECT": self.labels_stage3['RECT'][idx],
            "label_stage3_AB": item_label_stage3_AB,
        }
        
        return item


def build_hierarchical_dataset_v6(
    record: BlockRecord, 
    augmentation=None,
    stage: str = 'stage1'
) -> HierarchicalBlockDatasetV6:
    """Build V6 hierarchical dataset"""
    
    torch_record = record.to_torch()
    stage1_np = map_to_stage1_v6(record.labels)
    stage2_np, valid_mask = map_to_stage2_v6(record.labels)
    stage3_np = map_to_stage3_v6(record.labels)
    
    stage3_tensors = {
        head: torch.from_numpy(values.astype(np.int64)) 
        for head, values in stage3_np.items()
    }
    
    return HierarchicalBlockDatasetV6(
        record=torch_record,
        stage1_labels=torch.from_numpy(stage1_np.astype(np.int64)),
        stage2_labels=torch.from_numpy(stage2_np.astype(np.int64)),
        stage3_labels=stage3_tensors,
        augmentation=augmentation,
        stage=stage
    )


# ---------------------------------------------------------------------------
# Sampling strategies for imbalanced classes
# ---------------------------------------------------------------------------

def get_class_weights(labels: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """
    Compute class weights using effective number of samples
    (Cui et al., 2019)
    """
    unique, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique)
    
    # Effective number
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / weights.sum() * num_classes
    
    # Map to all samples
    sample_weights = np.zeros(len(labels))
    for class_id, weight in zip(unique, weights):
        sample_weights[labels == class_id] = weight
    
    return sample_weights


def create_balanced_sampler(labels: np.ndarray, oversample_factor: Dict[int, float] = None):
    """
    Create weighted sampler for imbalanced classes
    
    Args:
        labels: Class labels
        oversample_factor: Dict mapping class_id to oversampling factor
                          e.g., {0: 1.0, 1: 3.0, 2: 5.0}
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    if oversample_factor is None:
        # Inverse frequency weighting
        class_weights = 1.0 / counts
    else:
        # Custom oversampling
        class_weights = np.array([oversample_factor.get(c, 1.0) for c in unique])
    
    # Normalize
    class_weights = class_weights / class_weights.sum() * len(unique)
    
    # Assign weight to each sample
    sample_weights = np.zeros(len(labels))
    for class_id, weight in zip(unique, class_weights):
        sample_weights[labels == class_id] = weight
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_ab_oversampled_dataset(record: BlockRecord, oversample_factors: Dict[int, int]):
    """
    Create oversampled dataset for AB classes
    
    Args:
        record: Original BlockRecord
        oversample_factors: e.g., {0: 1, 1: 5, 2: 5, 3: 1} for HORZ_A, HORZ_B, VERT_A, VERT_B
    """
    # Get AB labels
    stage3_labels = map_to_stage3_v6(record.labels)
    ab_labels = stage3_labels['AB']
    
    # Find AB samples
    ab_mask = ab_labels >= 0
    ab_indices = np.where(ab_mask)[0]
    
    # Oversample
    oversampled_indices = []
    for idx in ab_indices:
        label = ab_labels[idx]
        factor = oversample_factors.get(int(label), 1)
        oversampled_indices.extend([idx] * factor)
    
    oversampled_indices = np.array(oversampled_indices)
    
    # Create new record
    return BlockRecord(
        samples=record.samples[oversampled_indices],
        labels=record.labels[oversampled_indices],
        qps=record.qps[oversampled_indices]
    )


# ---------------------------------------------------------------------------
# Stage-specific dataset filtering
# ---------------------------------------------------------------------------

def filter_for_stage2(record: BlockRecord) -> BlockRecord:
    """Filter out NONE samples (handled by Stage 1) and 1TO4"""
    names = np.vectorize(PARTITION_ID_TO_NAME.get)(record.labels)
    
    # Keep only SPLIT, RECT, AB samples
    valid_names = set()
    for members in STAGE2_GROUPS_V6.values():
        valid_names.update(members)
    
    mask = np.isin(names, list(valid_names))
    
    return BlockRecord(
        samples=record.samples[mask],
        labels=record.labels[mask],
        qps=record.qps[mask]
    )


def filter_for_stage3(record: BlockRecord, head: str) -> BlockRecord:
    """Filter for specific Stage 3 head (RECT or AB)"""
    if head not in STAGE3_GROUPS_V6:
        raise ValueError(f"Unknown head: {head}")
    
    valid_names = STAGE3_GROUPS_V6[head]
    names = np.vectorize(PARTITION_ID_TO_NAME.get)(record.labels)
    mask = np.isin(names, valid_names)
    
    return BlockRecord(
        samples=record.samples[mask],
        labels=record.labels[mask],
        qps=record.qps[mask]
    )


# ---------------------------------------------------------------------------
# Metadata and utilities
# ---------------------------------------------------------------------------

def save_metadata(path: Path, info: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)


def compute_class_distribution_v6(labels: Iterable[int]) -> Dict[str, float]:
    labels = list(labels)
    total = len(labels)
    counts: Dict[str, int] = {}
    for label in labels:
        name = PARTITION_ID_TO_NAME.get(int(label), "UNKNOWN")
        counts[name] = counts.get(name, 0) + 1
    return {name: count / total for name, count in counts.items()}


__all__ = [
    'PARTITION_ID_TO_NAME',
    'PARTITION_NAME_TO_ID',
    'FLATTEN_ID_TO_NAME',
    'FLATTEN_NAME_TO_ID',
    'STAGE2_GROUPS_V6',
    'STAGE3_GROUPS_V6',
    'STAGE2_NAME_TO_ID_V6',
    'STAGE3_NAME_TO_ID_V6',
    'map_to_stage1_v6',
    'map_to_stage2_v6',
    'map_to_stage3_v6',
    'HierarchicalBlockDatasetV6',
    'build_hierarchical_dataset_v6',
    'get_class_weights',
    'create_balanced_sampler',
    'create_ab_oversampled_dataset',
    'filter_for_stage2',
    'filter_for_stage3',
    'save_metadata',
    'compute_class_distribution_v6',
]
