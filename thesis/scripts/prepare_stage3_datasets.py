"""
Script 002: Prepare Stage 3 Datasets for Thesis
Creates specialized datasets for RECT and AB heads with:
- Oversampling for AB minority classes
- 3 ensemble versions for AB (different augmentation seeds)
- Standalone thesis pipeline module usage
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Import thesis pipeline utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.pipeline.data_hub import (
    load_block_records,
    train_test_split,
    filter_for_stage3,
    create_ab_oversampled_dataset,
    save_metadata,
    compute_class_distribution_v6,
    map_to_stage3_v6
)


def prepare_stage3_datasets(
    base_path: Path,
    block_size: str = "16",
    output_base: Path = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    ab_oversample_factors: dict = None
):
    """
    Prepare Stage 3 specialized datasets
    
    Args:
        base_path: Path to raw data
        block_size: Block size
        output_base: Base output directory
        test_ratio: Train/val split
        seed: Random seed
        ab_oversample_factors: Oversampling factors for AB classes
                              e.g., {0: 1, 1: 5, 2: 5, 3: 1}
    """
    
    print(f"\n{'='*70}")
    print(f"  Preparing Stage 3 Datasets - Block Size {block_size}")
    print(f"{'='*70}\n")
    
    # Default output
    if output_base is None:
        output_base = Path(__file__).parent.parent / "v7_dataset_stage3"
    
    # Default AB oversampling: 5x for minority classes (HORZ_B, VERT_A)
    if ab_oversample_factors is None:
        ab_oversample_factors = {
            0: 1,  # HORZ_A
            1: 5,  # HORZ_B (minority)
            2: 5,  # VERT_A (minority)
            3: 1,  # VERT_B
        }
    
    # Load and split data
    print(f"[1/4] Loading and splitting data...")
    record = load_block_records(base_path, block_size)
    train_record, val_record = train_test_split(record, test_ratio=test_ratio, seed=seed)
    
    # ============ RECT Dataset ============
    print(f"\n[2/4] Preparing RECT dataset (HORZ vs VERT)...")
    
    train_rect = filter_for_stage3(train_record, 'RECT')
    val_rect = filter_for_stage3(val_record, 'RECT')
    
    print(f"  Train samples: {len(train_rect.samples)}")
    print(f"  Val samples:   {len(val_rect.samples)}")
    
    dist_train = compute_class_distribution_v6(train_rect.labels)
    dist_val = compute_class_distribution_v6(val_rect.labels)
    
    print(f"  Train distribution:")
    for name, ratio in sorted(dist_train.items()):
        print(f"    {name:20s}: {ratio*100:6.2f}%")
    
    # Save RECT dataset
    rect_dir = output_base / "RECT" / f"block_{block_size}"
    rect_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'samples': train_rect.samples,
        'labels': train_rect.labels,
        'qps': train_rect.qps,
    }, rect_dir / "train.pt")
    
    torch.save({
        'samples': val_rect.samples,
        'labels': val_rect.labels,
        'qps': val_rect.qps,
    }, rect_dir / "val.pt")
    
    save_metadata(rect_dir / "metadata.json", {
        'block_size': block_size,
        'stage': 'stage3_rect',
        'train_samples': len(train_rect.samples),
        'val_samples': len(val_rect.samples),
        'train_distribution': dist_train,
        'val_distribution': dist_val,
        'classes': ['PARTITION_HORZ', 'PARTITION_VERT']
    })
    
    print(f"  ✅ Saved to: {rect_dir}")
    
    # ============ AB Dataset (Base) ============
    print(f"\n[3/4] Preparing AB dataset (HORZ_A, HORZ_B, VERT_A, VERT_B)...")
    
    train_ab = filter_for_stage3(train_record, 'AB')
    val_ab = filter_for_stage3(val_record, 'AB')
    
    print(f"  Original train samples: {len(train_ab.samples)}")
    print(f"  Original val samples:   {len(val_ab.samples)}")
    
    # Show AB label distribution
    ab_labels_train = map_to_stage3_v6(train_ab.labels)['AB']
    ab_labels_val = map_to_stage3_v6(val_ab.labels)['AB']
    
    unique_train, counts_train = np.unique(ab_labels_train, return_counts=True)
    unique_val, counts_val = np.unique(ab_labels_val, return_counts=True)
    
    ab_class_names = ['HORZ_A', 'HORZ_B', 'VERT_A', 'VERT_B']
    
    print(f"  Original train AB distribution:")
    for class_id, count in zip(unique_train, counts_train):
        print(f"    {ab_class_names[class_id]:10s}: {count:6d} samples")
    
    # Apply oversampling
    print(f"\n  Applying oversampling (factors: {ab_oversample_factors})...")
    train_ab_oversampled = create_ab_oversampled_dataset(train_ab, ab_oversample_factors)
    
    ab_labels_oversampled = map_to_stage3_v6(train_ab_oversampled.labels)['AB']
    unique_over, counts_over = np.unique(ab_labels_oversampled, return_counts=True)
    
    print(f"  Oversampled train AB distribution:")
    for class_id, count in zip(unique_over, counts_over):
        print(f"    {ab_class_names[class_id]:10s}: {count:6d} samples")
    
    # ============ AB Ensemble (3 versions) ============
    print(f"\n[4/4] Creating 3 ensemble versions for AB...")
    
    ab_dir = output_base / "AB" / f"block_{block_size}"
    ab_dir.mkdir(parents=True, exist_ok=True)
    
    # Save validation set (same for all)
    torch.save({
        'samples': val_ab.samples,
        'labels': val_ab.labels,
        'qps': val_ab.qps,
    }, ab_dir / "val.pt")
    
    # Create 3 different training sets with different random seeds
    for i in range(1, 4):
        print(f"\n  Creating ensemble version {i}...")
        
        # Use different seed for shuffling
        ensemble_seed = seed + i * 100
        np.random.seed(ensemble_seed)
        
        # Shuffle oversampled data
        indices = np.random.permutation(len(train_ab_oversampled.samples))
        
        train_data = {
            'samples': train_ab_oversampled.samples[indices],
            'labels': train_ab_oversampled.labels[indices],
            'qps': train_ab_oversampled.qps[indices],
            'ensemble_version': i,
            'random_seed': ensemble_seed
        }
        
        train_path = ab_dir / f"train_v{i}.pt"
        torch.save(train_data, train_path)
        print(f"    Saved to: {train_path}")
    
    # Save AB metadata
    save_metadata(ab_dir / "metadata.json", {
        'block_size': block_size,
        'stage': 'stage3_ab',
        'train_samples_original': len(train_ab.samples),
        'train_samples_oversampled': len(train_ab_oversampled.samples),
        'val_samples': len(val_ab.samples),
        'oversample_factors': ab_oversample_factors,
        'ensemble_versions': 3,
        'classes': ab_class_names,
        'original_train_distribution': {
            ab_class_names[c]: int(cnt) for c, cnt in zip(unique_train, counts_train)
        },
        'oversampled_train_distribution': {
            ab_class_names[c]: int(cnt) for c, cnt in zip(unique_over, counts_over)
        },
        'val_distribution': {
            ab_class_names[c]: int(cnt) for c, cnt in zip(unique_val, counts_val)
        }
    })
    
    print(f"  ✅ AB ensemble saved to: {ab_dir}")
    
    print(f"\n{'='*70}")
    print(f"  ✅ Stage 3 datasets prepared successfully!")
    print(f"  📁 Output directory: {output_base}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Stage 3 datasets")
    parser.add_argument(
        "--base-path",
        type=str,
        default="/home/chiarorosa/experimentos/uvg/",
        help="Path to raw dataset"
    )
    parser.add_argument(
        "--block-size",
        type=str,
        default="16",
        choices=["8", "16", "32", "64"],
        help="Block size"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Output base directory (default: thesis/v7_dataset_stage3)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Train/val split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    prepare_stage3_datasets(
        base_path=Path(args.base_path),
        block_size=args.block_size,
        output_base=Path(args.output_base) if args.output_base else None,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
