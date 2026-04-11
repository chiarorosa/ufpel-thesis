"""
Script 001: Prepare Thesis Dataset
Adapts dataset for thesis architecture with key changes:
1. Stage 2 has only 3 classes (SPLIT, RECT, AB)
2. Filters out NONE (handled by Stage 1) and 1TO4
3. Saves metadata for each stage
4. Standalone thesis pipeline module usage
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json

# Import thesis pipeline utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.pipeline.data_hub import (
    load_block_records,
    train_test_split,
    build_hierarchical_dataset_v6,
    filter_for_stage2,
    filter_for_stage3,
    save_metadata,
    compute_class_distribution_v6,
    STAGE2_GROUPS_V6,
    STAGE3_GROUPS_V6
)

def prepare_v7_dataset(
    base_path: Path,
    block_size: str = "16",
    output_dir: Path = None,
    test_ratio: float = 0.2,
    seed: int = 42
):
    """
    Prepare thesis dataset from raw data
    
    Args:
        base_path: Path to raw data
        block_size: Block size to process
        output_dir: Where to save processed data
        test_ratio: Train/val split ratio
        seed: Random seed
    """
    
    print(f"\n{'='*70}")
    print(f"  Preparing V7 Dataset - Block Size {block_size}")
    print(f"{'='*70}\n")
    
    # Set default output dir
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / f"v7_dataset" / f"block_{block_size}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load raw data
    print(f"[1/6] Loading raw block data...")
    record = load_block_records(base_path, block_size)
    print(f"  Total samples: {len(record.samples)}")
    
    # Show original distribution
    dist_orig = compute_class_distribution_v6(record.labels)
    print(f"\n  Original distribution:")
    for name, ratio in sorted(dist_orig.items()):
        print(f"    {name:20s}: {ratio*100:6.2f}%")
    
    # 2. Split train/val
    print(f"\n[2/6] Splitting train/val ({1-test_ratio:.0%}/{test_ratio:.0%})...")
    train_record, val_record = train_test_split(record, test_ratio=test_ratio, seed=seed)
    print(f"  Train samples: {len(train_record.samples)}")
    print(f"  Val samples:   {len(val_record.samples)}")
    
    # 3. Build hierarchical datasets (for Stage 1 + 2)
    print(f"\n[3/6] Building hierarchical datasets...")
    train_dataset = build_hierarchical_dataset_v6(train_record, stage='stage1')
    val_dataset = build_hierarchical_dataset_v6(val_record, stage='stage1')
    
    # Save combined dataset (for Stage 1 and Stage 2)
    train_path = output_dir / "train.pt"
    val_path = output_dir / "val.pt"
    
    print(f"  Saving train to: {train_path}")
    torch.save({
        'samples': train_dataset.samples,
        'labels_stage0': train_dataset.labels_stage0,
        'labels_stage1': train_dataset.labels_stage1,
        'labels_stage2': train_dataset.labels_stage2,
        'labels_stage3_RECT': train_dataset.labels_stage3['RECT'],
        'labels_stage3_AB': train_dataset.labels_stage3['AB'],
        'qps': train_dataset.qps,
    }, train_path)
    
    print(f"  Saving val to: {val_path}")
    torch.save({
        'samples': val_dataset.samples,
        'labels_stage0': val_dataset.labels_stage0,
        'labels_stage1': val_dataset.labels_stage1,
        'labels_stage2': val_dataset.labels_stage2,
        'labels_stage3_RECT': val_dataset.labels_stage3['RECT'],
        'labels_stage3_AB': val_dataset.labels_stage3['AB'],
        'qps': val_dataset.qps,
    }, val_path)
    
    # 4. Compute Stage 2 distribution (3 classes only)
    print(f"\n[4/6] Computing Stage 2 statistics (SPLIT, RECT, AB)...")
    train_stage2_filtered = filter_for_stage2(train_record)
    val_stage2_filtered = filter_for_stage2(val_record)
    
    dist_stage2_train = compute_class_distribution_v6(train_stage2_filtered.labels)
    dist_stage2_val = compute_class_distribution_v6(val_stage2_filtered.labels)
    
    print(f"  Train Stage 2 ({len(train_stage2_filtered.samples)} samples):")
    for name, ratio in sorted(dist_stage2_train.items()):
        print(f"    {name:20s}: {ratio*100:6.2f}%")
    
    print(f"  Val Stage 2 ({len(val_stage2_filtered.samples)} samples):")
    for name, ratio in sorted(dist_stage2_val.items()):
        print(f"    {name:20s}: {ratio*100:6.2f}%")
    
    # 5. Compute Stage 3 statistics
    print(f"\n[5/6] Computing Stage 3 statistics...")
    
    for head in ['RECT', 'AB']:
        train_stage3 = filter_for_stage3(train_record, head)
        val_stage3 = filter_for_stage3(val_record, head)
        
        dist_train = compute_class_distribution_v6(train_stage3.labels)
        dist_val = compute_class_distribution_v6(val_stage3.labels)
        
        print(f"\n  Stage 3-{head} Train ({len(train_stage3.samples)} samples):")
        for name, ratio in sorted(dist_train.items()):
            print(f"    {name:20s}: {ratio*100:6.2f}%")
        
        print(f"  Stage 3-{head} Val ({len(val_stage3.samples)} samples):")
        for name, ratio in sorted(dist_val.items()):
            print(f"    {name:20s}: {ratio*100:6.2f}%")
    
    # 6. Save metadata
    print(f"\n[6/6] Saving metadata...")
    
    metadata = {
        'block_size': block_size,
        'total_samples': len(record.samples),
        'train_samples': len(train_record.samples),
        'val_samples': len(val_record.samples),
        'test_ratio': test_ratio,
        'seed': seed,
        'original_distribution': dist_orig,
        'stage2_groups': {k: list(v) for k, v in STAGE2_GROUPS_V6.items()},
        'stage3_groups': {k: list(v) for k, v in STAGE3_GROUPS_V6.items()},
        'train_stage2_distribution': dist_stage2_train,
        'val_stage2_distribution': dist_stage2_val,
    }
    
    metadata_path = output_dir / "metadata.json"
    save_metadata(metadata_path, metadata)
    print(f"  Saved to: {metadata_path}")
    
    # Calculate class weights for losses
    stage2_labels = train_dataset.labels_stage2.numpy()
    stage2_labels_valid = stage2_labels[stage2_labels >= 0]
    unique, counts = np.unique(stage2_labels_valid, return_counts=True)
    
    stage2_weights = {
        'class_counts': {int(c): int(cnt) for c, cnt in zip(unique, counts)},
        'class_names': ['SPLIT', 'RECT', 'AB']
    }
    
    print(f"\n  Stage 2 class weights (for loss):")
    for class_id, count in stage2_weights['class_counts'].items():
        class_name = stage2_weights['class_names'][class_id]
        print(f"    {class_name:10s}: {count:6d} samples")
    
    metadata['stage2_class_weights'] = stage2_weights
    save_metadata(metadata_path, metadata)
    
    print(f"\n{'='*70}")
    print(f"  ✅ V7 Dataset prepared successfully!")
    print(f"  📁 Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare V7 dataset")
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
        help="Block size to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: thesis/v7_dataset/block_<size>)"
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
    
    prepare_v7_dataset(
        base_path=Path(args.base_path),
        block_size=args.block_size,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
