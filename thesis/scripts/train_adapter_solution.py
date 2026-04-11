"""
Script 020: Train Solution 1 - Conv-Adapter
Parameter Efficient Transfer Learning for AV1 Partition Prediction

Based on: Chen et al. (CVPR 2024) - "Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets"

Workflow:
1. Train Stage 1 normally (backbone + binary head)
2. Freeze backbone after Stage 1
3. Train Stage 2 with Conv-Adapter (3.5% params) + 3-way head
4. Train Stage 3 with Conv-Adapter + specialist heads

Expected improvements:
- Stage 2 F1: 46% → 60-65% (solves negative transfer)
- Parameter efficiency: 3.5% trainable vs 100% full fine-tuning
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import json
from tqdm import tqdm
import argparse

# Import thesis pipeline (standalone)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.pipeline.data_hub import (
    build_hierarchical_dataset_v6,
    create_balanced_sampler,
    get_class_weights,
    STAGE2_GROUPS_V6,
    compute_class_distribution_v6
)
from thesis.pipeline.backbone import create_stage1_head, create_stage2_head, create_stage3_rect_head, create_stage3_ab_head
from thesis.pipeline.conv_adapter import ConvAdapter, AdapterBackbone
from thesis.pipeline.losses import FocalLoss, ClassBalancedFocalLoss
from thesis.pipeline.evaluation import MetricsCalculator


def train_stage1_adapter_solution(
    dataset_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 128,
    epochs: int = 100,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    patience: int = 15,
    seed: int = 42
):
    """
    Train Stage 1 for Conv-Adapter solution
    Trains backbone + binary head normally (end-to-end)
    """
    print(f"\n{'='*80}")
    print(f"  SOLUTION 1 - Conv-Adapter: Training Stage 1")
    print(f"{'='*80}\n")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset (already processed tensors)
    print(f"[1/6] Loading dataset...")
    train_data = torch.load(dataset_dir / "train.pt")
    val_data = torch.load(dataset_dir / "val.pt")

    # Create datasets directly from tensors
    from torch.utils.data import TensorDataset
    
    train_dataset = TensorDataset(
        train_data['samples'],
        train_data['labels_stage1'],  # Binary labels for Stage 1
        train_data['qps']
    )
    
    val_dataset = TensorDataset(
        val_data['samples'],
        val_data['labels_stage1'],  # Binary labels for Stage 1
        val_data['qps']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create model
    print(f"\n[2/6] Creating Stage 1 model...")
    from thesis.pipeline.backbone import ImprovedBackbone, ClassificationHead
    
    backbone = ImprovedBackbone(pretrained=True)
    head = ClassificationHead(num_classes=2, hidden_dims=[256], dropout=0.3)
    
    class Stage1Model(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
    
    model = Stage1Model(backbone, head)
    model = model.to(device)

    # Loss function
    criterion = FocalLoss(gamma=2.0, alpha=0.25)

    # Optimizer with discriminative learning rates
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.head.parameters(), 'lr': lr_head}
    ])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    print(f"\n[3/6] Training Stage 1...")
    best_f1 = 0.0
    patience_counter = 0

    history = {
        'train_loss': [], 'train_f1': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
        'lr_backbone': [], 'lr_head': []
    }

    metrics_calc = MetricsCalculator

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images, targets, _ = batch  # Unpack (images, labels, qps)
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images, targets, _ = batch  # Unpack (images, labels, qps)
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        train_metrics = MetricsCalculator.calculate_classification_metrics(
            np.array(train_targets), np.array(train_preds)
        )
        val_metrics = MetricsCalculator.calculate_classification_metrics(
            np.array(val_targets), np.array(val_preds)
        )

        # Update history
        history['train_loss'].append(np.mean(train_losses))
        history['train_f1'].append(train_metrics['f1_macro'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(np.mean(val_losses))
        history['val_f1'].append(val_metrics['f1_macro'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr_backbone'].append(optimizer.param_groups[0]['lr'])
        history['lr_head'].append(optimizer.param_groups[1]['lr'])

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {history['train_loss'][-1]:.4f}, F1: {history['train_f1'][-1]:.4f}")
        print(f"  Val   - Loss: {history['val_loss'][-1]:.4f}, F1: {history['val_f1'][-1]:.4f}")

        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'history': history
            }
            torch.save(checkpoint, output_dir / "stage1_model_best.pt")
            print(f"  ✅ Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        scheduler.step(val_metrics['f1_macro'])

    # Save final model and history
    torch.save(history, output_dir / "stage1_history.pt")

    final_metrics = {
        'best_f1': best_f1,
        'final_train_f1': history['train_f1'][-1],
        'final_val_f1': history['val_f1'][-1],
        'epochs_trained': len(history['train_loss']),
        'early_stopped': patience_counter >= patience
    }

    with open(output_dir / "stage1_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  Stage 1 training completed!")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Model saved: {output_dir / 'stage1_model_best.pt'}")
    print(f"{'='*80}\n")

    return best_f1


def train_stage2_with_adapter(
    dataset_dir: Path,
    stage1_checkpoint: Path,
    output_dir: Path,
    adapter_reduction: int = 4,
    device: str = "cuda",
    batch_size: int = 128,
    epochs: int = 100,
    lr_adapter: float = 1e-3,
    lr_head: float = 1e-3,
    patience: int = 15,
    seed: int = 42
):
    """
    Train Stage 2 with Conv-Adapter
    Backbone frozen, only adapter (3.5% params) + head trainable
    """
    print(f"\n{'='*80}")
    print(f"  SOLUTION 1 - Conv-Adapter: Training Stage 2")
    print(f"  Adapter reduction: {adapter_reduction}")
    print(f"{'='*80}\n")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load Stage 1 checkpoint
    print(f"[1/7] Loading Stage 1 checkpoint...")
    stage1_ckpt = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)

    # Load dataset (already processed tensors)
    print(f"[2/7] Loading dataset...")
    train_data = torch.load(dataset_dir / "train.pt")
    val_data = torch.load(dataset_dir / "val.pt")

    # Filter for Stage 2 (remove NONE samples)
    train_stage2_mask = train_data['labels_stage2'] >= 0
    val_stage2_mask = val_data['labels_stage2'] >= 0
    
    train_stage2_data = {
        'samples': train_data['samples'][train_stage2_mask],
        'labels': train_data['labels_stage2'][train_stage2_mask],
        'qps': train_data['qps'][train_stage2_mask]
    }
    
    val_stage2_data = {
        'samples': val_data['samples'][val_stage2_mask],
        'labels': val_data['labels_stage2'][val_stage2_mask],
        'qps': val_data['qps'][val_stage2_mask]
    }

    # Create datasets directly from tensors
    from torch.utils.data import TensorDataset
    
    train_dataset = TensorDataset(
        train_stage2_data['samples'],
        train_stage2_data['labels'],  # Stage 2 labels (0,1,2)
        train_stage2_data['qps']
    )
    
    val_dataset = TensorDataset(
        val_stage2_data['samples'],
        val_stage2_data['labels'],  # Stage 2 labels (0,1,2)
        val_stage2_data['qps']
    )

    # Create balanced sampler for Stage 2 (3 classes)
    stage2_labels = train_stage2_data['labels'].numpy()
    sampler = create_balanced_sampler(stage2_labels)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_dataset)} (balanced)")
    print(f"  Val samples: {len(val_dataset)}")

    # Show Stage 2 distribution
    stage2_names = list(STAGE2_GROUPS_V6.keys())
    train_dist = compute_class_distribution_v6(train_stage2_data['labels'].numpy())
    val_dist = compute_class_distribution_v6(val_stage2_data['labels'].numpy())

    print(f"\n  Stage 2 distribution:")
    print(f"  Train: {[f'{k}: {v*100:.1f}%' for k, v in train_dist.items()]}")
    print(f"  Val:   {[f'{k}: {v*100:.1f}%' for k, v in val_dist.items()]}")

    # Create model with adapter
    print(f"\n[3/7] Creating Stage 2 model with Conv-Adapter...")

    # Load backbone and head from Stage 1 checkpoint
    from thesis.pipeline.backbone import ImprovedBackbone
    backbone = ImprovedBackbone(pretrained=False)  # Will load weights from checkpoint
    backbone_state_dict = {k: v for k, v in stage1_ckpt['model_state_dict'].items() if k.startswith('backbone.')}
    backbone_state_dict = {k.replace('backbone.', '', 1): v for k, v in backbone_state_dict.items()}
    backbone.load_state_dict(backbone_state_dict)
    
    head = create_stage1_head()
    head_state_dict = {k: v for k, v in stage1_ckpt['model_state_dict'].items() if k.startswith('head.')}
    head_state_dict = {k.replace('head.', '', 1): v for k, v in head_state_dict.items()}
    head.load_state_dict(head_state_dict)

    # Create adapter backbone (frozen backbone + adapter)
    adapter_config = {
        'reduction': adapter_reduction,
        'layers': ['layer3', 'layer4'],  # Adapt deep layers as per Chen et al.
        'variant': 'conv_parallel'
    }
    adapter_backbone = AdapterBackbone(backbone, adapter_config=adapter_config)

    # Create Stage 2 head
    stage2_head = create_stage2_head()

    # Create wrapper to extract features from AdapterBackbone
    class AdapterBackboneWrapper(nn.Module):
        def __init__(self, adapter_backbone):
            super().__init__()
            self.adapter_backbone = adapter_backbone
        
        def forward(self, x):
            features, _ = self.adapter_backbone(x)
            return features
    
    adapter_backbone_wrapper = AdapterBackboneWrapper(adapter_backbone)
    
    # Combine
    model = nn.Sequential(adapter_backbone_wrapper, stage2_head)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_efficiency = trainable_params / total_params * 100

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter efficiency: {param_efficiency:.1f}%")

    # Loss function with class balancing
    # Count samples per class for ClassBalancedFocalLoss
    unique_labels, counts = np.unique(stage2_labels, return_counts=True)
    samples_per_class = np.zeros(3)  # 3 classes in Stage 2
    for label, count in zip(unique_labels, counts):
        samples_per_class[label] = count
    
    criterion = ClassBalancedFocalLoss(
        samples_per_class=samples_per_class,
        beta=0.9999,
        gamma=2.0
    )

    # Optimizer (only adapter and head)
    optimizer = optim.AdamW([
        {'params': adapter_backbone.adapters.parameters(), 'lr': lr_adapter},
        {'params': stage2_head.parameters(), 'lr': lr_head}
    ])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    print(f"\n[4/7] Training Stage 2 with Conv-Adapter...")
    best_f1 = 0.0
    patience_counter = 0

    history = {
        'train_loss': [], 'train_f1': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
        'lr_adapter': [], 'lr_head': []
    }

    metrics_calc = MetricsCalculator

    for epoch in range(epochs):
        # Training
        model.train()
        # CRITICAL FIX (Issue #2): Force backbone BatchNorm to eval mode
        # Prevents distribution shift between train/val since backbone is frozen
        adapter_backbone.backbone.eval()
        
        train_losses = []
        train_preds = []
        train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images, targets, _ = batch  # Unpack (images, labels, qps)
            images = images.to(device)
            targets = targets.to(device)

            # Skip samples not in Stage 2 (-1 labels)
            valid_mask = targets >= 0
            if valid_mask.sum() == 0:
                continue

            images = images[valid_mask]
            targets = targets[valid_mask]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images, targets, _ = batch  # Unpack (images, labels, qps)
                images = images.to(device)
                targets = targets.to(device)

                valid_mask = targets >= 0
                if valid_mask.sum() == 0:
                    continue

                images = images[valid_mask]
                targets = targets[valid_mask]

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        if train_preds and val_preds:
            train_metrics = MetricsCalculator.calculate_classification_metrics(
                np.array(train_targets), np.array(train_preds)
            )
            val_metrics = MetricsCalculator.calculate_classification_metrics(
                np.array(val_targets), np.array(val_preds)
            )

            # Update history
            history['train_loss'].append(np.mean(train_losses))
            history['train_f1'].append(train_metrics['f1_macro'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(np.mean(val_losses))
            history['val_f1'].append(val_metrics['f1_macro'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['lr_adapter'].append(optimizer.param_groups[0]['lr'])
            history['lr_head'].append(optimizer.param_groups[1]['lr'])

            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train - Loss: {history['train_loss'][-1]:.4f}, F1: {history['train_f1'][-1]:.4f}")
            print(f"  Val   - Loss: {history['val_loss'][-1]:.4f}, F1: {history['val_f1'][-1]:.4f}")

            # Save best model
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'adapter_backbone_state_dict': adapter_backbone.state_dict(),
                    'stage2_head_state_dict': stage2_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_f1,
                    'adapter_reduction': adapter_reduction,
                    'param_efficiency': param_efficiency,
                    'history': history
                }
                torch.save(checkpoint, output_dir / "stage2_adapter_model_best.pt")
                print(f"  ✅ Saved best model (F1: {best_f1:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            scheduler.step(val_metrics['f1_macro'])

    # Save final model and history
    torch.save(history, output_dir / "stage2_adapter_history.pt")

    final_metrics = {
        'best_f1': best_f1,
        'final_train_f1': history['train_f1'][-1] if history['train_f1'] else 0,
        'final_val_f1': history['val_f1'][-1] if history['val_f1'] else 0,
        'epochs_trained': len(history['train_loss']),
        'early_stopped': patience_counter >= patience,
        'adapter_reduction': adapter_reduction,
        'param_efficiency': param_efficiency,
        'total_params': total_params,
        'trainable_params': trainable_params
    }

    with open(output_dir / "stage2_adapter_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  Stage 2 Conv-Adapter training completed!")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Parameter efficiency: {param_efficiency:.1f}%")
    print(f"  Model saved: {output_dir / 'stage2_adapter_model_best.pt'}")
    print(f"{'='*80}\n")

    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Train Solution 1: Conv-Adapter")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default=None,
        help="Path to Stage 1 checkpoint (if None, trains Stage 1 first)"
    )
    parser.add_argument(
        "--adapter-reduction",
        type=int,
        default=4,
        choices=[2, 4, 8, 16],
        help="Adapter reduction ratio (default: 4, ablation showed γ=2 gives no improvement)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum epochs"
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-4,
        help="Learning rate for backbone (Stage 1)"
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="Learning rate for heads"
    )
    parser.add_argument(
        "--lr-adapter",
        type=float,
        default=1e-3,
        help="Learning rate for adapter"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train Stage 1 if checkpoint not provided
    if args.stage1_checkpoint is None:
        print("No Stage 1 checkpoint provided. Training Stage 1 first...")
        stage1_dir = output_dir / "stage1"
        stage1_dir.mkdir(exist_ok=True)

        train_stage1_adapter_solution(
            dataset_dir=dataset_dir,
            output_dir=stage1_dir,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            patience=args.patience,
            seed=args.seed
        )

        stage1_checkpoint = stage1_dir / "stage1_model_best.pt"
    else:
        stage1_checkpoint = Path(args.stage1_checkpoint)

    # Train Stage 2 with adapter
    stage2_dir = output_dir / "stage2_adapter"
    stage2_dir.mkdir(exist_ok=True)

    train_stage2_with_adapter(
        dataset_dir=dataset_dir,
        stage1_checkpoint=stage1_checkpoint,
        output_dir=stage2_dir,
        adapter_reduction=args.adapter_reduction,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_adapter=args.lr_adapter,
        lr_head=args.lr_head,
        patience=args.patience,
        seed=args.seed
    )

    print(f"\n{'='*80}")
    print(f"  SOLUTION 1 (Conv-Adapter) training completed!")
    print(f"  Results saved in: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
