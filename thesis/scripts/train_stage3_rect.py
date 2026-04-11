"""
Script 014: Train Stage 3-RECT with Conv-Adapter (v7)
Stage 3-RECT: Binary classification (HORZ vs VERT)

v7 Improvements:
1. Conv-Adapter parameter-efficient transfer learning (Chen et al., CVPR 2024)
2. BatchNorm fix for frozen backbone (critical v7 discovery)
3. Focal Loss with tuned gamma (Exp04 findings)
4. No negative transfer (backbone stays frozen)

Architecture:
- Frozen Stage 2 Conv-Adapter backbone (97.13% params)
- New RECT classification head (2.87% params)
- Total: ~11.5M params, only ~331k trainable

References:
- Chen et al., CVPR 2024: Conv-Adapter: Exploring Parameter Efficient Transfer Learning
- Lin et al., ICCV 2017: Focal Loss for Dense Object Detection
- Howard & Ruder, ACL 2018: Universal Language Model Fine-tuning
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

# Add thesis pipeline to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.pipeline.data_hub import BlockRecord, build_hierarchical_dataset_v6
from thesis.pipeline.conv_adapter import AdapterBackbone
from thesis.pipeline.losses import FocalLoss, ClassBalancedFocalLoss
from thesis.pipeline.backbone import ImprovedBackbone
from thesis.pipeline.evaluation import MetricsCalculator


class Stage3RectModelV7(nn.Module):
    """
    Stage 3-RECT model with frozen Conv-Adapter backbone.
    
    Binary classification: PARTITION_HORZ (0) vs PARTITION_VERT (1)
    """
    def __init__(self, adapter_backbone, freeze_backbone=True):
        super().__init__()
        self.adapter_backbone = adapter_backbone
        self.freeze_backbone = freeze_backbone
        
        # Freeze backbone (including adapters from Stage 2)
        if freeze_backbone:
            for param in self.adapter_backbone.parameters():
                param.requires_grad = False
        
        # Binary classification head
        in_features = 512  # ResNet-18 final feature dimension
        self.rect_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # HORZ vs VERT
        )
    
    def forward(self, x):
        # Extract features with frozen backbone + adapters
        features_output = self.adapter_backbone(x)
        # AdapterBackbone returns (features, intermediates) - we only need features
        if isinstance(features_output, tuple):
            features = features_output[0]
        else:
            features = features_output
        # Classify RECT
        logits = self.rect_head(features)
        return logits
    
    def get_trainable_params(self):
        """Get parameters that require gradients"""
        return [p for p in self.parameters() if p.requires_grad]


def load_stage2_adapter_backbone(checkpoint_path, device):
    """
    Load Stage 2 Conv-Adapter model and extract backbone.
    
    Returns adapter_backbone with frozen weights.
    """
    print(f"\n[Loading] Stage 2 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract hyperparameters
    adapter_reduction = checkpoint.get('adapter_reduction', 4)
    
    # Rebuild adapter backbone
    base_backbone = ImprovedBackbone(pretrained=False)  # Will load weights from checkpoint
    adapter_config = {
        'reduction': adapter_reduction,
        'layers': ['layer3', 'layer4'],
        'variant': 'conv_parallel'
    }
    adapter_backbone = AdapterBackbone(base_backbone, adapter_config=adapter_config)
    
    # Load adapter_backbone weights (checkpoint has dedicated key)
    if 'adapter_backbone_state_dict' in checkpoint:
        adapter_backbone.load_state_dict(checkpoint['adapter_backbone_state_dict'], strict=True)
    else:
        # Fallback: extract from full model_state_dict
        state_dict = checkpoint['model_state_dict']
        adapter_state = {k.replace('adapter_backbone.', ''): v 
                         for k, v in state_dict.items() 
                         if k.startswith('adapter_backbone.')}
        adapter_backbone.load_state_dict(adapter_state, strict=False)
    
    adapter_backbone.to(device)
    
    print(f"  ✅ Loaded adapter_reduction={adapter_reduction}")
    print(f"  ✅ Adapter backbone ready (frozen)")
    
    return adapter_backbone


def load_datasets(dataset_dir, augmentation=None):
    """Load RECT train/val datasets"""
    train_path = Path(dataset_dir) / "train.pt"
    val_path = Path(dataset_dir) / "val.pt"
    
    print(f"\n[Loading datasets]")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    
    # Load raw data
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    
    # Data is already in numpy format [N, H, W, C]
    train_record = BlockRecord(
        samples=train_data['samples'],  # Already [N,16,16,1] numpy
        labels=train_data['labels'],     # Already numpy
        qps=train_data['qps']            # Already numpy
    )
    val_record = BlockRecord(
        samples=val_data['samples'],
        labels=val_data['labels'],
        qps=val_data['qps']
    )
    
    # Build datasets
    train_dataset = build_hierarchical_dataset_v6(train_record, augmentation=augmentation, stage='stage3_rect')
    val_dataset = build_hierarchical_dataset_v6(val_record, augmentation=None, stage='stage3_rect')
    
    print(f"  ✅ Train samples: {len(train_dataset)}")
    print(f"  ✅ Val samples: {len(val_dataset)}")
    
    # Check distribution (data is already numpy)
    train_labels = train_data['labels']
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"\n  Train distribution:")
    for label, count in zip(unique, counts):
        label_name = "PARTITION_HORZ" if label == 0 else "PARTITION_VERT"
        print(f"    {label_name}: {count:5d} ({count/len(train_labels)*100:5.2f}%)")
    
    return train_dataset, val_dataset


def train_epoch(model, dataloader, criterion, optimizer, device, fix_batchnorm=True):
    """Train for one epoch with BatchNorm fix"""
    model.train()
    
    # CRITICAL: Fix BatchNorm in frozen backbone (v7 discovery)
    if fix_batchnorm and model.freeze_backbone:
        model.adapter_backbone.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label_stage3_RECT'].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
        optimizer.step()
        
        # Collect predictions
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Calculate metrics
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics_dict = MetricsCalculator.calculate_classification_metrics(all_labels, all_preds, num_classes=2)
    metrics_dict['loss'] = total_loss / len(dataloader)
    
    # Rename keys to match expected format
    return {
        'loss': metrics_dict['loss'],
        'macro_f1': metrics_dict['f1_macro'],
        'macro_precision': metrics_dict['precision_macro'],
        'macro_recall': metrics_dict['recall_macro'],
        'per_class_f1': metrics_dict['f1_per_class']
    }


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            labels = batch['label_stage3_RECT'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
    
    # Calculate metrics
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics_dict = MetricsCalculator.calculate_classification_metrics(all_labels, all_preds, num_classes=2)
    metrics_dict['loss'] = total_loss / len(dataloader)
    
    # Rename keys to match expected format
    return {
        'loss': metrics_dict['loss'],
        'macro_f1': metrics_dict['f1_macro'],
        'macro_precision': metrics_dict['precision_macro'],
        'macro_recall': metrics_dict['recall_macro'],
        'per_class_f1': metrics_dict['f1_per_class']
    }


def main():
    parser = argparse.ArgumentParser(description="Train Stage 3-RECT with Conv-Adapter (v7)")
    
    # Paths
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to Stage 3-RECT dataset (e.g., v7_dataset_stage3/RECT/block_16)')
    parser.add_argument('--stage2-checkpoint', type=str, required=True,
                        help='Path to Stage 2 adapter checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for logs and checkpoints')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr-head', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    
    # Loss function
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'ce'])
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--alpha', type=float, default=0.25, help='Focal loss alpha')
    
    # v7 specific
    parser.add_argument('--fix-batchnorm', action='store_true',
                        help='Fix BatchNorm in frozen backbone (CRITICAL v7 fix)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 80)
    print("  Stage 3-RECT Training (v7 Conv-Adapter)")
    print("=" * 80)
    print(f"  Output: {output_dir}")
    print(f"  Stage 2 checkpoint: {args.stage2_checkpoint}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR (head): {args.lr_head}")
    print(f"  Loss: {args.loss} (gamma={args.gamma}, alpha={args.alpha})")
    print(f"  BatchNorm fix: {args.fix_batchnorm}")
    print(f"  Augmentation: {not args.no_augmentation}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    print("=" * 80)
    
    device = torch.device(args.device)
    
    # Load Stage 2 adapter backbone
    adapter_backbone = load_stage2_adapter_backbone(args.stage2_checkpoint, device)
    
    # Build model
    model = Stage3RectModelV7(adapter_backbone, freeze_backbone=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model]")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Frozen params: {total_params-trainable_params:,} ({(1-trainable_params/total_params)*100:.2f}%)")
    
    # Load datasets (without augmentation for now to simplify)
    train_dataset, val_dataset = load_datasets(args.dataset_dir, augmentation=None)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Loss function
    if args.loss == 'focal':
        criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha)
        print(f"\n[Loss] Focal Loss (γ={args.gamma}, α={args.alpha})")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"\n[Loss] Cross-Entropy")
    
    # Optimizer (only head is trainable)
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    max_patience = 10
    
    history = {
        'train_loss': [], 'train_f1': [], 'train_precision': [], 'train_recall': [],
        'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
        'per_class_f1': []
    }
    
    print("\n" + "=" * 80)
    print("  Starting Training")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, 
                                     fix_batchnorm=args.fix_batchnorm)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['macro_f1'])
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['macro_f1'])
        history['train_precision'].append(train_metrics['macro_precision'])
        history['train_recall'].append(train_metrics['macro_recall'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['macro_f1'])
        history['val_precision'].append(val_metrics['macro_precision'])
        history['val_recall'].append(val_metrics['macro_recall'])
        history['per_class_f1'].append(val_metrics['per_class_f1'])
        
        print(f"\n  Train: Loss={train_metrics['loss']:.4f}, F1={train_metrics['macro_f1']:.4f}, "
              f"P={train_metrics['macro_precision']:.4f}, R={train_metrics['macro_recall']:.4f}")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, F1={val_metrics['macro_f1']:.4f}, "
              f"P={val_metrics['macro_precision']:.4f}, R={val_metrics['macro_recall']:.4f}")
        
        print(f"\n  Per-class F1:")
        for class_name, f1 in zip(['HORZ', 'VERT'], val_metrics['per_class_f1']):
            print(f"    {class_name}: {f1:.4f}")
        
        # Save best model
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, output_dir / 'model_best.pt')
            print(f"  ✅ New best F1: {best_f1:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{max_patience})")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    # Save final metrics
    final_metrics = {
        'best_f1': float(best_f1),
        'best_epoch': best_epoch,
        'final_train_f1': float(history['train_f1'][-1]),
        'final_val_f1': float(history['val_f1'][-1]),
        'epochs_trained': epoch,
        'early_stopped': patience_counter >= max_patience,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_efficiency': float(trainable_params / total_params * 100)
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    with open(output_dir / 'history.json', 'w') as f:
        # Convert numpy to list for JSON serialization
        history_json = {k: [float(x) if isinstance(x, (np.floating, float)) else x 
                            for x in v] for k, v in history.items()}
        json.dump(history_json, f, indent=2)
    
    print("\n" + "=" * 80)
    print("  Training Complete!")
    print("=" * 80)
    print(f"  Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"  Metrics saved to: {output_dir / 'metrics.json'}")
    print(f"  Best model saved to: {output_dir / 'model_best.pt'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
