#!/usr/bin/env python3
"""
Script 017: Train Stage 3-AB BINARY (HORZ vs VERT)

Simplifies 4-way AB classification to binary:
- HORZ = HORZ_A (4) + HORZ_B (5)
- VERT = VERT_A (6) + VERT_B (7)

Hypothesis: Binary classification will achieve F1 > 60% (vs 21% in 4-way)

Architecture: Reuses Stage3RectModelV7 (proven to work, F1=69.61%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm

from thesis.pipeline.backbone import ImprovedBackbone
from thesis.pipeline.conv_adapter import AdapterBackbone
from thesis.pipeline.losses import FocalLoss
from thesis.pipeline.evaluation import MetricsCalculator


class ABBinaryDataset(Dataset):
    """
    Dataset for AB binary classification
    
    Labels mapping:
    - 0: HORZ (HORZ_A + HORZ_B)
    - 1: VERT (VERT_A + VERT_B)
    """
    def __init__(self, samples, labels_stage0):
        # Filter only AB samples (labels 4,5,6,7)
        ab_mask = (labels_stage0 >= 4) & (labels_stage0 <= 7)
        self.samples = samples[ab_mask]
        
        # Map to binary: 4,5 → 0 (HORZ), 6,7 → 1 (VERT)
        ab_labels = labels_stage0[ab_mask]
        self.labels = ((ab_labels == 6) | (ab_labels == 7)).long()
        
        print(f"  AB Binary Dataset created:")
        print(f"    Total samples: {len(self.samples)}")
        print(f"    HORZ: {(self.labels == 0).sum()} ({100*(self.labels == 0).sum()/len(self.labels):.2f}%)")
        print(f"    VERT: {(self.labels == 1).sum()} ({100*(self.labels == 1).sum()/len(self.labels):.2f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'image': self.samples[idx],
            'label': self.labels[idx]
        }


class Stage3ABBinaryModel(nn.Module):
    """
    Stage 3-AB Binary Model (HORZ vs VERT)
    
    Architecture: Same as Stage3RectModelV7 (proven design)
    - Frozen adapter backbone from Stage 2
    - Binary classification head
    """
    def __init__(self, adapter_backbone, freeze_backbone=True):
        super().__init__()
        
        self.adapter_backbone = adapter_backbone
        
        # Freeze backbone (same as RECT)
        if freeze_backbone:
            for param in self.adapter_backbone.parameters():
                param.requires_grad = False
        
        # Binary head (same structure as RECT for consistency)
        self.ab_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary: HORZ vs VERT
        )
    
    def forward(self, x):
        # AdapterBackbone returns (features, intermediates)
        features, _ = self.adapter_backbone(x)
        return self.ab_head(features)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = MetricsCalculator.calculate_classification_metrics(
        np.array(all_labels), np.array(all_preds), num_classes=2
    )
    
    return total_loss / len(dataloader), metrics


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = MetricsCalculator.calculate_classification_metrics(
        np.array(all_labels), np.array(all_preds), num_classes=2
    )
    
    return total_loss / len(dataloader), metrics


def main():
    parser = argparse.ArgumentParser(description="Train Stage 3-AB Binary (HORZ vs VERT)")
    
    # Paths
    parser.add_argument('--dataset-dir', type=str, 
                        default='thesis/v7_dataset/block_16',
                        help='Path to dataset directory')
    parser.add_argument('--stage2-checkpoint', type=str,
                        default='thesis/logs/v7_experiments/solution1_adapter_bn_fix/stage2_adapter/stage2_adapter_model_best.pt',
                        help='Path to Stage 2 checkpoint (for adapter backbone)')
    parser.add_argument('--output-dir', type=str,
                        default='thesis/logs/v7_experiments/stage3_ab_binary',
                        help='Output directory')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    
    # Loss function
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'ce'])
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    print("="*80)
    print("  Stage 3-AB Binary Training (HORZ vs VERT)")
    print("="*80)
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Stage 2 checkpoint: {args.stage2_checkpoint}")
    print(f"  Output: {args.output_dir}")
    print(f"  Loss: {args.loss}")
    print(f"  Device: {device}")
    print("="*80)
    
    # Load datasets
    print("\n[1/5] Loading datasets...")
    train_data = torch.load(Path(args.dataset_dir) / 'train.pt', weights_only=False)
    val_data = torch.load(Path(args.dataset_dir) / 'val.pt', weights_only=False)
    
    print("  Train:")
    train_dataset = ABBinaryDataset(train_data['samples'], train_data['labels_stage0'])
    print("  Val:")
    val_dataset = ABBinaryDataset(val_data['samples'], val_data['labels_stage0'])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load Stage 2 adapter backbone
    print("\n[2/5] Loading Stage 2 adapter backbone...")
    stage2_ckpt = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
    
    base_backbone = ImprovedBackbone(pretrained=False)
    adapter_config = {
        'reduction': stage2_ckpt.get('adapter_reduction', 4),
        'layers': ['layer3', 'layer4'],
        'variant': 'conv_parallel'
    }
    adapter_backbone = AdapterBackbone(base_backbone, adapter_config=adapter_config)
    adapter_backbone.load_state_dict(stage2_ckpt['adapter_backbone_state_dict'], strict=True)
    print(f"  ✅ Adapter backbone loaded (reduction={adapter_config['reduction']})")
    
    # Create model
    print("\n[3/5] Creating Stage 3-AB Binary model...")
    model = Stage3ABBinaryModel(adapter_backbone, freeze_backbone=True)
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Loss function
    if args.loss == 'focal':
        criterion = FocalLoss(gamma=2.5, alpha=0.25)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n[4/5] Training...")
    best_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['f1_macro'])
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1_macro'])
        
        print(f"  Train Loss: {train_loss:.4f}, F1: {train_metrics['f1_macro']*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, F1: {val_metrics['f1_macro']*100:.2f}%")
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'adapter_config': adapter_config
            }
            torch.save(checkpoint, output_dir / 'model_best.pt')
            print(f"  ✅ New best F1: {best_f1*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    # Save final metrics
    print("\n[5/5] Saving results...")
    
    final_metrics = {
        'best_f1': best_f1,
        'final_train_f1': history['train_f1'][-1],
        'final_val_f1': history['val_f1'][-1],
        'epochs_trained': epoch,
        'early_stopped': patience_counter >= args.patience,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_efficiency': 100 * trainable_params / total_params
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    torch.save(history, output_dir / 'history.pt')
    
    print(f"\n{'='*80}")
    print(f"  Training Complete!")
    print(f"{'='*80}")
    print(f"  Best F1: {best_f1*100:.2f}%")
    print(f"  Model saved: {output_dir / 'model_best.pt'}")
    print(f"  Metrics saved: {output_dir / 'metrics.json'}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
