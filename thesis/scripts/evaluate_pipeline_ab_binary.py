"""
Script 018: Evaluate Full Hierarchical Pipeline (v7) with AB Binary

End-to-end evaluation: Stage 1 → Stage 2 → Stage 3 (RECT/AB-Binary)

Changes from 016:
- Stage 3-AB uses BINARY model (HORZ vs VERT) instead of 4-way
- Mapping: HORZ → HORZ_A, VERT → VERT_A (heuristic)

Expected improvement:
- AB F1: 5.53% → 28-30%
- Pipeline accuracy: 57.66% → 58.5-59.5%
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Add thesis pipeline to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis.pipeline.data_hub import BlockRecord, build_hierarchical_dataset_v6
from thesis.pipeline.conv_adapter import AdapterBackbone
from thesis.pipeline.backbone import ImprovedBackbone, ClassificationHead

# AV1 partition mappings
PARTITION_NAMES = ['NONE', 'HORZ', 'VERT', 'SPLIT', 'HORZ_A', 'HORZ_B', 'VERT_A', 'VERT_B', 'HORZ_4', 'VERT_4']

STAGE2_GROUPS = {
    'SPLIT': [3],
    'RECT': [1, 2],
    'AB': [4, 5, 6, 7]
}

# Reverse mapping: partition_id -> stage2_class
PARTITION_TO_STAGE2 = {}
for stage2_class, (group_name, partitions) in enumerate(STAGE2_GROUPS.items()):
    for pid in partitions:
        PARTITION_TO_STAGE2[pid] = stage2_class


class PipelineV7(nn.Module):
    """
    Full hierarchical pipeline v7: Stage 1 → Stage 2 → Stage 3
    
    Uses Conv-Adapter frozen backbones for all stages.
    """
    def __init__(self, stage1_model, stage2_model, stage3_rect_model=None, stage3_ab_model=None):
        super().__init__()
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.stage3_rect = stage3_rect_model
        self.stage3_ab = stage3_ab_model
    
    def forward(self, x, return_intermediates=False):
        """
        Args:
            x: [B, 1, 16, 16] input images
            return_intermediates: If True, return stage1/stage2 predictions
        
        Returns:
            final_pred: [B] final partition predictions (0-9)
            intermediates: Dict with stage1_pred, stage2_pred, stage3_pred
        """
        batch_size = x.size(0)
        
        # Stage 1: NONE vs PARTITION
        stage1_logits = self.stage1(x)
        stage1_pred = torch.argmax(stage1_logits, dim=1)  # 0=NONE, 1=PARTITION
        
        # Initialize final predictions as PARTITION_NONE (0)
        final_pred = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Initialize intermediate predictions
        stage2_pred_full = torch.full((batch_size,), -1, dtype=torch.long, device=x.device)
        
        # Only process samples classified as PARTITION (stage1_pred == 1)
        partition_mask = stage1_pred == 1
        
        if partition_mask.sum() > 0:
            # Stage 2: SPLIT, RECT, AB
            x_partition = x[partition_mask]
            stage2_logits = self.stage2(x_partition)
            stage2_pred = torch.argmax(stage2_logits, dim=1)  # 0=SPLIT, 1=RECT, 2=AB
            
            # Store Stage 2 predictions in full array
            stage2_pred_full[partition_mask] = stage2_pred
            
            # Process SPLIT samples (directly map to partition 3)
            split_mask = stage2_pred == 0
            if split_mask.sum() > 0:
                partition_indices = torch.where(partition_mask)[0][split_mask]
                final_pred[partition_indices] = 3  # PARTITION_SPLIT
            
            # Process RECT samples (Stage 3-RECT)
            rect_mask = stage2_pred == 1
            if rect_mask.sum() > 0 and self.stage3_rect is not None:
                x_rect = x_partition[rect_mask]
                stage3_rect_logits = self.stage3_rect(x_rect)
                stage3_rect_pred = torch.argmax(stage3_rect_logits, dim=1)  # 0=HORZ, 1=VERT
                
                partition_indices = torch.where(partition_mask)[0][rect_mask]
                # Map: 0→1 (HORZ), 1→2 (VERT)
                final_pred[partition_indices] = stage3_rect_pred + 1
            
            # Process AB samples (Stage 3-AB BINARY)
            ab_mask = stage2_pred == 2
            if ab_mask.sum() > 0 and self.stage3_ab is not None:
                x_ab = x_partition[ab_mask]
                stage3_ab_logits = self.stage3_ab(x_ab)
                stage3_ab_binary_pred = torch.argmax(stage3_ab_logits, dim=1)  # 0=HORZ, 1=VERT
                
                partition_indices = torch.where(partition_mask)[0][ab_mask]
                
                # Mapping heuristic: HORZ → HORZ_A (4), VERT → VERT_A (6)
                # This is a simplified mapping that loses A/B granularity
                # but significantly improves standalone F1 (21% → 57%)
                ab_final_pred = torch.where(
                    stage3_ab_binary_pred == 0,
                    torch.tensor(4, device=x.device),  # HORZ → PARTITION_HORZ_A
                    torch.tensor(6, device=x.device)   # VERT → PARTITION_VERT_A
                )
                final_pred[partition_indices] = ab_final_pred
        
        if return_intermediates:
            intermediates = {
                'stage1_pred': stage1_pred,
                'stage2_pred': stage2_pred_full,
                'stage3_pred': torch.full((batch_size,), -1, dtype=torch.long, device=x.device)
            }
            return final_pred, intermediates
        
        return final_pred


def load_stage1_model(checkpoint_path, device):
    """Load Stage 1 model (binary: NONE vs PARTITION)"""
    print(f"\n[Loading Stage 1] {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Rebuild model (MUST match training architecture)
    backbone = ImprovedBackbone(pretrained=False)
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
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device).eval()
    print(f"  ✅ Stage 1 loaded")
    return model


def load_stage2_adapter_model(checkpoint_path, device):
    """Load Stage 2 Conv-Adapter model (3-way: SPLIT, RECT, AB)"""
    print(f"\n[Loading Stage 2] {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    adapter_reduction = checkpoint.get('adapter_reduction', 4)
    
    # Rebuild adapter backbone
    base_backbone = ImprovedBackbone(pretrained=False)
    adapter_config = {
        'reduction': adapter_reduction,
        'layers': ['layer3', 'layer4'],
        'variant': 'conv_parallel'
    }
    adapter_backbone = AdapterBackbone(base_backbone, adapter_config=adapter_config)
    
    # Load adapter_backbone weights
    if 'adapter_backbone_state_dict' in checkpoint:
        adapter_backbone.load_state_dict(checkpoint['adapter_backbone_state_dict'], strict=True)
    
    # Build full Stage 2 model (MUST match training architecture)
    stage2_head = ClassificationHead(num_classes=3, hidden_dims=[256, 128], dropout=0.4)
    
    # Load head weights
    if 'stage2_head_state_dict' in checkpoint:
        stage2_head.load_state_dict(checkpoint['stage2_head_state_dict'], strict=True)
    
    class Stage2AdapterModel(nn.Module):
        def __init__(self, adapter_backbone, head):
            super().__init__()
            self.adapter_backbone = adapter_backbone
            self.stage2_head = head
        
        def forward(self, x):
            # AdapterBackbone returns (features, intermediates)
            features, _ = self.adapter_backbone(x)
            return self.stage2_head(features)
    
    model = Stage2AdapterModel(adapter_backbone, stage2_head)
    model.to(device).eval()
    print(f"  ✅ Stage 2 loaded (adapter_reduction={adapter_reduction})")
    return model


def load_stage3_model(checkpoint_path, stage2_checkpoint_path, num_classes, device):
    """Load Stage 3 model (RECT or AB)"""
    print(f"\n[Loading Stage 3 ({num_classes}-way)] {checkpoint_path}")
    
    # Load Stage 2 checkpoint to get adapter backbone
    stage2_ckpt = torch.load(stage2_checkpoint_path, map_location=device, weights_only=False)
    adapter_reduction = stage2_ckpt.get('adapter_reduction', 4)
    
    # Rebuild adapter backbone
    base_backbone = ImprovedBackbone(pretrained=False)
    adapter_config = {
        'reduction': adapter_reduction,
        'layers': ['layer3', 'layer4'],
        'variant': 'conv_parallel'
    }
    adapter_backbone = AdapterBackbone(base_backbone, adapter_config=adapter_config)
    
    # Load adapter_backbone from Stage 2
    if 'adapter_backbone_state_dict' in stage2_ckpt:
        adapter_backbone.load_state_dict(stage2_ckpt['adapter_backbone_state_dict'], strict=True)
    
    # Build Stage 3 model
    class Stage3Model(nn.Module):
        def __init__(self, adapter_backbone, num_classes):
            super().__init__()
            self.adapter_backbone = adapter_backbone
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            features_output = self.adapter_backbone(x)
            if isinstance(features_output, tuple):
                features = features_output[0]
            else:
                features = features_output
            return self.head(features)
    
    model = Stage3Model(adapter_backbone, num_classes)
    
    # Load Stage 3 weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device).eval()
    print(f"  ✅ Stage 3 loaded")
    return model


def load_stage3_ab_binary_model(checkpoint_path, stage2_checkpoint_path, device):
    """
    Load Stage 3-AB BINARY model (HORZ vs VERT)
    
    Returns model that predicts:
        0 = HORZ (A+B grouped)
        1 = VERT (A+B grouped)
    """
    print(f"\n[Loading Stage 3-AB Binary] {checkpoint_path}")
    
    # Load Stage 2 checkpoint to get adapter backbone
    stage2_ckpt = torch.load(stage2_checkpoint_path, map_location=device, weights_only=False)
    adapter_reduction = stage2_ckpt.get('adapter_reduction', 4)
    
    # Rebuild adapter backbone
    base_backbone = ImprovedBackbone(pretrained=False)
    adapter_config = {
        'reduction': adapter_reduction,
        'layers': ['layer3', 'layer4'],
        'variant': 'conv_parallel'
    }
    adapter_backbone = AdapterBackbone(base_backbone, adapter_config=adapter_config)
    
    # Load adapter_backbone from Stage 2
    if 'adapter_backbone_state_dict' in stage2_ckpt:
        adapter_backbone.load_state_dict(stage2_ckpt['adapter_backbone_state_dict'], strict=True)
    
    # Build Stage 3-AB Binary model
    class Stage3ABBinaryModel(nn.Module):
        def __init__(self, adapter_backbone):
            super().__init__()
            self.adapter_backbone = adapter_backbone
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
            features_output = self.adapter_backbone(x)
            if isinstance(features_output, tuple):
                features = features_output[0]
            else:
                features = features_output
            return self.ab_head(features)
    
    model = Stage3ABBinaryModel(adapter_backbone)
    
    # Load Stage 3-AB binary weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device).eval()
    print(f"  ✅ Stage 3-AB Binary loaded (HORZ vs VERT)")
    return model


def evaluate_pipeline(pipeline, dataloader, device):
    """Evaluate the pipeline and compute metrics."""
    pipeline.eval()
    
    all_preds = []
    all_labels = []
    
    # Diagnostic counters
    stage1_none_count = 0
    stage1_partition_count = 0
    stage2_split_count = 0
    stage2_rect_count = 0
    stage2_ab_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Pipeline"):
            images = batch['image'].to(device)
            labels = batch['label_stage0'].to(device)
            
            preds, intermediates = pipeline(images, return_intermediates=True)
            
            # Track stage distributions
            stage1_none_count += (intermediates['stage1_pred'] == 0).sum().item()
            stage1_partition_count += (intermediates['stage1_pred'] == 1).sum().item()
            
            # Only count stage2 for partition samples
            stage2_pred = intermediates['stage2_pred']
            partition_mask = intermediates['stage1_pred'] == 1
            if partition_mask.sum() > 0:
                stage2_split_count += (stage2_pred[partition_mask] == 0).sum().item()
                stage2_rect_count += (stage2_pred[partition_mask] == 1).sum().item()
                stage2_ab_count += (stage2_pred[partition_mask] == 2).sum().item()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Print diagnostic info
    total_samples = len(all_labels)
    print("\n" + "="*80)
    print("  Pipeline Stage Diagnostics")
    print("="*80)
    print(f"  Stage 1 Predictions:")
    print(f"    NONE:      {stage1_none_count:6d} ({100*stage1_none_count/total_samples:.2f}%)")
    print(f"    PARTITION: {stage1_partition_count:6d} ({100*stage1_partition_count/total_samples:.2f}%)")
    print(f"\n  Stage 2 Predictions (within PARTITION samples):")
    if stage1_partition_count > 0:
        print(f"    SPLIT:     {stage2_split_count:6d} ({100*stage2_split_count/stage1_partition_count:.2f}%)")
        print(f"    RECT:      {stage2_rect_count:6d} ({100*stage2_rect_count/stage1_partition_count:.2f}%)")
        print(f"    AB:        {stage2_ab_count:6d} ({100*stage2_ab_count/stage1_partition_count:.2f}%)")
    else:
        print(f"    (No PARTITION samples predicted by Stage 1)")
    print("="*80 + "\n")
    
    # Compute metrics
    accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
    f1_macro = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels.numpy(), all_preds.numpy(), average=None, zero_division=0)
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), labels=list(range(10)))
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate v7 Pipeline (Stage 1→2→3)")
    
    # Checkpoints
    parser.add_argument('--stage1-checkpoint', type=str, required=True)
    parser.add_argument('--stage2-checkpoint', type=str, required=True)
    parser.add_argument('--stage3-rect-checkpoint', type=str, default=None)
    parser.add_argument('--stage3-ab-checkpoint', type=str, default=None)
    
    # Dataset
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to v7_dataset/block_16')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    
    # Output
    parser.add_argument('--output', type=str, required=True)
    
    # Misc
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  v7 Pipeline Evaluation (Stage 1→2→3)")
    print("=" * 80)
    print(f"  Stage 1: {args.stage1_checkpoint}")
    print(f"  Stage 2: {args.stage2_checkpoint}")
    print(f"  Stage 3-RECT: {args.stage3_rect_checkpoint or 'Not provided'}")
    print(f"  Stage 3-AB: {args.stage3_ab_checkpoint or 'Not provided'}")
    print(f"  Dataset: {args.dataset_dir} ({args.split})")
    print(f"  Output: {output_dir}")
    print("=" * 80)
    
    # Load models
    stage1_model = load_stage1_model(args.stage1_checkpoint, device)
    stage2_model = load_stage2_adapter_model(args.stage2_checkpoint, device)
    
    stage3_rect_model = None
    if args.stage3_rect_checkpoint:
        stage3_rect_model = load_stage3_model(
            args.stage3_rect_checkpoint, 
            args.stage2_checkpoint,
            num_classes=2,  # HORZ, VERT
            device=device
        )
    
    stage3_ab_model = None
    if args.stage3_ab_checkpoint:
        # Use BINARY AB model (HORZ vs VERT)
        stage3_ab_model = load_stage3_ab_binary_model(
            args.stage3_ab_checkpoint,
            args.stage2_checkpoint,
            device=device
        )
    
    # Build pipeline
    pipeline = PipelineV7(stage1_model, stage2_model, stage3_rect_model, stage3_ab_model)
    pipeline.to(device)
    
    # Load dataset
    data_path = Path(args.dataset_dir) / f"{args.split}.pt"
    print(f"\n[Loading dataset] {data_path}")
    data = torch.load(data_path, weights_only=False)
    
    # Create simple dataset directly from tensors
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, labels):
            self.samples = samples  # [N, 1, 16, 16]
            self.labels = labels    # [N]
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return {
                'image': self.samples[idx],
                'label_stage0': self.labels[idx]
            }
    
    # Extract data (handle both v7 and legacy formats)
    if isinstance(data, dict):
        samples = data['samples']
        labels = data.get('labels_stage0', data.get('labels'))
        
        # Convert numpy to torch if needed
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).float().transpose(1, 3).transpose(2, 3) / 1023.0
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    dataset = SimpleDataset(samples, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    print(f"  ✅ {len(dataset)} samples loaded")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("  Running Evaluation")
    print("=" * 80)
    
    results = evaluate_pipeline(pipeline, dataloader, device)
    
    # Print results
    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  F1 (macro): {results['f1_macro']*100:.2f}%")
    print(f"  F1 (weighted): {results['f1_weighted']*100:.2f}%")
    
    print(f"\n  Per-class F1:")
    for i, (name, f1) in enumerate(zip(PARTITION_NAMES, results['f1_per_class'])):
        print(f"    {name:12s}: {f1*100:5.2f}%")
    
    # Save results
    output_file = output_dir / f"pipeline_{args.split}_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to python types for JSON
        results_json = {
            'accuracy': float(results['accuracy']),
            'f1_macro': float(results['f1_macro']),
            'f1_weighted': float(results['f1_weighted']),
            'f1_per_class': [float(x) for x in results['f1_per_class']],
            'confusion_matrix': results['confusion_matrix'],
            'config': {
                'stage1_checkpoint': args.stage1_checkpoint,
                'stage2_checkpoint': args.stage2_checkpoint,
                'stage3_rect_checkpoint': args.stage3_rect_checkpoint,
                'stage3_ab_checkpoint': args.stage3_ab_checkpoint,
                'dataset': str(data_path),
                'split': args.split,
                'timestamp': datetime.now().isoformat()
            }
        }
        json.dump(results_json, f, indent=2)
    
    print(f"\n  ✅ Results saved to {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
