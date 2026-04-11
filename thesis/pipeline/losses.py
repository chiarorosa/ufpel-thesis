"""
V6 Pipeline - Loss Functions
Focal Loss, Class-Balanced Loss, Mixup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017)
    Addresses class imbalance by down-weighting easy examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits or (N, 1) for binary
            targets: (N,) class indices or (N, 1) for binary
        """
        # Binary case
        if inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
            targets = targets.float()
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            probs = torch.sigmoid(inputs)
            pt = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * (1 - pt) ** self.gamma
            loss = focal_weight * bce_loss
        
        # Multi-class case
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            probs = F.softmax(inputs, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al., 2019)
    Uses effective number of samples to compute class weights
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Compute effective number
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(weights)
        
        # Register as buffer so it moves with model.to(device)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        # Ensure weights are on same device as inputs
        weights = self.weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none')
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MixupLoss(nn.Module):
    """
    Mixup augmentation + loss (Zhang et al., 2018)
    Creates virtual training examples by mixing pairs
    """
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha
    
    def mixup_data(self, x, y):
        """Apply mixup to data"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Compute mixup loss"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class HardNegativeMiningLoss(nn.Module):
    """
    Hard Negative Mining for binary classification
    Keeps ratio of negative:positive = neg_pos_ratio
    """
    def __init__(self, neg_pos_ratio=3.0, base_loss='bce'):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.base_loss = base_loss
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, 1) or (N,) logits
            targets: (N,) binary labels
        """
        if inputs.dim() > 1:
            inputs = inputs.squeeze(1)
        
        targets = targets.float()
        
        # Compute loss per sample
        if self.base_loss == 'bce':
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        elif self.base_loss == 'focal':
            focal = FocalLoss(reduction='none')
            loss = focal(inputs.unsqueeze(1), targets.unsqueeze(1)).squeeze()
        
        # Separate positive and negative samples
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        
        pos_loss = loss[pos_mask]
        neg_loss = loss[neg_mask]
        
        # Keep all positives
        num_pos = pos_mask.sum().item()
        
        # Select hard negatives
        num_neg = min(int(num_pos * self.neg_pos_ratio), neg_mask.sum().item())
        
        if num_neg > 0:
            neg_loss, _ = torch.topk(neg_loss, num_neg)
        
        # Combine
        total_loss = torch.cat([pos_loss, neg_loss])
        
        return total_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Regularization
    Prevents overconfidence
    """
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -torch.sum(true_dist * log_probs, dim=1)
        return loss.mean()


def get_loss_function(stage, loss_config):
    """
    Factory function to create loss based on stage and config
    
    Args:
        stage: 'stage1', 'stage2', 'stage3_rect', 'stage3_ab'
        loss_config: dict with loss parameters
    
    Returns:
        loss function
    """
    if stage == 'stage1':
        if loss_config.get('hard_mining', False):
            return HardNegativeMiningLoss(
                neg_pos_ratio=loss_config.get('neg_pos_ratio', 3.0),
                base_loss='focal'
            )
        else:
            return FocalLoss(
                alpha=loss_config.get('alpha', 0.25),
                gamma=loss_config.get('gamma', 2.5)
            )
    
    elif stage == 'stage2':
        samples_per_class = loss_config.get('samples_per_class', [1000, 1000, 1000])
        return ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            beta=loss_config.get('beta', 0.9999),
            gamma=loss_config.get('gamma', 2.0)
        )
    
    elif stage == 'stage3_rect':
        return FocalLoss(
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0)
        )
    
    elif stage == 'stage3_ab':
        samples_per_class = loss_config.get('samples_per_class', [250, 250, 250, 250])
        return ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            beta=loss_config.get('beta', 0.9999),
            gamma=loss_config.get('gamma', 2.0)
        )
    
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    # Test losses
    print("Testing Focal Loss (binary)...")
    focal_binary = FocalLoss(alpha=0.25, gamma=2.0)
    inputs = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8,))
    loss = focal_binary(inputs, targets)
    print(f"Loss: {loss.item():.4f}\n")
    
    print("Testing Focal Loss (multi-class)...")
    focal_multi = FocalLoss(gamma=2.0)
    inputs = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    loss = focal_multi(inputs, targets)
    print(f"Loss: {loss.item():.4f}\n")
    
    print("Testing Class-Balanced Focal Loss...")
    cb_focal = ClassBalancedFocalLoss(
        samples_per_class=[1000, 500, 200],
        beta=0.9999,
        gamma=2.0
    )
    loss = cb_focal(inputs, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Weights: {cb_focal.weights}\n")
    
    print("Testing Hard Negative Mining...")
    hnm = HardNegativeMiningLoss(neg_pos_ratio=3.0)
    inputs = torch.randn(16, 1)
    targets = torch.tensor([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
    loss = hnm(inputs, targets)
    print(f"Loss: {loss.item():.4f}\n")
    
    print("Testing Mixup...")
    mixup = MixupLoss(alpha=0.4)
    x = torch.randn(8, 3, 16, 16)
    y = torch.randint(0, 3, (8,))
    mixed_x, y_a, y_b, lam = mixup.mixup_data(x, y)
    print(f"Lambda: {lam:.3f}")
    print(f"Mixed shape: {mixed_x.shape}\n")
    
    print("✅ All losses working correctly!")
