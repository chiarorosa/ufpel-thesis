"""Thesis standalone backbone architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class ImprovedBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, return_intermediate: bool = False):
        super().__init__()
        self.return_intermediate = return_intermediate
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight[:] = resnet.conv1.weight.mean(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        self.spatial_attn = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        intermediates = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)
        if self.return_intermediate:
            intermediates["layer1"] = x

        x = self.layer2(x)
        x = self.se2(x)
        if self.return_intermediate:
            intermediates["layer2"] = x

        x = self.layer3(x)
        x = self.se3(x)
        if self.return_intermediate:
            intermediates["layer3"] = x

        x = self.layer4(x)
        x = self.se4(x)
        x = self.spatial_attn(x)
        if self.return_intermediate:
            intermediates["layer4"] = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.return_intermediate:
            intermediates["features"] = x
            return x, intermediates
        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]

        layers: list[nn.Module] = []
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def create_stage1_head(**kwargs):
    return ClassificationHead(num_classes=2, hidden_dims=[256], dropout=0.3, **kwargs)


def create_stage2_head(**kwargs):
    return ClassificationHead(num_classes=3, hidden_dims=[256, 128], dropout=0.4, **kwargs)


def create_stage3_rect_head(**kwargs):
    return ClassificationHead(num_classes=2, hidden_dims=[128, 64], dropout=0.2, **kwargs)


def create_stage3_ab_head(**kwargs):
    return ClassificationHead(num_classes=4, hidden_dims=[256, 128], dropout=0.5, **kwargs)
