"""Thesis standalone Conv-Adapter implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        kernel_size: int = 3,
        variant: str = "conv_parallel",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.variant = variant

        hidden_channels = in_channels // reduction
        padding = kernel_size // 2

        self.down_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_channels,
            bias=False,
        )
        self.up_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.ones(in_channels))
        self.bn = nn.BatchNorm2d(hidden_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.down_proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.up_proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.dw_conv.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            self.down_proj.weight *= 0.01
            self.up_proj.weight *= 0.01
            self.dw_conv.weight *= 0.01

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta_h = self.down_proj(h)
        delta_h = self.bn(delta_h)
        delta_h = self.activation(delta_h)
        delta_h = self.dw_conv(delta_h)
        delta_h = self.activation(delta_h)
        delta_h = self.up_proj(delta_h)
        alpha = self.alpha.view(1, -1, 1, 1)
        return h + alpha * delta_h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdapterBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, adapter_config: dict | None = None):
        super().__init__()
        self.backbone = backbone
        if adapter_config is None:
            adapter_config = {
                "reduction": 4,
                "layers": ["layer3", "layer4"],
                "variant": "conv_parallel",
            }
        self.config = adapter_config

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleDict()
        layer_channels = {
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
        }
        for layer_name in adapter_config["layers"]:
            if layer_name in layer_channels:
                self.adapters[layer_name] = ConvAdapter(
                    in_channels=layer_channels[layer_name],
                    reduction=adapter_config["reduction"],
                    variant=adapter_config["variant"],
                )

    def forward(self, x: torch.Tensor):
        intermediates = {}

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.se1(x)
        if "layer1" in self.adapters:
            x = self.adapters["layer1"](x)
            intermediates["adapter1"] = x

        x = self.backbone.layer2(x)
        x = self.backbone.se2(x)
        if "layer2" in self.adapters:
            x = self.adapters["layer2"](x)
            intermediates["adapter2"] = x

        x = self.backbone.layer3(x)
        x = self.backbone.se3(x)
        if "layer3" in self.adapters:
            x = self.adapters["layer3"](x)
            intermediates["adapter3"] = x

        x = self.backbone.layer4(x)
        x = self.backbone.se4(x)
        x = self.backbone.spatial_attn(x)
        if "layer4" in self.adapters:
            x = self.adapters["layer4"](x)
            intermediates["adapter4"] = x

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return features, intermediates

    def num_adapter_parameters(self) -> dict[str, float | int]:
        adapter_params = sum(adapter.num_parameters() for adapter in self.adapters.values())
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "adapter_params": adapter_params,
            "total_params": total_params,
            "percentage": 100 * adapter_params / total_params,
        }


def create_adapter_model(backbone, stage, adapter_config=None):
    from .backbone import (
        create_stage1_head,
        create_stage2_head,
        create_stage3_ab_head,
        create_stage3_rect_head,
    )

    adapter_backbone = AdapterBackbone(backbone, adapter_config)

    if stage == 1:
        head = create_stage1_head()
    elif stage == 2:
        head = create_stage2_head()
    elif stage == "3_rect":
        head = create_stage3_rect_head()
    elif stage == "3_ab":
        head = create_stage3_ab_head()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    class AdapterModel(nn.Module):
        def __init__(self, adapter_backbone, head):
            super().__init__()
            self.adapter_backbone = adapter_backbone
            self.head = head

        def forward(self, x):
            features, _ = self.adapter_backbone(x)
            return self.head(features)

        def num_adapter_parameters(self):
            return self.adapter_backbone.num_adapter_parameters()

    return AdapterModel(adapter_backbone, head)
