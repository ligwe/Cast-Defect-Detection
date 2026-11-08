from typing import Optional

import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "head" not in name and "fc" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(model_name: str, num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    return ViTClassifier(model_name=model_name, num_classes=num_classes, pretrained=True, freeze_backbone=freeze_backbone)
