from __future__ import annotations

import timm
import torch.nn as nn


class FrameClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
        freeze_backbone: bool = False,
        hidden_dim: int = 0,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise RuntimeError("Cannot resolve backbone feature dimension")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if hidden_dim and hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
    hidden_dim: int = 0,
) -> nn.Module:
    return FrameClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
