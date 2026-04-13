from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ProtocolFrameClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def build_protocol_frame_model(model_cfg: dict, num_classes: int) -> nn.Module:
    return ProtocolFrameClassifier(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        hidden_dim=int(model_cfg.get("hidden_dim", 0) or 0),
    )
