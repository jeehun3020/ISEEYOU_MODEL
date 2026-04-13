from __future__ import annotations

import torch
import torch.nn as nn
import timm


class TemporalLSTMClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        freeze_backbone: bool = False,
        head_type: str = "lstm",
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
            raise RuntimeError("Cannot resolve feature dimension from backbone")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head_type = str(head_type).lower()
        if self.head_type == "mean_pool":
            self.temporal = None
            temporal_dim = feature_dim
        elif self.head_type == "temporal_conv":
            self.temporal = nn.Sequential(
                nn.Conv1d(feature_dim, hidden_size, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.GELU(),
            )
            temporal_dim = hidden_size
        elif self.head_type == "gru":
            rnn_dropout = dropout if num_layers > 1 else 0.0
            self.temporal = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=rnn_dropout,
            )
            temporal_dim = hidden_size * (2 if bidirectional else 1)
        elif self.head_type == "lstm":
            rnn_dropout = dropout if num_layers > 1 else 0.0
            self.temporal = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=rnn_dropout,
            )
            temporal_dim = hidden_size * (2 if bidirectional else 1)
        else:
            raise ValueError("Unsupported head_type. Use one of: mean_pool, temporal_conv, gru, lstm")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(temporal_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        feat = self.backbone(x)  # [B*T, F]
        feat = feat.reshape(b, t, -1)

        if self.head_type == "mean_pool":
            seq_repr = feat.mean(dim=1)
        elif self.head_type == "temporal_conv":
            conv_in = feat.transpose(1, 2)
            conv_out = self.temporal(conv_in)
            seq_repr = conv_out.mean(dim=2)
        else:
            out, _ = self.temporal(feat)  # [B, T, H]
            if lengths is None:
                seq_repr = out[:, -1, :]
            else:
                lengths = lengths.to(out.device)
                lengths = torch.clamp(lengths, min=1, max=t)
                gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
                seq_repr = out.gather(dim=1, index=gather_idx).squeeze(1)

        logits = self.classifier(self.dropout(seq_repr))
        return logits


def build_temporal_model(model_cfg: dict, num_classes: int) -> nn.Module:
    return TemporalLSTMClassifier(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        bidirectional=bool(model_cfg.get("bidirectional", True)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        head_type=str(model_cfg.get("head_type", "lstm")),
    )
