import logging
import math

from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, None] * self.weight[None] + self.bias[None]


class FTTransformerBackbone(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d = config["d_token"]
        n_features = config["input_dim"]
        seq_len = n_features + 1

        self.tokenizer = FeatureTokenizer(n_features, d)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config["attention_heads"],
            dim_feedforward=int(d * config["ffn_dim_multiplier"]),
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config["n_blocks"],
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d)

        logger.info(
            "FTTransformerBackbone | features=%d | seq_len=%d | d_token=%d | blocks=%d | heads=%d | params=%s",
            n_features,
            seq_len,
            d,
            config["n_blocks"],
            config["attention_heads"],
            f"{sum(p.numel() for p in self.parameters()):,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        tokens = tokens + self.pos_embedding

        out = self.transformer(tokens)

        cls_repr = self.norm(out[:, 0, :])
        return cls_repr


class IIoTFLNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d = config["d_token"]

        self.backbone = FTTransformerBackbone(config)

        # RUL Head w/ Softplut for non-negative output
        self.rul_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(d // 2, 1),
            nn.Softplus(),
        )

        # Failure head, activation applied on BCEWithLogitsLoss
        self.failure_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(d // 2, 1),
        )

        total = sum(p.numel() for p in self.parameters())
        backbone_p = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            "IIoTFLNet | total_params=%s | backbone=%s | heads=%s",
            f"{total:,}",
            f"{backbone_p:,}",
            f"{total - backbone_p:,}",
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        rul_pred = self.rul_head(x)
        failure_logit = self.failure_head(x)
        return rul_pred, failure_logit


class DualTaskLoss(nn.Module):
    def __init__(
        self,
        rul_weight: float = 1.0,
        fail_weight: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.rul_weight = rul_weight
        self.fail_weight = fail_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def set_pos_weight(self, pos_weight: torch.Tensor):
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        rul_pred: torch.Tensor,
        rul_true: torch.Tensor,
        failure_logit: torch.Tensor,
        failure_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rul_loss = self.mse(
            torch.log1p(rul_pred.squeeze()),
            torch.log1p(rul_true),
        )
        failure_loss = self.bce(failure_logit.squeeze(), failure_true)

        total = self.rul_weight * rul_loss + self.fail_weight * failure_loss
        return total, rul_loss.detach(), failure_loss.detach()
