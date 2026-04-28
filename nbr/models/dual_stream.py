"""Dual-stream model placeholder built on vanilla backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from nbr.models.vanilla import VanillaNBR


class DualStreamNBR(nn.Module):
    """Dual-stream NBR (placeholder implementation).

    This scaffolds the interface needed by training; it currently delegates to
    the vanilla model and returns no auxiliary losses.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        encoder_layers: int,
        gpt_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = VanillaNBR(
            vocab_size=vocab_size,
            dim=dim,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            gpt_layers=gpt_layers,
            dropout=dropout,
        )

    def forward(
        self,
        items: torch.Tensor,
        item_mask: torch.Tensor,
        basket_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = self.backbone(items, item_mask, basket_mask)
        return {"logits": logits, "losses": {}}

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        basket_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone.loss(logits, target, basket_mask)
