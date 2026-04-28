"""BERT-style intra-basket encoder using TransformerEncoderLayer."""

from __future__ import annotations

import torch
import torch.nn as nn


class IntraBasketEncoder(nn.Module):
    """Encodes items within each basket using a Transformer encoder with a CLS token.

    Baskets are unordered sets, so no positional encoding is applied.
    A learned CLS token is prepended to each basket; the CLS output serves
    as the basket-level representation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            dim: embedding dimension D.
            num_heads: number of attention heads.
            num_layers: number of TransformerEncoderLayer blocks.
            dropout: dropout probability.
        """
        super().__init__()
        self.dim = dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode items within baskets.

        Args:
            x: (B*T, S, D) — item embeddings for all baskets in the batch,
               flattened across time (B batches * T timesteps).
            mask: (B*T, S) bool — True for real items, False for padding.

        Returns:
            cls_repr: (B*T, D) — CLS token output (basket-level representation).
            item_reprs: (B*T, S, D) — per-item outputs from the encoder.
        """
        bt, s, d = x.shape  # (B*T, S, D)

        # Prepend CLS token to each basket
        cls_expanded = self.cls_token.expand(bt, -1, -1)  # (B*T, 1, D)
        x_with_cls = torch.cat([cls_expanded, x], dim=1)  # (B*T, S+1, D)

        # Build mask with CLS: CLS is always real (True), rest from item mask
        cls_mask = torch.ones(bt, 1, dtype=torch.bool, device=x.device)
        mask_with_cls = torch.cat([cls_mask, mask], dim=1)  # (B*T, S+1)

        # Transformer expects key_padding_mask where True means "to be ignored"
        # so we invert our mask
        key_padding_mask = ~mask_with_cls  # (B*T, S+1), True = ignore

        out = self.transformer(x_with_cls, src_key_padding_mask=key_padding_mask)
        out = self.layer_norm(out)

        cls_repr = out[:, 0, :]  # (B*T, D)
        item_reprs = out[:, 1:, :]  # (B*T, S, D)

        return cls_repr, item_reprs
