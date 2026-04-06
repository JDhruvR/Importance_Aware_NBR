"""Vanilla BERT+GPT Next Basket Recommendation model.

Intra-basket: TransformerEncoder with CLS token (BERT-style).
Inter-basket: Causal GPT with RoPE attention.
Prediction: dot product against item embedding weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nbr.models.embeddings import ItemEmbedding
from nbr.models.encoder import IntraBasketEncoder
from nbr.models.gpt import CausalBasketGPT


class BertGptNBR(nn.Module):
    """BERT-style intra-basket encoder + GPT-style inter-basket model."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 2,
        gpt_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            vocab_size: total number of unique items.
            dim: embedding / hidden dimension D.
            num_heads: number of attention heads.
            encoder_layers: number of intra-basket encoder layers.
            gpt_layers: number of inter-basket GPT layers.
            dropout: dropout probability.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.item_embedding = ItemEmbedding(vocab_size, dim, padding_idx=0)
        self.encoder = IntraBasketEncoder(dim, num_heads, encoder_layers, dropout)
        self.gpt = CausalBasketGPT(dim, num_heads, gpt_layers, dropout)

    def forward(
        self,
        items: torch.Tensor,
        item_mask: torch.Tensor,
        basket_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            items: (B, T, S) int64 — item IDs per basket per timestep.
            item_mask: (B, T, S) bool — True for real items.
            basket_mask: (B, T) bool — True for real baskets.

        Returns:
            (B, T, V) float32 — logits over vocabulary at each timestep.
        """
        b, t, s = items.shape

        # Embed items: (B, T, S, D)
        x = self.item_embedding(items)

        # Flatten across B and T for encoder: (B*T, S, D)
        x_flat = x.view(b * t, s, self.dim)
        mask_flat = item_mask.view(b * t, s)

        # Intra-basket encoding
        cls_repr, _ = self.encoder(x_flat, mask_flat)  # (B*T, D)

        # Reshape back to (B, T, D)
        basket_reprs = cls_repr.view(b, t, self.dim)

        # Inter-basket causal modeling: (B, T, D)
        next_pred = self.gpt(basket_reprs, basket_mask)

        # Dot product against item embedding weights: (B, T, D) @ (V, D).T -> (B, T, V)
        logits = next_pred @ self.item_embedding.embedding.weight.T  # (B, T, V)

        return logits

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        basket_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss over real baskets.

        Args:
            logits: (B, T, V) — predicted logits.
            target: (B, V) — multi-hot ground truth for the target basket.
            basket_mask: (B, T) bool — True for real baskets.

        Returns:
            Scalar loss averaged over real baskets.
        """
        target_expanded = target.unsqueeze(1).expand_as(logits)  # (B, T, V)
        per_element_loss = F.binary_cross_entropy_with_logits(
            logits, target_expanded, reduction="none"
        )  # (B, T, V)

        mask = basket_mask.unsqueeze(-1).float()  # (B, T, 1)
        masked_loss = (per_element_loss * mask).sum() / mask.sum().clamp(min=1.0)
        return masked_loss
