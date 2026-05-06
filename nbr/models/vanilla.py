"""Vanilla Next Basket Recommendation model.

ItemEmbedding -> IntraBasketEncoder (mean pool over items) ->
CausalBasketGPT -> dot product against ItemEmbedding.weight -> logits over vocab.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nbr.models.encoder import IntraBasketEncoder
from nbr.models.gpt import CausalBasketGPT
from nbr.models.embeddings import ItemEmbedding


class VanillaNBR(nn.Module):
    """Vanilla NBR: mean-pool intra-basket encoding + causal inter-basket modeling."""

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
        cls_repr, item_reprs = self.encoder(x_flat, mask_flat)
        # cls_repr: (B*T, D), item_reprs: (B*T, S, D)

        # Mean pool over items for basket representation: (B*T, D)
        # Only average over real items
        valid_counts = mask_flat.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B*T, 1)
        basket_reprs = (item_reprs * mask_flat.unsqueeze(-1)).sum(dim=1) / valid_counts  # (B*T, D)

        # Reshape back to (B, T, D)
        basket_reprs = basket_reprs.view(b, t, self.dim)

        # Inter-basket causal modeling: (B, T, D)
        next_pred = self.gpt(basket_reprs, basket_mask)

        # Dot product against item embedding weights: (B, T, D) @ (V, D).T -> (B, T, V)
        logits = next_pred @ self.item_embedding.embedding.weight.T  # (B, T, V)

        return logits

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        basket_mask_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss over real baskets.

        Args:
            logits: (B, T, V) — predicted logits.
            targets: (B, T, V) — multi-hot ground truth distributions.
            basket_mask_target: (B, T) bool — True for real baskets with a target.

        Returns:
            Scalar loss averaged over real baskets.
        """
        # Convert normalised multi-hot back to pure binary multi-hot
        binary_targets = (targets > 0).float()

        # Compute BCE per element
        per_element_loss = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        )  # (B, T, V)

        # Sum over V axis
        per_pos_loss = per_element_loss.sum(dim=-1)

        # Mask out padding baskets and average
        mask = basket_mask_target.float()  # (B, T)
        num_valid = mask.sum().clamp(min=1.0)
        return (per_pos_loss * mask).sum() / num_valid