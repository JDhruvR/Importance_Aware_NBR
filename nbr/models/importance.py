"""Importance head and initialization loss for per-item importance scoring.

The importance head is a lightweight MLP that maps contextualized item
representations from the BERT encoder to per-item importance weights in [0, 1].
It is initialized to reproduce the geometric alpha_idf scores (T3.1) and then
trained freely, allowing it to learn basket-specific context that the global
score cannot capture.

Reference: paper Section 5.2, Equation 6.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImportanceHead(nn.Module):
    """Two-layer MLP producing per-item importance weights.

    Architecture: Linear(D, D//2) → GELU → Linear(D//2, 1) → Sigmoid

    The MLP operates on contextualized item representations h_i (not raw
    embeddings) because h_i encodes item i in the context of the current
    basket: the same item may be more or less load-bearing depending on
    what else is present.  A sigmoid (not softmax) output allows
    independent per-item scores rather than a zero-sum competition.
    """

    def __init__(self, dim: int) -> None:
        """
        Args:
            dim: embedding dimension D (must be even).
        """
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, item_reprs: torch.Tensor) -> torch.Tensor:
        """Compute per-item importance weights.

        Args:
            item_reprs: (B*T, S, D) — contextualized item representations
                from the BERT encoder.

        Returns:
            importance: (B*T, S) — importance weights in [0, 1].
        """
        return self.mlp(item_reprs).squeeze(-1)  # (B*T, S, 1) → (B*T, S)


def importance_init_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for pre-training the importance head against alpha_idf targets.

    Only real (non-padding) items contribute to the loss.  The target
    alpha_idf scores should already be normalized to [0, 1] before
    being passed here.

    Args:
        predicted: (B*T, S) — importance weights from ImportanceHead.
        target: (B*T, S) — alpha_idf target scores, normalized to [0, 1].
        mask: (B*T, S) bool — True for real items, False for padding.

    Returns:
        Scalar MSE loss averaged over all real items.
    """
    diff = (predicted - target) ** 2  # (B*T, S)
    diff = diff * mask.float()  # zero out padding positions
    n_real = mask.sum().clamp(min=1)  # avoid division by zero
    return diff.sum() / n_real
