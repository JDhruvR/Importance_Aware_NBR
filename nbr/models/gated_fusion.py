import torch
import torch.nn as torch_nn


class DualStreamFusion(torch_nn.Module):
    """
    Fuses the full basket representation and the importance-weighted core
    basket representation using a learned gate.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # W_g learns to project the concatenated [full; core] into gate values
        self.W_g = torch_nn.Linear(2 * dim, dim, bias=True)

    def forward(
        self,
        cls_repr: torch.Tensor,
        item_reprs: torch.Tensor,
        importance: torch.Tensor,
        item_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_repr: (B*T, D) - Full basket summary (from [CLS] token)
            item_reprs: (B*T, S, D) - Individual item representations
            importance: (B*T, S) - Importance scores [0, 1] from ImportanceHead
            item_mask: (B*T, S) - Boolean mask for real items (True for real)

        Returns:
            fused_repr: (B*T, D) - The gated combination of full and core baskets
        """
        # 1. Mask importance scores to ignore padding items
        masked_importance = importance * item_mask.float()

        # 2. Normalize importance scores so they sum to 1 over the sequence (S) dimension
        # Add epsilon to prevent division by zero for completely empty padding baskets
        importance_sum = masked_importance.sum(dim=1, keepdim=True) + 1e-9
        normalized_importance = masked_importance / importance_sum

        # 3. Compute basket_core: the importance-weighted mean of items
        # normalized_importance shape: (B*T, S) -> unsqueeze to (B*T, S, 1) for broadcasting
        basket_core = (normalized_importance.unsqueeze(-1) * item_reprs).sum(dim=1)

        # 4. Compute the gate `g`
        # Concatenate along the hidden dimension (D)
        concat_repr = torch.cat([cls_repr, basket_core], dim=-1)  # (B*T, 2*D)
        g = torch.sigmoid(self.W_g(concat_repr))                  # (B*T, D)

        # 5. Fuse representations
        fused_repr = g * cls_repr + (1 - g) * basket_core         # (B*T, D)

        return fused_repr