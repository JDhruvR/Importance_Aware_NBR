import torch
import torch.nn as nn

class IntentProjection(nn.Module):
    def __init__(self, dim: int, intent_dim: int):
        super().__init__()
        self.dim = dim
        self.intent_dim = intent_dim
        # P is shape (D, dk). Initialize orthogonally.
        self.P = nn.Parameter(torch.empty(dim, intent_dim))
        nn.init.orthogonal_(self.P)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, D) vector (e.g., from GPT)
        Returns:
            intent_repr: (B, D)
            fill_repr: (B, D)
        """
        # P @ P.T is the projection matrix onto the intent subspace
        proj = self.P @ self.P.T  # (D, D)
        intent_repr = x @ proj    # (B, D)
        fill_repr = x - intent_repr # (B, D)
        return intent_repr, fill_repr

    def orthogonalize_(self):
        """In-place Gram-Schmidt re-orthonormalization of P's columns."""
        with torch.no_grad():
            for i in range(self.intent_dim):
                v = self.P[:, i].clone()
                for j in range(i):
                    u = self.P[:, j]
                    # subtract projection of v onto u
                    v -= (torch.dot(v, u) / (torch.dot(u, u) + 1e-9)) * u
                # normalize
                self.P[:, i] = v / (torch.norm(v) + 1e-9)

    def orthogonality_loss(self, intent_repr: torch.Tensor, fill_repr: torch.Tensor) -> torch.Tensor:
        """Computes the penalty to ensure intent and fill representations are strictly orthogonal."""
        # Dot product across the hidden dimension, squared, then averaged over batch
        return (intent_repr * fill_repr).sum(dim=-1).pow(2).mean()
