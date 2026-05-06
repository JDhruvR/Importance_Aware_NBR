"""Causal GPT-style inter-basket encoder with RoPE attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange


class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings.

    Applies cos/sin rotation to Q and K pairs based on position index.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        """
        Args:
            dim: total attention dimension D (must be divisible by num_heads).
            num_heads: number of attention heads.
            dropout: attention dropout probability.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def _apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding to pairs of dimensions.

        Args:
            x: (B, H, T, head_dim) — queries or keys.
            pos: (T,) — position indices.

        Returns:
            (B, H, T, head_dim) — rotated tensor.
        """
        # Split head_dim into pairs
        x1, x2 = x[..., ::2], x[..., 1::2]  # (B, H, T, head_dim//2) each
        sin = torch.sin(pos).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, T, 1)
        cos = torch.cos(pos).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, T, 1)
        # Rotate: [x1*cos - x2*sin, x1*sin + x2*cos]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) — input tensor.
            causal_mask: (T, T) bool — True where attention is allowed.

        Returns:
            (B, T, D) — attention output.
        """
        b, t, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, D)

        # Reshape for multi-head: (B, T, H, head_dim) -> (B, H, T, head_dim)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        # Apply RoPE to Q and K
        pos = torch.arange(t, dtype=q.dtype, device=q.device)  # (T,)
        q = self._apply_rope(q, pos)
        k = self._apply_rope(k, pos)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Apply causal mask: set masked positions to -inf
        attn = attn.masked_fill(~causal_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, H, T, head_dim)
        out = rearrange(out, "b h t d -> b t (h d)")  # (B, T, D)
        return self.out_proj(out)


class CausalBasketGPT(nn.Module):
    """Stack of RoPEAttention + FFN blocks for inter-basket modeling.

    Causal mask prevents basket t from attending to baskets t' > t.
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
            dim: hidden dimension D.
            num_heads: number of attention heads.
            num_layers: number of blocks.
            dropout: dropout probability.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict(
                {
                    "attn": RoPEAttention(dim, num_heads, dropout),
                    "ln1": nn.LayerNorm(dim),
                    "ffn": nn.Sequential(
                        nn.Linear(dim, dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim * 4, dim),
                        nn.Dropout(dropout),
                    ),
                    "ln2": nn.LayerNorm(dim),
                }
            )
            self.layers.append(block)
        self.final_ln = nn.LayerNorm(dim)

    def _make_causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        """Create a (T, T) causal mask where mask[i, j] = True iff j <= i."""
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))
        return mask

    def forward(
        self,
        basket_reprs: torch.Tensor,
        basket_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            basket_reprs: (B, T, D) — basket-level representations.
            basket_mask: (B, T) bool — True for real baskets.

        Returns:
            (B, T, D) — predicted next-basket representations at each position.
        """
        b, t, d = basket_reprs.shape
        causal_mask = self._make_causal_mask(t, basket_reprs.device)  # (T, T)

        x = basket_reprs
        for block in self.layers:
            # Attention with residual + LayerNorm
            attn_out = block["attn"](block["ln1"](x), causal_mask)
            x = x + attn_out

            # FFN with residual + LayerNorm
            ffn_out = block["ffn"](block["ln2"](x))
            x = x + ffn_out

        x = self.final_ln(x)

        # Apply basket mask: zero out padding positions
        x = x.masked_fill(~basket_mask.unsqueeze(-1), 0.0)

        return x
