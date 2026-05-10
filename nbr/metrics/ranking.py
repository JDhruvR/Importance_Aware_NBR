"""Ranking metrics for next-basket evaluation."""

from __future__ import annotations

import math

import torch


def recall_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Compute Recall@K for a batch.

    Args:
        preds: (B, V) scores (higher is better).
        targets: (B, V) multi-hot ground truth.
        k: cutoff.

    Returns:
        (B,) tensor with recall@k per sample.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    topk = torch.topk(preds, k=min(k, preds.shape[-1]), dim=-1).indices
    hits = torch.gather(targets, 1, topk).sum(dim=-1)
    denom = targets.sum(dim=-1).clamp(min=1.0)
    return hits / denom


def mrr_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Compute MRR@K for a batch.

    Args:
        preds: (B, V) scores (higher is better).
        targets: (B, V) multi-hot ground truth.
        k: cutoff.

    Returns:
        (B,) tensor with mrr@k per sample.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    k_eff = min(k, preds.shape[-1])
    topk = torch.topk(preds, k=k_eff, dim=-1).indices
    rel = torch.gather(targets, 1, topk)
    
    hits = rel > 0
    first_hit_idx = hits.long().argmax(dim=1)
    has_hit = hits.any(dim=1)
    
    mrr = torch.zeros(preds.shape[0], device=preds.device)
    mrr[has_hit] = 1.0 / (first_hit_idx[has_hit].float() + 1.0)
    return mrr


def ndcg_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Compute NDCG@K for a batch.

    Args:
        preds: (B, V) scores (higher is better).
        targets: (B, V) multi-hot ground truth.
        k: cutoff.

    Returns:
        (B,) tensor with ndcg@k per sample.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    k_eff = min(k, preds.shape[-1])
    topk = torch.topk(preds, k=k_eff, dim=-1).indices
    rel = torch.gather(targets, 1, topk)
    discounts = 1.0 / torch.log2(torch.arange(k_eff, device=preds.device) + 2.0)
    dcg = (rel * discounts).sum(dim=-1)

    ideal = torch.topk(targets, k=k_eff, dim=-1).values
    idcg = (ideal * discounts).sum(dim=-1).clamp(min=1e-8)
    return dcg / idcg


def repeat_explore_masks(
    targets: torch.Tensor,
    history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split targets into repeat/explore based on history.

    Args:
        targets: (B, V) multi-hot ground truth for next basket.
        history: (B, V) multi-hot history (items seen in past baskets).

    Returns:
        repeat_targets, explore_targets: both (B, V) multi-hot.
    """
    repeat_targets = targets * history
    explore_targets = targets * (1.0 - history)
    return repeat_targets, explore_targets


def build_history_multihot(
    items: torch.Tensor,
    item_mask: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Build multi-hot history from padded basket sequences.

    Args:
        items: (B, T, S) int64 item ids.
        item_mask: (B, T, S) bool mask for real items.
        vocab_size: vocab size.

    Returns:
        history: (B, V) float32 multi-hot (>=1 for seen items).
    """
    b, t, s = items.shape
    history = torch.zeros((b, vocab_size), device=items.device, dtype=torch.float32)
    flat_items = items.view(b, t * s)
    flat_mask = item_mask.view(b, t * s)
    for i in range(b):
        idx = flat_items[i][flat_mask[i]]
        idx = idx[(idx >= 0) & (idx < vocab_size)]
        if idx.numel() == 0:
            continue
        history[i].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    history = (history > 0).float()
    return history