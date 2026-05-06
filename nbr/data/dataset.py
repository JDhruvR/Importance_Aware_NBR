"""PyTorch Dataset and collator for causal GPT-style basket sequences.

One sample per user: all baskets are returned.
The collator builds input (all baskets) and target (shifted-by-1) tensors
so that a single forward pass trains all n-1 next-basket prediction positions.
"""

from __future__ import annotations

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


class BasketSequenceDataset(Dataset):
    """All-baskets dataset for causal GPT-style training.

    For each user, returns ALL baskets in order. The collator handles the
    causal shift: input at position t predicts target at position t+1.

    Users with fewer than 2 baskets are excluded (need at least one
    input-target pair).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        max_seq_len: int,
        min_history_len: int = 0,
    ) -> None:
        """
        Args:
            df: train/val/test DataFrame with columns
                [user_id: i32, order_idx: i32, item_id: i32].
            max_seq_len: maximum total baskets to keep per user (most recent).
            min_history_len: minimum number of baskets required (before filtering).
        """
        if min_history_len < 0:
            raise ValueError("min_history_len must be >= 0")
        self.max_seq_len = max_seq_len
        self.min_history_len = min_history_len

        self.user_baskets: dict[int, list[list[int]]] = {}
        self.user_ids: list[int] = []

        df_sorted = df.sort(["user_id", "order_idx"])

        current_user: int | None = None
        current_order: int | None = None
        current_basket: list[int] = []
        user_basket_list: list[list[int]] = []

        def _flush_user(uid: int, baskets: list[list[int]]) -> None:
            if len(baskets) < 2:          # need at least one input→target pair
                return
            self.user_baskets[uid] = baskets
            self.user_ids.append(uid)

        for row in df_sorted.iter_rows(named=True):
            uid = int(row["user_id"])
            oid = int(row["order_idx"])
            iid = int(row["item_id"])

            if uid != current_user:
                if current_user is not None:
                    if current_basket:
                        user_basket_list.append(current_basket)
                    _flush_user(current_user, user_basket_list)
                current_user = uid
                current_order = oid
                current_basket = [iid]
                user_basket_list = []
            elif oid != current_order:
                user_basket_list.append(current_basket)
                current_order = oid
                current_basket = [iid]
            else:
                current_basket.append(iid)

        # flush last user
        if current_user is not None:
            if current_basket:
                user_basket_list.append(current_basket)
            _flush_user(current_user, user_basket_list)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return all baskets for a user.

        Returns:
            {
                "baskets": list[list[int]]  — all baskets, up to max_seq_len
                "user_id": int
            }
        """
        uid = self.user_ids[idx]
        baskets = self.user_baskets[uid]

        # Keep most recent max_seq_len baskets
        if len(baskets) > self.max_seq_len:
            baskets = baskets[-self.max_seq_len:]

        return {
            "baskets": baskets,
            "user_id": uid,
        }


class BasketCollator:
    """Pad variable-length basket sequences into a batch of tensors.

    Causal GPT convention (identical to nanoGPT):
        items[:, t, :]  = basket t   (input)
        targets[:, t, :] = multi-hot of basket t+1  (supervision)

    The last basket position (T-1) has no target — it is masked out in the
    loss via basket_mask_target which is basket_mask with the last position False.
    """

    def __init__(self, vocab_size: int, item_id_offset: int = 0) -> None:
        """
        Args:
            vocab_size: total number of unique items (raw, without offset).
            item_id_offset: offset for item IDs (e.g. 2 for BERT to reserve PAD/MASK).
        """
        self.raw_vocab_size = vocab_size
        self.item_id_offset = item_id_offset
        self.vocab_size = vocab_size + item_id_offset

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a list of samples into a padded batch.

        Returns:
            {
                "items":         (B, T, S) int64  — item IDs for each basket position
                "item_mask":     (B, T, S) bool   — True for real items
                "basket_mask":   (B, T) bool       — True for real basket positions
                "targets":       (B, T, V) float32 — normalised multi-hot for basket t+1
                                                     (position T-1 is zeros / masked)
                "basket_mask_target": (B, T) bool  — valid target positions (basket_mask
                                                     shifted: positions 0..T-2 are True
                                                     for real baskets that have a next basket)
                "user_ids":      (B,) int64
            }
        """
        batch_size = len(batch)

        # Sequence length T = full basket list length (all baskets are input)
        max_t = max(len(sample["baskets"]) for sample in batch)
        max_s = (
            max(len(b) for sample in batch for b in sample["baskets"])
            if max_t > 0
            else 1
        )
        max_s = max(max_s, 1)

        items = torch.zeros((batch_size, max_t, max_s), dtype=torch.int64)
        basket_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)
        item_mask = torch.zeros((batch_size, max_t, max_s), dtype=torch.bool)
        targets = torch.zeros((batch_size, max_t, self.vocab_size), dtype=torch.float32)
        basket_mask_target = torch.zeros((batch_size, max_t), dtype=torch.bool)
        user_ids = torch.zeros(batch_size, dtype=torch.int64)

        for b, sample in enumerate(batch):
            user_ids[b] = sample["user_id"]
            baskets = sample["baskets"]
            n = len(baskets)          # actual number of baskets for this user

            # Fill input baskets
            for t, basket in enumerate(baskets):
                basket_mask[b, t] = True
                for s, item_id in enumerate(basket):
                    # Shift item ID by offset (e.g. for BERT)
                    items[b, t, s] = item_id + self.item_id_offset
                    item_mask[b, t, s] = True

            # Fill targets: target at position t = multi-hot of basket t+1
            for t in range(n - 1):
                next_basket = baskets[t + 1]
                raw = torch.zeros(self.vocab_size, dtype=torch.float32)
                for item_id in next_basket:
                    shifted_id = item_id + self.item_id_offset
                    if 0 <= shifted_id < self.vocab_size:
                        raw[shifted_id] = 1.0
                s = raw.sum()
                if s > 0:
                    targets[b, t] = raw / s          # uniform dist over basket items
                basket_mask_target[b, t] = True      # this position has a valid target

        return {
            "items": items,
            "item_mask": item_mask,
            "basket_mask": basket_mask,
            "targets": targets,
            "basket_mask_target": basket_mask_target,
            "user_ids": user_ids,
        }
