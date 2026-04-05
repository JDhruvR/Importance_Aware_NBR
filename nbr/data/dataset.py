"""PyTorch Dataset and collator for basket sequences."""

from __future__ import annotations

from collections import defaultdict

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


class BasketSequenceDataset(Dataset):
    """One training example per user: predict the last basket from history.

    For each user, the input is up to ``max_seq_len`` most recent baskets
    (excluding the last one), and the target is the last basket.
    """

    def __init__(self, df: pl.DataFrame, max_seq_len: int) -> None:
        """
        Args:
            df: train DataFrame with columns [user_id: i32, order_idx: i32, item_id: i32].
            max_seq_len: maximum number of historical baskets to use as input.
        """
        self.max_seq_len = max_seq_len

        # Group baskets by user, sorted by order_idx
        # user_baskets[user_id] = [[item_ids for basket 0], [item_ids for basket 1], ...]
        self.user_baskets: dict[int, list[list[int]]] = {}
        self.user_ids: list[int] = []

        # Sort by user_id then order_idx
        df_sorted = df.sort(["user_id", "order_idx"])

        current_user: int | None = None
        current_order: int | None = None
        current_basket: list[int] = []
        user_basket_list: list[list[int]] = []

        for row in df_sorted.iter_rows(named=True):
            uid = int(row["user_id"])
            oid = int(row["order_idx"])
            iid = int(row["item_id"])

            if uid != current_user:
                # Save previous user
                if current_user is not None:
                    if current_basket:
                        user_basket_list.append(current_basket)
                    if user_basket_list:
                        self.user_baskets[current_user] = user_basket_list
                        self.user_ids.append(current_user)
                current_user = uid
                current_order = oid
                current_basket = [iid]
                user_basket_list = []
            elif oid != current_order:
                # New basket for same user
                user_basket_list.append(current_basket)
                current_order = oid
                current_basket = [iid]
            else:
                # Same basket, add item
                current_basket.append(iid)

        # Don't forget the last user
        if current_user is not None:
            if current_basket:
                user_basket_list.append(current_basket)
            if user_basket_list:
                self.user_baskets[current_user] = user_basket_list
                self.user_ids.append(current_user)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training example.

        Returns:
            {
                "item_seqs": list[list[int]]  # up to max_seq_len historical baskets
                "target_items": list[int]     # items in the target (last) basket
                "user_id": int
            }
        """
        uid = self.user_ids[idx]
        baskets = self.user_baskets[uid]

        # Last basket is the target
        target_items = baskets[-1]

        # All baskets before the last one are the input history
        history = baskets[:-1]

        # Take up to max_seq_len most recent baskets
        if len(history) > self.max_seq_len:
            history = history[-self.max_seq_len :]

        return {
            "item_seqs": history,
            "target_items": target_items,
            "user_id": uid,
        }


class BasketCollator:
    """Pad variable-length basket sequences into a batch of tensors."""

    def __init__(self, vocab_size: int) -> None:
        """
        Args:
            vocab_size: total number of unique items (for multi-hot target).
        """
        self.vocab_size = vocab_size

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a list of samples into a padded batch.

        Returns:
            {
                "items": (B, T, S) int64 — item IDs, 0-padded
                "basket_mask": (B, T) bool — True for real baskets
                "item_mask": (B, T, S) bool — True for real items
                "target": (B, V) float32 — multi-hot ground truth
                "user_ids": (B,) int64
            }
        """
        batch_size = len(batch)

        # Find max sequence length (T) and max basket size (S) in this batch
        max_t = max(len(sample["item_seqs"]) for sample in batch)
        max_s = (
            max(len(basket) for sample in batch for basket in sample["item_seqs"])
            if max_t > 0
            else 0
        )

        # 0 is the padding index
        items = torch.zeros((batch_size, max_t, max_s), dtype=torch.int64)
        basket_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)
        item_mask = torch.zeros((batch_size, max_t, max_s), dtype=torch.bool)
        target = torch.zeros((batch_size, self.vocab_size), dtype=torch.float32)
        user_ids = torch.zeros(batch_size, dtype=torch.int64)

        for b, sample in enumerate(batch):
            user_ids[b] = sample["user_id"]

            # Multi-hot target
            for item_id in sample["target_items"]:
                if 0 < item_id < self.vocab_size:
                    target[b, item_id] = 1.0

            # Pad item sequences
            for t, basket in enumerate(sample["item_seqs"]):
                basket_mask[b, t] = True
                for s, item_id in enumerate(basket):
                    items[b, t, s] = item_id
                    item_mask[b, t, s] = True

        return {
            "items": items,
            "basket_mask": basket_mask,
            "item_mask": item_mask,
            "target": target,
            "user_ids": user_ids,
        }
