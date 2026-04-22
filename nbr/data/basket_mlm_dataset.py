"""Basket-level dataset and collator for BERT MLM warmup."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BasketSample:
    """Single basket sample."""

    user_id: int
    order_idx: int
    items: list[int]


class BasketMLMDataset(Dataset):
    """Dataset of individual baskets for MLM warmup."""

    def __init__(
        self,
        df: pl.DataFrame,
        max_items_per_basket: int | None = None,
        max_baskets: int | None = None,
    ) -> None:
        grouped = (
            df.group_by(["user_id", "order_idx"])
            .agg(pl.col("item_id").alias("items"))
            .sort(["user_id", "order_idx"])
        )

        self.samples: list[BasketSample] = []
        for row in grouped.iter_rows(named=True):
            items = [int(x) for x in row["items"]]
            if max_items_per_basket is not None and len(items) > max_items_per_basket:
                items = items[:max_items_per_basket]
            if not items:
                continue
            self.samples.append(
                BasketSample(
                    user_id=int(row["user_id"]),
                    order_idx=int(row["order_idx"]),
                    items=items,
                )
            )

        if max_baskets is not None:
            self.samples = self.samples[:max_baskets]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BasketSample:
        return self.samples[idx]


class BasketMLMCollator:
    """Pad baskets and apply masked item modeling corruption."""

    def __init__(
        self,
        num_items: int,
        mask_prob: float = 0.15,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        item_id_offset: int = 2,
        apply_mlm: bool = True,
    ) -> None:
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if item_id_offset < 2:
            raise ValueError("item_id_offset must be >= 2 to reserve PAD and MASK")
        self.num_items = num_items
        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.item_id_offset = item_id_offset
        self.apply_mlm = apply_mlm

        self.vocab_size = self.num_items + self.item_id_offset
        self.min_item_token_id = self.item_id_offset
        self.max_item_token_id = self.item_id_offset + self.num_items - 1

    def _shift_item_id(self, item_id: int) -> int:
        return item_id + self.item_id_offset

    def __call__(self, batch: list[BasketSample]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch received by BasketMLMCollator")

        batch_size = len(batch)
        max_len = max(len(sample.items) for sample in batch)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.int64)
        user_ids = torch.zeros(batch_size, dtype=torch.int64)
        order_idxs = torch.zeros(batch_size, dtype=torch.int64)

        for b, sample in enumerate(batch):
            user_ids[b] = sample.user_id
            order_idxs[b] = sample.order_idx

            seq = [self._shift_item_id(item_id) for item_id in sample.items]
            seq_len = len(seq)

            input_ids[b, :seq_len] = torch.tensor(seq, dtype=torch.int64)
            attention_mask[b, :seq_len] = True

            if not self.apply_mlm:
                continue

            candidates = torch.arange(seq_len)
            if len(candidates) == 0:
                continue

            num_to_mask = max(1, int(round(self.mask_prob * seq_len)))
            num_to_mask = min(num_to_mask, seq_len)
            perm = torch.randperm(seq_len)
            mask_positions = candidates[perm[:num_to_mask]]

            original_tokens = input_ids[b, mask_positions].clone()
            labels[b, mask_positions] = original_tokens

            probs = torch.rand(num_to_mask)
            mask_mask = probs < 0.8
            random_mask = (probs >= 0.8) & (probs < 0.9)

            if mask_mask.any():
                input_ids[b, mask_positions[mask_mask]] = self.mask_token_id

            if random_mask.any():
                random_tokens = torch.randint(
                    low=self.min_item_token_id,
                    high=self.max_item_token_id + 1,
                    size=(int(random_mask.sum().item()),),
                )
                input_ids[b, mask_positions[random_mask]] = random_tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "user_ids": user_ids,
            "order_idxs": order_idxs,
        }
