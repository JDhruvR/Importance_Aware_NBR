"""Data loaders for basket-BERT MLM warmup."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from nbr.data.basket_mlm_dataset import BasketMLMCollator, BasketMLMDataset
from nbr.data.split import split_user_baskets


class BasketBERTDataModule:
    """Build train/val dataloaders of basket sentences for MLM."""

    def __init__(
        self,
        processed_dir: str | Path,
        batch_size: int,
        num_workers: int,
        mask_prob: float,
        val_mask_prob: float,
        max_items_per_basket: int | None = None,
        max_train_baskets: int | None = None,
        max_val_baskets: int | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask_prob = mask_prob
        self.val_mask_prob = val_mask_prob
        self.max_items_per_basket = max_items_per_basket
        self.max_train_baskets = max_train_baskets
        self.max_val_baskets = max_val_baskets

        self._num_items: int | None = None
        self._train: BasketMLMDataset | None = None
        self._val: BasketMLMDataset | None = None

    @property
    def num_items(self) -> int:
        if self._num_items is None:
            raise RuntimeError("DataModule not setup yet")
        return self._num_items

    @property
    def val_dataset(self) -> BasketMLMDataset:
        if self._val is None:
            raise RuntimeError("DataModule not setup yet")
        return self._val

    def setup(self, stage: str | None = None) -> None:
        baskets_path = self.processed_dir / "baskets.parquet"
        df = pl.read_parquet(baskets_path)
        train_df, val_df, _ = split_user_baskets(df)

        self._num_items = int(df["item_id"].max()) + 1
        self._train = BasketMLMDataset(
            train_df,
            max_items_per_basket=self.max_items_per_basket,
            max_baskets=self.max_train_baskets,
        )
        self._val = BasketMLMDataset(
            val_df,
            max_items_per_basket=self.max_items_per_basket,
            max_baskets=self.max_val_baskets,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train is None:
            raise RuntimeError("DataModule not setup yet")
        collator = BasketMLMCollator(
            num_items=self.num_items,
            mask_prob=self.mask_prob,
            apply_mlm=True,
        )
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        if self._val is None:
            raise RuntimeError("DataModule not setup yet")
        collator = BasketMLMCollator(
            num_items=self.num_items,
            mask_prob=self.val_mask_prob,
            apply_mlm=True,
        )
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
