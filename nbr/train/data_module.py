"""Lightning DataModule for basket sequence datasets."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from nbr.data.dataset import BasketCollator, BasketSequenceDataset
from nbr.data.split import split_user_baskets


class BasketDataModule(LightningDataModule):
    """Build train/val/test dataloaders from processed parquet."""

    def __init__(
        self,
        processed_dir: str | Path,
        batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self._train: BasketSequenceDataset | None = None
        self._val: BasketSequenceDataset | None = None
        self._test: BasketSequenceDataset | None = None
        self._collator: BasketCollator | None = None

    @property
    def vocab_size(self) -> int:
        if self._collator is None:
            raise RuntimeError("DataModule not setup yet.")
        return self._collator.vocab_size

    def setup(self, stage: str | None = None) -> None:
        baskets_path = self.processed_dir / "baskets.parquet"
        df = pl.read_parquet(baskets_path)
        train_df, val_df, test_df = split_user_baskets(df)

        vocab_size = int(df["item_id"].max()) + 1
        self._collator = BasketCollator(vocab_size=vocab_size)

        self._train = BasketSequenceDataset(train_df, max_seq_len=self.max_seq_len)
        self._val = BasketSequenceDataset(val_df, max_seq_len=self.max_seq_len)
        self._test = BasketSequenceDataset(test_df, max_seq_len=self.max_seq_len)

    def train_dataloader(self) -> DataLoader:
        if self._train is None or self._collator is None:
            raise RuntimeError("DataModule not setup yet.")
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collator,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        if self._val is None or self._collator is None:
            raise RuntimeError("DataModule not setup yet.")
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collator,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        if self._test is None or self._collator is None:
            raise RuntimeError("DataModule not setup yet.")
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collator,
            pin_memory=torch.cuda.is_available(),
        )
