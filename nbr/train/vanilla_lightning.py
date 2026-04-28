"""Lightning module for the vanilla and BERT+GPT baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from nbr.metrics.ranking import build_history_multihot, ndcg_at_k, recall_at_k
from nbr.models.bert_gpt.model import BertGptNBR
from nbr.models.vanilla import VanillaNBR


class VanillaLitModule(LightningModule):
    """LightningModule wrapper for vanilla or BERT+GPT models."""

    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        dim: int,
        num_heads: int,
        encoder_layers: int,
        gpt_layers: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        max_epochs: int,
        k_values: list[int],
        pretrained_bert_path: str | None = None,
        item_id_offset: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_type == "vanilla":
            self.model: nn.Module = VanillaNBR(
                vocab_size=vocab_size,
                dim=dim,
                num_heads=num_heads,
                encoder_layers=encoder_layers,
                gpt_layers=gpt_layers,
                dropout=dropout,
            )
        elif model_type == "bert_gpt":
            self.model = BertGptNBR(
                vocab_size=vocab_size,
                dim=dim,
                num_heads=num_heads,
                encoder_layers=encoder_layers,
                gpt_layers=gpt_layers,
                dropout=dropout,
                item_id_offset=item_id_offset,
            )
            if pretrained_bert_path is not None:
                print(f"[BertGptNBR] Loading pretrained BERT weights from {pretrained_bert_path}")
                bundle = torch.load(pretrained_bert_path, map_location="cpu")
                
                # The bundle from train_bert.py has keys: state_dict -> {embedding.weight, encoder}
                state_dict = bundle["state_dict"]
                
                # Load ItemEmbedding weights
                self.model.item_embedding.embedding.load_state_dict({"weight": state_dict["embedding.weight"]})
                
                # Load IntraBasketEncoder weights
                self.model.encoder.load_state_dict(state_dict["encoder"])
                
                print("[BertGptNBR] Successfully loaded pretrained item_embedding and encoder.")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["items"], batch["item_mask"], batch["basket_mask"])

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self.forward(batch)
        loss = self.model.loss(logits, batch["target"], batch["basket_mask"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _shared_eval(self, batch: dict[str, torch.Tensor], prefix: str) -> None:
        # Skip batches with no historical baskets (e.g., users with empty history)
        if batch["basket_mask"].shape[1] == 0 or not batch["basket_mask"].any():
            return

        logits = self.forward(batch)
        loss = self.model.loss(logits, batch["target"], batch["basket_mask"])

        last_logits = logits[:, -1]
        # For metrics, we only care about predicting the LAST basket in the sequence
        target = batch["target"][:, -1]
        history = build_history_multihot(batch["items"], batch["item_mask"], target.shape[-1])

        for k in self.hparams.k_values:
            recall = recall_at_k(last_logits, target, k).mean()
            ndcg = ndcg_at_k(last_logits, target, k).mean()
            repeat_target, explore_target = target * history, target * (1.0 - history)
            repeat_recall = recall_at_k(last_logits, repeat_target, k).mean()
            explore_recall = recall_at_k(last_logits, explore_target, k).mean()
            self.log(f"{prefix}/recall@{k}", recall, prog_bar=False, on_epoch=True)
            self.log(f"{prefix}/ndcg@{k}", ndcg, prog_bar=False, on_epoch=True)
            self.log(f"{prefix}/repeat_recall@{k}", repeat_recall, prog_bar=False, on_epoch=True)
            self.log(f"{prefix}/explore_recall@{k}", explore_recall, prog_bar=False, on_epoch=True)

        self.log(f"{prefix}/loss", loss, prog_bar=False, on_epoch=True)
        if prefix == "val":
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_eval(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_eval(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=max(1, int(self.hparams.warmup_steps)),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
