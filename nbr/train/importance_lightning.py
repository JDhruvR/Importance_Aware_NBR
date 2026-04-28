"""Lightning module for importance-aware models (dual-stream/full)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from nbr.metrics.ranking import build_history_multihot, ndcg_at_k, recall_at_k


class ImportanceLitModule(LightningModule):
    """LightningModule wrapper for importance-aware models."""

    def __init__(
        self,
        model: nn.Module,
        loss_weights: dict[str, float],
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        max_epochs: int,
        k_values: list[int],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_weights = loss_weights

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.model(batch["items"], batch["item_mask"], batch["basket_mask"])
        if isinstance(output, dict):
            logits = output["logits"]
            aux_losses = output.get("losses", {})
        else:
            logits = output
            aux_losses = {}
        return logits, aux_losses

    def _primary_loss(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, "loss"):
            return self.model.loss(logits, batch["target"], batch["basket_mask"])
        target_expanded = batch["target"].unsqueeze(1).expand_as(logits)
        per_element_loss = F.binary_cross_entropy_with_logits(
            logits, target_expanded, reduction="none"
        )
        mask = batch["basket_mask"].unsqueeze(-1).float()
        return (per_element_loss * mask).sum() / mask.sum().clamp(min=1.0)

    def _combine_losses(
        self,
        primary_loss: torch.Tensor,
        aux_losses: dict[str, torch.Tensor],
        prefix: str,
    ) -> torch.Tensor:
        total = primary_loss * float(self.loss_weights.get("lambda", 1.0))
        self.log(f"{prefix}/loss_primary", primary_loss, on_epoch=True, prog_bar=False)

        weight_map = {
            "importance": float(self.loss_weights.get("gamma", 0.0)),
            "mlm": float(self.loss_weights.get("eta", 0.0)),
            "orth": float(self.loss_weights.get("orth", 0.0)),
        }
        for name, value in aux_losses.items():
            weight = weight_map.get(name, 0.0)
            if weight == 0.0:
                continue
            total = total + value * weight
            self.log(f"{prefix}/loss_{name}", value, on_epoch=True, prog_bar=False)

        self.log(f"{prefix}/loss", total, on_epoch=True, prog_bar=True)
        return total

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits, aux_losses = self.forward(batch)
        primary_loss = self._primary_loss(logits, batch)
        total_loss = self._combine_losses(primary_loss, aux_losses, "train")
        return total_loss

    def _shared_eval(self, batch: dict[str, torch.Tensor], prefix: str) -> None:
        logits, aux_losses = self.forward(batch)
        primary_loss = self._primary_loss(logits, batch)
        self._combine_losses(primary_loss, aux_losses, prefix)

        last_logits = logits[:, -1]
        target = batch["target"]
        history = build_history_multihot(batch["items"], batch["item_mask"], target.shape[-1])

        for k in self.hparams.k_values:
            recall = recall_at_k(last_logits, target, k).mean()
            ndcg = ndcg_at_k(last_logits, target, k).mean()
            repeat_target = target * history
            explore_target = target * (1.0 - history)
            repeat_recall = recall_at_k(last_logits, repeat_target, k).mean()
            explore_recall = recall_at_k(last_logits, explore_target, k).mean()
            self.log(f"{prefix}/recall@{k}", recall, prog_bar=False, on_epoch=True)
            self.log(f"{prefix}/ndcg@{k}", ndcg, prog_bar=False, on_epoch=True)
            self.log(f"{prefix}/repeat_recall@{k}", repeat_recall, prog_bar=False, on_epoch=True)
            self.log(f"{prefix}/explore_recall@{k}", explore_recall, prog_bar=False, on_epoch=True)

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
