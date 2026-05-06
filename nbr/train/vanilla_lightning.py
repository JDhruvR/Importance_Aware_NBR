"""Lightning module for the vanilla and BERT+GPT NBR models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule

from nbr.metrics.ranking import build_history_multihot, ndcg_at_k, recall_at_k
from nbr.models.bert_gpt.model import BertGptNBR
from nbr.models.vanilla import VanillaNBR


class VanillaLitModule(LightningModule):
    """LightningModule wrapper for vanilla or BERT+GPT NBR models."""

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
        # BERT+GPT specific
        bert_bundle_path: str | None = None,
        output_dir: str | None = None,
        dataset_name: str = "tafeng",
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
            if not bert_bundle_path:
                raise ValueError("bert_bundle_path is required for model_type='bert_gpt'")
            self.model = BertGptNBR(
                bert_bundle_path=bert_bundle_path,
                gpt_layers=gpt_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["items"], batch["item_mask"], batch["basket_mask"])

    def _compute_loss(
        self, logits: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(self.model, BertGptNBR):
            # New causal paradigm: targets (B, T, V), basket_mask_target (B, T)
            return self.model.loss(
                logits,
                batch["targets"],
                batch["basket_mask_target"],
            )
        else:
            # VanillaNBR legacy path: single (B, V) target
            import torch.nn.functional as F
            target_expanded = batch["target"].unsqueeze(1).expand_as(logits)
            per_element_loss = F.binary_cross_entropy_with_logits(
                logits, target_expanded, reduction="none"
            )
            mask = batch["basket_mask"].unsqueeze(-1).float()
            return (per_element_loss * mask).sum() / mask.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    # Training / Eval steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self._forward_logits(batch)
        loss = self._compute_loss(logits, batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _shared_eval(self, batch: dict[str, torch.Tensor], prefix: str) -> None:
        logits = self._forward_logits(batch)
        loss = self._compute_loss(logits, batch)
        self.log(f"{prefix}/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if isinstance(self.model, BertGptNBR):
            bmt = batch["basket_mask_target"]          # (B, T)
            last_idx = bmt.sum(dim=1) - 1              # (B,)  index of last valid pos
            last_idx = last_idx.clamp(min=0).long()

            # Ranking metrics use the last valid prediction position
            b_idx = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[b_idx, last_idx]      # (B, V)

            target = torch.stack(
                [batch["targets"][i, last_idx[i]] for i in range(len(last_idx))]
            )  # (B, V)  normalised; convert to binary for metrics
            target = (target > 0).float()
        else:
            last_logits = logits[:, -1]                # (B, V) Legacy fallback
            target = batch["target"]                   # (B, V) multi-hot

        history = build_history_multihot(batch["items"], batch["item_mask"], target.shape[-1])

        for k in self.hparams.k_values:
            recall = recall_at_k(last_logits, target, k).mean()
            ndcg = ndcg_at_k(last_logits, target, k).mean()
            repeat_target = target * history
            explore_target = target * (1.0 - history)
            repeat_recall = recall_at_k(last_logits, repeat_target, k).mean()
            explore_recall = recall_at_k(last_logits, explore_target, k).mean()
            self.log(f"{prefix}/recall@{k}", recall, on_epoch=True, sync_dist=True)
            self.log(f"{prefix}/ndcg@{k}", ndcg, on_epoch=True, sync_dist=True)
            self.log(f"{prefix}/repeat_recall@{k}", repeat_recall, on_epoch=True, sync_dist=True)
            self.log(f"{prefix}/explore_recall@{k}", explore_recall, on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_eval(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_eval(batch, "test")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
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
