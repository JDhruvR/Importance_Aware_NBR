"""BERT+GPT Next Basket Recommendation model.

Intra-basket:  pre-trained BasketBERT (fine-tuned end-to-end) → CLS token
Inter-basket:  CausalBasketGPT (RoPE attention) → predicted next-basket embedding
Loss:          softmax(pred_logits) vs normalised multi-hot target → MSE
Inference:     pred_logits.topk(K) → top-K recommended items
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nbr.models.encoder import IntraBasketEncoder
from nbr.models.gpt import CausalBasketGPT


def _load_bert_bundle(path: str | Path, device: torch.device) -> dict:
    """Load a saved BasketBERT bundle dict onto the target device."""
    bundle = torch.load(str(path), map_location=device)
    return bundle


class BertGptNBR(nn.Module):
    """BERT-style intra-basket encoder (fine-tuned) + GPT inter-basket model.

    Architecture
    ------------
    1. Item embedding table (from BERT bundle, fine-tunable).
    2. IntraBasketEncoder with CLS token (from BERT bundle, fine-tunable).
    3. CausalBasketGPT operating on CLS sequences.
    4. Dot-product head: predicted_emb @ item_emb.weight.T → logits (B, T, V).

    Loss
    ----
    softmax(logits) vs normalised multi-hot targets → MSE, averaged over
    valid (non-padding) basket positions.
    """

    def __init__(
        self,
        bert_bundle_path: str | Path,
        gpt_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        # kept for config compatibility with VanillaNBR, ignored here:
        vocab_size: int | None = None,
        dim: int | None = None,
        encoder_layers: int | None = None,
    ) -> None:
        """
        Args:
            bert_bundle_path: path to ``bert_encoder_bundle_*.pt``.
            gpt_layers: number of CausalBasketGPT blocks.
            num_heads: attention heads for the GPT.
            dropout: GPT dropout probability.
            vocab_size / dim / encoder_layers: ignored (inferred from bundle).
        """
        super().__init__()
        bundle = _load_bert_bundle(bert_bundle_path, device=torch.device("cpu"))

        num_items: int = int(bundle["num_items"])
        d: int = int(bundle["dim"])
        pad_token_id: int = int(bundle["pad_token_id"])
        item_id_offset: int = int(bundle["item_id_offset"])
        vocab_sz: int = num_items + item_id_offset

        self.vocab_size = vocab_sz
        self.dim = d
        self.item_id_offset = item_id_offset

        # ── Item embedding table (fine-tunable) ──────────────────────────────
        self.item_embedding = nn.Embedding(vocab_sz, d, padding_idx=pad_token_id)
        self.item_embedding.weight.data.copy_(
            bundle["state_dict"]["embedding.weight"]
        )

        # ── Intra-basket BERT encoder (fine-tunable) ──────────────────────────
        # We reconstruct the encoder from the bundle; number of layers / heads
        # are inferred from the saved state_dict keys.
        encoder_sd = bundle["state_dict"]["encoder"]
        # Count layers by the pattern "transformer.layers.N.*"
        layer_indices = {
            int(k.split(".")[2])
            for k in encoder_sd
            if k.startswith("transformer.layers.")
        }
        n_enc_layers = len(layer_indices) if layer_indices else 2

        # Primary heuristic: num_heads from first layer's in_proj weight shape
        # in_proj_weight shape = (3*d, d) so heads can't be inferred; use 4.
        # The encoder was trained with whatever num_heads the BERT config had,
        # but PyTorch TransformerEncoderLayer defaults to nhead and doesn't
        # store it in state_dict keys, so we default to 4.
        n_enc_heads = num_heads

        self.intra_encoder = IntraBasketEncoder(
            dim=d,
            num_heads=n_enc_heads,
            num_layers=n_enc_layers,
            dropout=dropout,
        )
        self.intra_encoder.load_state_dict(encoder_sd, strict=True)

        # ── Inter-basket CausalBasketGPT (trainable from scratch) ────────────
        self.gpt = CausalBasketGPT(
            dim=d,
            num_heads=num_heads,
            num_layers=gpt_layers,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        items: torch.Tensor,
        item_mask: torch.Tensor,
        basket_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            items:       (B, T, S) int64 — item IDs (BERT token IDs, so include
                         item_id_offset if items are raw 0-based IDs).
            item_mask:   (B, T, S) bool  — True for real items.
            basket_mask: (B, T) bool     — True for real basket positions.

        Returns:
            logits: (B, T, V) float32 — dot-product scores over full vocabulary.
        """
        b, t, s = items.shape

        # ── Embed items via fine-tunable embedding table ──────────────────────
        # items already contain BERT token IDs (raw item_id + item_id_offset)
        x = self.item_embedding(items)              # (B, T, S, D)

        # ── Intra-basket encoding: flatten B×T, encode, reshape ──────────────
        x_flat = x.view(b * t, s, self.dim)          # (B*T, S, D)
        mask_flat = item_mask.view(b * t, s)          # (B*T, S)

        cls_repr, _ = self.intra_encoder(x_flat, mask_flat)   # (B*T, D)
        basket_reprs = cls_repr.view(b, t, self.dim)           # (B, T, D)

        # ── Inter-basket causal GPT ───────────────────────────────────────────
        next_pred = self.gpt(basket_reprs, basket_mask)        # (B, T, D)

        # ── Dot-product head ──────────────────────────────────────────────────
        # item_embedding.weight: (V, D)
        logits = next_pred @ self.item_embedding.weight.T      # (B, T, V)
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        basket_mask_target: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax-MSE loss.

        For each valid position t:
            pred_probs  = softmax(logits[:, t, :])           (B, V)
            target_dist = targets[:, t, :]                   (B, V) already normalised
            loss_t      = MSE(pred_probs, target_dist)        scalar

        Final loss is the mean over all valid (non-padding) positions.

        Args:
            logits:             (B, T, V) — raw logit scores.
            targets:            (B, T, V) — normalised multi-hot target distributions.
            basket_mask_target: (B, T) bool — True for positions that have a target.

        Returns:
            Scalar MSE loss.
        """
        # Convert normalised multi-hot back to pure binary multi-hot (1.0 for targets, 0.0 otherwise)
        binary_targets = (targets > 0).float()
        
        # BCEWithLogits computes binary cross entropy independently per item
        # pred_probs = F.softmax(...) is no longer needed
        bce_loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction="none")  # (B, T, V)
    
        #     # Average over V axis -> (B, T)
        # per_pos_loss = bce_loss.mean(dim=-1)
        # Sum over V axis to retain gradient magnitude -> (B, T)
        per_pos_loss = bce_loss.sum(dim=-1)
        
        # Mask padding positions
        mask = basket_mask_target.float()                      # (B, T)
        num_valid = mask.sum().clamp(min=1.0)
        loss = (per_pos_loss * mask).sum() / num_valid
        return loss

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_bert_bundle(self, path: str | Path, dataset_name: str) -> Path:
        """Save the (fine-tuned) BERT components as a bundle compatible with
        the original ``bert_encoder_bundle_*.pt`` format.

        Args:
            path: output directory.
            dataset_name: appended to filename.

        Returns:
            Path to the saved file.
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"bert_encoder_bundle_{dataset_name}_finetuned.pt"
        bundle = {
            "dataset": dataset_name,
            "num_items": self.vocab_size - self.item_id_offset,
            "dim": self.dim,
            "pad_token_id": self.item_embedding.padding_idx,
            "mask_token_id": 1,          # convention from original BERT training
            "item_id_offset": self.item_id_offset,
            "state_dict": {
                "embedding.weight": self.item_embedding.weight.detach().cpu(),
                "encoder": {
                    k: v.detach().cpu()
                    for k, v in self.intra_encoder.state_dict().items()
                },
            },
        }
        torch.save(bundle, out_path)
        return out_path
