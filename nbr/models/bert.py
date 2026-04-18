"""Plain basket-BERT model for MLM warmup."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.word2vec import KeyedVectors

from nbr.models.encoder import IntraBasketEncoder


class BasketBERT(nn.Module):
    """BERT-style intra-basket model trained with MLM."""

    def __init__(
        self,
        num_items: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        item_id_offset: int = 2,
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if item_id_offset < 2:
            raise ValueError("item_id_offset must be >= 2")

        self.num_items = num_items
        self.dim = dim
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.item_id_offset = item_id_offset
        self.vocab_size = self.num_items + self.item_id_offset

        self.embedding = nn.Embedding(self.vocab_size, dim, padding_idx=pad_token_id)
        self.encoder = IntraBasketEncoder(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0.0)

    def init_item_embeddings_from_word2vec(self, path: str | Path) -> tuple[int, int]:
        """Initialize item token rows from Word2Vec vectors.

        Returns:
            (loaded_count, missing_count)
        """
        kv = KeyedVectors.load(str(path))
        if kv.vector_size != self.dim:
            raise ValueError(f"Word2Vec dim mismatch: got {kv.vector_size}, expected {self.dim}")

        loaded_count = 0
        missing_count = 0
        with torch.no_grad():
            for item_id in range(self.num_items):
                token_id = item_id + self.item_id_offset
                key = str(item_id)
                if key in kv:
                    self.embedding.weight[token_id] = torch.tensor(kv[key], dtype=torch.float32)
                    loaded_count += 1
                else:
                    missing_count += 1
            self.embedding.weight[self.pad_token_id].fill_(0.0)
        return loaded_count, missing_count

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode basket tokens and predict masked item IDs.

        Args:
            input_ids: (B, S) token IDs.
            attention_mask: (B, S) bool, True for non-padding tokens.

        Returns:
            Dict containing cls_repr, item_repr, and mlm_logits.
        """
        token_emb = self.embedding(input_ids)  # (B, S, D)
        cls_repr, item_repr = self.encoder(token_emb, attention_mask)
        mlm_logits = item_repr @ self.embedding.weight.T + self.output_bias
        return {
            "cls_repr": cls_repr,
            "item_repr": item_repr,
            "mlm_logits": mlm_logits,
        }

    @staticmethod
    def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on masked positions only (ignore_index=-100)."""
        vocab_size = logits.shape[-1]
        return F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
