"""Item embedding module: lookup table + Word2Vec pre-training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from gensim.models.word2vec import KeyedVectors


class ItemEmbedding(nn.Module):
    """Learnable item embedding lookup table.

    Supports initialization from scratch or from pre-trained Word2Vec weights.
    """

    def __init__(self, vocab_size: int, dim: int, padding_idx: int = 0) -> None:
        """
        Args:
            vocab_size: total number of unique items (including padding token).
            dim: embedding dimension D.
            padding_idx: index used for padding (gradients not updated).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings.

        Args:
            x: (...) int64 — item IDs (any shape).

        Returns:
            (..., D) float32 — item embeddings.
        """
        return self.embedding(x)  # (*, D)

    @classmethod
    def from_word2vec(
        cls,
        path: str | Path,
        vocab_size: int,
        dim: int,
        padding_idx: int = 0,
    ) -> ItemEmbedding:
        """Load embedding weights from a gensim KeyedVectors file.

        Args:
            path: path to saved KeyedVectors (.kv or .bin).
            vocab_size: total vocab size (including padding token).
            dim: embedding dimension.
            padding_idx: index to zero out.

        Returns:
            ItemEmbedding with weights initialized from Word2Vec.
        """
        kv = KeyedVectors.load(str(path))
        module = cls(vocab_size, dim, padding_idx=padding_idx)

        # Copy weights for known vocabulary items
        with torch.no_grad():
            for i in range(1, min(vocab_size, len(kv.index_to_key) + 1)):
                token = kv.index_to_key[i - 1]
                module.embedding.weight[i] = torch.tensor(kv[token], dtype=torch.float32)

            # Zero padding
            if padding_idx is not None:
                module.embedding.weight[padding_idx].fill_(0.0)

        return module


class Word2VecTrainer:
    """Train Word2Vec embeddings on basket sequences.

    Treats each basket as a sentence (list of item IDs as "words").
    """

    @staticmethod
    def train(
        basket_sequences: list[list[int]],
        dim: int = 128,
        window: int = 5,
        epochs: int = 10,
        min_count: int = 1,
        workers: int = 4,
    ) -> KeyedVectors:
        """Train Word2Vec model on basket sequences.

        Args:
            basket_sequences: list of baskets, each basket is a list of item IDs.
            dim: embedding dimension.
            window: context window size.
            epochs: number of training epochs.
            min_count: minimum item frequency to include.
            workers: number of training threads.

        Returns:
            KeyedVectors — trained word embeddings, ready to save or load.
        """
        # Convert int IDs to strings for gensim (it expects string tokens)
        sentences = [[str(item_id) for item_id in basket] for basket in basket_sequences]

        model = Word2Vec(
            sentences=sentences,
            vector_size=dim,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=1,  # skip-gram
        )

        return model.wv
