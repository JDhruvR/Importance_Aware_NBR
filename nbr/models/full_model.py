"""
full_model.py — IntentAwareNBR
==============================
Intent-aware next-basket recommendation model as described in the paper.

forward() returns every tensor that total_loss() needs, named to match
the loss function's argument list exactly so the training loop can call:

    loss, log = total_loss(
        intent_logits  = out["intent_logits"],
        fill_logits    = out["fill_logits"],
        targets        = targets,
        alpha_idf      = alpha_idf,
        tau_alpha      = cfg.model.tau_alpha,
        intent_repr    = out["intent_repr"],   # (B, T, D) — for L_orth
        fill_repr      = out["fill_repr"],     # (B, T, D) — for L_orth
        mlm_logits     = out["mlm_logits"],    # (B, T, S, V)
        mlm_targets    = mlm_targets,          # (B, T, S) from data pipeline
        weights        = cfg.loss.weights,
        mask_token_id  = cfg.data.mask_token_id,
    )

Pipeline (paper section references):
    §V-A  Item Embedding Initialization      — ItemEmbedding
    §V-B  Intra-Basket Encoder               — IntraBasketEncoder  (BERT-style, no pos enc)
    §V-B  Importance Head                    — ImportanceHead      (2-layer MLP, sigmoid)
    §V-D  Dual-Stream Gated Fusion           — DualStreamFusion    (Eqs. 8–11)
    §V-E  Inter-Basket Causal GPT            — CausalBasketGPT     (RoPE, causal mask)
    §V-F  Orthogonal Intent-Fill Decomp.     — TwoStageDecoder     (Eqs. 13–14)
    §V-G  Conditioned Two-Stage Decoding     — TwoStageDecoder     (Eqs. 15–21)
    §V-B  MLM Auxiliary Head                 — nn.Linear → vocab   (Eq. 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nbr.models.embeddings import ItemEmbedding
from nbr.models.encoder import IntraBasketEncoder
from nbr.models.importance import ImportanceHead
from nbr.models.gated_fusion import DualStreamFusion
from nbr.models.gpt import CausalBasketGPT
from nbr.models.decoder import TwoStageDecoder


class IntentAwareNBR(nn.Module):
    """
    Full intent-aware NBR model.

    Args:
        vocab_size         : total number of items |V|
        dim                : model hidden dimension d
        intent_dim         : rank of the low-rank intent projection (d_k < d)
        num_heads          : attention heads for both encoder and GPT
        num_encoder_layers : number of BERT-style transformer layers (L1)
        num_gpt_layers     : number of causal GPT transformer layers (L2)
        dropout            : dropout probability throughout
        temperature        : softmax temperature τ for soft intent context (Eq. 16)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        intent_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_gpt_layers: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        # §V-A — shared item embedding matrix E ∈ R^{|V| × d}
        # Initialized via skip-gram in training Phase 0; fully trainable after.
        self.item_embedding = ItemEmbedding(vocab_size, dim)

        # §V-B — BERT-style intra-basket encoder.
        # No positional encoding (baskets are unordered sets).
        # No causal mask (every item attends to every other item bidirectionally).
        # Returns: cls_repr (B*T, D), encoded_items (B*T, S, D)
        self.encoder = IntraBasketEncoder(dim, num_heads, num_encoder_layers, dropout)

        # §V-B — Importance head: 2-layer MLP with sigmoid output.
        # Operates on contextualized h_i, not raw embeddings E[i].
        # Sigmoid (not softmax) → independent per-item scores in [0, 1].
        # Initialized in Phase 2 to reproduce alpha_IDF via MSE loss.
        self.importance_head = ImportanceHead(dim)

        # §V-D — Dual-stream gated basket representation (Eqs. 8–11).
        # Blends full-basket CLS summary (b_full) with importance-weighted
        # centroid (b_core) via a learned elementwise gate g.
        self.fusion = DualStreamFusion(dim)

        # §V-E — Causal GPT with RoPE (Eq. 12).
        # Models temporal sequence of basket representations.
        # RoPE applied to Q and K only; basket vectors are pure semantic summaries.
        self.gpt = CausalBasketGPT(dim, num_heads, num_gpt_layers, dropout)

        # §V-F & §V-G — Low-rank orthogonal projection + two-stage conditioned decoder.
        # Decomposes GPT output h_{T+1} into h^intent and h^fill (Eqs. 13–14).
        # Stage 1: core items via residual loop in intent subspace (Eqs. 15, 17–18).
        # Stage 2: fill items conditioned on soft intent context c~ (Eqs. 16, 19–21).
        self.decoder = TwoStageDecoder(dim, intent_dim, temperature)

        # §V-B — MLM auxiliary head (Eq. 3).
        # Projects encoded item representations to vocab logits for masked items.
        # Provides a direct training signal for the intra-basket encoder independent
        # of the downstream temporal prediction task.
        self.mlm_head = nn.Linear(dim, vocab_size)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        items: torch.Tensor,
        item_mask: torch.Tensor,
        basket_mask: torch.Tensor,
        mlm_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Full forward pass. Returns every tensor needed by total_loss().

        Args:
            items       : (B, T, S)  item ids per basket per timestep
            item_mask   : (B, T, S)  1 for real items, 0 for padding
            basket_mask : (B, T)     1 for real baskets, 0 for padding
            mlm_mask    : (B, T, S)  1 at positions that were replaced by
                                     [MASK] token, 0 elsewhere.
                                     When provided, mlm_targets in the
                                     returned dict is the original item id
                                     at masked positions and mask_token_id
                                     (0) elsewhere — ready for mlm_loss().
                                     Pass None during inference.

        Returns dict with keys:
            intent_logits  : (B, T, V)    raw intent scores  s^intent  (Eq. 15)
            fill_logits    : (B, T, V)    raw fill scores    s^fill    (Eq. 20)
            intent_repr    : (B, T, D)    h^intent component (Eq. 13)  for L_orth
            fill_repr      : (B, T, D)    h^fill   component (Eq. 14)  for L_orth
            soft_intent    : (B, T, D)    soft intent context c~        (Eq. 16)
            importance     : (B, T, S)    per-item importance weights α_i
            mlm_logits     : (B, T, S, V) item logits at all positions  (Eq. 3)
            next_basket_repr: (B, T, D)   GPT output — use [:, -1, :] at inference
        """
        B, T, S = items.shape
        D = self.item_embedding.embedding.embedding_dim

        # ------------------------------------------------------------------ #
        # Step 1 — Item embeddings                                  §V-A      #
        # ------------------------------------------------------------------ #
        # (B, T, S) → (B, T, S, D)
        item_embs = self.item_embedding(items)

        # Flatten batch and time so the intra-basket encoder processes each
        # basket independently as an unordered set. (B*T, S, D)
        flat_embs = item_embs.view(B * T, S, D)
        flat_mask = item_mask.view(B * T, S)

        # ------------------------------------------------------------------ #
        # Step 2 — BERT intra-basket encoding                       §V-B      #
        # ------------------------------------------------------------------ #
        # cls_repr      : (B*T, D)    — h_CLS = b_full  (Eq. 8)
        # encoded_items : (B*T, S, D) — h_i per item in basket context
        cls_repr, encoded_items = self.encoder(flat_embs, flat_mask)

        # ------------------------------------------------------------------ #
        # Step 3 — Importance head                                  §V-B      #
        # ------------------------------------------------------------------ #
        # α_i = σ(W2 GELU(W1 h_i + b1) + b2)  ∈ [0,1]           (Eq. 2)
        # Shape: (B*T, S)
        importance_scores = self.importance_head(encoded_items)

        # ------------------------------------------------------------------ #
        # Step 4 — Dual-stream gated fusion                         §V-D      #
        # ------------------------------------------------------------------ #
        # b_full = h_CLS                                            (Eq. 8)
        # b_core = Σ α_i h_i / Σ α_i   (masked centroid)          (Eq. 9)
        # g      = σ(W_g [b_full ; b_core])                        (Eq. 10)
        # b_t    = g ⊙ b_full + (1-g) ⊙ b_core                    (Eq. 11)
        # Shape: (B*T, D)
        fused_repr = self.fusion(cls_repr, encoded_items, importance_scores, flat_mask)

        # Reshape back to sequence for the GPT: (B, T, D)
        basket_seq = fused_repr.view(B, T, D)

        # ------------------------------------------------------------------ #
        # Step 5 — Causal inter-basket GPT                          §V-E      #
        # ------------------------------------------------------------------ #
        # Applies RoPE to Q and K; causal mask prevents attending to future
        # baskets. Loss is computed at every position t = 1, ..., T-1.
        # next_basket_repr[:, t, :] is the model's prediction for basket t+1
        # given baskets 1..t.
        # Shape: (B, T, D)
        next_basket_repr = self.gpt(basket_seq, basket_mask)

        # ------------------------------------------------------------------ #
        # Step 6 — Two-stage conditioned decoder                    §V-F,G    #
        # ------------------------------------------------------------------ #
        # The decoder:
        #   (a) Decomposes next_basket_repr into h^intent and h^fill (Eqs. 13-14)
        #   (b) Scores all vocab items against h^intent → intent_logits (Eq. 15)
        #   (c) Builds soft intent context c~ via temperature softmax (Eq. 16)
        #   (d) Shifts fill query by c~ via W_cond → fill query (Eq. 19)
        #   (e) Scores all vocab items against fill query → fill_logits (Eq. 20)
        #
        # IMPORTANT: next_basket_repr is passed here, NOT cls_repr.
        # cls_repr encodes past observed baskets; next_basket_repr is the GPT's
        # prediction of the *next* basket — the correct input to the decoder.
        #
        # vocab_embeddings is passed so the decoder can compute dot-product
        # scores against the shared item embedding matrix E (tied weights).
        vocab_embeddings = self.item_embedding.embedding.weight  # (V, D)

        decoder_out = self.decoder(next_basket_repr, vocab_embeddings)
        # decoder_out contains:
        #   "intent_logits" : (B, T, V)  s^intent_i = e_i · h^intent
        #   "fill_logits"   : (B, T, V)  s^fill_i   = e_i · h^fill|intent
        #   "intent_repr"   : (B, T, D)  h^intent = P P^T h_{T+1}
        #   "fill_repr"     : (B, T, D)  h^fill   = h_{T+1} - P P^T h_{T+1}
        #   "soft_intent"   : (B, T, D)  c~ = Σ softmax_τ(s^intent)_i · e_i

        # ------------------------------------------------------------------ #
        # Step 7 — MLM auxiliary logits                             §V-B      #
        # ------------------------------------------------------------------ #
        # Project all encoded item positions to vocab logits.
        # The loss masks out non-masked positions via ignore_index in mlm_loss().
        # Shape: (B*T, S, V) → (B, T, S, V)
        mlm_logits = self.mlm_head(encoded_items).view(B, T, S, -1)

        # ------------------------------------------------------------------ #
        # Assemble output dict                                                #
        # ------------------------------------------------------------------ #
        return {
            # ── loss inputs ─────────────────────────────────────────────── #
            # These map 1-to-1 onto total_loss() arguments.
            "intent_logits":   decoder_out["intent_logits"],   # (B, T, V)
            "fill_logits":     decoder_out["fill_logits"],     # (B, T, V)
            "intent_repr":     decoder_out["intent_repr"],     # (B, T, D) → L_orth
            "fill_repr":       decoder_out["fill_repr"],       # (B, T, D) → L_orth
            "mlm_logits":      mlm_logits,                     # (B, T, S, V)

            # ── diagnostics / inference ─────────────────────────────────── #
            "soft_intent":     decoder_out["soft_intent"],     # (B, T, D) — c~
            "importance":      importance_scores.view(B, T, S),# (B, T, S) — α_i
            # Use next_basket_repr[:, -1, :] as the decoding vector at inference.
            # Do NOT use cls_repr — that encodes observed baskets, not predictions.
            "next_basket_repr": next_basket_repr,              # (B, T, D)
        }

    # ------------------------------------------------------------------
    # Convenience: periodic Gram-Schmidt re-orthonormalization
    # ------------------------------------------------------------------

    def orthogonalize_projection(self) -> None:
        """
        §V-F: Enforce P^T P = I_dk via Gram-Schmidt re-orthonormalization.
        Call every N steps from the training loop (paper suggests every 100).
        This prevents the low-rank projection from drifting toward the identity
        even when L_orth keeps the gradient signal small.
        """
        self.decoder.projection.orthogonalize_()