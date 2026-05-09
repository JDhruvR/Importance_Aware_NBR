import torch
import torch.nn as nn
import torch.nn.functional as F
from nbr.models.projection import IntentProjection

class TwoStageDecoder(nn.Module):
    def __init__(self, dim: int, intent_dim: int, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.projection = IntentProjection(dim, intent_dim)
        self.W_cond = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self, 
        next_pred: torch.Tensor, 
        item_embeddings: torch.Tensor, 
        core_mask: torch.Tensor | None = None
    ) -> dict:
        """
        Args:
            next_pred: (B, D) from GPT output representing the predicted next basket.
            item_embeddings: (V, D) full vocabulary embedding matrix.
            core_mask: (B, V) bool tensor marking predicted intent items (inference only).
        """
        # 1. Decompose
        intent_repr, fill_repr = self.projection(next_pred)

        # 2. Intent Logits
        # (B, D) @ (D, V) -> (B, V)
        intent_logits = torch.matmul(intent_repr, item_embeddings.T)

        # 3. Soft Intent Context (Training) or Hard Intent (Inference)
        if core_mask is None:
            # Training: Soft weighted combination of vocabulary
            attn_weights = F.softmax(intent_logits / self.temperature, dim=-1) # (B, V)
            intent_context = torch.matmul(attn_weights, item_embeddings) # (B, V) @ (V, D) -> (B, D)
        else:
            # Inference: Mean pool over predicted core items
            # core_mask is (B, V) boolean.
            mask_floats = core_mask.float()
            counts = mask_floats.sum(dim=-1, keepdim=True).clamp(min=1.0)
            intent_context = torch.matmul(mask_floats, item_embeddings) / counts # (B, D)

        # 4. Fill Query Conditioning
        fill_query = self.layer_norm(fill_repr + self.W_cond(intent_context)) # (B, D)

        # 5. Fill Logits
        fill_logits = torch.matmul(fill_query, item_embeddings.T) # (B, V)

        return {
            "intent_logits": intent_logits,
            "fill_logits":   fill_logits,
            "soft_intent":   intent_context,
            "intent_repr":   intent_repr,   # (B, T, D) — needed for L_orth
            "fill_repr":     fill_repr,     # (B, T, D) — needed for L_orth
        }

def residual_decode(
    repr_vec:        torch.Tensor,
    item_embeddings: torch.Tensor,
    decoder:         "TwoStageDecoder",
    k1:              int,
    k2:              int,
    excluded:        set[int],
) -> list[int]:
    """Two-stage residual decode (Section VI, Steps 4-6).

    Args:
        repr_vec:        (D,) predicted next-basket vector h_{T+1}.
        item_embeddings: (V, D) full vocabulary embedding matrix.
        decoder:         TwoStageDecoder — needed for W_cond + LayerNorm.
        k1:              number of core items (Stage 1, intent subspace).
        k2:              number of fill items (Stage 2, conditioned on core).
        excluded:        item IDs to exclude from selection.

    Returns:
        List of k1 + k2 item IDs: core items first, then fill items.
    """
    projection = decoder.projection
    vocab_size = item_embeddings.size(0)
    proj_op    = projection.P @ projection.P.T  # (D, D)

    valid_mask = torch.ones(vocab_size, dtype=torch.bool, device=repr_vec.device)
    if excluded:
        valid_mask[list(excluded)] = False

    intent_repr = torch.matmul(proj_op, repr_vec)        # h^intent
    fill_repr   = repr_vec - intent_repr                  # h^fill

    # Stage 1 — core items in intent subspace (Eqs. 17-18)
    core_ids = []
    r_intent  = intent_repr.clone()
    for _ in range(k1):
        scores = torch.matmul(item_embeddings, r_intent)
        scores[~valid_mask] = -float("inf")
        best = scores.argmax().item()
        core_ids.append(best)
        valid_mask[best] = False
        r_intent = r_intent - torch.matmul(proj_op, item_embeddings[best])

    # Stage 2 — fill items conditioned on discovered core (§VI Steps 5-6)
    # Use the decoder's learned W_cond and LayerNorm to match training behavior
    core_embs   = item_embeddings[torch.tensor(core_ids, device=repr_vec.device)]
    mean_core   = core_embs.mean(dim=0)                   # hard c~ (Eq. 32)
    fill_query  = decoder.layer_norm(
        fill_repr + decoder.W_cond(mean_core)
    )
    r_fill      = fill_query.clone()
    fill_proj   = torch.eye(repr_vec.size(-1), device=repr_vec.device) - proj_op

    fill_ids = []
    for _ in range(k2):
        scores = torch.matmul(item_embeddings, r_fill)
        scores[~valid_mask] = -float("inf")
        best = scores.argmax().item()
        fill_ids.append(best)
        valid_mask[best] = False
        r_fill = r_fill - torch.matmul(fill_proj, item_embeddings[best])

    return core_ids + fill_ids