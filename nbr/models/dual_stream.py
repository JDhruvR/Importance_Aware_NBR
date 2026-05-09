import torch
import torch.nn as nn
import torch.nn.functional as F

from nbr.models.embeddings import ItemEmbedding
from nbr.models.encoder import IntraBasketEncoder
from nbr.models.gpt import CausalBasketGPT
from nbr.models.importance import ImportanceHead
from nbr.models.gated_fusion import DualStreamFusion


class DualStreamNBR(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_gpt_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_embedding = ItemEmbedding(vocab_size, dim)
        self.encoder = IntraBasketEncoder(dim, num_heads, num_encoder_layers, dropout)
        self.importance_head = ImportanceHead(dim)
        self.fusion = DualStreamFusion(dim)
        self.gpt = CausalBasketGPT(dim, num_heads, num_gpt_layers, dropout)
        
        # MLM Head for auxiliary loss (predicting masked items from context)
        self.mlm_head = nn.Linear(dim, vocab_size)

    def forward(
        self, 
        items: torch.Tensor, 
        item_mask: torch.Tensor, 
        basket_mask: torch.Tensor
    ) -> dict:
        """
        Args:
            items: (B, T, S)
            item_mask: (B, T, S)
            basket_mask: (B, T)
        """
        B, T, S = items.shape
        D = self.item_embedding.embedding.embedding_dim

        # 1. Embed items: (B, T, S) -> (B, T, S, D)
        item_embs = self.item_embedding(items)

        # Flatten B and T for intra-basket processing
        flat_item_embs = item_embs.view(B * T, S, D)
        flat_item_mask = item_mask.view(B * T, S)

        # 2. Encode baskets (BERT-style)
        cls_repr, encoded_items = self.encoder(flat_item_embs, flat_item_mask)

        # 3. Compute Importance
        importance_scores = self.importance_head(encoded_items) # (B*T, S)

        # 4. Gate and Fuse
        fused_repr = self.fusion(cls_repr, encoded_items, importance_scores, flat_item_mask)
        
        # Also capture the gate values 'g' for your analysis (we have to recompute or extract)
        # To strictly analyze 'g', we recreate it here to log it easily
        with torch.no_grad():
            masked_imp = importance_scores * flat_item_mask.float()
            imp_sum = masked_imp.sum(dim=1, keepdim=True) + 1e-9
            norm_imp = masked_imp / imp_sum
            b_core = (norm_imp.unsqueeze(-1) * encoded_items).sum(dim=1)
            gate_g = torch.sigmoid(self.fusion.W_g(torch.cat([cls_repr, b_core], dim=-1)))

        # Reshape back to sequence
        basket_seq = fused_repr.view(B, T, D)

        # 5. Inter-basket sequence modeling (GPT-style)
        next_basket_repr = self.gpt(basket_seq, basket_mask) # (B, T, D)

        # 6. Predict next items (Dot product with item embeddings)
        # (B, T, D) @ (V, D).T -> (B, T, V)
        logits = F.linear(next_basket_repr, self.item_embedding.embedding.weight)

        # 7. MLM predictions for auxiliary loss
        mlm_logits = self.mlm_head(encoded_items) # (B*T, S, V)

        return {
            "logits": logits,
            "importance": importance_scores.view(B, T, S),
            "mlm_logits": mlm_logits.view(B, T, S, -1),
            "gate_values": gate_g.view(B, T, -1).mean(dim=-1) # Mean gate value per basket
        }