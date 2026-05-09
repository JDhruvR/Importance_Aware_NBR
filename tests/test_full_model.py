import torch
import pytest
from nbr.models.projection import IntentProjection
from nbr.models.decoder import TwoStageDecoder, residual_decode
from nbr.losses import total_loss

class TestPhase5:
    def test_intent_projection(self) -> None:
        """Projection splits vectors and maintains orthogonality."""
        dim, intent_dim = 128, 32
        proj = IntentProjection(dim, intent_dim)
        x = torch.randn(10, dim)
        
        # Test shapes
        intent_repr, fill_repr = proj(x)
        assert intent_repr.shape == (10, dim)
        assert fill_repr.shape == (10, dim)
        assert torch.allclose(intent_repr + fill_repr, x, atol=1e-5)
        
        # Test orthogonalize
        proj.P.data += 0.5 # Corrupt orthogonality
        proj.orthogonalize_()
        identity = torch.eye(intent_dim)
        assert torch.allclose(proj.P.T @ proj.P, identity, atol=1e-5)

    def test_two_stage_decoder(self) -> None:
        """Two stage decoder outputs correctly shaped logits."""
        dim, intent_dim, vocab_size = 64, 16, 1000
        decoder = TwoStageDecoder(dim, intent_dim)
        next_pred = torch.randn(8, dim)
        vocab_embeddings = torch.randn(vocab_size, dim)
        
        out = decoder(next_pred, vocab_embeddings)
        assert out["intent_logits"].shape == (8, vocab_size)
        assert out["fill_logits"].shape == (8, vocab_size)
        assert out["soft_intent"].shape == (8, dim)

    def test_residual_decode(self) -> None:
        """Residual loop does not predict duplicate/excluded items."""
        repr_vec = torch.randn(64)
        item_embeddings = torch.randn(100, 64)
        excluded = {5, 10, 15}
        
        recs = residual_decode(repr_vec, item_embeddings, None, k=10, excluded=excluded)
        assert len(recs) == 10
        assert len(set(recs)) == 10 # All unique (no duplicates)
        assert not any(x in excluded for x in recs)

    def test_losses_returns_scalar(self) -> None:
        """Total loss combines inputs correctly into a scalar."""
        B, V, S = 4, 100, 5
        intent_logits = torch.randn(B, V)
        fill_logits = torch.randn(B, V)
        targets = torch.rand(B, V) # Multi-hot BCE targets
        alpha_idf = torch.rand(V)
        orth_loss = torch.tensor(0.5)
        mlm_logits = torch.randn(B, S, V)
        mlm_targets = torch.randint(0, V, (B, S))
        
        loss, loss_dict = total_loss(
            intent_logits, fill_logits, targets, alpha_idf, 
            orth_loss, mlm_logits, mlm_targets, 
            weights={"intent": 1.0, "fill": 1.0, "orth": 0.1, "mlm": 0.5}
        )
        
        assert loss.dim() == 0
        assert "intent_loss" in loss_dict