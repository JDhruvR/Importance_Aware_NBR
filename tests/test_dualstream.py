import torch
from nbr.models.gated_fusion import DualStreamFusion

class TestDualStreamFusion:
    def test_output_shape(self) -> None:
        """Fusion outputs (B*T, D)."""
        dim = 128
        fusion = DualStreamFusion(dim)
        cls_repr = torch.randn(10, dim)               # B*T=10, D=128
        item_reprs = torch.randn(10, 8, dim)          # S=8
        importance = torch.rand(10, 8)
        item_mask = torch.ones(10, 8, dtype=torch.bool)
        
        out = fusion(cls_repr, item_reprs, importance, item_mask)
        assert out.shape == (10, dim)

    def test_gate_gradient_flow(self) -> None:
        """Gradients flow back to the learnable gate W_g."""
        dim = 64
        fusion = DualStreamFusion(dim)
        cls_repr = torch.randn(4, dim, requires_grad=True)
        item_reprs = torch.randn(4, 5, dim, requires_grad=True)
        importance = torch.rand(4, 5, requires_grad=True)
        item_mask = torch.ones(4, 5, dtype=torch.bool)

        out = fusion(cls_repr, item_reprs, importance, item_mask)
        loss = out.sum()
        loss.backward()

        assert fusion.W_g.weight.grad is not None
        assert cls_repr.grad is not None
        assert item_reprs.grad is not None

    def test_masking_ignores_padding(self) -> None:
        """Padding items shouldn't affect the fused representation."""
        dim = 32
        fusion = DualStreamFusion(dim)
        cls_repr = torch.randn(2, dim)
        item_reprs = torch.randn(2, 3, dim)
        importance = torch.rand(2, 3)
        
        # Mask out the last item entirely
        item_mask1 = torch.tensor([[True, True, True], [True, True, True]])
        item_mask2 = torch.tensor([[True, True, False], [True, True, False]])
        
        out1 = fusion(cls_repr, item_reprs, importance, item_mask1)
        out2 = fusion(cls_repr, item_reprs, importance, item_mask2)
        
        assert not torch.allclose(out1, out2)
