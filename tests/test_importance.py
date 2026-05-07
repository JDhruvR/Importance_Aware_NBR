"""Tests for ImportanceHead and importance_init_loss."""

from __future__ import annotations

import torch

from nbr.models.importance import ImportanceHead, importance_init_loss


class TestImportanceHead:
    """Shape and value-range tests for ImportanceHead."""

    def test_output_shape(self) -> None:
        """Output shape is (B*T, S) for input (B*T, S, D)."""
        head = ImportanceHead(dim=128)
        x = torch.randn(10, 8, 128)  # (B*T=10, S=8, D=128)
        out = head(x)
        assert out.shape == (10, 8)

    def test_output_range(self) -> None:
        """All outputs are in [0, 1] due to sigmoid."""
        head = ImportanceHead(dim=64)
        x = torch.randn(5, 12, 64)
        out = head(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_item_basket(self) -> None:
        """Works for baskets with a single item (S=1)."""
        head = ImportanceHead(dim=32)
        x = torch.randn(4, 1, 32)
        out = head(x)
        assert out.shape == (4, 1)

    def test_gradient_flows(self) -> None:
        """Gradients flow from loss back through the head."""
        head = ImportanceHead(dim=64)
        x = torch.randn(3, 5, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_different_dims(self) -> None:
        """Works for various even dimensions."""
        for dim in [16, 32, 64, 128, 256]:
            head = ImportanceHead(dim=dim)
            x = torch.randn(2, 4, dim)
            out = head(x)
            assert out.shape == (2, 4)

    def test_dim_too_small_raises(self) -> None:
        """dim < 2 raises ValueError."""
        try:
            ImportanceHead(dim=1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestImportanceInitLoss:
    """Tests for the importance initialization MSE loss."""

    def test_zero_loss_on_perfect_prediction(self) -> None:
        """Loss is zero when predicted matches target exactly."""
        target = torch.tensor([[0.5, 0.8, 0.3]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        loss = importance_init_loss(target, target, mask)
        assert loss.item() < 1e-7

    def test_loss_is_positive_on_mismatch(self) -> None:
        """Loss is positive when predicted differs from target."""
        predicted = torch.tensor([[0.5, 0.5, 0.5]])
        target = torch.tensor([[0.1, 0.9, 0.3]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        loss = importance_init_loss(predicted, target, mask)
        assert loss.item() > 0

    def test_padding_ignored(self) -> None:
        """Padding positions do not contribute to loss."""
        predicted = torch.tensor([[0.5, 0.5, 999.0]])  # padded position has wild value
        target = torch.tensor([[0.5, 0.5, 0.0]])
        mask = torch.tensor([[True, True, False]])
        loss = importance_init_loss(predicted, target, mask)
        assert loss.item() < 1e-7  # only real positions match

    def test_loss_decreases_with_optimization(self) -> None:
        """Loss decreases when we optimize the head toward targets."""
        torch.manual_seed(42)
        head = ImportanceHead(dim=64)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

        x = torch.randn(4, 6, 64)
        target = torch.rand(4, 6)  # random targets in [0, 1]
        mask = torch.ones(4, 6, dtype=torch.bool)

        initial_loss = importance_init_loss(head(x), target, mask).item()

        for _ in range(50):
            optimizer.zero_grad()
            predicted = head(x)
            loss = importance_init_loss(predicted, target, mask)
            loss.backward()
            optimizer.step()

        final_loss = importance_init_loss(head(x), target, mask).item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.6f} → {final_loss:.6f}"
        )

    def test_scalar_output(self) -> None:
        """Loss is a scalar tensor."""
        predicted = torch.rand(3, 5)
        target = torch.rand(3, 5)
        mask = torch.ones(3, 5, dtype=torch.bool)
        loss = importance_init_loss(predicted, target, mask)
        assert loss.dim() == 0

    def test_all_masked_no_crash(self) -> None:
        """No crash when all items are masked (edge case)."""
        predicted = torch.rand(2, 4)
        target = torch.rand(2, 4)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        loss = importance_init_loss(predicted, target, mask)
        assert loss.item() == 0.0
