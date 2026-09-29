"""Unit tests for smooth L1 loss."""

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.losses.smooth_l1 import smooth_l1_loss


class TestSmoothL1Loss:
    """Tests for smooth_l1_loss factory."""

    def test_factory(self):
        """smooth_l1_loss() returns a callable."""
        loss_fn = smooth_l1_loss()
        assert callable(loss_fn)

    def test_forward_shape(self):
        """Scalar output."""
        loss_fn = smooth_l1_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = smooth_l1_loss()
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss exactly 0."""
        loss_fn = smooth_l1_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_custom_beta(self):
        """Custom beta parameter works."""
        loss_fn = smooth_l1_loss(beta=2.0)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_numerical_correctness(self):
        """Matches nn.SmoothL1Loss reference."""
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        expected = nn.SmoothL1Loss()(student, teacher)
        actual = smooth_l1_loss()(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-6)

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_custom_reduction(self, reduction):
        """The reduction kwarg is forwarded to F.smooth_l1_loss."""
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        actual = smooth_l1_loss(reduction=reduction)(student, teacher)
        expected = nn.SmoothL1Loss(reduction=reduction)(student, teacher)
        assert torch.allclose(actual, expected, atol=1e-6)
