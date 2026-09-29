"""Unit tests for angular magnitude loss."""

import pytest
import torch
import torch.nn.functional as F

from silverspoon_kd.losses.angular import angular_magnitude_loss
from silverspoon_kd.losses.registry import get_loss_function


class TestAngularMagnitudeLoss:
    """Tests for angular_magnitude loss function."""

    def test_factory(self):
        """get_loss_function('angular_magnitude') returns a callable."""
        loss_fn = get_loss_function("angular_magnitude")
        assert callable(loss_fn)

    def test_forward_shape(self):
        """Scalar output."""
        loss_fn = angular_magnitude_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = angular_magnitude_loss()
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss approx 0."""
        loss_fn = angular_magnitude_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_alpha_zero_magnitude_only(self):
        """alpha=0 -> loss == magnitude component only."""
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss_fn = angular_magnitude_loss(alpha=0.0, beta=1.0)
        loss = loss_fn(student, teacher)

        # Manual magnitude-only computation
        s_norm = student.norm(dim=-1)
        t_norm = teacher.norm(dim=-1)
        expected = F.mse_loss(s_norm, t_norm)
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_beta_zero_angular_only(self):
        """beta=0 -> loss == angular component only."""
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss_fn = angular_magnitude_loss(alpha=1.0, beta=0.0)
        loss = loss_fn(student, teacher)

        # Manual angular-only computation
        expected = 1 - F.cosine_similarity(student, teacher, dim=-1).mean()
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_custom_dim(self):
        """dim=1 works correctly."""
        loss_fn = angular_magnitude_loss(dim=1)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_numerical_correctness(self):
        """Manual alpha*(1-cosine)+beta*mse(norms) matches function output."""
        alpha, beta = 0.7, 0.3
        loss_fn = angular_magnitude_loss(alpha=alpha, beta=beta)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)

        angular = 1 - F.cosine_similarity(student, teacher, dim=-1).mean()
        s_norm = student.norm(dim=-1)
        t_norm = teacher.norm(dim=-1)
        magnitude = F.mse_loss(s_norm, t_norm)
        expected = alpha * angular + beta * magnitude

        actual = loss_fn(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-5)
