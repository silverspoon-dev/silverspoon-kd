"""Unit tests for cosine similarity loss."""

import pytest
import torch
import torch.nn.functional as F

from silverspoon_kd.losses.cosine import cosine_loss


class TestCosineLoss:
    """Tests for cosine_loss factory."""

    def test_factory(self):
        """cosine_loss() returns a callable."""
        loss_fn = cosine_loss()
        assert callable(loss_fn)

    def test_forward_shape(self):
        """Scalar output."""
        loss_fn = cosine_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        result = loss_fn(student, teacher)
        assert result.shape == ()

    def test_custom_dim(self):
        """Custom dim parameter works."""
        loss_fn = cosine_loss(dim=1)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        result = loss_fn(student, teacher)
        assert result.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = cosine_loss()
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss approx 0."""
        loss_fn = cosine_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_parallel_vectors_zero_loss(self):
        """Parallel vectors (scaled) -> loss approx 0."""
        loss_fn = cosine_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x * 3.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors -> loss approx 1."""
        loss_fn = cosine_loss()
        # Construct orthogonal pair in 2D for a single sample
        student = torch.tensor([[1.0, 0.0]])
        teacher = torch.tensor([[0.0, 1.0]])
        loss = loss_fn(student, teacher)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_numerical_correctness(self):
        """Manual 1-cosine_similarity matches function output."""
        loss_fn = cosine_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        expected = 1 - F.cosine_similarity(student, teacher, dim=-1).mean()
        actual = loss_fn(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-6)
