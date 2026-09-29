"""Unit tests for MSE-based loss functions."""

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.losses.mse import mse_loss, normalized_mse_loss
from silverspoon_kd.losses.registry import get_loss_function


class TestMSELoss:
    """Tests for mse_loss factory."""

    def test_factory(self):
        """mse_loss() returns a callable."""
        loss_fn = mse_loss()
        assert callable(loss_fn)

    def test_forward_shape_2d(self):
        """Scalar output for 2D input."""
        loss_fn = mse_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_forward_shape_3d(self):
        """Scalar output for 3D input."""
        loss_fn = mse_loss()
        student = torch.randn(2, 16, 64)
        teacher = torch.randn(2, 16, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = mse_loss()
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss exactly 0."""
        loss_fn = mse_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_numerical_correctness(self):
        """Matches nn.MSELoss reference."""
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        expected = nn.MSELoss()(student, teacher)
        actual = mse_loss()(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-6)

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_custom_reduction(self, reduction):
        """The reduction kwarg is forwarded to F.mse_loss."""
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        actual = mse_loss(reduction=reduction)(student, teacher)
        expected = nn.MSELoss(reduction=reduction)(student, teacher)
        assert torch.allclose(actual, expected, atol=1e-6)


class TestNormalizedMSELoss:
    """Tests for normalized_mse loss function."""

    def test_factory(self):
        """get_loss_function('normalized_mse') returns a callable."""
        loss_fn = get_loss_function("normalized_mse")
        assert callable(loss_fn)

    def test_forward_shape_2d(self):
        """Scalar output for 2D input."""
        loss_fn = normalized_mse_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_forward_shape_3d(self):
        """Scalar output for 3D input."""
        loss_fn = normalized_mse_loss()
        student = torch.randn(2, 16, 64)
        teacher = torch.randn(2, 16, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = normalized_mse_loss()
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss approx 0."""
        loss_fn = normalized_mse_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_scale_invariance(self):
        """loss(s, t) approx loss(s*100, t*100) due to z-score normalization."""
        loss_fn = normalized_mse_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss_original = loss_fn(student, teacher)
        loss_scaled = loss_fn(student * 100, teacher * 100)
        assert loss_original.item() == pytest.approx(loss_scaled.item(), rel=1e-4)

    def test_custom_eps(self):
        """Passing eps=1e-3 doesn't break computation."""
        loss_fn = normalized_mse_loss(eps=1e-3)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss)

    def test_numerical_correctness(self):
        """Manual z-score + MSE matches function output."""
        eps = 1e-6
        loss_fn = normalized_mse_loss(eps=eps)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)

        # Manual computation
        s_norm = (student - student.mean(dim=-1, keepdim=True)) / (
            student.std(dim=-1, keepdim=True) + eps
        )
        t_norm = (teacher - teacher.mean(dim=-1, keepdim=True)) / (
            teacher.std(dim=-1, keepdim=True) + eps
        )
        expected = torch.nn.functional.mse_loss(s_norm, t_norm)

        actual = loss_fn(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-5)


class TestNormalizedMSEConstantFeatures:
    """Regression test: constant features (std=0) must produce finite loss."""

    def test_constant_student_features(self):
        """When student has constant features (std=0), loss must be finite."""
        loss_fn = normalized_mse_loss()
        student = torch.ones(4, 8, 64)  # all identical → std=0
        teacher = torch.randn(4, 8, 64)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"

    def test_constant_both(self):
        """When both are constant, loss should be finite (likely 0)."""
        loss_fn = normalized_mse_loss()
        student = torch.ones(4, 8, 64) * 3.0
        teacher = torch.ones(4, 8, 64) * 7.0
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss)

    def test_constant_student_gradient_flow(self):
        """Gradients through constant-feature normalization must be finite."""
        loss_fn = normalized_mse_loss()
        student = torch.ones(4, 8, 64, requires_grad=True)
        teacher = torch.randn(4, 8, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_constant_features_across_dtypes(self, dtype):
        """Constant features with reduced-precision dtypes must stay finite."""
        loss_fn = normalized_mse_loss()
        student = torch.ones(4, 8, 64, dtype=dtype)
        teacher = torch.randn(4, 8, 64, dtype=dtype)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"Non-finite loss with {dtype}"
