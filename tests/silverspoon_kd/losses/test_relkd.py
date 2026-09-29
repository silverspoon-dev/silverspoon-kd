"""Unit tests for relation-based KD (RelKD) losses."""

import logging

import pytest
import torch

from silverspoon_kd.losses.relkd import (
    _mean_normalize_distances,
    _pairwise_distances,
    relkd_angle_loss,
    relkd_da_loss,
    relkd_distance_loss,
)


class TestPairwiseDistances:
    """Tests for _pairwise_distances helper."""

    def test_shape(self):
        """Output is [B, B]."""
        x = torch.randn(4, 64)
        dist = _pairwise_distances(x)
        assert dist.shape == (4, 4)

    def test_diagonal_is_zero(self):
        """Self-distance is zero."""
        x = torch.randn(4, 64)
        dist = _pairwise_distances(x)
        assert torch.allclose(dist.diag(), torch.zeros(4), atol=1e-6)

    def test_symmetric(self):
        """Distance matrix is symmetric."""
        x = torch.randn(4, 64)
        dist = _pairwise_distances(x)
        assert torch.allclose(dist, dist.T, atol=1e-6)

    def test_numerical_correctness(self):
        """Manual Euclidean distance matches."""
        x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        dist = _pairwise_distances(x)
        assert dist[0, 1].item() == pytest.approx(5.0, abs=1e-5)
        assert dist[1, 0].item() == pytest.approx(5.0, abs=1e-5)


class TestMeanNormalizeDistances:
    """Tests for _mean_normalize_distances helper."""

    def test_normalized_mean_is_one(self):
        """After normalization, mean of positive entries is ~1."""
        x = torch.randn(8, 64)
        dist = _pairwise_distances(x)
        normed = _mean_normalize_distances(dist)
        positive_mean = normed[normed > 0].mean()
        assert positive_mean.item() == pytest.approx(1.0, abs=1e-5)


class TestRelKDDistanceLoss:
    """Tests for relkd_distance_loss."""

    def test_factory(self):
        """Factory returns a callable."""
        loss_fn = relkd_distance_loss()
        assert callable(loss_fn)

    def test_forward_shape(self):
        """Scalar output."""
        loss_fn = relkd_distance_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 128)
        result = loss_fn(student, teacher)
        assert result.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = relkd_distance_loss()
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 128)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss approx 0 (identical distance structure)."""
        loss_fn = relkd_distance_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_dims(self):
        """Works with different student/teacher feature dimensions."""
        loss_fn = relkd_distance_loss()
        student = torch.randn(4, 32)
        teacher = torch.randn(4, 256)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_3d_input_flattens(self):
        """3D input [B, L, D] is flattened to [B, L*D]."""
        loss_fn = relkd_distance_loss()
        student = torch.randn(4, 8, 32)
        teacher = torch.randn(4, 8, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_non_negative(self):
        """Smooth L1 loss is non-negative."""
        loss_fn = relkd_distance_loss()
        student = torch.randn(8, 64)
        teacher = torch.randn(8, 128)
        loss = loss_fn(student, teacher)
        assert loss.item() >= 0

    def test_numerical_distance_structure(self):
        """Verify the loss captures distance structure correctly.

        When student has the same pairwise distance ratios as teacher
        (just scaled), the mu-normalized distances match -> loss ~0.
        """
        loss_fn = relkd_distance_loss()
        teacher = torch.randn(4, 64)
        # Uniformly scaled teacher has identical mu-normalized distances
        student = teacher * 3.0
        loss = loss_fn(student, teacher)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestRelKDAngleLoss:
    """Tests for relkd_angle_loss."""

    def test_factory(self):
        """Factory returns a callable."""
        loss_fn = relkd_angle_loss()
        assert callable(loss_fn)

    def test_forward_shape(self):
        """Scalar output."""
        loss_fn = relkd_angle_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 128)
        result = loss_fn(student, teacher)
        assert result.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = relkd_angle_loss()
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 128)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss approx 0 (identical angle structure)."""
        loss_fn = relkd_angle_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_dims(self):
        """Works with different student/teacher feature dimensions."""
        loss_fn = relkd_angle_loss()
        student = torch.randn(4, 32)
        teacher = torch.randn(4, 256)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_3d_input_flattens(self):
        """3D input [B, L, D] is flattened to [B, L*D]."""
        loss_fn = relkd_angle_loss()
        student = torch.randn(4, 8, 32)
        teacher = torch.randn(4, 8, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_non_negative(self):
        """Smooth L1 loss is non-negative."""
        loss_fn = relkd_angle_loss()
        student = torch.randn(8, 64)
        teacher = torch.randn(8, 128)
        loss = loss_fn(student, teacher)
        assert loss.item() >= 0

    def test_scale_invariant(self):
        """Angle loss is invariant to uniform scaling of features.

        Scaling all features by a constant doesn't change the angles.
        """
        loss_fn = relkd_angle_loss()
        teacher = torch.randn(4, 64)
        student = teacher * 5.0
        loss = loss_fn(student, teacher)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestRelKDDALoss:
    """Tests for relkd_da_loss (combined distance + angle)."""

    def test_factory(self):
        """Factory returns a callable."""
        loss_fn = relkd_da_loss()
        assert callable(loss_fn)

    def test_forward_shape(self):
        """Scalar output."""
        loss_fn = relkd_da_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 128)
        result = loss_fn(student, teacher)
        assert result.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = relkd_da_loss()
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 128)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss approx 0."""
        loss_fn = relkd_da_loss()
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_default_weights(self):
        """Default weights are dist=1, angle=2."""
        loss_fn_da = relkd_da_loss()
        loss_fn_d = relkd_distance_loss()
        loss_fn_a = relkd_angle_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 128)
        da_loss = loss_fn_da(student, teacher)
        d_loss = loss_fn_d(student, teacher)
        a_loss = loss_fn_a(student, teacher)
        expected = 1.0 * d_loss + 2.0 * a_loss
        assert da_loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_custom_weights(self):
        """Custom weights are applied correctly."""
        loss_fn_da = relkd_da_loss(dist_weight=3.0, angle_weight=0.5)
        loss_fn_d = relkd_distance_loss()
        loss_fn_a = relkd_angle_loss()
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 128)
        da_loss = loss_fn_da(student, teacher)
        d_loss = loss_fn_d(student, teacher)
        a_loss = loss_fn_a(student, teacher)
        expected = 3.0 * d_loss + 0.5 * a_loss
        assert da_loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_different_dims(self):
        """Works with different student/teacher feature dimensions."""
        loss_fn = relkd_da_loss()
        student = torch.randn(4, 32)
        teacher = torch.randn(4, 256)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_3d_input_flattens(self):
        """3D input [B, L, D] is flattened to [B, L*D]."""
        loss_fn = relkd_da_loss()
        student = torch.randn(4, 8, 32)
        teacher = torch.randn(4, 8, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()


class TestRelKDBatchSize1:
    """Regression tests: batch_size=1 must produce finite zero loss, not NaN.

    Relational losses operate on pairwise/ternary relations between batch
    elements.  With a single element the distance matrix is [1, 1] = [[0]],
    so there are no positive entries to average over; the loss must still be
    a finite zero with finite gradients.
    """

    def test_distance_loss_batch1_is_finite_zero(self):
        """Distance loss with batch_size=1 is finite and zero."""
        loss_fn = relkd_distance_loss()
        student = torch.randn(1, 64, requires_grad=True)
        teacher = torch.randn(1, 128)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
        assert loss.item() == pytest.approx(0.0, abs=1e-7)
        # Backward must not produce NaN gradients
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()

    def test_angle_loss_batch1_is_finite_zero(self):
        """Angle loss with batch_size=1 is finite and zero."""
        loss_fn = relkd_angle_loss()
        student = torch.randn(1, 64, requires_grad=True)
        teacher = torch.randn(1, 128)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
        assert loss.item() == pytest.approx(0.0, abs=1e-7)
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()

    def test_da_loss_batch1_is_finite_zero(self):
        """Combined DA loss with batch_size=1 is finite and zero."""
        loss_fn = relkd_da_loss()
        student = torch.randn(1, 64, requires_grad=True)
        teacher = torch.randn(1, 128)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
        assert loss.item() == pytest.approx(0.0, abs=1e-7)
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()

    def test_distance_loss_batch1_warns(self, caplog):
        """Distance loss with batch_size=1 logs a warning."""
        loss_fn = relkd_distance_loss()
        student = torch.randn(1, 64)
        teacher = torch.randn(1, 128)
        with caplog.at_level(logging.WARNING):
            loss_fn(student, teacher)
        assert any("batch_size=1" in msg for msg in caplog.messages)

    def test_angle_loss_batch1_warns(self, caplog):
        """Angle loss with batch_size=1 logs a warning."""
        loss_fn = relkd_angle_loss()
        student = torch.randn(1, 64)
        teacher = torch.randn(1, 128)
        with caplog.at_level(logging.WARNING):
            loss_fn(student, teacher)
        assert any("batch_size=1" in msg for msg in caplog.messages)

    def test_mean_normalize_distances_empty_positive(self):
        """_mean_normalize_distances returns dist unchanged when no positive entries."""
        dist = torch.zeros(1, 1)
        result = _mean_normalize_distances(dist)
        assert torch.isfinite(result).all()
        assert result.item() == pytest.approx(0.0)
