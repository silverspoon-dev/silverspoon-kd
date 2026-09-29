"""Unit tests for Mahalanobis-metric loss functions."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from silverspoon_kd.losses.mahalanobis import mahal_cosine_loss, mahal_mse_loss
from silverspoon_kd.losses.registry import get_loss_function

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_weight_matrix(vocab: int = 100, dim: int = 64) -> torch.Tensor:
    """Random (V, D) weight matrix for tests."""
    return torch.randn(vocab, dim)


def _make_metric_matrix(dim: int = 64) -> torch.Tensor:
    """Random PSD (D, D) metric matrix for tests."""
    A = torch.randn(dim, dim)
    return A.T @ A  # guaranteed PSD


# ═══════════════════════════════════════════════════════════════════════════
#  mahal_mse
# ═══════════════════════════════════════════════════════════════════════════


class TestMahalMSE:
    """Tests for mahal_mse_loss factory."""

    def test_requires_matrix_kwarg(self):
        """Raises ValueError when neither weight_matrix nor metric_matrix given."""
        with pytest.raises(ValueError, match=r"weight_matrix.*metric_matrix"):
            mahal_mse_loss()

    def test_factory_with_weight_matrix(self):
        """Factory returns callable when weight_matrix is provided."""
        W = _make_weight_matrix()
        loss_fn = mahal_mse_loss(weight_matrix=W)
        assert callable(loss_fn)

    def test_factory_with_metric_matrix(self):
        """Factory returns callable when metric_matrix is provided."""
        M = _make_metric_matrix()
        loss_fn = mahal_mse_loss(metric_matrix=M)
        assert callable(loss_fn)

    def test_registry_lookup(self):
        """get_loss_function('mahal_mse', ...) works."""
        W = _make_weight_matrix()
        loss_fn = get_loss_function("mahal_mse", weight_matrix=W)
        assert callable(loss_fn)

    def test_forward_shape_2d(self):
        """Scalar output for 2D input."""
        W = _make_weight_matrix()
        loss_fn = mahal_mse_loss(weight_matrix=W)
        loss = loss_fn(torch.randn(4, 64), torch.randn(4, 64))
        assert loss.shape == ()

    def test_forward_shape_3d(self):
        """Scalar output for 3D input (batch, seq, dim)."""
        W = _make_weight_matrix()
        loss_fn = mahal_mse_loss(weight_matrix=W)
        loss = loss_fn(torch.randn(2, 16, 64), torch.randn(2, 16, 64))
        assert loss.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad on student."""
        W = _make_weight_matrix()
        loss_fn = mahal_mse_loss(weight_matrix=W)
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss exactly 0."""
        W = _make_weight_matrix()
        loss_fn = mahal_mse_loss(weight_matrix=W)
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_numerical_correctness_weight_matrix(self):
        """Manual (s-t)^T W^T W (s-t) matches function output."""
        W = _make_weight_matrix(vocab=50, dim=32)
        loss_fn = mahal_mse_loss(weight_matrix=W)

        student = torch.randn(4, 32)
        teacher = torch.randn(4, 32)

        # Manual reference
        M = W.T @ W
        diff = student - teacher
        expected = (diff @ M * diff).sum(-1).mean()

        actual = loss_fn(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_numerical_correctness_metric_matrix(self):
        """Direct metric_matrix gives same result as weight_matrix-derived M."""
        W = _make_weight_matrix(vocab=50, dim=32)
        M = W.T @ W

        loss_w = mahal_mse_loss(weight_matrix=W)
        loss_m = mahal_mse_loss(metric_matrix=M)

        student = torch.randn(4, 32)
        teacher = torch.randn(4, 32)

        assert loss_w(student, teacher).item() == pytest.approx(
            loss_m(student, teacher).item(), rel=1e-4
        )

    def test_equivalent_to_logit_mse(self):
        """(s-t)^T M (s-t) == ||W(s-t)||^2 / N (the whole point of Mahalanobis MSE)."""
        W = _make_weight_matrix(vocab=50, dim=32)
        loss_fn = mahal_mse_loss(weight_matrix=W)

        student = torch.randn(4, 32)
        teacher = torch.randn(4, 32)

        # Logit-space MSE (O(V*D))
        diff = student - teacher
        logit_diff = diff @ W.T  # (4, 50)
        logit_mse = (logit_diff**2).sum(-1).mean()

        # Mahalanobis MSE (O(D^2))
        mahal = loss_fn(student, teacher)

        assert mahal.item() == pytest.approx(logit_mse.item(), rel=1e-4)

    def test_with_pre_norm(self):
        """pre_norm is applied before metric computation."""
        W = _make_weight_matrix(vocab=50, dim=32)
        norm = nn.LayerNorm(32)
        loss_no_norm = mahal_mse_loss(weight_matrix=W)
        loss_with_norm = mahal_mse_loss(weight_matrix=W, pre_norm=norm)

        student = torch.randn(4, 32)
        teacher = torch.randn(4, 32)

        # Results should differ (norm changes the inputs)
        val_no = loss_no_norm(student, teacher)
        val_with = loss_with_norm(student, teacher)
        assert val_no.item() != pytest.approx(val_with.item(), abs=1e-3)

    def test_pre_norm_numerical_correctness(self):
        """Manual norm + metric matches function output."""
        W = _make_weight_matrix(vocab=50, dim=32)
        norm = nn.LayerNorm(32)
        loss_fn = mahal_mse_loss(weight_matrix=W, pre_norm=norm)

        student = torch.randn(4, 32)
        teacher = torch.randn(4, 32)

        # Manual reference
        M = W.T @ W
        s_normed = norm(student)
        with torch.no_grad():
            t_normed = norm(teacher)
        diff = s_normed - t_normed
        expected = (diff @ M * diff).sum(-1).mean()

        actual = loss_fn(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_pre_norm_teacher_no_grad(self):
        """Teacher path through pre_norm does not accumulate gradients."""
        W = _make_weight_matrix(vocab=50, dim=32)
        norm = nn.LayerNorm(32)
        loss_fn = mahal_mse_loss(weight_matrix=W, pre_norm=norm)

        student = torch.randn(4, 32, requires_grad=True)
        teacher = torch.randn(4, 32, requires_grad=True)
        loss = loss_fn(student, teacher)
        loss.backward()

        assert student.grad is not None
        assert teacher.grad is None


# ═══════════════════════════════════════════════════════════════════════════
#  mahal_cosine
# ═══════════════════════════════════════════════════════════════════════════


class TestMahalCosine:
    """Tests for mahal_cosine_loss factory."""

    def test_requires_matrix_kwarg(self):
        """Raises ValueError when neither weight_matrix nor metric_matrix given."""
        with pytest.raises(ValueError, match=r"weight_matrix.*metric_matrix"):
            mahal_cosine_loss()

    def test_factory_with_weight_matrix(self):
        """Factory returns callable when weight_matrix is provided."""
        W = _make_weight_matrix()
        loss_fn = mahal_cosine_loss(weight_matrix=W)
        assert callable(loss_fn)

    def test_registry_lookup(self):
        """get_loss_function('mahal_cosine', ...) works."""
        W = _make_weight_matrix()
        loss_fn = get_loss_function("mahal_cosine", weight_matrix=W)
        assert callable(loss_fn)

    def test_forward_shape_2d(self):
        """Scalar output for 2D input."""
        W = _make_weight_matrix()
        loss_fn = mahal_cosine_loss(weight_matrix=W)
        loss = loss_fn(torch.randn(4, 64), torch.randn(4, 64))
        assert loss.shape == ()

    def test_forward_shape_3d(self):
        """Scalar output for 3D input."""
        W = _make_weight_matrix()
        loss_fn = mahal_cosine_loss(weight_matrix=W)
        loss = loss_fn(torch.randn(2, 16, 64), torch.randn(2, 16, 64))
        assert loss.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad on student."""
        W = _make_weight_matrix()
        loss_fn = mahal_cosine_loss(weight_matrix=W)
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_zero_loss(self):
        """Same input -> loss approx 0 (cos_M = 1)."""
        W = _make_weight_matrix()
        loss_fn = mahal_cosine_loss(weight_matrix=W)
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_numerical_correctness(self):
        """Manual cos_M computation matches function output."""
        W = _make_weight_matrix(vocab=50, dim=32)
        eps = 1e-8
        loss_fn = mahal_cosine_loss(weight_matrix=W, eps=eps)

        student = torch.randn(4, 32)
        teacher = torch.randn(4, 32)

        # Manual reference
        M = W.T @ W
        sM = student @ M
        tM = teacher @ M
        sMt = (sM * teacher).sum(-1)
        sMs = (sM * student).sum(-1)
        tMt = (tM * teacher).sum(-1)
        cos_M = sMt / (sMs.sqrt() * tMt.sqrt() + eps)
        expected = (1.0 - cos_M).mean()

        actual = loss_fn(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_equivalent_to_logit_cosine(self):
        """cos_M(s, t) == cos(Ws, Wt) (the whole point of Mahalanobis cosine)."""
        W = _make_weight_matrix(vocab=50, dim=32)
        loss_fn = mahal_cosine_loss(weight_matrix=W)

        student = torch.randn(4, 32)
        teacher = torch.randn(4, 32)

        # Logit-space cosine (O(V*D))
        s_logits = student @ W.T
        t_logits = teacher @ W.T
        logit_cosine = (1 - F.cosine_similarity(s_logits, t_logits, dim=-1)).mean()

        # Mahalanobis cosine (O(D^2))
        mahal = loss_fn(student, teacher)

        assert mahal.item() == pytest.approx(logit_cosine.item(), rel=1e-4)

    def test_with_pre_norm(self):
        """pre_norm is applied before metric computation."""
        gen = torch.Generator().manual_seed(99)
        W = _make_weight_matrix(vocab=50, dim=32)
        norm = nn.LayerNorm(32)
        loss_no_norm = mahal_cosine_loss(weight_matrix=W)
        loss_with_norm = mahal_cosine_loss(weight_matrix=W, pre_norm=norm)

        student = torch.randn(4, 32, generator=gen)
        teacher = torch.randn(4, 32, generator=gen)

        val_no = loss_no_norm(student, teacher)
        val_with = loss_with_norm(student, teacher)
        assert val_no.item() != pytest.approx(val_with.item(), abs=1e-4)

    def test_pre_norm_teacher_no_grad(self):
        """Teacher path through pre_norm does not accumulate gradients."""
        W = _make_weight_matrix(vocab=50, dim=32)
        norm = nn.LayerNorm(32)
        loss_fn = mahal_cosine_loss(weight_matrix=W, pre_norm=norm)

        student = torch.randn(4, 32, requires_grad=True)
        teacher = torch.randn(4, 32, requires_grad=True)
        loss = loss_fn(student, teacher)
        loss.backward()

        assert student.grad is not None
        assert teacher.grad is None

    def test_near_zero_vectors_finite_loss(self):
        """Near-zero student/teacher vectors must produce finite loss and gradients.

        Regression test: the denominator is clamped with clamp(min=eps)
        rather than computed as (product + eps), which can still be tiny
        when both sqrt terms are near zero.
        """
        W = _make_weight_matrix(vocab=50, dim=32)
        loss_fn = mahal_cosine_loss(weight_matrix=W)

        student = (torch.randn(4, 32) * 1e-10).requires_grad_(True)
        teacher = torch.randn(4, 32) * 1e-10
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()
