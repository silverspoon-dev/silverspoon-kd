"""Unit tests for contrastive loss."""

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.losses.contrastive import (
    ContrastiveDistillationLoss,
    contrastive_loss,
)


class TestContrastiveDistillationLoss:
    """Tests for ContrastiveDistillationLoss."""

    def test_init_with_explicit_dims_same(self):
        """Test initialization with same student/teacher dims (no projector)."""
        loss_fn = ContrastiveDistillationLoss(student_dim=128, teacher_dim=128)
        assert loss_fn.projector is None
        assert loss_fn._initialized is True

    def test_init_with_explicit_dims_different(self):
        """Test initialization with different dims creates projector."""
        loss_fn = ContrastiveDistillationLoss(student_dim=64, teacher_dim=128)
        assert loss_fn.projector is not None
        assert isinstance(loss_fn.projector, nn.Linear)
        assert loss_fn.projector.in_features == 64
        assert loss_fn.projector.out_features == 128
        assert loss_fn._initialized is True

    def test_init_without_dims(self):
        """Test initialization without dims (lazy init)."""
        loss_fn = ContrastiveDistillationLoss()
        assert loss_fn.projector is None
        assert loss_fn._initialized is False

    def test_init_custom_temperature(self):
        """Test custom temperature parameter."""
        loss_fn = ContrastiveDistillationLoss(temperature=0.5)
        assert loss_fn.temperature == 0.5

    def test_forward_same_dims_2d(self):
        """Test forward pass with same dimensions, 2D input."""
        loss_fn = ContrastiveDistillationLoss(temperature=0.07)
        student = torch.randn(8, 64)
        teacher = torch.randn(8, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_forward_different_dims_2d(self):
        """Test forward pass with different dimensions, 2D input (lazy init)."""
        loss_fn = ContrastiveDistillationLoss()
        student = torch.randn(8, 64)
        teacher = torch.randn(8, 128)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()
        assert loss_fn.projector is not None
        assert loss_fn._initialized is True

    def test_forward_3d_input_flattens(self):
        """Test that 3D input is flattened to 2D."""
        loss_fn = ContrastiveDistillationLoss()
        student = torch.randn(4, 16, 64)  # [B, L, D]
        teacher = torch.randn(4, 16, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_forward_3d_input_different_dims(self):
        """Test 3D input with different dims triggers projector."""
        loss_fn = ContrastiveDistillationLoss()
        student = torch.randn(4, 16, 32)
        teacher = torch.randn(4, 16, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()
        assert loss_fn.projector is not None

    def test_forward_with_explicit_dims(self):
        """Test forward when dims are set at init."""
        loss_fn = ContrastiveDistillationLoss(student_dim=32, teacher_dim=64)
        student = torch.randn(8, 32)
        teacher = torch.randn(8, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_forward_no_projector_needed_lazy(self):
        """Test lazy init when dims match (no projector created)."""
        loss_fn = ContrastiveDistillationLoss()
        student = torch.randn(8, 64)
        teacher = torch.randn(8, 64)
        loss_fn(student, teacher)
        assert loss_fn.projector is None
        assert loss_fn._initialized is True

    def test_forward_produces_gradient(self):
        """Test that loss supports backward pass."""
        loss_fn = ContrastiveDistillationLoss()
        student = torch.randn(8, 64, requires_grad=True)
        teacher = torch.randn(8, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None


class TestContrastiveLossFactory:
    """Tests for contrastive_loss factory function."""

    def test_contrastive_loss_factory(self):
        """Test contrastive loss factory."""
        loss_fn = contrastive_loss()
        assert isinstance(loss_fn, ContrastiveDistillationLoss)

    def test_contrastive_loss_factory_with_kwargs(self):
        """Test contrastive loss factory passes kwargs."""
        loss_fn = contrastive_loss(temperature=0.5)
        assert loss_fn.temperature == 0.5


class TestContrastiveBatchSize1:
    """Regression tests: contrastive loss with a single sample.

    InfoNCE with N=1 gives logits [[s@t]], labels [0], so cross_entropy is
    log(softmax([s@t])[0]) = log(1) = 0.  The loss is degenerate (always 0)
    but must still be finite and differentiable.
    """

    def test_batch1_2d_is_finite(self):
        """batch_size=1, 2D input produces a finite (zero) loss."""
        loss_fn = ContrastiveDistillationLoss()
        student = torch.randn(1, 64, requires_grad=True)
        teacher = torch.randn(1, 64)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_batch1_gradient_flow(self):
        """Backward on batch=1 contrastive loss does not produce NaN grads."""
        loss_fn = ContrastiveDistillationLoss()
        student = torch.randn(1, 64, requires_grad=True)
        teacher = torch.randn(1, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()


class TestLossEdgeCases:
    """Edge case coverage across loss functions.

    These catch silent failure modes that could produce NaN/Inf losses
    mid-training without crashing loudly.
    """

    @pytest.mark.parametrize(
        "loss_name,kwargs",
        [
            ("mse", {}),
            ("cosine", {}),
            ("smooth_l1", {}),
            ("kl_div", {}),
            ("normalized_mse", {}),
            ("angular_magnitude", {}),
        ],
    )
    def test_identical_inputs_near_zero_loss(self, loss_name, kwargs):
        """Identical student/teacher produce ~zero loss (no spurious gradient)."""
        from silverspoon_kd.losses.registry import get_loss_function

        loss_fn = get_loss_function(loss_name, **kwargs)
        x = torch.randn(4, 64)
        loss = loss_fn(x, x.clone())
        assert torch.isfinite(loss)
        # Tolerance: cosine/angular can have tiny round-off; others are exact
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize(
        "loss_name,kwargs",
        [
            ("mse", {}),
            ("cosine", {}),
            ("smooth_l1", {}),
            ("kl_div", {}),
            ("normalized_mse", {}),
        ],
    )
    def test_all_zero_tensors_finite(self, loss_name, kwargs):
        """All-zero student and teacher produce finite loss."""
        from silverspoon_kd.losses.registry import get_loss_function

        loss_fn = get_loss_function(loss_name, **kwargs)
        student = torch.zeros(4, 64, requires_grad=True)
        teacher = torch.zeros(4, 64)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"{loss_name} produced non-finite loss on zero inputs"

    @pytest.mark.parametrize("temperature", [0.01, 0.1, 10.0, 100.0])
    def test_kl_extreme_temperatures(self, temperature):
        """KL divergence with extreme temperatures stays finite."""
        from silverspoon_kd.losses.registry import get_loss_function

        loss_fn = get_loss_function("kl_div", temperature=temperature)
        student = torch.randn(4, 100)
        teacher = torch.randn(4, 100)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"KL produced non-finite loss at temperature={temperature}"


class TestLossFactoryRejectsUnknownKwargs:
    """Regression tests (issue #27): loss factories must reject typos.

    Each factory has an explicit keyword-only signature, so an unknown
    keyword such as ``temprature=2.0`` raises ``TypeError`` instead of being
    silently ignored.
    """

    @pytest.mark.parametrize(
        "loss_name",
        [
            "mse",
            "cosine",
            "smooth_l1",
            "kl_div",
            "jsd",
            "normalized_mse",
            "angular_magnitude",
            "contrastive",
            "logit_lens_kl",
            "mahal_mse",
            "mahal_cosine",
            "relkd_distance",
            "relkd_angle",
            "relkd_da",
        ],
    )
    def test_unknown_kwarg_raises(self, loss_name):
        """A typo'd kwarg must raise TypeError instead of being silently ignored."""
        from silverspoon_kd.losses.registry import get_loss_function

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            get_loss_function(loss_name, totally_made_up_kwarg=42)

    def test_no_factory_uses_var_keyword(self):
        """Meta-test: no loss factory may accept ``**kwargs``.

        The ``**kwargs: Any`` pattern silently swallows typos.  This test
        inspects every factory in ``LOSS_REGISTRY`` and asserts that none
        has a VAR_KEYWORD parameter, guarding against a future loss
        factory being added with that pattern.
        """
        import inspect

        from silverspoon_kd.losses.registry import LOSS_REGISTRY

        offenders = []
        for name, factory in LOSS_REGISTRY.items():
            sig = inspect.signature(factory)
            for param in sig.parameters.values():
                if param.kind is inspect.Parameter.VAR_KEYWORD:
                    offenders.append(f"{name} (factory={factory.__name__})")
                    break
        assert not offenders, (
            f"Loss factories must use explicit keyword-only parameters "
            f"instead of **kwargs (which silently drops typos). Offenders: "
            f"{offenders}"
        )
