"""Unit tests for KL divergence loss functions."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from silverspoon_kd.losses.kl import jsd_loss, kl_divergence_loss, logit_lens_kl_loss
from silverspoon_kd.losses.registry import get_loss_function


class TestKLDivergenceLoss:
    """Tests for kl_divergence_loss factory."""

    def test_factory(self):
        """kl_divergence_loss() returns a callable."""
        loss_fn = kl_divergence_loss()
        assert callable(loss_fn)

    def test_forward_shape(self):
        """Scalar output."""
        loss_fn = kl_divergence_loss()
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        result = loss_fn(student, teacher)
        assert result.shape == ()
        assert result.item() >= 0

    def test_custom_temperature(self):
        """Custom temperature works."""
        loss_fn = kl_divergence_loss(temperature=2.0)
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        result = loss_fn(student, teacher)
        assert result.shape == ()

    def test_rejects_unknown_kwargs(self):
        """Unknown kwargs (e.g. typos) raise TypeError.

        The factory has an explicit keyword-only signature, so a misspelled
        or unsupported argument (``reduction`` is not supported) is rejected
        instead of being silently dropped.
        """
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            kl_divergence_loss(temprature=2.0)  # typo
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            kl_divergence_loss(reduction="sum")  # never supported

    def test_gradient_flow(self):
        """Backward produces non-None grad."""
        loss_fn = kl_divergence_loss()
        student = torch.randn(4, 10, requires_grad=True)
        teacher = torch.randn(4, 10)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_identical_inputs_near_zero_loss(self):
        """Same input -> loss approx 0."""
        loss_fn = kl_divergence_loss()
        x = torch.randn(4, 10)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_numerical_correctness(self):
        """Manual softmax/log_softmax/kl_div matches function output."""
        temperature = 2.0
        loss_fn = kl_divergence_loss(temperature=temperature)
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)

        student_log_probs = F.log_softmax(student / temperature, dim=-1)
        teacher_probs = F.softmax(teacher / temperature, dim=-1)
        expected = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            temperature**2
        )

        actual = loss_fn(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-5)


class TestLogitLensKLLoss:
    """Tests for logit_lens_kl loss function."""

    def test_factory_requires_output_head(self):
        """Calling without output_head raises ValueError."""
        with pytest.raises(ValueError, match="output_head"):
            get_loss_function("logit_lens_kl")

    def test_factory_with_output_head(self):
        """get_loss_function('logit_lens_kl', output_head=...) returns callable."""
        lm_head = nn.Linear(64, 100)
        loss_fn = get_loss_function("logit_lens_kl", output_head=lm_head)
        assert callable(loss_fn)

    def test_unknown_kwarg_rejected(self):
        """A misspelled kwarg raises instead of being silently ignored."""
        with pytest.raises(TypeError):
            logit_lens_kl_loss(lm_head=nn.Linear(64, 100))  # type: ignore[call-arg]

    def test_forward_shape(self):
        """Scalar output."""
        lm_head = nn.Linear(64, 100)
        loss_fn = logit_lens_kl_loss(output_head=lm_head)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_gradient_flow(self):
        """Backward produces non-None grad on student (teacher is no-grad)."""
        lm_head = nn.Linear(64, 100)
        loss_fn = logit_lens_kl_loss(output_head=lm_head)
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None

    def test_teacher_no_grad(self):
        """Teacher tensor should not have grad after backward."""
        lm_head = nn.Linear(64, 100)
        loss_fn = logit_lens_kl_loss(output_head=lm_head)
        student = torch.randn(4, 64, requires_grad=True)
        teacher = torch.randn(4, 64, requires_grad=True)
        loss = loss_fn(student, teacher)
        loss.backward()
        # Teacher path uses torch.no_grad(), so no grad accumulated
        assert teacher.grad is None

    def test_temperature_scaling(self):
        """Different temperatures produce different losses."""
        lm_head = nn.Linear(64, 100)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        loss_t1 = logit_lens_kl_loss(output_head=lm_head, temperature=1.0)(student, teacher)
        loss_t4 = logit_lens_kl_loss(output_head=lm_head, temperature=4.0)(student, teacher)
        assert loss_t1.item() != pytest.approx(loss_t4.item(), abs=1e-4)

    def test_numerical_correctness(self):
        """Manual output_head -> softmax/log_softmax -> kl_div * T^2 matches."""
        lm_head = nn.Linear(64, 100)
        temperature = 2.0
        loss_fn = logit_lens_kl_loss(output_head=lm_head, temperature=temperature)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)

        # Manual computation
        with torch.no_grad():
            teacher_logits = lm_head(teacher)
        student_logits = lm_head(student)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        expected = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            temperature**2
        )

        actual = loss_fn(student, teacher)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-5)


class TestJSDLossBetaValidation:
    """Regression tests: jsd_loss must reject invalid beta values."""

    @pytest.mark.parametrize("beta", [0.0, 1.0, -0.5, 1.5, 2.0])
    def test_invalid_beta_raises(self, beta):
        """beta outside (0, 1) must raise ValueError at factory time."""
        with pytest.raises(ValueError, match="0 < beta < 1"):
            jsd_loss(beta=beta)

    @pytest.mark.parametrize("beta", [0.01, 0.1, 0.5, 0.9, 0.99])
    def test_valid_beta_succeeds(self, beta):
        """beta strictly between 0 and 1 must create a working loss."""
        loss_fn = jsd_loss(beta=beta)
        student = torch.randn(4, 10, requires_grad=True)
        teacher = torch.randn(4, 10)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss), f"beta={beta} produced non-finite loss"
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()


class TestKLChunkedAccumulator:
    """Chunked KL/JSD must match the non-chunked results.

    The chunked paths accumulate in a tensor of the inputs' device and
    dtype, so both paths produce equivalent results.
    """

    def test_kl_chunked_matches_unchunked(self):
        """Chunked and non-chunked KL produce the same result."""
        student = torch.randn(8, 100)
        teacher = torch.randn(8, 100)
        loss_full = kl_divergence_loss(temperature=2.0)(student, teacher)
        loss_chunked = kl_divergence_loss(temperature=2.0, chunk_size=3)(student, teacher)
        assert loss_chunked.item() == pytest.approx(loss_full.item(), rel=1e-4)

    def test_jsd_chunked_matches_unchunked(self):
        """Chunked and non-chunked JSD produce the same result."""
        student = torch.randn(8, 100)
        teacher = torch.randn(8, 100)
        loss_full = jsd_loss(temperature=2.0, beta=0.3)(student, teacher)
        loss_chunked = jsd_loss(temperature=2.0, beta=0.3, chunk_size=3)(student, teacher)
        assert loss_chunked.item() == pytest.approx(loss_full.item(), rel=1e-4)

    def test_kl_chunked_gradient_flow(self):
        """Chunked KL produces valid gradients."""
        student = torch.randn(8, 100, requires_grad=True)
        teacher = torch.randn(8, 100)
        loss = kl_divergence_loss(chunk_size=3)(student, teacher)
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()
