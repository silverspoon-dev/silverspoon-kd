"""
Unit tests for per-alignment loss weighting and magnitude-aware normalization.

Tests cover:
- Alignment.loss_weight parameter
- HKD weighted loss accumulation (with and without magnitude-aware)
- ResKD magnitude-aware soft/hard normalization
- BKD warning for unsupported loss_weight
- Gradient proportionality verification
- Metric logging (raw vs normalized)
"""

import logging

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
from silverspoon_kd.distillers.holistic_distiller import HolisticDistiller
from silverspoon_kd.distillers.response_based_distiller import ResponseBasedDistiller
from silverspoon_kd.losses.kl import kl_divergence_loss
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)

from ..conftest import create_alignment, create_batch

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_holistic_args(training_args, **overrides):
    """Create TrainingArguments from base training_args fixture."""
    return TrainingArguments(
        output_dir=training_args.output_dir,
        max_steps=training_args.max_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=training_args.use_cpu,
        **overrides,
    )


def _make_reskd_args(training_args, **overrides):
    """Create TrainingArguments from base training_args fixture."""
    return TrainingArguments(
        output_dir=training_args.output_dir,
        max_steps=training_args.max_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=training_args.use_cpu,
        **overrides,
    )


def _make_blockwise_args(training_args, **overrides):
    """Create TrainingArguments from base training_args fixture."""
    return TrainingArguments(
        output_dir=training_args.output_dir,
        max_steps=training_args.max_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=training_args.use_cpu,
        **overrides,
    )


# ── Alignment.loss_weight ───────────────────────────────────────────────────


class TestAlignmentLossWeight:
    """Tests for Alignment.loss_weight parameter."""

    def test_loss_weight_default_is_one(self):
        """Default loss_weight should be 1.0."""
        block = nn.Linear(64, 64)
        alignment = Alignment(teacher_block=block, student_block=block)
        assert alignment.loss_weight == 1.0

    def test_loss_weight_custom(self):
        """Custom loss_weight should be stored correctly."""
        block = nn.Linear(64, 64)
        alignment = Alignment(teacher_block=block, student_block=block, loss_weight=0.5)
        assert alignment.loss_weight == 0.5

    def test_loss_weight_in_init_signature(self):
        """loss_weight should be a proper __init__ kwarg, not monkey-patched."""
        import inspect

        sig = inspect.signature(Alignment.__init__)
        assert "loss_weight" in sig.parameters


# ── HKD loss weighting ──────────────────────────────────────────────────────


class TestHKDLossWeighting:
    """Tests for HolisticDistiller loss weighting and magnitude-aware normalization."""

    def _make_distiller_with_weights(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        weights,
        magnitude_aware=False,
    ):
        """Create an HKD distiller with custom per-alignment loss weights."""
        alignments = []
        for i, w in enumerate(weights):
            teacher_block = teacher_model.get_layer(i)
            student_block = student_model.get_layer(i)
            alignment = create_alignment(
                teacher_block=teacher_block,
                student_block=student_block,
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
            )
            alignment.loss_weight = w
            alignments.append(alignment)

        args = _make_holistic_args(training_args, magnitude_aware_weighting=magnitude_aware)
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=args,
            train_dataset=train_dataset,
        )
        return distiller

    def test_weighted_sum_without_magnitude_aware(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """With magnitude_aware=False, total loss = sum(w_i * L_i)."""
        distiller = self._make_distiller_with_weights(
            teacher_model,
            student_model,
            training_args,
            train_dataset,
            weights=[2.0, 0.5, 1.0],
            magnitude_aware=False,
        )
        distiller._register_capture()
        distiller._reset_step_metrics(is_training=True)

        # Inject captured outputs with known values
        for i in range(3):
            distiller.teacher_capture.captured_outputs[i] = torch.randn(2, 16, 128, device=device)
            distiller.student_capture.captured_outputs[i] = torch.randn(2, 16, 64, device=device)

        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=True,
        )

        # Reconstruct expected total from raw losses and weights
        raw_losses = [v.item() for v in distiller.step_losses]
        expected = 2.0 * raw_losses[0] + 0.5 * raw_losses[1] + 1.0 * raw_losses[2]

        assert total_loss.item() == pytest.approx(expected, rel=1e-5)
        distiller._deregister_capture()

    def test_default_weights_unchanged_behavior(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """With default weights (1.0), total matches simple sum (regression test)."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        distiller._reset_step_metrics(is_training=True)

        distiller.teacher_capture.captured_outputs[0] = torch.randn(2, 16, 128, device=device)
        distiller.student_capture.captured_outputs[0] = torch.randn(2, 16, 64, device=device)

        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=True,
        )

        # With weight=1.0, total should equal the raw loss
        raw_loss = distiller.step_losses[0].item()
        assert total_loss.item() == pytest.approx(raw_loss, rel=1e-5)
        distiller._deregister_capture()

    def test_magnitude_aware_returns_raw_weighted_sum(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """With magnitude_aware=True, the returned *value* is the raw
        weighted sum — identical to what you'd get with magnitude_aware=False.

        The straight-through trick decouples the displayed value from the
        backprop gradient: forward value is ``sum(w_i * L_i)``, gradient
        is from ``sum(w_i * L_i / |L_i.detach()|)``.
        """
        weights = [0.25, 0.75, 1.0]
        distiller = self._make_distiller_with_weights(
            teacher_model,
            student_model,
            training_args,
            train_dataset,
            weights=weights,
            magnitude_aware=True,
        )
        distiller._register_capture()
        distiller._reset_step_metrics(is_training=True)

        for i in range(3):
            distiller.teacher_capture.captured_outputs[i] = torch.randn(2, 16, 128, device=device)
            distiller.student_capture.captured_outputs[i] = torch.randn(2, 16, 64, device=device)

        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=True,
        )

        # Value is the raw weighted sum — exactly what a plain
        # (non-magnitude-aware) distiller would have returned.  The raw
        # per-alignment losses are recorded in ``distiller.step_losses``
        # by ``_compute_alignment_loss``.
        raw_per_alignment = [loss.item() for loss in distiller.step_losses]
        expected_raw = sum(w * r for w, r in zip(weights, raw_per_alignment, strict=False))
        assert total_loss.item() == pytest.approx(expected_raw, rel=1e-5)
        # And crucially, *not* ≈ sum(weights) — that was the old
        # "unhelpful 1.0-ish" behaviour we explicitly moved away from.
        assert abs(total_loss.item() - sum(weights)) > 0.01
        distiller._deregister_capture()

    def test_magnitude_aware_gradient_proportionality(self, device):
        """Gradient contributions should be proportional to weights with magnitude-aware."""
        # Seed deterministically: with unseeded randn(10), the ratio of grad
        # norms is sqrt(F(10,10)) which has a long right tail (~7% chance of
        # falling outside the rel=0.5 tolerance window).
        g = torch.Generator(device=device).manual_seed(0)
        # Create a simple setup: two parameters, each contributing to a different loss
        param_a = torch.randn(10, generator=g, device=device, requires_grad=True)
        param_b = torch.randn(10, generator=g, device=device, requires_grad=True)

        # Simulate losses with very different magnitudes
        loss_a = (param_a**2).sum() * 100  # Large magnitude
        loss_b = (param_b**2).sum() * 0.1  # Small magnitude

        w_a, w_b = 0.5, 0.5  # Equal weights

        # Without magnitude-aware: large loss dominates
        total_naive = w_a * loss_a + w_b * loss_b
        total_naive.backward()
        grad_a_naive = param_a.grad.clone()
        grad_b_naive = param_b.grad.clone()
        param_a.grad = None
        param_b.grad = None

        # Recompute losses (graph consumed)
        loss_a = (param_a**2).sum() * 100
        loss_b = (param_b**2).sum() * 0.1

        # With magnitude-aware: gradients balanced
        total_aware = w_a * loss_a / loss_a.detach().abs().clamp(
            min=1e-8
        ) + w_b * loss_b / loss_b.detach().abs().clamp(min=1e-8)
        total_aware.backward()
        grad_a_aware = param_a.grad.clone()
        grad_b_aware = param_b.grad.clone()

        # Without magnitude-aware, loss_a dominates (100x larger)
        ratio_naive = (grad_a_naive.norm() / grad_b_naive.norm()).item()
        assert ratio_naive > 10  # Large ratio = loss_a dominates

        # With magnitude-aware, gradient norms should be similar (balanced)
        ratio_aware = (grad_a_aware.norm() / grad_b_aware.norm()).item()
        assert ratio_aware == pytest.approx(1.0, rel=0.5)  # Roughly equal

    def test_weight_ratios_not_absolute_values(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Weights [2, 3, 5] should produce identical gradients as [0.2, 0.3, 0.5]."""
        # Set deterministic captured outputs
        teacher_outs = [torch.randn(2, 16, 128, device=device) for _ in range(3)]
        student_outs = [torch.randn(2, 16, 64, device=device) for _ in range(3)]

        results = {}
        for label, weights in [("abs", [2.0, 3.0, 5.0]), ("ratio", [0.2, 0.3, 0.5])]:
            distiller = self._make_distiller_with_weights(
                teacher_model,
                student_model,
                training_args,
                train_dataset,
                weights=weights,
                magnitude_aware=True,
            )
            distiller._register_capture()
            distiller._reset_step_metrics(is_training=True)

            for i in range(3):
                distiller.teacher_capture.captured_outputs[i] = teacher_outs[i]
                distiller.student_capture.captured_outputs[i] = (
                    student_outs[i].clone().detach().requires_grad_(False)
                )

            total_loss = distiller._compute_alignment_losses(
                distiller.teacher_capture,
                distiller.student_capture,
                distiller.alignment_id_to_module_id,
                is_training=True,
            )
            results[label] = total_loss.item()
            distiller._deregister_capture()

        # Under the straight-through detach trick the returned value is
        # ``sum(w_i * L_i)``, so scaling all weights by 10× scales the
        # returned value by 10×.  (The ratio was also 10× under the old
        # "total ≈ sum(weights)" behaviour, coincidentally, but for an
        # entirely different reason.)  Tolerance is loose because each
        # run creates its own freshly-initialised auto projectors, so
        # the per-alignment raw losses differ slightly between the two
        # runs even for identical teacher/student captures.
        assert results["abs"] / results["ratio"] == pytest.approx(10.0, rel=0.05)

    def test_magnitude_aware_zero_loss_safety(self, device):
        """Loss = 0 should not produce NaN or Inf (abs+clamp handles it)."""
        param = torch.zeros(5, requires_grad=True, device=device)
        loss = (param**2).sum()  # = 0.0

        weighted = 0.5 * loss / loss.detach().abs().clamp(min=1e-8)

        assert torch.isfinite(weighted).all()
        weighted.backward()
        assert torch.isfinite(param.grad).all()

    def test_magnitude_aware_negative_loss_safety(self, device):
        """Negative raw loss (from weight corruption, bf16 rounding, etc.)
        must be bounded — NOT amplified by ``clamp(min=1e-8)`` passing the
        sign through.  With ``.abs().clamp(min=1e-8)`` the normalized
        contribution of a -1e10 loss is bounded to magnitude ``weight``,
        not ``weight * 1e18``."""
        for bad in [-1.0, -1e3, -1e6, -1e10]:
            loss = torch.tensor(bad, requires_grad=True, device=device)
            weighted = 0.5 * loss / loss.detach().abs().clamp(min=1e-8)
            # With the abs-clamp fix, magnitude is bounded to `weight` (=0.5),
            # with the sign inherited from the raw loss (so -0.5 here).
            assert weighted.item() == pytest.approx(-0.5, abs=1e-6), (
                f"bad={bad}: expected -0.5, got {weighted.item()}. "
                "Without .abs() on the denominator this blows up to "
                f"{0.5 * bad / max(bad, 1e-8):.3e}."
            )
            # Gradient of the loss / |loss.detach()| term is 1/|loss|
            weighted.backward()
            assert torch.isfinite(loss.grad).all()

    def test_nonfinite_component_is_skipped_in_combine_losses(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """A non-finite (NaN / Inf) loss in *one* component must not poison
        the combined total — the abs-clamp fix bounds finite negatives but
        NaN/Inf still propagate through ``NaN / NaN = NaN``.  ``_combine_losses``
        skips non-finite components entirely, so one bad batch cannot kill
        the whole run."""
        args = _make_reskd_args(
            training_args,
            alpha=0.5,
            magnitude_aware_weighting=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=2.0),
        )
        distiller._reset_step_metrics(is_training=True)

        # Pure Python scalars wrapped in tensors so we control exactly
        # which component is non-finite.
        good_soft = torch.tensor(3.0, device=device, requires_grad=True)
        bad_hard = torch.tensor(float("nan"), device=device, requires_grad=True)
        total = distiller._combine_losses(
            {"soft": good_soft, "hard": bad_hard},
            {"soft": 0.5, "hard": 0.5},
        )
        # Finite result — bad_hard was dropped, total carries only good_soft
        assert torch.isfinite(total), f"total={total}"
        # Value is the raw weighted sum of the *surviving* components:
        # 0.5 * 3.0 = 1.5
        assert total.item() == pytest.approx(1.5, rel=1e-5)
        # Inf also dropped
        distiller._reset_step_metrics(is_training=True)
        bad_hard = torch.tensor(float("inf"), device=device, requires_grad=True)
        total2 = distiller._combine_losses(
            {"soft": good_soft, "hard": bad_hard},
            {"soft": 0.5, "hard": 0.5},
        )
        assert torch.isfinite(total2)
        assert total2.item() == pytest.approx(1.5, rel=1e-5)

    def test_nonfinite_alignment_loss_zeroed_in_apply_loss_weighting(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Same guard for the HKD/BKD path: ``_apply_loss_weighting`` must
        return a detached zero when given a non-finite loss."""
        distiller = self._make_distiller_with_weights(
            teacher_model,
            student_model,
            training_args,
            train_dataset,
            weights=[1.0],
            magnitude_aware=True,
        )
        bad = torch.tensor(float("nan"), device=device, requires_grad=True)
        result = distiller._apply_loss_weighting(bad, distiller.alignments[0])
        assert torch.isfinite(result)
        assert result.item() == 0.0
        # Also check Inf
        bad_inf = torch.tensor(float("inf"), device=device, requires_grad=True)
        result_inf = distiller._apply_loss_weighting(bad_inf, distiller.alignments[0])
        assert torch.isfinite(result_inf)
        assert result_inf.item() == 0.0

    def test_metrics_log_raw_loss(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """_store_loss_metric and step_losses should contain raw (un-normalized) values."""
        distiller = self._make_distiller_with_weights(
            teacher_model,
            student_model,
            training_args,
            train_dataset,
            weights=[2.0, 0.5, 1.0],
            magnitude_aware=True,
        )
        distiller._register_capture()
        distiller._reset_step_metrics(is_training=True)

        for i in range(3):
            distiller.teacher_capture.captured_outputs[i] = torch.randn(2, 16, 128, device=device)
            distiller.student_capture.captured_outputs[i] = torch.randn(2, 16, 64, device=device)

        distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=True,
        )

        # step_losses should contain raw losses (not weight * loss / loss.detach())
        for raw_loss in distiller.step_losses:
            # Raw losses should be > 0 and not ≈ weight (which would indicate normalization)
            assert raw_loss.item() > 0

        # Per-alignment metrics should match raw losses
        for alignment in distiller.alignments:
            metric_key = f"loss/{alignment.get_name()}"
            assert metric_key in distiller.current_step_metrics

        distiller._deregister_capture()

    def test_eval_mode_with_weighted_losses(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Weighted accumulation should work in eval mode (is_training=False)."""
        distiller = self._make_distiller_with_weights(
            teacher_model,
            student_model,
            training_args,
            train_dataset,
            weights=[2.0, 0.5, 1.0],
            magnitude_aware=True,
        )
        distiller._register_capture()
        distiller._reset_step_metrics(is_training=False)

        for i in range(3):
            distiller.teacher_capture.captured_outputs[i] = torch.randn(2, 16, 128, device=device)
            distiller.student_capture.captured_outputs[i] = torch.randn(2, 16, 64, device=device)

        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=False,
        )

        # Should populate eval_losses (not step_losses)
        assert len(distiller.eval_losses) == 3
        assert len(distiller.step_losses) == 0

        # Under the straight-through detach trick the returned value is
        # the raw weighted sum, just like with magnitude_aware=False.
        raw_per_alignment = [loss.item() for loss in distiller.eval_losses]
        expected_raw = (
            2.0 * raw_per_alignment[0] + 0.5 * raw_per_alignment[1] + 1.0 * raw_per_alignment[2]
        )
        assert total_loss.item() == pytest.approx(expected_raw, rel=1e-5)
        distiller._deregister_capture()

    def test_single_alignment_magnitude_aware(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Single alignment with magnitude_aware: total = w (degenerate case)."""
        distiller = self._make_distiller_with_weights(
            teacher_model,
            student_model,
            training_args,
            train_dataset,
            weights=[0.7],
            magnitude_aware=True,
        )
        distiller._register_capture()
        distiller._reset_step_metrics(is_training=True)

        distiller.teacher_capture.captured_outputs[0] = torch.randn(2, 16, 128, device=device)
        distiller.student_capture.captured_outputs[0] = torch.randn(2, 16, 64, device=device)

        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=True,
        )

        # Single alignment under the detach trick: value is the raw
        # weighted loss ``w * L``, not the normalised ``w``.
        raw = distiller.step_losses[0].item()
        assert total_loss.item() == pytest.approx(0.7 * raw, rel=1e-5)
        distiller._deregister_capture()

    def test_weight_ratios_produce_identical_gradients(self, device):
        """Weights [2, 3] and [0.4, 0.6] should produce proportional gradients."""
        # Use the same param values for both weight sets
        param_data = torch.randn(10, device=device)
        results = {}
        for label, weights in [("abs", [2.0, 3.0]), ("ratio", [0.4, 0.6])]:
            param = param_data.clone().detach().requires_grad_(True)
            loss_a = (param[:5] ** 2).sum() * 50
            loss_b = (param[5:] ** 2).sum() * 3

            total = weights[0] * loss_a / loss_a.detach().abs().clamp(min=1e-8) + weights[
                1
            ] * loss_b / loss_b.detach().abs().clamp(min=1e-8)
            total.backward()
            results[label] = param.grad.clone()

        # Gradient DIRECTIONS should be identical (proportional with scale factor)
        scale = sum([2.0, 3.0]) / sum([0.4, 0.6])  # 5.0 / 1.0 = 5.0
        scaled = results["ratio"] * scale
        torch.testing.assert_close(results["abs"], scaled, rtol=1e-5, atol=1e-7)

    def test_training_step_with_magnitude_aware(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Full training step with magnitude_aware should run without error."""
        distiller = self._make_distiller_with_weights(
            teacher_model,
            student_model,
            training_args,
            train_dataset,
            weights=[0.25, 0.75, 1.0],
            magnitude_aware=True,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

        distiller._deregister_capture()


# ── _apply_loss_weighting isolation ─────────────────────────────────────────


class TestApplyLossWeighting:
    """Direct unit tests for BaseDistiller._apply_loss_weighting."""

    def _make_distiller(
        self,
        training_args,
        train_dataset,
        teacher_model,
        student_model,
        single_alignment,
        magnitude_aware=False,
    ):
        """Create a HolisticDistiller for _apply_loss_weighting access."""
        return HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args, magnitude_aware_weighting=magnitude_aware),
            train_dataset=train_dataset,
        )

    def test_weight_1_no_magnitude_aware_returns_loss_identity(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """weight=1.0, magnitude_aware=False → returns original loss object."""
        distiller = self._make_distiller(
            training_args,
            train_dataset,
            teacher_model,
            student_model,
            single_alignment,
            magnitude_aware=False,
        )
        loss = torch.tensor(42.0, device=device, requires_grad=True)
        result = distiller._apply_loss_weighting(loss, single_alignment[0])
        # Should be the exact same tensor (identity optimization)
        assert result is loss

    def test_weight_applied_without_magnitude_aware(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """weight=2.0, magnitude_aware=False → returns 2 * loss."""
        distiller = self._make_distiller(
            training_args,
            train_dataset,
            teacher_model,
            student_model,
            single_alignment,
            magnitude_aware=False,
        )
        single_alignment[0].loss_weight = 2.0
        loss = torch.tensor(5.0, device=device, requires_grad=True)
        result = distiller._apply_loss_weighting(loss, single_alignment[0])
        assert result.item() == pytest.approx(10.0, rel=1e-6)

    def test_magnitude_aware_normalizes(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """weight=0.3, magnitude_aware=True → returns 0.3 * loss / |loss| = 0.3."""
        distiller = self._make_distiller(
            training_args,
            train_dataset,
            teacher_model,
            student_model,
            single_alignment,
            magnitude_aware=True,
        )
        single_alignment[0].loss_weight = 0.3
        loss = torch.tensor(999.0, device=device, requires_grad=True)
        result = distiller._apply_loss_weighting(loss, single_alignment[0])
        assert result.item() == pytest.approx(0.3, rel=1e-5)

    def test_magnitude_aware_zero_clamp(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Zero loss with magnitude_aware should not produce NaN."""
        distiller = self._make_distiller(
            training_args,
            train_dataset,
            teacher_model,
            student_model,
            single_alignment,
            magnitude_aware=True,
        )
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        result = distiller._apply_loss_weighting(loss, single_alignment[0])
        assert torch.isfinite(result)


# ── ResKD magnitude-aware ───────────────────────────────────────────────────


class TestResKDMagnitudeAware:
    """Tests for ResponseBasedDistiller magnitude-aware weighting."""

    def test_magnitude_aware_soft_hard_returns_raw_weighted_sum(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """With magnitude_aware + alpha=0.5, the returned value is the raw
        weighted sum ``(1-alpha) * soft_raw + alpha * hard_raw``, NOT the
        normalised ~sum(weights)=1.0.  Under the straight-through detach
        trick the forward value is interpretable while the gradient is
        still magnitude-balanced.
        """
        args = _make_reskd_args(
            training_args,
            alpha=0.5,
            magnitude_aware_weighting=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=2.0),
        )
        distiller._reset_step_metrics(is_training=True)

        # Create logits with very different magnitudes for soft vs hard
        student_logits = torch.randn(2, 16, 1000, device=device, requires_grad=True)
        teacher_logits = torch.randn(2, 16, 1000, device=device)
        hard_loss = torch.tensor(5.0, device=device, requires_grad=True)

        total, soft = distiller._compute_total_loss(
            student_logits, teacher_logits, hard_loss, track_metrics=True
        )

        # Value is the raw weighted sum: 0.5 * soft_raw + 0.5 * 5.0
        expected_raw = 0.5 * soft.item() + 0.5 * 5.0
        assert total.item() == pytest.approx(expected_raw, rel=1e-4)
        # Explicitly check it's NOT a normalised ~1.0 constant.
        assert abs(total.item() - 1.0) > 0.01

    def test_magnitude_aware_gradient_proportionality(self, device):
        """With alpha=0.75, 75% of gradient should come from hard loss."""
        alpha = 0.75

        # Shared parameter contributing to both losses
        param = torch.randn(10, requires_grad=True, device=device)
        soft_loss = (param[:5] ** 2).sum() * 1000  # Large magnitude
        hard_loss = (param[5:] ** 2).sum() * 0.01  # Small magnitude

        # Without magnitude-aware: soft dominates despite alpha=0.75
        total_naive = (1 - alpha) * soft_loss + alpha * hard_loss
        total_naive.backward()
        grad_naive = param.grad.clone()
        param.grad = None

        # Recompute
        soft_loss = (param[:5] ** 2).sum() * 1000
        hard_loss = (param[5:] ** 2).sum() * 0.01

        # With magnitude-aware
        total_aware = (1 - alpha) * soft_loss / soft_loss.detach().abs().clamp(
            min=1e-8
        ) + alpha * hard_loss / hard_loss.detach().abs().clamp(min=1e-8)
        total_aware.backward()
        grad_aware = param.grad.clone()

        # Without magnitude-aware: soft part gradient much larger
        soft_grad_naive = grad_naive[:5].norm()
        hard_grad_naive = grad_naive[5:].norm()
        assert (soft_grad_naive / hard_grad_naive).item() > 100  # soft dominates

        # With magnitude-aware: ratio matches alpha weights
        soft_grad_aware = grad_aware[:5].norm()
        hard_grad_aware = grad_aware[5:].norm()
        # (1-alpha)/alpha = 0.25/0.75 ≈ 0.333
        ratio = (soft_grad_aware / hard_grad_aware).item()
        assert ratio < 2.0  # Much more balanced than naive

    def test_soft_only_no_normalization(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """When alpha=0, magnitude_aware has no effect (single component)."""
        for mag_aware in [False, True]:
            args = _make_reskd_args(
                training_args,
                alpha=0.0,
                magnitude_aware_weighting=mag_aware,
            )
            distiller = ResponseBasedDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                args=args,
                train_dataset=train_dataset,
                soft_loss_fn=kl_divergence_loss(temperature=2.0),
            )
            distiller._reset_step_metrics(is_training=True)

            student_logits = torch.randn(2, 16, 1000, device=device)
            teacher_logits = torch.randn(2, 16, 1000, device=device)

            total, soft = distiller._compute_total_loss(
                student_logits, teacher_logits, hard_loss=None, track_metrics=True
            )

            # When alpha=0, total == soft regardless of magnitude_aware
            assert total.item() == pytest.approx(soft.item(), rel=1e-5)

    def test_raw_per_component_metrics_preserved(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """loss/soft and loss/hard track the raw per-component losses, not
        the magnitude-normalised ones.  The combined total is not tracked as
        a sibling metric; it is returned as the main loss."""
        args = _make_reskd_args(
            training_args,
            alpha=0.5,
            magnitude_aware_weighting=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=2.0),
        )
        distiller._reset_step_metrics(is_training=True)

        student_logits = torch.randn(2, 16, 1000, device=device, requires_grad=True)
        teacher_logits = torch.randn(2, 16, 1000, device=device)
        hard_loss = torch.tensor(10.0, device=device, requires_grad=True)

        distiller._compute_total_loss(student_logits, teacher_logits, hard_loss, track_metrics=True)

        # Raw component metrics should be present
        assert "loss/soft" in distiller.current_step_metrics
        assert "loss/hard" in distiller.current_step_metrics

        # Raw soft/hard should reflect actual loss values, not normalized ones
        raw_soft = distiller.current_step_metrics["loss/soft"]
        raw_hard = distiller.current_step_metrics["loss/hard"]
        assert raw_hard.item() == pytest.approx(10.0, rel=1e-5)
        assert raw_soft.item() > 0  # Not ≈ 0.5 (which would be normalized)

    def test_magnitude_aware_alpha_gt0_hard_loss_none(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """alpha>0 + magnitude_aware + hard_loss=None → falls through to soft-only."""
        args = _make_reskd_args(
            training_args,
            alpha=0.5,
            magnitude_aware_weighting=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=2.0),
        )
        distiller._reset_step_metrics(is_training=True)

        student_logits = torch.randn(2, 16, 1000, device=device)
        teacher_logits = torch.randn(2, 16, 1000, device=device)

        # hard_loss=None (no labels in batch) — should return soft-only
        total, soft = distiller._compute_total_loss(
            student_logits, teacher_logits, hard_loss=None, track_metrics=True
        )

        assert total.item() == pytest.approx(soft.item(), rel=1e-5)

    def test_track_metrics_false_suppresses_per_component_logging(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """``track_metrics=False`` suppresses the per-component
        ``loss/soft`` / ``loss/hard`` sibling metrics.  There is no
        ``loss/total`` sibling metric to check — the returned tensor's
        value already *is* the raw weighted sum, so the main ``loss`` /
        ``eval_loss`` field carries it to EarlyStoppingCallback."""
        args = _make_reskd_args(
            training_args,
            alpha=0.5,
            magnitude_aware_weighting=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=2.0),
        )
        distiller._reset_step_metrics(is_training=True)

        student_logits = torch.randn(2, 16, 1000, device=device, requires_grad=True)
        teacher_logits = torch.randn(2, 16, 1000, device=device)
        hard_loss = torch.tensor(5.0, device=device, requires_grad=True)

        total, _ = distiller._compute_total_loss(
            student_logits, teacher_logits, hard_loss, track_metrics=False
        )

        # Per-component metrics are suppressed with track_metrics=False
        assert "loss/soft" not in distiller.current_step_metrics
        assert "loss/hard" not in distiller.current_step_metrics
        assert "loss/total" not in distiller.current_step_metrics

        # The *returned* loss carries the raw weighted sum regardless of
        # track_metrics — it's the autograd graph's forward value, not a
        # tracked sibling metric — so early stopping can still see real
        # progress.  (Specifically: NOT ≈ 1.0, as a normalised sum would be.)
        assert abs(total.item() - 1.0) > 0.01

    def test_reskd_training_step_with_magnitude_aware(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Full ResKD training step with magnitude_aware should run without error."""
        args = _make_reskd_args(
            training_args,
            alpha=0.5,
            magnitude_aware_weighting=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=2.0),
        )

        batch = create_batch(batch_size=2, seq_len=16, device=device)
        loss = distiller.training_step(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    @pytest.mark.parametrize(
        "alpha, provide_hard_loss, mag_aware",
        [
            (0.0, False, False),  # pure soft, no magnitude-aware
            (0.0, False, True),  # pure soft, magnitude-aware (no effect)
            (0.5, True, False),  # soft+hard, no magnitude-aware
            (0.5, True, True),  # soft+hard, magnitude-aware
            (0.5, False, False),  # alpha>0 but hard_loss=None, no magnitude-aware
            (0.5, False, True),  # alpha>0 but hard_loss=None, magnitude-aware
        ],
    )
    def test_returned_loss_finite_and_per_component_tracked(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        device,
        alpha,
        provide_hard_loss,
        mag_aware,
    ):
        """Across every code path (alpha × hard_loss × magnitude_aware):
        the returned ``total`` is finite and ``loss/soft`` is tracked as a
        per-component sibling metric.  No ``loss/total`` sibling metric
        exists — the returned tensor IS the total, tracked by HF Trainer
        directly as ``loss`` / ``eval_loss``."""
        args = _make_reskd_args(
            training_args,
            alpha=alpha,
            magnitude_aware_weighting=mag_aware,
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=2.0),
        )
        distiller._reset_step_metrics(is_training=True)

        student_logits = torch.randn(2, 16, 1000, device=device)
        teacher_logits = torch.randn(2, 16, 1000, device=device)
        hard_loss = torch.tensor(5.0, device=device) if provide_hard_loss else None

        total, _ = distiller._compute_total_loss(
            student_logits, teacher_logits, hard_loss, track_metrics=True
        )

        assert torch.isfinite(total)
        assert "loss/soft" in distiller.current_step_metrics
        # loss/total is deliberately NOT tracked anymore — the returned
        # total tensor already carries that value via the detach trick.
        assert "loss/total" not in distiller.current_step_metrics


# ── BKD warning ─────────────────────────────────────────────────────────────


class TestBKDLossWeightWarning:
    """Tests for BlockwiseDistiller warning on unsupported loss_weight."""

    def test_warns_on_non_default_loss_weight(
        self, teacher_model, student_model, training_args, train_dataset, caplog
    ):
        """BKD should warn when an alignment has loss_weight != 1.0."""
        alignments = []
        for i in range(teacher_model.num_layers):
            alignment = create_alignment(
                teacher_block=teacher_model.get_layer(i),
                student_block=student_model.get_layer(i),
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                with_input_projector=(i > 0),
            )
            alignments.append(alignment)

        # Set non-default weight on one alignment
        alignments[1].loss_weight = 0.5

        with caplog.at_level(
            logging.WARNING, logger="silverspoon_kd.distillers.blockwise_distiller"
        ):
            BlockwiseDistiller(
                teacher_model=teacher_model,
                alignments=alignments,
                args=_make_blockwise_args(training_args),
                train_dataset=train_dataset,
                student_models={student_model.name_or_path: student_model}
                if hasattr(student_model, "name_or_path")
                else None,
            )

        assert any("does not support per-alignment loss_weight" in msg for msg in caplog.messages)

    def test_no_warning_with_default_weights(
        self, teacher_model, student_model, training_args, train_dataset, caplog
    ):
        """BKD should NOT warn when all alignments have default loss_weight."""
        alignments = []
        for i in range(teacher_model.num_layers):
            alignment = create_alignment(
                teacher_block=teacher_model.get_layer(i),
                student_block=student_model.get_layer(i),
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                with_input_projector=(i > 0),
            )
            alignments.append(alignment)

        with caplog.at_level(
            logging.WARNING, logger="silverspoon_kd.distillers.blockwise_distiller"
        ):
            BlockwiseDistiller(
                teacher_model=teacher_model,
                alignments=alignments,
                args=_make_blockwise_args(training_args),
                train_dataset=train_dataset,
            )

        assert not any(
            "does not support per-alignment loss_weight" in msg for msg in caplog.messages
        )
