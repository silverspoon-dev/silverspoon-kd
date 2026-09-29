"""
Unit tests for ResponseBasedDistiller class.
"""

import contextlib
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from silverspoon_kd.distillers.response_based_distiller import ResponseBasedDistiller
from silverspoon_kd.losses.kl import jsd_loss, kl_divergence_loss
from silverspoon_kd.training_arguments import TrainingArguments
from tests.silverspoon_kd.conftest import load_module_copy


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


class TestResponseBasedDistiller:
    """Test suite for ResponseBasedDistiller."""

    def test_initialization(self, teacher_model, student_model, training_args, train_dataset):
        """Test that ResponseBasedDistiller initializes correctly."""
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=4.0),
        )

        assert distiller.teacher_model is teacher_model
        assert distiller.student_model is student_model
        assert distiller.args.alpha == 0.5
        assert distiller.soft_loss_fn is not None

    def test_initialization_default_args(self, teacher_model, student_model, train_dataset, device):
        """Test that default TrainingArguments are created when none provided."""
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=TrainingArguments(use_cpu=(device.type == "cpu")),
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.args, TrainingArguments)
        assert distiller.args.alpha == 0.0
        assert distiller.soft_loss_fn is not None

    def test_initialization_pure_distillation(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test initialization with alpha=0 (pure distillation)."""
        args = _make_reskd_args(training_args, alpha=0.0)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        assert distiller.args.alpha == 0.0

    def test_teacher_gradients_disabled(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that teacher model has gradients disabled."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        # All teacher parameters should not require grad
        for param in distiller.teacher_model.parameters():
            assert param.requires_grad is False

    def test_extract_logits_from_tensor(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that _extract_logits works with raw tensors."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        logits = torch.randn(2, 16, 1000)
        result = distiller._extract_logits(logits)

        assert result is logits

    def test_extract_logits_from_output_object(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that _extract_logits works with objects having logits attribute."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        logits = torch.randn(2, 16, 1000)
        output = MockOutput(logits)
        result = distiller._extract_logits(output)

        assert torch.equal(result, logits)

    def test_extract_logits_invalid_type(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that _extract_logits raises error for invalid types."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        with pytest.raises(ValueError, match="Cannot extract logits"):
            distiller._extract_logits("invalid")

    def test_extract_logits_empty_iterable(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that _extract_logits raises clear error for empty iterables."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        with pytest.raises(ValueError, match="empty iterable"):
            distiller._extract_logits([])
        with pytest.raises(ValueError, match="empty iterable"):
            distiller._extract_logits(())
        with pytest.raises(ValueError, match="empty iterable"):
            distiller._extract_logits(map(str, []))

    def test_soft_loss_fn_no_chunking(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test that soft_loss_fn works without chunking."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(chunk_size=0),
        )

        student_logits = torch.randn(2, 16, 1000, device=device)
        teacher_logits = torch.randn(2, 16, 1000, device=device)

        loss = distiller.soft_loss_fn(student_logits, teacher_logits)

        # Should return a scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_soft_loss_fn_with_chunking(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test that soft_loss_fn works with chunking."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(chunk_size=8),
        )

        # Create logits with 32 tokens (4 chunks of 8)
        student_logits = torch.randn(2, 16, 1000, device=device)  # 32 tokens total
        teacher_logits = torch.randn(2, 16, 1000, device=device)

        loss = distiller.soft_loss_fn(student_logits, teacher_logits)

        # Should return a scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_soft_loss_fn_temperature_scaling(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test that temperature affects soft loss computation."""
        student_logits = torch.randn(2, 16, 1000, device=device)
        teacher_logits = torch.randn(2, 16, 1000, device=device)

        # Compute with temperature=1
        args1 = _make_reskd_args(training_args)
        distiller1 = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args1,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=1.0),
        )
        loss1 = distiller1.soft_loss_fn(student_logits, teacher_logits)

        # Compute with temperature=4
        args2 = _make_reskd_args(training_args)
        distiller2 = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args2,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(temperature=4.0),
        )
        loss2 = distiller2.soft_loss_fn(student_logits, teacher_logits)

        # Losses should be different (temperature affects the loss magnitude)
        assert not torch.isclose(loss1, loss2)

    def test_hard_loss_from_model_output(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test that hard loss comes from model output when alpha > 0."""
        args = _make_reskd_args(training_args, alpha=1.0)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        loss = distiller.training_step(student_model, batch)

        # Should return a scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # With alpha=1.0, loss should be dominated by hard loss
        assert "loss/hard" in distiller.current_step_metrics

    def test_hard_loss_propagation_3d_logits(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test that model.loss with 3D logits (batch, seq_len, vocab) propagates correctly.

        The distiller does not reshape logits itself; the model computes its
        own loss.  Verify that loss is used and produces correct gradient flow.
        """
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        distiller.training_step(student_model, batch)

        # Should have both soft and hard loss
        assert "loss/soft" in distiller.current_step_metrics
        assert "loss/hard" in distiller.current_step_metrics

        # Hard loss should be a reasonable CE value (not zero, not huge)
        hard = distiller.current_step_metrics["loss/hard"]
        assert 0 < hard.item() < 20, f"hard loss {hard.item()} out of expected range"

        # Gradients should exist (loss flowed back through both soft and hard)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in student_model.parameters()
        )
        assert has_grad, "No gradients — hard loss may not have propagated"

    def test_training_step_pure_distillation(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test training_step with pure distillation (alpha=0)."""
        args = _make_reskd_args(training_args, alpha=0.0)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(student_model, batch)

        # Should return a scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Should have soft loss metric
        assert "loss/soft" in distiller.current_step_metrics

        # Total loss equals soft loss when alpha=0
        assert abs(loss.item() - distiller.current_step_metrics["loss/soft"]) < 1e-4

        # Should not have hard loss metric
        assert "loss/hard" not in distiller.current_step_metrics

    def test_training_step_with_hard_labels(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test training_step with both soft and hard losses (alpha > 0)."""
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        loss = distiller.training_step(student_model, batch)

        # Should return a scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Should have soft and hard loss metrics
        assert "loss/soft" in distiller.current_step_metrics
        assert "loss/hard" in distiller.current_step_metrics

        # Returned loss should be the weighted combination
        # (train/loss is logged by the Trainer during the full training loop)
        soft_loss = distiller.current_step_metrics["loss/soft"]
        hard_loss = distiller.current_step_metrics["loss/hard"]

        expected_total = (1 - 0.5) * soft_loss + 0.5 * hard_loss
        assert abs(loss.item() - expected_total) < 1e-4

    def test_compute_loss_evaluation(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test compute_loss during evaluation."""
        args = _make_reskd_args(training_args, alpha=0.0)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.compute_loss(student_model, batch)

        # Should return a scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Test with return_outputs
        loss, outputs = distiller.compute_loss(student_model, batch, return_outputs=True)
        assert isinstance(loss, torch.Tensor)
        assert outputs is None

    def test_prediction_step(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test prediction_step."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss, logits, labels = distiller.prediction_step(
            student_model, batch, prediction_loss_only=True
        )

        # Should return loss and None for logits and labels
        assert isinstance(loss, torch.Tensor)
        assert logits is None
        assert labels is None

    def test_prepare_teacher_inputs_custom(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test custom prepare_teacher_inputs function."""

        def custom_prepare(inputs):
            return {**inputs, "teacher_key": "teacher_value"}

        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            prepare_teacher_inputs=custom_prepare,
        )

        inputs = {"input_ids": torch.tensor([1, 2, 3])}
        result = distiller._prepare_teacher_inputs(inputs)

        assert "teacher_key" in result
        assert result["teacher_key"] == "teacher_value"

    def test_prepare_student_inputs_custom(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test custom prepare_student_inputs function."""

        def custom_prepare(inputs):
            return {**inputs, "student_key": "student_value"}

        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            prepare_student_inputs=custom_prepare,
        )

        inputs = {"input_ids": torch.tensor([1, 2, 3])}
        result = distiller._prepare_student_inputs(inputs)

        assert "student_key" in result
        assert result["student_key"] == "student_value"

    def test_custom_soft_loss_fn(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test that custom soft_loss_fn is used."""

        def custom_loss(student_logits, teacher_logits):
            return torch.tensor(42.0, device=student_logits.device)

        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=custom_loss,
        )

        student_logits = torch.randn(2, 16, 1000, device=device)
        teacher_logits = torch.randn(2, 16, 1000, device=device)

        loss = distiller.soft_loss_fn(student_logits, teacher_logits)

        # Should use custom loss function
        assert loss.item() == 42.0

    def test_evaluation_with_dataset(
        self, teacher_model, student_model, training_args, train_dataset, eval_dataset
    ):
        """Test that evaluation works with dataset."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Run evaluation
        metrics = distiller.evaluate()

        # Should return metrics dictionary
        assert isinstance(metrics, dict)
        assert "eval_loss" in metrics

    def test_chunking_consistency(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test that chunked and non-chunked computations give similar results."""
        student_logits = torch.randn(2, 16, 1000, device=device)
        teacher_logits = torch.randn(2, 16, 1000, device=device)

        # Compute without chunking
        loss_fn_no_chunk = kl_divergence_loss(chunk_size=0)
        loss_no_chunk = loss_fn_no_chunk(student_logits, teacher_logits)

        # Compute with chunking
        loss_fn_chunked = kl_divergence_loss(chunk_size=16)
        loss_chunked = loss_fn_chunked(student_logits, teacher_logits)

        # Should be approximately equal (small numerical differences are acceptable)
        assert torch.isclose(loss_no_chunk, loss_chunked, rtol=1e-4, atol=1e-5)

    def test_alpha_weighting(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test that alpha correctly weights soft and hard losses."""
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        # Test with alpha=0 (pure soft)
        args_soft = _make_reskd_args(training_args, alpha=0.0)
        distiller_soft = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args_soft,
            train_dataset=train_dataset,
        )
        loss_soft = distiller_soft.training_step(student_model, batch)

        # Test with alpha=1 (pure hard)
        args_hard = _make_reskd_args(training_args, alpha=1.0)
        distiller_hard = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args_hard,
            train_dataset=train_dataset,
        )
        loss_hard = distiller_hard.training_step(student_model, batch)

        # Test with alpha=0.5 (mixed)
        args_mixed = _make_reskd_args(training_args, alpha=0.5)
        distiller_mixed = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args_mixed,
            train_dataset=train_dataset,
        )
        loss_mixed = distiller_mixed.training_step(student_model, batch)

        # All losses should be different
        assert not torch.isclose(loss_soft, loss_hard)
        assert not torch.isclose(loss_soft, loss_mixed)
        assert not torch.isclose(loss_hard, loss_mixed)


class TestResponseBasedDistillerAutoDeviceDtypeMatch:
    """Test suite for auto_device_match and auto_dtype_match in ResponseBasedDistiller."""

    def test_auto_dtype_match_initialization(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that auto_dtype_match parameter is stored correctly."""
        args = _make_reskd_args(training_args, auto_dtype_match=True)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )
        assert distiller.args.auto_dtype_match is True

        args2 = _make_reskd_args(training_args, auto_dtype_match=False)
        distiller2 = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args2,
            train_dataset=train_dataset,
        )
        assert distiller2.args.auto_dtype_match is False

    def test_auto_device_match_initialization(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that auto_device_match parameter is stored correctly."""
        args = _make_reskd_args(training_args, auto_device_match=True)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )
        assert distiller.args.auto_device_match is True

        args2 = _make_reskd_args(training_args, auto_device_match=False)
        distiller2 = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args2,
            train_dataset=train_dataset,
        )
        assert distiller2.args.auto_device_match is False

    def test_training_step_with_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Test training_step with teacher float32 and student bfloat16."""
        # Create student in bfloat16
        from tests.silverspoon_kd.conftest import SimpleModel

        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(
            device=device, dtype=torch.bfloat16
        )

        # Teacher stays float32
        assert next(teacher_model.parameters()).dtype == torch.float32
        assert next(student_model.parameters()).dtype == torch.bfloat16

        args = _make_reskd_args(training_args, auto_dtype_match=True)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Should succeed without dtype mismatch error
        loss = distiller.training_step(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_training_step_without_auto_dtype_match(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Test training_step works without auto_dtype_match (implicit PyTorch casting)."""
        from tests.silverspoon_kd.conftest import SimpleModel

        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(
            device=device, dtype=torch.bfloat16
        )

        args = _make_reskd_args(training_args, auto_dtype_match=False)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # May succeed due to implicit PyTorch casting, or fail on some backends
        # The point of auto_dtype_match is to ensure explicit, predictable behavior
        loss = distiller.training_step(student_model, batch)
        assert isinstance(loss, torch.Tensor)

    def test_compute_loss_with_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Test compute_loss (eval) with teacher float32 and student bfloat16."""
        from tests.silverspoon_kd.conftest import SimpleModel

        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(
            device=device, dtype=torch.bfloat16
        )

        args = _make_reskd_args(training_args, auto_dtype_match=True)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Should succeed without dtype mismatch error
        loss = distiller.compute_loss(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_training_step_with_hard_labels_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Test training_step with hard labels and dtype mismatch."""
        from tests.silverspoon_kd.conftest import SimpleModel

        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(
            device=device, dtype=torch.bfloat16
        )

        args = _make_reskd_args(training_args, auto_dtype_match=True, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        # Should succeed without dtype mismatch error
        loss = distiller.training_step(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Both soft and hard loss should be logged
        assert "loss/soft" in distiller.current_step_metrics
        assert "loss/hard" in distiller.current_step_metrics

    def test_combined_device_and_dtype_match(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Test that both auto_device_match and auto_dtype_match work together."""
        from tests.silverspoon_kd.conftest import SimpleModel

        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(
            device=device, dtype=torch.bfloat16
        )

        args = _make_reskd_args(training_args, auto_device_match=True, auto_dtype_match=True)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Should succeed with both flags enabled
        loss = distiller.training_step(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestResponseBasedDistillerProfiler:
    """Test profiler methods in ResponseBasedDistiller."""

    def test_init_profiler_when_enabled(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        """Test _init_profiler creates a profiler when profiling is enabled."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=5,
            enable_profiling=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        assert distiller.profiler is not None

    def test_profiler_step_with_profiler(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        """Test _profiler_step calls step() on profiler."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=5,
            enable_profiling=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        # Start profiler, step, stop
        distiller.profiler.__enter__()
        step_before = distiller.profiler.step_num
        distiller._profiler_step()
        step_after = distiller.profiler.step_num
        distiller.profiler.__exit__(None, None, None)

        assert step_after == step_before + 1


class TestResponseBasedDistillerFlopCounting:
    """Test FLOP counting methods in ResponseBasedDistiller."""

    def test_should_count_flops_enabled(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        """Test _should_count_flops returns True when enabled on step 0."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            count_flops=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        distiller.state.global_step = 0
        assert distiller._should_count_flops() is True

    def test_get_flop_context_count_now(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test _get_flop_context with count_now=True returns FlopCounterMode."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        ctx = distiller._get_flop_context(count_now=True)
        assert isinstance(ctx, FlopCounterMode)

    def test_get_flop_context_no_count(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test _get_flop_context with count_now=False returns nullcontext."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        ctx = distiller._get_flop_context(count_now=False)
        assert isinstance(ctx, contextlib.nullcontext)

    def test_record_flops_with_flop_counter(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test _record_flops records flops from FlopCounterMode."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        mock_counter = MagicMock(spec=FlopCounterMode)
        mock_counter.get_total_flops.return_value = 5000

        distiller._record_flops(mock_counter)
        assert distiller.flops_per_step == 5000

    def test_update_flop_counter_enabled(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        """Test _update_flop_counter increments when counting is enabled."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            count_flops=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        distiller.flops_per_step = 2000
        distiller._update_flop_counter()
        assert distiller.flop_counter == 2000


class TestResponseBasedDistillerTeacherInputs:
    """Test teacher input auto-filter in ResponseBasedDistiller."""

    def test_auto_filter_without_kwargs(self, student_model, training_args, train_dataset, device):
        """Test auto-filter when teacher does NOT accept **kwargs."""

        class StrictTeacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return self.linear(input_ids.float())

        teacher = StrictTeacher().to(device)

        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher,
            args=args,
            train_dataset=train_dataset,
        )

        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([0, 1, 2]),
        }
        result = distiller._prepare_teacher_inputs(inputs)

        # Should filter out labels
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" not in result
        assert result["use_cache"] is False

    def test_auto_filter_with_kwargs_teacher(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test auto-filter when teacher accepts **kwargs strips labels but passes rest."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "extra_key": "value",
        }
        result = distiller._prepare_teacher_inputs(inputs)

        # Teacher has **kwargs — non-label keys pass through
        assert "input_ids" in result
        assert "extra_key" in result

    def test_get_teacher_accepted_params_caching(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that _get_teacher_accepted_params caches result."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        result1 = distiller._get_teacher_accepted_params()
        result2 = distiller._get_teacher_accepted_params()
        assert result1 is result2


class TestResponseBasedDistillerTrainMethod:
    """Test train() method in ResponseBasedDistiller."""

    def test_train_with_profiler_lifecycle(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        """Test that train() manages profiler lifecycle."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=2,
            enable_profiling=True,
            profiling_wait=0,
            profiling_warmup=1,
            profiling_active=1,
            use_cpu=(device.type == "cpu"),
        )
        # Prevent DataParallel wrapping on multi-GPU machines
        args._n_gpu = 1

        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        assert distiller.profiler is not None

        # Train should enter and exit profiler context
        result = distiller.train()
        assert result is not None

    def test_train_resets_flop_counter(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that train() resets flop counter."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        distiller.flop_counter = 999
        distiller.train()
        assert distiller.flop_counter == 0


class TestResponseBasedDistillerComputeLossHardLabels:
    """Test compute_loss with hard labels in eval mode."""

    def test_compute_loss_with_alpha_and_labels(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test compute_loss in eval mode with alpha > 0 and labels."""
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        loss = distiller.compute_loss(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestResponseBasedDistillerLogOverride:
    """Test log() method override in ResponseBasedDistiller."""

    def test_log_injects_averaged_training_metrics(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test that log() injects averaged training metrics from accumulator."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        # Simulate accumulated training metrics (from _track_metric calls)
        # Running sums over 2 steps: soft 0.5+0.7=1.2, hard 0.3+0.5=0.8
        distiller._training_metric_accumulator = {
            "loss/soft": 1.2,
            "loss/hard": 0.8,
        }
        distiller._metric_accumulator_steps = 2

        captured_logs = None
        from transformers import Trainer

        original_log = Trainer.log

        def capture_logs(self_arg, logs_dict, start_time=None):
            nonlocal captured_logs
            captured_logs = logs_dict

        Trainer.log = capture_logs

        try:
            distiller.log({"train_loss": 0.4})

            assert captured_logs is not None
            assert "loss/soft" in captured_logs
            assert abs(captured_logs["loss/soft"] - 0.6) < 1e-6
            assert "loss/hard" in captured_logs
            assert abs(captured_logs["loss/hard"] - 0.4) < 1e-6
            assert "train_loss" in captured_logs
        finally:
            Trainer.log = original_log


class TestResponseBasedDistillerWeightWatcher:
    """Test WeightWatcher integration in ResponseBasedDistiller."""

    def test_evaluate_with_weightwatcher(
        self, teacher_model, student_model, training_args, train_dataset, eval_dataset
    ):
        """Test evaluate with mocked WeightWatcher."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        distiller.args.use_weightwatcher = True

        with patch(
            "silverspoon_kd.distillers.response_based_distiller.WEIGHTWATCHER_AVAILABLE",
            True,
        ):
            mock_ww = MagicMock()
            mock_details = MagicMock()
            mock_summary = {"alpha": 2.5, "log_norm": 1.0}
            mock_ww.analyze.return_value = mock_details
            mock_ww.get_summary.return_value = mock_summary

            mock_ww_module = MagicMock()
            mock_ww_module.WeightWatcher.return_value = mock_ww
            with patch.dict("sys.modules", {"weightwatcher": mock_ww_module}):
                metrics = distiller.evaluate()

            assert isinstance(metrics, dict)
            # WW metrics should be included
            assert any("ww_alpha" in k for k in metrics)

    def test_run_weightwatcher_analysis_handles_exception(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Test _run_weightwatcher_analysis handles exceptions gracefully."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        # weightwatcher is imported lazily inside the method, so patch it
        # in sys.modules so the lazy import picks up the mock.
        mock_ww = MagicMock()
        mock_ww.WeightWatcher.side_effect = RuntimeError("WW failed")
        with patch.dict(sys.modules, {"weightwatcher": mock_ww}):
            metrics = distiller._run_weightwatcher_analysis("eval")

        assert metrics == {}


class TestResponseBasedDistillerImportFallback:
    """Test import error fallback for optional dependencies."""

    def test_weightwatcher_import_error(self):
        """Test WEIGHTWATCHER_AVAILABLE is False when weightwatcher is not installed."""
        import silverspoon_kd.distillers.base_distiller as base_mod
        import silverspoon_kd.distillers.response_based_distiller as mod

        # The flag is defined in base_distiller and re-exported here.
        assert mod.WEIGHTWATCHER_AVAILABLE is base_mod.WEIGHTWATCHER_AVAILABLE
        with patch.dict(sys.modules, {"weightwatcher": None}):
            probe = load_module_copy(base_mod)
        assert probe.WEIGHTWATCHER_AVAILABLE is False


class TestResponseBasedDistillerDefaultArgsNone:
    """Test default TrainingArguments creation when args is None."""

    def test_initialization_with_none_args(self, teacher_model, student_model, train_dataset):
        """Test that ResponseBasedDistiller creates default args when args is None."""
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=None,
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.args, TrainingArguments)


class TestResponseBasedDistillerComputeLossAutoDevice:
    """Test auto_device_match in compute_loss (eval path)."""

    def test_compute_loss_with_auto_device_match(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        """Test compute_loss applies auto_device_match during evaluation."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            use_cpu=(device.type == "cpu"),
            auto_device_match=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.compute_loss(student_model, batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestResponseBasedDistillerPredictionStepTuple:
    """Test prediction_step handling of tuple returns from compute_loss."""

    def test_prediction_step_with_tuple_loss(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """Test prediction_step extracts loss from tuple return."""
        args = _make_reskd_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        expected_loss = torch.tensor(2.0, device=device)
        with patch.object(distiller, "compute_loss", return_value=(expected_loss, None)):
            loss, logits, labels = distiller.prediction_step(student_model, batch, True)

        assert loss.item() == 2.0
        assert logits is None
        assert labels is None


class TestResponseBasedDistillerMissingLabels:
    """Test behavior when labels are absent from batches (alpha > 0)."""

    def test_hard_loss_missing_without_labels(
        self, teacher_model, student_model, training_args, train_dataset, device
    ):
        """loss/hard should NOT appear in current_step_metrics when labels missing."""
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            # No "labels" key
        }

        distiller.training_step(student_model, batch)

        assert "loss/soft" in distiller.current_step_metrics
        assert "loss/hard" not in distiller.current_step_metrics

    def test_warning_when_labels_missing_with_alpha(
        self, teacher_model, student_model, training_args, train_dataset, device, caplog
    ):
        """A warning should be logged when alpha > 0 but no labels in batch."""
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        with caplog.at_level(logging.WARNING):
            distiller.training_step(student_model, batch)

        assert any("no hard loss available" in msg for msg in caplog.messages)

    def test_warning_only_once(
        self, teacher_model, student_model, training_args, train_dataset, device, caplog
    ):
        """The missing-labels warning should only be emitted once."""
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        with caplog.at_level(logging.WARNING):
            distiller.training_step(student_model, batch)
            distiller.training_step(student_model, batch)

        warning_count = sum(1 for msg in caplog.messages if "no hard loss available" in msg)
        assert warning_count == 1

    def test_warning_when_model_does_not_compute_loss(
        self, teacher_model, training_args, train_dataset, device, caplog
    ):
        """Warning when model ignores labels and doesn't return .loss."""
        from collections import namedtuple

        LosslessOutput = namedtuple("LosslessOutput", ["logits", "loss"])

        class LosslessModel(nn.Module):
            """Model that returns logits but never computes loss."""

            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(128, 128)
                self.head = nn.Linear(128, 128)

            def forward(self, input_ids, **kwargs):
                x = self.embedding(input_ids)
                logits = self.head(x)
                return LosslessOutput(logits=logits, loss=None)

        student = LosslessModel().to(device)
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )

        # Batch WITH labels — but model ignores them
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        with caplog.at_level(logging.WARNING):
            loss = distiller.training_step(student, batch)

        # Should still produce a valid loss (soft-only)
        assert torch.isfinite(loss)
        # Should warn that hard loss is missing
        assert any("no hard loss available" in msg for msg in caplog.messages)
        # loss/hard should NOT be in metrics
        assert "loss/hard" not in distiller.current_step_metrics


class TestResponseBasedDistillerMetricVisibility:
    """Reproduce and verify that loss/soft and loss/hard appear in all expected places.

    These tests check the full Trainer pipeline (train/evaluate), not just
    current_step_metrics, to catch issues where metrics are computed but
    never make it into log_history or eval results.
    """

    def _make_args(self, tmp_path, device, **overrides):
        kwargs = {
            "output_dir": str(tmp_path / "output"),
            "max_steps": 5,
            "logging_steps": 1,
            "per_device_train_batch_size": 2,
            "dataloader_num_workers": 0,
            "report_to": [],
            "use_cpu": (device.type == "cpu"),
        }
        kwargs.update(overrides)
        args = TrainingArguments(**kwargs)
        args._n_gpu = 1
        return args

    def test_train_loss_soft_in_log_history(
        self,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        device,
        tmp_path,
    ):
        """train/loss/soft must appear in log_history (alpha=0, pure distillation)."""
        args = self._make_args(tmp_path, device, alpha=0.0)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )
        distiller.train()

        training_entries = [
            e for e in distiller.state.log_history if "loss" in e and "eval_loss" not in e
        ]
        assert len(training_entries) > 0, "No training log entries found"
        assert all("loss/soft" in e for e in training_entries), (
            f"loss/soft missing from training entries: {training_entries}"
        )
        # With alpha=0, loss/hard should NOT appear
        assert not any("loss/hard" in e for e in training_entries), (
            f"loss/hard should not appear with alpha=0: {training_entries}"
        )
        # alpha=0 ⇒ loss is 100% soft, so loss/soft must match loss
        for entry in training_entries:
            assert abs(entry["loss/soft"] - entry["loss"]) < 0.05, (
                f"With alpha=0, loss/soft ({entry['loss/soft']}) should "
                f"match loss ({entry['loss']})"
            )

    def test_train_loss_hard_in_log_history(
        self,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        device,
        tmp_path,
    ):
        """train/loss/hard must appear in log_history when alpha > 0 and labels present."""
        args = self._make_args(tmp_path, device, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )
        distiller.train()

        training_entries = [
            e for e in distiller.state.log_history if "loss" in e and "eval_loss" not in e
        ]
        assert len(training_entries) > 0, "No training log entries found"
        assert all("loss/hard" in e for e in training_entries), (
            f"loss/hard missing from training entries: {training_entries}"
        )
        assert all("loss/soft" in e for e in training_entries), (
            f"loss/soft missing from training entries: {training_entries}"
        )
        # alpha=0.5 ⇒ loss = 0.5*soft + 0.5*hard
        for entry in training_entries:
            expected = 0.5 * entry["loss/soft"] + 0.5 * entry["loss/hard"]
            assert abs(entry["loss"] - expected) < 0.05, (
                f"loss ({entry['loss']}) should equal "
                f"0.5*soft ({entry['loss/soft']}) + "
                f"0.5*hard ({entry['loss/hard']}) = {expected}"
            )

    def test_eval_loss_soft_in_eval_metrics(
        self,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        device,
        tmp_path,
    ):
        """eval metrics should include loss/soft breakdown with correct value."""
        args = self._make_args(tmp_path, device, eval_strategy="steps", eval_steps=5, alpha=0.0)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        metrics = distiller.evaluate()

        assert "eval_loss" in metrics, f"eval_loss missing: {metrics}"
        assert "eval_loss/soft" in metrics, f"eval_loss/soft missing from eval metrics: {metrics}"
        # alpha=0 ⇒ eval_loss is 100% soft, so they must be close.
        # Minor discrepancy expected when the last eval batch is smaller: the
        # Trainer weights losses per-sample (via losses.repeat(batch_size))
        # while our component averages weight equally per-batch (sum/len).
        # A divisor bug would cause a multi-x discrepancy, so 0.05 catches it.
        assert abs(metrics["eval_loss/soft"] - metrics["eval_loss"]) < 0.05, (
            f"With alpha=0, eval_loss/soft ({metrics['eval_loss/soft']}) should "
            f"be close to eval_loss ({metrics['eval_loss']})"
        )

    def test_eval_loss_hard_in_eval_metrics(
        self,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        device,
        tmp_path,
    ):
        """eval metrics should include loss/hard breakdown with correct values."""
        args = self._make_args(tmp_path, device, eval_strategy="steps", eval_steps=5, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        metrics = distiller.evaluate()

        assert "eval_loss" in metrics, f"eval_loss missing: {metrics}"
        assert "eval_loss/soft" in metrics, f"eval_loss/soft missing from eval metrics: {metrics}"
        assert "eval_loss/hard" in metrics, f"eval_loss/hard missing from eval metrics: {metrics}"
        # alpha=0.5 ⇒ eval_loss ≈ 0.5*soft + 0.5*hard.
        # Minor discrepancy expected when the last eval batch is smaller: the
        # Trainer weights losses per-sample (via losses.repeat(batch_size))
        # while our component averages weight equally per-batch (sum/len).
        # A divisor bug would cause a multi-x discrepancy, so 0.05 catches it.
        expected = 0.5 * metrics["eval_loss/soft"] + 0.5 * metrics["eval_loss/hard"]
        assert abs(metrics["eval_loss"] - expected) < 0.05, (
            f"eval_loss ({metrics['eval_loss']}) should be close to "
            f"0.5*soft ({metrics['eval_loss/soft']}) + "
            f"0.5*hard ({metrics['eval_loss/hard']}) = {expected}"
        )

    def test_eval_metrics_in_log_history_during_training(
        self,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        device,
        tmp_path,
    ):
        """eval/loss/soft and eval/loss/hard should appear in log_history when eval runs during training."""
        args = self._make_args(tmp_path, device, eval_strategy="steps", eval_steps=5, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        distiller.train()

        eval_entries = [e for e in distiller.state.log_history if "eval_loss" in e]
        assert len(eval_entries) > 0, (
            f"No eval log entries found. Full log_history: {distiller.state.log_history}"
        )
        assert any("eval_loss/soft" in e for e in eval_entries), (
            f"eval_loss/soft missing from eval entries: {eval_entries}"
        )
        assert any("eval_loss/hard" in e for e in eval_entries), (
            f"eval_loss/hard missing from eval entries: {eval_entries}"
        )
        # Verify component values are consistent with combined loss.
        # Tolerance of 0.05 accounts for per-sample vs per-batch weighting
        # difference when the last eval batch is smaller.
        for entry in eval_entries:
            if "eval_loss/soft" in entry and "eval_loss/hard" in entry:
                expected = 0.5 * entry["eval_loss/soft"] + 0.5 * entry["eval_loss/hard"]
                assert abs(entry["eval_loss"] - expected) < 0.05, (
                    f"eval_loss ({entry['eval_loss']}) should be close to "
                    f"0.5*soft ({entry['eval_loss/soft']}) + "
                    f"0.5*hard ({entry['eval_loss/hard']}) = {expected}"
                )

    def test_no_stale_training_metrics_in_eval_entry(
        self,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        device,
        tmp_path,
    ):
        """Eval log entries must NOT contain bare loss/soft or loss/hard from training."""
        args = self._make_args(tmp_path, device, eval_strategy="steps", eval_steps=5, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        distiller.train()

        eval_entries = [e for e in distiller.state.log_history if "eval_loss" in e]
        assert len(eval_entries) > 0
        for entry in eval_entries:
            # These are training metrics that should NOT leak into eval entries
            assert "loss/soft" not in entry or "eval_loss/soft" in entry, (
                f"Stale training loss/soft leaked into eval entry: {entry}"
            )
            assert "loss/hard" not in entry or "eval_loss/hard" in entry, (
                f"Stale training loss/hard leaked into eval entry: {entry}"
            )

    def test_compute_loss_accumulates_component_losses(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        eval_dataset,
        device,
    ):
        """compute_loss should accumulate soft/hard losses in _eval_component_losses."""
        args = _make_reskd_args(training_args, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Simulate what evaluate() does: set up the accumulator
        distiller._eval_component_losses = {"soft": [], "hard": []}

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        distiller.compute_loss(student_model, batch)
        distiller.compute_loss(student_model, batch)

        assert len(distiller._eval_component_losses["soft"]) == 2
        assert len(distiller._eval_component_losses["hard"]) == 2
        assert all(v > 0 for v in distiller._eval_component_losses["soft"])
        assert all(v > 0 for v in distiller._eval_component_losses["hard"])

    def test_compute_loss_no_hard_accumulation_without_alpha(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        eval_dataset,
        device,
    ):
        """compute_loss should not accumulate hard losses when alpha=0."""
        args = _make_reskd_args(training_args, alpha=0.0)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        distiller._eval_component_losses = {"soft": [], "hard": []}

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        distiller.compute_loss(student_model, batch)

        assert len(distiller._eval_component_losses["soft"]) == 1
        assert len(distiller._eval_component_losses["hard"]) == 0

    def test_eval_with_custom_metric_prefix(
        self,
        teacher_model,
        student_model,
        train_dataset,
        eval_dataset,
        device,
        tmp_path,
    ):
        """Component metrics should use the custom metric_key_prefix."""
        args = self._make_args(tmp_path, device, alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        metrics = distiller.evaluate(metric_key_prefix="test")

        assert "test_loss" in metrics, f"test_loss missing: {metrics}"
        assert "test_loss/soft" in metrics, f"test_loss/soft missing: {metrics}"
        assert "test_loss/hard" in metrics, f"test_loss/hard missing: {metrics}"
        # Should NOT have eval_ prefix
        assert "eval_loss/soft" not in metrics
        # Component values must be consistent with combined loss.
        # Tolerance of 0.05 accounts for per-sample vs per-batch weighting
        # difference when the last eval batch is smaller.
        expected = 0.5 * metrics["test_loss/soft"] + 0.5 * metrics["test_loss/hard"]
        assert abs(metrics["test_loss"] - expected) < 0.05, (
            f"test_loss ({metrics['test_loss']}) should be close to "
            f"0.5*soft ({metrics['test_loss/soft']}) + "
            f"0.5*hard ({metrics['test_loss/hard']}) = {expected}"
        )


class TestResponseBasedDistillerTorchCompile:
    """Test torch.compile handling in ResponseBasedDistiller."""

    def test_teacher_compiled_when_torch_compile_true(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        """Test that teacher is compiled when torch_compile=True."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ResponseBasedDistiller(
            teacher_model=teacher_model,
            student_model=student_model,
            args=args,
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.teacher_model, torch._dynamo.eval_frame.OptimizedModule)

    def test_training_works_with_torch_compile(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        """Test that training works after torch.compile is applied."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ResponseBasedDistiller(
            teacher_model=teacher_model,
            student_model=student_model,
            args=args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        loss = distiller.training_step(student_model, batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestResponseBasedDistillerJSD:
    """Tests for JSD soft loss support."""

    def test_jsd_init(self, teacher_model, student_model, train_dataset, device):
        """Test that JSD soft_loss_fn initializes correctly."""
        args = TrainingArguments(
            use_cpu=(device.type == "cpu"),
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=jsd_loss(),
        )
        assert distiller.soft_loss_fn is not None
        assert distiller._liger_loss is None

    def test_invalid_soft_loss_fn_string_raises(
        self, teacher_model, student_model, train_dataset, device
    ):
        """Test that invalid soft_loss_fn string raises an error."""
        args = TrainingArguments(
            use_cpu=(device.type == "cpu"),
        )
        with pytest.raises(ValueError, match=r"Unknown loss type.*invalid_loss_name"):
            ResponseBasedDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                args=args,
                train_dataset=train_dataset,
                soft_loss_fn="invalid_loss_name",
            )

    def test_default_is_kl(self, teacher_model, student_model, train_dataset, device):
        """Test that default soft_loss_fn is KL divergence."""
        args = TrainingArguments(use_cpu=(device.type == "cpu"))
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )
        # Default should be kl_divergence_loss — verify by comparing output
        # against an explicit KL computation
        student_logits = torch.randn(2, 10, device=device)
        teacher_logits = torch.randn(2, 10, device=device)
        default_loss = distiller.soft_loss_fn(student_logits, teacher_logits)
        expected_loss = kl_divergence_loss()(student_logits, teacher_logits)
        assert torch.allclose(default_loss, expected_loss, atol=1e-6)

    def test_jsd_vs_kl_different(self, teacher_model, student_model, train_dataset, device):
        """Test that JSD and KL produce different losses."""
        from tests.silverspoon_kd.conftest import create_batch

        batch = create_batch(device=str(device))

        kl_args = TrainingArguments(
            use_cpu=(device.type == "cpu"),
        )
        kl_distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=kl_args,
            train_dataset=train_dataset,
            soft_loss_fn=kl_divergence_loss(),
        )

        jsd_args = TrainingArguments(
            use_cpu=(device.type == "cpu"),
        )
        jsd_distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=jsd_args,
            train_dataset=train_dataset,
            soft_loss_fn=jsd_loss(),
        )

        kl_loss_val = kl_distiller.compute_distillation_loss(
            student_model, batch, is_training=False
        )
        jsd_loss_val = jsd_distiller.compute_distillation_loss(
            student_model, batch, is_training=False
        )

        # They should be different values (both finite, non-zero)
        assert torch.isfinite(kl_loss_val) and torch.isfinite(jsd_loss_val)
        assert kl_loss_val.item() != pytest.approx(jsd_loss_val.item(), abs=1e-6)

    def test_jsd_beta_extremes(self, teacher_model, student_model, train_dataset, device):
        """Test JSD at beta extremes (0 and 1)."""
        from tests.silverspoon_kd.conftest import create_batch

        batch = create_batch(device=str(device))

        for beta in [0.01, 0.99]:
            args = TrainingArguments(
                use_cpu=(device.type == "cpu"),
            )
            distiller = ResponseBasedDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                args=args,
                train_dataset=train_dataset,
                soft_loss_fn=jsd_loss(beta=beta),
            )
            loss = distiller.compute_distillation_loss(student_model, batch, is_training=False)
            assert torch.isfinite(loss)

    def test_jsd_chunking_consistency(self, teacher_model, student_model, train_dataset, device):
        """Test that chunked and non-chunked JSD produce similar results."""
        from tests.silverspoon_kd.conftest import create_batch

        batch = create_batch(device=str(device))

        args = TrainingArguments(
            use_cpu=(device.type == "cpu"),
        )

        d1 = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=jsd_loss(chunk_size=0),
        )
        d2 = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=jsd_loss(chunk_size=8),
        )

        loss1 = d1.compute_distillation_loss(student_model, batch, is_training=False)
        loss2 = d2.compute_distillation_loss(student_model, batch, is_training=False)
        assert loss1.item() == pytest.approx(loss2.item(), rel=1e-4)

    def test_jsd_training_step(self, teacher_model, student_model, train_dataset, device, tmp_path):
        """Test that JSD training runs to completion and logs finite losses."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            logging_steps=1,
            dataloader_num_workers=0,
            report_to=[],
            use_cpu=(device.type == "cpu"),
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            soft_loss_fn=jsd_loss(),
        )
        distiller.train()

        assert distiller.state.global_step == 2
        losses = [e["loss"] for e in distiller.state.log_history if "loss" in e]
        assert losses, "no training loss logged"
        assert torch.isfinite(torch.tensor(losses)).all()


class TestResponseBasedDistillerLiger:
    """Tests for Liger fused kernel support."""

    @pytest.fixture(autouse=True)
    def _reset_dynamo(self):
        """Reset torch._dynamo before each test to avoid stale compile caches."""
        import torch._dynamo

        torch._dynamo.reset()

    def test_liger_unavailable_raises(self, teacher_model, student_model, train_dataset, device):
        """Test that use_liger_kernel=True raises ImportError when not installed."""
        from unittest.mock import patch

        args = TrainingArguments(
            use_cpu=(device.type == "cpu"),
        )
        with (
            patch(
                "silverspoon_kd.distillers.response_based_distiller.LIGER_KERNEL_AVAILABLE",
                False,
            ),
            pytest.raises(ImportError, match="liger-kernel"),
        ):
            ResponseBasedDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                args=args,
                train_dataset=train_dataset,
                use_liger_kernel=True,
                output_head_layer="lm_head",
            )

    def test_missing_output_head_layer_raises(
        self, teacher_model, student_model, train_dataset, device
    ):
        """Test that use_liger_kernel=True without output_head_layer raises ValueError."""
        from unittest.mock import patch

        args = TrainingArguments(
            use_cpu=(device.type == "cpu"),
        )
        with (
            patch(
                "silverspoon_kd.distillers.response_based_distiller.LIGER_KERNEL_AVAILABLE",
                True,
            ),
            pytest.raises(ValueError, match="output_head_layer is required"),
        ):
            ResponseBasedDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                args=args,
                train_dataset=train_dataset,
                use_liger_kernel=True,
                # output_head_layer intentionally omitted
            )

    def test_get_output_head_resolves_path(
        self, teacher_model, student_model, train_dataset, device
    ):
        """Test _get_output_head resolves the user-provided attribute path."""
        args = TrainingArguments(use_cpu=(device.type == "cpu"))
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            output_head_layer="lm_head",
        )
        head = distiller._get_output_head(student_model)
        assert head is student_model.lm_head

    def test_get_output_head_none_raises(self, teacher_model, student_model, train_dataset, device):
        """Test _get_output_head raises when output_head_layer is None."""
        args = TrainingArguments(use_cpu=(device.type == "cpu"))
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )
        with pytest.raises(ValueError, match="output_head_layer is required"):
            distiller._get_output_head(student_model)

    def test_extract_hidden_states_via_hook(
        self, teacher_model, student_model, train_dataset, device
    ):
        """Test _extract_hidden_states captures hidden states via hook."""
        from tests.silverspoon_kd.conftest import create_batch

        args = TrainingArguments(use_cpu=(device.type == "cpu"))
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            output_head_layer="lm_head",
        )
        batch = create_batch(device=str(device))
        hidden = distiller._extract_hidden_states(student_model, batch)
        # hidden should be (batch, seq_len, hidden_dim)
        assert hidden.dim() == 3
        assert hidden.shape[0] == batch["input_ids"].shape[0]
        assert hidden.shape[1] == batch["input_ids"].shape[1]
        assert hidden.shape[2] == student_model.hidden_dim

    @pytest.mark.cuda
    def test_fused_kl_numerical_correctness(self, device):
        """FusedLinearKLDivLoss matches manual KL computation."""
        pytest.importorskip("liger_kernel")
        from silverspoon_kd.losses.liger import (
            FusedLinearKLDivLoss,
        )

        temperature = 4.0
        fused_loss = FusedLinearKLDivLoss(temperature=temperature)

        B, S, H, V = 2, 8, 32, 64
        # Liger 0.7.0 expects 2D input: (batch*seq_len, hidden_size)
        student_hidden = torch.randn(B * S, H, device=device, requires_grad=True)
        teacher_hidden = torch.randn(B * S, H, device=device)
        lm_head = nn.Linear(H, V, bias=False).to(device)

        # Fused result — pass valid dummy labels so Liger's per-token
        # normalisation doesn't zero out the loss
        dummy_labels = torch.zeros(B * S, dtype=torch.long, device=device)
        fused_result = fused_loss(
            student_hidden,
            lm_head.weight,
            teacher_hidden,
            lm_head.weight,  # same head for both
            target=dummy_labels,
        )

        # Manual KL computation
        with torch.no_grad():
            teacher_logits = teacher_hidden @ lm_head.weight.T
        student_logits_manual = student_hidden.detach().requires_grad_(True) @ lm_head.weight.T
        student_log_probs = torch.nn.functional.log_softmax(
            student_logits_manual / temperature, dim=-1
        )
        teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        manual_result = torch.nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="sum",
        )

        # The fused kernel normalizes per-token and the base class pre-divides
        # logits by temperature (so the KL gets an implicit 1/T² scaling
        # compared to the manual sum-reduction computation).
        # Verify both are positive and related by a known factor.
        assert fused_result.item() > 0
        assert manual_result.item() > 0
        # Expected ratio: 1/T² (per-token norm) = 1/(4²) = 0.0625 for T=4
        ratio = fused_result.item() / manual_result.item()
        expected_ratio = 1.0 / (temperature**2)
        assert abs(ratio - expected_ratio) / expected_ratio < 0.05, (
            f"ratio={ratio}, expected ~{expected_ratio}"
        )

    @pytest.mark.cuda
    def test_fused_jsd_numerical_correctness(self, device):
        """LigerFusedLinearJSDLoss matches manual JSD computation."""
        pytest.importorskip("liger_kernel")
        from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

        temperature = 2.0
        beta = 0.5
        fused_loss = LigerFusedLinearJSDLoss(beta=beta, temperature=temperature)

        B, S, H, V = 2, 8, 32, 64
        # Liger 0.7.0 expects 2D input: (batch*seq_len, hidden_size)
        student_hidden = torch.randn(B * S, H, device=device, requires_grad=True)
        teacher_hidden = torch.randn(B * S, H, device=device)
        lm_head = nn.Linear(H, V, bias=False).to(device)
        # 0.7.0 requires true_labels; use valid dummy labels (not -100)
        # so Liger's per-token normalisation doesn't zero out the loss
        dummy_labels = torch.zeros(B * S, dtype=torch.long, device=device)

        fused_result = fused_loss(
            student_hidden,
            lm_head.weight,
            teacher_hidden,
            lm_head.weight,
            dummy_labels,
        )

        # Manual JSD computation
        with torch.no_grad():
            teacher_logits = teacher_hidden @ lm_head.weight.T
        student_logits = student_hidden.detach() @ lm_head.weight.T
        s_log = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        t_log = torch.nn.functional.log_softmax(teacher_logits / temperature, dim=-1)
        import math

        log_m = torch.logaddexp(t_log + math.log(beta), s_log + math.log(1 - beta))
        t_probs = t_log.exp()
        s_probs = s_log.exp()
        kl_t_m = torch.nn.functional.kl_div(log_m.view(-1, V), t_probs.view(-1, V), reduction="sum")
        kl_s_m = torch.nn.functional.kl_div(log_m.view(-1, V), s_probs.view(-1, V), reduction="sum")
        manual_result = beta * kl_t_m + (1 - beta) * kl_s_m

        assert fused_result.item() > 0
        assert manual_result.item() > 0
        ratio = fused_result.item() / manual_result.item()
        assert 0.1 < ratio < 10.0, f"ratio={ratio}"

    @pytest.mark.cuda
    def test_compute_fused_loss_integration(
        self, teacher_model, student_model, train_dataset, device
    ):
        """Full integration test calling _compute_fused_loss with a real model."""
        pytest.importorskip("liger_kernel")
        from silverspoon_kd.losses.liger import (
            LIGER_KERNEL_AVAILABLE,
        )
        from tests.silverspoon_kd.conftest import create_batch

        if not LIGER_KERNEL_AVAILABLE:
            pytest.skip("liger-kernel not available at module level")

        args = TrainingArguments(
            use_cpu=(device.type == "cpu"),
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            use_liger_kernel=True,
            output_head_layer="lm_head",
        )
        batch = create_batch(device=str(device))
        teacher_inputs = distiller._prepare_teacher_inputs(batch)
        student_inputs = distiller._prepare_student_inputs(batch)
        loss = distiller._compute_fused_loss(
            student_model, batch, teacher_inputs, student_inputs, is_training=True
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() >= 0


class TestEvalComponentLossAccumulator:
    """The soft/hard component accumulator exists only while evaluate() runs."""

    def test_accumulator_is_cleared_after_evaluate(
        self, teacher_model, student_model, train_dataset, eval_dataset, training_args
    ):
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=_make_reskd_args(training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        assert distiller._eval_component_losses is None

        metrics = distiller.evaluate()

        assert "eval_loss/soft" in metrics
        assert distiller._eval_component_losses is None
        assert distiller._eval_metric_prefix is None


class TestFlopLogging:
    """count_flops=True logs the per-step and cumulative FLOPs with the training metrics."""

    def test_flops_appear_in_log_history(
        self, teacher_model, student_model, train_dataset, device, tmp_path
    ):
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            count_flops=True,
            use_cpu=(device.type == "cpu"),
        )
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )
        distiller.train()

        entries = [e for e in distiller.state.log_history if "flops/step" in e]
        assert len(entries) == 2
        assert entries[0]["flops/step"] > 0
        assert entries[-1]["flops/total"] == pytest.approx(2 * entries[-1]["flops/step"])


class TestTeacherPlacementOnEvaluate:
    """evaluate() places the teacher even when train() has not run."""

    def test_evaluate_before_train_places_teacher_once(
        self, teacher_model, student_model, train_dataset, eval_dataset, training_args, monkeypatch
    ):
        from silverspoon_kd.distributed import strategies

        placed = []

        def record(teacher, device):
            placed.append(device)
            return teacher

        monkeypatch.setattr(strategies, "place_teacher_replicated", record)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=_make_reskd_args(training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        assert distiller._teacher_placed is False

        distiller.evaluate()
        assert distiller._teacher_placed is True
        if distiller.args.device.type == "cpu":
            assert placed == [distiller.args.device]

        distiller.train()
        # Placement runs once; train() does not repeat it.
        if distiller.args.device.type == "cpu":
            assert placed == [distiller.args.device]
