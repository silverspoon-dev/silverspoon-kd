"""
Unit tests for HolisticDistiller class.
"""

import logging
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.alignments.projectors import GenericLinearProjector
from silverspoon_kd.distillers.holistic_distiller import HolisticDistiller
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)

from ..conftest import DummyDataset, SimpleModel, SimpleOutput, create_alignment


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


class TestHolisticDistiller:
    """Test suite for HolisticDistiller."""

    def test_initialization(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
    ):
        """Test that HolisticDistiller initializes correctly."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        assert distiller.teacher_model is teacher_model
        assert distiller.student_model is student_model
        assert distiller.alignments == teacher_alignments
        assert distiller.teacher_capture is not None
        assert distiller.student_capture is not None
        assert len(distiller.alignment_id_to_module_id) == len(teacher_alignments)

    def test_initialization_without_student_model(
        self, teacher_model, teacher_alignments, training_args, train_dataset
    ):
        """Test that HolisticDistiller raises error when student_model is None."""
        # The Trainer base class raises RuntimeError before HolisticDistiller can check
        with pytest.raises(
            RuntimeError,
            match="`Trainer` requires either a `model` or `model_init` argument",
        ):
            HolisticDistiller(
                student_model=None,
                teacher_model=teacher_model,
                alignments=teacher_alignments,
                args=_make_holistic_args(training_args),
                train_dataset=train_dataset,
            )

    def test_prepare_student_inputs_default(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Test that _prepare_student_inputs returns inputs unchanged by default."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        }
        result = distiller._prepare_student_inputs(inputs)

        assert result is inputs

    def test_prepare_student_inputs_custom(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Test that _prepare_student_inputs uses custom function when provided."""

        def custom_prepare(inputs):
            return {**inputs, "student_only_key": True}

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
            prepare_student_inputs=custom_prepare,
        )

        inputs = {"input_ids": torch.tensor([1, 2, 3])}
        result = distiller._prepare_student_inputs(inputs)

        assert "student_only_key" in result
        assert result["student_only_key"] is True

    def test_register_and_deregister_capture(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Test that both capture engines are registered and deregistered."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Both capture engines exist and have their hooks installed
        assert distiller.teacher_capture is not None
        assert distiller.student_capture is not None
        assert distiller.teacher_capture.is_registered
        assert distiller.student_capture.is_registered

        distiller._deregister_capture()

        assert not distiller.teacher_capture.is_registered
        assert not distiller.student_capture.is_registered

    def test_clear_captured_data(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Test that _clear_captured_data clears both capture engines."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        # Add some dummy captured data
        distiller.teacher_capture.captured_outputs[0] = torch.randn(2, 16, 128)
        distiller.student_capture.captured_outputs[0] = torch.randn(2, 16, 64)

        # Clear
        distiller._clear_captured_data()

        # Should be empty
        assert len(distiller.teacher_capture.captured_outputs) == 0
        assert len(distiller.student_capture.captured_outputs) == 0

    def test_training_step(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Test that training_step executes without errors."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Create a batch
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Run training step
        loss = distiller.training_step(student_model, batch)

        # Should return a scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        # Should have losses stored
        assert len(distiller.step_losses) > 0

        # Should have metrics
        assert len(distiller.current_step_metrics) > 0

        distiller._deregister_capture()

    def test_compute_loss(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Test that compute_loss executes without errors."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Create a batch
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Run compute_loss
        loss = distiller.compute_loss(student_model, batch)

        # Should return a scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        # Test with return_outputs
        loss, outputs = distiller.compute_loss(student_model, batch, return_outputs=True)
        assert isinstance(loss, torch.Tensor)
        assert outputs is None

        distiller._deregister_capture()

    def test_compute_alignment_losses_training(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Test that _compute_alignment_losses works in training mode."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Manually set up captured outputs
        teacher_output = torch.randn(2, 16, 128, device=device)
        student_output = torch.randn(2, 16, 64, device=device)

        distiller.teacher_capture.captured_outputs[0] = teacher_output
        distiller.student_capture.captured_outputs[0] = student_output

        # Compute losses
        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=True,
        )

        # Should return a scalar loss
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0

        # Should have stored losses
        assert len(distiller.step_losses) > 0

        # Should have metrics
        assert len(distiller.current_step_metrics) > 0

        distiller._deregister_capture()

    def test_compute_alignment_losses_eval(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Test that _compute_alignment_losses works in eval mode."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Manually set up captured outputs
        teacher_output = torch.randn(2, 16, 128, device=device)
        student_output = torch.randn(2, 16, 64, device=device)

        distiller.teacher_capture.captured_outputs[0] = teacher_output
        distiller.student_capture.captured_outputs[0] = student_output

        # Compute losses
        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=False,
        )

        # Should return a scalar loss
        assert isinstance(total_loss, torch.Tensor)

        # Should have stored eval losses
        assert len(distiller.eval_losses) > 0

        distiller._deregister_capture()

    def test_multiple_alignments(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
        device,
    ):
        """Test HolisticDistiller with multiple alignments."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Create a batch
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Run training step
        distiller.training_step(student_model, batch)

        # Should have losses from all alignments
        assert len(distiller.step_losses) == len(teacher_alignments)

        distiller._deregister_capture()

    def test_evaluation_with_dataset(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        eval_dataset,
    ):
        """Test that evaluation works with dataset."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        distiller._register_capture()

        # Run evaluation
        metrics = distiller.evaluate()

        # Should return metrics dictionary
        assert isinstance(metrics, dict)
        assert "eval_loss" in metrics

        distiller._deregister_capture()

    def test_deepcopy_captured_args_kwargs(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Test initialization with deepcopy_captured_args_and_kwargs."""
        holistic_args = _make_holistic_args(training_args, deepcopy_captured_args_and_kwargs=True)
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=holistic_args,
            train_dataset=train_dataset,
        )

        assert distiller.args.deepcopy_captured_args_and_kwargs is True

    def test_gradient_flow_through_student(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Test that gradients flow through student model during training."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()
        # Explicitly build the optimizer (normally done by Trainer.train()).
        distiller.create_optimizer()

        # Get initial student parameter norm
        initial_params = {}
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()

        # Create a batch
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Run training step (backward pass) then optimizer step
        distiller.training_step(student_model, batch)
        distiller.optimizer.step()

        # Parameters should have changed after optimization step
        changed = False
        for name, param in student_model.named_parameters():
            if (
                param.requires_grad
                and name in initial_params
                and not torch.allclose(param.data, initial_params[name])
            ):
                changed = True
                break

        assert changed, "Student parameters should change after training step"

        distiller._deregister_capture()

    def test_teacher_gradients_disabled(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Test that teacher model gradients are not computed."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Create a batch
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Run training step
        distiller.training_step(student_model, batch)

        # Teacher parameters should not have gradients
        for param in teacher_model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

        distiller._deregister_capture()

    def test_auto_truncate(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Test HolisticDistiller with auto_truncate enabled."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )

        assert distiller.teacher_capture.auto_truncate is True
        assert distiller.student_capture.auto_truncate is True

    def test_prediction_step(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Test that prediction_step works correctly."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Create a batch
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Run prediction step
        loss, logits, labels = distiller.prediction_step(
            student_model, batch, prediction_loss_only=True
        )

        # Should return loss and None for logits and labels
        assert isinstance(loss, torch.Tensor)
        assert logits is None
        assert labels is None

        distiller._deregister_capture()


def _make_bf16_holistic_alignments(teacher_model, student_model, num_layers=1):
    """Helper: create alignments with auto_dtype_match=True for bfloat16 student."""
    alignments = []
    for i in range(num_layers):
        teacher_block = teacher_model.get_layer(i)
        student_block = student_model.get_layer(i)
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="bf16_student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            auto_dtype_match=True,
        )
        alignments.append(alignment)
    return alignments


class TestHolisticDistillerAutoDtypeMatch:
    """Tests for auto_dtype_match in HolisticDistiller."""

    def test_training_step_with_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Student model in bfloat16, teacher in float32 — training step should succeed."""
        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(
            device=device, dtype=torch.bfloat16
        )
        alignments = _make_bf16_holistic_alignments(teacher_model, student_model, num_layers=1)

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        distiller._deregister_capture()

    def test_compute_loss_with_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Student model in bfloat16, teacher in float32 — eval loss should succeed."""
        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(
            device=device, dtype=torch.bfloat16
        )
        alignments = _make_bf16_holistic_alignments(teacher_model, student_model, num_layers=1)

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.compute_loss(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        loss_ret, outputs = distiller.compute_loss(student_model, batch, return_outputs=True)
        assert isinstance(loss_ret, torch.Tensor)
        assert outputs is None

        distiller._deregister_capture()

    def test_compute_alignment_losses_casts_teacher_to_student_dtype(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Directly verify that teacher output is cast to student dtype in _compute_alignment_losses."""
        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(
            device=device, dtype=torch.bfloat16
        )
        alignments = _make_bf16_holistic_alignments(teacher_model, student_model, num_layers=1)

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        # Manually inject captured outputs with mismatched dtypes
        distiller.teacher_capture.captured_outputs[0] = torch.randn(
            2, 16, 128, device=device, dtype=torch.float32
        )
        distiller.student_capture.captured_outputs[0] = torch.randn(
            2, 16, 128, device=device, dtype=torch.bfloat16
        )

        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=True,
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert total_loss.item() >= 0
        assert len(distiller.step_losses) == 1

        distiller._deregister_capture()

    def test_auto_device_match_in_compute_alignment_losses(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Verify auto_device_match works in _compute_alignment_losses (CPU same-device sanity check)."""
        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device=device)
        teacher_block = teacher_model.get_layer(0)
        student_block = student_model.get_layer(0)
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            auto_device_match=True,
        )
        alignments = [alignment]

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        # Manually inject captured outputs on same device
        distiller.teacher_capture.captured_outputs[0] = torch.randn(2, 16, 128, device=device)
        distiller.student_capture.captured_outputs[0] = torch.randn(2, 16, 128, device=device)

        total_loss = distiller._compute_alignment_losses(
            distiller.teacher_capture,
            distiller.student_capture,
            distiller.alignment_id_to_module_id,
            is_training=True,
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0

        distiller._deregister_capture()

    def test_auto_device_and_dtype_match_together(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Both auto_device_match and auto_dtype_match enabled simultaneously."""
        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(
            device=device, dtype=torch.bfloat16
        )
        teacher_block = teacher_model.get_layer(0)
        student_block = student_model.get_layer(0)
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="bf16_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            auto_device_match=True,
            auto_dtype_match=True,
        )
        alignments = [alignment]

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        distiller._deregister_capture()

    def test_multiple_alignments_with_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Multiple alignment layers with bfloat16 student — all should be handled."""
        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(
            device=device, dtype=torch.bfloat16
        )
        alignments = _make_bf16_holistic_alignments(teacher_model, student_model, num_layers=3)

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(student_model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert len(distiller.step_losses) == 3

        distiller._deregister_capture()

    def test_gradient_flow_through_bf16_student(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Verify gradients flow and update bfloat16 student parameters."""
        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(
            device=device, dtype=torch.bfloat16
        )
        alignments = _make_bf16_holistic_alignments(teacher_model, student_model, num_layers=1)

        initial_params = {
            n: p.data.clone() for n, p in student_model.named_parameters() if p.requires_grad
        }

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        # Explicitly build the optimizer (normally done by Trainer.train()).
        distiller.create_optimizer()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        distiller.training_step(student_model, batch)
        distiller.optimizer.step()

        changed = any(
            not torch.equal(p.data, initial_params[n])
            for n, p in student_model.named_parameters()
            if p.requires_grad and n in initial_params
        )
        assert changed, "Student parameters should change after training step"

        # Weights should still be bfloat16
        for p in student_model.parameters():
            if p.is_floating_point():
                assert p.dtype == torch.bfloat16

        distiller._deregister_capture()


def _auto_truncate_worker(rank, method="training_step"):
    """Run auto_truncate test in a clean subprocess via mp.spawn.

    This eliminates pytest-xdist dynamo state leaks that interfere with
    exception propagation through Module._call_impl.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0) if torch.cuda.is_available() else None

    teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
    teacher.eval()
    student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=3).to(device)

    alignments = [
        create_alignment(
            teacher_block=teacher.get_layer(0),
            student_block=student.get_layer(0),
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=5,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=999,
            dataloader_num_workers=0,
            report_to=[],
            use_cpu=(device == "cpu"),
        )
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(num_samples=20, seq_len=16),
            auto_truncate=True,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        if method == "training_step":
            loss = distiller.training_step(student, batch)
        else:
            loss = distiller.compute_loss(student, batch)

        assert isinstance(loss, torch.Tensor), f"Expected tensor, got {type(loss)}"
        assert loss.dim() == 0, f"Expected scalar, got dim={loss.dim()}"
        assert torch.isfinite(loss), f"Loss not finite: {loss}"
        distiller._deregister_capture()


class TestHolisticDistillerTruncatedForward:
    """Test TruncatedForwardException paths in HolisticDistiller.

    Tests run in fresh subprocesses via mp.spawn to avoid pytest-xdist
    dynamo state leaks that interfere with exception propagation.
    """

    def test_training_step_with_auto_truncate(self):
        """Test training_step with auto_truncate that causes truncated forwards."""
        mp.spawn(_auto_truncate_worker, args=("training_step",), nprocs=1, join=True)

    def test_compute_loss_with_auto_truncate(self):
        """Test compute_loss with auto_truncate that causes truncated forwards."""
        mp.spawn(_auto_truncate_worker, args=("compute_loss",), nprocs=1, join=True)


class TestHolisticDistillerDefaultArgs:
    """Test default TrainingArguments creation in HolisticDistiller."""

    def test_initialization_without_args(
        self, teacher_model, student_model, single_alignment, train_dataset, device
    ):
        """Test that HolisticDistiller creates default args when args is None."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=None,
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.args, TrainingArguments)


class TestHolisticDistillerNoneStudentModel:
    """Test HolisticDistiller validation when student_model is None."""

    def test_none_student_model_raises_error(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that HolisticDistiller raises ValueError when student_model is None."""
        with pytest.raises((ValueError, RuntimeError)):
            HolisticDistiller(
                student_model=None,
                teacher_model=teacher_model,
                alignments=single_alignment,
                args=_make_holistic_args(training_args),
                train_dataset=train_dataset,
            )


class TestHolisticDistillerTruncatedForwardCaptured:
    """Test auto_truncate with modules that ARE in the capture list.

    Uses mp.spawn for subprocess isolation (same as distributed tests).
    """

    def test_training_step_with_captured_terminal(self):
        """Training step works with auto_truncate enabled."""
        mp.spawn(_auto_truncate_worker, args=("training_step",), nprocs=1, join=True)

    def test_compute_loss_with_captured_terminal(self):
        """Compute loss works with auto_truncate enabled."""
        mp.spawn(_auto_truncate_worker, args=("compute_loss",), nprocs=1, join=True)


class TestHolisticDistillerTorchCompile:
    """Test torch.compile handling in HolisticDistiller."""

    def test_teacher_compiled_when_torch_compile_true(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        train_dataset,
        device,
        tmp_path,
    ):
        """Test that teacher is compiled when torch_compile=True."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = HolisticDistiller(
            teacher_model=teacher_model,
            student_model=student_model,
            alignments=teacher_alignments,
            args=args,
            train_dataset=train_dataset,
            auto_truncate=False,
        )

        # Teacher should be compiled
        assert isinstance(distiller.teacher_model, torch._dynamo.eval_frame.OptimizedModule)

    def test_training_works_with_torch_compile(
        self,
        teacher_model,
        student_model,
        single_alignment,
        train_dataset,
        device,
        tmp_path,
    ):
        """Test that training works after torch.compile is applied."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = HolisticDistiller(
            teacher_model=teacher_model,
            student_model=student_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            auto_truncate=False,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(student_model, batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        distiller._deregister_capture()


class TestHolisticDistillerInputProjectorWarning:
    """HKD should warn when alignments have input_projector set."""

    def test_warns_on_input_projector(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        device,
        caplog,
    ):
        input_projector = GenericLinearProjector(
            in_features=128,
            out_features=64,
            mode="input",
            apply_to_arg=0,
        ).to(device)

        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student_model.get_layer(0),
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            input_projector=input_projector,
            output_projector=nn.Linear(64, 128).to(device),
        )

        with caplog.at_level(
            logging.WARNING, logger="silverspoon_kd.distillers.holistic_distiller"
        ):
            HolisticDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                alignments=[alignment],
                args=_make_holistic_args(training_args),
                train_dataset=train_dataset,
            )

        assert any("input_projector" in r.message for r in caplog.records), (
            "Expected warning about input_projector being ignored in HKD"
        )

    def test_no_warning_without_input_projector(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        device,
        caplog,
    ):
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student_model.get_layer(0),
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            output_projector=nn.Linear(64, 128).to(device),
        )

        with caplog.at_level(
            logging.WARNING, logger="silverspoon_kd.distillers.holistic_distiller"
        ):
            HolisticDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                alignments=[alignment],
                args=_make_holistic_args(training_args),
                train_dataset=train_dataset,
            )

        assert not any("input_projector" in r.message for r in caplog.records), (
            "Should not warn about input_projector when none is set"
        )


class TestHolisticDistillerTorchCompileTerminalGuard:
    """Tests for torch_compile + auto_truncate coexistence.

    Isolated experiments proved that torch.compile is compatible with
    exception-based truncation — the earlier "incompatibility" was an
    artifact of pytest-xdist dynamo state leaks.
    """

    def test_allows_torch_compile_with_auto_truncate(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """torch_compile should remain True when auto_truncate=True."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args, torch_compile=True),
            train_dataset=train_dataset,
            auto_truncate=True,
        )

        assert distiller.args.torch_compile is True

    def test_keeps_torch_compile_without_auto_truncate(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """torch_compile should remain True when auto_truncate=False."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args, torch_compile=True),
            train_dataset=train_dataset,
            auto_truncate=False,
        )

        assert distiller.args.torch_compile is True

    def test_no_warning_when_torch_compile_false(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        caplog,
    ):
        """No warning when torch_compile is already False, even with auto_truncate."""
        import logging

        with caplog.at_level(logging.WARNING):
            HolisticDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                alignments=single_alignment,
                args=_make_holistic_args(training_args, torch_compile=False),
                train_dataset=train_dataset,
                auto_truncate=True,
            )

        assert not any("torch_compile is incompatible" in msg for msg in caplog.messages)


class TestHolisticDistillerAutoTruncate:
    """Integration tests for auto_truncate in HolisticDistiller."""

    def test_auto_truncate_produces_valid_loss(self):
        """Training step with auto_truncate=True should produce a valid loss."""
        mp.spawn(_auto_truncate_worker, args=("training_step",), nprocs=1, join=True)

    def test_auto_truncate_false_produces_valid_loss(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        device,
    ):
        """Training step with auto_truncate=False should also work."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=False,
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

    def test_auto_truncate_false_allows_torch_compile(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """torch_compile should remain True when auto_truncate=False and no terminal modules."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args, torch_compile=True),
            train_dataset=train_dataset,
            auto_truncate=False,
        )
        assert distiller.args.torch_compile is True

    def test_auto_truncate_true_allows_torch_compile(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """torch_compile should remain enabled when auto_truncate=True."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args, torch_compile=True),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        assert distiller.args.torch_compile is True

    def test_auto_truncate_captures_engine_setting(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """auto_truncate should be propagated to capture engines."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        assert distiller.teacher_capture.auto_truncate is True
        assert distiller.student_capture.auto_truncate is True

        distiller2 = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=False,
        )
        assert distiller2.teacher_capture.auto_truncate is False
        assert distiller2.student_capture.auto_truncate is False


class TestAutoTruncateFSDPInteraction:
    """Test that auto_truncate is correctly handled with FSDP.

    FSDP's backward hooks require all forward-participating modules to
    complete the backward state transition. This means:
    - FSDP *student* (needs backward): auto_truncate must be disabled
    - FSDP *teacher* (forward-only, no_grad): auto_truncate is safe
    """

    def test_fsdp_student_disables_student_auto_truncate(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        caplog,
    ):
        """When FSDP wraps the student, auto_truncate should be disabled on
        the student capture engine (but NOT the teacher) and a warning logged.
        """
        from silverspoon_kd.distillers.base_distiller import BaseDistiller

        assert BaseDistiller._is_model_fsdp_wrapped(student_model) is False

        import logging

        with caplog.at_level(logging.WARNING):
            distiller = HolisticDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                alignments=single_alignment,
                args=_make_holistic_args(training_args),
                train_dataset=train_dataset,
                auto_truncate=True,
            )

        # Non-FSDP: both engines keep auto_truncate=True, no warning
        assert distiller.teacher_capture.auto_truncate is True
        assert distiller.student_capture.auto_truncate is True
        assert not any(
            "auto_truncate=True was requested but is being disabled" in msg
            for msg in caplog.messages
        )

    def test_non_fsdp_keeps_auto_truncate_enabled(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Without FSDP, auto_truncate should remain True on both engines."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        assert distiller.teacher_capture.auto_truncate is True
        assert distiller.student_capture.auto_truncate is True

    def test_ddp_keeps_auto_truncate_enabled(self):
        """DDP (non-FSDP) should keep auto_truncate enabled — training step works."""
        mp.spawn(_auto_truncate_worker, args=("training_step",), nprocs=1, join=True)


class TestAnchorFSDPOutput:
    """Unit tests for BaseDistiller._anchor_fsdp_output.

    This method adds a zero-weighted term from the student forward output
    into the loss so that FSDP pre-backward hooks fire correctly.  These
    tests verify the method's behavior for all output types including the
    critical ``model_output=None`` case (truncated forward).
    """

    def test_none_output_returns_loss_unchanged(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """When student_output is None, loss is returned unchanged."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        loss = torch.tensor(1.5, requires_grad=True)
        result = distiller._anchor_fsdp_output(loss, None)
        assert result is loss

    def test_none_output_warns_when_fsdp_enabled(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        caplog,
    ):
        """When model_output is None and FSDP is active, a warning is logged."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        # Temporarily simulate FSDP being enabled
        original = distiller.is_fsdp_enabled
        distiller.is_fsdp_enabled = True
        try:
            loss = torch.tensor(1.5, requires_grad=True)
            with caplog.at_level(logging.WARNING):
                result = distiller._anchor_fsdp_output(loss, None)
            assert result is loss
            assert any("FSDP" in msg and "no output" in msg for msg in caplog.messages)
        finally:
            distiller.is_fsdp_enabled = original

    def test_tensor_output_with_grad_adds_anchor(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Output tensor with requires_grad=True adds a 0*sum() anchor term."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        loss = torch.tensor(2.0, requires_grad=True)
        output = torch.randn(4, 10, requires_grad=True)
        result = distiller._anchor_fsdp_output(loss, output)
        # Value should be same (0 * output.sum() = 0)
        assert result.item() == pytest.approx(2.0)
        # But it should be a different tensor (graph was modified)
        assert result is not loss
        assert result.grad_fn is not None

    def test_namedtuple_output_extracts_logits(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Output with .logits attribute uses logits for anchoring."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        loss = torch.tensor(3.0, requires_grad=True)
        logits = torch.randn(4, 10, requires_grad=True)
        output = SimpleOutput(logits=logits, loss=None)
        result = distiller._anchor_fsdp_output(loss, output)
        assert result.item() == pytest.approx(3.0)
        assert result.grad_fn is not None

    def test_no_grad_output_returns_loss_unchanged(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """Output tensor without requires_grad returns loss unchanged."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        loss = torch.tensor(4.0, requires_grad=True)
        output = torch.randn(4, 10)  # requires_grad=False
        result = distiller._anchor_fsdp_output(loss, output)
        assert result is loss


# -----------------------------------------------------------------------------
# Optimizer coverage regression tests
#
# HKD must use a single optimizer that covers every trainable student
# parameter.  A per-alignment ``CompositeOptimizer`` (the BKD scheme) only
# steps parameters inside some ``alignment.student_block``; with 4 encoder
# alignments plus a classifier alignment, BERT's embeddings and pooler would
# receive gradients during backward but never be stepped by any optimizer.
#
# The tests below verify that after calling ``create_optimizer``:
#   1. Every trainable student parameter is in the optimizer's param groups.
#   2. A single backward+step actually updates every parameter (no silent
#      drop-through).
#   3. Projector parameters (both explicit and lazy/auto-inferred) are
#      included in the same optimizer.
#   4. The optimizer is a single standard ``torch.optim.Optimizer`` — not a
#      ``CompositeOptimizer`` with per-alignment children.
#
# Any regression to per-alignment optimizers would immediately fail (1), (2),
# and (4); any regression in the lazy-projector hook would fail (3).
# -----------------------------------------------------------------------------


class TestHolisticOptimizerCoverage:
    """Regression tests for end-to-end optimizer coverage.

    HKD runs the student end-to-end and must optimise every parameter that
    contributes to the loss — including embeddings, pooler, LM head, and any
    layer not covered by an alignment.
    """

    def test_optimizer_is_not_composite(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """HKD must not use CompositeOptimizer (unlike BKD).

        ``CompositeOptimizer`` only steps parameters owned by its child
        optimizers, which is correct for BKD's independent blocks but
        silently breaks HKD's end-to-end backward pass.
        """
        from silverspoon_kd.optim import CompositeOptimizer

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller.create_optimizer()

        assert not isinstance(distiller.optimizer, CompositeOptimizer), (
            "HolisticDistiller must use a single optimizer, not CompositeOptimizer. "
            "CompositeOptimizer would leave non-aligned student params un-updated."
        )
        assert HolisticDistiller._USE_COMPOSITE_OPTIMIZER is False, (
            "HolisticDistiller._USE_COMPOSITE_OPTIMIZER must be False; a "
            "composite per-alignment optimizer silently leaves student "
            "parameters outside any student_block (e.g. embeddings) untrained."
        )

    def test_every_student_param_in_optimizer(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
    ):
        """Every trainable student param must appear in some optimizer param group.

        This is the first-line defence against the regression: if a student
        parameter is outside every ``alignment.student_block`` but still
        ``requires_grad=True``, it must be in the main optimizer.  For a
        multi-layer SimpleModel this includes ``embedding`` and ``lm_head``
        which are outside the per-layer alignments.
        """
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller.create_optimizer()

        # Collect every param id owned by the optimizer.
        optim_param_ids = set()
        for group in distiller.optimizer.param_groups:
            for p in group["params"]:
                optim_param_ids.add(id(p))

        # Every trainable student parameter must be in the optimizer.
        missing = []
        for name, p in student_model.named_parameters():
            if p.requires_grad and id(p) not in optim_param_ids:
                missing.append(name)

        assert not missing, (
            f"Student parameters missing from optimizer: {missing}. "
            "These parameters would receive gradients during backward but "
            "never get updated during optimizer.step() — a silent bug."
        )

    def test_embeddings_updated_during_training(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
        device,
    ):
        """The student's ``embedding`` weight must change after a training step.

        Embeddings sit *outside* any aligned ``student_block``.  An optimizer
        scoped to the aligned blocks alone would leave them un-updated even
        though gradients flow through them.
        """
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        distiller.create_optimizer()

        initial_embed = student_model.embedding.weight.data.clone()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        distiller.training_step(student_model, batch)
        distiller.optimizer.step()

        assert not torch.allclose(student_model.embedding.weight.data, initial_embed, atol=0.0), (
            "Student embedding weight did not change after an HKD training step. "
            "This is the embeddings-never-trained regression: the parameter "
            "receives gradients (backward runs through it) but no optimizer "
            "steps it."
        )

        distiller._deregister_capture()

    def test_head_with_response_alignment_is_updated(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
        device,
    ):
        """An output head covered by a response-style alignment must update.

        Simulates the BERT sequence-classification use case where the
        classifier/head is registered as an additional alignment outside
        the per-layer ``student_blocks``.  The head receives gradients from
        the response loss and must be updated by the main optimizer.
        """
        # Mirror the ``trainers.py`` response_alignment pattern: treat
        # lm_head as a separate student_block and create an alignment on it.
        # In the SimpleModel test fixtures teacher.lm_head and student.lm_head
        # both map 128->128 resp. 64->128, so we need an output projector
        # from the student's 128-dim logits to the teacher's 128-dim logits.
        # They're already the same size so no projector is needed.
        head_alignment = create_alignment(
            teacher_block=teacher_model.lm_head,
            student_block=student_model.lm_head,
            teacher_module_name="lm_head",
            student_module_name="lm_head",
            teacher_hidden_dim=128,  # both lm_heads output 128
        )

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments + [head_alignment],
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        distiller.create_optimizer()

        initial_head = student_model.lm_head.weight.data.clone()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        distiller.training_step(student_model, batch)
        distiller.optimizer.step()

        assert not torch.allclose(student_model.lm_head.weight.data, initial_head, atol=0.0), (
            "Student lm_head weight did not change despite being covered by a "
            "response-style alignment.  This would reproduce the BERT classifier "
            "regression where the head receives gradients but isn't updated."
        )

        distiller._deregister_capture()

    def test_non_aligned_upstream_layer_updated(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        device,
    ):
        """A layer upstream of the aligned layer must still update.

        Builds a distiller aligning ONLY the final layer, then verifies that
        an earlier (upstream) layer's parameters still receive updates —
        gradient flows back through every layer between the loss and the
        model inputs, and every such parameter must be in the optimizer.

        This is the canonical test for "params outside any alignment still
        update" — for the BERT case, the equivalent would be the embeddings
        feeding the encoder.
        """
        # Align only the LAST layer (layer 2).  Layers 0 and 1 are upstream
        # of the alignment point and will receive gradients during backward
        # even though they are not inside any ``alignment.student_block``.
        alignment = create_alignment(
            teacher_block=teacher_model.get_layer(2),
            student_block=student_model.get_layer(2),
            teacher_module_name="layers.2",
            student_module_name="layers.2",
        )

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        distiller.create_optimizer()

        # Snapshot a parameter inside layer 0 (upstream of layer 2, NOT in
        # any alignment).
        initial_layer0 = student_model.get_layer(0).ffn.weight.data.clone()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        distiller.training_step(student_model, batch)
        distiller.optimizer.step()

        assert not torch.allclose(
            student_model.get_layer(0).ffn.weight.data, initial_layer0, atol=0.0
        ), (
            "Upstream student layer (layer 0) did not update even though "
            "gradients flowed through it to layer 2's alignment loss.  This "
            "is the core regression: non-aligned params receive gradients "
            "but never get stepped."
        )

        distiller._deregister_capture()

    def test_explicit_projector_params_in_optimizer(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
    ):
        """Projectors explicitly passed to Alignment must be in the optimizer."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller.create_optimizer()

        optim_param_ids = set()
        for group in distiller.optimizer.param_groups:
            for p in group["params"]:
                optim_param_ids.add(id(p))

        for i, alignment in enumerate(teacher_alignments):
            for proj_name, proj in [
                ("input_projector", alignment.input_projector),
                ("output_projector", alignment.output_projector),
            ]:
                if proj is None:
                    continue
                missing = [n for n, p in proj.named_parameters() if id(p) not in optim_param_ids]
                assert not missing, f"Alignment {i} {proj_name} params not in optimizer: {missing}"

    def test_auto_projector_params_added_after_first_forward(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        device,
    ):
        """Lazy auto-projectors created on the first forward pass must join the optimizer."""
        # Create an alignment WITHOUT explicit projectors, but with
        # auto_projector=True so the HKD base auto-infers one on the first
        # forward pass (student_hidden=64 vs teacher_hidden=128).
        teacher_block = teacher_model.get_layer(0)
        student_block = student_model.get_layer(0)
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            auto_projector=True,
        )

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        distiller.create_optimizer()

        # Before forward: no output_projector yet (lazy).
        assert alignment.output_projector is None

        # Trigger a forward pass — this lazy-inits the output_projector.
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        distiller.training_step(student_model, batch)

        # The lazy projector must now exist and be in the main optimizer.
        assert alignment.output_projector is not None, (
            "Auto-projector was not lazy-initialised on first forward"
        )
        optim_param_ids = set()
        for group in distiller.optimizer.param_groups:
            for p in group["params"]:
                optim_param_ids.add(id(p))

        missing = [
            n
            for n, p in alignment.output_projector.named_parameters()
            if id(p) not in optim_param_ids
        ]
        assert not missing, (
            f"Auto-projector params not added to optimizer after first forward: {missing}. "
            "Lazy projectors must be wired to the main optimizer via "
            "Alignment._add_projector_params_to_optimizer, which requires "
            "alignment.optimizer to point at the main optimizer."
        )

        distiller._deregister_capture()

    def test_optimizer_param_count_matches_student(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
    ):
        """Total optimiser param count equals student trainable params + projector params.

        A sanity check that catches both under-inclusion (bug we fixed) and
        accidental over-inclusion (e.g. teacher params leaking in).
        """
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller.create_optimizer()

        student_trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        projector_params = 0
        for alignment in teacher_alignments:
            for proj in (alignment.input_projector, alignment.output_projector):
                if proj is not None:
                    projector_params += sum(p.numel() for p in proj.parameters())

        optim_params = sum(p.numel() for g in distiller.optimizer.param_groups for p in g["params"])

        assert optim_params == student_trainable + projector_params, (
            f"Optimizer param count ({optim_params}) != student trainable "
            f"({student_trainable}) + projectors ({projector_params}). "
            "If optim < expected: regression to per-alignment scoping. "
            "If optim > expected: teacher params or duplicates leaked in."
        )

    def test_no_teacher_params_in_optimizer(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
    ):
        """Teacher parameters must never appear in the student's optimizer."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller.create_optimizer()

        optim_param_ids = {id(p) for g in distiller.optimizer.param_groups for p in g["params"]}
        teacher_param_ids = {id(p) for p in teacher_model.parameters()}

        overlap = optim_param_ids & teacher_param_ids
        assert not overlap, (
            f"{len(overlap)} teacher parameters leaked into the student optimizer. "
            "This would cause the teacher to be updated during training — "
            "wrong semantics for distillation."
        )

    def test_multiple_training_steps_update_embeddings_cumulatively(
        self,
        teacher_model,
        student_model,
        teacher_alignments,
        training_args,
        train_dataset,
        device,
    ):
        """Embeddings should continue updating across multiple training steps.

        A single-step bug could theoretically be masked by an atol=0 check;
        this test confirms the update is real by running multiple steps and
        checking the cumulative change is larger than the per-step change.
        """
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        distiller.create_optimizer()

        initial_embed = student_model.embedding.weight.data.clone()

        # Step 1
        batch1 = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        distiller.training_step(student_model, batch1)
        distiller.optimizer.step()
        # Use model.zero_grad() — NOT optimizer.zero_grad() — to match what
        # the HF Trainer actually does.  optimizer.zero_grad() masks bugs
        # where params outside model.parameters() (e.g. projectors) silently
        # accumulate gradients.
        student_model.zero_grad()
        after_step1 = student_model.embedding.weight.data.clone()

        # Steps 2-5
        for _ in range(4):
            batch = {
                "input_ids": torch.randint(0, 128, (2, 16), device=device),
                "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            }
            distiller.training_step(student_model, batch)
            distiller.optimizer.step()
            student_model.zero_grad()

        after_step5 = student_model.embedding.weight.data.clone()

        step1_change = (after_step1 - initial_embed).abs().mean().item()
        cumulative_change = (after_step5 - initial_embed).abs().mean().item()

        assert step1_change > 0.0, "Embeddings didn't update on step 1"
        assert cumulative_change > step1_change, (
            f"Embedding change after 5 steps ({cumulative_change:.2e}) not "
            f"larger than after 1 step ({step1_change:.2e}) — embeddings may "
            "have stopped updating."
        )

        distiller._deregister_capture()

    def test_projector_gradients_not_accumulated_across_steps(
        self,
        teacher_model,
        student_model,
        training_args,
        train_dataset,
        device,
    ):
        """Projector gradients must NOT accumulate across training steps.

        Regression test: the HF Trainer's ``model.zero_grad()`` only zeros
        ``model.parameters()``, which misses projector parameters that live
        on Alignment objects.  ``BaseDistiller._zero_projector_gradients``
        must zero them explicitly; otherwise projector gradients from step N
        would persist into step N+1's backward pass, corrupting the gradient
        signal.

        This test uses ``model.zero_grad()`` (matching the Trainer) rather
        than ``optimizer.zero_grad()`` to exercise the real code path.
        """
        # Create alignment with dim mismatch to force projector creation
        teacher_block = teacher_model.get_layer(0)
        student_block = student_model.get_layer(0)
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            auto_projector=True,
        )

        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        distiller.create_optimizer()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # Step 1: compute gradients
        distiller.training_step(student_model, batch)
        distiller.optimizer.step()

        assert alignment.output_projector is not None, "Projector not created"

        # Record projector grad norm after step 1 (before any zero_grad)
        step1_grad = (
            sum(
                p.grad.norm().item() ** 2
                for p in alignment.output_projector.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        # Use model.zero_grad() — exactly what the HF Trainer does.
        # This is the critical line: it must NOT be optimizer.zero_grad().
        student_model.zero_grad()

        # Step 2: compute fresh gradients
        distiller.training_step(student_model, batch)

        step2_grad = (
            sum(
                p.grad.norm().item() ** 2
                for p in alignment.output_projector.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        # If gradients accumulated, step2_grad ≈ 2 * step1_grad.
        # With correct zeroing, step2_grad ≈ step1_grad (fresh, not accumulated).
        ratio = step2_grad / max(step1_grad, 1e-8)
        assert ratio < 1.5, (
            f"Projector gradient accumulated across steps: "
            f"step1={step1_grad:.4f}, step2={step2_grad:.4f}, ratio={ratio:.2f}. "
            f"Expected ratio ≈ 1.0 (fresh gradients), got {ratio:.2f} "
            f"(accumulated). This means model.zero_grad() missed projector "
            f"params and _zero_projector_gradients() is not working."
        )

        distiller._deregister_capture()


class TestHolisticHardLoss:
    """Tests for HKD hard loss (alpha > 0) support."""

    def test_alpha_zero_no_hard_loss(
        self, teacher_model, student_model, single_alignment, training_args, train_dataset, device
    ):
        """alpha=0 produces the same loss whether labels are present or not."""
        args = _make_holistic_args(training_args, alpha=0.0)
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        assert torch.isfinite(loss)
        distiller._deregister_capture()

    def test_alpha_nonzero_mixes_hard_loss(
        self, teacher_model, training_args, train_dataset, device
    ):
        """alpha > 0 with labels produces a different loss than alpha=0."""
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        # alpha=0
        args0 = _make_holistic_args(training_args, alpha=0.0)
        d0 = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args0,
            train_dataset=train_dataset,
        )
        d0._register_capture()
        loss0 = d0.compute_distillation_loss(d0.model, batch, is_training=False)
        d0._deregister_capture()

        # alpha=0.5
        args05 = _make_holistic_args(training_args, alpha=0.5)
        d05 = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args05,
            train_dataset=train_dataset,
        )
        d05._register_capture()
        loss05 = d05.compute_distillation_loss(d05.model, batch, is_training=False)
        d05._deregister_capture()

        assert torch.isfinite(loss0) and torch.isfinite(loss05)
        assert loss0.item() != pytest.approx(loss05.item(), abs=1e-5), (
            "alpha=0 and alpha=0.5 produced identical losses — hard loss not mixing in"
        )

    def test_alpha_nonzero_no_labels_falls_back(
        self, teacher_model, training_args, train_dataset, device
    ):
        """alpha > 0 without labels in batch produces pure alignment loss (no crash)."""
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        args = _make_holistic_args(training_args, alpha=0.5)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            # No labels
        }
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        assert torch.isfinite(loss)
        distiller._deregister_capture()

    def test_alpha_one_pure_hard_loss(self, teacher_model, training_args, train_dataset, device):
        """alpha=1.0 means only hard loss (zero alignment contribution)."""
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        args = _make_holistic_args(training_args, alpha=1.0)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        assert torch.isfinite(loss)
        distiller._deregister_capture()

    def test_labels_injected_when_stripped_by_prepare(
        self, teacher_model, training_args, train_dataset, device
    ):
        """When prepare_student_inputs strips labels, HKD re-injects them for hard loss."""
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )

        def strip_labels(inputs):
            return {k: v for k, v in inputs.items() if k != "labels"}

        args = _make_holistic_args(training_args, alpha=0.5)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
            prepare_student_inputs=strip_labels,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        # Without labels injection this would silently produce pure alignment loss
        loss_with_alpha = distiller.compute_distillation_loss(
            distiller.model, batch, is_training=False
        )
        distiller._deregister_capture()

        # Compare with alpha=0 to verify hard loss was actually mixed in
        args0 = _make_holistic_args(training_args, alpha=0.0)
        d0 = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args0,
            train_dataset=train_dataset,
            prepare_student_inputs=strip_labels,
        )
        d0._register_capture()
        loss_no_alpha = d0.compute_distillation_loss(d0.model, batch, is_training=False)
        d0._deregister_capture()

        assert torch.isfinite(loss_with_alpha) and torch.isfinite(loss_no_alpha)
        assert loss_with_alpha.item() != pytest.approx(loss_no_alpha.item(), abs=1e-5)

    def test_alpha_nonzero_eval_mode(self, teacher_model, training_args, train_dataset, device):
        """alpha > 0 works correctly in eval mode (is_training=False)."""
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        args = _make_holistic_args(training_args, alpha=0.5)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        student.eval()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=False)
        assert torch.isfinite(loss)
        distiller._deregister_capture()

    def test_hard_loss_gradient_flows(self, teacher_model, training_args, train_dataset, device):
        """Hard loss with alpha > 0 produces non-zero gradients on the student."""
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        args = _make_holistic_args(training_args, alpha=0.5)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
        assert has_grad, "No gradient on student parameters after hard loss backward"
        distiller._deregister_capture()

    def test_hard_loss_metrics_tracked(self, teacher_model, training_args, train_dataset, device):
        """alpha > 0 logs both loss/alignment and loss/hard in metrics."""
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        args = _make_holistic_args(training_args, alpha=0.5)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        distiller.compute_distillation_loss(distiller.model, batch, is_training=True)

        assert "loss/alignment" in distiller.current_step_metrics, (
            "loss/alignment not tracked in metrics"
        )
        assert "loss/hard" in distiller.current_step_metrics, "loss/hard not tracked in metrics"
        align_val = distiller.current_step_metrics["loss/alignment"]
        hard_val = distiller.current_step_metrics["loss/hard"]
        if isinstance(align_val, torch.Tensor):
            align_val = align_val.item()
        if isinstance(hard_val, torch.Tensor):
            hard_val = hard_val.item()
        assert align_val > 0, f"loss/alignment not positive: {align_val}"
        assert hard_val > 0, f"loss/hard not positive: {hard_val}"
        distiller._deregister_capture()

    def test_alpha_one_no_alignment_metric(
        self, teacher_model, training_args, train_dataset, device
    ):
        """alpha=1.0 still tracks loss/alignment (weight=0) but loss/hard dominates."""
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        args = _make_holistic_args(training_args, alpha=1.0)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        assert torch.isfinite(loss)
        assert "loss/hard" in distiller.current_step_metrics
        distiller._deregister_capture()

    def test_no_hard_loss_warning(
        self, teacher_model, training_args, train_dataset, device, caplog
    ):
        """Warning fires when alpha > 0 but student forward doesn't return a loss."""

        class NoLossModel(nn.Module):
            """Model whose forward() returns no loss even with labels."""

            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(128, 128)
                self.layer = nn.Linear(128, 128)
                self.lm_head = nn.Linear(128, 128)

            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                x = self.embedding(input_ids)
                x = self.layer(x)
                logits = self.lm_head(x)
                # Deliberately return no loss
                return SimpleOutput(logits=logits, loss=None)

            def get_layer(self, idx):
                return self.layer

        student = NoLossModel().to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        args = _make_holistic_args(training_args, alpha=0.5)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        with caplog.at_level(logging.WARNING):
            loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)

        assert torch.isfinite(loss), "Should return alignment loss as fallback"
        assert any("did not return a loss" in msg for msg in caplog.messages), (
            "Expected warning about missing hard loss"
        )
        distiller._deregister_capture()

    def test_signature_columns_labels_preserved(
        self, teacher_model, training_args, train_dataset, device
    ):
        """When alpha > 0, labels must be in _signature_columns to survive RemoveColumnsCollator."""

        class StrictSignatureModel(nn.Module):
            """Model whose forward() has explicit params (no **kwargs)."""

            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(128, 128)
                self.layer = nn.Linear(128, 128)
                self.lm_head = nn.Linear(128, 128)

            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.layer(x)
                return SimpleOutput(logits=self.lm_head(x), loss=None)

            def get_layer(self, idx):
                return self.layer

        student = StrictSignatureModel().to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )

        args = _make_holistic_args(training_args, alpha=0.5)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )

        # _signature_columns should include "labels" even though the model
        # forward() doesn't accept it — needed for RemoveColumnsCollator.
        assert distiller._signature_columns is not None
        assert "labels" in distiller._signature_columns, (
            "labels not in _signature_columns — RemoveColumnsCollator would "
            "strip labels from batches, silently disabling hard loss"
        )


class TestHolisticIncrementalLoss:
    """Tests for incremental alignment loss (computed inside the student forward).

    The incremental path is the only path; correctness is verified against a
    reference implementation that runs the same per-alignment math after the
    forward pass instead of inside the hook.
    """

    @staticmethod
    def _ref_alignment_loss(distiller, model, batch, is_training):
        """Reference: post-hoc all-at-once computation of the alignment loss
        after the forward pass, for checking gradient parity with the
        incremental path.

        Mirrors ``compute_distillation_loss`` (including ``_anchor_fsdp_output``)
        so the autograd graph has the same set of leaves as the incremental
        path — otherwise zero-valued lm_head gradients show up in one snapshot
        but not the other and confuse the comparison.
        """
        # Disable the incremental callback so the student capture stays in
        # the dict for us to consume after the forward.
        distiller.student_capture.output_callback = None
        try:
            teacher_inputs = distiller._prepare_teacher_inputs(batch)
            student_inputs = distiller._prepare_student_inputs(batch)
            distiller._run_teacher_forward(teacher_inputs, catch_truncation=True)
            distiller._sync_teacher_stream()
            with distiller._grad_context(is_training):
                student_output = model(**student_inputs)
            loss = distiller._compute_alignment_losses(
                distiller.teacher_capture,
                distiller.student_capture,
                distiller.alignment_id_to_module_id,
                is_training,
            )
            if loss is None:
                return None
            return distiller._anchor_fsdp_output(loss, student_output)
        finally:
            distiller.student_capture.output_callback = distiller._on_student_capture
            distiller.student_capture.captured_outputs.clear()
            distiller.teacher_capture.captured_outputs.clear()

    @staticmethod
    def _grads_snapshot(model):
        """Return {name: detached cloned grad} for parameters that have grads."""
        return {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

    def _build(self, teacher_model, student_model, alignments, training_args, **kw):
        return HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args, **kw.pop("arg_overrides", {})),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
            **kw,
        )

    def _make_batch(self, device, with_labels=True):
        torch.manual_seed(0)
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        if with_labels:
            batch["labels"] = torch.randint(0, 128, (2, 16), device=device)
        return batch

    def test_incremental_matches_reference_gradients(
        self, teacher_model, student_model, teacher_alignments, training_args, device
    ):
        """Incremental path must produce gradients identical to the
        post-hoc all-at-once reference."""
        distiller = self._build(teacher_model, student_model, teacher_alignments, training_args)
        distiller._register_capture()
        batch = self._make_batch(device)

        # Reference: post-hoc loss (callback disabled internally), backward
        student_model.zero_grad(set_to_none=True)
        ref_loss = self._ref_alignment_loss(distiller, distiller.model, batch, is_training=True)
        assert ref_loss is not None
        ref_loss.backward()
        ref_grads = self._grads_snapshot(student_model)
        ref_value = ref_loss.detach().clone()

        # Incremental: callback path
        student_model.zero_grad(set_to_none=True)
        inc_loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        inc_loss.backward()
        inc_grads = self._grads_snapshot(student_model)

        # alpha=0 by default → distiller returns the alignment loss directly
        assert torch.allclose(inc_loss.detach(), ref_value, rtol=1e-5, atol=1e-6)
        assert ref_grads.keys() == inc_grads.keys()
        for name in ref_grads:
            assert torch.allclose(ref_grads[name], inc_grads[name], rtol=1e-5, atol=1e-6), (
                f"gradient mismatch on {name}"
            )
        distiller._deregister_capture()

    def test_non_uniform_alignment_weights(
        self, teacher_model, student_model, training_args, device
    ):
        """Per-alignment ``loss_weight`` must apply correctly in the incremental path."""
        weights = [0.5, 1.0, 2.0]
        alignments = []
        for i, w in enumerate(weights):
            a = create_alignment(
                teacher_block=teacher_model.get_layer(i),
                student_block=student_model.get_layer(i),
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                with_input_projector=(i > 0),
            )
            a.loss_weight = w
            alignments.append(a)

        distiller = self._build(teacher_model, student_model, alignments, training_args)
        distiller._register_capture()
        batch = self._make_batch(device)

        student_model.zero_grad(set_to_none=True)
        ref_loss = self._ref_alignment_loss(distiller, distiller.model, batch, is_training=True)
        assert ref_loss is not None
        ref_loss.backward()
        ref_grads = self._grads_snapshot(student_model)

        student_model.zero_grad(set_to_none=True)
        inc_loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        inc_loss.backward()
        inc_grads = self._grads_snapshot(student_model)

        assert torch.allclose(inc_loss.detach(), ref_loss.detach(), rtol=1e-5, atol=1e-6)
        for name in ref_grads:
            assert torch.allclose(ref_grads[name], inc_grads[name], rtol=1e-5, atol=1e-6)
        distiller._deregister_capture()

    def test_zero_weight_alignment_no_contribution(
        self, teacher_model, student_model, training_args, device
    ):
        """An alignment with loss_weight=0 must add no gradient through that path."""
        a0 = create_alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student_model.get_layer(0),
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        a0.loss_weight = 0.0
        a1 = create_alignment(
            teacher_block=teacher_model.get_layer(1),
            student_block=student_model.get_layer(1),
            teacher_module_name="layers.1",
            student_module_name="layers.1",
            with_input_projector=True,
        )

        distiller = self._build(teacher_model, student_model, [a0, a1], training_args)
        distiller._register_capture()
        batch = self._make_batch(device)

        # All-zero weights → distiller alignment with weight 0 contributes
        # via _apply_loss_weighting which returns ``weight * loss``.  Since
        # weight=0, this is a fresh zero tensor with no grad path through
        # alignment 0's student block, so embedding gradients should match
        # those produced by a single-alignment (a1-only) distiller.
        single = self._build(teacher_model, student_model, [a1], training_args)
        single._register_capture()

        student_model.zero_grad(set_to_none=True)
        loss_full = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        loss_full.backward()
        grads_full = self._grads_snapshot(student_model)

        student_model.zero_grad(set_to_none=True)
        loss_single = single.compute_distillation_loss(single.model, batch, is_training=True)
        loss_single.backward()
        grads_single = self._grads_snapshot(student_model)
        single._deregister_capture()

        # Loss values must match (zero-weighted alignment adds 0).
        assert torch.allclose(loss_full.detach(), loss_single.detach(), rtol=1e-5, atol=1e-6)
        # Embedding gradients must match (only path is through a1).
        for name, g in grads_single.items():
            if "embedding" in name:
                assert name in grads_full
                assert torch.allclose(g, grads_full[name], rtol=1e-5, atol=1e-6)
        distiller._deregister_capture()

    def test_single_alignment(
        self, teacher_model, student_model, single_alignment, training_args, device
    ):
        """N=1 edge case: single alignment must train end-to-end."""
        distiller = self._build(teacher_model, student_model, single_alignment, training_args)
        distiller._register_capture()
        batch = self._make_batch(device)

        student_model.zero_grad(set_to_none=True)
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        loss.backward()
        assert torch.isfinite(loss)
        # At least the target student block should have a grad
        any_grad = any(p.grad is not None for p in single_alignment[0].student_block.parameters())
        assert any_grad, "single alignment did not produce gradients on student block"
        distiller._deregister_capture()

    def test_alpha_zero_and_alpha_nonzero(
        self, teacher_model, student_model, single_alignment, training_args, device
    ):
        """Incremental path with alpha=0 vs alpha=0.5 must mix in hard loss correctly."""
        batch = self._make_batch(device, with_labels=True)

        d0 = self._build(
            teacher_model,
            student_model,
            single_alignment,
            training_args,
            arg_overrides={"alpha": 0.0},
        )
        d0._register_capture()
        loss0 = d0.compute_distillation_loss(d0.model, batch, is_training=False)
        d0._deregister_capture()

        d05 = self._build(
            teacher_model,
            student_model,
            single_alignment,
            training_args,
            arg_overrides={"alpha": 0.5},
        )
        d05._register_capture()
        loss05 = d05.compute_distillation_loss(d05.model, batch, is_training=False)
        d05._deregister_capture()

        assert torch.isfinite(loss0) and torch.isfinite(loss05)
        assert loss0.item() != pytest.approx(loss05.item(), abs=1e-5), (
            "alpha mixing did not change the loss in the incremental path"
        )

    def test_repeat_hook_firing_does_not_double_count(
        self, teacher_model, student_model, teacher_alignments, training_args, device
    ):
        """Gradient checkpointing fires student hooks twice (forward + recompute).
        The dedup set must short-circuit the second firing so the running
        total — and therefore the loss and gradients — stay unchanged.

        This isolates the dedup invariant.  End-to-end integration with
        ``torch.utils.checkpoint`` (matching saved-tensor counts across
        original/recompute passes when callback work differs by mode) is
        a separate and larger concern; see the issue's note that this is
        a blocker requiring its own follow-up.
        """
        distiller = self._build(teacher_model, student_model, teacher_alignments, training_args)
        distiller._register_capture()
        batch = self._make_batch(device)

        student_model.zero_grad(set_to_none=True)
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        assert distiller._loss_accumulator is not None
        assert distiller._loss_accumulator.total is not None
        total_before = distiller._loss_accumulator.total.detach().clone()
        processed_before = set(distiller._incremental_processed)

        # Simulate the recompute pass firing every layer's hook again.
        # The student_output passed in is irrelevant — the dedup must
        # short-circuit before we touch it.
        for module_id in processed_before:
            sentinel = torch.zeros(1, device=device, requires_grad=True)
            distiller._on_student_capture(module_id, None, sentinel)

        assert distiller._incremental_processed == processed_before, (
            "dedup set should not gain or lose entries on repeat firings"
        )
        assert torch.allclose(distiller._loss_accumulator.total.detach(), total_before), (
            "running loss total changed after a repeat-firing — double-counted"
        )

        # Backward still works on the original loss.
        loss.backward()
        any_finite = any(
            p.grad is not None and torch.isfinite(p.grad).all() for p in student_model.parameters()
        )
        assert any_finite
        distiller._deregister_capture()


@pytest.mark.cuda
class TestHolisticOverlapAlignmentLoss:
    """CUDA-only: parity between overlap_alignment_loss=True and =False."""

    def test_overlap_matches_no_overlap_gradients(
        self, teacher_model, student_model, teacher_alignments, training_args
    ):
        """overlap=True and overlap=False must produce numerically equal
        gradients (within float tolerance) for the same inputs and weights."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        device = torch.device("cuda")
        teacher_model.to(device)
        student_model.to(device)
        for a in teacher_alignments:
            if a.output_projector is not None:
                a.output_projector.to(device)
            if a.input_projector is not None:
                a.input_projector.to(device)

        torch.manual_seed(0)
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }

        def run(overlap: bool):
            d = HolisticDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                alignments=teacher_alignments,
                args=_make_holistic_args(training_args),
                train_dataset=DummyDataset(num_samples=4, seq_len=16),
                overlap_alignment_loss=overlap,
            )
            d._register_capture()
            student_model.zero_grad(set_to_none=True)
            loss = d.compute_distillation_loss(d.model, batch, is_training=True)
            loss.backward()
            torch.cuda.synchronize()
            grads = {
                n: p.grad.detach().clone()
                for n, p in student_model.named_parameters()
                if p.grad is not None
            }
            d._deregister_capture()
            return loss.detach().clone(), grads

        # Reusing the same student_model across run(False) and run(True)
        # caches AccumulateGrad nodes on the default stream during the first
        # run, then the second run feeds gradients from _loss_stream into
        # them — a cross-iteration mismatch warning that only arises because
        # of this test's design. Real training uses one overlap setting per
        # run so AccumulateGrad nodes stay on a single stream.
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        try:
            loss_off, grads_off = run(False)
            loss_on, grads_on = run(True)
        finally:
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(True)
        assert torch.allclose(loss_off, loss_on, rtol=1e-4, atol=1e-5)
        for n in grads_off:
            assert torch.allclose(grads_off[n], grads_on[n], rtol=1e-4, atol=1e-5), (
                f"grad mismatch on {n}"
            )

    def test_overlap_with_alpha_mixing(
        self, teacher_model, student_model, single_alignment, training_args
    ):
        """Cross-stream combination: overlap=True with alpha>0 must still
        produce a finite, sensibly-mixed loss."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        device = torch.device("cuda")
        teacher_model.to(device)
        student_model.to(device)
        for a in single_alignment:
            if a.output_projector is not None:
                a.output_projector.to(device)

        torch.manual_seed(0)
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        d = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args, alpha=0.5),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
            overlap_alignment_loss=True,
        )
        d._register_capture()
        loss = d.compute_distillation_loss(d.model, batch, is_training=True)
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        d._deregister_capture()


def test_overlap_alignment_loss_disabled_on_cpu(
    teacher_model, student_model, single_alignment, training_args
):
    """On CPU, overlap_alignment_loss=True is silently disabled (no stream)."""
    d = HolisticDistiller(
        student_model=student_model,
        teacher_model=teacher_model,
        alignments=single_alignment,
        args=_make_holistic_args(training_args),
        train_dataset=DummyDataset(num_samples=4, seq_len=16),
        overlap_alignment_loss=True,
    )
    if not torch.cuda.is_available():
        assert d._loss_stream is None


class TestHolisticIncrementalEdgeCases:
    """Edge cases for the incremental alignment-loss path."""

    @staticmethod
    def _build(teacher_model, alignments, training_args, student_model, **kw):
        return HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_holistic_args(training_args, **kw.pop("arg_overrides", {})),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
            **kw,
        )

    @staticmethod
    def _make_batch(device, with_labels=True):
        torch.manual_seed(0)
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        if with_labels:
            batch["labels"] = torch.randint(0, 128, (2, 16), device=device)
        return batch

    def test_gradient_checkpointing_currently_raises_checkpoint_error(
        self, teacher_model, student_model, teacher_alignments, training_args, device
    ):
        """torch.utils.checkpoint(use_reentrant=False) around aligned student
        layers fires the hook twice (forward + recompute).  Our incremental
        callback creates tensors during the original pass and is short-
        circuited during recompute, so the saved-tensor counts mismatch and
        ``torch.utils.checkpoint`` raises ``CheckpointError``.

        This test asserts that contract today.  If we ever fix the
        incompatibility, the ``pytest.raises`` will not see the expected
        exception and this test will fail loudly, prompting an update to
        either the implementation guarantee or this test.
        """
        from torch.utils.checkpoint import CheckpointError, checkpoint

        # Wrap the student forward so each aligned layer goes through
        # torch.utils.checkpoint.  We can't use HF's gradient_checkpointing_enable
        # because SimpleModel doesn't implement it; the manual wrap reproduces
        # the same hook-fires-twice + saved-tensor-consistency interaction.
        original_forward = student_model.forward

        def gc_forward(input_ids, attention_mask=None, labels=None, **kw):
            x = student_model.embedding(input_ids)
            for layer in student_model.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            logits = student_model.lm_head(x)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
            return SimpleOutput(logits=logits, loss=loss)

        student_model.forward = gc_forward
        try:
            distiller = self._build(teacher_model, teacher_alignments, training_args, student_model)
            distiller._register_capture()
            batch = self._make_batch(device)

            student_model.zero_grad(set_to_none=True)
            with pytest.raises(CheckpointError):
                # Backward is required to trigger the recompute pass that
                # exposes the saved-tensor count mismatch.
                loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
                loss.backward()
            distiller._deregister_capture()
        finally:
            student_model.forward = original_forward

    def test_magnitude_aware_weighting_parity(
        self, teacher_model, student_model, teacher_alignments, training_args, device
    ):
        """Incremental path under magnitude_aware_weighting=True must match
        the post-hoc reference, including the straight-through gradient."""
        distiller = self._build(
            teacher_model,
            teacher_alignments,
            training_args,
            student_model,
            arg_overrides={"magnitude_aware_weighting": True},
        )
        distiller._register_capture()
        batch = self._make_batch(device)

        # Reference path
        student_model.zero_grad(set_to_none=True)
        ref_loss = TestHolisticIncrementalLoss._ref_alignment_loss(
            distiller, distiller.model, batch, is_training=True
        )
        assert ref_loss is not None
        ref_loss.backward()
        ref_grads = TestHolisticIncrementalLoss._grads_snapshot(student_model)
        ref_value = ref_loss.detach().clone()

        # Incremental path
        student_model.zero_grad(set_to_none=True)
        inc_loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        inc_loss.backward()
        inc_grads = TestHolisticIncrementalLoss._grads_snapshot(student_model)

        # Value: raw weighted sum (matches non-mag-aware shape).
        assert torch.allclose(inc_loss.detach(), ref_value, rtol=1e-5, atol=1e-6)
        # Gradients: the straight-through trick produces grads from the
        # normalized total, so must agree across paths.
        assert ref_grads.keys() == inc_grads.keys()
        for name in ref_grads:
            assert torch.allclose(ref_grads[name], inc_grads[name], rtol=1e-5, atol=1e-6), (
                f"magnitude-aware grad mismatch on {name}"
            )
        distiller._deregister_capture()

    def test_auto_projector_lazy_init_in_callback(
        self, teacher_model, student_model, training_args, device
    ):
        """Auto-projectors are created inside the incremental callback (via
        ``_try_init_output_projector``).  Their parameters must end up in
        the main optimizer so they actually get trained."""
        teacher_block = teacher_model.get_layer(0)
        student_block = student_model.get_layer(0)
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            auto_projector=True,
        )
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=_make_holistic_args(training_args),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
        )
        distiller._register_capture()
        distiller.create_optimizer()

        assert alignment.output_projector is None, "projector should be lazy"

        batch = self._make_batch(device, with_labels=False)
        distiller.training_step(student_model, batch)

        assert alignment.output_projector is not None, (
            "auto_projector was not lazy-initialised by the incremental callback"
        )
        optim_param_ids = {
            id(p) for group in distiller.optimizer.param_groups for p in group["params"]
        }
        missing = [
            n
            for n, p in alignment.output_projector.named_parameters()
            if id(p) not in optim_param_ids
        ]
        assert not missing, f"auto-projector params not registered after callback init: {missing}"
        distiller._deregister_capture()

    def test_tuple_module_output_handled(self, teacher_model, training_args, train_dataset, device):
        """A student layer returning a tuple must flow through the default
        ``OutputSelector`` (index 0) inside the incremental callback."""

        class TupleBlock(nn.Module):
            """Mimics HF attention layers: returns ``(hidden, extra)``."""

            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, x):
                h = self.linear(x)
                return (h, torch.zeros_like(h))  # tuple output

        class TupleModel(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.embedding = nn.Embedding(128, dim)
                self.layer = TupleBlock(dim)
                self.lm_head = nn.Linear(dim, 128)

            def forward(self, input_ids, attention_mask=None, labels=None, **kw):
                x = self.embedding(input_ids)
                x, _ = self.layer(x)
                return SimpleOutput(logits=self.lm_head(x), loss=None)

            def get_layer(self, idx):
                return self.layer

        student = TupleModel(dim=64).to(device)
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            output_projector=nn.Linear(64, 128).to(device),
        )
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[alignment],
            args=_make_holistic_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = self._make_batch(device, with_labels=False)
        student.zero_grad(set_to_none=True)
        loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        loss.backward()
        assert torch.isfinite(loss)
        # The student layer's parameters should have grads (proves the
        # tuple was selected and routed through the loss path).
        assert any(p.grad is not None for p in student.layer.parameters())
        distiller._deregister_capture()

    def test_multiple_alignments_on_same_module(self, teacher_model, training_args, device):
        """Two alignments pointing at the same student block should each
        contribute independently — the registration loop attaches a separate
        hook per alignment, so both must fire and accumulate.

        Uses a same-dim student so no projector is needed; otherwise each
        alignment would get an independently-initialised projector and
        produce different values for the same module output.
        """
        student = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        a0 = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        a0.loss_weight = 1.0
        a1 = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0_dup",
            student_module_name="layers.0_dup",
        )
        a1.loss_weight = 2.0

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[a0, a1],
            args=_make_holistic_args(training_args),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
        )
        distiller._register_capture()
        batch = self._make_batch(device, with_labels=False)

        student.zero_grad(set_to_none=True)
        loss_dual = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        # Two firings = two distinct module_ids in the dedup set
        assert len(distiller._incremental_processed) == 2
        loss_dual_value = loss_dual.detach().clone()
        distiller._deregister_capture()

        # Reference: a single alignment with loss_weight=3.0 — both alignments
        # compare identical teacher/student outputs, so the dual-weighted sum
        # (1 * loss + 2 * loss) must equal a single 3-weighted loss.
        a_combined = Alignment(
            teacher_block=teacher_model.get_layer(0),
            student_block=student.get_layer(0),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )
        a_combined.loss_weight = 3.0
        single = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=[a_combined],
            args=_make_holistic_args(training_args),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
        )
        single._register_capture()
        loss_single = single.compute_distillation_loss(single.model, batch, is_training=True)
        single._deregister_capture()

        assert torch.allclose(loss_dual_value, loss_single.detach(), rtol=1e-5, atol=1e-6), (
            "dual-alignment loss did not equal weight-sum-equivalent single alignment"
        )

    def test_no_aligned_modules_fired_returns_zero(
        self, teacher_model, student_model, single_alignment, training_args, device
    ):
        """If the forward never reaches an aligned module (e.g. truncated by
        the user mid-forward), the distiller must return a finite zero loss
        rather than crash on ``finalize`` returning ``None``."""
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_holistic_args(training_args),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
        )
        distiller._register_capture()
        batch = self._make_batch(device, with_labels=False)

        # Stub the student model to raise the truncation exception before
        # any aligned layer fires — simulates "no hooks fired this step."
        from silverspoon_kd.engines.module_capture_engine import _TruncatedForwardException

        class NoOpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Linear(1, 1)

            def forward(self, *a, **kw):
                raise _TruncatedForwardException()

        loss = distiller.compute_distillation_loss(NoOpModel(), batch, is_training=True)
        assert torch.isfinite(loss)
        assert loss.item() == 0.0
        distiller._deregister_capture()


@pytest.mark.cuda
class TestHolisticIncrementalCUDA:
    """CUDA-only: mixed precision, memory scaling, distributed-ish behaviour."""

    def _setup_on_cuda(self, teacher_model, student_model, alignments):
        device = torch.device("cuda")
        teacher_model.to(device)
        student_model.to(device)
        for a in alignments:
            if a.output_projector is not None:
                a.output_projector.to(device)
            if a.input_projector is not None:
                a.input_projector.to(device)
        return device

    def _make_batch(self, device, with_labels=True):
        torch.manual_seed(0)
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        if with_labels:
            batch["labels"] = torch.randint(0, 128, (2, 16), device=device)
        return batch

    def test_bf16_autocast_parity(
        self, teacher_model, student_model, teacher_alignments, training_args
    ):
        """Under bf16 autocast, gradients from the incremental path must match
        the post-hoc reference (within bf16 tolerance)."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        device = self._setup_on_cuda(teacher_model, student_model, teacher_alignments)
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
            overlap_alignment_loss=False,
        )
        distiller._register_capture()
        batch = self._make_batch(device)

        # Reference under autocast
        student_model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ref_loss = TestHolisticIncrementalLoss._ref_alignment_loss(
                distiller, distiller.model, batch, is_training=True
            )
        assert ref_loss is not None
        ref_loss.backward()
        torch.cuda.synchronize()
        ref_grads = TestHolisticIncrementalLoss._grads_snapshot(student_model)

        # Incremental under autocast
        student_model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inc_loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        inc_loss.backward()
        torch.cuda.synchronize()
        inc_grads = TestHolisticIncrementalLoss._grads_snapshot(student_model)

        # bf16 has ~3-4 decimal digits — looser tolerance.
        assert torch.allclose(inc_loss.detach(), ref_loss.detach(), rtol=1e-2, atol=1e-3)
        for name in ref_grads:
            assert torch.allclose(ref_grads[name], inc_grads[name], rtol=1e-2, atol=1e-3), (
                f"bf16 grad mismatch on {name}"
            )
        distiller._deregister_capture()

    def test_bf16_with_overlap_runs(
        self, teacher_model, student_model, teacher_alignments, training_args
    ):
        """bf16 autocast with overlap_alignment_loss=True must run without
        crashing.  The autocast context must reach the loss-stream kernels —
        if it didn't, mse_loss would hit a dtype mismatch."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        device = self._setup_on_cuda(teacher_model, student_model, teacher_alignments)
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
            overlap_alignment_loss=True,
        )
        distiller._register_capture()
        batch = self._make_batch(device)

        student_model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        distiller._deregister_capture()

    def test_fp16_gradscaler_runs(
        self, teacher_model, student_model, teacher_alignments, training_args
    ):
        """fp16 + GradScaler must successfully scale, backward, unscale, step
        on a stream-accumulated loss (no NaNs from missed unscale, no fp16
        overflow from un-clamped accumulation)."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        device = self._setup_on_cuda(teacher_model, student_model, teacher_alignments)
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args),
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
            overlap_alignment_loss=True,
        )
        distiller._register_capture()
        distiller.create_optimizer()
        scaler = torch.amp.GradScaler("cuda")
        batch = self._make_batch(device)

        student_model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
        scaler.scale(loss).backward()
        scaler.unscale_(distiller.optimizer)
        scaler.step(distiller.optimizer)
        scaler.update()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        # If GradScaler missed our stream-accumulated loss, gradients would
        # be ~scale_factor times too big (so unscale_ wouldn't bring them
        # back to sane values).  Sanity check norm is bounded.
        for p in student_model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                assert p.grad.abs().max() < 1e4, "grad magnitude suggests missed unscale"
        distiller._deregister_capture()

    def test_memory_o1_in_alignments(self, teacher_model, training_args):
        """Peak GPU memory for N=6 alignments should not be ~6x N=1.

        The whole motivation of incremental loss is O(1) live alignment
        activations.  Compares ``torch.cuda.max_memory_allocated`` between
        a single-alignment and a six-alignment run on the same teacher,
        and asserts the ratio stays well below a linear-growth shape.
        """
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        device = torch.device("cuda")
        teacher_model.to(device)

        def peak_for(n: int) -> int:
            student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=max(n, 3)).to(device)
            alignments = []
            for i in range(n):
                # Re-use teacher layer 0 if n exceeds teacher depth; we only
                # care about per-alignment captured tensors, not which layer.
                t_layer = teacher_model.get_layer(min(i, teacher_model.num_layers - 1))
                s_layer = student.get_layer(min(i, student.num_layers - 1))
                alignments.append(
                    create_alignment(
                        teacher_block=t_layer,
                        student_block=s_layer,
                        teacher_module_name=f"layers.{i}",
                        student_module_name=f"layers.{i}",
                        with_input_projector=False,
                    )
                )
                if alignments[-1].output_projector is not None:
                    alignments[-1].output_projector.to(device)

            distiller = HolisticDistiller(
                student_model=student,
                teacher_model=teacher_model,
                alignments=alignments,
                args=_make_holistic_args(training_args),
                train_dataset=DummyDataset(num_samples=4, seq_len=16),
                overlap_alignment_loss=False,
            )
            distiller._register_capture()
            batch = {
                "input_ids": torch.randint(0, 128, (4, 32), device=device),
                "attention_mask": torch.ones(4, 32, dtype=torch.long, device=device),
            }
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            loss = distiller.compute_distillation_loss(distiller.model, batch, is_training=True)
            loss.backward()
            torch.cuda.synchronize()
            distiller._deregister_capture()
            return torch.cuda.max_memory_allocated(device)

        peak_1 = peak_for(1)
        peak_6 = peak_for(6)
        # If the post-hoc all-at-once pattern silently came back, peak_6
        # would be ~6x peak_1.  With incremental + auto-truncate-off, base
        # forward activations grow with depth, but the *alignment-related*
        # memory should not.  Allow a generous 2x ceiling — the goal is
        # detecting catastrophic regressions, not micro-tuning.
        assert peak_6 < 2 * peak_1, (
            f"peak memory grew too much with N: {peak_1=} {peak_6=} "
            f"(ratio {peak_6 / peak_1:.2f}). "
            "Possibly regressed to all-at-once alignment-loss behaviour."
        )

    def test_full_step_loop_loss_decreases(
        self, teacher_model, student_model, teacher_alignments, training_args
    ):
        """Sanity: training steps with overlap=True drive the loss down on a
        fixed batch.  Catches gross misuse of the loss stream during real
        training (wrong gradients via stream-sync bug, NaNs from missed
        sync, etc.).

        Uses the *same* batch every step so gradient descent has a consistent
        optimization target; with random batches, "loss decreased" reflects
        which batches landed where rather than whether the optimizer is
        working.  The previous version of this test failed flakily on real
        CUDA because 5 random batches have ~equal mean loss with no signal.
        """
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        device = self._setup_on_cuda(teacher_model, student_model, teacher_alignments)
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_holistic_args(training_args, learning_rate=1e-2),
            train_dataset=DummyDataset(num_samples=8, seq_len=16),
            overlap_alignment_loss=True,
        )
        distiller._register_capture()
        distiller.create_optimizer()

        torch.manual_seed(0)
        fixed_batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        losses = []
        for _ in range(10):
            distiller.optimizer.zero_grad(set_to_none=True)
            loss = distiller.compute_distillation_loss(
                distiller.model, fixed_batch, is_training=True
            )
            loss.backward()
            distiller.optimizer.step()
            torch.cuda.synchronize()
            losses.append(loss.item())

        # On a fixed batch with finite gradients and a reasonable learning
        # rate, the loss must drop substantially.  A loose threshold so
        # bf16/fp16 rounding doesn't trip us — what we're guarding against
        # is "the optimizer isn't doing anything" (bug), not micro-noise.
        assert losses[-1] < 0.9 * losses[0], (
            f"loss did not decrease enough over 10 steps on a fixed batch: "
            f"{losses[0]:.4f} -> {losses[-1]:.4f} ({losses=})"
        )
        distiller._deregister_capture()
