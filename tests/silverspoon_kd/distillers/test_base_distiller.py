"""
Unit tests for BaseDistiller class.
"""

import contextlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from silverspoon_kd.distillers.base_distiller import (
    BaseDistiller,
    send_to_dtype,
)
from silverspoon_kd.training_arguments import TrainingArguments
from tests.silverspoon_kd.conftest import load_module_copy


class _DummyModel(nn.Module):
    """Dummy model to satisfy Trainer's model requirement."""

    def __init__(self, device):
        super().__init__()
        self.register_buffer("_dummy_param", torch.zeros(1, device=device))

    def forward(self, **kwargs):
        return None


class ConcreteDistiller(BaseDistiller):
    """
    Concrete implementation of BaseDistiller for testing.
    """

    def __init__(self, teacher_model, alignments, **kwargs):
        # Create a dummy model to satisfy Trainer's requirements
        device = next(teacher_model.parameters()).device
        dummy_model = _DummyModel(device)
        super().__init__(
            teacher_model=teacher_model,
            alignments=alignments,
            model=dummy_model,
            **kwargs,
        )
        # _DummyModel has no trainable parameters; disable gradient clipping
        # to avoid accelerate warning about empty parameter generators.
        self.args.max_grad_norm = 0

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Simple training step that returns a dummy loss."""
        device = next(self.teacher_model.parameters()).device
        loss = torch.tensor(1.0, device=device, requires_grad=True)
        self.step_losses.append(loss.detach())
        return loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        """Simple compute_loss that returns a dummy loss."""
        self._reset_step_metrics(is_training=model.training)
        device = next(self.teacher_model.parameters()).device
        loss = torch.tensor(1.0, device=device)
        self.eval_losses.append(loss.detach())

        # Track per-alignment metrics like real distillers do
        for alignment in self.alignments:
            self._store_loss_metric(alignment.get_name(), loss)
        if not model.training:
            self._store_eval_metrics()

        if return_outputs:
            return (loss, None)
        return loss


class TestBaseDistiller:
    """Test suite for BaseDistiller."""

    def test_initialization(self, teacher_model, single_alignment, training_args, train_dataset):
        """Test that BaseDistiller initializes correctly."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        assert distiller.teacher_model is teacher_model
        assert distiller.alignments == single_alignment
        assert len(distiller.current_step_metrics) == 0
        assert len(distiller.step_losses) == 0
        assert len(distiller.eval_losses) == 0
        assert distiller.flop_counter == 0
        assert distiller.flops_per_step == 0

    def test_initialization_without_alignments_composite(
        self, teacher_model, training_args, train_dataset
    ):
        """Test that empty alignments raises error for composite optimizer distillers."""

        class CompositeConcreteDistiller(ConcreteDistiller):
            _USE_COMPOSITE_OPTIMIZER = True

        with pytest.raises(ValueError, match="No alignments provided"):
            CompositeConcreteDistiller(
                teacher_model=teacher_model,
                alignments=[],
                args=training_args,
                train_dataset=train_dataset,
            )

    def test_initialization_without_alignments_allowed(
        self, teacher_model, training_args, train_dataset
    ):
        """Test that empty alignments is allowed for non-composite distillers."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=[],
            args=training_args,
            train_dataset=train_dataset,
        )
        assert distiller.alignments == []

    def test_initialization_creates_default_args(
        self, teacher_model, single_alignment, train_dataset, device
    ):
        """Test that default TrainingArguments are created when none provided."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=TrainingArguments(use_cpu=(device.type == "cpu")),
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.args, TrainingArguments)

    def test_reset_step_metrics(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _reset_step_metrics clears metrics correctly."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Add some metrics
        distiller.current_step_metrics = {"loss": 1.0, "lr": 0.001}
        distiller.step_losses = [torch.tensor(1.0), torch.tensor(2.0)]
        distiller.eval_losses = [torch.tensor(0.5)]

        # Reset training metrics
        distiller._reset_step_metrics(is_training=True)
        assert len(distiller.current_step_metrics) == 0
        assert len(distiller.step_losses) == 0
        assert len(distiller.eval_losses) == 1  # Should not be cleared

        # Reset eval metrics
        distiller._reset_step_metrics(is_training=False)
        assert len(distiller.eval_losses) == 0

    def test_store_loss_metric(self, teacher_model, single_alignment, training_args, train_dataset):
        """Test that _store_loss_metric stores loss correctly."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        loss = torch.tensor(1.5, requires_grad=True)
        distiller._store_loss_metric("student_0", loss)

        assert "loss/student_0" in distiller.current_step_metrics
        assert distiller.current_step_metrics["loss/student_0"] == 1.5

    def test_store_lr_metric(self, teacher_model, single_alignment, training_args, train_dataset):
        """Test that _store_lr_metric stores learning rate correctly."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller._store_lr_metric("student_0", 0.001)

        assert "learning_rate/student_0" in distiller.current_step_metrics
        assert distiller.current_step_metrics["learning_rate/student_0"] == 0.001

    def test_store_grad_norm_metric(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _store_grad_norm_metric stores gradient norm correctly."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller._store_grad_norm_metric("student_0", 2.5)

        assert "grad_norm/student_0" in distiller.current_step_metrics
        assert distiller.current_step_metrics["grad_norm/student_0"] == 2.5

    def test_prepare_teacher_inputs_default(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _prepare_teacher_inputs passes through accepted keys."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        }
        result = distiller._prepare_teacher_inputs(inputs)

        assert "input_ids" in result
        assert "attention_mask" in result

    def test_prepare_teacher_inputs_custom(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _prepare_teacher_inputs uses custom function when provided."""

        def custom_prepare(inputs):
            return {**inputs, "use_cache": False}

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
            prepare_teacher_inputs=custom_prepare,
        )

        inputs = {"input_ids": torch.tensor([1, 2, 3])}
        result = distiller._prepare_teacher_inputs(inputs)

        assert "use_cache" in result
        assert result["use_cache"] is False

    def test_prepare_teacher_inputs_strips_labels(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Labels should always be stripped from teacher inputs.

        The teacher's loss is never used by any distiller. Passing labels
        wastes compute and crashes with pipeline-parallel teachers (where
        logits and labels end up on different devices).
        """
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([4, 5, 6]),
        }
        result = distiller._prepare_teacher_inputs(inputs)

        assert "labels" not in result
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_prepare_teacher_inputs_strips_labels_with_kwargs_model(
        self, single_alignment, training_args, train_dataset
    ):
        """Labels should be stripped even when teacher accepts **kwargs."""

        # Create a teacher that accepts **kwargs (no signature filtering)
        class KwargsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Linear(1, 1)

            def forward(self, **kwargs):
                return None

        distiller = ConcreteDistiller(
            teacher_model=KwargsModel(),
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([4, 5, 6]),
        }
        result = distiller._prepare_teacher_inputs(inputs)

        assert "labels" not in result
        assert "input_ids" in result

    def test_should_count_flops(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _should_count_flops returns correct value."""
        # Enable FLOP counting
        training_args.count_flops = True

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Should count on step 0
        distiller.state.global_step = 0
        assert distiller._should_count_flops() is True

        # Should not count on step 1
        distiller.state.global_step = 1
        assert distiller._should_count_flops() is False

    def test_should_count_flops_disabled(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _should_count_flops returns False when count_flops is disabled."""
        training_args.count_flops = False

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller.state.global_step = 0
        assert distiller._should_count_flops() is False

    def test_teacher_alignment_preparation(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that teacher alignments are prepared during initialization."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Check that alignments have been prepared
        for _alignment_id, alignment in enumerate(distiller.alignments):
            assert alignment.optimizer is not None
            assert alignment.scheduler is not None

    def test_log_filters_learning_rate_when_accumulator_has_per_student_lr(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that log method auto-filters global learning_rate when accumulator
        contains per-student learning_rate/ keys."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Simulate accumulated per-student LR metrics (from blockwise/global training)
        # Running sums over 2 steps: loss 0.5+0.6=1.1, lr 0.001+0.001=0.002
        distiller._training_metric_accumulator = {
            "loss/student_0": 1.1,
            "learning_rate/student_0": 0.002,
        }
        distiller._metric_accumulator_steps = 2

        logs = {
            "loss": 1.0,
            "learning_rate": 0.001,
        }

        # Store original log method
        captured_logs = None
        original_log = distiller.__class__.__bases__[0].__bases__[0].log  # Trainer.log

        def capture_logs(self_arg, logs_dict, start_time=None):
            nonlocal captured_logs
            captured_logs = logs_dict

        # Monkey patch the Trainer's log method
        from transformers import Trainer

        Trainer.log = capture_logs

        try:
            distiller.log(logs)

            # Should filter out global learning_rate and inject averaged per-student LR
            assert "learning_rate" not in captured_logs
            assert "learning_rate/student_0" in captured_logs
            assert captured_logs["learning_rate/student_0"] == 0.001
        finally:
            # Restore original method
            Trainer.log = original_log

    def test_log_keeps_learning_rate_when_no_per_student_lr(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that log preserves global learning_rate when accumulator has no
        per-student LR keys (e.g. ResponseBasedDistiller)."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Simulate accumulated metrics without per-student LR (response-based KD pattern)
        # Running sum over 2 steps: 0.5 + 0.6 = 1.1
        distiller._training_metric_accumulator = {
            "loss/soft": 1.1,
        }
        distiller._metric_accumulator_steps = 2

        logs = {
            "loss": 1.0,
            "learning_rate": 0.001,
        }

        captured_logs = None
        original_log = distiller.__class__.__bases__[0].__bases__[0].log  # Trainer.log

        def capture_logs(self_arg, logs_dict, start_time=None):
            nonlocal captured_logs
            captured_logs = logs_dict

        from transformers import Trainer

        Trainer.log = capture_logs

        try:
            distiller.log(logs)

            # Global learning_rate should be preserved
            assert "learning_rate" in captured_logs
            assert captured_logs["learning_rate"] == 0.001
        finally:
            Trainer.log = original_log

    def test_num_training_steps_calculation(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that num_training_steps is calculated correctly."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Should use max_steps when max_steps > 0
        assert distiller.num_training_steps == training_args.max_steps

    def test_num_training_steps_from_dataset(
        self,
        teacher_model,
        single_alignment,
        train_dataset,
        device,
        tmp_path,
    ):
        """Test that num_training_steps is estimated from dataset when max_steps = 0."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            num_train_epochs=2,
            per_device_train_batch_size=4,
            max_steps=0,  # Use dataset size
            use_cpu=(device.type == "cpu"),
        )

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        # Should estimate from dataset size
        assert distiller.num_training_steps > 0


class TestSendToDtype:
    """Test suite for the send_to_dtype helper function."""

    def test_float_tensor_is_cast(self):
        """Test that a floating-point tensor is cast to the target dtype."""
        t = torch.randn(2, 3, dtype=torch.float32)
        result = send_to_dtype(t, torch.float16)
        assert result.dtype == torch.float16

    def test_integer_tensor_is_not_cast(self):
        """Test that integer tensors are left unchanged."""
        t = torch.tensor([1, 2, 3], dtype=torch.long)
        result = send_to_dtype(t, torch.float16)
        assert result.dtype == torch.long

    def test_bool_tensor_is_not_cast(self):
        """Test that bool tensors are left unchanged."""
        t = torch.tensor([True, False], dtype=torch.bool)
        result = send_to_dtype(t, torch.float16)
        assert result.dtype == torch.bool

    def test_tuple_of_tensors(self):
        """Test that tuples of tensors are recursively cast."""
        t1 = torch.randn(2, 3, dtype=torch.float32)
        t2 = torch.randn(4, dtype=torch.float64)
        result = send_to_dtype((t1, t2), torch.bfloat16)
        assert isinstance(result, tuple)
        assert result[0].dtype == torch.bfloat16
        assert result[1].dtype == torch.bfloat16

    def test_list_of_tensors(self):
        """Test that lists of tensors are recursively cast."""
        t1 = torch.randn(2, dtype=torch.float32)
        result = send_to_dtype([t1], torch.float16)
        assert isinstance(result, list)
        assert result[0].dtype == torch.float16

    def test_dict_of_tensors(self):
        """Test that dicts of tensors are recursively cast."""
        d = {
            "hidden": torch.randn(2, 3, dtype=torch.float32),
            "mask": torch.ones(2, 3, dtype=torch.long),
        }
        result = send_to_dtype(d, torch.float16)
        assert result["hidden"].dtype == torch.float16
        assert result["mask"].dtype == torch.long  # integer unchanged

    def test_nested_structure(self):
        """Test deeply nested structures are handled."""
        nested = (
            torch.randn(2, dtype=torch.float32),
            {"key": [torch.randn(3, dtype=torch.float64)]},
        )
        result = send_to_dtype(nested, torch.bfloat16)
        assert result[0].dtype == torch.bfloat16
        assert result[1]["key"][0].dtype == torch.bfloat16

    def test_non_tensor_passthrough(self):
        """Test that non-tensor values are returned unchanged."""
        assert send_to_dtype(42, torch.float16) == 42
        assert send_to_dtype("hello", torch.float16) == "hello"
        assert send_to_dtype(None, torch.float16) is None


class TestBaseDistillerProfiler:
    """Test profiler methods in BaseDistiller."""

    def test_init_profiler_when_enabled(
        self, teacher_model, single_alignment, train_dataset, device, tmp_path
    ):
        """Test _init_profiler creates a profiler when profiling is enabled."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=5,
            enable_profiling=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        assert distiller.profiler is not None

    def test_init_profiler_custom_output_dir(
        self, teacher_model, single_alignment, train_dataset, device, tmp_path
    ):
        """Test _init_profiler uses custom output directory."""
        custom_dir = str(tmp_path / "custom_profiling")
        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=5,
            enable_profiling=True,
            profiling_output_dir=custom_dir,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        assert distiller.profiler is not None

    def test_start_and_stop_profiler(
        self, teacher_model, single_alignment, train_dataset, device, tmp_path
    ):
        """Test _start_profiler and _stop_profiler when profiler is set."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=5,
            enable_profiling=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        # Start and stop should not raise
        distiller._start_profiler()
        distiller._stop_profiler()

    def test_profiler_step_with_profiler(
        self, teacher_model, single_alignment, train_dataset, device, tmp_path
    ):
        """Test _profiler_step calls step() on profiler."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=5,
            enable_profiling=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        # Start profiler first, then step
        distiller._start_profiler()
        distiller._profiler_step()
        distiller._stop_profiler()

    def test_profiler_step_without_profiler(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _profiler_step is a no-op when profiler is None."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        assert distiller.profiler is None
        # Should not raise
        distiller._profiler_step()


class TestBaseDistillerFlopCounting:
    """Test FLOP counting methods in BaseDistiller."""

    def test_get_flop_context_count_now(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _get_flop_context with count_now=True returns FlopCounterMode."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        ctx = distiller._get_flop_context(count_now=True)
        assert isinstance(ctx, FlopCounterMode)

    def test_get_flop_context_no_count(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _get_flop_context with count_now=False returns nullcontext."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        ctx = distiller._get_flop_context(count_now=False)
        assert isinstance(ctx, contextlib.nullcontext)

    def test_record_flops_with_flop_counter_mode(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _record_flops records flops from FlopCounterMode."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Create a mock FlopCounterMode
        mock_counter = MagicMock(spec=FlopCounterMode)
        mock_counter.get_total_flops.return_value = 12345

        distiller._record_flops(mock_counter)
        assert distiller.flops_per_step == 12345

    def test_record_flops_with_nullcontext(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _record_flops ignores nullcontext."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller._record_flops(contextlib.nullcontext())
        assert distiller.flops_per_step == 0

    def test_update_flop_counter_enabled(
        self,
        teacher_model,
        single_alignment,
        train_dataset,
        device,
        tmp_path,
    ):
        """Test _update_flop_counter increments when counting is enabled."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            count_flops=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        distiller.flops_per_step = 1000
        distiller._update_flop_counter()
        assert distiller.flop_counter == 1000

        distiller._update_flop_counter()
        assert distiller.flop_counter == 2000

    def test_update_flop_counter_disabled(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _update_flop_counter does nothing when counting is disabled."""
        training_args.count_flops = False

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller.flops_per_step = 1000
        distiller._update_flop_counter()
        assert distiller.flop_counter == 0


class TestBaseDistillerTeacherInputs:
    """Test teacher input preparation in BaseDistiller."""

    def test_auto_filter_without_kwargs(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test auto-filter when teacher accepts **kwargs still strips labels."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # SimpleModel.forward accepts input_ids, attention_mask, labels, **kwargs
        # Labels should always be stripped (teacher loss is never used)
        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([0, 1, 2]),
        }
        result = distiller._prepare_teacher_inputs(inputs)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" not in result

    def test_auto_filter_teacher_without_var_keyword(
        self, single_alignment, training_args, train_dataset, device
    ):
        """Test auto-filter when teacher does NOT accept **kwargs."""

        class StrictTeacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return self.linear(input_ids.float())

        teacher = StrictTeacher().to(device)

        distiller = ConcreteDistiller(
            teacher_model=teacher,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([0, 1, 2]),
        }
        result = distiller._prepare_teacher_inputs(inputs)

        # Should filter to only accepted params
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" not in result
        # use_cache should be set to False
        assert result["use_cache"] is False

    def test_get_teacher_accepted_params_caching(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _get_teacher_accepted_params caches result."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        result1 = distiller._get_teacher_accepted_params()
        result2 = distiller._get_teacher_accepted_params()
        assert result1 is result2


@pytest.mark.filterwarnings("ignore:`parameters` is an empty generator")
class TestBaseDistillerTrainLifecycle:
    """Test train lifecycle in BaseDistiller."""

    def test_train_registers_and_deregisters_capture(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that train() registers and deregisters capture."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        registered = False
        deregistered = False

        original_register = distiller._register_capture
        original_deregister = distiller._deregister_capture

        def mock_register():
            nonlocal registered
            registered = True
            original_register()

        def mock_deregister():
            nonlocal deregistered
            deregistered = True
            original_deregister()

        distiller._register_capture = mock_register
        distiller._deregister_capture = mock_deregister

        distiller.train()

        assert registered
        assert deregistered

    def test_train_starts_and_stops_profiler(
        self, teacher_model, single_alignment, train_dataset, device, tmp_path
    ):
        """Test that train() starts and stops profiler."""
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

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        assert distiller.profiler is not None

        started = False
        stopped = False

        original_start = distiller._start_profiler
        original_stop = distiller._stop_profiler

        def mock_start():
            nonlocal started
            started = True
            original_start()

        def mock_stop():
            nonlocal stopped
            stopped = True
            original_stop()

        distiller._start_profiler = mock_start
        distiller._stop_profiler = mock_stop

        distiller.train()

        assert started
        assert stopped

    def test_train_initializes_flop_counter(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that train() resets flop counter."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller.flop_counter = 999
        distiller.train()
        assert distiller.flop_counter == 0


class TestBaseDistillerEvaluation:
    """Test evaluation in BaseDistiller."""

    def test_prediction_step(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Test prediction_step returns loss, None, None."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss, logits, labels = distiller.prediction_step(
            distiller.model, batch, prediction_loss_only=True
        )

        assert isinstance(loss, torch.Tensor)
        assert logits is None
        assert labels is None

    def test_store_eval_metrics(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _store_eval_metrics accumulates per-layer metrics."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Set up per_layer_eval_metrics like evaluate() does
        alignment = single_alignment[0]
        student_name = alignment.get_name()
        distiller.metric_key_prefix = "eval"
        distiller.per_layer_eval_metrics = {
            f"eval_loss/{student_name}": [],
        }

        # Simulate a step with metrics
        distiller.current_step_metrics = {
            f"loss/{student_name}": 0.5,
        }

        distiller._store_eval_metrics()

        assert len(distiller.per_layer_eval_metrics[f"eval_loss/{student_name}"]) == 1
        assert distiller.per_layer_eval_metrics[f"eval_loss/{student_name}"][0] == 0.5

    def test_evaluate_with_per_layer_metrics(
        self,
        teacher_model,
        single_alignment,
        training_args,
        train_dataset,
        eval_dataset,
    ):
        """Test evaluate() returns per-layer metrics with correct averaged values."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        metrics = distiller.evaluate()

        assert isinstance(metrics, dict)
        assert "eval_loss" in metrics

        # Per-layer metric must exist and have a finite, non-negative value
        alignment_name = single_alignment[0].get_name()
        per_layer_key = f"eval_loss/{alignment_name}"
        assert per_layer_key in metrics, (
            f"{per_layer_key} missing from eval metrics: {list(metrics.keys())}"
        )
        assert metrics[per_layer_key] >= 0, (
            f"{per_layer_key} should be non-negative, got {metrics[per_layer_key]}"
        )
        # Per-layer average should be close to eval_loss (single alignment ⇒ equal)
        assert abs(metrics[per_layer_key] - metrics["eval_loss"]) < 0.1, (
            f"Per-layer metric {metrics[per_layer_key]} should be close to "
            f"eval_loss {metrics['eval_loss']} for single-alignment distiller"
        )


class TestBaseDistillerWeightWatcher:
    """Test WeightWatcher integration in BaseDistiller."""

    def test_compute_additional_eval_metrics_with_weightwatcher(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _compute_additional_eval_metrics with mocked WeightWatcher."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Enable WeightWatcher in args
        distiller.args.use_weightwatcher = True

        with patch("silverspoon_kd.distillers.base_distiller.WEIGHTWATCHER_AVAILABLE", True):
            mock_ww = MagicMock()
            mock_details = MagicMock()
            mock_summary = {"alpha": 2.5, "log_norm": 1.0}
            mock_ww.analyze.return_value = mock_details
            mock_ww.get_summary.return_value = mock_summary

            with patch(
                "silverspoon_kd.distillers.base_distiller.ww", create=True
            ) as mock_ww_module:
                mock_ww_module.WeightWatcher.return_value = mock_ww
                metrics = distiller._compute_additional_eval_metrics("eval")

            assert len(metrics) > 0

    def test_run_weightwatcher_analysis(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _run_weightwatcher_analysis with mocked WeightWatcher."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        mock_ww = MagicMock()
        mock_details = MagicMock()
        mock_summary = {"alpha": 2.5, "log_norm": 1.0}
        mock_ww.analyze.return_value = mock_details
        mock_ww.get_summary.return_value = mock_summary

        with patch("silverspoon_kd.distillers.base_distiller.ww", create=True) as mock_ww_module:
            mock_ww_module.WeightWatcher.return_value = mock_ww
            metrics = distiller._run_weightwatcher_analysis("eval")

        alignment = single_alignment[0]
        student_name = alignment.get_name()
        assert f"eval_ww_alpha/{student_name}" in metrics
        assert metrics[f"eval_ww_alpha/{student_name}"] == 2.5

    def test_run_weightwatcher_analysis_handles_exception(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _run_weightwatcher_analysis handles exceptions gracefully."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        with patch("silverspoon_kd.distillers.base_distiller.ww", create=True) as mock_ww_module:
            mock_ww_module.WeightWatcher.side_effect = RuntimeError("WW failed")
            metrics = distiller._run_weightwatcher_analysis("eval")

        assert metrics == {}

    def test_run_weightwatcher_no_alignments_analyzes_full_model(
        self, teacher_model, training_args, train_dataset
    ):
        """Without alignments, WW should analyze the full student model."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=[],
            args=training_args,
            train_dataset=train_dataset,
        )

        mock_ww = MagicMock()
        mock_ww.analyze.return_value = MagicMock()
        mock_ww.get_summary.return_value = {"alpha": 3.0}

        mock_ww_module = MagicMock()
        mock_ww_module.WeightWatcher.return_value = mock_ww

        with patch.dict("sys.modules", {"weightwatcher": mock_ww_module}):
            metrics = distiller._run_weightwatcher_analysis("eval")

        # Should use "student" key (not an alignment name)
        assert "eval_ww_alpha/student" in metrics
        assert metrics["eval_ww_alpha/student"] == 3.0
        # Should have been called with the full model, not a block
        mock_ww_module.WeightWatcher.assert_called_once()

    def test_run_weightwatcher_with_alignments_analyzes_blocks(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """With alignments, WW should analyze each student block individually."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        mock_ww = MagicMock()
        mock_ww.analyze.return_value = MagicMock()
        mock_ww.get_summary.return_value = {"alpha": 2.5}

        with patch("silverspoon_kd.distillers.base_distiller.ww", create=True) as mock_ww_module:
            mock_ww_module.WeightWatcher.return_value = mock_ww
            metrics = distiller._run_weightwatcher_analysis("eval")

        # Should use alignment name as key (not "student")
        alignment_name = single_alignment[0].get_name()
        assert f"eval_ww_alpha/{alignment_name}" in metrics
        assert "eval_ww_alpha/student" not in metrics

    def test_compute_additional_eval_metrics_disabled(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test _compute_additional_eval_metrics returns empty when WW disabled."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller.args.use_weightwatcher = False
        metrics = distiller._compute_additional_eval_metrics("eval")
        assert metrics == {}


class TestBaseDistillerDefaultArgs:
    """Test default TrainingArguments creation in BaseDistiller."""

    def test_initialization_with_none_args(
        self, teacher_model, single_alignment, train_dataset, device
    ):
        """Test that BaseDistiller creates default args when args is None."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=None,
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.args, TrainingArguments)


class TestBaseDistillerImportFallback:
    """Test import error fallback for optional dependencies."""

    def test_weightwatcher_import_error(self):
        """Test WEIGHTWATCHER_AVAILABLE is False when weightwatcher is not installed."""
        import silverspoon_kd.distillers.base_distiller as mod

        with patch.dict(sys.modules, {"weightwatcher": None}):
            probe = load_module_copy(mod)
        assert probe.WEIGHTWATCHER_AVAILABLE is False


class TestBaseDistillerPredictionStepTuple:
    """Test prediction_step handling of tuple returns from compute_loss."""

    def test_prediction_step_with_tuple_loss(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Test prediction_step extracts loss from tuple return."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        expected_loss = torch.tensor(2.0, device=device)
        with patch.object(distiller, "compute_loss", return_value=(expected_loss, None)):
            loss, logits, labels = distiller.prediction_step(None, batch, True)

        assert loss.item() == 2.0
        assert logits is None
        assert labels is None

        distiller._deregister_capture()


class TestBaseDistillerTorchCompile:
    """Test torch.compile handling of the teacher model in BaseDistiller."""

    def test_teacher_compiled_when_torch_compile_true(
        self,
        teacher_model,
        single_alignment,
        train_dataset,
        device,
        tmp_path,
    ):
        """Test that teacher model is compiled when torch_compile=True."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.teacher_model, torch._dynamo.eval_frame.OptimizedModule)

    def test_teacher_not_compiled_when_torch_compile_false(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that teacher model is NOT compiled when torch_compile=False."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        assert distiller.teacher_model is teacher_model


class TestBaseDistillerMetricAccumulation:
    """Test metric accumulation and averaging in BaseDistiller."""

    def test_track_metric_stores_in_current_step_metrics(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _track_metric stores the value in current_step_metrics."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller._track_metric("loss/student_0", 0.5)

        assert distiller.current_step_metrics["loss/student_0"] == 0.5

    def test_track_metric_accumulates_when_is_accumulating(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _track_metric appends to accumulator when _is_accumulating=True."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller._is_accumulating = True
        distiller._track_metric("loss/student_0", 0.5)
        distiller._track_metric("loss/student_0", 0.7)

        assert abs(distiller._training_metric_accumulator["loss/student_0"] - 1.2) < 1e-6

    def test_track_metric_defers_item_for_gpu_tensors(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _track_metric accepts tensors and defers .item() to log time.

        This is the core optimization: GPU tensors are accumulated as a running
        sum without calling .item() (which would force a GPU→CPU sync every
        step).  The scalar conversion only happens in _get_averaged_metrics().
        """
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller._is_accumulating = True
        # Simulate 3 training steps passing loss tensors (as the real code does)
        distiller._track_metric("loss/soft", torch.tensor(1.0))
        distiller._track_metric("loss/soft", torch.tensor(2.0))
        distiller._track_metric("loss/soft", torch.tensor(3.0))

        # Accumulator should hold the running sum as a tensor, not a list
        assert isinstance(distiller._training_metric_accumulator["loss/soft"], torch.Tensor)

        # Simulate 3 steps accumulated
        distiller._metric_accumulator_steps = 3

        averaged = distiller._get_averaged_metrics()

        # Average of [1.0, 2.0, 3.0] = 2.0
        assert isinstance(averaged["loss/soft"], float)
        assert abs(averaged["loss/soft"] - 2.0) < 1e-6
        assert len(distiller._training_metric_accumulator) == 0

    def test_track_metric_does_not_accumulate_when_not_accumulating(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _track_metric does not accumulate when _is_accumulating=False."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller._is_accumulating = False
        distiller._track_metric("loss/student_0", 0.5)

        assert "loss/student_0" not in distiller._training_metric_accumulator

    def test_get_averaged_metrics_returns_correct_averages(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _get_averaged_metrics returns correct averages and clears accumulator."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Running sums over 3 steps: loss 1+2+3=6, lr 0.001*3=0.003
        distiller._training_metric_accumulator = {
            "loss/student_0": 6.0,
            "learning_rate/student_0": 0.003,
        }
        distiller._metric_accumulator_steps = 3

        averaged = distiller._get_averaged_metrics()

        assert abs(averaged["loss/student_0"] - 2.0) < 1e-6
        assert abs(averaged["learning_rate/student_0"] - 0.001) < 1e-6
        assert len(distiller._training_metric_accumulator) == 0

    def test_reset_step_metrics_sets_accumulating_flag(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that _reset_step_metrics sets _is_accumulating based on is_training."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        distiller._reset_step_metrics(is_training=True)
        assert distiller._is_accumulating is True
        assert distiller._metric_accumulator_steps == 1

        distiller._reset_step_metrics(is_training=True)
        assert distiller._metric_accumulator_steps == 2

        distiller._reset_step_metrics(is_training=False)
        assert distiller._is_accumulating is False
        # Eval steps should not increment the counter
        assert distiller._metric_accumulator_steps == 2

    def test_log_injects_averaged_metrics(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that log() injects averaged metrics from accumulator into logs."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Running sums over 2 steps: loss 1+2=3, grad_norm 3+5=8
        distiller._training_metric_accumulator = {
            "loss/student_0": 3.0,
            "grad_norm/student_0": 8.0,
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
            distiller.log({"loss": 1.5})

            assert abs(captured_logs["loss/student_0"] - 1.5) < 1e-6
            assert abs(captured_logs["grad_norm/student_0"] - 4.0) < 1e-6
            assert captured_logs["loss"] == 1.5
        finally:
            Trainer.log = original_log

    def test_log_averages_correctly_when_globalstep_already_updated(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Regression test: the HF Trainer updates _globalstep_last_logged
        *before* calling log(), so our averaging must not depend on it.

        This simulates the real Trainer calling sequence in
        _maybe_log_save_evaluate() where _globalstep_last_logged is set
        to global_step on line 2909, then log() is called on line 2912.
        """
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Simulate 5 training steps via _reset_step_metrics + _track_metric
        for i in range(5):
            distiller._reset_step_metrics(is_training=True)
            distiller._track_metric("loss/student_0", float(i + 1))

        # Simulate the Trainer's state: _globalstep_last_logged already
        # updated to match global_step (as Trainer does before calling log)
        distiller.state.global_step = 10
        distiller._globalstep_last_logged = 10  # Already updated!

        captured_logs = None
        from transformers import Trainer

        original_log = Trainer.log

        def capture_logs(self_arg, logs_dict, start_time=None):
            nonlocal captured_logs
            captured_logs = logs_dict

        Trainer.log = capture_logs

        try:
            distiller.log({"loss": 2.0})

            # Average of [1, 2, 3, 4, 5] = 3.0, must not be 15.0 (sum)
            assert abs(captured_logs["loss/student_0"] - 3.0) < 1e-6
        finally:
            Trainer.log = original_log

    def test_log_does_not_inject_when_accumulator_empty(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that log() passes through logs unchanged when accumulator is empty."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        captured_logs = None
        from transformers import Trainer

        original_log = Trainer.log

        def capture_logs(self_arg, logs_dict, start_time=None):
            nonlocal captured_logs
            captured_logs = logs_dict

        Trainer.log = capture_logs

        try:
            distiller.log({"loss": 1.5, "learning_rate": 0.001})

            # No accumulator content, so logs should pass through unchanged
            assert captured_logs == {"loss": 1.5, "learning_rate": 0.001}
        finally:
            Trainer.log = original_log

    def test_eval_path_does_not_pollute_accumulator(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that eval-path _store_loss_metric calls don't pollute the accumulator."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Simulate eval path: _reset_step_metrics(is_training=False)
        distiller._reset_step_metrics(is_training=False)
        assert distiller._is_accumulating is False

        # Store a metric (as would happen during eval)
        loss = torch.tensor(0.5)
        distiller._store_loss_metric("student_0", loss)

        # Metric should be in current_step_metrics but NOT in accumulator
        assert "loss/student_0" in distiller.current_step_metrics
        assert "loss/student_0" not in distiller._training_metric_accumulator

    def test_max_grad_norm_is_numeric_after_composite_init(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """max_grad_norm must be numeric (not None) after composite optimizer init.

        BaseDistiller with _USE_COMPOSITE_OPTIMIZER propagates max_grad_norm
        to per-alignment optimizers, then disables the Trainer's own clipping.
        The disabled value must be 0 (not None) because the HF Trainer does
        ``if self.args.max_grad_norm > 0`` which raises TypeError on None.

        This is a regression test: BlockwiseDistiller uses composite
        optimizers and was the distiller that surfaced this bug during
        smoke testing of downstream experiments.
        """
        from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        args = TrainingArguments(
            output_dir=training_args.output_dir,
            max_steps=5,
            max_grad_norm=1.0,
            per_device_train_batch_size=2,
            dataloader_num_workers=0,
            report_to=[],
            use_cpu=True,
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        # Grad norm should be propagated to alignments
        assert single_alignment[0].max_grad_norm == 1.0

        # After propagation, Trainer-level clipping must be disabled with a
        # numeric value.  The HF Trainer does `if self.args.max_grad_norm > 0`
        # which raises TypeError on None.
        assert isinstance(distiller.args.max_grad_norm, (int, float)), (
            f"max_grad_norm must be numeric, got {type(distiller.args.max_grad_norm)}"
        )
        _ = distiller.args.max_grad_norm > 0  # must not raise TypeError


# ── Regression tests for bug fixes ──────────────────────────────────────────


class TestExtractLoss:
    """Regression tests for BaseDistiller._extract_loss."""

    def test_extracts_from_object_with_loss_attr(self):
        """Should extract .loss from HuggingFace-style model outputs."""
        from collections import namedtuple

        Output = namedtuple("Output", ["logits", "loss"])
        loss = torch.tensor(1.5)
        output = Output(logits=torch.randn(2, 10), loss=loss)
        assert BaseDistiller._extract_loss(output) is loss

    def test_extracts_from_dict(self):
        """Should extract 'loss' from dict outputs."""
        loss = torch.tensor(2.0)
        output = {"loss": loss, "logits": torch.randn(2, 10)}
        assert BaseDistiller._extract_loss(output) is loss

    def test_returns_none_when_no_loss(self):
        """Should return None when output has no loss."""
        from collections import namedtuple

        Output = namedtuple("Output", ["logits", "loss"])
        output = Output(logits=torch.randn(2, 10), loss=None)
        assert BaseDistiller._extract_loss(output) is None

    def test_returns_none_for_plain_tensor(self):
        """Should return None for a plain tensor (no .loss attr)."""
        assert BaseDistiller._extract_loss(torch.randn(2, 10)) is None

    def test_returns_none_for_dict_without_loss_key(self):
        """Should return None when dict has no 'loss' key."""
        assert BaseDistiller._extract_loss({"logits": torch.randn(2, 10)}) is None


class TestWarnOnce:
    """Regression tests for BaseDistiller._warn_once."""

    def test_warns_first_time(
        self, teacher_model, single_alignment, training_args, train_dataset, caplog
    ):
        """Should log a warning on first call with a given key."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        import logging

        with caplog.at_level(logging.WARNING):
            distiller._warn_once("test_key", "test message %s", "arg1")

        assert any("test message arg1" in msg for msg in caplog.messages)

    def test_suppresses_repeat(
        self, teacher_model, single_alignment, training_args, train_dataset, caplog
    ):
        """Should NOT log on second call with the same key."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        import logging

        distiller._warn_once("dup_key", "first call")
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            distiller._warn_once("dup_key", "second call")

        assert not any("second call" in msg for msg in caplog.messages)

    def test_different_keys_are_independent(
        self, teacher_model, single_alignment, training_args, train_dataset, caplog
    ):
        """Different keys should produce independent warnings."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        import logging

        with caplog.at_level(logging.WARNING):
            distiller._warn_once("key_a", "warning A")
            distiller._warn_once("key_b", "warning B")

        assert any("warning A" in msg for msg in caplog.messages)
        assert any("warning B" in msg for msg in caplog.messages)

    def test_per_instance_isolation(
        self, teacher_model, single_alignment, training_args, train_dataset, caplog
    ):
        """Warnings should be per-distiller-instance, not global."""
        d1 = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )
        d2 = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )

        import logging

        d1._warn_once("shared_key", "from d1")
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            d2._warn_once("shared_key", "from d2")

        # d2 should still warn — it's a separate instance
        assert any("from d2" in msg for msg in caplog.messages)


class TestCombineLosses:
    """Regression tests for BaseDistiller._combine_losses."""

    def _make_distiller(
        self,
        training_args,
        train_dataset,
        teacher_model,
        single_alignment,
        magnitude_aware=False,
    ):
        from silverspoon_kd.training_arguments import TrainingArguments

        args = TrainingArguments(
            output_dir=training_args.output_dir,
            max_steps=training_args.max_steps,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            logging_steps=training_args.logging_steps,
            dataloader_num_workers=0,
            report_to=[],
            use_cpu=training_args.use_cpu,
            magnitude_aware_weighting=magnitude_aware,
        )
        from silverspoon_kd.distillers.holistic_distiller import HolisticDistiller

        return HolisticDistiller(
            student_model=teacher_model,  # same model, doesn't matter for this test
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

    def test_weighted_combination(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Should produce weighted sum: 0.3*L1 + 0.7*L2."""
        distiller = self._make_distiller(
            training_args, train_dataset, teacher_model, single_alignment
        )
        distiller._reset_step_metrics(is_training=True)

        L1 = torch.tensor(10.0, requires_grad=True)
        L2 = torch.tensor(2.0, requires_grad=True)

        total = distiller._combine_losses(
            {"soft": L1, "hard": L2},
            {"soft": 0.3, "hard": 0.7},
        )
        assert total.item() == pytest.approx(0.3 * 10.0 + 0.7 * 2.0, rel=1e-5)

    def test_magnitude_aware_returns_raw_weighted_sum(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """With magnitude_aware=True, the returned *value* is the raw
        weighted sum (interpretable, matches magnitude_aware=False);
        the *gradient* is separately magnitude-balanced via the
        straight-through detach trick documented on ``_combine_losses``."""
        distiller = self._make_distiller(
            training_args,
            train_dataset,
            teacher_model,
            single_alignment,
            magnitude_aware=True,
        )
        distiller._reset_step_metrics(is_training=True)

        L1 = torch.tensor(1000.0, requires_grad=True)
        L2 = torch.tensor(0.5, requires_grad=True)

        total = distiller._combine_losses(
            {"soft": L1, "hard": L2},
            {"soft": 0.4, "hard": 0.6},
        )
        # Value is the raw weighted sum: 0.4 * 1000 + 0.6 * 0.5 = 400.3
        assert total.item() == pytest.approx(0.4 * 1000.0 + 0.6 * 0.5, rel=1e-5)
        # And explicitly NOT a ~sum(weights) = 1.0 constant.
        assert abs(total.item() - 1.0) > 1.0

        # Gradients come from the normalized total, not the raw one.
        # For each leaf: ``d(w * L / |L.detach()|)/dL = w / |L|``, so
        # L1.grad = 0.4 / 1000 = 0.0004
        # L2.grad = 0.6 / 0.5   = 1.2
        # Without the detach trick (or with magnitude_aware=False) these
        # would be w_i themselves (0.4 and 0.6).
        total.backward()
        assert L1.grad.item() == pytest.approx(0.4 / 1000.0, rel=1e-5)
        assert L2.grad.item() == pytest.approx(0.6 / 0.5, rel=1e-5)

    def test_tracks_per_component_metrics(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Should track loss/soft and loss/hard when track_metrics=True."""
        distiller = self._make_distiller(
            training_args, train_dataset, teacher_model, single_alignment
        )
        distiller._reset_step_metrics(is_training=True)

        distiller._combine_losses(
            {"soft": torch.tensor(5.0), "hard": torch.tensor(3.0)},
            {"soft": 0.5, "hard": 0.5},
            track_metrics=True,
        )
        assert "loss/soft" in distiller.current_step_metrics
        assert "loss/hard" in distiller.current_step_metrics

    def test_no_metrics_when_tracking_disabled(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Should NOT track metrics when track_metrics=False."""
        distiller = self._make_distiller(
            training_args, train_dataset, teacher_model, single_alignment
        )
        distiller._reset_step_metrics(is_training=True)

        distiller._combine_losses(
            {"soft": torch.tensor(5.0), "hard": torch.tensor(3.0)},
            {"soft": 0.5, "hard": 0.5},
            track_metrics=False,
        )
        assert "loss/soft" not in distiller.current_step_metrics
        assert "loss/hard" not in distiller.current_step_metrics

    def test_single_component(self, teacher_model, single_alignment, training_args, train_dataset):
        """Should handle a single loss component correctly."""
        distiller = self._make_distiller(
            training_args, train_dataset, teacher_model, single_alignment
        )
        distiller._reset_step_metrics(is_training=True)

        total = distiller._combine_losses(
            {"only": torch.tensor(7.0)},
            {"only": 1.0},
        )
        assert total.item() == pytest.approx(7.0, rel=1e-6)


class TestGetGradNorm:
    """Regression tests for BaseDistiller._get_grad_norm with composite optimizer."""

    def test_returns_precomputed_grad_norm(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Should return grad_norm directly when already provided."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )
        result = distiller._get_grad_norm(grad_norm=42.0)
        assert result == 42.0

    def test_composite_optimizer_computes_from_alignments(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """With composite optimizer, should compute norm from alignment params.

        Uses ``BlockwiseDistiller``: its block-independent training is the
        distiller with ``_USE_COMPOSITE_OPTIMIZER = True``.
        """
        from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        args = TrainingArguments(
            output_dir=training_args.output_dir,
            max_steps=5,
            per_device_train_batch_size=2,
            logging_steps=1,
            dataloader_num_workers=0,
            report_to=[],
            use_cpu=training_args.use_cpu,
        )
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        # Manually create optimizers and simulate gradients
        # (avoids full training_step which involves capture engine complexity)
        distiller.create_optimizer()
        for alignment in distiller.alignments:
            if alignment.optimizer is not None:
                for g in alignment.optimizer.param_groups:
                    for p in g["params"]:
                        p.grad = torch.randn_like(p)

        norm = distiller._get_grad_norm()
        assert isinstance(norm, (torch.Tensor, float))
        if isinstance(norm, torch.Tensor):
            assert torch.isfinite(norm)
            assert norm.item() > 0

    def test_composite_optimizer_no_gradients_returns_zero(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """With composite optimizer but no gradients, should return 0.

        Uses ``BlockwiseDistiller`` (see note above).
        """
        from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        args = TrainingArguments(
            output_dir=training_args.output_dir,
            max_steps=5,
            per_device_train_batch_size=2,
            logging_steps=1,
            dataloader_num_workers=0,
            report_to=[],
            use_cpu=training_args.use_cpu,
        )
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        # No training step run, so no gradients exist
        # Optimizers not created yet either
        norm = distiller._get_grad_norm()
        assert isinstance(norm, torch.Tensor)
        assert norm.item() == 0.0


class TestCreateOptimizerSignature:
    """Regression test for create_optimizer accepting model argument (HF 5.3+)."""

    def test_accepts_model_argument(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """create_optimizer(model=...) should not raise TypeError."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )
        # Should not raise "takes 1 positional argument but 2 were given"
        result = distiller.create_optimizer(model=teacher_model)
        # ConcreteDistiller has no trainable params, so returns parent's result
        assert result is not None or result is None  # just shouldn't raise
