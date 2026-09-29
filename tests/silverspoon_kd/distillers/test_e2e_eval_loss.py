"""
Tests for the end-to-end evaluation loss feature (e2e_eval_loss).
"""

import logging
from unittest.mock import patch

import torch
import torch.nn as nn

from silverspoon_kd.distillers.base_distiller import BaseDistiller
from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
from silverspoon_kd.distillers.holistic_distiller import HolisticDistiller
from silverspoon_kd.distillers.response_based_distiller import (
    ResponseBasedDistiller,
)
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)

from ..conftest import DummyDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blockwise_args(training_args, **overrides):
    """Create TrainingArguments from base training_args."""
    return TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        max_steps=training_args.max_steps,
        dataloader_num_workers=training_args.dataloader_num_workers,
        report_to=[],
        use_cpu=training_args.use_cpu,
        **overrides,
    )


def _make_holistic_args(training_args, **overrides):
    """Create TrainingArguments from base training_args."""
    return TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        max_steps=training_args.max_steps,
        dataloader_num_workers=training_args.dataloader_num_workers,
        report_to=[],
        use_cpu=training_args.use_cpu,
        **overrides,
    )


def _make_response_based_args(training_args, **overrides):
    """Create TrainingArguments from base training_args."""
    return TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        max_steps=training_args.max_steps,
        dataloader_num_workers=training_args.dataloader_num_workers,
        report_to=[],
        use_cpu=training_args.use_cpu,
        **overrides,
    )


class _DummyModel(nn.Module):
    """Dummy model to satisfy Trainer's model requirement."""

    def __init__(self, device):
        super().__init__()
        self.register_buffer("_dummy_param", torch.zeros(1, device=device))

    def forward(self, **kwargs):
        return None


class ConcreteDistiller(BaseDistiller):
    """Concrete implementation of BaseDistiller for testing."""

    def __init__(self, teacher_model, alignments, **kwargs):
        device = next(teacher_model.parameters()).device
        dummy_model = _DummyModel(device)
        super().__init__(
            teacher_model=teacher_model,
            alignments=alignments,
            model=dummy_model,
            **kwargs,
        )
        self.args.max_grad_norm = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        device = next(self.teacher_model.parameters()).device
        loss = torch.tensor(1.0, device=device, requires_grad=True)
        self.step_losses.append(loss.detach())
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(self.teacher_model.parameters()).device
        loss = torch.tensor(1.0, device=device)
        self.eval_losses.append(loss.detach())
        if return_outputs:
            return (loss, None)
        return loss


class LossComputingModel(nn.Module):
    """A model that computes loss when labels are provided."""

    def __init__(self, input_dim=64, hidden_dim=128, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(128, input_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.lm_head = nn.Linear(hidden_dim, 128)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        class Output:
            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss

            def __iter__(self):
                yield self.logits

        return Output(logits, loss)

    def get_layer(self, idx):
        return self.layers[idx]


class SimpleModelWithLoss(nn.Module):
    """SimpleModel variant that computes cross-entropy loss when labels are provided.

    Uses the same SimpleBlock layers as conftest.SimpleModel so that layer objects
    can be shared with Alignment fixtures.
    """

    def __init__(self, input_dim=64, hidden_dim=128, num_layers=3):
        super().__init__()
        from ..conftest import SimpleBlock

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(128, input_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                SimpleBlock(
                    input_dim=input_dim if i == 0 else hidden_dim,
                    output_dim=hidden_dim,
                )
            )
        self.lm_head = nn.Linear(hidden_dim, 128)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        class Output:
            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss

            def __iter__(self):
                yield self.logits

        return Output(logits, loss)

    def get_layer(self, idx):
        return self.layers[idx]


class NoLossModel(nn.Module):
    """A model that never returns a loss."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, input_ids, attention_mask=None, labels=None):
        class Output:
            def __init__(self):
                self.loss = None

        return Output()


# ---------------------------------------------------------------------------
# Tests for _get_e2e_student_models
# ---------------------------------------------------------------------------


class TestGetE2eStudentModels:
    """Test _get_e2e_student_models for each distiller type."""

    def test_base_distiller_returns_none(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """BaseDistiller default returns None."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
        )
        assert distiller._get_e2e_student_models() is None

    def test_blockwise_returns_student_models(
        self,
        teacher_model,
        single_alignment,
        training_args,
        train_dataset,
        eval_dataset,
        student_model,
    ):
        """BlockwiseDistiller returns student_models dict when provided."""
        args = _make_blockwise_args(training_args)
        student_models = {"test_student": student_model}
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            student_models=student_models,
        )
        assert distiller._get_e2e_student_models() is student_models

    def test_blockwise_returns_none_when_no_student_models(
        self,
        teacher_model,
        single_alignment,
        training_args,
        train_dataset,
        eval_dataset,
    ):
        """BlockwiseDistiller returns None when student_models not provided."""
        args = _make_blockwise_args(training_args)
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        assert distiller._get_e2e_student_models() is None

    def test_global_returns_student_model(
        self,
        teacher_model,
        student_model,
        single_alignment,
        training_args,
        train_dataset,
        eval_dataset,
    ):
        """HolisticDistiller returns dict with student model."""
        args = _make_holistic_args(training_args)
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        result = distiller._get_e2e_student_models()
        assert isinstance(result, dict)
        assert len(result) == 1
        model_name = single_alignment[0].student_model_name
        assert model_name in result
        assert result[model_name] is student_model


# ---------------------------------------------------------------------------
# Tests for _compute_e2e_eval_loss
# ---------------------------------------------------------------------------


class TestComputeE2eEvalLoss:
    """Test _compute_e2e_eval_loss method."""

    def test_disabled_when_e2e_eval_loss_is_none(
        self,
        teacher_model,
        single_alignment,
        training_args,
        train_dataset,
        eval_dataset,
    ):
        """Returns empty dict when e2e_eval_loss is None (default)."""
        assert training_args.e2e_eval_loss is None
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        result = distiller._compute_e2e_eval_loss(eval_dataset)
        assert result == {}

    def test_disabled_when_no_student_models(
        self, teacher_model, single_alignment, train_dataset, eval_dataset, device, tmp_path
    ):
        """Returns empty dict when _get_e2e_student_models returns None."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        # BaseDistiller._get_e2e_student_models returns None
        result = distiller._compute_e2e_eval_loss(eval_dataset)
        assert result == {}

    def test_forward_mode_single_model(
        self, teacher_model, single_alignment, train_dataset, eval_dataset, device, tmp_path
    ):
        """Forward mode computes e2e loss for a single student model."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )
        student = LossComputingModel().to(device)
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Override _get_e2e_student_models to return our loss-computing model
        distiller._get_e2e_student_models = lambda: {"test_student": student}

        result = distiller._compute_e2e_eval_loss(eval_dataset)
        assert "eval_loss/e2e" in result
        assert isinstance(result["eval_loss/e2e"], float)
        assert result["eval_loss/e2e"] > 0

    def test_forward_mode_multiple_models(
        self, teacher_model, single_alignment, train_dataset, eval_dataset, device, tmp_path
    ):
        """Forward mode uses per-model metric keys for multiple student models."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )
        student_a = LossComputingModel().to(device)
        student_b = LossComputingModel().to(device)
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        distiller._get_e2e_student_models = lambda: {
            "model_a": student_a,
            "model_b": student_b,
        }

        result = distiller._compute_e2e_eval_loss(eval_dataset)
        assert "eval_loss/e2e/model_a" in result
        assert "eval_loss/e2e/model_b" in result
        assert "eval_loss/e2e" not in result  # No bare key for multi-model

    def test_custom_metric_key_prefix(
        self, teacher_model, single_alignment, train_dataset, eval_dataset, device, tmp_path
    ):
        """Metric key prefix is used correctly."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )
        student = LossComputingModel().to(device)
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        distiller._get_e2e_student_models = lambda: {"test_student": student}

        result = distiller._compute_e2e_eval_loss(eval_dataset, metric_key_prefix="test")
        assert "test_loss/e2e" in result

    def test_metric_keys_use_eval_underscore_prefix(
        self, teacher_model, single_alignment, train_dataset, eval_dataset, device, tmp_path
    ):
        """Metric keys must start with '{prefix}_' so wandb/tensorboard categorize them
        under the correct section (e.g. 'eval') instead of defaulting to 'train'."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )
        student = LossComputingModel().to(device)
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        distiller._get_e2e_student_models = lambda: {"test_student": student}

        # Single model
        result = distiller._compute_e2e_eval_loss(eval_dataset, metric_key_prefix="eval")
        for key in result:
            assert key.startswith("eval_"), (
                f"Metric key '{key}' does not start with 'eval_'. "
                "Wandb/tensorboard will mis-categorize it under 'train'."
            )

        # Multiple models
        student_b = LossComputingModel().to(device)
        distiller._get_e2e_student_models = lambda: {
            "model_a": student,
            "model_b": student_b,
        }
        result = distiller._compute_e2e_eval_loss(eval_dataset, metric_key_prefix="eval")
        for key in result:
            assert key.startswith("eval_"), (
                f"Metric key '{key}' does not start with 'eval_'. "
                "Wandb/tensorboard will mis-categorize it under 'train'."
            )

    def test_warning_when_model_returns_no_loss(
        self,
        teacher_model,
        single_alignment,
        train_dataset,
        eval_dataset,
        device,
        caplog,
        tmp_path,
    ):
        """Warning logged when model doesn't return a loss."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )
        no_loss_model = NoLossModel().to(device)
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        distiller._get_e2e_student_models = lambda: {"test_student": no_loss_model}

        with caplog.at_level(logging.WARNING):
            result = distiller._compute_e2e_eval_loss(eval_dataset)

        assert result == {}  # No metrics since no loss was returned
        assert any("did not return a loss" in msg for msg in caplog.messages)

    def test_uses_self_eval_dataset_when_none_passed(
        self, teacher_model, single_alignment, train_dataset, eval_dataset, device, tmp_path
    ):
        """Falls back to self.eval_dataset when eval_dataset arg is None."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )
        student = LossComputingModel().to(device)
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        distiller._get_e2e_student_models = lambda: {"test_student": student}

        # Pass None - should use self.eval_dataset
        result = distiller._compute_e2e_eval_loss(None)
        assert "eval_loss/e2e" in result


# ---------------------------------------------------------------------------
# Tests for evaluate() integration
# ---------------------------------------------------------------------------


class TestEvaluateE2eIntegration:
    """Test that evaluate() includes e2e metrics when configured."""

    def test_evaluate_includes_e2e_metric(
        self, teacher_model, single_alignment, train_dataset, eval_dataset, device, tmp_path
    ):
        """evaluate() includes e2e metric when e2e_eval_loss is set."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )
        student = LossComputingModel().to(device)
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        distiller._get_e2e_student_models = lambda: {"test_student": student}

        metrics = distiller.evaluate()
        assert "eval_loss/e2e" in metrics

    def test_evaluate_no_e2e_metric_when_disabled(
        self,
        teacher_model,
        single_alignment,
        training_args,
        train_dataset,
        eval_dataset,
    ):
        """evaluate() does not include e2e metric when e2e_eval_loss is None."""
        distiller = ConcreteDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        metrics = distiller.evaluate()
        assert "eval_loss/e2e" not in metrics


# ---------------------------------------------------------------------------
# Tests for ResponseBasedDistiller e2e_eval_loss support
# ---------------------------------------------------------------------------


class TestResponseBasedDistillerE2e:
    """Test that response-based KD supports e2e_eval_loss."""

    def test_e2e_eval_loss_preserved(
        self, teacher_model, student_model, train_dataset, device, caplog, tmp_path
    ):
        """Response-based KD preserves e2e_eval_loss without warning."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )

        with caplog.at_level(logging.WARNING):
            distiller = ResponseBasedDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                args=args,
                train_dataset=train_dataset,
            )

        assert not any("redundant" in msg for msg in caplog.messages)
        assert distiller.args.e2e_eval_loss == "forward"

    def test_no_warning_when_e2e_eval_loss_none(
        self, teacher_model, student_model, training_args, train_dataset, caplog
    ):
        """Response-based KD does not warn when e2e_eval_loss is None (default)."""
        args = _make_response_based_args(training_args)
        with caplog.at_level(logging.WARNING):
            ResponseBasedDistiller(
                student_model=student_model,
                teacher_model=teacher_model,
                args=args,
                train_dataset=train_dataset,
            )

        assert not any("redundant" in msg for msg in caplog.messages)

    def test_get_e2e_student_models_returns_student(
        self, teacher_model, student_model, training_args, train_dataset
    ):
        """Response-based KD returns the student model for e2e eval."""
        args = _make_response_based_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
        )
        result = distiller._get_e2e_student_models()
        assert isinstance(result, dict)
        assert len(result) == 1
        assert list(result.values())[0] is student_model

    def test_evaluate_has_no_e2e_metric_when_disabled(
        self, teacher_model, student_model, training_args, train_dataset, eval_dataset
    ):
        """Response-based KD evaluate() excludes e2e metric when e2e_eval_loss is None."""
        args = _make_response_based_args(training_args)
        distiller = ResponseBasedDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        metrics = distiller.evaluate()
        assert not any("e2e" in k for k in metrics)


# ---------------------------------------------------------------------------
# Integration tests: real distillers with capture hooks + e2e eval
# ---------------------------------------------------------------------------


def _make_same_dim_alignments(teacher_model, student_model):
    """Build single-student-per-teacher alignments for same-dim models."""
    from silverspoon_kd.alignments import Alignment

    alignments = []
    for i in range(teacher_model.num_layers):
        alignment = Alignment(
            teacher_block=teacher_model.get_layer(i),
            student_block=student_model.get_layer(i),
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
        )
        alignments.append(alignment)
    return alignments


class TestE2eEvalTerminalModuleHandling:
    """Tests verifying auto_truncate is suspended during e2e eval and restored after.

    HKD uses ModuleCaptureEngine with auto_truncate=True to truncate forward
    passes for efficiency.  During e2e eval, truncation must be suspended so
    the full student model runs end-to-end.  After e2e eval, it must be restored
    so subsequent training continues to truncate correctly.
    """

    def test_hkd_clears_truncation_during_e2e_eval(self, teacher_model, device, tmp_path):
        """HKD suspends auto_truncate before running e2e eval."""
        student = SimpleModelWithLoss(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignments = _make_same_dim_alignments(teacher_model, student)
        eval_ds = DummyDataset(num_samples=6, seq_len=16)

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(num_samples=10, seq_len=16),
            eval_dataset=eval_ds,
            auto_truncate=True,
        )

        distiller._register_capture()

        # Patch _run_e2e_eval to capture auto_truncate state during e2e eval
        captured_states = {}
        original_run = BaseDistiller._run_e2e_eval

        def spy_run(self_inner, eval_dataset=None, metric_key_prefix="eval"):
            captured_states["teacher_auto_truncate"] = self_inner.teacher_capture.auto_truncate
            captured_states["student_auto_truncate"] = self_inner.student_capture.auto_truncate
            return original_run(self_inner, eval_dataset, metric_key_prefix)

        with patch.object(BaseDistiller, "_run_e2e_eval", spy_run):
            distiller._compute_e2e_eval_loss(eval_ds)

        # auto_truncate must be False during e2e forward pass
        assert captured_states["teacher_auto_truncate"] is False, (
            "Teacher auto_truncate was not disabled during e2e eval"
        )
        assert captured_states["student_auto_truncate"] is False, (
            "Student auto_truncate was not disabled during e2e eval"
        )

        distiller._deregister_capture()

    def test_hkd_restores_truncation_after_e2e_eval(self, teacher_model, device, tmp_path):
        """HKD restores auto_truncate after e2e eval completes."""
        student = SimpleModelWithLoss(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignments = _make_same_dim_alignments(teacher_model, student)
        eval_ds = DummyDataset(num_samples=6, seq_len=16)

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(num_samples=10, seq_len=16),
            eval_dataset=eval_ds,
            auto_truncate=True,
        )

        distiller._register_capture()

        # Run e2e eval
        distiller._compute_e2e_eval_loss(eval_ds)

        # auto_truncate should be restored after e2e eval
        assert distiller.teacher_capture.auto_truncate is True, (
            "Teacher auto_truncate was not restored after e2e eval"
        )
        assert distiller.student_capture.auto_truncate is True, (
            "Student auto_truncate was not restored after e2e eval"
        )

        distiller._deregister_capture()


class TestE2eEvalWithCaptureHooks:
    """Integration tests verifying e2e eval works with active capture engines.

    HolisticDistiller registers forward hooks on
    the student model via ModuleCaptureEngine.  Running a clean e2e forward pass
    requires those hooks to be temporarily removed.  These tests exercise the
    full evaluate() path on real distiller instances to verify that the
    deregister/re-register pattern works correctly.
    """

    def test_holistic_distiller_evaluate_with_e2e(self, teacher_model, device, tmp_path):
        """HolisticDistiller.evaluate() produces e2e metric without hook interference."""
        student = SimpleModelWithLoss(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignments = _make_same_dim_alignments(teacher_model, student)
        train_ds = DummyDataset(num_samples=10, seq_len=16)
        eval_ds = DummyDataset(num_samples=6, seq_len=16)

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher_model,
            alignments=alignments,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )

        # Hooks must be active for evaluate() since compute_loss needs captured
        # data.  In normal usage evaluate() runs inside train() where hooks are
        # registered; here we register them manually.
        distiller._register_capture()
        try:
            metrics = distiller.evaluate()
        finally:
            distiller._deregister_capture()

        assert "eval_loss/e2e" in metrics
        assert isinstance(metrics["eval_loss/e2e"], float)
        assert metrics["eval_loss/e2e"] > 0

    def test_blockwise_distiller_evaluate_with_e2e(self, teacher_model, device, tmp_path):
        """BlockwiseDistiller.evaluate() works — hooks are only on teacher, not student."""
        student = SimpleModelWithLoss(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        alignments = _make_same_dim_alignments(teacher_model, student)
        train_ds = DummyDataset(num_samples=10, seq_len=16)
        eval_ds = DummyDataset(num_samples=6, seq_len=16)

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            e2e_eval_loss="forward",
            use_cpu=(device.type == "cpu"),
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            student_models={"test_student": student},
        )

        # BlockwiseDistiller hooks are only on the teacher, so the student
        # model can run a clean e2e forward without deregistering.
        distiller._register_capture()
        try:
            metrics = distiller.evaluate()
        finally:
            distiller._deregister_capture()

        assert "eval_loss/e2e" in metrics
        assert isinstance(metrics["eval_loss/e2e"], float)
        assert metrics["eval_loss/e2e"] > 0


# ---------------------------------------------------------------------------
# Tests: compute_metrics via e2e eval pass
# ---------------------------------------------------------------------------


def _counting_accuracy_metric(counter):
    """Return a compute_metrics callback that tracks how many times it's called."""

    def compute_metrics(eval_pred):
        counter["calls"] += 1
        preds = eval_pred.predictions.argmax(-1)
        acc = float((preds == eval_pred.label_ids).mean())
        assert 0.0 <= acc <= 1.0, f"accuracy {acc} out of [0, 1] range"
        return {"accuracy": acc}

    return compute_metrics


def _make_bkd_with_metrics(teacher_model, device, tmp_path, compute_metrics=None, **args_overrides):
    """Helper: create a BKD distiller with student_models and optional compute_metrics."""
    student = SimpleModelWithLoss(input_dim=64, hidden_dim=128, num_layers=3).to(device)
    alignments = _make_same_dim_alignments(teacher_model, student)
    ds = DummyDataset(num_samples=6, seq_len=16)
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        use_cpu=(device.type == "cpu"),
        **args_overrides,
    )
    distiller = BlockwiseDistiller(
        teacher_model=teacher_model,
        alignments=alignments,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,
        student_models={"test_student": student},
        compute_metrics=compute_metrics,
    )
    return distiller, student


def _make_hkd_with_metrics(teacher_model, device, tmp_path, compute_metrics=None, **args_overrides):
    """Helper: create an HKD distiller with optional compute_metrics."""
    student = SimpleModelWithLoss(input_dim=64, hidden_dim=128, num_layers=3).to(device)
    alignments = _make_same_dim_alignments(teacher_model, student)
    ds = DummyDataset(num_samples=6, seq_len=16)
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        use_cpu=(device.type == "cpu"),
        **args_overrides,
    )
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher_model,
        alignments=alignments,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,
        compute_metrics=compute_metrics,
    )
    return distiller, student


def _evaluate(distiller):
    """Register capture, evaluate, deregister — returns metrics dict."""
    distiller._register_capture()
    try:
        return distiller.evaluate()
    finally:
        distiller._deregister_capture()


class TestComputeMetricsViaE2E:
    """Tests for compute_metrics integration across distiller paradigms.

    When prediction_step cannot provide logits (BKD), compute_metrics
    should be called from the e2e eval pass.  When prediction_step does
    provide logits (HKD, ResKD), compute_metrics should come from the
    main eval loop and NOT trigger a redundant e2e pass.
    """

    # -- BKD: compute_metrics via e2e pass --

    def test_bkd_accuracy_with_e2e_loss(self, teacher_model, device, tmp_path):
        """BKD + e2e_eval_loss + compute_metrics: both e2e loss and accuracy
        from one forward pass."""
        counter = {"calls": 0}
        distiller, student = _make_bkd_with_metrics(
            teacher_model,
            device,
            tmp_path,
            compute_metrics=_counting_accuracy_metric(counter),
            e2e_eval_loss="forward",
        )

        metrics = _evaluate(distiller)

        assert "eval_accuracy" in metrics, f"Missing eval_accuracy; keys: {list(metrics)}"
        assert 0.0 <= metrics["eval_accuracy"] <= 1.0
        assert "eval_loss/e2e" in metrics
        assert counter["calls"] == 1, f"compute_metrics called {counter['calls']} times, expected 1"

    def test_bkd_accuracy_without_e2e_loss(self, teacher_model, device, tmp_path):
        """BKD + compute_metrics but NO e2e_eval_loss: e2e pass triggers for
        metrics only, no e2e loss produced."""
        counter = {"calls": 0}
        distiller, _ = _make_bkd_with_metrics(
            teacher_model,
            device,
            tmp_path,
            compute_metrics=_counting_accuracy_metric(counter),
        )

        metrics = _evaluate(distiller)

        assert "eval_accuracy" in metrics, f"Missing eval_accuracy; keys: {list(metrics)}"
        assert "eval_loss/e2e" not in metrics
        assert counter["calls"] == 1

    def test_bkd_no_e2e_pass_without_metrics_or_loss(self, teacher_model, device, tmp_path):
        """BKD without compute_metrics or e2e_eval_loss: student model must
        never be called — verify zero forward passes to prevent waste."""
        distiller, student = _make_bkd_with_metrics(teacher_model, device, tmp_path)

        call_count = 0
        orig_forward = student.forward

        def counting_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return orig_forward(*args, **kwargs)

        student.forward = counting_forward
        try:
            _evaluate(distiller)
        finally:
            student.forward = orig_forward

        assert call_count == 0, (
            f"Student forward called {call_count} times but neither "
            f"e2e_eval_loss nor compute_metrics was configured"
        )

    def test_bkd_single_forward_pass_for_loss_and_metrics(self, teacher_model, device, tmp_path):
        """BKD + e2e_eval_loss + compute_metrics must use ONE forward pass,
        not separate passes for loss and metrics."""
        counter = {"calls": 0}
        distiller, student = _make_bkd_with_metrics(
            teacher_model,
            device,
            tmp_path,
            compute_metrics=_counting_accuracy_metric(counter),
            e2e_eval_loss="forward",
        )

        fwd_count = 0
        orig_forward = student.forward

        def counting_forward(*args, **kwargs):
            nonlocal fwd_count
            fwd_count += 1
            return orig_forward(*args, **kwargs)

        student.forward = counting_forward
        try:
            _evaluate(distiller)
        finally:
            student.forward = orig_forward

        # 6 samples / batch_size 2 = 3 batches = 3 forward calls
        num_batches = 3
        assert fwd_count == num_batches, (
            f"Student forward called {fwd_count} times for {num_batches} batches — "
            f"expected exactly one forward per batch (no redundant passes)"
        )

    # -- HKD: compute_metrics from main eval loop --

    def test_hkd_accuracy_from_main_loop(self, teacher_model, device, tmp_path):
        """HKD + compute_metrics: accuracy from prediction_step, called exactly once."""
        counter = {"calls": 0}
        distiller, _ = _make_hkd_with_metrics(
            teacher_model,
            device,
            tmp_path,
            compute_metrics=_counting_accuracy_metric(counter),
        )

        metrics = _evaluate(distiller)

        assert "eval_accuracy" in metrics, f"Missing eval_accuracy; keys: {list(metrics)}"
        assert 0.0 <= metrics["eval_accuracy"] <= 1.0
        assert "eval_loss/e2e" not in metrics
        assert counter["calls"] == 1, (
            f"compute_metrics called {counter['calls']} times, expected exactly 1 "
            f"(from main eval loop only)"
        )

    def test_hkd_accuracy_with_e2e_loss_no_duplicate(self, teacher_model, device, tmp_path):
        """HKD + compute_metrics + e2e_eval_loss: accuracy from main loop,
        e2e loss from separate pass, compute_metrics called exactly once."""
        counter = {"calls": 0}
        distiller, _ = _make_hkd_with_metrics(
            teacher_model,
            device,
            tmp_path,
            compute_metrics=_counting_accuracy_metric(counter),
            e2e_eval_loss="forward",
        )

        metrics = _evaluate(distiller)

        assert "eval_accuracy" in metrics
        assert "eval_loss/e2e" in metrics
        assert counter["calls"] == 1, (
            f"compute_metrics called {counter['calls']} times, expected 1 — "
            f"e2e pass must not duplicate the main loop's metric computation"
        )

    # -- ResKD: compute_metrics from main eval loop --

    @staticmethod
    def _make_reskd(device, tmp_path, compute_metrics=None, **args_overrides):
        """Helper: create a ResKD distiller with optional compute_metrics."""
        from silverspoon_kd.losses import kl_divergence_loss

        teacher = SimpleModelWithLoss(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        student = SimpleModelWithLoss(input_dim=64, hidden_dim=128, num_layers=3).to(device)
        args_overrides.setdefault("report_to", [])
        return ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            soft_loss_fn=kl_divergence_loss(temperature=4.0),
            args=TrainingArguments(
                output_dir=str(tmp_path / "out"),
                max_steps=0,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                use_cpu=(device.type == "cpu"),
                **args_overrides,
            ),
            train_dataset=DummyDataset(num_samples=6, seq_len=16),
            eval_dataset=DummyDataset(num_samples=6, seq_len=16),
            compute_metrics=compute_metrics,
        )

    def test_reskd_accuracy_from_main_loop(self, device, tmp_path):
        """ResKD + compute_metrics: accuracy from prediction_step, called exactly once."""
        counter = {"calls": 0}
        distiller = self._make_reskd(
            device,
            tmp_path,
            compute_metrics=_counting_accuracy_metric(counter),
        )

        metrics = distiller.evaluate()

        assert "eval_accuracy" in metrics, f"Missing eval_accuracy; keys: {list(metrics)}"
        assert 0.0 <= metrics["eval_accuracy"] <= 1.0
        assert "eval_loss/e2e" not in metrics
        assert counter["calls"] == 1

    def test_reskd_accuracy_with_e2e_loss_no_duplicate(self, device, tmp_path):
        """ResKD + compute_metrics + e2e_eval_loss: accuracy from main loop,
        e2e loss from separate pass, compute_metrics called exactly once."""
        counter = {"calls": 0}
        distiller = self._make_reskd(
            device,
            tmp_path,
            compute_metrics=_counting_accuracy_metric(counter),
            e2e_eval_loss="forward",
        )

        metrics = distiller.evaluate()

        assert "eval_accuracy" in metrics
        assert "eval_loss/e2e" in metrics
        assert counter["calls"] == 1, (
            f"compute_metrics called {counter['calls']} times, expected 1 — "
            f"e2e pass must not duplicate the main loop's metric computation"
        )
