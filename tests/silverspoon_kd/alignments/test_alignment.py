"""
Unit tests for the Alignment class, focused on auto-projector inference.
"""

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.training_arguments import TrainingArguments


class ParameterlessBlock(nn.Module):
    """A module with no parameters (e.g. activation, pooling)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu()


class SimpleBlock(nn.Module):
    """A block with parameters for baseline tests."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestTryInferOutputProjector:
    """Tests for Alignment._try_infer_output_projector."""

    def test_same_shape_returns_none(self):
        """No projector needed when shapes match."""
        alignment = Alignment(
            teacher_block=SimpleBlock(64, 64),
            student_block=SimpleBlock(64, 64),
        )
        student_out = torch.randn(2, 8, 64)
        teacher_out = torch.randn(2, 8, 64)

        result = alignment._try_infer_output_projector(student_out, teacher_out)
        assert result is None

    def test_different_last_dim_creates_linear(self):
        """Mismatched last dim should create a linear projector."""
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=SimpleBlock(64, 64),
        )
        student_out = torch.randn(2, 8, 64)
        teacher_out = torch.randn(2, 8, 128)

        result = alignment._try_infer_output_projector(student_out, teacher_out)
        assert result is not None
        # Should project from student dim to teacher dim
        projected = result(student_out)
        assert projected.shape == teacher_out.shape

    def test_parameterless_student_block(self):
        """Parameterless student block should not raise StopIteration."""
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=ParameterlessBlock(),
        )
        student_out = torch.randn(2, 8, 64)
        teacher_out = torch.randn(2, 8, 128)

        # Must not raise StopIteration for a block with no parameters
        result = alignment._try_infer_output_projector(student_out, teacher_out)
        assert result is not None
        projected = result(student_out)
        assert projected.shape == teacher_out.shape

    def test_parameterless_block_device_from_tensor(self):
        """Projector for parameterless block should match the student output device."""
        alignment = Alignment(
            teacher_block=ParameterlessBlock(),
            student_block=ParameterlessBlock(),
        )
        student_out = torch.randn(2, 8, 32)
        teacher_out = torch.randn(2, 8, 64)

        result = alignment._try_infer_output_projector(student_out, teacher_out)
        assert result is not None
        # Projector should be on same device as student_output
        proj_device = next(result.parameters()).device
        assert proj_device == student_out.device

    def test_4d_channel_mismatch_creates_conv2d(self):
        """4D tensors with channel mismatch should create Conv2d projector."""
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=SimpleBlock(64, 64),
        )
        student_out = torch.randn(2, 64, 7, 7)
        teacher_out = torch.randn(2, 128, 7, 7)

        result = alignment._try_infer_output_projector(student_out, teacher_out)
        assert result is not None
        projected = result(student_out)
        assert projected.shape == teacher_out.shape


class TestTryInitOutputProjector:
    """Tests for Alignment._try_init_output_projector (lazy init from captured outputs)."""

    def test_creates_projector_on_dim_mismatch(self):
        """Should create and store output projector when dims differ."""
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=SimpleBlock(64, 64),
            auto_projector=True,
        )
        assert alignment.output_projector is None
        assert alignment._awaiting_output_shape is True

        student_out = torch.randn(2, 8, 64)
        teacher_out = torch.randn(2, 8, 128)
        alignment._try_init_output_projector(student_out, teacher_out)

        assert alignment.output_projector is not None
        assert alignment._auto_projector_initialized is True
        assert alignment._awaiting_output_shape is False

    def test_no_projector_on_same_dims(self):
        """Should not create projector when dims match."""
        alignment = Alignment(
            teacher_block=SimpleBlock(64, 64),
            student_block=SimpleBlock(64, 64),
            auto_projector=True,
        )
        student_out = torch.randn(2, 8, 64)
        teacher_out = torch.randn(2, 8, 64)
        alignment._try_init_output_projector(student_out, teacher_out)

        assert alignment.output_projector is None
        assert alignment._auto_projector_initialized is True

    def test_skips_if_not_awaiting_output_shape(self):
        """Should be a no-op if _awaiting_output_shape is False."""
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=SimpleBlock(64, 64),
            auto_projector=True,
        )
        alignment._awaiting_output_shape = False

        student_out = torch.randn(2, 8, 64)
        teacher_out = torch.randn(2, 8, 128)
        alignment._try_init_output_projector(student_out, teacher_out)

        # Should not have created a projector since not awaiting shape
        assert alignment.output_projector is None
        assert alignment._auto_projector_initialized is True

    def test_skips_if_output_projector_already_set(self):
        """Should be a no-op if output_projector was manually provided."""
        existing_proj = nn.Linear(64, 128)
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=SimpleBlock(64, 64),
            output_projector=existing_proj,
            auto_projector=True,
        )

        student_out = torch.randn(2, 8, 64)
        teacher_out = torch.randn(2, 8, 128)
        alignment._try_init_output_projector(student_out, teacher_out)

        # Should keep the original projector
        assert alignment.output_projector is existing_proj

    def test_parameterless_block(self):
        """Should work with parameterless student blocks."""
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=ParameterlessBlock(),
            auto_projector=True,
        )

        student_out = torch.randn(2, 8, 64)
        teacher_out = torch.randn(2, 8, 128)
        alignment._try_init_output_projector(student_out, teacher_out)

        assert alignment.output_projector is not None
        assert alignment._auto_projector_initialized is True


class TestAlignmentLossWeight:
    """Tests for Alignment.loss_weight constructor parameter."""

    def test_default_loss_weight(self):
        """Default loss_weight should be 1.0."""
        alignment = Alignment(
            teacher_block=SimpleBlock(64, 64),
            student_block=SimpleBlock(64, 64),
        )
        assert alignment.loss_weight == 1.0

    def test_custom_loss_weight(self):
        """Custom loss_weight should be stored as-is."""
        alignment = Alignment(
            teacher_block=SimpleBlock(64, 64),
            student_block=SimpleBlock(64, 64),
            loss_weight=0.3,
        )
        assert alignment.loss_weight == 0.3


class TestAlignmentOptimizerCreation:
    """Tests for optimizer and scheduler creation in Alignment."""

    def _make_training_args(self, tmp_path, **kwargs):
        defaults = {
            "output_dir": str(tmp_path),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "max_steps": 10,
            "weight_decay": 0.01,
            "report_to": [],
            "use_cpu": True,
        }
        defaults.update(kwargs)
        return TrainingArguments(**defaults)

    def test_param_groups_has_decay_and_no_decay(self, tmp_path):
        """_get_param_groups should return two groups: decay and no-decay."""
        args = self._make_training_args(tmp_path)
        alignment = Alignment(
            teacher_block=SimpleBlock(64, 64),
            student_block=SimpleBlock(64, 64),
        )
        groups = alignment._get_param_groups(args)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] > 0
        assert groups[1]["weight_decay"] == 0.0

    def test_projector_params_included_in_groups(self, tmp_path):
        """Projector params should be included in optimizer param groups."""
        args = self._make_training_args(tmp_path)
        output_projector = nn.Linear(64, 128)
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=SimpleBlock(64, 64),
            output_projector=output_projector,
        )
        groups = alignment._get_param_groups(args)
        all_params = [p for g in groups for p in g["params"]]
        proj_weight = output_projector.weight
        assert any(p is proj_weight for p in all_params)

    def test_bias_in_no_decay_group(self, tmp_path):
        """Bias parameters should land in the no-decay group."""
        args = self._make_training_args(tmp_path)
        alignment = Alignment(
            teacher_block=SimpleBlock(64, 64),
            student_block=SimpleBlock(64, 64),
        )
        groups = alignment._get_param_groups(args)
        no_decay_params = groups[1]["params"]
        # SimpleBlock has a Linear with bias — bias should be in no-decay
        bias_param = alignment.student_block.linear.bias
        assert any(p is bias_param for p in no_decay_params)

    def test_ensure_raises_without_num_training_steps(self, tmp_path):
        """_ensure_optimizer_and_scheduler raises ValueError without num_training_steps."""
        args = self._make_training_args(tmp_path)
        alignment = Alignment(
            teacher_block=SimpleBlock(64, 64),
            student_block=SimpleBlock(64, 64),
        )
        with pytest.raises(ValueError, match="num_training_steps"):
            alignment._ensure_optimizer_and_scheduler(args, num_training_steps=None)

    def test_prepare_sets_id_and_grad_norm(self, tmp_path):
        """_prepare_for_training should set alignment.id and alignment.max_grad_norm."""
        args = self._make_training_args(tmp_path)
        alignment = Alignment(
            teacher_block=SimpleBlock(64, 64),
            student_block=SimpleBlock(64, 64),
        )
        assert alignment.id is None
        alignment._prepare_for_training(alignment_id=7, training_args=args, num_training_steps=100)
        assert alignment.id == 7
        assert alignment.max_grad_norm == args.max_grad_norm

    def test_add_projector_params_deduplicates(self, tmp_path):
        """Calling _add_projector_params_to_optimizer twice should not duplicate params."""
        args = self._make_training_args(tmp_path)
        output_projector = nn.Linear(64, 128)
        alignment = Alignment(
            teacher_block=SimpleBlock(128, 128),
            student_block=SimpleBlock(64, 64),
            output_projector=output_projector,
        )
        alignment._prepare_for_training(alignment_id=0, training_args=args, num_training_steps=100)
        param_count_before = sum(len(g["params"]) for g in alignment.optimizer.param_groups)
        alignment._add_projector_params_to_optimizer()
        param_count_after = sum(len(g["params"]) for g in alignment.optimizer.param_groups)
        assert param_count_after == param_count_before
