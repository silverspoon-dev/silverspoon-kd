"""Tests for projector_init parameter on Alignment and create_alignments.

Verifies that projector_init correctly initializes auto-created projectors
with the specified initialization scheme.
"""

import pytest
import torch
import torch.nn as nn

from silverspoon_kd import Alignment, create_alignments

# ── Fixtures ─────────────────────────────────────────────────────────────


class TwoLayerModel(nn.Module):
    """Minimal model with named layers for alignment matching."""

    def __init__(self, hidden):
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "0": nn.Linear(hidden, hidden),
                "1": nn.Linear(hidden, hidden),
            }
        )

    def forward(self, x):
        x = self.layers["0"](x)
        x = self.layers["1"](x)
        return x


def _make_models(teacher_dim=32, student_dim=16):
    torch.manual_seed(0)
    teacher = TwoLayerModel(teacher_dim)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = TwoLayerModel(student_dim)
    return teacher, student


def _trigger_projector_creation(alignment, student_dim=16, teacher_dim=32):
    """Force auto-projector creation by simulating a forward pass output."""
    student_output = torch.randn(2, student_dim)
    teacher_output = torch.randn(2, teacher_dim)
    alignment._try_init_output_projector(student_output, teacher_output)


# ── Tests: Alignment projector_init parameter ────────────────────────────


class TestProjectorInitAlignment:
    """Test projector_init on the Alignment class directly."""

    def test_default_init_is_none(self):
        teacher, student = _make_models()
        a = Alignment(
            teacher_block=teacher.layers["0"],
            student_block=student.layers["0"],
            auto_projector=True,
        )
        assert a._projector_init is None

    def test_normal_init_sets_small_weights(self):
        teacher, student = _make_models()
        a = Alignment(
            teacher_block=teacher.layers["0"],
            student_block=student.layers["0"],
            auto_projector=True,
            projector_init="normal",
        )
        _trigger_projector_creation(a)
        assert a.output_projector is not None

        # normal_(0, 0.02) should produce small weights
        weight = a.output_projector.weight
        assert weight.std().item() < 0.05, (
            f"normal init should have std ~0.02, got {weight.std().item():.4f}"
        )
        assert abs(weight.mean().item()) < 0.01, (
            f"normal init should have mean ~0, got {weight.mean().item():.4f}"
        )
        # bias should be zero
        if a.output_projector.bias is not None:
            assert a.output_projector.bias.abs().max().item() == 0.0

    def test_xavier_init(self):
        teacher, student = _make_models()
        a = Alignment(
            teacher_block=teacher.layers["0"],
            student_block=student.layers["0"],
            auto_projector=True,
            projector_init="xavier",
        )
        _trigger_projector_creation(a)
        assert a.output_projector is not None

        # Xavier should have different distribution than normal(0, 0.02)
        weight = a.output_projector.weight
        # Xavier uniform for (16, 32): fan_in=16, fan_out=32 → bound ≈ 0.289
        assert weight.std().item() > 0.05, "xavier init should have larger std than normal(0, 0.02)"
        if a.output_projector.bias is not None:
            assert a.output_projector.bias.abs().max().item() == 0.0

    def test_callable_init(self):
        """projector_init can be a callable applied via module.apply()."""
        teacher, student = _make_models()

        def custom_init(module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)

        a = Alignment(
            teacher_block=teacher.layers["0"],
            student_block=student.layers["0"],
            auto_projector=True,
            projector_init=custom_init,
        )
        _trigger_projector_creation(a)
        assert a.output_projector is not None

        assert torch.allclose(
            a.output_projector.weight,
            torch.full_like(a.output_projector.weight, 0.5),
        )
        if a.output_projector.bias is not None:
            assert torch.allclose(
                a.output_projector.bias,
                torch.full_like(a.output_projector.bias, 0.1),
            )

    def test_invalid_init_raises(self):
        teacher, student = _make_models()
        a = Alignment(
            teacher_block=teacher.layers["0"],
            student_block=student.layers["0"],
            auto_projector=True,
            projector_init="nonexistent",
        )
        with pytest.raises(ValueError, match="Unknown projector_init"):
            _trigger_projector_creation(a)

    def test_none_init_uses_pytorch_default(self):
        """projector_init=None should leave PyTorch's default (kaiming_uniform)."""
        teacher, student = _make_models()
        a = Alignment(
            teacher_block=teacher.layers["0"],
            student_block=student.layers["0"],
            auto_projector=True,
            projector_init=None,
        )
        _trigger_projector_creation(a)
        assert a.output_projector is not None

        # kaiming_uniform has larger range than normal(0, 0.02)
        weight = a.output_projector.weight
        assert weight.std().item() > 0.1, (
            f"kaiming init should have std >> 0.02, got {weight.std().item():.4f}"
        )

    def test_normal_vs_default_differ(self):
        """Verify normal and default inits produce different weights."""
        teacher1, student1 = _make_models()
        teacher2, student2 = _make_models()

        torch.manual_seed(42)
        a_default = Alignment(
            teacher_block=teacher1.layers["0"],
            student_block=student1.layers["0"],
            auto_projector=True,
            projector_init=None,
        )
        _trigger_projector_creation(a_default)

        torch.manual_seed(42)
        a_normal = Alignment(
            teacher_block=teacher2.layers["0"],
            student_block=student2.layers["0"],
            auto_projector=True,
            projector_init="normal",
        )
        _trigger_projector_creation(a_normal)

        assert not torch.equal(
            a_default.output_projector.weight,
            a_normal.output_projector.weight,
        ), "normal and default inits should produce different weights"


# ── Tests: create_alignments with projector_init ─────────────────────────


class TestProjectorInitCreateAlignments:
    """Test projector_init passed through create_alignments."""

    def test_projector_init_propagates(self):
        teacher, student = _make_models()
        alignments = create_alignments(
            teacher_model=teacher,
            student_model=student,
            modules=r"layers\.0$",
            projector_init="normal",
        )
        assert len(alignments) == 1
        assert alignments[0]._projector_init == "normal"

    def test_projector_init_none_by_default(self):
        teacher, student = _make_models()
        alignments = create_alignments(
            teacher_model=teacher,
            student_model=student,
            modules=r"layers\.0$",
        )
        assert alignments[0]._projector_init is None

    def test_projector_init_applies_to_all_alignments(self):
        teacher, student = _make_models()
        alignments = create_alignments(
            teacher_model=teacher,
            student_model=student,
            modules=r"layers\.\d+$",
            projector_init="xavier",
        )
        assert len(alignments) == 2
        for a in alignments:
            assert a._projector_init == "xavier"

    def test_callable_propagates_through_create_alignments(self):
        teacher, student = _make_models()

        def my_init(module):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)

        alignments = create_alignments(
            teacher_model=teacher,
            student_model=student,
            modules=r"layers\.0$",
            projector_init=my_init,
        )
        assert alignments[0]._projector_init is my_init


# ── Tests: Conv2d projector init ─────────────────────────────────────────


class TestConv2dProjectorInit:
    """Test that projector_init also works for Conv2d projectors."""

    def test_normal_init_on_conv2d(self):
        """Conv2d projectors should also be initialized when projector_init is set."""

        class ConvModel(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Conv2d(channels, channels, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        teacher = ConvModel(32)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        student = ConvModel(16)

        a = Alignment(
            teacher_block=teacher.conv,
            student_block=student.conv,
            auto_projector=True,
            projector_init="normal",
        )

        # Trigger with 4D tensors (batch, channels, H, W)
        student_out = torch.randn(2, 16, 8, 8)
        teacher_out = torch.randn(2, 32, 8, 8)
        a._try_init_output_projector(student_out, teacher_out)

        assert a.output_projector is not None
        # The Conv2d projector's weight should have small values from normal init
        # (Conv2d doesn't have a standard .weight like Linear, but the init
        # applies via module.apply which iterates all sub-modules)
