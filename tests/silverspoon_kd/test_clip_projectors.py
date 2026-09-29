"""Tests for clip_projectors training argument.

Verifies that clip_projectors=False excludes projector parameters from
gradient clipping, and clip_projectors=True (default) includes them.
"""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from silverspoon_kd import (
    Alignment,
    HolisticDistiller,
    TrainingArguments,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


class TinyModel(nn.Module):
    """Minimal 2-layer model for testing. Input dim is always `inp`."""

    def __init__(self, inp=8, hidden=16, out=4):
        super().__init__()
        self.layer0 = nn.Linear(inp, hidden)
        self.layer1 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return self.head(x)


def _make_models(inp=8, teacher_hidden=16, student_hidden=8, out=4):
    torch.manual_seed(0)
    teacher = TinyModel(inp=inp, hidden=teacher_hidden, out=out)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = TinyModel(inp=inp, hidden=student_hidden, out=out)
    student.train()
    return teacher, student


def _make_alignment(teacher, student):
    """Create a single alignment with an auto-projector (dimension mismatch)."""
    return Alignment(
        teacher_block=teacher.layer1,
        student_block=student.layer1,
        auto_projector=True,
    )


def _make_dataset(n=64, inp=8):
    """Create a simple tensor dataset."""
    from torch.utils.data import TensorDataset

    x = torch.randn(n, inp)
    return TensorDataset(x)


def _collator(batch):
    return {"x": torch.stack([b[0] for b in batch])}


def _run_one_step(teacher, student, alignment, clip_projectors, output_dir, max_grad_norm=1.0):
    """Run one training step and return gradient norms for student and projector."""
    _on_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=1,
        per_device_train_batch_size=16,
        learning_rate=1e-3,
        max_grad_norm=max_grad_norm,
        report_to=[],
        logging_steps=999,
        save_strategy="no",
        clip_projectors=clip_projectors,
        dataloader_pin_memory=not _on_mps,
    )

    dataset = _make_dataset()

    class SimpleHKD(HolisticDistiller):
        def _get_train_sampler(self, dataset=None):
            return torch.utils.data.SequentialSampler(dataset or self.train_dataset)

    trainer = SimpleHKD(
        student_model=student,
        teacher_model=teacher,
        alignments=[alignment],
        args=args,
        train_dataset=dataset,
        data_collator=_collator,
    )
    trainer.train()

    # Collect gradient norms (they'll be zero after optimizer.zero_grad,
    # so we check param changes instead)
    return trainer


# ── Tests ────────────────────────────────────────────────────────────────


class TestClipProjectorsFlag:
    """Tests for the clip_projectors TrainingArguments parameter."""

    def test_default_is_true(self, tmp_path):
        args = TrainingArguments(output_dir=str(tmp_path / "out"))
        assert args.clip_projectors is True

    def test_can_set_false(self, tmp_path):
        args = TrainingArguments(output_dir=str(tmp_path / "out"), clip_projectors=False)
        assert args.clip_projectors is False

    def test_can_set_true_explicitly(self, tmp_path):
        args = TrainingArguments(output_dir=str(tmp_path / "out"), clip_projectors=True)
        assert args.clip_projectors is True


class TestClipProjectorsGradients:
    """Verify that clip_projectors affects gradient clipping scope."""

    def test_clip_all_vs_student_only_produces_different_updates(self, tmp_path):
        """With projectors in the norm, student gradients are scaled more."""
        teacher, student1 = _make_models()
        teacher2 = deepcopy(teacher)
        student2 = deepcopy(student1)

        align1 = _make_alignment(teacher, student1)
        align2 = _make_alignment(teacher2, student2)

        # Run with clip_projectors=True (default)
        _run_one_step(
            teacher, student1, align1, clip_projectors=True, output_dir=tmp_path / "clip_all"
        )

        # Run with clip_projectors=False
        _run_one_step(
            teacher2,
            student2,
            align2,
            clip_projectors=False,
            output_dir=tmp_path / "student_only",
        )

        # Parameters should differ because clipping scope differs
        any_diff = False
        for (n1, p1), (n2, p2) in zip(
            student1.named_parameters(), student2.named_parameters(), strict=True
        ):
            if not torch.equal(p1, p2):
                any_diff = True
                break
        assert any_diff, "Student params should differ between clip_projectors=True and False"

    def test_clip_projectors_false_disables_trainer_clipping(self, tmp_path):
        """When clip_projectors=False, Trainer's max_grad_norm should be 0."""
        teacher, student = _make_models()
        align = _make_alignment(teacher, student)

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=0,  # don't actually train
            per_device_train_batch_size=16,
            max_grad_norm=1.0,
            report_to=[],
            clip_projectors=False,
        )

        trainer = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=[align],
            args=args,
            train_dataset=_make_dataset(),
            data_collator=_collator,
        )

        # After init, Trainer's max_grad_norm should be disabled
        assert trainer.args.max_grad_norm == 0, (
            "Trainer's max_grad_norm should be 0 when clip_projectors=False"
        )
        # But the original norm is stored for manual clipping
        assert trainer._student_only_clip_norm == 1.0

    def test_clip_projectors_true_preserves_trainer_clipping(self, tmp_path):
        """When clip_projectors=True (default), Trainer's max_grad_norm is preserved."""
        teacher, student = _make_models()
        align = _make_alignment(teacher, student)

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=0,
            per_device_train_batch_size=16,
            max_grad_norm=1.0,
            report_to=[],
            clip_projectors=True,
        )

        trainer = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=[align],
            args=args,
            train_dataset=_make_dataset(),
            data_collator=_collator,
        )

        # max_grad_norm should be preserved (unless composite optimizer overrides it)
        assert trainer._student_only_clip_norm is None

    def test_no_clipping_when_max_grad_norm_zero(self, tmp_path):
        """clip_projectors=False should not activate when max_grad_norm=0."""
        teacher, student = _make_models()
        align = _make_alignment(teacher, student)

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=0,
            per_device_train_batch_size=16,
            max_grad_norm=0,  # no clipping at all
            report_to=[],
            clip_projectors=False,
        )

        trainer = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=[align],
            args=args,
            train_dataset=_make_dataset(),
            data_collator=_collator,
        )

        assert trainer._student_only_clip_norm is None, (
            "Should not activate student-only clipping when max_grad_norm=0"
        )


# ── BlockwiseDistiller (per-alignment clipping) ─────────────────────────────


class TestClipProjectorsBlockwise:
    """clip_projectors applies to BlockwiseDistiller's per-alignment clipping."""

    @staticmethod
    def _gradient_norms(tmp_path, clip_projectors, max_grad_norm):
        """Run one blockwise training step; return (student block norm, projector norm)."""
        from silverspoon_kd import BlockwiseDistiller
        from silverspoon_kd.alignments.projectors import GenericLinearProjector

        teacher, student = _make_models()  # seeds torch before building the models
        input_proj = GenericLinearProjector(16, 8, mode="input", apply_to_arg=0)
        output_proj = GenericLinearProjector(8, 16, mode="output")
        alignment = Alignment(
            teacher_block=teacher.layer1,
            student_block=student.layer1,
            input_projector=input_proj,
            output_projector=output_proj,
        )
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=1,
            per_device_train_batch_size=16,
            max_grad_norm=max_grad_norm,
            report_to=[],
            logging_steps=999,
            save_strategy="no",
            clip_projectors=clip_projectors,
            use_cpu=True,
            dataloader_pin_memory=False,
        )
        dataset = _make_dataset()
        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=[alignment],
            args=args,
            train_dataset=dataset,
            data_collator=_collator,
        )
        distiller._register_capture()
        distiller.training_step(distiller.model, _collator([dataset[i] for i in range(16)]))

        def norm(params):
            return torch.norm(torch.stack([p.grad.norm() for p in params])).item()

        return (
            norm(student.layer1.parameters()),
            norm([*input_proj.parameters(), *output_proj.parameters()]),
        )

    def test_projectors_are_left_unclipped_when_disabled(self, tmp_path):
        tiny = 1e-3
        _, proj_unclipped = self._gradient_norms(tmp_path / "ref", True, max_grad_norm=1e6)
        student_norm, proj_norm = self._gradient_norms(tmp_path / "off", False, max_grad_norm=tiny)

        assert student_norm <= tiny * 1.01
        assert proj_norm == pytest.approx(proj_unclipped, rel=1e-5)
        assert proj_unclipped > tiny

    def test_projectors_are_clipped_by_default(self, tmp_path):
        tiny = 1e-3
        student_norm, proj_norm = self._gradient_norms(tmp_path / "on", True, max_grad_norm=tiny)

        assert (student_norm**2 + proj_norm**2) ** 0.5 <= tiny * 1.01
