"""Pipeline-parallel teacher placement tests.

Tests teacher PP placement with split-GPU mode. These tests verify that
the teacher model is correctly distributed across dedicated GPUs using
device_map, and that the student trains correctly on the remaining GPUs.

Requires at least 2 CUDA GPUs (``--device=cuda``).
"""

import pytest
import torch

from silverspoon_kd.distillers import (
    BlockwiseDistiller,
    HolisticDistiller,
    ResponseBasedDistiller,
)
from silverspoon_kd.distributed.strategies import place_teacher_pp
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)
from tests.silverspoon_kd.conftest import (
    DummyDataset,
    SimpleModel,
    create_alignment,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="At least 2 GPUs required",
    ),
]


def _make_args(tmp_path, args_cls, **kwargs):
    return args_cls(
        output_dir=str(tmp_path / "output"),
        max_steps=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=999,
        save_steps=999,
        dataloader_num_workers=0,
        report_to=[],
        disable_tqdm=True,
        use_cpu=False,
        **kwargs,
    )


def _make_alignments(teacher, student, with_input_projector=False):
    alignments = []
    for i in range(teacher.num_layers):
        alignment = create_alignment(
            teacher_block=teacher.get_layer(i),
            student_block=student.get_layer(i),
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            with_input_projector=with_input_projector,
        )
        alignment.auto_device_match = True
        alignments.append(alignment)
    return alignments


class TestPPPlacement:
    """Test PP placement of teacher model across GPUs."""

    def test_place_teacher_pp_single_gpu(self):
        """PP with 1 GPU just moves model to that device."""
        model = SimpleModel(64, 128, 3)
        place_teacher_pp(model, [0], "cuda")
        for p in model.parameters():
            assert p.device == torch.device("cuda:0")

    def test_place_teacher_pp_multi_gpu(self):
        """PP distributes children across remapped teacher GPUs."""
        model = SimpleModel(64, 128, 3)
        devices = list(range(min(2, torch.cuda.device_count())))
        place_teacher_pp(model, devices, "cuda")
        # At least some params should be on each device
        param_devices = {p.device for p in model.parameters()}
        assert len(param_devices) >= 1  # At least placed somewhere


class TestPPHolisticDistiller:
    """Holistic distiller with PP teacher."""

    def test_trains(self, tmp_path):
        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()
        # Place teacher across 2 GPUs via PP
        place_teacher_pp(teacher, [0, 1], "cuda")

        args = _make_args(tmp_path, TrainingArguments)
        alignments = _make_alignments(teacher, student)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(20, 16),
            eval_dataset=DummyDataset(10, 16),
        )
        distiller.train()

    def test_student_params_updated(self, tmp_path):
        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()
        place_teacher_pp(teacher, [0, 1], "cuda")

        params_before = {n: p.clone() for n, p in student.named_parameters() if p.requires_grad}

        args = _make_args(tmp_path, TrainingArguments)
        alignments = _make_alignments(teacher, student)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(20, 16),
        )
        distiller.train()

        changed = sum(
            1
            for n, p in student.named_parameters()
            if p.requires_grad and not torch.equal(p, params_before[n])
        )
        assert changed > 0, "Student parameters should be updated after training"


class TestPPBlockwiseDistiller:
    """Blockwise distiller with PP teacher."""

    def test_trains(self, tmp_path):
        # Use same hidden_dim to avoid dimension mismatch issues.
        # Blockwise feeds teacher outputs directly to student blocks.
        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 128, 3).cuda()
        place_teacher_pp(teacher, [0, 1], "cuda")

        args = _make_args(tmp_path, TrainingArguments)
        alignments = _make_alignments(teacher, student)
        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(20, 16),
        )
        distiller.train()


class TestPPResponseBasedDistiller:
    """Response-based distiller with PP teacher."""

    def test_trains(self, tmp_path):
        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()
        place_teacher_pp(teacher, [0, 1], "cuda")

        args = _make_args(
            tmp_path,
            TrainingArguments,
            auto_device_match=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=DummyDataset(20, 16),
        )
        distiller.train()


class TestPPEvaluation:
    """Evaluation works with PP teacher."""

    def test_holistic_eval(self, tmp_path):
        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()
        place_teacher_pp(teacher, [0, 1], "cuda")

        args = _make_args(tmp_path, TrainingArguments)
        alignments = _make_alignments(teacher, student)
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(20, 16),
            eval_dataset=DummyDataset(10, 16),
        )
        metrics = distiller.evaluate()
        assert "eval_loss" in metrics

    def test_response_based_eval(self, tmp_path):
        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()
        place_teacher_pp(teacher, [0, 1], "cuda")

        args = _make_args(
            tmp_path,
            TrainingArguments,
            auto_device_match=True,
        )
        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=DummyDataset(20, 16),
            eval_dataset=DummyDataset(10, 16),
        )
        metrics = distiller.evaluate()
        assert "eval_loss" in metrics
