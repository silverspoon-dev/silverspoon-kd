"""Multi-GPU tests for distillers.

Tests correctness of all distiller types under DataParallel mode, which is the
standard multi-GPU configuration supported by the HF Trainer.

When ``_n_gpu > 1`` the Trainer wraps the model in ``torch.nn.DataParallel``.
The base distiller template uses ``self.model`` (the unwrapped model) rather
than the ``model`` parameter (which may be DP-wrapped) to avoid duplicating
capture engine hooks across replicas.

Requires at least 2 CUDA GPUs (``--device=cuda``).

.. note::

    Cross-device teacher/student placement (teacher on GPU 0, student on GPU 1)
    is **not** supported because HF Trainer's ``accelerator.prepare()`` moves
    the model to ``args.device`` unconditionally during ``_inner_training_loop``.
    There is no way to prevent this without overriding the inner training loop.
"""

import pytest
import torch

from silverspoon_kd.distillers import (
    BlockwiseDistiller,
    HolisticDistiller,
    ResponseBasedDistiller,
)
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)
from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

# All tests in this file require CUDA and at least 2 real GPUs
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        sum(
            1
            for i in range(torch.cuda.device_count())
            if torch.cuda.get_device_properties(i).total_memory > 4 * 1024**3
        )
        < 2,
        reason="At least 2 GPUs with >4GB required",
    ),
]


def _real_gpu_ids():
    """Return list of GPU ids with >4GB memory (filtering display-only GPUs)."""
    return [
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_properties(i).total_memory > 4 * 1024**3
    ]


def _make_args(tmp_path, max_steps=10, args_cls=TrainingArguments, **kwargs):
    """Create training args for multi-GPU tests."""
    return args_cls(
        output_dir=str(tmp_path / "output"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        save_steps=999,
        eval_steps=999,
        max_steps=max_steps,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        disable_tqdm=True,
        **kwargs,
    )


def _verify_dp_training(distiller, student, initial_params):
    """Verify DataParallel training: loss decreasing + params changed."""
    # Losses finite, positive, and decreasing
    losses = [e["loss"] for e in distiller.state.log_history if "loss" in e]
    assert len(losses) > 0, "No losses logged"
    for v in losses:
        assert torch.isfinite(torch.tensor(v)), f"Non-finite loss: {v}"
        assert v > 0, f"Non-positive loss: {v}"
    if len(losses) >= 4:
        mid = len(losses) // 2
        early = sum(losses[:mid]) / mid
        late = sum(losses[mid:]) / (len(losses) - mid)
        assert late < early, f"Loss not decreasing: early_avg={early:.6f}, late_avg={late:.6f}"

    # Student params changed
    changed = any(
        not torch.equal(v, initial_params[k])
        for k, v in student.state_dict().items()
        if k in initial_params
    )
    assert changed, "No student parameters were updated"


def _make_alignments(teacher, student):
    """Create teacher-student alignments with projectors on correct devices."""
    import torch.nn as nn

    from silverspoon_kd.alignments import Alignment
    from silverspoon_kd.alignments.projectors import GenericLinearProjector

    teacher_dim = teacher.hidden_dim
    alignments = []
    for i in range(teacher.num_layers):
        teacher_block = teacher.get_layer(i)
        student_block = student.get_layer(i)
        student_device = next(student_block.parameters()).device

        student_dim = None
        for mod in student_block.modules():
            if isinstance(mod, nn.Linear):
                student_dim = mod.out_features
                break

        output_projector = None
        if student_dim is not None and student_dim != teacher_dim:
            output_projector = nn.Linear(student_dim, teacher_dim).to(student_device)

        input_projector = None
        if i > 0 and student_dim is not None and student_dim != teacher_dim:
            input_projector = GenericLinearProjector(
                in_features=teacher_dim,
                out_features=student_dim,
                mode="input",
                apply_to_arg=0,
            ).to(student_device)

        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            input_projector=input_projector,
            output_projector=output_projector,
        )
        alignments.append(alignment)
    return alignments


# =======================================================================
#  DataParallel Mode (Trainer wraps model when _n_gpu > 1)
# =======================================================================


class TestDataParallel:
    """Verify training works when Trainer enables DataParallel (_n_gpu > 1).

    Regression test: training_step and compute_loss use self.model
    (unwrapped) rather than the model parameter (potentially
    DataParallel-wrapped).
    """

    def test_holistic_dataparallel(self, tmp_path):
        """HolisticDistiller trains correctly under DataParallel wrapping."""
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        alignments = _make_alignments(teacher, student)
        initial_params = {k: v.clone() for k, v in student.state_dict().items()}

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(40, 16),
        )
        distiller.train()
        _verify_dp_training(distiller, student, initial_params)

    def test_blockwise_dataparallel(self, tmp_path):
        """BlockwiseDistiller trains correctly under DataParallel."""
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        alignments = _make_alignments(teacher, student)
        initial_params = {k: v.clone() for k, v in student.state_dict().items()}

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(40, 16),
        )
        distiller.train()
        _verify_dp_training(distiller, student, initial_params)

    def test_response_based_dataparallel(self, tmp_path):
        """ResponseBasedDistiller trains correctly under DataParallel."""
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        initial_params = {k: v.clone() for k, v in student.state_dict().items()}

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())

        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=DummyDataset(40, 16),
        )
        distiller.train()
        _verify_dp_training(distiller, student, initial_params)


# =======================================================================
#  DataParallel Gradient Correctness
# =======================================================================


class TestDataParallelGradients:
    """Verify per-block parameter updates under DataParallel."""

    def test_blockwise_dp_all_blocks_updated(self, tmp_path):
        """Every blockwise student block has parameters updated after DP training."""
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        alignments = _make_alignments(teacher, student)

        initial_block_params = {}
        for alignment in alignments:
            name = alignment.get_name()
            initial_block_params[name] = {
                k: v.clone() for k, v in alignment.student_block.state_dict().items()
            }

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(40, 16),
        )
        distiller.train()

        for alignment in alignments:
            name = alignment.get_name()
            any_changed = any(
                not torch.equal(v, initial_block_params[name][k])
                for k, v in alignment.student_block.state_dict().items()
            )
            assert any_changed, f"Block {name} was not updated during DP training"


# =======================================================================
#  DataParallel Evaluation
# =======================================================================


class TestDataParallelEvaluation:
    """Verify evaluation works correctly under DataParallel wrapping."""

    def test_holistic_dp_eval(self, tmp_path):
        """HolisticDistiller evaluates correctly under DataParallel."""
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        alignments = _make_alignments(teacher, student)

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())
        args.eval_strategy = "steps"
        args.eval_steps = 5

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(40, 16),
            eval_dataset=DummyDataset(10, 16),
        )
        distiller.train()

        metrics = distiller.evaluate()
        assert "eval_loss" in metrics
        assert metrics["eval_loss"] > 0

    def test_blockwise_dp_eval(self, tmp_path):
        """BlockwiseDistiller evaluates correctly under DataParallel."""
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        alignments = _make_alignments(teacher, student)

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())
        args.eval_strategy = "steps"
        args.eval_steps = 5

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(40, 16),
            eval_dataset=DummyDataset(10, 16),
        )
        distiller.train()

        metrics = distiller.evaluate()
        assert "eval_loss" in metrics
        assert metrics["eval_loss"] > 0

    def test_response_based_dp_eval(self, tmp_path):
        """ResponseBasedDistiller evaluates correctly under DataParallel."""
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())
        args.eval_strategy = "steps"
        args.eval_steps = 5

        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=DummyDataset(40, 16),
            eval_dataset=DummyDataset(10, 16),
        )
        distiller.train()

        metrics = distiller.evaluate()
        assert "eval_loss" in metrics
        assert metrics["eval_loss"] > 0


# =======================================================================
#  DataParallel Teacher Frozen
# =======================================================================


class TestDataParallelTeacherFrozen:
    """Verify teacher parameters do not change during DataParallel training."""

    def test_holistic_teacher_frozen(self, tmp_path):
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        alignments = _make_alignments(teacher, student)
        teacher_params_before = {k: v.clone() for k, v in teacher.state_dict().items()}

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(40, 16),
        )
        distiller.train()

        for k, v in teacher.state_dict().items():
            assert torch.equal(v, teacher_params_before[k]), (
                f"Teacher param {k} changed during DP training"
            )

    def test_blockwise_teacher_frozen(self, tmp_path):
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        alignments = _make_alignments(teacher, student)
        teacher_params_before = {k: v.clone() for k, v in teacher.state_dict().items()}

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(40, 16),
        )
        distiller.train()

        for k, v in teacher.state_dict().items():
            assert torch.equal(v, teacher_params_before[k]), (
                f"Teacher param {k} changed during DP training"
            )

    def test_response_based_teacher_frozen(self, tmp_path):
        gpu0 = torch.device("cuda:0")
        teacher = SimpleModel(64, 128, 3).to(gpu0)
        teacher.eval()
        student = SimpleModel(64, 64, 3).to(gpu0)
        teacher_params_before = {k: v.clone() for k, v in teacher.state_dict().items()}

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = len(_real_gpu_ids())

        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=DummyDataset(40, 16),
        )
        distiller.train()

        for k, v in teacher.state_dict().items():
            assert torch.equal(v, teacher_params_before[k]), (
                f"Teacher param {k} changed during DP training"
            )
