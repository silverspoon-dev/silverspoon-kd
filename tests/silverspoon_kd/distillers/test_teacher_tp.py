"""Tensor-parallel teacher placement tests.

Tests teacher TP placement with split-GPU mode. Includes end-to-end
training tests that exercise the full ``distiller.train()`` path.

Requires at least 4 CUDA GPUs (``--device=cuda``).

TP with split GPUs requires switching ``torch.cuda.set_device()``
to the teacher GPU during teacher forward because DTensor operations check
``torch.cuda.current_device()`` against the DeviceMesh.
``install_tp_device_hooks()`` handles this via pre/post forward hooks.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from silverspoon_kd.distributed.strategies import parallelize_teacher_tp


def _find_free_port():
    """Find a free TCP port by briefly binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


_requires_4_gpus = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 4 if torch.cuda.is_available() else True,
        reason="At least 4 GPUs required for TP tests",
    ),
]


def _init_process(rank, world_size, port, fn, *args):
    """Initialize distributed process group and run fn."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Strategy-level workers (wrap + forward only)
# ---------------------------------------------------------------------------


def _tp_placement_worker(rank, world_size, tmpdir):
    """Worker that tests TP placement of a model with _tp_plan."""
    from tests.silverspoon_kd.conftest import TPSimpleModel

    student_gpu = rank
    teacher = TPSimpleModel(64, 128, 2)
    remapped = [2, 3]
    teacher = parallelize_teacher_tp(teacher, remapped, "cuda")

    # TP requires current CUDA device to match the DeviceMesh during forward
    teacher_gpu = remapped[rank]
    torch.cuda.set_device(teacher_gpu)

    batch = torch.randint(0, 128, (2, 16), device=f"cuda:{teacher_gpu}")
    with torch.no_grad():
        output = teacher(batch)
    assert output.logits is not None
    assert output.logits.shape[0] == 2

    # Restore student device
    torch.cuda.set_device(student_gpu)


# ---------------------------------------------------------------------------
# End-to-end training workers (full distiller.train() path)
# ---------------------------------------------------------------------------


def _tp_train_worker(rank, world_size, tmpdir):
    """Full training with TP teacher on split GPUs.

    Teacher TP-sharded on GPUs 2,3. Student on GPUs 0,1.
    install_tp_device_hooks() switches torch.cuda.current_device()
    around teacher forward to satisfy DTensor's DeviceMesh check.
    """
    from silverspoon_kd import HolisticDistiller, create_alignments
    from silverspoon_kd.distributed.strategies import install_tp_device_hooks
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel, TPSimpleModel

    student_gpu = rank

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = SimpleModel(64, 64, 2)
    student.to(f"cuda:{student_gpu}")

    alignments = create_alignments(
        teacher_model=teacher,
        student_model=student,
        modules=r"layers\.\d+$",
        output_selector_index=None,
        auto_projector=True,
    )

    remapped = [2, 3]
    teacher = parallelize_teacher_tp(teacher, remapped, "cuda")
    install_tp_device_hooks(teacher)

    for a in alignments:
        a.auto_device_match = True

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=3,
        per_device_train_batch_size=2,
        logging_steps=999,
        save_steps=999,
        learning_rate=1e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
    )

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=DummyDataset(num_samples=20, seq_len=16),
    )

    distiller.train()
    assert distiller.state.global_step == 3


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestTPPlacement:
    """Test TP placement of teacher model."""

    def test_no_tp_plan_raises(self):
        """Model without _tp_plan should raise ValueError."""
        from tests.silverspoon_kd.conftest import SimpleModel

        model = SimpleModel(64, 128, 3)
        with pytest.raises(ValueError, match="_tp_plan"):
            parallelize_teacher_tp(model, [0, 1], "cuda")

    def test_tp_model_has_plan(self):
        """TPSimpleModel should have a _tp_plan attribute."""
        from tests.silverspoon_kd.conftest import TPSimpleModel

        model = TPSimpleModel(64, 128, 2)
        plan = model._tp_plan
        assert isinstance(plan, dict)
        assert len(plan) > 0

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        torch.cuda.device_count() < 4 if torch.cuda.is_available() else True,
        reason="At least 4 GPUs required",
    )
    def test_tp_placement_multi_process(self, tmp_path):
        """TP placement works in multi-process context with 2 GPUs."""
        port = _find_free_port()
        mp.spawn(
            _init_process,
            args=(2, port, _tp_placement_worker, str(tmp_path)),
            nprocs=2,
            join=True,
        )


class TestTPTraining:
    """End-to-end training with TP teacher."""

    pytestmark = _requires_4_gpus

    def test_full_training(self, tmp_path):
        """TP teacher completes full distiller.train() loop."""
        port = _find_free_port()
        mp.spawn(
            _init_process,
            args=(2, port, _tp_train_worker, str(tmp_path)),
            nprocs=2,
            join=True,
        )


class TestTPValidation:
    """TP validation tests (CPU-safe)."""

    def test_non_distributed_raises(self):
        """TP without distributed raises an error."""
        from tests.silverspoon_kd.conftest import TPSimpleModel

        model = TPSimpleModel(64, 128, 2)
        if not dist.is_initialized():
            with pytest.raises((RuntimeError, ValueError)):
                parallelize_teacher_tp(model, [0, 1], "cuda")
