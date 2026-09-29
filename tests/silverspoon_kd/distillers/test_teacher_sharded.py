"""FSDP-sharded teacher placement tests.

Tests both:
- String ``"sharded"`` (all-ranks FSDP)
- ``TeacherPlacement(strategy="sharded")`` (split-GPU FSDP)

Includes end-to-end training tests that exercise the full
``distiller.train()`` path (not just strategy functions in isolation).

Requires 4+ GPUs and multi-process context for distributed tests.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from silverspoon_kd.distributed.strategies import (
    _build_wrap_policy,
    shard_teacher_fsdp_all_ranks,
    shard_teacher_fsdp_split,
)


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
        reason="At least 4 GPUs required for sharded tests",
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


def _all_ranks_sharded_worker(rank, world_size, tmpdir):
    """Worker that tests all-ranks FSDP sharding."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from tests.silverspoon_kd.conftest import SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)

    assert isinstance(teacher, FSDP)

    # Forward pass should work
    batch = torch.randint(0, 128, (2, 16), device=f"cuda:{rank}")
    with torch.no_grad():
        output = teacher(batch)
    assert output.logits is not None


def _split_sharded_worker(rank, world_size, tmpdir):
    """Worker that tests split-GPU FSDP sharding."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from tests.silverspoon_kd.conftest import SimpleModel

    teacher = SimpleModel(64, 128, 3)
    remapped = [2, 3]  # Teacher on remapped GPUs
    teacher = shard_teacher_fsdp_split(teacher, remapped, "cuda")

    assert isinstance(teacher, FSDP)

    # Forward pass on teacher GPU
    teacher_gpu = remapped[rank]
    batch = torch.randint(0, 128, (2, 16), device=f"cuda:{teacher_gpu}")
    with torch.no_grad():
        output = teacher(batch)
    assert output.logits is not None


# ---------------------------------------------------------------------------
# End-to-end training workers (full distiller.train() path)
# ---------------------------------------------------------------------------


def _all_ranks_sharded_train_worker(rank, world_size, tmpdir):
    """Full training with all-ranks FSDP-sharded teacher."""
    from silverspoon_kd import HolisticDistiller, create_alignments
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = SimpleModel(64, 64, 3)
    student.to(f"cuda:{rank}")

    # Create alignments before FSDP wrapping
    alignments = create_alignments(
        teacher_model=teacher,
        student_model=student,
        modules=r"layers\.\d+$",
        output_selector_index=None,
        auto_projector=True,
    )

    # FSDP-shard teacher on same GPUs as student (all-ranks mode)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    for a in alignments:
        a.auto_device_match = True

    params_before = {
        n: p.clone().detach() for n, p in student.named_parameters() if p.requires_grad
    }

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

    assert args.device == torch.device(f"cuda:{rank}"), (
        f"Rank {rank}: args.device={args.device}, expected cuda:{rank}"
    )

    distiller.train()
    assert distiller.state.global_step == 3

    changed = any(
        not torch.equal(p.data, params_before[n])
        for n, p in student.named_parameters()
        if p.requires_grad and n in params_before
    )
    assert changed, "Student params should update during training"


def _split_sharded_train_worker(rank, world_size, tmpdir):
    """Full training with split-GPU FSDP-sharded teacher.

    Teacher on GPUs 2,3 (FSDP). Student on GPUs 0,1.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        ShardingStrategy,
    )

    from silverspoon_kd import HolisticDistiller, create_alignments
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    student_gpu = rank
    teacher_gpu = rank + 2

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = SimpleModel(64, 64, 3)
    student.to(f"cuda:{student_gpu}")

    alignments = create_alignments(
        teacher_model=teacher,
        student_model=student,
        modules=r"layers\.\d+$",
        output_selector_index=None,
        auto_projector=True,
    )

    # FSDP-shard teacher on dedicated GPUs
    torch.cuda.set_device(teacher_gpu)
    teacher.to(f"cuda:{teacher_gpu}")
    teacher_group = dist.new_group(ranks=list(range(world_size)))
    wrap_policy = _build_wrap_policy(teacher, wrap_cls="SimpleBlock")
    teacher = FSDP(
        teacher,
        process_group=teacher_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.device(f"cuda:{teacher_gpu}"),
        auto_wrap_policy=wrap_policy,
    )
    torch.cuda.set_device(student_gpu)

    for a in alignments:
        a.auto_device_match = True

    params_before = {
        n: p.clone().detach() for n, p in student.named_parameters() if p.requires_grad
    }

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

    assert args.device == torch.device(f"cuda:{rank}"), (
        f"Rank {rank}: args.device={args.device}, expected cuda:{rank}"
    )

    distiller.train()
    assert distiller.state.global_step == 3

    # Verify teacher still on teacher GPU (not moved to student device)
    teacher_param_device = next(teacher.parameters()).device
    assert teacher_param_device == torch.device(f"cuda:{teacher_gpu}"), (
        f"Rank {rank}: teacher on {teacher_param_device}, expected cuda:{teacher_gpu}"
    )

    changed = any(
        not torch.equal(p.data, params_before[n])
        for n, p in student.named_parameters()
        if p.requires_grad and n in params_before
    )
    assert changed, "Student params should update during training"


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestAllRanksSharded:
    """String 'sharded' — FSDP across all ranks."""

    pytestmark = _requires_4_gpus

    def test_fsdp_wrapping(self, tmp_path):
        """All-ranks FSDP wraps teacher correctly."""
        port = _find_free_port()
        mp.spawn(
            _init_process,
            args=(2, port, _all_ranks_sharded_worker, str(tmp_path)),
            nprocs=2,
            join=True,
        )

    def test_full_training(self, tmp_path):
        """All-ranks FSDP teacher completes full distiller.train() loop."""
        port = _find_free_port()
        mp.spawn(
            _init_process,
            args=(2, port, _all_ranks_sharded_train_worker, str(tmp_path)),
            nprocs=2,
            join=True,
        )


class TestSplitSharded:
    """TeacherPlacement(strategy='sharded') — split-GPU FSDP."""

    pytestmark = _requires_4_gpus

    def test_fsdp_split_wrapping(self, tmp_path):
        """Split-GPU FSDP wraps teacher on dedicated GPUs."""
        port = _find_free_port()
        mp.spawn(
            _init_process,
            args=(2, port, _split_sharded_worker, str(tmp_path)),
            nprocs=2,
            join=True,
        )

    def test_full_training(self, tmp_path):
        """Split-GPU FSDP teacher completes full distiller.train() loop."""
        port = _find_free_port()
        mp.spawn(
            _init_process,
            args=(2, port, _split_sharded_train_worker, str(tmp_path)),
            nprocs=2,
            join=True,
        )


class TestWrapPolicy:
    """FSDP wrap policy resolution (CPU-safe, no CUDA required)."""

    def test_no_split_modules(self):
        """Model with _no_split_modules gets transformer_auto_wrap_policy."""
        from tests.silverspoon_kd.conftest import TPSimpleModel

        model = TPSimpleModel(64, 128, 2)
        policy = _build_wrap_policy(model)
        assert policy is not None

    def test_explicit_wrap_cls(self):
        """Explicit wrap_cls overrides _no_split_modules."""
        from tests.silverspoon_kd.conftest import SimpleModel

        model = SimpleModel(64, 128, 3)
        policy = _build_wrap_policy(model, wrap_cls="SimpleBlock")
        assert policy is not None

    def test_fallback_size_policy(self):
        """Model without _no_split_modules or wrap_cls falls back to size-based."""
        from tests.silverspoon_kd.conftest import SimpleModel

        model = SimpleModel(64, 128, 3)
        policy = _build_wrap_policy(model)
        assert policy is not None

    def test_non_distributed_sharded_raises(self):
        """Sharded without distributed raises during FSDP init."""
        from tests.silverspoon_kd.conftest import SimpleModel

        model = SimpleModel(64, 128, 3)
        if not dist.is_initialized():
            with pytest.raises(ValueError):
                shard_teacher_fsdp_all_ranks(model, device_id=0)


class TestSetupTeacherPlacementFSDPBypass:
    """Verify _setup_teacher_placement() skips FSDP-wrapped teachers (CPU-safe)."""

    def test_fsdp_teacher_not_moved(self, tmp_path):
        """_setup_teacher_placement should be a no-op when teacher is FSDP-wrapped.

        Calling place_teacher_replicated() on an FSDP-wrapped teacher would move
        it to the student's device and break FSDP's device assertions.
        """
        from unittest.mock import MagicMock

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from silverspoon_kd import HolisticDistiller, create_alignments
        from silverspoon_kd.training_arguments import TrainingArguments
        from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

        teacher = SimpleModel(64, 128, 3)
        teacher.eval()
        student = SimpleModel(64, 64, 3)

        alignments = create_alignments(
            teacher_model=teacher,
            student_model=student,
            modules=r"layers\.\d+$",
            output_selector_index=None,
            auto_projector=True,
        )

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=1,
            per_device_train_batch_size=1,
            report_to=[],
            use_cpu=True,
        )

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(num_samples=4, seq_len=8),
        )

        # Replace teacher with an FSDP mock — _setup_teacher_placement should skip it
        mock_fsdp_teacher = MagicMock(spec=FSDP)
        distiller.teacher_model = mock_fsdp_teacher

        distiller._setup_teacher_placement()

        # Verify the teacher was not replaced or .to()'d
        assert distiller.teacher_model is mock_fsdp_teacher
        # .to() should not have been called on the FSDP mock
        mock_fsdp_teacher.to.assert_not_called()
