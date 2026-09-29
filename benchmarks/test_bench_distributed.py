"""Distributed training throughput benchmarks.

Measures wall-clock time for full distiller.train() under different
teacher placement x student wrapping x distiller type combinations
using mp.spawn.

Requires at least 4 CUDA GPUs.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 4 if torch.cuda.is_available() else True,
        reason="At least 4 GPUs required",
    ),
]


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_process(rank, world_size, port, fn, *args):
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
# Helpers
# ---------------------------------------------------------------------------


def _make_teacher_student(rank):
    from tests.silverspoon_kd.conftest import SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3)
    student.to(f"cuda:{rank}")
    return teacher, student


def _make_alignments(teacher, student):
    from silverspoon_kd import create_alignments

    alignments = create_alignments(
        teacher_model=teacher,
        student_model=student,
        modules=r"layers\.\d+$",
        output_selector_index=None,
        auto_projector=True,
    )
    for a in alignments:
        a.auto_device_match = True
    return alignments


def _dataset(n=60):
    from tests.silverspoon_kd.conftest import DummyDataset

    return DummyDataset(num_samples=n, seq_len=16)


def _fsdp_split_teacher(teacher, rank, world_size):
    """Place teacher on dedicated GPUs (rank+2) with FSDP sharding."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        ShardingStrategy,
    )

    from silverspoon_kd.distributed.strategies import _build_wrap_policy

    teacher_gpu = rank + 2
    torch.cuda.set_device(teacher_gpu)
    teacher.to(f"cuda:{teacher_gpu}")
    teacher_group = dist.new_group(ranks=list(range(world_size)))
    teacher = FSDP(
        teacher,
        process_group=teacher_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.device(f"cuda:{teacher_gpu}"),
        auto_wrap_policy=_build_wrap_policy(teacher, wrap_cls="SimpleBlock"),
    )
    torch.cuda.set_device(rank)
    return teacher


# ---------------------------------------------------------------------------
# Holistic workers
# ---------------------------------------------------------------------------


def _rep_ddp_hol_worker(rank, world_size, tmpdir):
    from silverspoon_kd import HolisticDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
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
        train_dataset=_dataset(200),
    )
    distiller.train()


def _fsdp_all_ddp_hol_worker(rank, world_size, tmpdir):
    from silverspoon_kd import HolisticDistiller, TrainingArguments
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
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
        train_dataset=_dataset(200),
    )
    distiller.train()


def _fsdp_split_ddp_hol_worker(rank, world_size, tmpdir):
    from silverspoon_kd import HolisticDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
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
        train_dataset=_dataset(200),
    )
    distiller.train()


def _rep_fsdp_hol_worker(rank, world_size, tmpdir):
    from silverspoon_kd import HolisticDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        fsdp="full_shard",
    )
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(200),
    )
    distiller.train()


# ---------------------------------------------------------------------------
# Blockwise workers
# ---------------------------------------------------------------------------


def _rep_ddp_bkd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import BlockwiseDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
    )
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(200),
    )
    distiller.train()


def _fsdp_split_ddp_bkd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import BlockwiseDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
    )
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(200),
    )
    distiller.train()


def _rep_fsdp_bkd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import BlockwiseDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        fsdp="full_shard",
    )
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(200),
    )
    distiller.train()


# ---------------------------------------------------------------------------
# ResponseBased workers
# ---------------------------------------------------------------------------


def _rep_ddp_reskd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        auto_device_match=True,
    )
    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=_dataset(200),
    )
    distiller.train()


def _fsdp_split_ddp_reskd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        auto_device_match=True,
    )
    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=_dataset(200),
    )
    distiller.train()


def _rep_fsdp_reskd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

    teacher, student = _make_teacher_student(rank)
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=20,
        per_device_train_batch_size=4,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        auto_device_match=True,
        fsdp="full_shard",
    )
    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=_dataset(200),
    )
    distiller.train()


# ---------------------------------------------------------------------------
# Spawn helper
# ---------------------------------------------------------------------------


def _spawn(fn, tmp_path):
    port = _find_free_port()
    mp.spawn(
        _init_process,
        args=(2, port, fn, str(tmp_path)),
        nprocs=2,
        join=True,
    )


# ---------------------------------------------------------------------------
# Benchmark classes
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="dist-holistic")
class TestHolisticDistributed:
    """Benchmark HolisticDistiller under different distributed configs.

    All benchmarks use 20 training steps with batch_size=4.
    """

    def test_replicated_ddp(self, benchmark, tmp_path):
        """Baseline: replicated teacher + DDP student."""
        benchmark.pedantic(
            _spawn,
            args=(_rep_ddp_hol_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )

    def test_fsdp_all_ranks_ddp(self, benchmark, tmp_path):
        """FSDP teacher (all ranks) + DDP student."""
        benchmark.pedantic(
            _spawn,
            args=(_fsdp_all_ddp_hol_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )

    def test_fsdp_split_ddp(self, benchmark, tmp_path):
        """Split FSDP teacher (GPUs 2,3) + DDP student (GPUs 0,1)."""
        benchmark.pedantic(
            _spawn,
            args=(_fsdp_split_ddp_hol_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )

    def test_replicated_fsdp_student(self, benchmark, tmp_path):
        """Replicated teacher + FSDP student."""
        benchmark.pedantic(
            _spawn,
            args=(_rep_fsdp_hol_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )


@pytest.mark.benchmark(group="dist-blockwise")
class TestBlockwiseDistributed:
    """Benchmark BlockwiseDistiller under different distributed configs.

    All benchmarks use 20 training steps with batch_size=4.
    """

    def test_replicated_ddp(self, benchmark, tmp_path):
        """Baseline: replicated teacher + DDP student."""
        benchmark.pedantic(
            _spawn,
            args=(_rep_ddp_bkd_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )

    def test_fsdp_split_ddp(self, benchmark, tmp_path):
        """Split FSDP teacher (GPUs 2,3) + DDP student (GPUs 0,1)."""
        benchmark.pedantic(
            _spawn,
            args=(_fsdp_split_ddp_bkd_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )

    def test_replicated_fsdp_student(self, benchmark, tmp_path):
        """Replicated teacher + FSDP student."""
        benchmark.pedantic(
            _spawn,
            args=(_rep_fsdp_bkd_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )


@pytest.mark.benchmark(group="dist-response-based")
class TestResponseBasedDistributed:
    """Benchmark ResponseBasedDistiller under different distributed configs.

    All benchmarks use 20 training steps with batch_size=4.
    """

    def test_replicated_ddp(self, benchmark, tmp_path):
        """Baseline: replicated teacher + DDP student."""
        benchmark.pedantic(
            _spawn,
            args=(_rep_ddp_reskd_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )

    def test_fsdp_split_ddp(self, benchmark, tmp_path):
        """Split FSDP teacher (GPUs 2,3) + DDP student (GPUs 0,1)."""
        benchmark.pedantic(
            _spawn,
            args=(_fsdp_split_ddp_reskd_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )

    def test_replicated_fsdp_student(self, benchmark, tmp_path):
        """Replicated teacher + FSDP student."""
        benchmark.pedantic(
            _spawn,
            args=(_rep_fsdp_reskd_worker, tmp_path),
            rounds=3,
            warmup_rounds=1,
        )
