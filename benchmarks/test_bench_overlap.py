"""Forward overlap benchmarks.

Compares training throughput with and without overlapped teacher-student
forward passes (overlap_teacher_forward=True vs False).

Uses FSDP-split teacher placement (teacher on GPUs 2,3, student DDP on
GPUs 0,1) because overlap benefits are most pronounced when teacher and
student reside on different physical GPUs.  Larger models than the
standard distributed benchmarks make the difference measurable.

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


def _make_large_teacher_student(rank):
    """Create larger models so overlap benefits are measurable."""
    from tests.silverspoon_kd.conftest import SimpleModel

    teacher = SimpleModel(64, 2048, 12)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 1024, 12)
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


def _dataset():
    from tests.silverspoon_kd.conftest import DummyDataset

    return DummyDataset(num_samples=2000, seq_len=128)


# ---------------------------------------------------------------------------
# Holistic overlap workers
# ---------------------------------------------------------------------------


def _overlap_on_hol_worker(rank, world_size, tmpdir):
    from silverspoon_kd import HolisticDistiller, TrainingArguments

    teacher, student = _make_large_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=50,
        per_device_train_batch_size=16,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        overlap_teacher_forward=True,
    )
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()


def _overlap_off_hol_worker(rank, world_size, tmpdir):
    from silverspoon_kd import HolisticDistiller, TrainingArguments

    teacher, student = _make_large_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=50,
        per_device_train_batch_size=16,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        overlap_teacher_forward=False,
    )
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()


# ---------------------------------------------------------------------------
# Blockwise overlap workers
# ---------------------------------------------------------------------------


def _overlap_on_bkd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import BlockwiseDistiller, TrainingArguments

    teacher, student = _make_large_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=50,
        per_device_train_batch_size=16,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        overlap_teacher_forward=True,
    )
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()


def _overlap_off_bkd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import BlockwiseDistiller, TrainingArguments

    teacher, student = _make_large_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=50,
        per_device_train_batch_size=16,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        overlap_teacher_forward=False,
    )
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()


# ---------------------------------------------------------------------------
# ResponseBased overlap workers
# ---------------------------------------------------------------------------


def _overlap_on_reskd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

    teacher, student = _make_large_teacher_student(rank)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=50,
        per_device_train_batch_size=16,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        auto_device_match=True,
        overlap_teacher_forward=True,
    )
    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()


def _overlap_off_reskd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

    teacher, student = _make_large_teacher_student(rank)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)

    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=50,
        per_device_train_batch_size=16,
        logging_steps=999,
        save_steps=999,
        learning_rate=5e-3,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        local_rank=rank,
        auto_device_match=True,
        overlap_teacher_forward=False,
    )
    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=_dataset(),
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
# Benchmark class
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="overlap-forward")
class TestForwardOverlap:
    """Compare training throughput with/without teacher-student forward overlap.

    Uses FSDP-split teacher (GPUs 2,3) + DDP student (GPUs 0,1) with
    larger models to make the overlap benefit measurable.

    Teacher: SimpleModel(64, 2048, 12) -- ~50M params
    Student: SimpleModel(64, 1024, 12) -- ~12M params
    """

    def test_holistic_overlap_off(self, benchmark, tmp_path):
        """Holistic: serial teacher -> student forward."""
        benchmark.pedantic(
            _spawn,
            args=(_overlap_off_hol_worker, tmp_path),
            rounds=5,
            warmup_rounds=1,
        )

    def test_holistic_overlap_on(self, benchmark, tmp_path):
        """Holistic: overlapped teacher / student forward."""
        benchmark.pedantic(
            _spawn,
            args=(_overlap_on_hol_worker, tmp_path),
            rounds=5,
            warmup_rounds=1,
        )

    def test_blockwise_overlap_off(self, benchmark, tmp_path):
        """Blockwise: serial teacher -> student forward (no pipelining)."""
        benchmark.pedantic(
            _spawn,
            args=(_overlap_off_bkd_worker, tmp_path),
            rounds=5,
            warmup_rounds=1,
        )

    def test_blockwise_overlap_on(self, benchmark, tmp_path):
        """Blockwise: pipelined teacher / student forward."""
        benchmark.pedantic(
            _spawn,
            args=(_overlap_on_bkd_worker, tmp_path),
            rounds=5,
            warmup_rounds=1,
        )

    def test_response_based_overlap_off(self, benchmark, tmp_path):
        """ResponseBased: serial teacher -> student forward."""
        benchmark.pedantic(
            _spawn,
            args=(_overlap_off_reskd_worker, tmp_path),
            rounds=5,
            warmup_rounds=1,
        )

    def test_response_based_overlap_on(self, benchmark, tmp_path):
        """ResponseBased: overlapped teacher / student forward."""
        benchmark.pedantic(
            _spawn,
            args=(_overlap_on_reskd_worker, tmp_path),
            rounds=5,
            warmup_rounds=1,
        )
