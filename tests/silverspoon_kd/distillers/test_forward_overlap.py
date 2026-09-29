"""Forward overlap tests — verify correctness when teacher/student forwards
run on separate CUDA streams (HKD/response-based KD) or pipelined via per-block events (BKD).

Tests cover:
- Explicit overlap_teacher_forward=True with split-GPU (all 3 distillers)
- Explicit overlap_teacher_forward=False override (disables auto-detection)
- Auto-detection: overlap enabled when teacher/student on different GPUs
- Auto-detection: overlap NOT enabled when teacher/student on same GPU

Requires at least 4 CUDA GPUs (teacher FSDP on GPUs 2,3; student DDP on 0,1).
"""

import os

import pytest
import torch
import torch.distributed as dist

from tests.silverspoon_kd.distillers._distributed_helpers import _spawn_with_retry

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 4 if torch.cuda.is_available() else True,
        reason="At least 4 GPUs required",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_process(rank, world_size, port, fn, *args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # A rank that raises must exit without destroy_process_group(): that call
    # blocks on the other rank, which is then killed only by the NCCL timeout,
    # and the real traceback is lost. Exiting promptly lets mp.spawn stop the
    # other rank and report this one's error.
    completed = False
    try:
        fn(rank, world_size, *args)
        completed = True
    finally:
        if completed:
            dist.destroy_process_group()


def _spawn(fn, tmp_path):
    # Retries on rendezvous port collisions, which happen when several xdist
    # workers spawn process groups at the same time.
    _spawn_with_retry(_init_process, 2, fn, tmp_path)


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


def _snapshot_params(model):
    return {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}


def _dataset():
    from tests.silverspoon_kd.conftest import DummyDataset

    return DummyDataset(num_samples=40, seq_len=16)


def _verify(rank, distiller, student, params_before):
    """Check training completed, loss valid, params updated."""
    assert distiller.state.global_step == 10
    losses = [e["loss"] for e in distiller.state.log_history if "loss" in e]
    assert len(losses) > 0
    for v in losses:
        assert torch.isfinite(torch.tensor(v)) and v > 0
    changed = any(
        not torch.equal(p.data, params_before[n])
        for n, p in student.named_parameters()
        if p.requires_grad and n in params_before
    )
    assert changed, f"Rank {rank}: student params unchanged"
    assert not distiller.teacher_model.training


def _make_holistic_args(tmpdir, rank, **extra):
    from silverspoon_kd.training_arguments import TrainingArguments

    defaults = {
        "output_dir": tmpdir,
        "max_steps": 10,
        "per_device_train_batch_size": 2,
        "logging_steps": 1,
        "save_steps": 999,
        "learning_rate": 5e-3,
        "dataloader_num_workers": 0,
        "report_to": [],
        "remove_unused_columns": False,
        "local_rank": rank,
    }
    defaults.update(extra)
    return TrainingArguments(**defaults)


def _make_blockwise_args(tmpdir, rank, **extra):
    from silverspoon_kd.training_arguments import TrainingArguments

    defaults = {
        "output_dir": tmpdir,
        "max_steps": 10,
        "per_device_train_batch_size": 2,
        "logging_steps": 1,
        "save_steps": 999,
        "learning_rate": 5e-3,
        "dataloader_num_workers": 0,
        "report_to": [],
        "remove_unused_columns": False,
        "local_rank": rank,
    }
    defaults.update(extra)
    return TrainingArguments(**defaults)


def _make_response_args(tmpdir, rank, **extra):
    from silverspoon_kd.training_arguments import TrainingArguments

    defaults = {
        "output_dir": tmpdir,
        "max_steps": 10,
        "per_device_train_batch_size": 2,
        "logging_steps": 1,
        "save_steps": 999,
        "learning_rate": 5e-3,
        "dataloader_num_workers": 0,
        "report_to": [],
        "remove_unused_columns": False,
        "local_rank": rank,
        "alpha": 0.0,
        "auto_device_match": True,
        "auto_dtype_match": True,
    }
    defaults.update(extra)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# Workers: overlap=True (explicit), split-GPU
# ---------------------------------------------------------------------------


def _overlap_on_hol_worker(rank, world_size, tmpdir):
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)
    params_before = _snapshot_params(student)

    args = _make_holistic_args(tmpdir, rank, overlap_teacher_forward=True)
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()

    assert distiller._overlap_teacher_forward is True
    assert distiller._teacher_stream is not None
    _verify(rank, distiller, student, params_before)


def _overlap_on_bkd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)
    params_before = _snapshot_params(student)

    args = _make_blockwise_args(tmpdir, rank, overlap_teacher_forward=True)
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()

    assert distiller._overlap_teacher_forward is True
    assert distiller._teacher_stream is not None
    _verify(rank, distiller, student, params_before)


def _overlap_on_reskd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)
    params_before = _snapshot_params(student)

    args = _make_response_args(tmpdir, rank, overlap_teacher_forward=True)
    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()

    assert distiller._overlap_teacher_forward is True
    assert distiller._teacher_stream is not None
    _verify(rank, distiller, student, params_before)


# ---------------------------------------------------------------------------
# Workers: overlap=False (explicit override), split-GPU
# ---------------------------------------------------------------------------


def _overlap_off_hol_worker(rank, world_size, tmpdir):
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)
    params_before = _snapshot_params(student)

    args = _make_holistic_args(tmpdir, rank, overlap_teacher_forward=False)
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()

    # Explicit False should override auto-detection even with split-GPU
    assert distiller._overlap_teacher_forward is False
    assert distiller._teacher_stream is None
    _verify(rank, distiller, student, params_before)


def _overlap_off_bkd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)
    params_before = _snapshot_params(student)

    args = _make_blockwise_args(tmpdir, rank, overlap_teacher_forward=False)
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()

    assert distiller._overlap_teacher_forward is False
    assert distiller._teacher_stream is None
    _verify(rank, distiller, student, params_before)


def _overlap_off_reskd_worker(rank, world_size, tmpdir):
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)
    params_before = _snapshot_params(student)

    args = _make_response_args(tmpdir, rank, overlap_teacher_forward=False)
    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()

    assert distiller._overlap_teacher_forward is False
    assert distiller._teacher_stream is None
    _verify(rank, distiller, student, params_before)


# ---------------------------------------------------------------------------
# Workers: auto-detection
# ---------------------------------------------------------------------------


def _auto_detect_split_worker(rank, world_size, tmpdir):
    """Split-GPU with default overlap=None → should auto-enable."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _fsdp_split_teacher(teacher, rank, world_size)
    params_before = _snapshot_params(student)

    # overlap_teacher_forward not set → auto-detect
    args = _make_holistic_args(tmpdir, rank)
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()

    assert distiller._overlap_teacher_forward is True, (
        "Auto-detection should enable overlap when teacher/student on different GPUs"
    )
    assert distiller._teacher_stream is not None
    _verify(rank, distiller, student, params_before)


def _auto_detect_same_device_worker(rank, world_size, tmpdir):
    """Replicated teacher on same GPU → should NOT auto-enable."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    # Keep teacher on same GPU as student (replicated)
    teacher.to(f"cuda:{rank}")
    params_before = _snapshot_params(student)

    args = _make_holistic_args(tmpdir, rank)
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=_dataset(),
    )
    distiller.train()

    assert distiller._overlap_teacher_forward is False, (
        "Auto-detection should NOT enable overlap when teacher/student on same GPU"
    )
    assert distiller._teacher_stream is None
    _verify(rank, distiller, student, params_before)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverlapOn:
    """Training correctness with overlap explicitly enabled (split-GPU)."""

    def test_holistic(self, tmp_path):
        _spawn(_overlap_on_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_overlap_on_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_overlap_on_reskd_worker, tmp_path)


class TestOverlapOff:
    """Explicit overlap=False overrides auto-detection (split-GPU)."""

    def test_holistic(self, tmp_path):
        _spawn(_overlap_off_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_overlap_off_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_overlap_off_reskd_worker, tmp_path)


class TestAutoDetection:
    """Overlap auto-detection based on device placement."""

    def test_split_gpu_enables_overlap(self, tmp_path):
        """Different GPUs → auto-enable."""
        _spawn(_auto_detect_split_worker, tmp_path)

    def test_same_device_no_overlap(self, tmp_path):
        """Same GPU → no overlap."""
        _spawn(_auto_detect_same_device_worker, tmp_path)


# ---------------------------------------------------------------------------
# Workers: numerical equivalence (overlap vs no-overlap)
# ---------------------------------------------------------------------------


def _equivalence_hol_worker(rank, world_size, tmpdir):
    """Train holistic with and without overlap, verify similar final loss."""
    import os

    from silverspoon_kd import HolisticDistiller
    from tests.silverspoon_kd.conftest import DummyDataset

    # Train with overlap=False
    torch.manual_seed(42 + rank)
    teacher1, student1 = _make_teacher_student(rank)
    alignments1 = _make_alignments(teacher1, student1)
    teacher1 = _fsdp_split_teacher(teacher1, rank, world_size)

    args1 = _make_holistic_args(
        os.path.join(tmpdir, "no_overlap"),
        rank,
        overlap_teacher_forward=False,
    )
    distiller1 = HolisticDistiller(
        student_model=student1,
        teacher_model=teacher1,
        alignments=alignments1,
        args=args1,
        train_dataset=DummyDataset(num_samples=40, seq_len=16, seed=99),
    )
    distiller1.train()
    losses_no_overlap = [e["loss"] for e in distiller1.state.log_history if "loss" in e]

    torch.cuda.set_device(rank)

    # Train with overlap=True
    torch.manual_seed(42 + rank)
    teacher2, student2 = _make_teacher_student(rank)
    alignments2 = _make_alignments(teacher2, student2)
    teacher2 = _fsdp_split_teacher(teacher2, rank, world_size)

    args2 = _make_holistic_args(
        os.path.join(tmpdir, "overlap"),
        rank,
        overlap_teacher_forward=True,
    )
    distiller2 = HolisticDistiller(
        student_model=student2,
        teacher_model=teacher2,
        alignments=alignments2,
        args=args2,
        train_dataset=DummyDataset(num_samples=40, seq_len=16, seed=99),
    )
    distiller2.train()
    losses_overlap = [e["loss"] for e in distiller2.state.log_history if "loss" in e]

    # Compare: final losses should be in the same ballpark
    assert len(losses_no_overlap) > 0 and len(losses_overlap) > 0
    final_no = losses_no_overlap[-1]
    final_yes = losses_overlap[-1]
    ratio = final_yes / final_no if final_no > 0 else float("inf")
    assert 0.3 < ratio < 3.0, (
        f"Rank {rank}: overlap vs no-overlap final loss diverged. "
        f"no_overlap={final_no:.6f}, overlap={final_yes:.6f}, ratio={ratio:.2f}"
    )


def _equivalence_bkd_worker(rank, world_size, tmpdir):
    """Train blockwise with and without overlap, verify similar final loss."""
    import os

    from silverspoon_kd import BlockwiseDistiller
    from tests.silverspoon_kd.conftest import DummyDataset

    # Train with overlap=False
    torch.manual_seed(42 + rank)
    teacher1, student1 = _make_teacher_student(rank)
    alignments1 = _make_alignments(teacher1, student1)
    teacher1 = _fsdp_split_teacher(teacher1, rank, world_size)

    args1 = _make_blockwise_args(
        os.path.join(tmpdir, "no_overlap"),
        rank,
        overlap_teacher_forward=False,
    )
    distiller1 = BlockwiseDistiller(
        teacher_model=teacher1,
        alignments=alignments1,
        args=args1,
        train_dataset=DummyDataset(num_samples=40, seq_len=16, seed=99),
    )
    distiller1.train()
    losses_no_overlap = [e["loss"] for e in distiller1.state.log_history if "loss" in e]

    torch.cuda.set_device(rank)

    # Train with overlap=True
    torch.manual_seed(42 + rank)
    teacher2, student2 = _make_teacher_student(rank)
    alignments2 = _make_alignments(teacher2, student2)
    teacher2 = _fsdp_split_teacher(teacher2, rank, world_size)

    args2 = _make_blockwise_args(
        os.path.join(tmpdir, "overlap"),
        rank,
        overlap_teacher_forward=True,
    )
    distiller2 = BlockwiseDistiller(
        teacher_model=teacher2,
        alignments=alignments2,
        args=args2,
        train_dataset=DummyDataset(num_samples=40, seq_len=16, seed=99),
    )
    distiller2.train()
    losses_overlap = [e["loss"] for e in distiller2.state.log_history if "loss" in e]

    assert len(losses_no_overlap) > 0 and len(losses_overlap) > 0
    final_no = losses_no_overlap[-1]
    final_yes = losses_overlap[-1]
    ratio = final_yes / final_no if final_no > 0 else float("inf")
    assert 0.3 < ratio < 3.0, (
        f"Rank {rank}: BKD overlap vs no-overlap final loss diverged. "
        f"no_overlap={final_no:.6f}, overlap={final_yes:.6f}, ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Tests: numerical equivalence
# ---------------------------------------------------------------------------


class TestOverlapNumericalEquivalence:
    """Verify overlap produces numerically similar results to non-overlap."""

    def test_holistic(self, tmp_path):
        """HKD: overlap and non-overlap produce similar final losses."""
        _spawn(_equivalence_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        """BKD: overlap and non-overlap produce similar final losses."""
        _spawn(_equivalence_bkd_worker, tmp_path)
