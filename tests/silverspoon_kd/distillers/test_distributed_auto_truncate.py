"""Auto-truncate with distributed training tests."""

import torch

from tests.silverspoon_kd.distillers._distributed_helpers import (
    _dataset,
    _make_alignments,
    _make_blockwise_args,
    _make_fsdp_teacher,
    _make_holistic_args,
    _make_pp_teacher,
    _make_teacher_student,
    _requires_4_gpus,
    _snapshot_params,
    _spawn,
    _verify_training,
)

# =========================================================================
# Auto-truncate with distributed training
# =========================================================================
# These tests verify that auto_truncate=True works with all teacher
# placement strategies and student wrapping. Each runs in a clean
# subprocess via mp.spawn, so there are no dynamo state leak issues.
#
# Coverage matrix:
#   Teacher placement         Student    HKD  BKD
#   ---------------------     -------    ---  ---
#   Replicated                DDP         v    v
#   FSDP split (GPUs 2,3)    DDP         v    v
#   FSDP all-ranks            DDP         v    v
#   PP (GPUs 2,3)             DDP         v    v
#   Replicated + compile      DDP         v    -  (BKD disables compile)
#   Replicated                FSDP        v    -  (student auto-disabled)
# =========================================================================


def _at_rep_ddp_hol_worker(rank, world_size, tmpdir):
    """auto_truncate + Replicated teacher + DDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        check_cross_rank=True,
        world_size=world_size,
    )


def _at_rep_ddp_bkd_worker(rank, world_size, tmpdir):
    """auto_truncate + Replicated teacher + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        check_cross_rank=True,
        world_size=world_size,
    )


def _at_fsdp_split_ddp_hol_worker(rank, world_size, tmpdir):
    """auto_truncate + Split-GPU FSDP teacher + DDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        expected_teacher_device=torch.device(f"cuda:{rank + 2}"),
        check_cross_rank=True,
        world_size=world_size,
    )


def _at_fsdp_split_ddp_bkd_worker(rank, world_size, tmpdir):
    """auto_truncate + Split-GPU FSDP teacher + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        expected_teacher_device=torch.device(f"cuda:{rank + 2}"),
    )


def _at_fsdp_all_ddp_hol_worker(rank, world_size, tmpdir):
    """auto_truncate + All-ranks FSDP teacher + DDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        check_cross_rank=True,
        world_size=world_size,
    )


def _at_fsdp_all_ddp_bkd_worker(rank, world_size, tmpdir):
    """auto_truncate + All-ranks FSDP teacher + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        check_cross_rank=True,
        world_size=world_size,
    )


def _at_pp_ddp_hol_worker(rank, world_size, tmpdir):
    """auto_truncate + PP teacher + DDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _at_pp_ddp_bkd_worker(rank, world_size, tmpdir):
    """auto_truncate + PP teacher + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _at_compile_ddp_hol_worker(rank, world_size, tmpdir):
    """auto_truncate + torch.compile + DDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, torch_compile=True),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _at_rep_fsdp_hol_worker(rank, world_size, tmpdir):
    """auto_truncate + Replicated teacher + FSDP student + Holistic.

    FSDP student should auto-disable student auto_truncate (backward
    state machine incompatibility) while keeping teacher auto_truncate
    enabled. Training should still work — only the student runs full
    forward, the teacher still truncates.
    """
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
        auto_truncate=True,
    )
    # Verify auto-disable: student should be False, teacher should be True
    assert distiller.student_capture.auto_truncate is False, (
        "FSDP student should auto-disable auto_truncate"
    )
    assert distiller.teacher_capture.auto_truncate is True, (
        "Teacher (forward-only) should keep auto_truncate"
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        check_cross_rank=True,
        world_size=world_size,
    )


class TestAutoTruncateDistributed:
    """auto_truncate=True with all teacher placement strategies."""

    pytestmark = _requires_4_gpus

    # Replicated teacher + DDP student
    def test_holistic_replicated(self, tmp_path):
        _spawn(_at_rep_ddp_hol_worker, tmp_path)

    def test_blockwise_replicated(self, tmp_path):
        _spawn(_at_rep_ddp_bkd_worker, tmp_path)

    # FSDP split teacher (GPUs 2,3) + DDP student
    def test_holistic_fsdp_split_teacher(self, tmp_path):
        _spawn(_at_fsdp_split_ddp_hol_worker, tmp_path)

    def test_blockwise_fsdp_split_teacher(self, tmp_path):
        _spawn(_at_fsdp_split_ddp_bkd_worker, tmp_path)

    # FSDP all-ranks teacher + DDP student
    def test_holistic_fsdp_all_teacher(self, tmp_path):
        _spawn(_at_fsdp_all_ddp_hol_worker, tmp_path)

    def test_blockwise_fsdp_all_teacher(self, tmp_path):
        _spawn(_at_fsdp_all_ddp_bkd_worker, tmp_path)

    # PP teacher + DDP student
    def test_holistic_pp_teacher(self, tmp_path):
        _spawn(_at_pp_ddp_hol_worker, tmp_path)

    def test_blockwise_pp_teacher(self, tmp_path):
        _spawn(_at_pp_ddp_bkd_worker, tmp_path)

    # torch.compile + auto_truncate
    def test_holistic_torch_compile(self, tmp_path):
        _spawn(_at_compile_ddp_hol_worker, tmp_path)

    # FSDP student (auto-disable verification)
    def test_holistic_fsdp_student_auto_disable(self, tmp_path):
        _spawn(_at_rep_fsdp_hol_worker, tmp_path)
