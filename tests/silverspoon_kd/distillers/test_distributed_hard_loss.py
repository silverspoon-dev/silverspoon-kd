"""Distributed tests for HKD hard loss (alpha > 0).

Verifies that the hard loss path works correctly when the teacher and
student are distributed across multiple GPUs via DDP, FSDP, and split-GPU
FSDP placements.  These tests exercise device placement, gradient flow,
and loss computation correctness under configurations where device
mismatches and FSDP state issues are most likely to surface.
"""

import torch

from tests.silverspoon_kd.distillers._distributed_helpers import (
    _dataset,
    _make_alignments,
    _make_fsdp_teacher,
    _make_holistic_args,
    _make_pp_teacher,
    _make_teacher_student,
    _requires_4_gpus,
    _snapshot_params,
    _spawn,
    _verify_training,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verify_hard_loss_logged(rank, distiller, alpha=0.5):
    """Assert that both 'loss/hard' and 'loss/alignment' appear in metrics
    and that their logged values are individually finite and positive.

    This catches three failure modes:
    1. Hard loss code path silently skipped (``loss/hard`` missing).
    2. Alignment loss silently zeroed (``loss/alignment`` missing or zero).
    3. Either component is garbage (non-finite or negative).

    For alpha < 1.0, also verifies ``loss/alignment`` is present.
    """
    all_keys = set()
    hard_values = []
    alignment_values = []
    for entry in distiller.state.log_history:
        all_keys.update(entry.keys())
        if "loss/hard" in entry:
            hard_values.append(entry["loss/hard"])
        if "loss/alignment" in entry:
            alignment_values.append(entry["loss/alignment"])

    # Hard loss must always be logged when alpha > 0
    assert "loss/hard" in all_keys, (
        f"Rank {rank}: 'loss/hard' not found in logged metrics — hard loss "
        f"was not mixed in despite alpha > 0. Logged keys: {sorted(all_keys)}"
    )
    assert all(torch.isfinite(torch.tensor(v)) and v > 0 for v in hard_values), (
        f"Rank {rank}: loss/hard values not all finite and positive: {hard_values[:5]}"
    )

    # Alignment loss must be logged when alpha < 1.0
    if alpha < 1.0:
        assert "loss/alignment" in all_keys, (
            f"Rank {rank}: 'loss/alignment' not found — alignment loss path "
            f"may have silently failed. Logged keys: {sorted(all_keys)}"
        )
        assert all(torch.isfinite(torch.tensor(v)) and v > 0 for v in alignment_values), (
            f"Rank {rank}: loss/alignment values not all finite and positive: {alignment_values[:5]}"
        )


# ---------------------------------------------------------------------------
# Workers — each runs inside mp.spawn with rank, world_size, tmpdir
# ---------------------------------------------------------------------------


def _rep_ddp_hol_alpha_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + HKD with alpha=0.5."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, alpha=0.5),
        train_dataset=_dataset(),
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
    _verify_hard_loss_logged(rank, distiller)


def _fsdp_all_fsdp_hol_alpha_worker(rank, world_size, tmpdir):
    """FSDP teacher (all-ranks) + FSDP student + HKD with alpha=0.5."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, alpha=0.5, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        expected_teacher_device=torch.device(f"cuda:{rank + 2}"),
    )
    _verify_hard_loss_logged(rank, distiller)


def _fsdp_split_fsdp_hol_alpha_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + FSDP student + HKD with alpha=0.5.

    Teacher FSDP-sharded on GPUs 2,3; student FSDP-sharded on GPUs 0,1.
    This is the most demanding placement for hard loss: labels must flow
    correctly to the student on the right device, and the combined
    alignment+hard loss must work with FSDP's pre-backward hooks.
    """
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, alpha=0.5, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        expected_teacher_device=torch.device(f"cuda:{rank + 2}"),
    )
    _verify_hard_loss_logged(rank, distiller)


def _fsdp_split_ddp_hol_alpha_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DDP student + HKD with alpha=0.5.

    Teacher FSDP-sharded on GPUs 2,3; student DDP on GPUs 0,1.
    Mixed FSDP teacher + DDP student configuration.
    """
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, alpha=0.5),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        expected_teacher_device=torch.device(f"cuda:{rank + 2}"),
    )
    _verify_hard_loss_logged(rank, distiller)


def _pp_ddp_hol_alpha_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student (GPUs 0,1) + HKD with alpha=0.5."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, alpha=0.5),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
    )
    _verify_hard_loss_logged(rank, distiller)


def _rep_ddp_hol_alpha_labels_stripped_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + HKD with alpha=0.5.

    Uses prepare_student_inputs that strips labels — verifies the
    label re-injection logic works under DDP.
    """
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    def strip_labels(inputs):
        return {k: v for k, v in inputs.items() if k != "labels"}

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, alpha=0.5),
        train_dataset=_dataset(),
        prepare_student_inputs=strip_labels,
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
    _verify_hard_loss_logged(rank, distiller)


def _rep_ddp_hol_alpha_one_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + HKD with alpha=1.0 (pure hard loss)."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, alpha=1.0),
        train_dataset=_dataset(),
    )
    distiller.train()

    # With alpha=1.0, the loss is 100% hard loss (CE).
    losses = [e["loss"] for e in distiller.state.log_history if "loss" in e]
    assert len(losses) > 0, f"Rank {rank}: no losses logged"
    for v in losses:
        assert torch.isfinite(torch.tensor(v)), f"Rank {rank}: non-finite loss {v}"

    changed = {
        n
        for n, p in student.named_parameters()
        if p.requires_grad and n in params_before and not torch.equal(p.data, params_before[n])
    }
    assert changed, f"Rank {rank}: student params unchanged with alpha=1.0"
    _verify_hard_loss_logged(rank, distiller, alpha=1.0)


def _rep_ddp_hol_alpha_magnitude_aware_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP + HKD with alpha=0.5 + magnitude_aware_weighting.

    Tests the interaction between hard loss mixing and magnitude-aware
    normalization — both paths go through ``_combine_losses`` and could
    interact poorly if magnitude normalization divides by a hard-loss
    magnitude that's on a different scale than alignment losses.
    """
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, alpha=0.5, magnitude_aware_weighting=True),
        train_dataset=_dataset(),
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
    _verify_hard_loss_logged(rank, distiller)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestHKDHardLossReplicatedDDP:
    """HKD hard loss (alpha > 0) with replicated teacher + DDP student."""

    pytestmark = _requires_4_gpus

    def test_alpha_half(self, tmp_path):
        """alpha=0.5: mixed alignment + hard loss under DDP."""
        _spawn(_rep_ddp_hol_alpha_worker, tmp_path)

    def test_alpha_one(self, tmp_path):
        """alpha=1.0: pure hard loss under DDP."""
        _spawn(_rep_ddp_hol_alpha_one_worker, tmp_path)

    def test_labels_stripped(self, tmp_path):
        """alpha=0.5 with prepare_student_inputs that strips labels."""
        _spawn(_rep_ddp_hol_alpha_labels_stripped_worker, tmp_path)

    def test_magnitude_aware(self, tmp_path):
        """alpha=0.5 + magnitude_aware_weighting under DDP."""
        _spawn(_rep_ddp_hol_alpha_magnitude_aware_worker, tmp_path)


class TestHKDHardLossFSDPTeacher:
    """HKD hard loss with FSDP-sharded teacher on split GPUs."""

    pytestmark = _requires_4_gpus

    def test_fsdp_teacher_ddp_student(self, tmp_path):
        """FSDP teacher (GPUs 2,3) + DDP student (GPUs 0,1) + alpha=0.5."""
        _spawn(_fsdp_split_ddp_hol_alpha_worker, tmp_path)

    def test_fsdp_teacher_fsdp_student(self, tmp_path):
        """FSDP teacher (GPUs 2,3) + FSDP student (GPUs 0,1) + alpha=0.5."""
        _spawn(_fsdp_split_fsdp_hol_alpha_worker, tmp_path)

    def test_fsdp_all_ranks(self, tmp_path):
        """FSDP teacher (all-ranks) + FSDP student + alpha=0.5."""
        _spawn(_fsdp_all_fsdp_hol_alpha_worker, tmp_path)


class TestHKDHardLossPPTeacher:
    """HKD hard loss with pipeline-parallel teacher on split GPUs."""

    pytestmark = _requires_4_gpus

    def test_pp_teacher_ddp_student(self, tmp_path):
        """PP teacher (GPUs 2,3) + DDP student (GPUs 0,1) + alpha=0.5."""
        _spawn(_pp_ddp_hol_alpha_worker, tmp_path)
