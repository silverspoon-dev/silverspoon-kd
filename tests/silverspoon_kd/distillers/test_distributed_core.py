"""Core teacher placement x student wrapper matrix tests."""

import torch

from tests.silverspoon_kd.distillers._distributed_helpers import (
    _dataset,
    _make_alignments,
    _make_blockwise_args,
    _make_explicit_projector_alignments,
    _make_fsdp_teacher,
    _make_holistic_args,
    _make_pp_teacher,
    _make_response_args,
    _make_teacher_student,
    _make_tp_teacher,
    _requires_4_gpus,
    _snapshot_params,
    _spawn,
    _verify_training,
)

# =========================================================================
# Teacher Replicated + Student DDP
# =========================================================================


def _rep_ddp_hol_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + Holistic distiller."""
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


def _rep_ddp_bkd_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + Blockwise distiller."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
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


def _rep_ddp_rbd_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + ResponseBased distiller."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank),
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


# =========================================================================
# Teacher All-Ranks FSDP + Student DDP
# =========================================================================


def _fsdp_all_ddp_hol_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + DDP student + Holistic. Cross-rank + frozen."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

    # Teacher frozen: verify via summon_full_params
    with FSDP.summon_full_params(distiller.teacher_model):
        for n, p in distiller.teacher_model.named_parameters():
            assert not p.requires_grad, f"Rank {rank}: teacher param '{n}' has requires_grad=True"


def _fsdp_all_ddp_bkd_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + DDP student + Blockwise."""
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


def _fsdp_all_ddp_rbd_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + DDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank),
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


# =========================================================================
# Teacher Split-GPU FSDP (GPUs 2,3) + Student DDP (GPUs 0,1)
# =========================================================================


def _fsdp_split_ddp_hol_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DDP student + Holistic. Cross-rank + device."""
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


def _fsdp_split_ddp_bkd_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DDP student + Blockwise."""
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
    )
    distiller.train()
    _verify_training(
        rank,
        distiller,
        student,
        params_before,
        expected_teacher_device=torch.device(f"cuda:{rank + 2}"),
    )


def _fsdp_split_ddp_rbd_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank),
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


# =========================================================================
# Split-GPU FSDP: Capture engine + Projector device + Varied alignments
# =========================================================================


def _capture_verification_worker(rank, world_size, tmpdir):
    """Verify capture hooks fire through FSDP — all modules captured."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    num_alignments = len(alignments)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, max_steps=1),
        train_dataset=_dataset(10),
    )

    distiller._setup_teacher_placement()
    distiller._register_capture()

    batch = {
        "input_ids": torch.randint(0, 128, (2, 16), device=f"cuda:{rank}"),
        "attention_mask": torch.ones(2, 16, dtype=torch.long, device=f"cuda:{rank}"),
    }
    batch = distiller._prepare_inputs(batch)

    # Teacher forward
    teacher_inputs = distiller._prepare_teacher_inputs(batch)
    with torch.no_grad():
        distiller.teacher_model(**teacher_inputs)

    tc = distiller.teacher_capture
    assert len(tc.captured_outputs) == num_alignments, (
        f"Rank {rank}: teacher captured {len(tc.captured_outputs)}/{num_alignments} "
        f"modules. Hooks may not fire through FSDP."
    )
    for mid, out in tc.captured_outputs.items():
        assert isinstance(out, torch.Tensor), (
            f"Rank {rank}: module {mid} output is {type(out)}, expected Tensor"
        )
        assert out.shape[0] == 2, f"Rank {rank}: module {mid} batch dim={out.shape[0]}, expected 2"
        assert torch.isfinite(out).all(), f"Rank {rank}: module {mid} has non-finite values"

    # Student forward
    student_inputs = distiller._prepare_student_inputs(batch)
    distiller.student_model(**student_inputs)

    sc = distiller.student_capture
    assert len(sc.captured_outputs) == num_alignments, (
        f"Rank {rank}: student captured {len(sc.captured_outputs)}/{num_alignments}"
    )

    distiller._deregister_capture()


def _projector_device_worker(rank, world_size, tmpdir):
    """Auto-projectors must land on student device, not teacher device."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, max_steps=2),
        train_dataset=_dataset(10),
    )
    distiller.train()

    expected = torch.device(f"cuda:{rank}")
    for a in alignments:
        if a.output_projector is not None:
            for name, param in a.output_projector.named_parameters():
                assert param.device == expected, (
                    f"Rank {rank}: projector param '{name}' on {param.device}, expected {expected}"
                )


def _varied_alignment_worker(rank, world_size, tmpdir):
    """Single-alignment (layer 1 only) works with FSDP teacher."""
    from silverspoon_kd import HolisticDistiller, create_alignments

    teacher, student = _make_teacher_student(rank)
    alignments = create_alignments(
        teacher_model=teacher,
        student_model=student,
        modules=r"layers\.1$",
        output_selector_index=None,
        auto_projector=True,
    )
    for a in alignments:
        a.auto_device_match = True

    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Student FSDP (via Trainer wrapping)
# =========================================================================


def _rep_fsdp_hol_worker(rank, world_size, tmpdir):
    """Replicated teacher + FSDP student + Holistic.

    CompositeOptimizer creation is deferred until create_optimizer(),
    which runs after FSDP wraps the student model.
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


def _rep_fsdp_bkd_worker(rank, world_size, tmpdir):
    """Replicated teacher + FSDP student + Blockwise.

    Blockwise calls student blocks directly — summon_full_params
    materialises sharded FSDP params for the duration of the forward.
    """
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fsdp="full_shard"),
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


def _rep_fsdp_rbd_worker(rank, world_size, tmpdir):
    """Replicated teacher + FSDP student + ResponseBased.

    ResponseBased uses standard Trainer optimizer (not CompositeOptimizer),
    so student FSDP wrapping should be compatible.
    """
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, fsdp="full_shard"),
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


def _fsdp_all_fsdp_bkd_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + FSDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fsdp_all_fsdp_hol_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + FSDP student + Holistic."""
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
        args=_make_holistic_args(tmpdir, rank, fsdp="full_shard"),
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


def _fsdp_all_fsdp_rbd_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + FSDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fsdp_split_fsdp_hol_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + FSDP student + Holistic.

    CompositeOptimizer creation deferred until after FSDP wraps the student.
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
        args=_make_holistic_args(tmpdir, rank, fsdp="full_shard"),
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


def _fsdp_split_fsdp_bkd_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + FSDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fsdp="full_shard"),
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


def _fsdp_split_fsdp_rbd_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + FSDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, fsdp="full_shard"),
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


# =========================================================================
# _setup_teacher_placement dispatch via training args
# =========================================================================


def _dispatch_sharded_hol_worker(rank, world_size, tmpdir):
    """teacher_placement='sharded' -> FSDP wrapping by distiller dispatch."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, teacher_placement="sharded"),
        train_dataset=_dataset(),
    )
    distiller.train()

    assert isinstance(distiller.teacher_model, FSDP), (
        f"Rank {rank}: teacher should be FSDP after dispatch, got {type(distiller.teacher_model)}"
    )
    _verify_training(rank, distiller, student, params_before)


def _dispatch_sharded_bkd_worker(rank, world_size, tmpdir):
    """teacher_placement='sharded' dispatch + Blockwise."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, teacher_placement="sharded"),
        train_dataset=_dataset(),
    )
    distiller.train()

    assert isinstance(distiller.teacher_model, FSDP), (
        f"Rank {rank}: teacher should be FSDP after dispatch"
    )
    _verify_training(rank, distiller, student, params_before)


def _dispatch_sharded_rbd_worker(rank, world_size, tmpdir):
    """teacher_placement='sharded' dispatch + ResponseBased."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, teacher_placement="sharded"),
        train_dataset=_dataset(),
    )
    distiller.train()

    assert isinstance(distiller.teacher_model, FSDP), (
        f"Rank {rank}: teacher should be FSDP after dispatch"
    )
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Dispatch + Student FSDP
# =========================================================================


def _dispatch_fsdp_hol_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + FSDP student + Holistic."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(
            tmpdir,
            rank,
            teacher_placement="sharded",
            fsdp="full_shard",
        ),
        train_dataset=_dataset(),
    )
    distiller.train()

    assert isinstance(distiller.teacher_model, FSDP)
    _verify_training(rank, distiller, student, params_before)


def _dispatch_fsdp_bkd_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + FSDP student + Blockwise."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(
            tmpdir,
            rank,
            teacher_placement="sharded",
            fsdp="full_shard",
        ),
        train_dataset=_dataset(),
    )
    distiller.train()

    assert isinstance(distiller.teacher_model, FSDP)
    _verify_training(rank, distiller, student, params_before)


def _dispatch_fsdp_rbd_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + FSDP student + ResponseBased."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(
            tmpdir,
            rank,
            teacher_placement="sharded",
            fsdp="full_shard",
        ),
        train_dataset=_dataset(),
    )
    distiller.train()

    assert isinstance(distiller.teacher_model, FSDP)
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Teacher PP (device_map) + Student DDP
# =========================================================================


def _pp_ddp_hol_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student + Holistic."""
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
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _pp_ddp_bkd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student + Blockwise."""
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
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _pp_ddp_rbd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Teacher PP + Student FSDP
# =========================================================================


def _pp_fsdp_hol_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + FSDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _pp_fsdp_bkd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + FSDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _pp_fsdp_rbd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + FSDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Teacher TP + Student DDP
# =========================================================================


def _tp_ddp_hol_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller
    from tests.silverspoon_kd.conftest import SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2)
    student.to(f"cuda:{rank}")

    alignments = _make_alignments(teacher, student)
    teacher = _make_tp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _tp_ddp_bkd_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller
    from tests.silverspoon_kd.conftest import SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2)
    student.to(f"cuda:{rank}")

    alignments = _make_alignments(teacher, student)
    teacher = _make_tp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _tp_ddp_rbd_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller
    from tests.silverspoon_kd.conftest import SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2)
    student.to(f"cuda:{rank}")

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    # TP the teacher after distiller init (ResponseBased doesn't use alignments)
    teacher_tp = _make_tp_teacher(rank, world_size, teacher)
    distiller.teacher_model = teacher_tp

    params_before = _snapshot_params(student)
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Teacher TP + Student FSDP
# =========================================================================


def _tp_fsdp_hol_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + FSDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller
    from tests.silverspoon_kd.conftest import SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2)
    student.to(f"cuda:{rank}")

    alignments = _make_alignments(teacher, student)
    teacher = _make_tp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _tp_fsdp_bkd_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + FSDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller
    from tests.silverspoon_kd.conftest import SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2)
    student.to(f"cuda:{rank}")

    alignments = _make_alignments(teacher, student)
    teacher = _make_tp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _tp_fsdp_rbd_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + FSDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller
    from tests.silverspoon_kd.conftest import SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2)
    student.to(f"cuda:{rank}")

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    teacher_tp = _make_tp_teacher(rank, world_size, teacher)
    distiller.teacher_model = teacher_tp

    params_before = _snapshot_params(student)
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Explicit projector tests (non-auto projectors in distributed context)
# =========================================================================


def _explicit_proj_ddp_bkd_worker(rank, world_size, tmpdir):
    """Explicit input+output projectors + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_explicit_projector_alignments(teacher, student)
    params_before = _snapshot_params(student)

    # Snapshot projector params
    proj_params_before = {}
    for a in alignments:
        for proj_name, proj in [
            ("input", a.input_projector),
            ("output", a.output_projector),
        ]:
            if proj is not None:
                for n, p in proj.named_parameters():
                    key = f"{a.get_name()}.{proj_name}.{n}"
                    proj_params_before[key] = p.clone().detach()

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)

    # Verify projector params changed
    proj_changed = False
    for a in alignments:
        for proj_name, proj in [
            ("input", a.input_projector),
            ("output", a.output_projector),
        ]:
            if proj is not None:
                for n, p in proj.named_parameters():
                    key = f"{a.get_name()}.{proj_name}.{n}"
                    if key in proj_params_before and not torch.equal(p, proj_params_before[key]):
                        proj_changed = True
    assert proj_changed, f"Rank {rank}: projector params unchanged after training"


def _explicit_proj_ddp_hol_worker(rank, world_size, tmpdir):
    """Explicit output projectors + DDP student + Holistic."""
    import torch.nn as nn

    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.alignments import Alignment

    teacher, student = _make_teacher_student(rank)
    teacher_dim = teacher.hidden_dim
    student_device = f"cuda:{rank}"

    # Create alignments with explicit output projectors only (holistic ignores input projectors)
    alignments = []
    for i in range(min(teacher.num_layers, student.num_layers)):
        teacher_block = teacher.get_layer(i)
        student_block = student.get_layer(i)
        student_dim = None
        for mod in student_block.modules():
            if isinstance(mod, nn.Linear):
                student_dim = mod.out_features
                break

        output_projector = None
        if student_dim is not None and student_dim != teacher_dim:
            output_projector = nn.Linear(student_dim, teacher_dim).to(student_device)

        alignments.append(
            Alignment(
                teacher_block=teacher_block,
                student_block=student_block,
                teacher_model_name="test_teacher",
                student_model_name="test_student",
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                output_projector=output_projector,
                auto_device_match=True,
            )
        )

    params_before = _snapshot_params(student)

    # Snapshot output projector params
    proj_params_before = {}
    for a in alignments:
        if a.output_projector is not None:
            for n, p in a.output_projector.named_parameters():
                key = f"{a.get_name()}.output.{n}"
                proj_params_before[key] = p.clone().detach()

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)

    # Verify projector params changed
    proj_changed = False
    for a in alignments:
        if a.output_projector is not None:
            for n, p in a.output_projector.named_parameters():
                key = f"{a.get_name()}.output.{n}"
                if key in proj_params_before and not torch.equal(p, proj_params_before[key]):
                    proj_changed = True
    assert proj_changed, f"Rank {rank}: output projector params unchanged"


def _explicit_proj_fsdp_bkd_worker(rank, world_size, tmpdir):
    """Explicit input+output projectors + FSDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_explicit_projector_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _explicit_proj_fsdp_hol_worker(rank, world_size, tmpdir):
    """Explicit output projectors + FSDP student + Holistic."""
    import torch.nn as nn

    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.alignments import Alignment

    teacher, student = _make_teacher_student(rank)
    teacher_dim = teacher.hidden_dim
    student_device = f"cuda:{rank}"

    alignments = []
    for i in range(min(teacher.num_layers, student.num_layers)):
        teacher_block = teacher.get_layer(i)
        student_block = student.get_layer(i)
        student_dim = None
        for mod in student_block.modules():
            if isinstance(mod, nn.Linear):
                student_dim = mod.out_features
                break

        output_projector = None
        if student_dim is not None and student_dim != teacher_dim:
            output_projector = nn.Linear(student_dim, teacher_dim).to(student_device)

        alignments.append(
            Alignment(
                teacher_block=teacher_block,
                student_block=student_block,
                teacher_model_name="test_teacher",
                student_model_name="test_student",
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                output_projector=output_projector,
                auto_device_match=True,
            )
        )

    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Explicit projectors + PP teacher
# =========================================================================


def _explicit_proj_pp_bkd_worker(rank, world_size, tmpdir):
    """Explicit input+output projectors + PP teacher + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_explicit_projector_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    # Snapshot projector params
    proj_params_before = {}
    for a in alignments:
        for proj_name, proj in [
            ("input", a.input_projector),
            ("output", a.output_projector),
        ]:
            if proj is not None:
                for n, p in proj.named_parameters():
                    key = f"{a.get_name()}.{proj_name}.{n}"
                    proj_params_before[key] = p.clone().detach()

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)

    # Verify projector params changed
    proj_changed = False
    for a in alignments:
        for proj_name, proj in [
            ("input", a.input_projector),
            ("output", a.output_projector),
        ]:
            if proj is not None:
                for n, p in proj.named_parameters():
                    key = f"{a.get_name()}.{proj_name}.{n}"
                    if key in proj_params_before and not torch.equal(p, proj_params_before[key]):
                        proj_changed = True
    assert proj_changed, f"Rank {rank}: projector params unchanged after training"


def _explicit_proj_pp_hol_worker(rank, world_size, tmpdir):
    """Explicit output projectors + PP teacher + Holistic."""
    import torch.nn as nn

    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.alignments import Alignment

    teacher, student = _make_teacher_student(rank)
    teacher_dim = teacher.hidden_dim
    student_device = f"cuda:{rank}"

    # Create alignments with explicit output projectors only
    alignments = []
    for i in range(min(teacher.num_layers, student.num_layers)):
        teacher_block = teacher.get_layer(i)
        student_block = student.get_layer(i)
        student_dim = None
        for mod in student_block.modules():
            if isinstance(mod, nn.Linear):
                student_dim = mod.out_features
                break

        output_projector = None
        if student_dim is not None and student_dim != teacher_dim:
            output_projector = nn.Linear(student_dim, teacher_dim).to(student_device)

        alignments.append(
            Alignment(
                teacher_block=teacher_block,
                student_block=student_block,
                teacher_model_name="test_teacher",
                student_model_name="test_student",
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                output_projector=output_projector,
                auto_device_match=True,
            )
        )

    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    # Snapshot output projector params
    proj_params_before = {}
    for a in alignments:
        if a.output_projector is not None:
            for n, p in a.output_projector.named_parameters():
                key = f"{a.get_name()}.output.{n}"
                proj_params_before[key] = p.clone().detach()

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)

    # Verify projector params changed
    proj_changed = False
    for a in alignments:
        if a.output_projector is not None:
            for n, p in a.output_projector.named_parameters():
                key = f"{a.get_name()}.output.{n}"
                if key in proj_params_before and not torch.equal(p, proj_params_before[key]):
                    proj_changed = True
    assert proj_changed, f"Rank {rank}: output projector params unchanged"


# =========================================================================
# Explicit projectors + FSDP-all teacher
# =========================================================================


def _explicit_proj_fsdp_all_bkd_worker(rank, world_size, tmpdir):
    """Explicit input+output projectors + FSDP-all teacher + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    alignments = _make_explicit_projector_alignments(teacher, student)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    # Snapshot projector params
    proj_params_before = {}
    for a in alignments:
        for proj_name, proj in [
            ("input", a.input_projector),
            ("output", a.output_projector),
        ]:
            if proj is not None:
                for n, p in proj.named_parameters():
                    key = f"{a.get_name()}.{proj_name}.{n}"
                    proj_params_before[key] = p.clone().detach()

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)

    # Verify projector params changed
    proj_changed = False
    for a in alignments:
        for proj_name, proj in [
            ("input", a.input_projector),
            ("output", a.output_projector),
        ]:
            if proj is not None:
                for n, p in proj.named_parameters():
                    key = f"{a.get_name()}.{proj_name}.{n}"
                    if key in proj_params_before and not torch.equal(p, proj_params_before[key]):
                        proj_changed = True
    assert proj_changed, f"Rank {rank}: projector params unchanged after training"


def _explicit_proj_fsdp_all_hol_worker(rank, world_size, tmpdir):
    """Explicit output projectors + FSDP-all teacher + Holistic."""
    import torch.nn as nn

    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.alignments import Alignment
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    teacher_dim = teacher.hidden_dim
    student_device = f"cuda:{rank}"

    # Create alignments with explicit output projectors only
    alignments = []
    for i in range(min(teacher.num_layers, student.num_layers)):
        teacher_block = teacher.get_layer(i)
        student_block = student.get_layer(i)
        student_dim = None
        for mod in student_block.modules():
            if isinstance(mod, nn.Linear):
                student_dim = mod.out_features
                break

        output_projector = None
        if student_dim is not None and student_dim != teacher_dim:
            output_projector = nn.Linear(student_dim, teacher_dim).to(student_device)

        alignments.append(
            Alignment(
                teacher_block=teacher_block,
                student_block=student_block,
                teacher_model_name="test_teacher",
                student_model_name="test_student",
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                output_projector=output_projector,
                auto_device_match=True,
            )
        )

    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    # Snapshot output projector params
    proj_params_before = {}
    for a in alignments:
        if a.output_projector is not None:
            for n, p in a.output_projector.named_parameters():
                key = f"{a.get_name()}.output.{n}"
                proj_params_before[key] = p.clone().detach()

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)

    # Verify projector params changed
    proj_changed = False
    for a in alignments:
        if a.output_projector is not None:
            for n, p in a.output_projector.named_parameters():
                key = f"{a.get_name()}.output.{n}"
                if key in proj_params_before and not torch.equal(p, proj_params_before[key]):
                    proj_changed = True
    assert proj_changed, f"Rank {rank}: output projector params unchanged"


# =========================================================================
# Test classes
# =========================================================================


class TestReplicatedTeacherDDPStudent:
    """Teacher replicated on each rank's GPU, student DDP-wrapped by Trainer."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_rep_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_rep_ddp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_rep_ddp_rbd_worker, tmp_path)


class TestAllRanksFSDPTeacherDDPStudent:
    """Teacher FSDP-sharded across all ranks (same GPUs), student DDP."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        """Includes cross-rank consistency + teacher frozen verification."""
        _spawn(_fsdp_all_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_fsdp_all_ddp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_fsdp_all_ddp_rbd_worker, tmp_path)


class TestSplitFSDPTeacherDDPStudent:
    """Teacher FSDP on GPUs 2,3; student DDP on GPUs 0,1."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        """Includes cross-rank + teacher device verification."""
        _spawn(_fsdp_split_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_fsdp_split_ddp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_fsdp_split_ddp_rbd_worker, tmp_path)

    def test_capture_engine_through_fsdp(self, tmp_path):
        """All capture hooks fire through FSDP wrapping."""
        _spawn(_capture_verification_worker, tmp_path)

    def test_projector_device_placement(self, tmp_path):
        """Auto-projectors created on student device."""
        _spawn(_projector_device_worker, tmp_path)

    def test_single_alignment(self, tmp_path):
        """Works with fewer alignments than layers."""
        _spawn(_varied_alignment_worker, tmp_path)


class TestReplicatedTeacherFSDPStudent:
    """Replicated teacher, student FSDP-wrapped by Trainer."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        """CompositeOptimizer deferred until after FSDP wraps student."""
        _spawn(_rep_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        """Per-block FSDP wrapping: each block is its own FSDP unit."""
        _spawn(_rep_fsdp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_rep_fsdp_rbd_worker, tmp_path)


class TestAllRanksFSDPTeacherFSDPStudent:
    """Teacher FSDP all-ranks + student FSDP. Both models FSDP-sharded."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_fsdp_all_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_fsdp_all_fsdp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_fsdp_all_fsdp_rbd_worker, tmp_path)


class TestSplitFSDPTeacherFSDPStudent:
    """Teacher FSDP on GPUs 2,3; student FSDP on GPUs 0,1."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        """CompositeOptimizer deferred until after FSDP wraps student."""
        _spawn(_fsdp_split_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_fsdp_split_fsdp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_fsdp_split_fsdp_rbd_worker, tmp_path)


class TestPPTeacherDDPStudent:
    """PP teacher on GPUs 2,3; student DDP on GPUs 0,1."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_pp_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_pp_ddp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_pp_ddp_rbd_worker, tmp_path)


class TestPPTeacherFSDPStudent:
    """PP teacher on GPUs 2,3; student FSDP on GPUs 0,1."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_pp_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_pp_fsdp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_pp_fsdp_rbd_worker, tmp_path)


class TestTPTeacherDDPStudent:
    """TP teacher on GPUs 2,3; student DDP on GPUs 0,1."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_tp_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_tp_ddp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_tp_ddp_rbd_worker, tmp_path)


class TestTPTeacherFSDPStudent:
    """TP teacher on GPUs 2,3; student FSDP on GPUs 0,1."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_tp_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_tp_fsdp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_tp_fsdp_rbd_worker, tmp_path)


class TestTeacherPlacementDispatch:
    """teacher_placement='sharded' triggers FSDP wrapping via distiller dispatch."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_dispatch_sharded_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_dispatch_sharded_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_dispatch_sharded_rbd_worker, tmp_path)


class TestDispatchFSDPStudent:
    """Dispatch sharded teacher + student FSDP. Both models FSDP-sharded."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_dispatch_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_dispatch_fsdp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_dispatch_fsdp_rbd_worker, tmp_path)


class TestExplicitProjectorsDDP:
    """Explicit (non-auto) projectors with DDP student."""

    pytestmark = _requires_4_gpus

    def test_blockwise_input_output_projectors(self, tmp_path):
        """Blockwise with explicit input+output projectors under DDP."""
        _spawn(_explicit_proj_ddp_bkd_worker, tmp_path)

    def test_holistic_output_projector(self, tmp_path):
        """Holistic with explicit output projector under DDP."""
        _spawn(_explicit_proj_ddp_hol_worker, tmp_path)


class TestExplicitProjectorsFSDP:
    """Explicit (non-auto) projectors with FSDP student."""

    pytestmark = _requires_4_gpus

    def test_blockwise_input_output_projectors(self, tmp_path):
        """Blockwise with explicit input+output projectors under FSDP."""
        _spawn(_explicit_proj_fsdp_bkd_worker, tmp_path)

    def test_holistic_output_projector(self, tmp_path):
        """Holistic with explicit output projector under FSDP."""
        _spawn(_explicit_proj_fsdp_hol_worker, tmp_path)


class TestExplicitProjectorsPPTeacher:
    """Explicit projectors with PP teacher on GPUs 2,3."""

    pytestmark = _requires_4_gpus

    def test_blockwise_input_output_projectors(self, tmp_path):
        """Blockwise with explicit input+output projectors + PP teacher."""
        _spawn(_explicit_proj_pp_bkd_worker, tmp_path)

    def test_holistic_output_projector(self, tmp_path):
        """Holistic with explicit output projector + PP teacher."""
        _spawn(_explicit_proj_pp_hol_worker, tmp_path)


class TestExplicitProjectorsFSDPAllTeacher:
    """Explicit projectors with FSDP-all teacher."""

    pytestmark = _requires_4_gpus

    def test_blockwise_input_output_projectors(self, tmp_path):
        """Blockwise with explicit input+output projectors + FSDP-all teacher."""
        _spawn(_explicit_proj_fsdp_all_bkd_worker, tmp_path)

    def test_holistic_output_projector(self, tmp_path):
        """Holistic with explicit output projector + FSDP-all teacher."""
        _spawn(_explicit_proj_fsdp_all_hol_worker, tmp_path)
