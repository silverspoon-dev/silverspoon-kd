"""DeepSpeed integration tests for distributed distillation."""

import json
import os

import torch

from tests.silverspoon_kd.distillers._distributed_helpers import (
    _assert_deepspeed_compatible_optimizer,
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
    _requires_deepspeed,
    _snapshot_params,
    _spawn_ds,
    _verify_eval,
    _verify_training,
    _write_ds3_config,
    _write_ds_config,
)

# =========================================================================
# Student DeepSpeed ZeRO-2
# =========================================================================


def _rep_ds_rbd_worker(rank, world_size, tmpdir):
    """Replicated teacher + DeepSpeed ZeRO-2 student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fsdp_split_ds_rbd_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DeepSpeed ZeRO-2 student + ResponseBased.

    Teacher FSDP on GPUs 2,3 (via torch.distributed).
    Student DeepSpeed on GPUs 0,1 (via HF Trainer + accelerate).
    """
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, deepspeed=ds_config),
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


def _rep_ds_hol_worker(rank, world_size, tmpdir):
    """Replicated teacher + DeepSpeed ZeRO-2 student + Holistic.

    HKD intentionally has ``_USE_COMPOSITE_OPTIMIZER = False`` (a previous
    inherited ``True`` caused an embeddings-never-trained bug), so under
    DeepSpeed it goes through HF Trainer's standard optimizer creation,
    not ``_build_deepspeed_compatible_optimizer``.  See BKD's worker for
    the per-alignment-tagged optimizer path.
    """
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _rep_ds_bkd_worker(rank, world_size, tmpdir):
    """Replicated teacher + DeepSpeed ZeRO-2 student + Blockwise.

    Uses _build_deepspeed_compatible_optimizer with per-alignment param_groups.
    """
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    _assert_deepspeed_compatible_optimizer(rank, distiller, alignments)


def _fsdp_all_ds_hol_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + DeepSpeed ZeRO-2 student + Holistic."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fsdp_all_ds_bkd_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + DeepSpeed ZeRO-2 student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fsdp_all_ds_rbd_worker(rank, world_size, tmpdir):
    """All-ranks FSDP teacher + DeepSpeed ZeRO-2 student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fsdp_split_ds_hol_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DeepSpeed ZeRO-2 student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, deepspeed=ds_config),
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


def _fsdp_split_ds_bkd_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DeepSpeed ZeRO-2 student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_fsdp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, deepspeed=ds_config),
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


def _pp_ds_rbd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DeepSpeed ZeRO-2 student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _pp_ds_hol_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DeepSpeed ZeRO-2 student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _pp_ds_bkd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DeepSpeed ZeRO-2 student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Teacher Dispatch "sharded" + Student DeepSpeed
# =========================================================================


def _dispatch_ds_hol_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + DeepSpeed ZeRO-2 student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(
            tmpdir,
            rank,
            teacher_placement="sharded",
            deepspeed=ds_config,
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _dispatch_ds_bkd_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + DeepSpeed ZeRO-2 student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(
            tmpdir,
            rank,
            teacher_placement="sharded",
            deepspeed=ds_config,
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _dispatch_ds_rbd_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + DeepSpeed ZeRO-2 student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(
            tmpdir,
            rank,
            teacher_placement="sharded",
            deepspeed=ds_config,
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Teacher TP + Student DeepSpeed
# =========================================================================


def _tp_ds_hol_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DeepSpeed ZeRO-2 student + Holistic."""
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
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _tp_ds_bkd_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DeepSpeed ZeRO-2 student + Blockwise."""
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
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _tp_ds_rbd_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DeepSpeed ZeRO-2 student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller
    from tests.silverspoon_kd.conftest import SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2)
    student.to(f"cuda:{rank}")

    ds_config = _write_ds_config(tmpdir, rank)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    teacher_tp = _make_tp_teacher(rank, world_size, teacher)
    distiller.teacher_model = teacher_tp

    params_before = _snapshot_params(student)
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Explicit projectors + DeepSpeed
# =========================================================================


def _explicit_proj_ds_bkd_worker(rank, world_size, tmpdir):
    """Explicit input+output projectors + DeepSpeed ZeRO-2 student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_explicit_projector_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _explicit_proj_ds_hol_worker(rank, world_size, tmpdir):
    """Explicit output projectors + DeepSpeed ZeRO-2 student + Holistic."""
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
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Gradient checkpointing + DeepSpeed
# =========================================================================


def _gc_ds_bkd_worker(rank, world_size, tmpdir):
    """Gradient checkpointing + DeepSpeed ZeRO-2 student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(
            tmpdir,
            rank,
            deepspeed=ds_config,
            gradient_checkpointing=True,
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _gc_ds_hol_worker(rank, world_size, tmpdir):
    """Gradient checkpointing + DeepSpeed ZeRO-2 student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(
            tmpdir,
            rank,
            deepspeed=ds_config,
            gradient_checkpointing=True,
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Checkpoint save/load — DeepSpeed student
# =========================================================================


def _ckpt_ds_hol_worker(rank, world_size, tmpdir):
    """DeepSpeed ZeRO-2 student + Holistic: train 5 steps, save, resume to 10."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    ds_config = _write_ds_config(tmpdir, rank)

    # Phase 1: train 5 steps with save_steps=5
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(
            tmpdir,
            rank,
            max_steps=5,
            save_steps=5,
            deepspeed=ds_config,
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    assert distiller.state.global_step == 5, (
        f"Rank {rank}: expected 5 steps, got {distiller.state.global_step}"
    )

    # Find checkpoint directory
    ckpt_dir = os.path.join(tmpdir, "checkpoint-5")
    assert os.path.isdir(ckpt_dir), f"Rank {rank}: checkpoint dir {ckpt_dir} not found"

    # Phase 2: resume from checkpoint, train to 10 steps
    teacher2, student2 = _make_teacher_student(rank)
    alignments2 = _make_alignments(teacher2, student2)
    ds_config2 = _write_ds_config(tmpdir, rank)

    distiller2 = HolisticDistiller(
        student_model=student2,
        teacher_model=teacher2,
        alignments=alignments2,
        args=_make_holistic_args(
            tmpdir,
            rank,
            max_steps=10,
            save_steps=999,
            deepspeed=ds_config2,
        ),
        train_dataset=_dataset(),
    )
    distiller2.train(resume_from_checkpoint=ckpt_dir)
    assert distiller2.state.global_step == 10, (
        f"Rank {rank}: expected 10 steps after resume, got {distiller2.state.global_step}"
    )


def _ckpt_ds_bkd_worker(rank, world_size, tmpdir):
    """DeepSpeed ZeRO-2 student + Blockwise: train 5 steps, save, resume to 10."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    ds_config = _write_ds_config(tmpdir, rank)

    # Phase 1: train 5 steps with save_steps=5
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(
            tmpdir,
            rank,
            max_steps=5,
            save_steps=5,
            deepspeed=ds_config,
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    assert distiller.state.global_step == 5, (
        f"Rank {rank}: expected 5 steps, got {distiller.state.global_step}"
    )

    # Find checkpoint directory
    ckpt_dir = os.path.join(tmpdir, "checkpoint-5")
    assert os.path.isdir(ckpt_dir), f"Rank {rank}: checkpoint dir {ckpt_dir} not found"

    # Phase 2: resume from checkpoint, train to 10 steps
    teacher2, student2 = _make_teacher_student(rank)
    alignments2 = _make_alignments(teacher2, student2)
    ds_config2 = _write_ds_config(tmpdir, rank)

    distiller2 = BlockwiseDistiller(
        teacher_model=teacher2,
        alignments=alignments2,
        args=_make_blockwise_args(
            tmpdir,
            rank,
            max_steps=10,
            save_steps=999,
            deepspeed=ds_config2,
        ),
        train_dataset=_dataset(),
    )
    distiller2.train(resume_from_checkpoint=ckpt_dir)
    assert distiller2.state.global_step == 10, (
        f"Rank {rank}: expected 10 steps after resume, got {distiller2.state.global_step}"
    )


# =========================================================================
# Evaluate in distributed — DeepSpeed student
# =========================================================================


def _eval_ds_hol_worker(rank, world_size, tmpdir):
    """DeepSpeed ZeRO-2 student + Holistic: train then evaluate."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_ds_bkd_worker(rank, world_size, tmpdir):
    """DeepSpeed ZeRO-2 student + Blockwise: train then evaluate."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_ds_rbd_worker(rank, world_size, tmpdir):
    """DeepSpeed ZeRO-2 student + ResponseBased: train then evaluate."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)
    ds_config = _write_ds_config(tmpdir, rank)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


# =========================================================================
# Gradient accumulation + DeepSpeed student
# =========================================================================


def _grad_accum_ds_hol_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + DeepSpeed ZeRO-2 student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config_path = os.path.join(tmpdir, f"ds_config_ga_{rank}.json")
    ds_config = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 5e-3, "weight_decay": 0.0},
        },
        "fp16": {"enabled": False},
        "zero_optimization": {"stage": 2},
    }
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(
            tmpdir,
            rank,
            deepspeed=ds_config_path,
            gradient_accumulation_steps=2,
        ),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _grad_accum_ds_bkd_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + DeepSpeed ZeRO-2 student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config_path = os.path.join(tmpdir, f"ds_config_ga_{rank}.json")
    ds_config = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 5e-3, "weight_decay": 0.0},
        },
        "fp16": {"enabled": False},
        "zero_optimization": {"stage": 2},
    }
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(
            tmpdir,
            rank,
            deepspeed=ds_config_path,
            gradient_accumulation_steps=2,
        ),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _grad_accum_ds_rbd_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + DeepSpeed ZeRO-2 student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)
    ds_config_path = os.path.join(tmpdir, f"ds_config_ga_{rank}.json")
    ds_config = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 5e-3, "weight_decay": 0.0},
        },
        "fp16": {"enabled": False},
        "zero_optimization": {"stage": 2},
    }
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(
            tmpdir,
            rank,
            deepspeed=ds_config_path,
            gradient_accumulation_steps=2,
        ),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# DeepSpeed ZeRO-3
# =========================================================================


def _zero3_hol_worker(rank, world_size, tmpdir):
    """DeepSpeed ZeRO-3 student + Holistic distiller."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds3_config(tmpdir, rank)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _zero3_bkd_worker(rank, world_size, tmpdir):
    """DeepSpeed ZeRO-3 student + Blockwise distiller."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)
    ds_config = _write_ds3_config(tmpdir, rank)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _zero3_rbd_worker(rank, world_size, tmpdir):
    """DeepSpeed ZeRO-3 student + ResponseBased distiller."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)
    ds_config = _write_ds3_config(tmpdir, rank)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, deepspeed=ds_config),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


# =========================================================================
# Test classes
# =========================================================================


class TestDeepSpeedReplicatedTeacher:
    """DeepSpeed ZeRO-2 student + replicated teacher."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_response_based(self, tmp_path):
        _spawn_ds(_rep_ds_rbd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn_ds(_rep_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_rep_ds_bkd_worker, tmp_path)


class TestDeepSpeedAllRanksFSDPTeacher:
    """DeepSpeed ZeRO-2 student + all-ranks FSDP teacher."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_response_based(self, tmp_path):
        _spawn_ds(_fsdp_all_ds_rbd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn_ds(_fsdp_all_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_fsdp_all_ds_bkd_worker, tmp_path)


class TestDeepSpeedSplitFSDPTeacher:
    """DeepSpeed ZeRO-2 student + split-GPU FSDP teacher."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_response_based(self, tmp_path):
        _spawn_ds(_fsdp_split_ds_rbd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn_ds(_fsdp_split_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_fsdp_split_ds_bkd_worker, tmp_path)


class TestDeepSpeedPPTeacher:
    """DeepSpeed ZeRO-2 student + PP teacher on GPUs 2,3."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_response_based(self, tmp_path):
        _spawn_ds(_pp_ds_rbd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn_ds(_pp_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_pp_ds_bkd_worker, tmp_path)


class TestDeepSpeedDispatchTeacher:
    """DeepSpeed ZeRO-2 student + dispatch sharded teacher."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_holistic(self, tmp_path):
        _spawn_ds(_dispatch_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_dispatch_ds_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn_ds(_dispatch_ds_rbd_worker, tmp_path)


class TestDeepSpeedTPTeacher:
    """DeepSpeed ZeRO-2 student + TP teacher on GPUs 2,3."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_holistic(self, tmp_path):
        _spawn_ds(_tp_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_tp_ds_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn_ds(_tp_ds_rbd_worker, tmp_path)


class TestExplicitProjectorsDeepSpeed:
    """Explicit (non-auto) projectors with DeepSpeed ZeRO-2 student."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_blockwise_input_output_projectors(self, tmp_path):
        """Blockwise with explicit input+output projectors under DeepSpeed."""
        _spawn_ds(_explicit_proj_ds_bkd_worker, tmp_path)

    def test_holistic_output_projector(self, tmp_path):
        """Holistic with explicit output projector under DeepSpeed."""
        _spawn_ds(_explicit_proj_ds_hol_worker, tmp_path)


class TestGradientCheckpointingDeepSpeed:
    """Gradient checkpointing with DeepSpeed ZeRO-2 student."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_blockwise(self, tmp_path):
        """Blockwise GC + DeepSpeed: checkpoint calls inside block forward."""
        _spawn_ds(_gc_ds_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        """Holistic GC + DeepSpeed: no-op GC stub."""
        _spawn_ds(_gc_ds_hol_worker, tmp_path)


class TestCheckpointSaveLoadDeepSpeed:
    """Checkpoint save and resume with DeepSpeed ZeRO-2 student."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_holistic(self, tmp_path):
        _spawn_ds(_ckpt_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_ckpt_ds_bkd_worker, tmp_path)


class TestEvaluateDistributedDeepSpeed:
    """Evaluate after training with DeepSpeed ZeRO-2 student."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_holistic(self, tmp_path):
        _spawn_ds(_eval_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_eval_ds_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn_ds(_eval_ds_rbd_worker, tmp_path)


class TestGradientAccumulationDeepSpeed:
    """Gradient accumulation with DeepSpeed ZeRO-2 student."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_holistic(self, tmp_path):
        _spawn_ds(_grad_accum_ds_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_grad_accum_ds_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn_ds(_grad_accum_ds_rbd_worker, tmp_path)


class TestDeepSpeedZeRO3:
    """DeepSpeed ZeRO-3 student with all distiller types."""

    pytestmark = [*_requires_4_gpus, _requires_deepspeed]

    def test_holistic(self, tmp_path):
        _spawn_ds(_zero3_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn_ds(_zero3_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn_ds(_zero3_rbd_worker, tmp_path)
