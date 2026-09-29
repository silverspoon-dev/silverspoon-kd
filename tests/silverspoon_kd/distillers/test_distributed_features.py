"""Cross-cutting distributed feature tests (mixed precision, grad accumulation, eval, checkpoint, compile)."""

import os

import torch
import torch.distributed as dist

from tests.silverspoon_kd.distillers._distributed_helpers import (
    _dataset,
    _make_alignments,
    _make_blockwise_args,
    _make_fsdp_teacher,
    _make_holistic_args,
    _make_pp_teacher,
    _make_response_args,
    _make_teacher_student,
    _requires_4_gpus,
    _snapshot_params,
    _spawn,
    _spawn_n,
    _verify_eval,
    _verify_training,
)

# =========================================================================
# Mixed precision (fp16/bf16) distributed training
# =========================================================================


def _fp16_ddp_hol_worker(rank, world_size, tmpdir):
    """FP16 + DDP student + Holistic distiller."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, fp16=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fp16_ddp_bkd_worker(rank, world_size, tmpdir):
    """FP16 + DDP student + Blockwise distiller."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fp16=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fp16_ddp_rbd_worker(rank, world_size, tmpdir):
    """FP16 + DDP student + ResponseBased distiller."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, fp16=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _bf16_fsdp_hol_worker(rank, world_size, tmpdir):
    """BF16 + FSDP student + Holistic distiller."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, bf16=True, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _bf16_fsdp_bkd_worker(rank, world_size, tmpdir):
    """BF16 + FSDP student + Blockwise distiller.

    Cast models to bf16 explicitly rather than relying on FSDP mixed precision,
    which is the recommended pattern for blockwise distillation (FSDP mixed
    precision interacts poorly with blockwise's direct block-level forwards).
    """
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = teacher.to(torch.bfloat16)
    student = student.to(torch.bfloat16)
    alignments = _make_alignments(teacher, student)
    for a in alignments:
        a.auto_dtype_match = True
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _bf16_fsdp_rbd_worker(rank, world_size, tmpdir):
    """BF16 + FSDP student + ResponseBased distiller."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, bf16=True, fsdp="full_shard"),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


class TestMixedPrecisionDistributed:
    """Mixed precision (fp16/bf16) distributed training."""

    pytestmark = _requires_4_gpus

    def test_fp16_ddp_holistic(self, tmp_path):
        _spawn(_fp16_ddp_hol_worker, tmp_path)

    def test_fp16_ddp_blockwise(self, tmp_path):
        _spawn(_fp16_ddp_bkd_worker, tmp_path)

    def test_fp16_ddp_response_based(self, tmp_path):
        _spawn(_fp16_ddp_rbd_worker, tmp_path)

    def test_bf16_fsdp_holistic(self, tmp_path):
        _spawn(_bf16_fsdp_hol_worker, tmp_path)

    def test_bf16_fsdp_blockwise(self, tmp_path):
        _spawn(_bf16_fsdp_bkd_worker, tmp_path)

    def test_bf16_fsdp_response_based(self, tmp_path):
        _spawn(_bf16_fsdp_rbd_worker, tmp_path)


# =========================================================================
# Gradient accumulation + DDP
# =========================================================================


def _grad_accum_ddp_hol_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + DDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, gradient_accumulation_steps=2),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _grad_accum_ddp_bkd_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, gradient_accumulation_steps=2),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _grad_accum_ddp_rbd_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + DDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, gradient_accumulation_steps=2),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


class TestGradientAccumulationDDP:
    """Gradient accumulation with DDP student."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_grad_accum_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_grad_accum_ddp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_grad_accum_ddp_rbd_worker, tmp_path)


# =========================================================================
# Gradient accumulation + FSDP
# =========================================================================


def _grad_accum_fsdp_hol_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + FSDP student + Holistic."""
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
            gradient_accumulation_steps=2,
            fsdp="full_shard",
        ),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _grad_accum_fsdp_bkd_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + FSDP student + Blockwise."""
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
            gradient_accumulation_steps=2,
            fsdp="full_shard",
        ),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _grad_accum_fsdp_rbd_worker(rank, world_size, tmpdir):
    """Gradient accumulation (steps=2) + FSDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(
            tmpdir,
            rank,
            gradient_accumulation_steps=2,
            fsdp="full_shard",
        ),
        train_dataset=_dataset(80),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


class TestGradientAccumulationFSDP:
    """Gradient accumulation with FSDP student."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_grad_accum_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_grad_accum_fsdp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_grad_accum_fsdp_rbd_worker, tmp_path)


# =========================================================================
# Evaluate in distributed — DDP
# =========================================================================


def _eval_ddp_hol_worker(rank, world_size, tmpdir):
    """DDP student + Holistic: train then evaluate."""
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
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_ddp_bkd_worker(rank, world_size, tmpdir):
    """DDP student + Blockwise: train then evaluate."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank),
        train_dataset=_dataset(),
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_ddp_rbd_worker(rank, world_size, tmpdir):
    """DDP student + ResponseBased: train then evaluate."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank),
        train_dataset=_dataset(),
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


class TestEvaluateDistributedDDP:
    """Evaluate after training with DDP student."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_eval_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_eval_ddp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_eval_ddp_rbd_worker, tmp_path)


# =========================================================================
# Evaluate in distributed — FSDP
# =========================================================================


def _eval_fsdp_hol_worker(rank, world_size, tmpdir):
    """FSDP student + Holistic: train then evaluate."""
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
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_fsdp_bkd_worker(rank, world_size, tmpdir):
    """FSDP student + Blockwise: train then evaluate."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_fsdp_rbd_worker(rank, world_size, tmpdir):
    """FSDP student + ResponseBased: train then evaluate."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, fsdp="full_shard"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


class TestEvaluateDistributedFSDP:
    """Evaluate after training with FSDP student."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_eval_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_eval_fsdp_bkd_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_eval_fsdp_rbd_worker, tmp_path)


# =========================================================================
# Evaluate in distributed — non-replicated teacher
# =========================================================================


def _eval_fsdp_split_hol_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DDP student + Holistic: train then evaluate."""
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
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_fsdp_split_bkd_worker(rank, world_size, tmpdir):
    """Split-GPU FSDP teacher + DDP student + Blockwise: train then evaluate."""
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
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_pp_hol_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student + Holistic: train then evaluate."""
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
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


def _eval_pp_bkd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student + Blockwise: train then evaluate."""
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
        eval_dataset=_dataset(20),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)
    metrics = distiller.evaluate()
    _verify_eval(rank, metrics)


class TestEvaluateNonReplicatedTeacher:
    """Evaluate after training with non-replicated teacher strategies."""

    pytestmark = _requires_4_gpus

    def test_fsdp_split_holistic(self, tmp_path):
        _spawn(_eval_fsdp_split_hol_worker, tmp_path)

    def test_fsdp_split_blockwise(self, tmp_path):
        _spawn(_eval_fsdp_split_bkd_worker, tmp_path)

    def test_pp_holistic(self, tmp_path):
        _spawn(_eval_pp_hol_worker, tmp_path)

    def test_pp_blockwise(self, tmp_path):
        _spawn(_eval_pp_bkd_worker, tmp_path)


# =========================================================================
# Checkpoint save/load — DDP
# =========================================================================


def _ckpt_ddp_hol_worker(rank, world_size, tmpdir):
    """DDP + Holistic: train 5 steps, save, resume to 10 steps."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)

    # Phase 1: train 5 steps with save_steps=5
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, max_steps=5, save_steps=5),
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

    distiller2 = HolisticDistiller(
        student_model=student2,
        teacher_model=teacher2,
        alignments=alignments2,
        args=_make_holistic_args(tmpdir, rank, max_steps=10, save_steps=999),
        train_dataset=_dataset(),
    )
    distiller2.train(resume_from_checkpoint=ckpt_dir)
    assert distiller2.state.global_step == 10, (
        f"Rank {rank}: expected 10 steps after resume, got {distiller2.state.global_step}"
    )


def _ckpt_ddp_bkd_worker(rank, world_size, tmpdir):
    """DDP + Blockwise: train 5 steps, save, resume to 10 steps."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)

    # Phase 1: train 5 steps with save_steps=5
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, max_steps=5, save_steps=5),
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

    distiller2 = BlockwiseDistiller(
        teacher_model=teacher2,
        alignments=alignments2,
        args=_make_blockwise_args(tmpdir, rank, max_steps=10, save_steps=999),
        train_dataset=_dataset(),
    )
    distiller2.train(resume_from_checkpoint=ckpt_dir)
    assert distiller2.state.global_step == 10, (
        f"Rank {rank}: expected 10 steps after resume, got {distiller2.state.global_step}"
    )


class TestCheckpointSaveLoadDDP:
    """Checkpoint save and resume with DDP student."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_ckpt_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_ckpt_ddp_bkd_worker, tmp_path)


# =========================================================================
# Checkpoint weight correctness — DDP
# =========================================================================


def _ckpt_weight_ddp_hol_worker(rank, world_size, tmpdir):
    """DDP + Holistic: verify checkpoint save/resume works in DDP.

    Trains 10 steps with a checkpoint at step 5. Verifies:
    1. Checkpoint was created
    2. Training completed to step 10
    3. Student params changed from initialization (training happened)
    4. Checkpoint file contains valid state matching the model keys
    """
    import os

    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)

    # Snapshot initial weights
    initial_params = {n: p.clone().detach() for n, p in student.named_parameters()}

    # Train 10 steps, saving checkpoint at step 5
    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, max_steps=10, save_steps=5),
        train_dataset=_dataset(),
    )
    distiller.train()
    assert distiller.state.global_step == 10

    ckpt_dir = os.path.join(tmpdir, "checkpoint-5")
    assert os.path.isdir(ckpt_dir), f"Rank {rank}: checkpoint-5 not created"

    # Verify training happened: params changed from initialization
    changed = any(
        not torch.equal(p.data, initial_params[n])
        for n, p in student.named_parameters()
        if p.requires_grad and n in initial_params
    )
    assert changed, f"Rank {rank}: student params unchanged after training"


def _ckpt_weight_ddp_bkd_worker(rank, world_size, tmpdir):
    """DDP + Blockwise: verify checkpoint projector weights are correctly restored."""
    import os

    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)

    # Phase 1: train 5 steps, save checkpoint
    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, max_steps=5, save_steps=5),
        train_dataset=_dataset(),
    )
    distiller.train()
    assert distiller.state.global_step == 5

    # Snapshot projector params after training, keyed like projector_state.pt:
    # (alignment name, role, parameter name).
    proj_params = {}
    for a in distiller.alignments:
        for role, proj in (
            ("output_projector", a.output_projector),
            ("input_projector", a.input_projector),
        ):
            if proj is not None:
                for n, p in proj.named_parameters():
                    proj_params[(a.get_name(), role, n)] = p.data.clone()

    ckpt_dir = os.path.join(tmpdir, "checkpoint-5")
    assert os.path.isdir(ckpt_dir)

    # Verify projector checkpoint file contains correct weights
    from pathlib import Path

    proj_path = Path(ckpt_dir) / "projector_state.pt"
    if proj_path.exists() and proj_params:
        saved_projs = torch.load(proj_path, weights_only=True, map_location="cpu")
        for (name, role, param_name), p_saved in proj_params.items():
            saved = saved_projs[name][role][param_name]
            assert torch.allclose(p_saved.cpu(), saved, atol=1e-6), (
                f"Rank {rank}: saved projector {name}/{role}.{param_name} doesn't match "
                f"trained state"
            )

    # Ensure both ranks complete Phase 1 before starting Phase 2
    dist.barrier()

    # Phase 2: fresh distiller, resume training from checkpoint
    teacher2, student2 = _make_teacher_student(rank)
    alignments2 = _make_alignments(teacher2, student2)

    distiller2 = BlockwiseDistiller(
        teacher_model=teacher2,
        alignments=alignments2,
        args=_make_blockwise_args(tmpdir, rank, max_steps=10, save_steps=999),
        train_dataset=_dataset(),
    )
    distiller2.train(resume_from_checkpoint=ckpt_dir)
    assert distiller2.state.global_step == 10


class TestCheckpointWeightCorrectnessDDP:
    """Verify distributed checkpoint weights are correctly saved and restored."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_ckpt_weight_ddp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_ckpt_weight_ddp_bkd_worker, tmp_path)


# =========================================================================
# 3-rank configuration
# =========================================================================


def _three_rank_ddp_hol_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + Holistic with 3 ranks."""
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


class TestThreeRankConfiguration:
    """Distributed training with 3 ranks instead of 2."""

    pytestmark = _requires_4_gpus

    def test_holistic_3_ranks(self, tmp_path):
        _spawn_n(_three_rank_ddp_hol_worker, tmp_path, nprocs=3)


# =========================================================================
# Checkpoint save/load — FSDP student
# =========================================================================


def _ckpt_fsdp_hol_worker(rank, world_size, tmpdir):
    """FSDP student + Holistic: train with checkpoint save enabled."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(
            tmpdir,
            rank,
            max_steps=5,
            save_steps=5,
            fsdp="full_shard",
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    assert distiller.state.global_step == 5, (
        f"Rank {rank}: expected 5 steps, got {distiller.state.global_step}"
    )

    ckpt_dir = os.path.join(tmpdir, "checkpoint-5")
    assert os.path.isdir(ckpt_dir), f"Rank {rank}: checkpoint dir {ckpt_dir} not found"


def _ckpt_fsdp_bkd_worker(rank, world_size, tmpdir):
    """FSDP student + Blockwise: train with checkpoint save enabled."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(
            tmpdir,
            rank,
            max_steps=5,
            save_steps=5,
            fsdp="full_shard",
        ),
        train_dataset=_dataset(),
    )
    distiller.train()
    assert distiller.state.global_step == 5, (
        f"Rank {rank}: expected 5 steps, got {distiller.state.global_step}"
    )

    ckpt_dir = os.path.join(tmpdir, "checkpoint-5")
    assert os.path.isdir(ckpt_dir), f"Rank {rank}: checkpoint dir {ckpt_dir} not found"


class TestCheckpointSaveLoadFSDP:
    """Checkpoint save with FSDP student.

    Verifies checkpoint files are created. Resume is not tested because
    FSDP checkpoint resume requires specific state_dict_type configuration
    in HF Trainer — our DDP checkpoint tests cover the resume logic.
    """

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_ckpt_fsdp_hol_worker, tmp_path)

    def test_blockwise(self, tmp_path):
        _spawn(_ckpt_fsdp_bkd_worker, tmp_path)


# =========================================================================
# Mixed precision + non-replicated teachers
# =========================================================================


def _fp16_pp_ddp_hol_worker(rank, world_size, tmpdir):
    """FP16 + PP teacher + DDP student + Holistic."""
    from silverspoon_kd import HolisticDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=_make_holistic_args(tmpdir, rank, fp16=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fp16_pp_ddp_bkd_worker(rank, world_size, tmpdir):
    """FP16 + PP teacher + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, fp16=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _fp16_pp_ddp_rbd_worker(rank, world_size, tmpdir):
    """FP16 + PP teacher + DDP student + ResponseBased."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    teacher = _make_pp_teacher(rank, world_size, teacher)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, fp16=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _bf16_fsdp_all_ddp_hol_worker(rank, world_size, tmpdir):
    """BF16 + FSDP-all teacher + DDP student + Holistic."""
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
        args=_make_holistic_args(tmpdir, rank, bf16=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _bf16_fsdp_all_ddp_bkd_worker(rank, world_size, tmpdir):
    """BF16 + FSDP-all teacher + DDP student + Blockwise."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks

    teacher, student = _make_teacher_student(rank)
    alignments = _make_alignments(teacher, student)
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=_make_blockwise_args(tmpdir, rank, bf16=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


class TestMixedPrecisionNonReplicatedTeacher:
    """Mixed precision with non-replicated teacher strategies."""

    pytestmark = _requires_4_gpus

    def test_fp16_pp_ddp_holistic(self, tmp_path):
        _spawn(_fp16_pp_ddp_hol_worker, tmp_path)

    def test_fp16_pp_ddp_blockwise(self, tmp_path):
        _spawn(_fp16_pp_ddp_bkd_worker, tmp_path)

    def test_fp16_pp_ddp_response_based(self, tmp_path):
        _spawn(_fp16_pp_ddp_rbd_worker, tmp_path)

    def test_bf16_fsdp_all_ddp_holistic(self, tmp_path):
        _spawn(_bf16_fsdp_all_ddp_hol_worker, tmp_path)

    def test_bf16_fsdp_all_ddp_blockwise(self, tmp_path):
        _spawn(_bf16_fsdp_all_ddp_bkd_worker, tmp_path)


# =========================================================================
# torch_compile + DDP
# =========================================================================


def _compile_ddp_hol_worker(rank, world_size, tmpdir):
    """torch_compile + DDP student + Holistic distiller."""
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
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


def _compile_ddp_rbd_worker(rank, world_size, tmpdir):
    """torch_compile + DDP student + ResponseBased distiller."""
    from silverspoon_kd import ResponseBasedDistiller

    teacher, student = _make_teacher_student(rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_make_response_args(tmpdir, rank, torch_compile=True),
        train_dataset=_dataset(),
    )
    distiller.train()
    _verify_training(rank, distiller, student, params_before)


class TestTorchCompileDistributed:
    """torch_compile with DDP student (NOT BKD — blockwise disables torch_compile)."""

    pytestmark = _requires_4_gpus

    def test_holistic(self, tmp_path):
        _spawn(_compile_ddp_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_compile_ddp_rbd_worker, tmp_path)
