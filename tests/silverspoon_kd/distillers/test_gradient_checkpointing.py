"""Gradient checkpointing tests for all distiller types.

Tests that gradient_checkpointing=True works correctly across:
- Single-GPU (CPU): all 3 distillers
- DDP student (4 GPUs, mp.spawn)
- FSDP student (4 GPUs, mp.spawn) — especially BlockwiseDistiller with per-block FSDP

BlockwiseDistiller wraps student block forward calls in
torch.utils.checkpoint.checkpoint when GC is enabled.
Holistic/ResponseBased get a no-op stub for plain nn.Module models.
"""

import os
from unittest.mock import patch

import pytest
import torch

from tests.silverspoon_kd.distillers._distributed_helpers import _spawn_with_retry

_requires_4_gpus = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 4 if torch.cuda.is_available() else True,
        reason="At least 4 GPUs required",
    ),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_teacher_student(device="cpu"):
    from tests.silverspoon_kd.conftest import SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3)
    if device != "cpu":
        teacher.to(device)
        student.to(device)
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


def _dataset(n=40):
    from tests.silverspoon_kd.conftest import DummyDataset

    return DummyDataset(num_samples=n, seq_len=16)


def _snapshot_params(model):
    return {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}


def _verify_gc_training(distiller, student, params_before, expected_steps=5):
    """Post-training assertions for GC tests."""
    assert distiller.state.global_step == expected_steps
    losses = [e["loss"] for e in distiller.state.log_history if "loss" in e]
    assert len(losses) > 0
    for v in losses:
        assert torch.isfinite(torch.tensor(v)), f"Non-finite loss: {v}"
        assert v > 0, f"Non-positive loss: {v}"
    changed = any(
        not torch.equal(p.data, params_before[n])
        for n, p in student.named_parameters()
        if p.requires_grad and n in params_before
    )
    assert changed, "Student params unchanged after training"


# ---------------------------------------------------------------------------
# Single-GPU tests (CPU)
# ---------------------------------------------------------------------------


class TestGradientCheckpointingSingleGPU:
    """Gradient checkpointing on CPU — all distiller types."""

    def test_blockwise_gc_trains(self, tmp_path):
        """BlockwiseDistiller with GC: training completes and params update."""
        from silverspoon_kd import BlockwiseDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        teacher, student = _make_teacher_student()
        alignments = _make_alignments(teacher, student)
        params_before = _snapshot_params(student)

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=5,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                logging_steps=1,
                use_cpu=True,
                report_to=[],
            ),
            train_dataset=_dataset(),
        )
        distiller.train()
        _verify_gc_training(distiller, student, params_before)
        assert getattr(distiller.model, "_gradient_checkpointing", False)

    def test_holistic_gc_trains(self, tmp_path):
        """HolisticDistiller with GC: no-op stub but must not crash."""
        from silverspoon_kd import HolisticDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        teacher, student = _make_teacher_student()
        alignments = _make_alignments(teacher, student)
        params_before = _snapshot_params(student)

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=5,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                logging_steps=1,
                use_cpu=True,
                report_to=[],
            ),
            train_dataset=_dataset(),
        )
        distiller.train()
        _verify_gc_training(distiller, student, params_before)

    def test_response_based_gc_trains(self, tmp_path):
        """ResponseBasedDistiller with GC: no-op stub but must not crash."""
        from silverspoon_kd import ResponseBasedDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        teacher, student = _make_teacher_student()
        params_before = _snapshot_params(student)

        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=5,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                logging_steps=1,
                use_cpu=True,
                report_to=[],
                alpha=0.0,
            ),
            train_dataset=_dataset(),
        )
        distiller.train()
        _verify_gc_training(distiller, student, params_before)

    def test_blockwise_gc_calls_checkpoint(self, tmp_path):
        """Verify torch.utils.checkpoint.checkpoint is actually called."""
        from silverspoon_kd import BlockwiseDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        teacher, student = _make_teacher_student()
        alignments = _make_alignments(teacher, student)
        real_fn = torch.utils.checkpoint.checkpoint

        call_count = 0

        def counting_checkpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return real_fn(*args, **kwargs)

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=3,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                logging_steps=1,
                use_cpu=True,
                report_to=[],
            ),
            train_dataset=_dataset(),
        )

        with patch("torch.utils.checkpoint.checkpoint", side_effect=counting_checkpoint):
            distiller.train()

        # 3 steps * 3 blocks = 9 calls minimum
        assert call_count > 0, "torch.utils.checkpoint.checkpoint was never called"

    def test_blockwise_gc_every_n_layers(self, tmp_path):
        """``every_n_layers`` checkpoints only every n-th block, in alignment order.

        Passed via ``gradient_checkpointing_kwargs`` so it works with both Trainer
        conventions: older transformers forward the whole dict, transformers >= 5.16
        pop it and pass ``every_n_layers=`` explicitly.
        """
        from silverspoon_kd import BlockwiseDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        real_fn = torch.utils.checkpoint.checkpoint

        def run(every_n_layers):
            teacher, student = _make_teacher_student()
            alignments = _make_alignments(teacher, student)  # 3 blocks
            calls = 0

            def counting_checkpoint(*args, **kwargs):
                nonlocal calls
                calls += 1
                return real_fn(*args, **kwargs)

            distiller = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=alignments,
                args=TrainingArguments(
                    output_dir=str(tmp_path / f"every_{every_n_layers}"),
                    max_steps=3,
                    per_device_train_batch_size=2,
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"every_n_layers": every_n_layers},
                    logging_steps=1,
                    use_cpu=True,
                    report_to=[],
                ),
                train_dataset=_dataset(),
            )
            with patch("torch.utils.checkpoint.checkpoint", side_effect=counting_checkpoint):
                distiller.train()
            return calls, distiller.model

        calls_all, model_all = run(1)
        calls_half, model_half = run(2)

        assert len(model_all._gradient_checkpointing_blocks) == 3
        assert len(model_half._gradient_checkpointing_blocks) == 2  # blocks 0 and 2
        # ``every_n_layers`` must not leak into the torch.utils.checkpoint kwargs
        assert "every_n_layers" not in model_half._gradient_checkpointing_kwargs
        assert calls_all > 0
        assert calls_half * 3 == calls_all * 2, (calls_all, calls_half)

    def test_offload_accepted_by_both_trainer_conventions(self):
        """``offload`` is taken as a keyword or from ``gradient_checkpointing_kwargs``.

        transformers >= 5.17 passes ``offload=`` explicitly; older Trainers put
        it in the kwargs dict.  Either way it must not leak into the
        ``torch.utils.checkpoint`` kwargs.
        """
        from silverspoon_kd.distillers.blockwise_distiller import StudentBlocksContainer

        teacher, student = _make_teacher_student()
        container = StudentBlocksContainer(teacher, _make_alignments(teacher, student))

        container.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=None, every_n_layers=1, offload=True
        )
        assert container._gradient_checkpointing_offload is True
        assert "offload" not in container._gradient_checkpointing_kwargs

        container.gradient_checkpointing_disable()
        assert container._gradient_checkpointing_offload is False

        container.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"offload": True, "use_reentrant": False}
        )
        assert container._gradient_checkpointing_offload is True
        assert container._gradient_checkpointing_kwargs == {"use_reentrant": False}

    def test_blockwise_gc_offload_trains(self, tmp_path):
        """Training runs with activation offloading requested."""
        from silverspoon_kd import BlockwiseDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        teacher, student = _make_teacher_student()
        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=_make_alignments(teacher, student),
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=2,
                per_device_train_batch_size=2,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"offload": True},
                logging_steps=1,
                use_cpu=True,
                report_to=[],
            ),
            train_dataset=_dataset(),
        )
        distiller.train()
        assert distiller.model._gradient_checkpointing_offload is True
        assert distiller.state.global_step == 2

    def test_every_n_layers_must_be_positive(self):
        from silverspoon_kd.distillers.blockwise_distiller import StudentBlocksContainer

        teacher, student = _make_teacher_student()
        container = StudentBlocksContainer(teacher, _make_alignments(teacher, student))
        with pytest.raises(ValueError, match="every_n_layers"):
            container.gradient_checkpointing_enable(every_n_layers=0)
        assert not container.is_gradient_checkpointing

    def test_blockwise_no_gc_skips_checkpoint(self, tmp_path):
        """Without GC, torch.utils.checkpoint.checkpoint should NOT be called."""
        from silverspoon_kd import BlockwiseDistiller
        from silverspoon_kd.training_arguments import TrainingArguments

        teacher, student = _make_teacher_student()
        alignments = _make_alignments(teacher, student)

        call_count = 0
        real_fn = torch.utils.checkpoint.checkpoint

        def counting_checkpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return real_fn(*args, **kwargs)

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=3,
                per_device_train_batch_size=2,
                gradient_checkpointing=False,
                logging_steps=1,
                use_cpu=True,
                report_to=[],
            ),
            train_dataset=_dataset(),
        )

        with patch("torch.utils.checkpoint.checkpoint", side_effect=counting_checkpoint):
            distiller.train()

        assert call_count == 0, f"checkpoint called {call_count} times with GC disabled"


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _init_process(rank, world_size, port, fn, *args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    import torch.distributed as dist

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


def _verify_gc_dist(rank, distiller, student, params_before, expected_steps=10):
    """Post-training assertions for distributed GC tests."""
    assert distiller.state.global_step == expected_steps, (
        f"Rank {rank}: expected {expected_steps} steps, got {distiller.state.global_step}"
    )
    losses = [e["loss"] for e in distiller.state.log_history if "loss" in e]
    assert len(losses) > 0, f"Rank {rank}: no losses"
    for v in losses:
        assert torch.isfinite(torch.tensor(v)), f"Rank {rank}: non-finite loss {v}"
        assert v > 0, f"Rank {rank}: non-positive loss {v}"
    changed = any(
        not torch.equal(p.data, params_before[n])
        for n, p in student.named_parameters()
        if p.requires_grad and n in params_before
    )
    assert changed, f"Rank {rank}: student params unchanged"


# ---------------------------------------------------------------------------
# DDP + GC workers
# ---------------------------------------------------------------------------


def _gc_ddp_bkd_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + Blockwise + gradient checkpointing."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_ddp_hol_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + Holistic + gradient checkpointing."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_ddp_rbd_worker(rank, world_size, tmpdir):
    """Replicated teacher + DDP student + ResponseBased + gradient checkpointing."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


# ---------------------------------------------------------------------------
# FSDP + GC workers
# ---------------------------------------------------------------------------


def _gc_fsdp_bkd_worker(rank, world_size, tmpdir):
    """Replicated teacher + FSDP student + Blockwise + gradient checkpointing."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_fsdp_hol_worker(rank, world_size, tmpdir):
    """Replicated teacher + FSDP student + Holistic + gradient checkpointing."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_fsdp_rbd_worker(rank, world_size, tmpdir):
    """Replicated teacher + FSDP student + ResponseBased + gradient checkpointing."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestGradientCheckpointingDDP:
    """Gradient checkpointing with DDP student."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        _spawn(_gc_ddp_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_ddp_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_ddp_rbd_worker, tmp_path)


class TestGradientCheckpointingFSDP:
    """Gradient checkpointing with FSDP student."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        """BlockwiseDistiller + FSDP + GC: per-block FSDP with checkpointing."""
        _spawn(_gc_fsdp_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_fsdp_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_fsdp_rbd_worker, tmp_path)


# ---------------------------------------------------------------------------
# GC + non-replicated teacher workers
# ---------------------------------------------------------------------------


def _gc_pp_teacher_bkd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student + Blockwise + gradient checkpointing."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import place_teacher_pp
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    teacher = place_teacher_pp(teacher, [2, 3], "cuda")
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_pp_teacher_hol_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student + Holistic + gradient checkpointing."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.distributed.strategies import place_teacher_pp
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    teacher = place_teacher_pp(teacher, [2, 3], "cuda")
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_fsdp_teacher_bkd_worker(rank, world_size, tmpdir):
    """FSDP-sharded teacher + DDP student + Blockwise + gradient checkpointing."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_fsdp_teacher_hol_worker(rank, world_size, tmpdir):
    """FSDP-sharded teacher + DDP student + Holistic + gradient checkpointing."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_pp_teacher_rbd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + DDP student + ResponseBased + gradient checkpointing."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import place_teacher_pp
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")
    teacher = place_teacher_pp(teacher, [2, 3], "cuda")
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


class TestGradientCheckpointingPPTeacher:
    """Gradient checkpointing with PP teacher on split GPUs."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        _spawn(_gc_pp_teacher_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_pp_teacher_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_pp_teacher_rbd_worker, tmp_path)


def _gc_fsdp_teacher_rbd_worker(rank, world_size, tmpdir):
    """FSDP-sharded teacher + DDP student + ResponseBased + gradient checkpointing."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


class TestGradientCheckpointingFSDPTeacher:
    """Gradient checkpointing with FSDP-sharded teacher (all-ranks)."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        _spawn(_gc_fsdp_teacher_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_fsdp_teacher_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_fsdp_teacher_rbd_worker, tmp_path)


# ---------------------------------------------------------------------------
# GC + TP teacher + DDP student workers
# ---------------------------------------------------------------------------


def _gc_tp_teacher_bkd_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DDP student + Blockwise + gradient checkpointing."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import (
        install_tp_device_hooks,
        parallelize_teacher_tp,
    )
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2).to(f"cuda:{rank}")

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
    teacher = parallelize_teacher_tp(teacher, [2, 3], "cuda")
    install_tp_device_hooks(teacher)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_tp_teacher_hol_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DDP student + Holistic + gradient checkpointing."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.distributed.strategies import (
        install_tp_device_hooks,
        parallelize_teacher_tp,
    )
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2).to(f"cuda:{rank}")

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
    teacher = parallelize_teacher_tp(teacher, [2, 3], "cuda")
    install_tp_device_hooks(teacher)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_tp_teacher_rbd_worker(rank, world_size, tmpdir):
    """TP teacher (GPUs 2,3) + DDP student + ResponseBased + gradient checkpointing."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import (
        install_tp_device_hooks,
        parallelize_teacher_tp,
    )
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel, TPSimpleModel

    teacher = TPSimpleModel(64, 128, 2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 2).to(f"cuda:{rank}")
    teacher = parallelize_teacher_tp(teacher, [2, 3], "cuda")
    install_tp_device_hooks(teacher)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


class TestGradientCheckpointingTPTeacher:
    """Gradient checkpointing with TP teacher on split GPUs."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        _spawn(_gc_tp_teacher_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_tp_teacher_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_tp_teacher_rbd_worker, tmp_path)


# ---------------------------------------------------------------------------
# GC + FSDP-split teacher + DDP student workers
# ---------------------------------------------------------------------------


def _gc_fsdp_split_teacher_bkd_worker(rank, world_size, tmpdir):
    """FSDP-split teacher (GPUs 2,3) + DDP student + Blockwise + gradient checkpointing."""
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        ShardingStrategy,
    )

    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import _build_wrap_policy
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_fsdp_split_teacher_hol_worker(rank, world_size, tmpdir):
    """FSDP-split teacher (GPUs 2,3) + DDP student + Holistic + gradient checkpointing."""
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        ShardingStrategy,
    )

    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.distributed.strategies import _build_wrap_policy
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_fsdp_split_teacher_rbd_worker(rank, world_size, tmpdir):
    """FSDP-split teacher (GPUs 2,3) + DDP student + ResponseBased + gradient checkpointing."""
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        ShardingStrategy,
    )

    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import _build_wrap_policy
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


class TestGradientCheckpointingFSDPSplitTeacher:
    """Gradient checkpointing with FSDP-split teacher on dedicated GPUs."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        _spawn(_gc_fsdp_split_teacher_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_fsdp_split_teacher_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_fsdp_split_teacher_rbd_worker, tmp_path)


# ---------------------------------------------------------------------------
# GC + dispatch teacher + DDP student workers
# ---------------------------------------------------------------------------


def _gc_dispatch_teacher_bkd_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + DDP student + Blockwise + gradient checkpointing."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            teacher_placement="sharded",
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_dispatch_teacher_hol_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + DDP student + Holistic + gradient checkpointing."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            teacher_placement="sharded",
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_dispatch_teacher_rbd_worker(rank, world_size, tmpdir):
    """Dispatch sharded teacher + DDP student + ResponseBased + gradient checkpointing."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            teacher_placement="sharded",
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


class TestGradientCheckpointingDispatchTeacher:
    """Gradient checkpointing with dispatch sharded teacher."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        _spawn(_gc_dispatch_teacher_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_dispatch_teacher_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_dispatch_teacher_rbd_worker, tmp_path)


# ---------------------------------------------------------------------------
# GC + PP teacher + FSDP student workers
# ---------------------------------------------------------------------------


def _gc_pp_fsdp_bkd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + FSDP student + Blockwise + gradient checkpointing."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import place_teacher_pp
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    teacher = place_teacher_pp(teacher, [2, 3], "cuda")
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_pp_fsdp_hol_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + FSDP student + Holistic + gradient checkpointing."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.distributed.strategies import place_teacher_pp
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    teacher = place_teacher_pp(teacher, [2, 3], "cuda")
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_pp_fsdp_rbd_worker(rank, world_size, tmpdir):
    """PP teacher (GPUs 2,3) + FSDP student + ResponseBased + gradient checkpointing."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import place_teacher_pp
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")
    teacher = place_teacher_pp(teacher, [2, 3], "cuda")
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


class TestGradientCheckpointingPPTeacherFSDPStudent:
    """Gradient checkpointing with PP teacher + FSDP student."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        _spawn(_gc_pp_fsdp_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_pp_fsdp_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_pp_fsdp_rbd_worker, tmp_path)


# ---------------------------------------------------------------------------
# GC + FSDP-all teacher + FSDP student workers
# ---------------------------------------------------------------------------


def _gc_fsdp_all_fsdp_bkd_worker(rank, world_size, tmpdir):
    """FSDP-all teacher + FSDP student + Blockwise + gradient checkpointing."""
    from silverspoon_kd import BlockwiseDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_fsdp_all_fsdp_hol_worker(rank, world_size, tmpdir):
    """FSDP-all teacher + FSDP student + Holistic + gradient checkpointing."""
    from silverspoon_kd import HolisticDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")

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
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


def _gc_fsdp_all_fsdp_rbd_worker(rank, world_size, tmpdir):
    """FSDP-all teacher + FSDP student + ResponseBased + gradient checkpointing."""
    from silverspoon_kd import ResponseBasedDistiller
    from silverspoon_kd.distributed.strategies import shard_teacher_fsdp_all_ranks
    from silverspoon_kd.training_arguments import TrainingArguments
    from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = SimpleModel(64, 64, 3).to(f"cuda:{rank}")
    teacher = shard_teacher_fsdp_all_ranks(teacher, device_id=rank)
    params_before = _snapshot_params(student)

    distiller = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=TrainingArguments(
            output_dir=tmpdir,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=999,
            learning_rate=5e-3,
            dataloader_num_workers=0,
            report_to=[],
            remove_unused_columns=False,
            local_rank=rank,
            fsdp="full_shard",
            alpha=0.0,
            auto_device_match=True,
            auto_dtype_match=True,
        ),
        train_dataset=DummyDataset(num_samples=40, seq_len=16),
    )
    distiller.train()
    _verify_gc_dist(rank, distiller, student, params_before)


class TestGradientCheckpointingFSDPAllTeacherFSDPStudent:
    """Gradient checkpointing with FSDP-all teacher + FSDP student."""

    pytestmark = _requires_4_gpus

    def test_blockwise(self, tmp_path):
        _spawn(_gc_fsdp_all_fsdp_bkd_worker, tmp_path)

    def test_holistic(self, tmp_path):
        _spawn(_gc_fsdp_all_fsdp_hol_worker, tmp_path)

    def test_response_based(self, tmp_path):
        _spawn(_gc_fsdp_all_fsdp_rbd_worker, tmp_path)


# ---------------------------------------------------------------------------
# GC memory savings verification
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGradientCheckpointingMemorySavings:
    """Verify gradient checkpointing actually reduces peak GPU memory.

    Uses a larger model to make the memory difference measurable.
    A regression that silently disables GC would cause this test to fail.
    """

    def test_blockwise_gc_reduces_peak_memory(self, tmp_path):
        """BlockwiseDistiller: GC=True uses less peak memory than GC=False.

        Uses a model with 4x-expansion FFN blocks (like a real transformer)
        so that each layer creates large intermediate activations.  With the
        test-suite's SimpleBlock (two Linears + LayerNorm) the intermediates
        are tiny and GC saves almost nothing.

        With 12 layers of hidden_dim=1024 and ffn_dim=4096, batch=32,
        seq_len=64, each layer's forward creates ~32MB of intermediates —
        without GC all 12 are kept (~384MB), with GC they are recomputed.
        """
        from silverspoon_kd import BlockwiseDistiller, create_alignments
        from silverspoon_kd.training_arguments import TrainingArguments
        from tests.silverspoon_kd.conftest import DummyDataset

        device = torch.device("cuda:0")
        num_layers = 12
        hidden_dim = 1024
        ffn_dim = hidden_dim * 4

        class _FFNBlock(torch.nn.Module):
            """Block with 4x FFN expansion — creates large intermediates."""

            def __init__(self, dim, ffn):
                super().__init__()
                self.up = torch.nn.Linear(dim, ffn)
                self.down = torch.nn.Linear(ffn, dim)
                self.norm = torch.nn.LayerNorm(dim)

            def forward(self, x):
                return self.norm(x + self.down(torch.nn.functional.gelu(self.up(x))))

        class _GCTestModel(torch.nn.Module):
            def __init__(self, dim, ffn, n_layers):
                super().__init__()
                self.embedding = torch.nn.Embedding(128, dim)
                self.layers = torch.nn.ModuleList([_FFNBlock(dim, ffn) for _ in range(n_layers)])
                self.lm_head = torch.nn.Linear(dim, 128)

            def forward(self, input_ids, **kwargs):
                x = self.embedding(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        def _measure_peak_memory(gc_enabled):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

            teacher = _GCTestModel(hidden_dim, ffn_dim, num_layers).to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            student = _GCTestModel(hidden_dim, ffn_dim, num_layers).to(device)

            alignments = create_alignments(
                teacher_model=teacher,
                student_model=student,
                modules=r"layers\.\d+$",
                output_selector_index=None,
                auto_projector=True,
            )

            args = TrainingArguments(
                output_dir=str(tmp_path / f"gc_{gc_enabled}"),
                max_steps=3,
                per_device_train_batch_size=32,
                gradient_checkpointing=gc_enabled,
                logging_steps=1,
                save_strategy="no",
                report_to=[],
            )
            args._n_gpu = 1

            distiller = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=DummyDataset(num_samples=128, seq_len=64),
            )
            distiller.train()

            peak = torch.cuda.max_memory_allocated(device)

            del distiller, teacher, student, alignments
            torch.cuda.empty_cache()

            return peak

        peak_no_gc = _measure_peak_memory(gc_enabled=False)
        peak_gc = _measure_peak_memory(gc_enabled=True)

        assert peak_gc < peak_no_gc * 0.90, (
            f"GC did not reduce peak memory by at least 10%: "
            f"no_gc={peak_no_gc / 1024**2:.1f}MB, gc={peak_gc / 1024**2:.1f}MB"
        )
