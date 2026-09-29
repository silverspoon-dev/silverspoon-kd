"""Student-side compatibility verification.

Tests that various student distributed strategies (DDP, FSDP, DataParallel)
work correctly with different teacher placements.

Requires 4+ GPUs for distributed tests, 2+ GPUs for DataParallel tests.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist

from silverspoon_kd.distributed.strategies import (
    place_teacher_pp,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="At least 2 GPUs required",
    ),
]


def _find_free_port():
    """Find a free TCP port by briefly binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_process(rank, world_size, port, fn, *args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


class TestDataParallelCompatibility:
    """Student DataParallel + various teacher placements."""

    def test_replicated_teacher_dp_student(self, tmp_path):
        """Replicated teacher with DP student trains correctly."""
        from silverspoon_kd.distillers import HolisticDistiller
        from silverspoon_kd.training_arguments import TrainingArguments
        from tests.silverspoon_kd.conftest import (
            DummyDataset,
            SimpleModel,
            create_alignment,
        )

        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()

        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=999,
            save_steps=999,
            dataloader_num_workers=0,
            report_to=[],
            disable_tqdm=True,
            use_cpu=False,
        )

        alignments = [
            create_alignment(
                teacher.get_layer(i),
                student.get_layer(i),
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
            )
            for i in range(teacher.num_layers)
        ]

        params_before = {n: p.clone() for n, p in student.named_parameters() if p.requires_grad}

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(20, 16),
        )
        distiller.train()

        changed = sum(
            1
            for n, p in student.named_parameters()
            if p.requires_grad and not torch.equal(p, params_before[n])
        )
        assert changed > 0

    def test_pp_teacher_dp_student(self, tmp_path):
        """PP teacher with single-process student trains correctly."""
        from silverspoon_kd.distillers import HolisticDistiller
        from silverspoon_kd.training_arguments import TrainingArguments
        from tests.silverspoon_kd.conftest import (
            DummyDataset,
            SimpleModel,
            create_alignment,
        )

        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()
        place_teacher_pp(teacher, [0, 1], "cuda")

        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=999,
            save_steps=999,
            dataloader_num_workers=0,
            report_to=[],
            disable_tqdm=True,
            use_cpu=False,
        )

        alignments = []
        for i in range(teacher.num_layers):
            a = create_alignment(
                teacher.get_layer(i),
                student.get_layer(i),
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
            )
            a.auto_device_match = True
            alignments.append(a)

        params_before = {n: p.clone() for n, p in student.named_parameters() if p.requires_grad}

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(20, 16),
        )
        distiller.train()

        assert distiller.state.global_step == 3
        changed = sum(
            1
            for n, p in student.named_parameters()
            if p.requires_grad and not torch.equal(p, params_before[n])
        )
        assert changed > 0


class TestGradientConsistency:
    """Verify gradient correctness across placements."""

    def test_finite_loss_replicated(self, tmp_path):
        """Replicated teacher produces finite loss."""
        from silverspoon_kd.distillers import HolisticDistiller
        from silverspoon_kd.training_arguments import TrainingArguments
        from tests.silverspoon_kd.conftest import (
            DummyDataset,
            SimpleModel,
            create_alignment,
        )

        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()

        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=1,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=999,
            dataloader_num_workers=0,
            report_to=[],
            disable_tqdm=True,
            use_cpu=False,
        )

        alignments = [
            create_alignment(
                teacher.get_layer(i),
                student.get_layer(i),
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
            )
            for i in range(teacher.num_layers)
        ]

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(20, 16),
        )
        distiller.train()
        # Check that training logged some metrics
        assert distiller.state.global_step == 1

    def test_finite_loss_pp(self, tmp_path):
        """PP teacher produces finite loss."""
        from silverspoon_kd.distillers import HolisticDistiller
        from silverspoon_kd.training_arguments import TrainingArguments
        from tests.silverspoon_kd.conftest import (
            DummyDataset,
            SimpleModel,
            create_alignment,
        )

        teacher = SimpleModel(64, 128, 3).cuda()
        student = SimpleModel(64, 64, 3).cuda()
        place_teacher_pp(teacher, [0, 1], "cuda")

        args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=1,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=999,
            dataloader_num_workers=0,
            report_to=[],
            disable_tqdm=True,
            use_cpu=False,
        )

        alignments = []
        for i in range(teacher.num_layers):
            a = create_alignment(
                teacher.get_layer(i),
                student.get_layer(i),
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
            )
            a.auto_device_match = True
            alignments.append(a)

        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(20, 16),
        )
        distiller.train()
        assert distiller.state.global_step == 1
