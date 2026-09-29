"""Benchmarks for teacher placement strategies.

Measures overhead of different teacher placement strategies compared to
baseline single-GPU training.

Requires 2+ CUDA GPUs.
"""

import pytest
import torch

from silverspoon_kd.distributed.strategies import (
    place_teacher_pp,
    place_teacher_replicated,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="At least 2 GPUs required for benchmarks",
    ),
]


def _make_teacher_and_student():
    from tests.silverspoon_kd.conftest import SimpleModel

    teacher = SimpleModel(64, 256, 6)
    student = SimpleModel(64, 128, 6)
    return teacher, student


def _forward_pass(model, device, n_steps=20):
    """Run n forward passes and return elapsed time."""
    batch = torch.randint(0, 1000, (4, 32), device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(batch)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(n_steps):
            model(batch)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


class TestTeacherPlacementOverhead:
    """Benchmark teacher placement strategies."""

    def test_replicated_baseline(self, benchmark):
        """Baseline: replicated teacher on single GPU."""
        teacher, _ = _make_teacher_and_student()
        place_teacher_replicated(teacher, torch.device("cuda:0"))

        def run():
            return _forward_pass(teacher, "cuda:0")

        benchmark(run)

    def test_pp_two_gpus(self, benchmark):
        """PP teacher across 2 GPUs."""
        teacher, _ = _make_teacher_and_student()
        place_teacher_pp(teacher, [0, 1], "cuda")

        def run():
            device = next(teacher.parameters()).device
            return _forward_pass(teacher, device)

        benchmark(run)

    def test_student_forward_with_replicated_teacher(self, benchmark):
        """Student forward when teacher is replicated."""
        teacher, student = _make_teacher_and_student()
        place_teacher_replicated(teacher, torch.device("cuda:0"))
        student.to("cuda:0")

        def run():
            batch = torch.randint(0, 1000, (4, 32), device="cuda:0")
            with torch.no_grad():
                teacher(batch)
            student(batch)
            return True

        benchmark(run)

    def test_student_forward_with_pp_teacher(self, benchmark):
        """Student forward when teacher is PP across 2 GPUs."""
        teacher, student = _make_teacher_and_student()
        place_teacher_pp(teacher, [0, 1], "cuda")
        student.to("cuda:0")

        def run():
            device = next(teacher.parameters()).device
            batch = torch.randint(0, 1000, (4, 32), device=device)
            with torch.no_grad():
                teacher(batch)
            batch_student = batch.to("cuda:0")
            student(batch_student)
            return True

        benchmark(run)
