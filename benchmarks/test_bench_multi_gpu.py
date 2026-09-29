"""Multi-GPU benchmarks for silverspoon-kd knowledge distillation.

Measures the overhead of DataParallel wrapping on each distiller type.
When ``_n_gpu > 1`` the Trainer wraps the model in ``torch.nn.DataParallel``.
The base distiller template uses ``self.model`` (unwrapped) to avoid
duplicating capture hooks, so DP wrapping should add minimal overhead.

Requires at least 2 CUDA GPUs with >4GB each (``--device=cuda``).
"""

import pytest
import torch
from bench_utils import (
    DummyDataset,
    SimpleModel,
    make_alignments,
    make_training_args,
)

from silverspoon_kd.distillers import (
    BlockwiseDistiller,
    HolisticDistiller,
    ResponseBasedDistiller,
)


def _real_gpu_ids():
    """Return list of GPU ids with >4GB memory (filtering display-only GPUs)."""
    return [
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_properties(i).total_memory > 4 * 1024**3
    ]


# All tests require CUDA + at least 2 real GPUs
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(len(_real_gpu_ids()) < 2, reason="At least 2 GPUs required"),
]


def _build_distiller(distiller_type, teacher, student, alignments, args, dataset):
    """Construct a distiller by type string."""
    if distiller_type == "blockwise":
        return BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=dataset,
        )
    elif distiller_type == "holistic":
        return HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=dataset,
        )
    elif distiller_type == "response_based":
        return ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=dataset,
        )
    raise ValueError(f"Unknown distiller type: {distiller_type}")


# =======================================================================
#  DataParallel Overhead
# =======================================================================


@pytest.mark.benchmark(group="multi-gpu-dataparallel-overhead")
class TestDataParallelOverhead:
    """Measure overhead of DataParallel wrapping on distillers.

    The base template uses self.model (unwrapped) to avoid capture engine
    issues with DataParallel, so DP wrapping should add minimal overhead.
    This benchmark quantifies that overhead.
    """

    DISTILLER_TYPES = [
        pytest.param("holistic", id="holistic"),
        pytest.param("blockwise", id="blockwise"),
        pytest.param("response_based", id="response_based"),
    ]

    @pytest.mark.parametrize("distiller_type", DISTILLER_TYPES)
    def test_single_gpu_baseline(self, benchmark, tmp_path, distiller_type):
        """Baseline: _n_gpu=1 (no DataParallel wrapping)."""
        gpus = _real_gpu_ids()
        gpu0 = torch.device(f"cuda:{gpus[0]}")

        def run():
            teacher = SimpleModel(64, 128, 3).to(gpu0)
            teacher.eval()
            student = SimpleModel(64, 64, 3).to(gpu0)
            alignments = make_alignments(teacher, student)

            args = make_training_args(
                tmp_path,
                gpu0,
                max_steps=5,
                batch_size=4,
            )
            dataset = DummyDataset(40, 32)

            distiller = _build_distiller(
                distiller_type,
                teacher,
                student,
                alignments,
                args,
                dataset,
            )
            distiller.train()

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)

    @pytest.mark.parametrize("distiller_type", DISTILLER_TYPES)
    def test_dataparallel_enabled(self, benchmark, tmp_path, distiller_type):
        """With DataParallel wrapping enabled (_n_gpu > 1)."""
        gpus = _real_gpu_ids()
        gpu0 = torch.device(f"cuda:{gpus[0]}")

        def run():
            teacher = SimpleModel(64, 128, 3).to(gpu0)
            teacher.eval()
            student = SimpleModel(64, 64, 3).to(gpu0)
            alignments = make_alignments(teacher, student)

            args = make_training_args(
                tmp_path,
                gpu0,
                max_steps=5,
                batch_size=4,
            )
            args._n_gpu = len(gpus)
            dataset = DummyDataset(40, 32)

            distiller = _build_distiller(
                distiller_type,
                teacher,
                student,
                alignments,
                args,
                dataset,
            )
            distiller.train()

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)
