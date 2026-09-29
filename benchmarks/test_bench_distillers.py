"""Speed benchmarks for all distiller types.

Measures training throughput, evaluation time, and the impact of key
configuration parameters (chunk sizes, model sizes, profiling).
"""

import pytest
from bench_utils import (
    DummyDataset,
    create_distiller,
    make_alignments,
    make_models,
    make_training_args,
)

from silverspoon_kd.distillers import (
    BlockwiseDistiller,
    HolisticDistiller,
    ResponseBasedDistiller,
)

# ─── Model size configurations ────────────────────────────────────────

SMALL = {
    "teacher_dim": 128,
    "student_dim": 64,
    "num_layers": 2,
    "batch_size": 2,
    "seq_len": 16,
    "num_samples": 20,
    "max_steps": 5,
}

MEDIUM = {
    "teacher_dim": 256,
    "student_dim": 128,
    "num_layers": 3,
    "batch_size": 4,
    "seq_len": 32,
    "num_samples": 40,
    "max_steps": 5,
}

CONFIGS = [
    pytest.param(SMALL, id="small"),
    pytest.param(MEDIUM, id="medium"),
]


# ═══════════════════════════════════════════════════════════════════════
#  BlockwiseDistiller
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="distillers-blockwise")
class TestBlockwiseDistillerSpeed:
    @pytest.mark.parametrize("config", CONFIGS)
    def test_train(self, benchmark, device, tmp_path, config):
        """End-to-end training throughput."""

        def run():
            distiller = create_distiller(
                "blockwise",
                device,
                tmp_path,
                **config,
            )
            distiller.train()

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)

    def test_evaluate(self, benchmark, device, tmp_path):
        """Evaluation pass after training."""

        def run():
            eval_dataset = DummyDataset(10, 16)
            distiller = create_distiller(
                "blockwise",
                device,
                tmp_path,
                max_steps=3,
            )
            distiller.train()
            distiller.evaluate(eval_dataset=eval_dataset)

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)


# ═══════════════════════════════════════════════════════════════════════
#  HolisticDistiller
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="distillers-holistic")
class TestHolisticDistillerSpeed:
    @pytest.mark.parametrize("config", CONFIGS)
    def test_train(self, benchmark, device, tmp_path, config):
        """End-to-end training throughput."""

        def run():
            distiller = create_distiller(
                "holistic",
                device,
                tmp_path,
                **config,
            )
            distiller.train()

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)

    def test_evaluate(self, benchmark, device, tmp_path):
        """Evaluation pass after training."""

        def run():
            eval_dataset = DummyDataset(10, 16)
            distiller = create_distiller(
                "holistic",
                device,
                tmp_path,
                max_steps=3,
            )
            distiller.train()
            distiller.evaluate(eval_dataset=eval_dataset)

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)


# ═══════════════════════════════════════════════════════════════════════
#  ResponseBasedDistiller
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="distillers-response")
class TestResponseBasedDistillerSpeed:
    @pytest.mark.parametrize("config", CONFIGS)
    def test_train(self, benchmark, device, tmp_path, config):
        """End-to-end training throughput."""

        def run():
            distiller = create_distiller(
                "response_based",
                device,
                tmp_path,
                **config,
            )
            distiller.train()

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)

    def test_evaluate(self, benchmark, device, tmp_path):
        """Evaluation pass after training."""

        def run():
            eval_dataset = DummyDataset(10, 16)
            distiller = create_distiller(
                "response_based",
                device,
                tmp_path,
                max_steps=3,
            )
            distiller.train()
            distiller.evaluate(eval_dataset=eval_dataset)

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)

    @pytest.mark.parametrize("chunk_size", [0, 256, 1024])
    def test_chunk_sizes(self, benchmark, device, tmp_path, chunk_size):
        """Impact of KL divergence chunk size on training speed."""
        from silverspoon_kd.losses.kl import kl_divergence_loss

        def run():
            teacher, student = make_models(device, 128, 64, 3)
            args = make_training_args(
                tmp_path,
                device,
                max_steps=5,
                batch_size=2,
            )
            dataset = DummyDataset(20, 16)
            distiller = ResponseBasedDistiller(
                student_model=student,
                teacher_model=teacher,
                args=args,
                train_dataset=dataset,
                soft_loss_fn=kl_divergence_loss(chunk_size=chunk_size),
            )
            distiller.train()

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)


# ═══════════════════════════════════════════════════════════════════════
#  Cross-Distiller Comparisons
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="distillers-comparison")
class TestDistillerComparison:
    """Same-config benchmarks across all distiller types for direct comparison."""

    @pytest.mark.parametrize(
        "distiller_type",
        [
            pytest.param("blockwise", id="blockwise"),
            pytest.param("holistic", id="holistic"),
            pytest.param("response_based", id="response_based"),
        ],
    )
    def test_train_uniform_config(self, benchmark, device, tmp_path, distiller_type):
        """All distillers with identical model/data configuration."""

        def run():
            distiller = create_distiller(
                distiller_type,
                device,
                tmp_path,
                teacher_dim=128,
                student_dim=64,
                num_layers=2,
                batch_size=2,
                seq_len=16,
                num_samples=20,
                max_steps=5,
            )
            distiller.train()

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)


# ═══════════════════════════════════════════════════════════════════════
#  Profiling Overhead
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="distillers-profiling")
class TestDistillerProfilingOverhead:
    """Measure the overhead profiling adds to training."""

    @pytest.mark.parametrize(
        "distiller_type",
        [
            pytest.param("blockwise", id="blockwise"),
            pytest.param("holistic", id="holistic"),
            pytest.param("response_based", id="response_based"),
        ],
    )
    def test_train_with_profiling(self, benchmark, device, tmp_path, distiller_type):
        """Training with profiling enabled — compare to test_train_uniform_config."""

        def run():
            teacher, student = make_models(device, 128, 64, 2)
            alignments = make_alignments(teacher, student)
            args = make_training_args(
                tmp_path,
                device,
                max_steps=5,
                batch_size=2,
                enable_profiling=True,
                profiling_wait=0,
                profiling_warmup=1,
                profiling_active=4,
                profiling_repeat=1,
            )
            dataset = DummyDataset(20, 16)

            if distiller_type == "blockwise":
                distiller = BlockwiseDistiller(
                    teacher_model=teacher,
                    alignments=alignments,
                    args=args,
                    train_dataset=dataset,
                )
            elif distiller_type == "holistic":
                distiller = HolisticDistiller(
                    student_model=student,
                    teacher_model=teacher,
                    alignments=alignments,
                    args=args,
                    train_dataset=dataset,
                )
            elif distiller_type == "response_based":
                distiller = ResponseBasedDistiller(
                    student_model=student,
                    teacher_model=teacher,
                    args=args,
                    train_dataset=dataset,
                )
            distiller.train()

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)
