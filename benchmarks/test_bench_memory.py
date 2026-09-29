"""Memory benchmarks for silverspoon-kd distillers.

Measures peak CPU memory (via tracemalloc) and peak GPU memory
(via torch.cuda) during training. Results are recorded via pytest's
record_property for inclusion in JUnit XML and CI reports.
"""

import pytest
from bench_utils import DummyDataset, MemoryTracker, create_distiller

# ═══════════════════════════════════════════════════════════════════════
#  Per-Distiller Peak Memory
# ═══════════════════════════════════════════════════════════════════════


class TestDistillerMemory:
    """Peak memory usage during a short training run for each distiller type."""

    @pytest.mark.parametrize(
        "distiller_type",
        [
            pytest.param("blockwise", id="blockwise"),
            pytest.param("holistic", id="holistic"),
            pytest.param("response_based", id="response_based"),
        ],
    )
    def test_training_peak_memory(self, distiller_type, device, tmp_path, record_property):
        distiller = create_distiller(
            distiller_type,
            device,
            tmp_path,
            max_steps=3,
            batch_size=2,
            teacher_dim=128,
            student_dim=64,
            num_layers=3,
        )

        with MemoryTracker() as mem:
            distiller.train()

        record_property("peak_cpu_mb", round(mem.peak_cpu_mb, 2))
        record_property("peak_gpu_mb", round(mem.peak_gpu_mb, 2))

        # Sanity check: training used some memory
        assert mem.peak_cpu_mb > 0

    @pytest.mark.parametrize(
        "distiller_type",
        [
            pytest.param("blockwise", id="blockwise"),
            pytest.param("holistic", id="holistic"),
            pytest.param("response_based", id="response_based"),
        ],
    )
    def test_evaluation_peak_memory(self, distiller_type, device, tmp_path, record_property):
        eval_dataset = DummyDataset(10, 16)
        distiller = create_distiller(
            distiller_type,
            device,
            tmp_path,
            max_steps=3,
            batch_size=2,
            teacher_dim=128,
            student_dim=64,
            num_layers=3,
        )
        distiller.train()

        with MemoryTracker() as mem:
            distiller.evaluate(eval_dataset=eval_dataset)

        record_property("eval_peak_cpu_mb", round(mem.peak_cpu_mb, 2))
        record_property("eval_peak_gpu_mb", round(mem.peak_gpu_mb, 2))

        assert mem.peak_cpu_mb > 0


# ═══════════════════════════════════════════════════════════════════════
#  Memory Scaling: Batch Size
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryScalingBatchSize:
    """How peak memory scales with increasing batch size."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_blockwise_batch_scaling(self, device, tmp_path, batch_size, record_property):
        distiller = create_distiller(
            "blockwise",
            device,
            tmp_path,
            max_steps=3,
            batch_size=batch_size,
            teacher_dim=128,
            student_dim=64,
            num_layers=3,
        )

        with MemoryTracker() as mem:
            distiller.train()

        record_property(f"peak_cpu_mb_bs{batch_size}", round(mem.peak_cpu_mb, 2))
        record_property(f"peak_gpu_mb_bs{batch_size}", round(mem.peak_gpu_mb, 2))

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_response_based_batch_scaling(self, device, tmp_path, batch_size, record_property):
        distiller = create_distiller(
            "response_based",
            device,
            tmp_path,
            max_steps=3,
            batch_size=batch_size,
            teacher_dim=128,
            student_dim=64,
            num_layers=3,
        )

        with MemoryTracker() as mem:
            distiller.train()

        record_property(f"peak_cpu_mb_bs{batch_size}", round(mem.peak_cpu_mb, 2))
        record_property(f"peak_gpu_mb_bs{batch_size}", round(mem.peak_gpu_mb, 2))


# ═══════════════════════════════════════════════════════════════════════
#  Memory Scaling: Model Size (layers)
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryScalingModelSize:
    """How peak memory scales with increasing model depth."""

    @pytest.mark.parametrize("num_layers", [2, 4, 6])
    def test_blockwise_layer_scaling(self, device, tmp_path, num_layers, record_property):
        distiller = create_distiller(
            "blockwise",
            device,
            tmp_path,
            max_steps=3,
            batch_size=2,
            num_layers=num_layers,
            teacher_dim=128,
            student_dim=64,
        )

        with MemoryTracker() as mem:
            distiller.train()

        record_property(f"peak_cpu_mb_{num_layers}layers", round(mem.peak_cpu_mb, 2))
        record_property(f"peak_gpu_mb_{num_layers}layers", round(mem.peak_gpu_mb, 2))

    @pytest.mark.parametrize("num_layers", [2, 4, 6])
    def test_holistic_layer_scaling(self, device, tmp_path, num_layers, record_property):
        distiller = create_distiller(
            "holistic",
            device,
            tmp_path,
            max_steps=3,
            batch_size=2,
            num_layers=num_layers,
            teacher_dim=128,
            student_dim=64,
        )

        with MemoryTracker() as mem:
            distiller.train()

        record_property(f"peak_cpu_mb_{num_layers}layers", round(mem.peak_cpu_mb, 2))
        record_property(f"peak_gpu_mb_{num_layers}layers", round(mem.peak_gpu_mb, 2))


# ═══════════════════════════════════════════════════════════════════════
#  Memory Scaling: Hidden Dimension
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryScalingHiddenDim:
    """How peak memory scales with increasing hidden dimension."""

    @pytest.mark.parametrize(
        "dims",
        [
            pytest.param((64, 32), id="64-32"),
            pytest.param((128, 64), id="128-64"),
            pytest.param((256, 128), id="256-128"),
        ],
    )
    def test_blockwise_dim_scaling(self, device, tmp_path, dims, record_property):
        teacher_dim, student_dim = dims
        distiller = create_distiller(
            "blockwise",
            device,
            tmp_path,
            max_steps=3,
            batch_size=2,
            num_layers=3,
            teacher_dim=teacher_dim,
            student_dim=student_dim,
        )

        with MemoryTracker() as mem:
            distiller.train()

        record_property(f"peak_cpu_mb_t{teacher_dim}s{student_dim}", round(mem.peak_cpu_mb, 2))
        record_property(f"peak_gpu_mb_t{teacher_dim}s{student_dim}", round(mem.peak_gpu_mb, 2))
