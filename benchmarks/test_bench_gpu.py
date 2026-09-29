"""GPU-specific benchmarks for silverspoon-kd knowledge distillation.

These benchmarks only run under ``make benchmark`` (--device=cuda).
They test scenarios where GPU behavior meaningfully diverges from CPU:
reduced-precision (bf16) training, GPU memory scaling, CUDA profiling
overhead, and capture engine hook overhead on GPU.
"""

import pytest
import torch
from bench_utils import (
    DummyDataset,
    MemoryTracker,
    SimpleModel,
    make_alignments,
    make_models,
    make_training_args,
)

from silverspoon_kd.distillers import (
    BlockwiseDistiller,
    HolisticDistiller,
    ResponseBasedDistiller,
)
from silverspoon_kd.engines.module_capture_engine import ModuleCaptureEngine

# Applied to every test in this file — auto-skipped under --device=cpu
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def _make_batch(device, batch_size=8, seq_len=64):
    """Create a random batch on the given device."""
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Shared parametrize values
# ═══════════════════════════════════════════════════════════════════════

# The distillers run models in their native parameter dtype (their custom
# training_step bypasses the Trainer's autocast/GradScaler path), so precision
# is selected by casting the models before the distiller is built.
PRECISION_DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, id="bf16"),
]

ALL_DISTILLER_TYPES = [
    pytest.param("blockwise", id="blockwise"),
    pytest.param("holistic", id="holistic"),
    pytest.param("response_based", id="response_based"),
]

FEATURE_DISTILLER_TYPES = [
    pytest.param("blockwise", id="blockwise"),
    pytest.param("holistic", id="holistic"),
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


# ═══════════════════════════════════════════════════════════════════════
#  Group 1: Mixed-Precision Training Throughput
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="gpu-mixed-precision")
class TestMixedPrecisionTraining:
    """Training throughput at fp32 and bf16 for every distiller type.

    Teacher and student are cast to the target dtype before the distiller
    is built; the projectors created by ``make_alignments`` follow the
    student's dtype.
    """

    @pytest.mark.parametrize("dtype", PRECISION_DTYPES)
    @pytest.mark.parametrize("distiller_type", ALL_DISTILLER_TYPES)
    def test_train_precision(self, benchmark, device, tmp_path, distiller_type, dtype):
        """Measure training throughput at fp32/bf16."""

        def run():
            teacher, student = make_models(device, 128, 64, 3, dtype=dtype)
            alignments = make_alignments(teacher, student)
            args = make_training_args(tmp_path, device, max_steps=5, batch_size=4)
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


# ═══════════════════════════════════════════════════════════════════════
#  Group 2: GPU Memory Scaling
# ═══════════════════════════════════════════════════════════════════════

GPU_MEMORY_CONFIGS = [
    pytest.param(
        {
            "teacher_dim": 128,
            "student_dim": 64,
            "num_layers": 3,
            "batch_size": 2,
            "seq_len": 32,
        },
        id="base",
    ),
    pytest.param(
        {
            "teacher_dim": 128,
            "student_dim": 64,
            "num_layers": 3,
            "batch_size": 8,
            "seq_len": 32,
        },
        id="large-batch",
    ),
    pytest.param(
        {
            "teacher_dim": 128,
            "student_dim": 64,
            "num_layers": 3,
            "batch_size": 2,
            "seq_len": 128,
        },
        id="long-seq",
    ),
    pytest.param(
        {
            "teacher_dim": 256,
            "student_dim": 128,
            "num_layers": 6,
            "batch_size": 2,
            "seq_len": 32,
        },
        id="large-model",
    ),
    pytest.param(
        {
            "teacher_dim": 256,
            "student_dim": 128,
            "num_layers": 6,
            "batch_size": 8,
            "seq_len": 128,
        },
        id="stress",
    ),
]


@pytest.mark.benchmark(group="gpu-memory-scaling")
class TestGPUMemoryScaling:
    """Peak GPU memory across model sizes, batch sizes, and sequence lengths.

    This is the primary constraint users hit on GPU.  Each test reports
    peak_gpu_mb via record_property for CI tracking.
    """

    @pytest.mark.parametrize("config", GPU_MEMORY_CONFIGS)
    @pytest.mark.parametrize("distiller_type", FEATURE_DISTILLER_TYPES)
    def test_training_peak_gpu_memory(
        self, benchmark, device, tmp_path, distiller_type, config, record_property
    ):
        """Peak GPU memory during a short training run."""

        def run():
            teacher, student = make_models(
                device,
                teacher_dim=config["teacher_dim"],
                student_dim=config["student_dim"],
                num_layers=config["num_layers"],
            )
            alignments = make_alignments(teacher, student)
            args = make_training_args(
                tmp_path,
                device,
                max_steps=3,
                batch_size=config["batch_size"],
            )
            dataset = DummyDataset(
                num_samples=max(20, config["batch_size"] * 5),
                seq_len=config["seq_len"],
            )
            distiller = _build_distiller(
                distiller_type,
                teacher,
                student,
                alignments,
                args,
                dataset,
            )

            with MemoryTracker() as mem:
                distiller.train()

            record_property("peak_gpu_mb", round(mem.peak_gpu_mb, 2))

        benchmark.pedantic(run, rounds=2, warmup_rounds=1)

    @pytest.mark.parametrize("dtype", PRECISION_DTYPES)
    def test_precision_memory_impact(self, benchmark, device, tmp_path, dtype, record_property):
        """Compare GPU memory between fp32 and bf16 for the same workload."""

        def run():
            teacher, student = make_models(device, 256, 128, 4, dtype=dtype)
            alignments = make_alignments(teacher, student)
            args = make_training_args(tmp_path, device, max_steps=3, batch_size=4)
            dataset = DummyDataset(40, 64)
            distiller = HolisticDistiller(
                student_model=student,
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
            )

            with MemoryTracker() as mem:
                distiller.train()

            record_property("peak_gpu_mb", round(mem.peak_gpu_mb, 2))

        benchmark.pedantic(run, rounds=2, warmup_rounds=1)


# ═══════════════════════════════════════════════════════════════════════
#  Group 3: CUDA Profiling Overhead
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="gpu-profiling-overhead")
class TestCUDAProfilingOverhead:
    """Measure the real cost of torch.profiler with CUDA activity tracing.

    CUDA profiling inserts synchronization barriers that force the GPU to
    drain its work queue.  This measures the actual overhead users pay when
    they enable profiling on GPU, compared to the baseline without profiling.
    """

    @pytest.mark.parametrize("distiller_type", FEATURE_DISTILLER_TYPES)
    def test_baseline_no_profiling(self, benchmark, device, tmp_path, distiller_type):
        """Baseline: training without profiling (for comparison)."""

        def run():
            teacher, student = make_models(device, 128, 64, 3)
            alignments = make_alignments(teacher, student)
            args = make_training_args(
                tmp_path,
                device,
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

    @pytest.mark.parametrize("distiller_type", FEATURE_DISTILLER_TYPES)
    def test_with_cuda_profiling(self, benchmark, device, tmp_path, distiller_type):
        """Training with CUDA profiling enabled (synchronization barriers active)."""

        def run():
            teacher, student = make_models(device, 128, 64, 3)
            alignments = make_alignments(teacher, student)
            args = make_training_args(
                tmp_path,
                device,
                max_steps=5,
                batch_size=4,
                enable_profiling=True,
                profiling_wait=0,
                profiling_warmup=1,
                profiling_active=4,
                profiling_repeat=1,
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


# ═══════════════════════════════════════════════════════════════════════
#  Group 4: Capture Engine Hook Overhead on GPU
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="gpu-capture-engine")
class TestGPUCaptureEngineOverhead:
    """Capture engine hook overhead specifically on GPU.

    Forward hooks interact with CUDA's async execution model.  On GPU,
    hooks break kernel pipelining and detach/deepcopy operations involve
    GPU memory copies.  This measures the real GPU-specific overhead.
    """

    def test_forward_no_engine_baseline(self, benchmark, device):
        """Baseline: forward pass with no capture engine."""
        model = SimpleModel(hidden_dim=128, num_layers=4).to(device)
        batch = _make_batch(device, batch_size=8, seq_len=64)

        def run():
            torch.cuda.synchronize()
            with torch.no_grad():
                model(**batch)
            torch.cuda.synchronize()

        benchmark(run)

    def test_forward_detach_mode(self, benchmark, device):
        """Capture with detached outputs (default mode)."""
        model = SimpleModel(hidden_dim=128, num_layers=4).to(device)
        modules = list(model.layers)
        engine = ModuleCaptureEngine(
            model=model,
            modules_to_capture=modules,
            detach_outputs=True,
        )
        engine.register()
        batch = _make_batch(device, batch_size=8, seq_len=64)

        def run():
            torch.cuda.synchronize()
            with torch.no_grad():
                model(**batch)
            torch.cuda.synchronize()
            engine.clear_captured()

        benchmark(run)
        engine.deregister()

    def test_forward_deepcopy_mode(self, benchmark, device):
        """Capture with deepcopy (expensive on GPU — involves GPU memory copies)."""
        model = SimpleModel(hidden_dim=128, num_layers=4).to(device)
        modules = list(model.layers)
        engine = ModuleCaptureEngine(
            model=model,
            modules_to_capture=modules,
            deepcopy_captured_args_and_kwargs=True,
        )
        engine.register()
        batch = _make_batch(device, batch_size=8, seq_len=64)

        def run():
            torch.cuda.synchronize()
            with torch.no_grad():
                model(**batch)
            torch.cuda.synchronize()
            engine.clear_captured()

        benchmark(run)
        engine.deregister()

    def test_forward_with_callback(self, benchmark, device):
        """Capture with output callback (Python callback per hook on GPU)."""
        model = SimpleModel(hidden_dim=128, num_layers=4).to(device)
        modules = list(model.layers)

        def noop_callback(module_id, inp, out):
            pass

        engine = ModuleCaptureEngine(
            model=model,
            modules_to_capture=modules,
            output_callback=noop_callback,
        )
        engine.register()
        batch = _make_batch(device, batch_size=8, seq_len=64)

        def run():
            torch.cuda.synchronize()
            with torch.no_grad():
                model(**batch)
            torch.cuda.synchronize()
            engine.clear_captured()

        benchmark(run)
        engine.deregister()

    @pytest.mark.parametrize("num_layers", [2, 4, 8])
    def test_hook_scaling_with_depth(self, benchmark, device, num_layers):
        """How hook overhead scales with model depth (more hooks = more sync points)."""
        model = SimpleModel(hidden_dim=128, num_layers=num_layers).to(device)
        modules = list(model.layers)
        engine = ModuleCaptureEngine(
            model=model,
            modules_to_capture=modules,
            detach_outputs=True,
        )
        engine.register()
        batch = _make_batch(device, batch_size=8, seq_len=64)

        def run():
            torch.cuda.synchronize()
            with torch.no_grad():
                model(**batch)
            torch.cuda.synchronize()
            engine.clear_captured()

        benchmark(run)
        engine.deregister()
