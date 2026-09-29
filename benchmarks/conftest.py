"""Benchmark conftest: device fixture and bench_utils import path."""

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# Add this directory to sys.path so test files can import bench_utils
sys.path.insert(0, str(Path(__file__).parent))


def pytest_addoption(parser):
    # ``--device`` may already be registered by ``tests/conftest.py``
    # when pytest collects both the benchmarks and tests directories in
    # the same run; suppressing the ValueError lets this conftest work
    # standalone or alongside the tests conftest.
    with contextlib.suppress(ValueError):
        parser.addoption(
            "--device",
            action="store",
            default="cpu",
            choices=["cpu", "cuda"],
            help="Device to run benchmarks on: cpu (default) or cuda",
        )
    parser.addoption(
        "--skip-env-checks",
        action="store_true",
        default=False,
        help="Skip environment checks for CPU/GPU utilization before benchmarking",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--device") != "cuda":
        skip_cuda = pytest.mark.skip(reason="CUDA benchmarks require --device=cuda")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


def _check_system_utilization(device_option):
    """Check if CPU or GPU is under significant load.

    Returns a list of warning messages. Empty list means the system is idle.
    """
    issues = []

    # Check CPU load
    try:
        load_1min = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        cpu_usage = load_1min / cpu_count
        if cpu_usage > 0.3:
            issues.append(
                f"High CPU load detected: {load_1min:.1f} ({cpu_usage:.0%} of {cpu_count} cores)."
            )
    except OSError:
        pass

    # Check GPU utilization
    if device_option == "cuda":
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    idx, gpu_util, mem_used, mem_total = line.split(", ")
                    gpu_util = float(gpu_util)
                    mem_pct = float(mem_used) / float(mem_total) * 100
                    if gpu_util > 0 or float(mem_used) > 500:
                        issues.append(
                            f"GPU {idx} is busy: {gpu_util:.0f}% utilization, "
                            f"{mem_pct:.0f}% memory used ({mem_used}/{mem_total} MiB)."
                        )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return issues


def pytest_sessionstart(session):
    """Check system utilization before running benchmarks."""
    if session.config.getoption("--skip-env-checks", default=False):
        return

    device_option = session.config.getoption("--device", default="cpu")
    issues = _check_system_utilization(device_option)
    if issues:
        msg = (
            "Environment check failed — system is not idle:\n"
            + "\n".join(f"  - {issue}" for issue in issues)
            + "\nBenchmark results would be unreliable. "
            "Use --skip-env-checks to run anyway."
        )
        pytest.exit(msg, returncode=1)


@pytest.fixture
def device(request):
    """Get the device to run benchmarks on, controlled by --device flag."""
    requested = request.config.getoption("--device")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
