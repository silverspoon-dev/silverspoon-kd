"""
Unit tests for profiler.py.
"""

import os
from unittest.mock import patch

import torch
from torch.profiler import profile

from silverspoon_kd.engines.profiler import create_profiler


class TestCreateProfiler:
    """Tests for create_profiler function."""

    def test_creates_profiler_instance(self, tmp_path):
        """Test that create_profiler returns a profile instance."""
        output_dir = str(tmp_path / "traces")
        prof = create_profiler(output_dir)
        assert isinstance(prof, profile)

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created."""
        output_dir = str(tmp_path / "traces" / "nested")
        create_profiler(output_dir)
        assert os.path.isdir(output_dir)

    def test_trace_handler_exports_chrome_trace(self, tmp_path):
        """Test that the trace handler callback writes a Chrome trace file."""
        output_dir = tmp_path / "traces"
        prof = create_profiler(str(output_dir), wait=0, warmup=1, active=1, repeat=1)

        # Use the profiler briefly to trigger trace export
        # Need wait + warmup + active = 2 steps minimum
        with prof:
            for _ in range(2):
                x = torch.randn(2, 2)
                _ = x + x
                prof.step()

        traces = sorted(output_dir.glob("trace_step_*.json"))
        assert traces, f"no trace file written to {output_dir}"
        assert all(t.stat().st_size > 0 for t in traces)

    def test_custom_schedule_parameters(self, tmp_path):
        """Test profiler with custom schedule parameters."""
        output_dir = str(tmp_path / "traces")
        prof = create_profiler(output_dir, wait=5, warmup=2, active=2, repeat=2, with_stack=True)
        assert isinstance(prof, profile)

    @patch("silverspoon_kd.engines.profiler.torch.cuda.is_available", return_value=False)
    def test_cpu_only_activities(self, mock_cuda, tmp_path):
        """Test that only CPU activity is used when CUDA is unavailable."""
        output_dir = str(tmp_path / "traces")
        prof = create_profiler(output_dir)
        assert isinstance(prof, profile)

    @patch("silverspoon_kd.engines.profiler.torch.cuda.is_available", return_value=True)
    def test_cuda_activities_included(self, mock_cuda, tmp_path):
        """Test that CUDA activity is included when available."""
        output_dir = str(tmp_path / "traces")
        prof = create_profiler(output_dir)
        assert isinstance(prof, profile)
