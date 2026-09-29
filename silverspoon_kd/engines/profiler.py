"""
Profiling utilities for SilverSpoon knowledge distillation.

Thin wrapper around torch.profiler for CPU/GPU kernel tracing and memory analysis.
Traces are saved as Chrome Trace JSON files. View at: https://ui.perfetto.dev/
"""

import inspect
import logging
import os

import torch
from torch.profiler import ProfilerActivity, profile, schedule

logger = logging.getLogger(__name__)

# ``acc_events`` was added in torch 2.5; probe once so older versions still work.
_PROFILE_SUPPORTS_ACC_EVENTS = "acc_events" in inspect.signature(profile).parameters


def create_profiler(
    output_dir: str,
    wait: int = 20,
    warmup: int = 3,
    active: int = 3,
    repeat: int = 1,
    with_stack: bool = False,
) -> profile:
    """
    Create a torch.profiler.profile instance with a configurable schedule.

    Traces are exported as Chrome Trace JSON files via the on_trace_ready callback.

    Args:
        output_dir: Directory to save trace files.
        wait: Number of steps to skip before profiling begins.
        warmup: Number of warmup steps (not traced).
        active: Number of steps to actively trace.
        repeat: Number of times to repeat the wait/warmup/active cycle.
        with_stack: Whether to record Python call stacks in traces.

    Returns:
        A torch.profiler.profile instance (use as context manager).
    """
    os.makedirs(output_dir, exist_ok=True)

    def trace_handler(prof):
        output_path = os.path.join(
            output_dir,
            f"trace_step_{prof.step_num}.json",
        )
        prof.export_chrome_trace(output_path)
        logger.info("Profiler trace saved to: %s", output_path)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    profile_kwargs = {
        "activities": activities,
        "schedule": schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        "on_trace_ready": trace_handler,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": with_stack,
        "with_flops": False,
    }
    if _PROFILE_SUPPORTS_ACC_EVENTS:
        profile_kwargs["acc_events"] = True
    return profile(**profile_kwargs)
