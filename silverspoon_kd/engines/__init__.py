"""Profiling, memory diagnostics, and module-capture utilities."""

from .memory_diagnostics import (
    MemoryLeakDetectionCallback,
    StorageDiffBucket,
    StorageInfo,
    TensorDiffBucket,
    TensorInfo,
    collect_storage_inventory,
    collect_tensor_inventory,
    describe_referrers,
    diff_storage_inventories,
    diff_tensor_inventories,
    find_tensors_for_data_ptrs,
    format_memory_stats,
    log_storage_diff,
    log_tensor_diff,
    record_cuda_memory_history,
)
from .module_capture_engine import ModuleCaptureEngine
from .profiler import create_profiler

__all__ = [
    "MemoryLeakDetectionCallback",
    "ModuleCaptureEngine",
    "StorageDiffBucket",
    "StorageInfo",
    "TensorDiffBucket",
    "TensorInfo",
    "collect_storage_inventory",
    "collect_tensor_inventory",
    "create_profiler",
    "describe_referrers",
    "diff_storage_inventories",
    "diff_tensor_inventories",
    "find_tensors_for_data_ptrs",
    "format_memory_stats",
    "log_storage_diff",
    "log_tensor_diff",
    "record_cuda_memory_history",
]
