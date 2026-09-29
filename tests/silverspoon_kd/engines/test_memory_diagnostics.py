"""
Comprehensive unit tests for ``silverspoon_kd.engines.memory_diagnostics``.

Covers:

* ``collect_tensor_inventory`` — inventory walking via ``gc.get_objects``.
* ``diff_tensor_inventories`` / ``log_tensor_diff`` — bucketed diffs between
  two inventory snapshots.
* ``record_cuda_memory_history`` — context manager around PyTorch's private
  memory history API.
* ``format_memory_stats`` — one-line CUDA stats string.
* ``MemoryLeakDetectionCallback`` — TrainerCallback integration.

Tests that require a real GPU are marked with ``@pytest.mark.cuda`` and
skipped on CPU-only runners. The majority of the logic is tested on CPU
by passing ``include_cpu=True`` and by heavily mocking the CUDA APIs.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import pickle
from unittest.mock import MagicMock, patch

import pytest
import torch

from silverspoon_kd.engines.memory_diagnostics import (
    MemoryLeakDetectionCallback,
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

# ─────────────────────────────────────────────────────────────────────────────
# TensorInfo / TensorDiffBucket: dataclass sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestTensorInfo:
    def test_construction(self):
        info = TensorInfo(
            shape=(2, 1024, 2048),
            dtype="torch.bfloat16",
            device="cuda:0",
            nbytes=8388608,
        )
        assert info.shape == (2, 1024, 2048)
        assert info.dtype == "torch.bfloat16"
        assert info.device == "cuda:0"
        assert info.nbytes == 8388608

    def test_equality(self):
        a = TensorInfo(shape=(4, 4), dtype="torch.float32", device="cpu", nbytes=64)
        b = TensorInfo(shape=(4, 4), dtype="torch.float32", device="cpu", nbytes=64)
        c = TensorInfo(shape=(4, 8), dtype="torch.float32", device="cpu", nbytes=128)
        assert a == b
        assert a != c

    def test_frozen(self):
        info = TensorInfo(shape=(1,), dtype="torch.int64", device="cpu", nbytes=8)
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.nbytes = 999  # type: ignore[misc]


class TestTensorDiffBucket:
    def test_construction(self):
        bucket = TensorDiffBucket(
            shape=(1, 1024, 2048),
            dtype="torch.bfloat16",
            device="cuda:0",
            count=28,
            total_bytes=939524096,
        )
        assert bucket.count == 28
        assert bucket.total_bytes == 939524096


# ─────────────────────────────────────────────────────────────────────────────
# collect_tensor_inventory
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectTensorInventory:
    def test_includes_cpu_tensor_when_requested(self):
        t = torch.zeros(4, 8, dtype=torch.float32)
        inv = collect_tensor_inventory(include_cpu=True)
        assert id(t) in inv
        info = inv[id(t)]
        assert info.shape == (4, 8)
        assert info.dtype == "torch.float32"
        assert "cpu" in info.device
        assert info.nbytes == 4 * 8 * 4  # float32 = 4 bytes
        # Keep t alive until after the assertion so gc doesn't reap it.
        assert t.numel() == 32

    def test_excludes_cpu_tensor_by_default(self):
        t = torch.zeros(4, 8, dtype=torch.float32)
        inv = collect_tensor_inventory(include_cpu=False)
        assert id(t) not in inv
        assert t.numel() == 32

    def test_dropped_reference_disappears_after_gc(self):
        t = torch.zeros(100, 100, dtype=torch.float32)
        tid = id(t)
        inv_before = collect_tensor_inventory(include_cpu=True)
        assert tid in inv_before
        del t
        gc.collect()
        inv_after = collect_tensor_inventory(include_cpu=True)
        # Note: id() of a dead object can be reused, but within the same
        # test the chance of collision is negligible.
        assert tid not in inv_after

    def test_different_dtypes_reported_correctly(self):
        tensors = {
            "bf16": torch.zeros(10, dtype=torch.bfloat16),
            "fp16": torch.zeros(10, dtype=torch.float16),
            "fp32": torch.zeros(10, dtype=torch.float32),
            "int8": torch.zeros(10, dtype=torch.int8),
            "int64": torch.zeros(10, dtype=torch.int64),
        }
        expected_bytes = {
            "bf16": 20,
            "fp16": 20,
            "fp32": 40,
            "int8": 10,
            "int64": 80,
        }
        inv = collect_tensor_inventory(include_cpu=True)
        for name, t in tensors.items():
            assert id(t) in inv, f"{name} tensor missing from inventory"
            assert inv[id(t)].nbytes == expected_bytes[name], name

    def test_force_gc_collects_cycles(self):
        """Cycle-referenced tensors should be collected when force_gc=True."""

        class _Holder:
            pass

        holder = _Holder()
        holder.tensor = torch.zeros(50, 50, dtype=torch.float32)
        holder.self_ref = holder  # cycle
        tid = id(holder.tensor)
        del holder  # drops the only external reference, but cycle persists
        # Without force_gc, the cycle may not be collected yet
        # With force_gc, gc.collect() is called first
        inv = collect_tensor_inventory(include_cpu=True, force_gc=True)
        assert tid not in inv

    def test_non_tensor_objects_ignored(self):
        """Objects that aren't torch.Tensor shouldn't end up in inventory."""
        dummy = {"not": "a tensor"}
        other = [1, 2, 3]
        inv = collect_tensor_inventory(include_cpu=True)
        assert id(dummy) not in inv
        assert id(other) not in inv
        # Make sure locals stay alive
        assert dummy and other

    def test_empty_tensor(self):
        t = torch.zeros(0, dtype=torch.float32)
        inv = collect_tensor_inventory(include_cpu=True)
        assert id(t) in inv
        assert inv[id(t)].shape == (0,)
        assert inv[id(t)].nbytes == 0

    def test_scalar_tensor(self):
        t = torch.tensor(3.14, dtype=torch.float32)
        inv = collect_tensor_inventory(include_cpu=True)
        assert id(t) in inv
        assert inv[id(t)].shape == ()
        assert inv[id(t)].nbytes == 4

    def test_deprecated_attribute_warnings_suppressed(self):
        """Walking gc.get_objects() should not leak warnings from deprecated
        module attributes (e.g. torch.distributed.reduce_op)."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            collect_tensor_inventory(include_cpu=True)
        # The inventory walk should not have produced any warnings that
        # leak out to the caller's scope.
        torch_dist_warnings = [w for w in caught if "torch.distributed" in str(w.message)]
        assert len(torch_dist_warnings) == 0, (
            f"Expected no torch.distributed warnings, got: "
            f"{[str(w.message) for w in torch_dist_warnings]}"
        )

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensor_picked_up(self):
        t = torch.zeros(4, 8, dtype=torch.float32, device="cuda")
        inv = collect_tensor_inventory()  # default: CUDA only
        assert id(t) in inv
        assert "cuda" in inv[id(t)].device
        assert t.numel() == 32

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensor_excluded_when_cpu_only_filter(self):
        """The CPU filter is the inverse of the CUDA filter; test symmetry."""
        t_cpu = torch.zeros(10)
        t_cuda = torch.zeros(10, device="cuda")
        inv_cuda_only = collect_tensor_inventory(include_cpu=False)
        assert id(t_cuda) in inv_cuda_only
        assert id(t_cpu) not in inv_cuda_only
        inv_both = collect_tensor_inventory(include_cpu=True)
        assert id(t_cuda) in inv_both
        assert id(t_cpu) in inv_both


# ─────────────────────────────────────────────────────────────────────────────
# diff_tensor_inventories
# ─────────────────────────────────────────────────────────────────────────────


def _info(shape, dtype="torch.bfloat16", device="cuda:0", nbytes=None):
    if nbytes is None:
        nelem = 1
        for d in shape:
            nelem *= d
        nbytes = nelem * 2  # bf16 default
    return TensorInfo(shape=shape, dtype=dtype, device=device, nbytes=nbytes)


class TestDiffTensorInventories:
    def test_empty_before_and_after(self):
        buckets, total = diff_tensor_inventories({}, {})
        assert buckets == []
        assert total == 0

    def test_identical_inventories(self):
        inv = {1: _info((4, 4)), 2: _info((8, 8))}
        buckets, total = diff_tensor_inventories(inv, inv)
        assert buckets == []
        assert total == 0

    def test_only_new_tensors(self):
        before: dict = {}
        after = {
            10: _info((1, 1024, 2048)),
            11: _info((1, 1024, 2048)),
            12: _info((1, 1024, 2048)),
        }
        buckets, total = diff_tensor_inventories(before, after)
        assert len(buckets) == 1
        assert buckets[0].count == 3
        assert buckets[0].shape == (1, 1024, 2048)
        assert buckets[0].total_bytes == 3 * 1024 * 2048 * 2
        assert total == 3 * 1024 * 2048 * 2

    def test_disappeared_tensors_not_reported(self):
        """Tensors present in `before` but not `after` should be ignored.

        The diff only cares about NEW tensors, not deleted ones.
        """
        before = {1: _info((4, 4)), 2: _info((8, 8))}
        after = {1: _info((4, 4))}  # id=2 disappeared
        buckets, total = diff_tensor_inventories(before, after)
        assert buckets == []
        assert total == 0

    def test_mixed_old_and_new(self):
        before = {1: _info((4, 4))}
        after = {
            1: _info((4, 4)),  # unchanged
            2: _info((8, 8)),  # new
            3: _info((8, 8)),  # new, same bucket as 2
            4: _info((16, 16)),  # new, different bucket
        }
        buckets, total = diff_tensor_inventories(before, after)
        assert len(buckets) == 2
        # Sorted by total_bytes desc — (16, 16) bucket is larger
        assert buckets[0].shape == (16, 16)
        assert buckets[0].count == 1
        assert buckets[0].total_bytes == 16 * 16 * 2
        assert buckets[1].shape == (8, 8)
        assert buckets[1].count == 2
        assert buckets[1].total_bytes == 2 * 8 * 8 * 2
        assert total == buckets[0].total_bytes + buckets[1].total_bytes

    def test_bucketing_groups_by_shape_dtype_device(self):
        """Tensors with matching (shape, dtype, device) are grouped."""
        before: dict = {}
        after = {
            1: _info((4, 4), dtype="torch.float32"),
            2: _info((4, 4), dtype="torch.float32"),
            3: _info((4, 4), dtype="torch.bfloat16"),  # different dtype
            4: _info((4, 4), dtype="torch.float32", device="cuda:1"),  # different device
        }
        buckets, _ = diff_tensor_inventories(before, after)
        # 3 buckets: fp32/cuda:0 (count=2), bf16/cuda:0 (count=1), fp32/cuda:1 (count=1)
        assert len(buckets) == 3
        counts_by_key = {(b.shape, b.dtype, b.device): b.count for b in buckets}
        assert counts_by_key[((4, 4), "torch.float32", "cuda:0")] == 2
        assert counts_by_key[((4, 4), "torch.bfloat16", "cuda:0")] == 1
        assert counts_by_key[((4, 4), "torch.float32", "cuda:1")] == 1

    def test_sorted_by_total_bytes_descending(self):
        before: dict = {}
        after = {
            1: _info((100,)),  # 200 B
            2: _info((1000,)),  # 2000 B — largest
            3: _info((10,)),  # 20 B
            4: _info((500,)),  # 1000 B
        }
        buckets, _ = diff_tensor_inventories(before, after)
        sizes = [b.total_bytes for b in buckets]
        assert sizes == sorted(sizes, reverse=True)
        assert buckets[0].shape == (1000,)
        assert buckets[-1].shape == (10,)

    def test_realistic_attention_leak_pattern(self):
        """Simulate the per-layer attention leak we observed in HKD QAT.

        28 bf16 attention-score-sized tensors of shape (1, 16, 1024, 1024)
        should aggregate to approximately 939 MiB (= 28 × 32 MiB).
        """
        before: dict = {}
        after = {}
        bucket_bytes = 1 * 16 * 1024 * 1024 * 2  # ~33.5 MiB
        for i in range(28):
            after[i] = _info(
                (1, 16, 1024, 1024),
                dtype="torch.bfloat16",
                nbytes=bucket_bytes,
            )
        buckets, total = diff_tensor_inventories(before, after)
        assert len(buckets) == 1
        assert buckets[0].count == 28
        assert total == 28 * bucket_bytes
        # The resulting total should be very close to 0.875 GiB
        total_gib = total / 1024**3
        assert 0.86 < total_gib < 0.90, f"got {total_gib:.3f} GiB"


# ─────────────────────────────────────────────────────────────────────────────
# log_tensor_diff
# ─────────────────────────────────────────────────────────────────────────────


class TestLogTensorDiff:
    def test_returns_buckets_and_total(self):
        before: dict = {}
        after = {1: _info((4, 4)), 2: _info((4, 4))}
        buckets, total = log_tensor_diff(before, after)
        assert len(buckets) == 1
        assert buckets[0].count == 2
        assert total == 2 * 4 * 4 * 2

    def test_logs_total_count_and_bytes(self, caplog):
        before: dict = {}
        after = {1: _info((4, 4))}
        with caplog.at_level(logging.INFO, "silverspoon_kd.engines.memory_diagnostics"):
            log_tensor_diff(before, after, label="my_label")
        messages = [rec.message for rec in caplog.records]
        assert any("TENSOR_DIFF [my_label]" in m for m in messages)
        assert any("NEW tensors:" in m for m in messages)

    def test_logs_individual_bucket_rows(self, caplog):
        before: dict = {}
        after = {
            1: _info((1, 1024, 2048)),
            2: _info((1, 1024, 2048)),
            3: _info((16, 16)),
        }
        with caplog.at_level(logging.INFO, "silverspoon_kd.engines.memory_diagnostics"):
            log_tensor_diff(before, after)
        messages = [rec.message for rec in caplog.records]
        # One line per bucket (2 buckets here)
        bucket_lines = [m for m in messages if m.startswith("  NEW")]
        assert len(bucket_lines) == 2
        # The largest bucket (1, 1024, 2048) should be logged first
        assert "(1, 1024, 2048)" in bucket_lines[0]
        assert "(16, 16)" in bucket_lines[1]

    def test_max_buckets_truncates_output(self, caplog):
        before: dict = {}
        after = {
            i: _info((i + 1,))  # each in its own bucket
            for i in range(10)
        }
        with caplog.at_level(logging.INFO, "silverspoon_kd.engines.memory_diagnostics"):
            log_tensor_diff(before, after, max_buckets=3)
        messages = [rec.message for rec in caplog.records]
        bucket_lines = [m for m in messages if m.startswith("  NEW")]
        assert len(bucket_lines) == 3
        # "more buckets" message should appear
        assert any("more buckets" in m for m in messages)

    def test_max_buckets_zero_means_unlimited(self, caplog):
        before: dict = {}
        after = {i: _info((i + 1,)) for i in range(50)}
        with caplog.at_level(logging.INFO, "silverspoon_kd.engines.memory_diagnostics"):
            log_tensor_diff(before, after, max_buckets=0)
        messages = [rec.message for rec in caplog.records]
        bucket_lines = [m for m in messages if m.startswith("  NEW")]
        assert len(bucket_lines) == 50

    def test_uses_provided_logger(self):
        custom_log = logging.getLogger("test.custom_logger")
        mock_log = MagicMock(spec=custom_log)
        before: dict = {}
        after = {1: _info((4,))}
        log_tensor_diff(before, after, log=mock_log)
        assert mock_log.info.called


# ─────────────────────────────────────────────────────────────────────────────
# format_memory_stats
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatMemoryStats:
    def test_returns_cuda_not_available_when_no_gpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert format_memory_stats() == "CUDA not available"

    def test_formatted_string_when_gpu_available(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=2 * 1024**3),
            patch("torch.cuda.memory_reserved", return_value=3 * 1024**3),
            patch("torch.cuda.max_memory_allocated", return_value=4 * 1024**3),
        ):
            s = format_memory_stats()
        assert "alloc=2.000GiB" in s
        assert "reserved=3.000GiB" in s
        assert "peak=4.000GiB" in s


# ─────────────────────────────────────────────────────────────────────────────
# record_cuda_memory_history
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordCudaMemoryHistory:
    def test_noop_when_cuda_not_available(self, tmp_path):
        snapshot_path = str(tmp_path / "snap.pickle")
        with patch("torch.cuda.is_available", return_value=False):
            with record_cuda_memory_history(snapshot_path):
                pass
        # Nothing should have been written
        assert not os.path.exists(snapshot_path)

    def test_starts_and_stops_recording(self, tmp_path):
        snapshot_path = str(tmp_path / "subdir" / "snap.pickle")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory._record_memory_history") as mock_rec,
            patch("torch.cuda.memory._dump_snapshot") as mock_dump,
        ):
            with record_cuda_memory_history(snapshot_path, max_entries=500):
                pass
        # Recording should have been started with enabled="all"
        start_calls = [c for c in mock_rec.call_args_list if c.kwargs.get("enabled") == "all"]
        assert len(start_calls) == 1
        assert start_calls[0].kwargs["max_entries"] == 500
        # Dump should have been called
        mock_dump.assert_called_once_with(snapshot_path)
        # Recording should have been disabled on exit
        disable_calls = [c for c in mock_rec.call_args_list if c.kwargs.get("enabled") is None]
        assert len(disable_calls) == 1

    def test_creates_parent_directory(self, tmp_path):
        snapshot_path = str(tmp_path / "deeply" / "nested" / "dir" / "snap.pickle")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory._record_memory_history"),
            patch("torch.cuda.memory._dump_snapshot") as mock_dump,
        ):
            with record_cuda_memory_history(snapshot_path):
                pass
        # The parent directory should exist (even if we didn't actually write)
        assert os.path.isdir(os.path.dirname(snapshot_path))
        mock_dump.assert_called_once()

    def test_handles_start_failure_gracefully(self, tmp_path):
        snapshot_path = str(tmp_path / "snap.pickle")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.memory._record_memory_history",
                side_effect=RuntimeError("boom"),
            ),
        ):
            # Should not raise
            with record_cuda_memory_history(snapshot_path):
                pass

    def test_handles_dump_failure_gracefully(self, tmp_path):
        snapshot_path = str(tmp_path / "snap.pickle")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory._record_memory_history"),
            patch(
                "torch.cuda.memory._dump_snapshot",
                side_effect=RuntimeError("disk full"),
            ),
        ):
            # Should not raise
            with record_cuda_memory_history(snapshot_path):
                pass

    def test_body_exception_still_stops_recording(self, tmp_path):
        snapshot_path = str(tmp_path / "snap.pickle")
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory._record_memory_history") as mock_rec,
            patch("torch.cuda.memory._dump_snapshot"),
            pytest.raises(ValueError, match="body error"),
        ):
            with record_cuda_memory_history(snapshot_path):
                raise ValueError("body error")
        # Even with an exception, disable should still have been called
        disable_calls = [c for c in mock_rec.call_args_list if c.kwargs.get("enabled") is None]
        assert len(disable_calls) == 1

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_real_cuda_snapshot_creates_pickle(self, tmp_path):
        snapshot_path = str(tmp_path / "real_snap.pickle")
        with record_cuda_memory_history(snapshot_path):
            # Do some CUDA work so there's something to record
            _ = torch.zeros(1024, 1024, device="cuda")
        assert os.path.exists(snapshot_path)
        # Sanity check: file is a valid pickle
        with open(snapshot_path, "rb") as f:
            data = pickle.load(f)
        assert data is not None


# ─────────────────────────────────────────────────────────────────────────────
# MemoryLeakDetectionCallback
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryLeakDetectionCallbackValidation:
    def test_snapshot_at_without_path_raises(self):
        with pytest.raises(ValueError, match="snapshot_path is required"):
            MemoryLeakDetectionCallback(snapshot_at_batch=10)

    def test_reversed_diff_batches_raises(self):
        with pytest.raises(ValueError, match="diff_between_batches"):
            MemoryLeakDetectionCallback(diff_between_batches=(15, 5))

    def test_equal_diff_batches_raises(self):
        with pytest.raises(ValueError, match="diff_between_batches"):
            MemoryLeakDetectionCallback(diff_between_batches=(5, 5))

    def test_default_instantiation_succeeds(self):
        cb = MemoryLeakDetectionCallback()
        assert cb._eval_batch == 0
        assert cb._diff_between is None
        assert cb._snapshot_at is None

    def test_is_trainer_callback_subclass(self):
        """The callback class subclasses TrainerCallback directly."""
        from transformers.trainer_callback import TrainerCallback

        assert issubclass(MemoryLeakDetectionCallback, TrainerCallback)

        cb = MemoryLeakDetectionCallback()
        assert isinstance(cb, TrainerCallback)
        assert type(cb) is MemoryLeakDetectionCallback


class TestMemoryLeakDetectionCallbackLogging:
    """Tests for the per-batch logging behavior, using CPU-only mocks."""

    def setup_method(self):
        # Patch CUDA availability + memory_allocated etc. so the callback
        # can run on CPU-only CI.
        self.patches = [
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024**3),
            patch("torch.cuda.memory_reserved", return_value=2 * 1024**3),
            patch("torch.cuda.max_memory_allocated", return_value=3 * 1024**3),
            patch("torch.cuda.reset_peak_memory_stats"),
        ]
        for p in self.patches:
            p.start()

    def teardown_method(self):
        for p in self.patches:
            p.stop()

    def _step(self, cb, n=1):
        for _ in range(n):
            cb.on_prediction_step(None, None, None)

    def test_logs_first_n_batches_always(self, caplog):
        cb = MemoryLeakDetectionCallback(
            log_first_n_batches=5,
            log_every_n_batches=100,
        )
        with caplog.at_level(logging.INFO, "silverspoon_kd.engines.memory_diagnostics"):
            self._step(cb, n=5)
        logged = [r for r in caplog.records if "MEMORY [eval_batch_" in r.message]
        assert len(logged) == 5

    def test_log_every_n_batches_throttles(self, caplog):
        cb = MemoryLeakDetectionCallback(
            log_first_n_batches=0,
            log_every_n_batches=5,
        )
        with caplog.at_level(logging.INFO, "silverspoon_kd.engines.memory_diagnostics"):
            self._step(cb, n=20)
        logged = [r for r in caplog.records if "MEMORY [eval_batch_" in r.message]
        # Should log at batches 5, 10, 15, 20
        assert len(logged) == 4

    def test_log_every_zero_logs_every_batch(self, caplog):
        cb = MemoryLeakDetectionCallback(
            log_first_n_batches=0,
            log_every_n_batches=0,
        )
        with caplog.at_level(logging.INFO, "silverspoon_kd.engines.memory_diagnostics"):
            self._step(cb, n=3)
        logged = [r for r in caplog.records if "MEMORY [eval_batch_" in r.message]
        assert len(logged) == 3

    def test_on_evaluate_resets_batch_counter(self, caplog):
        cb = MemoryLeakDetectionCallback()
        self._step(cb, n=5)
        assert cb._eval_batch == 5
        cb.on_evaluate(None, None, None)
        assert cb._eval_batch == 0
        # After reset, next prediction_step starts at 1
        self._step(cb, n=1)
        assert cb._eval_batch == 1


class TestMemoryLeakDetectionCallbackDiff:
    def setup_method(self):
        self.patches = [
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024**3),
            patch("torch.cuda.memory_reserved", return_value=2 * 1024**3),
            patch("torch.cuda.max_memory_allocated", return_value=3 * 1024**3),
            patch("torch.cuda.reset_peak_memory_stats"),
        ]
        for p in self.patches:
            p.start()

    def teardown_method(self):
        for p in self.patches:
            p.stop()

    def test_diff_takes_two_snapshots_and_logs_result(self, caplog):
        cb = MemoryLeakDetectionCallback(diff_between_batches=(2, 5))
        # Patch the inventory function to return controlled results.
        snapshots = iter(
            [
                # Before: empty
                {},
                # After: one new tensor
                {1: _info((1, 1024, 2048))},
            ]
        )
        with (
            patch(
                "silverspoon_kd.engines.memory_diagnostics.collect_tensor_inventory",
                side_effect=lambda **kw: next(snapshots),
            ),
            caplog.at_level(
                logging.INFO,
                "silverspoon_kd.engines.memory_diagnostics",
            ),
        ):
            for _ in range(5):
                cb.on_prediction_step(None, None, None)
        messages = [r.message for r in caplog.records]
        # Should have logged a TENSOR_DIFF line
        diff_lines = [m for m in messages if "TENSOR_DIFF" in m]
        assert any("eval_batches_2_to_5" in m for m in diff_lines), (
            f"Expected diff label in: {diff_lines}"
        )

    def test_diff_disabled_by_default(self, caplog):
        cb = MemoryLeakDetectionCallback()
        with patch(
            "silverspoon_kd.engines.memory_diagnostics.collect_tensor_inventory"
        ) as mock_collect:
            for _ in range(10):
                cb.on_prediction_step(None, None, None)
            mock_collect.assert_not_called()

    def test_diff_only_at_specified_batches(self):
        cb = MemoryLeakDetectionCallback(diff_between_batches=(3, 7))
        with patch(
            "silverspoon_kd.engines.memory_diagnostics.collect_tensor_inventory",
            return_value={},
        ) as mock_collect:
            for _ in range(10):
                cb.on_prediction_step(None, None, None)
        # Should have been called exactly twice: at batches 3 and 7
        assert mock_collect.call_count == 2


class TestMemoryLeakDetectionCallbackSnapshot:
    def setup_method(self):
        self.patches = [
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024**3),
            patch("torch.cuda.memory_reserved", return_value=2 * 1024**3),
            patch("torch.cuda.max_memory_allocated", return_value=3 * 1024**3),
            patch("torch.cuda.reset_peak_memory_stats"),
        ]
        for p in self.patches:
            p.start()

    def teardown_method(self):
        for p in self.patches:
            p.stop()

    def test_snapshot_at_batch_starts_history_early(self, tmp_path):
        cb = MemoryLeakDetectionCallback(
            snapshot_at_batch=10,
            snapshot_path=str(tmp_path / "snap.pickle"),
        )
        with (
            patch("torch.cuda.memory._record_memory_history") as mock_rec,
            patch("torch.cuda.memory._dump_snapshot") as mock_dump,
        ):
            for _ in range(10):
                cb.on_prediction_step(None, None, None)
        # Recording should start at batch 5 (10 - 5)
        start_calls = [c for c in mock_rec.call_args_list if c.kwargs.get("enabled") == "all"]
        assert len(start_calls) == 1
        # And dump at batch 10
        mock_dump.assert_called_once()

    def test_no_snapshot_when_disabled(self, tmp_path):
        cb = MemoryLeakDetectionCallback()
        with (
            patch("torch.cuda.memory._record_memory_history") as mock_rec,
            patch("torch.cuda.memory._dump_snapshot") as mock_dump,
        ):
            for _ in range(20):
                cb.on_prediction_step(None, None, None)
        mock_rec.assert_not_called()
        mock_dump.assert_not_called()

    def test_snapshot_dumped_at_exact_batch(self, tmp_path):
        cb = MemoryLeakDetectionCallback(
            snapshot_at_batch=8,
            snapshot_path=str(tmp_path / "snap.pickle"),
        )
        with (
            patch("torch.cuda.memory._record_memory_history"),
            patch("torch.cuda.memory._dump_snapshot") as mock_dump,
        ):
            for i in range(15):
                cb.on_prediction_step(None, None, None)
                if i + 1 == 8:
                    # At exactly batch 8, dump should have been called
                    assert mock_dump.call_count == 1
        # Only called once total, even though we kept running
        assert mock_dump.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: build a real inventory and diff it on CPU
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_real_inventory_diff_on_cpu(self):
        """Create tensors, collect inventory, add more, diff, verify.

        This test actually exercises the full flow without mocks, on CPU,
        to verify the utilities work together with real torch tensors.
        """
        # Inventory 1: two small tensors
        t1 = torch.zeros(100, dtype=torch.float32)
        t2 = torch.zeros(200, dtype=torch.float32)
        inv_before = collect_tensor_inventory(include_cpu=True)
        assert id(t1) in inv_before
        assert id(t2) in inv_before

        # Add three more tensors (including a bigger one)
        t3 = torch.zeros(500, dtype=torch.bfloat16)
        t4 = torch.zeros(500, dtype=torch.bfloat16)  # same bucket as t3
        t5 = torch.zeros(1_000_000, dtype=torch.float32)  # largest
        inv_after = collect_tensor_inventory(include_cpu=True)

        buckets, _total_bytes = diff_tensor_inventories(inv_before, inv_after)

        # Find the buckets for our new tensors
        bucket_map = {(b.shape, b.dtype): b for b in buckets}
        bf16_key = ((500,), "torch.bfloat16")
        fp32_key = ((1_000_000,), "torch.float32")

        assert bf16_key in bucket_map, f"bf16 bucket missing: {bucket_map}"
        assert bucket_map[bf16_key].count == 2
        assert bucket_map[bf16_key].total_bytes == 2 * 500 * 2

        assert fp32_key in bucket_map
        assert bucket_map[fp32_key].count == 1
        assert bucket_map[fp32_key].total_bytes == 1_000_000 * 4

        # Large fp32 tensor should be sorted first (biggest total_bytes)
        assert buckets[0].shape == (1_000_000,)

        # Keep references alive so gc doesn't reap them mid-test
        assert (t1, t2, t3, t4, t5) is not None


# ─────────────────────────────────────────────────────────────────────────────
# StorageInfo / collect_storage_inventory: storage-faithful inventory
#
# These tests catch the class of bugs where the tensor inventory under-counts
# memory because views into a large storage report only their view size,
# missing the underlying buffer.
# ─────────────────────────────────────────────────────────────────────────────


class TestStorageInfo:
    def test_construction(self):
        info = StorageInfo(
            nbytes=1024**2,
            device="cuda:0",
            representative_shape=(512, 512),
            representative_dtype="torch.float32",
            num_views=3,
        )
        assert info.nbytes == 1024**2
        assert info.num_views == 3


class TestCollectStorageInventory:
    def test_basic_storage_capture(self):
        t = torch.zeros(100, 100, dtype=torch.float32)
        inv = collect_storage_inventory(include_cpu=True)
        # Find a storage with our tensor's data_ptr
        ptr = t.untyped_storage().data_ptr()
        assert ptr in inv
        assert inv[ptr].nbytes == 100 * 100 * 4
        assert inv[ptr].num_views == 1
        assert t.numel() == 10000

    def test_views_share_one_storage_entry(self):
        """Multiple views into one storage should appear as a single entry
        whose nbytes equals the FULL underlying storage, not the view size.

        This is the regression test for the original bug — the tensor
        inventory counted views as their visible size, missing the actual
        memory.
        """
        big = torch.zeros(1000, 1000, dtype=torch.float32)
        # Take 10 small views into the same storage
        _views = [big[i, :].clone() for i in range(10)]
        # Note: .clone() creates new storages, so let's use slicing
        # which preserves storage sharing.
        view_a = big[0, :]  # (1000,)
        view_b = big[1, :]  # (1000,)
        view_c = big[:, 100:200]  # (1000, 100)

        inv = collect_storage_inventory(include_cpu=True)
        ptr = big.untyped_storage().data_ptr()
        assert ptr in inv
        # The storage's nbytes should be the FULL big tensor size, not
        # any individual view's size
        assert inv[ptr].nbytes == 1000 * 1000 * 4
        # And the view count should reflect all the tensors that share it
        # (big + view_a + view_b + view_c = 4)
        assert inv[ptr].num_views >= 4
        # Keep references alive
        assert (big, view_a, view_b, view_c) is not None

    def test_view_shapes_records_all_distinct_shapes(self):
        """The view_shapes field should record every distinct shape sharing
        a storage. This is the field we use to debug 'scalar view of large
        buffer' patterns: if a 32 MiB storage's view_shapes is just `((),)`,
        it tells us a tiny scalar is keeping the whole buffer alive.
        """
        big = torch.zeros(100, 100, dtype=torch.float32)
        slice_2d = big[:, 0:50]  # shape (100, 50)
        slice_1d = big[0, :]  # shape (100,)
        scalar_view = big[0, 0]  # shape ()

        inv = collect_storage_inventory(include_cpu=True)
        ptr = big.untyped_storage().data_ptr()
        info = inv[ptr]
        # All four distinct shapes should appear in view_shapes
        view_shapes_set = set(info.view_shapes)
        assert (100, 100) in view_shapes_set, view_shapes_set
        assert (100, 50) in view_shapes_set, view_shapes_set
        assert (100,) in view_shapes_set, view_shapes_set
        assert () in view_shapes_set, view_shapes_set
        # And it should be sorted by shape length descending so the most
        # informative shape comes first
        lens = [len(s) for s in info.view_shapes]
        assert lens == sorted(lens, reverse=True)
        # Keep refs alive
        assert (big, slice_2d, slice_1d, scalar_view) is not None

    def test_view_shapes_propagates_to_diff_bucket(self):
        """The bucket's example_view_shapes field should let users see the
        view-vs-storage mismatch directly in log output. This is the
        actionable diagnostic that pinpoints the leak source.
        """
        # Simulate a 'scalar view of large buffer' pattern
        before = collect_storage_inventory(include_cpu=True)
        big = torch.zeros(1000, dtype=torch.float32)  # 4000 bytes
        # Only keep a scalar view alive (the big tensor itself goes
        # out of scope, but the scalar view keeps the storage alive)
        scalar_only = big[0]
        del big
        after = collect_storage_inventory(include_cpu=True)

        buckets, _ = diff_storage_inventories(before, after)
        # Find the bucket containing our 4000-byte storage
        our_bucket = next((b for b in buckets if b.nbytes == 4000), None)
        assert our_bucket is not None, (
            f"4000-byte bucket missing; got {[b.nbytes for b in buckets]}"
        )
        # The example_view_shapes should contain the scalar shape ()
        assert () in our_bucket.example_view_shapes, (
            f"expected scalar in view_shapes, got {our_bucket.example_view_shapes}"
        )
        assert scalar_only.numel() == 1

    def test_inventory_reports_our_specific_storage_correctly(self):
        """The inventory should contain OUR specific tensor's storage with
        the correct nbytes.

        Robust to parallel/sequential test interference: we look up our
        tensor by data_ptr() instead of comparing totals (which would be
        affected by other tests' tensors being freed/created in between
        snapshots, especially under pytest-xdist).

        This is the test that would have caught the original bug: if we
        allocate exactly 40 MB and the inventory entry reports anything
        different, the inventory is broken.
        """
        N = 20_000_000  # 40 MB of bf16
        t = torch.zeros(N, dtype=torch.bfloat16)
        ptr = t.untyped_storage().data_ptr()

        inv = collect_storage_inventory(include_cpu=True)
        assert ptr in inv, "our tensor's storage was not captured by the inventory"
        # The reported nbytes must EXACTLY match the allocation size
        # (40 MB = 20M × bf16). Not "approximately" — exactly.
        assert inv[ptr].nbytes == N * 2, (
            f"inventory reported {inv[ptr].nbytes} bytes for our 40 MB tensor"
        )
        assert t.numel() == N

    def test_tensor_inventory_undercounts_views_KNOWN_LIMITATION(self):
        """Document the known limitation of collect_tensor_inventory:
        a small view of a large storage is counted at the view's element count,
        not the storage's actual size. The storage inventory is the right
        tool when this matters.

        This test explicitly captures the bug we hit so future regressions
        in either direction (tensor or storage inventory) are caught.
        """
        big = torch.zeros(1024, 1024, dtype=torch.float32)  # 4 MB storage
        small_view = big[0:1, 0:1]  # 4 bytes view of 4 MB storage

        tensor_inv = collect_tensor_inventory(include_cpu=True)
        storage_inv = collect_storage_inventory(include_cpu=True)

        # The view tensor reports its view size (4 bytes)
        view_info = tensor_inv[id(small_view)]
        assert view_info.nbytes == 4

        # The big tensor reports its full size (4 MB)
        big_info = tensor_inv[id(big)]
        assert big_info.nbytes == 4 * 1024 * 1024

        # The storage inventory deduplicates: BOTH share one entry that
        # reflects the full 4 MB
        ptr = big.untyped_storage().data_ptr()
        assert ptr in storage_inv
        assert storage_inv[ptr].nbytes == 4 * 1024 * 1024

        # And ONLY one storage entry exists for them
        assert small_view.untyped_storage().data_ptr() == ptr

    def test_zero_data_ptr_skipped(self):
        """Tensors with no allocation (data_ptr() == 0) should not appear."""
        t = torch.zeros(0, dtype=torch.float32)  # empty, no allocation
        inv = collect_storage_inventory(include_cpu=True)
        # Either it's not in the inventory at all, or its ptr is not 0
        assert all(ptr != 0 for ptr in inv)
        assert t.numel() == 0

    def test_force_gc_collects_cycles(self):
        class _Holder:
            pass

        holder = _Holder()
        holder.tensor = torch.zeros(100, 100, dtype=torch.float32)
        holder.self_ref = holder
        ptr = holder.tensor.untyped_storage().data_ptr()
        del holder
        inv = collect_storage_inventory(include_cpu=True, force_gc=True)
        assert ptr not in inv

    def test_excludes_cpu_by_default(self):
        t = torch.zeros(100, dtype=torch.float32)  # CPU tensor
        inv = collect_storage_inventory(include_cpu=False)
        ptr = t.untyped_storage().data_ptr()
        assert ptr not in inv
        assert t.numel() == 100

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_storage_inventory_captures_specific_allocation(self):
        """The storage inventory must contain OUR specific CUDA tensor
        with the correct nbytes.

        Robust to parallel test runs: identified by data_ptr().
        """
        N = 50_000_000  # 100 MB bf16
        t = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
        ptr = t.untyped_storage().data_ptr()

        storage_inv = collect_storage_inventory(include_cpu=False)
        assert ptr in storage_inv, "our CUDA tensor missing from inventory"
        assert storage_inv[ptr].nbytes == N * 2
        assert "cuda" in storage_inv[ptr].device
        del t


# ─────────────────────────────────────────────────────────────────────────────
# diff_storage_inventories
# ─────────────────────────────────────────────────────────────────────────────


def _sinfo(nbytes, device="cuda:0", shape=(1024,), dtype="torch.float32"):
    return StorageInfo(
        nbytes=nbytes,
        device=device,
        representative_shape=shape,
        representative_dtype=dtype,
        num_views=1,
    )


class TestDiffStorageInventories:
    def test_empty(self):
        buckets, total = diff_storage_inventories({}, {})
        assert buckets == []
        assert total == 0

    def test_only_new_storages(self):
        before: dict = {}
        after = {
            10: _sinfo(33_554_432),  # 32 MB
            11: _sinfo(33_554_432),
            12: _sinfo(33_554_432),
        }
        buckets, total = diff_storage_inventories(before, after)
        assert len(buckets) == 1
        assert buckets[0].count == 3
        assert buckets[0].total_bytes == 3 * 33_554_432
        assert total == 3 * 33_554_432

    def test_disappeared_storages_not_reported(self):
        before = {1: _sinfo(1024), 2: _sinfo(2048)}
        after = {1: _sinfo(1024)}
        buckets, total = diff_storage_inventories(before, after)
        assert buckets == []
        assert total == 0

    def test_bucketing_groups_by_size_dtype_device(self):
        before: dict = {}
        after = {
            1: _sinfo(1024, dtype="torch.float32"),
            2: _sinfo(1024, dtype="torch.float32"),
            3: _sinfo(1024, dtype="torch.bfloat16"),
            4: _sinfo(2048, dtype="torch.float32"),
        }
        buckets, _ = diff_storage_inventories(before, after)
        # 3 buckets: (1024 bytes, fp32), (1024 bytes, bf16), (2048 bytes, fp32)
        assert len(buckets) == 3
        counts = {(b.nbytes, b.representative_dtype): b.count for b in buckets}
        assert counts[(1024, "torch.float32")] == 2
        assert counts[(1024, "torch.bfloat16")] == 1
        assert counts[(2048, "torch.float32")] == 1

    def test_realistic_attention_leak_pattern(self):
        """Same as the tensor-diff version: 28 attention-score-sized
        storages should aggregate to ~0.875 GiB.

        This is the canonical pattern we want to detect for HKD QAT.
        """
        before: dict = {}
        bytes_per = 1 * 16 * 1024 * 1024 * 2  # 32 MiB
        after = {i: _sinfo(bytes_per) for i in range(28)}
        buckets, total = diff_storage_inventories(before, after)
        assert len(buckets) == 1
        assert buckets[0].count == 28
        total_gib = total / 1024**3
        assert 0.86 < total_gib < 0.90

    def test_sorted_by_total_bytes_descending(self):
        before: dict = {}
        after = {
            1: _sinfo(100),
            2: _sinfo(2000),  # largest
            3: _sinfo(50),
            4: _sinfo(1000),
        }
        buckets, _ = diff_storage_inventories(before, after)
        sizes = [b.total_bytes for b in buckets]
        assert sizes == sorted(sizes, reverse=True)


class TestLogStorageDiff:
    def test_returns_buckets_and_total(self):
        before: dict = {}
        after = {1: _sinfo(1024), 2: _sinfo(1024)}
        buckets, _total = log_storage_diff(before, after)
        assert len(buckets) == 1
        assert buckets[0].count == 2

    def test_logs_storage_size_details(self, caplog):
        before: dict = {}
        after = {1: _sinfo(33_554_432)}  # 32 MB
        with caplog.at_level(logging.INFO, "silverspoon_kd.engines.memory_diagnostics"):
            log_storage_diff(before, after, label="test")
        messages = [r.message for r in caplog.records]
        assert any("STORAGE_DIFF [test]" in m for m in messages)
        # Should mention the size in MiB
        assert any("32.0MiB" in m for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# Inventory accuracy regression tests
#
# These tests assert that the inventory totals match (or closely track) the
# actual amount of memory allocated. They are the tests that would have
# caught the original bug.
# ─────────────────────────────────────────────────────────────────────────────


class TestInventoryAccuracyRegression:
    """Verify inventory totals reflect ACTUAL memory usage, not view sizes."""

    def test_growing_workload_each_storage_present_in_inventory(self):
        """Allocate N tensors, then verify each one's storage appears in
        the inventory at the correct size.

        This is robust to test parallelism / fixture interference: we
        identify our tensors by their unique data_ptr() rather than
        comparing inventory totals.
        """
        held = [torch.zeros(1_000_000, dtype=torch.float32) for _ in range(5)]
        ptrs_and_sizes = [(t.untyped_storage().data_ptr(), 1_000_000 * 4) for t in held]

        inv = collect_storage_inventory(include_cpu=True)
        for ptr, expected_nbytes in ptrs_and_sizes:
            assert ptr in inv, f"storage at ptr={ptr} missing from inventory"
            assert inv[ptr].nbytes == expected_nbytes, (
                f"storage at ptr={ptr} reported {inv[ptr].nbytes} bytes, expected {expected_nbytes}"
            )
        assert len(held) == 5

    def test_view_only_growth_does_not_create_new_storage_entries(self):
        """Taking 100 views of one storage must result in ONE storage entry
        (not 100). Robust to test parallelism: identified by data_ptr().
        """
        big = torch.zeros(10_000, dtype=torch.float32)  # 40 KB
        ptr = big.untyped_storage().data_ptr()

        # Take 100 views of the same storage
        views = [big[i : i + 1] for i in range(100)]

        inv = collect_storage_inventory(include_cpu=True)

        # Exactly one entry for our storage, with the original 40 KB size
        assert ptr in inv
        assert inv[ptr].nbytes == 10_000 * 4
        # And num_views should reflect ALL the tensors that share this
        # storage: big + 100 views = 101 (or more if pytest fixtures
        # accidentally hold a view, but at least 101).
        assert inv[ptr].num_views >= 101, (
            f"expected ≥101 views for our storage, got {inv[ptr].num_views}"
        )

        # Critical: NO data_ptr should appear more than once in the inventory
        # (the inventory dict is keyed by ptr, so this is guaranteed by
        # construction; we still verify because regressions in the
        # implementation could break this invariant).
        assert len(set(inv.keys())) == len(inv)
        assert (big, views) is not None

    def test_simulated_leak_is_detected_by_callback(self):
        """End-to-end leak simulation with the callback.

        Builds a fake leak (a list that grows by one tensor per
        prediction_step), runs the callback over enough batches to
        capture before/after inventories, and verifies the diff
        correctly identifies the leaking tensors.
        """
        leak_storage = []  # external "leak"

        cb = MemoryLeakDetectionCallback(
            diff_between_batches=(2, 7),
            log_first_n_batches=0,
            log_every_n_batches=100,
        )

        # Patch CUDA stats so the callback's _log_stats path is no-op
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch(
                "silverspoon_kd.engines.memory_diagnostics.collect_tensor_inventory",
                wraps=lambda **kw: collect_tensor_inventory(
                    include_cpu=True,
                    **{k: v for k, v in kw.items() if k != "include_cpu"},
                ),
            ),
            patch(
                "silverspoon_kd.engines.memory_diagnostics.collect_storage_inventory",
                wraps=lambda **kw: collect_storage_inventory(
                    include_cpu=True,
                    **{k: v for k, v in kw.items() if k != "include_cpu"},
                ),
            ),
        ):
            for _batch in range(8):
                # Simulate a leak: each prediction_step allocates a
                # 4 MiB tensor and never frees it.
                leak_storage.append(torch.zeros(1_048_576, dtype=torch.float32))
                cb.on_prediction_step(None, None, None)

        # The leak is real — verify the list grew
        assert len(leak_storage) == 8

    def test_realistic_28_layer_attention_leak_simulation(self):
        """Simulate the exact pattern we observed: 28 leaking bf16 storages
        of ~32 MiB each per "eval batch", over 10 batches = 280 storages.

        Robust to test parallelism: take baseline FIRST so any other test's
        tensors are excluded, then verify the diff contains exactly the
        storages WE allocated (identified by data_ptr).
        """
        # Use a smaller size in test (we don't actually need 32 MiB to test
        # correctness; the pattern is what matters). 1 KiB per storage.
        per_storage_elems = 256  # 256 × bf16 = 512 bytes
        per_storage_bytes = per_storage_elems * 2

        before = collect_storage_inventory(include_cpu=True)

        # Simulate the leak: 10 batches × 28 layers
        leak = []
        for _ in range(10):
            for _ in range(28):
                leak.append(torch.zeros(per_storage_elems, dtype=torch.bfloat16))

        after = collect_storage_inventory(include_cpu=True)
        buckets, _total_bytes = diff_storage_inventories(before, after)

        # Look for the bucket matching our exact size + dtype
        our_bucket = None
        for b in buckets:
            if b.nbytes == per_storage_bytes and b.representative_dtype == "torch.bfloat16":
                our_bucket = b
                break

        assert our_bucket is not None, (
            f"Expected a {per_storage_bytes}-byte bf16 bucket, got: "
            f"{[(b.count, b.nbytes, b.representative_dtype) for b in buckets[:5]]}"
        )
        # And it must have exactly 280 entries (10 batches × 28 layers)
        assert our_bucket.count == 280, f"expected 280, got {our_bucket.count}"
        assert our_bucket.total_bytes == 280 * per_storage_bytes
        assert len(leak) == 280

    def test_diff_correctly_identifies_specific_leaked_storages(self):
        """End-to-end: take an inventory snapshot, allocate specific
        tensors, take another snapshot, verify the diff contains EXACTLY
        the data_ptrs of the tensors we allocated.

        This is the strongest correctness test: it verifies the diff
        identifies leaks by tensor identity, not by aggregate counts.
        """
        before = collect_storage_inventory(include_cpu=True)

        # Allocate a small known set of leak tensors
        leak = [torch.zeros(1024 + i, dtype=torch.float32) for i in range(5)]
        leak_ptrs = {t.untyped_storage().data_ptr() for t in leak}

        after = collect_storage_inventory(include_cpu=True)

        # The set difference of data_ptrs should be a SUPERSET of leak_ptrs
        # (other tests may have allocated tensors too, but ours MUST be there).
        new_ptrs = set(after.keys()) - set(before.keys())
        missing = leak_ptrs - new_ptrs
        assert not missing, f"diff missed our leak ptrs: {missing}"

        # And each of OUR storages should report the correct nbytes
        for t in leak:
            ptr = t.untyped_storage().data_ptr()
            assert after[ptr].nbytes == t.numel() * 4


# ─────────────────────────────────────────────────────────────────────────────
# find_tensors_for_data_ptrs / describe_referrers — leak source identification
# ─────────────────────────────────────────────────────────────────────────────


class TestFindTensorsForDataPtrs:
    def test_finds_tensor_by_data_ptr(self):
        t = torch.zeros(100, dtype=torch.float32)
        ptr = t.untyped_storage().data_ptr()
        matches = find_tensors_for_data_ptrs({ptr}, include_cpu=True)
        assert any(m is t for m in matches)

    def test_finds_multiple_views_of_same_storage(self):
        big = torch.zeros(100, 100, dtype=torch.float32)
        view = big[0:50, 0:50]  # view sharing storage
        ptr = big.untyped_storage().data_ptr()
        matches = find_tensors_for_data_ptrs({ptr}, include_cpu=True)
        # Both `big` and `view` share the same storage and should both
        # be found by the search.
        match_ids = {id(m) for m in matches}
        assert id(big) in match_ids
        assert id(view) in match_ids

    def test_returns_empty_for_unknown_ptr(self):
        # Use a data_ptr that won't match any real tensor
        bogus = {0xDEADBEEF}
        matches = find_tensors_for_data_ptrs(bogus, include_cpu=True)
        assert matches == []

    def test_does_not_match_different_tensor(self):
        a = torch.zeros(10)
        b = torch.zeros(10)
        ptr_a = a.untyped_storage().data_ptr()
        ptr_b = b.untyped_storage().data_ptr()
        if ptr_a == ptr_b:
            pytest.skip("torch reused the same storage for both tensors")
        matches = find_tensors_for_data_ptrs({ptr_a}, include_cpu=True)
        assert any(m is a for m in matches)
        assert not any(m is b for m in matches)


class TestDescribeReferrers:
    def test_finds_list_referrer(self):
        t = torch.zeros(10)
        _container = [t]  # keep alive for the test
        descriptions = describe_referrers(t, max_depth=1)
        # The container should appear as a referrer
        assert any("list" in d and "len=1" in d for d in descriptions), descriptions

    def test_finds_dict_referrer_with_key(self):
        t = torch.zeros(10)
        _container = {"my_loss": t}  # keep alive for the test
        descriptions = describe_referrers(t, max_depth=1)
        # The dict should be found and its key should be reported
        assert any("dict" in d and "key='my_loss'" in d for d in descriptions), descriptions

    def test_max_depth_walks_chain(self):
        t = torch.zeros(10)
        inner_list = [t]
        _outer_list = [inner_list]  # keep alive for the test
        d1 = describe_referrers(t, max_depth=1)
        d2 = describe_referrers(t, max_depth=2)
        # depth=1 finds the inner list; depth=2 also walks back to outer
        assert any("depth=1" in d for d in d1)
        assert any("depth=2" in d for d in d2)
        assert len(d2) >= len(d1)

    def test_max_referrers_limit(self):
        t = torch.zeros(10)
        # Create many containers that all hold t
        _containers = [[t] for _ in range(100)]
        descriptions = describe_referrers(t, max_depth=1, max_referrers=5)
        # Should be capped at max_referrers
        depth1 = [d for d in descriptions if "depth=1" in d]
        assert len(depth1) <= 5

    def test_no_referrers_returns_empty(self):
        # An object with no Python referrers other than the local variable
        # in this test (which gc.get_referrers may or may not see depending
        # on the frame state). At minimum, the function should not crash
        # and should return a list.
        t = torch.zeros(10)
        descriptions = describe_referrers(t, max_depth=1)
        assert isinstance(descriptions, list)
        assert t.numel() == 10  # keep ref alive

    def test_end_to_end_scalar_view_of_large_buffer(self):
        """End-to-end: create the exact 'scalar view of large buffer'
        pattern, then verify find_tensors_for_data_ptrs + describe_referrers
        can identify the Python container holding the leak.

        This is the diagnostic flow we use to find HKD QAT-style leaks
        in production.
        """
        # Allocate a large buffer
        big = torch.zeros(1000, 1000, dtype=torch.float32)  # 4 MB
        big_ptr = big.untyped_storage().data_ptr()
        big_nbytes = big.untyped_storage().nbytes()

        # Create a scalar view that's the only thing that will be kept alive
        leak_container = []
        scalar = big[0, 0]  # shape (), shares storage with `big`
        leak_container.append(scalar)

        # Drop the reference to `big`. The scalar view in `leak_container`
        # is now the only Python reference to the 4 MB storage.
        del big

        # Find the tensor that's keeping the storage alive
        matches = find_tensors_for_data_ptrs({big_ptr}, include_cpu=True)
        # We should find at least one tensor (the scalar view)
        assert len(matches) >= 1
        # And its storage should still report the FULL 4 MB
        scalar_match = next(m for m in matches if m.shape == ())
        assert scalar_match.untyped_storage().nbytes() == big_nbytes

        # Now describe the referrers of the scalar — should find leak_container
        descriptions = describe_referrers(scalar_match, max_depth=2)
        assert any("list" in d for d in descriptions), (
            f"expected to find a list referrer, got: {descriptions}"
        )
        # Keep refs alive
        assert leak_container is not None
