"""
Memory diagnostics utilities for distillation workflows.

Distillation training involves hooks on many model layers, captured
intermediate tensors, and large teacher/student forward passes. These all
interact with PyTorch's autograd machinery, the CUDA caching allocator, and
Python's reference counting in ways that can produce hard-to-diagnose memory
leaks (tensors that stay alive across eval batches when they shouldn't).

``torch.profiler`` (wrapped by :mod:`silverspoon_kd.engines.profiler`) is
excellent for per-op kernel timing and per-op memory allocations, but it does
not answer the question "which tensors are Python still holding references to
at this point in time?". This module provides that missing capability:

* :func:`collect_tensor_inventory` enumerates every live CUDA tensor that
  Python currently holds a reference to, using ``gc.get_objects()``.
* :func:`diff_tensor_inventories` compares two inventories and reports which
  tensors are **new** in the later one — i.e. the ones that are leaking.
* :func:`record_cuda_memory_history` is a context manager around
  ``torch.cuda.memory._record_memory_history`` that dumps a pickle snapshot
  viewable at https://pytorch.org/memory_viz, including full allocation
  backtraces.
* :class:`MemoryLeakDetectionCallback` is a :class:`TrainerCallback` that
  applies these utilities automatically during eval, with configurable batch
  ranges for diffing and snapshotting.

Example (hooking into a distiller's eval loop):

.. code-block:: python

    from silverspoon_kd.engines.memory_diagnostics import (
        MemoryLeakDetectionCallback,
    )
    distiller.add_callback(MemoryLeakDetectionCallback(
        diff_between_batches=(5, 15),   # diff live tensors from batch 5 → 15
        snapshot_at_batch=20,           # dump allocation history at batch 20
        snapshot_path="/tmp/leak.pickle",
        log_every_n_batches=5,
    ))
"""

from __future__ import annotations

import gc
import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

import torch
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorInfo:
    """Summary of a single live CUDA tensor.

    Only contains lightweight metadata — not a reference to the tensor itself
    — so that collecting an inventory cannot itself prevent garbage collection.
    """

    shape: tuple[int, ...]
    dtype: str
    device: str
    nbytes: int


def collect_tensor_inventory(
    include_cpu: bool = False,
    force_gc: bool = True,
) -> dict[int, TensorInfo]:
    """Return a dict of ``id(tensor) -> TensorInfo`` for all live tensors.

    Uses ``gc.get_objects()`` plus ``isinstance(obj, torch.Tensor)`` to walk
    every Python-reachable tensor. Tensors that are only kept alive via C++
    references (e.g. inside custom autograd function state) will not appear.

    Args:
        include_cpu: If ``True``, CPU tensors are included too. Defaults to
            ``False`` since CPU tensors are generally cheap and the goal is
            to find GPU memory leaks.
        force_gc: If ``True``, runs ``gc.collect()`` first so that unreachable
            cycles are cleared before inventory collection. This avoids
            false positives where an object is reachable only through a
            cycle that hasn't been collected yet.

    Returns:
        A dict keyed by ``id(tensor)``. Use ``id()`` as the key so inventories
        can be diffed by identity without depending on Python's
        ``__hash__`` / ``__eq__`` semantics (which PyTorch tensors do not
        implement in the usual way).
    """
    if force_gc:
        gc.collect()

    inventory: dict[int, TensorInfo] = {}
    # gc.get_objects() can yield objects in a partially-initialized state,
    # and isinstance() probes on deprecated module attributes can
    # trigger DeprecationWarning/FutureWarning via __getattr__ hooks.
    # Wrap each probe in try/except + warning suppression so a single bad
    # object or deprecated attribute can't crash or pollute the inventory walk.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor):
                    continue
                if not include_cpu and not obj.is_cuda:
                    continue
                nbytes = obj.element_size() * obj.nelement()
                inventory[id(obj)] = TensorInfo(
                    shape=tuple(obj.shape),
                    dtype=str(obj.dtype),
                    device=str(obj.device),
                    nbytes=nbytes,
                )
            except Exception:
                continue
    return inventory


@dataclass(frozen=True)
class StorageInfo:
    """Summary of a unique CUDA tensor storage.

    Different from :class:`TensorInfo` because it deduplicates by underlying
    storage data pointer, capturing the actual memory footprint regardless
    of how many tensor *views* refer to it. This catches the common case
    where a small "view" tensor (e.g. a slice of a layer output) keeps a
    much larger underlying buffer alive.

    The ``view_shapes`` field records ALL distinct tensor shapes that share
    this storage. Inspecting this is invaluable when debugging leaks: if a
    32 MiB storage is referenced only by tensors with shape ``()`` (scalars),
    something is keeping a tiny scalar view of a large buffer alive.
    """

    nbytes: int
    device: str
    # One representative tensor's metadata; multiple tensors may share this
    # storage with different shapes/dtypes (rare but possible).
    representative_shape: tuple[int, ...]
    representative_dtype: str
    num_views: int
    # All distinct tensor shapes that share this storage (sorted, deduplicated).
    # Captured for debugging view-vs-storage mismatches.
    view_shapes: tuple[tuple[int, ...], ...] = ()


def collect_storage_inventory(
    include_cpu: bool = False,
    force_gc: bool = True,
) -> dict[int, StorageInfo]:
    """Return a dict of ``storage_data_ptr -> StorageInfo`` for live storages.

    This is the **memory-faithful** counterpart to
    :func:`collect_tensor_inventory`: it deduplicates tensors that share an
    underlying storage and reports the storage's true ``nbytes`` (which can
    be much larger than any single view's element count × element size).

    This function is the right tool for finding leaks where the visible
    Python tensors are small but the underlying CUDA memory is large —
    e.g. a 1 MB view of a 1 GB buffer that nothing else has dropped.

    Args:
        include_cpu: If ``True``, CPU storages are included.
        force_gc: If ``True``, runs ``gc.collect()`` first.

    Returns:
        Dict keyed by storage's ``data_ptr()`` (an int). Each value is a
        :class:`StorageInfo` describing the storage's true memory footprint
        and one representative tensor view.
    """
    if force_gc:
        gc.collect()

    # First pass: bucket all live CUDA/CPU tensors by their storage data_ptr,
    # collecting every distinct shape per storage so the user can spot
    # view-vs-storage mismatches at a glance.
    raw: dict[int, dict[str, Any]] = {}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor):
                    continue
                if not include_cpu and not obj.is_cuda:
                    continue
                # Get the underlying storage. Some tensors (e.g. tensors
                # created by torch.empty(0)) have no storage
                # (``RuntimeError``), and some tensor subclasses drop
                # the ``untyped_storage`` method entirely
                # (``AttributeError``).
                try:
                    storage = obj.untyped_storage()
                except (RuntimeError, AttributeError):
                    continue
                ptr = storage.data_ptr()
                if ptr == 0:
                    continue  # empty / no allocation
                shape = tuple(obj.shape)
                if ptr in raw:
                    raw[ptr]["shapes"].add(shape)
                    raw[ptr]["num_views"] += 1
                else:
                    raw[ptr] = {
                        "nbytes": storage.nbytes(),
                        "device": str(obj.device),
                        "first_shape": shape,
                        "first_dtype": str(obj.dtype),
                        "shapes": {shape},
                        "num_views": 1,
                    }
            except Exception:
                continue

    storages: dict[int, StorageInfo] = {}
    for ptr, entry in raw.items():
        storages[ptr] = StorageInfo(
            nbytes=entry["nbytes"],
            device=entry["device"],
            representative_shape=entry["first_shape"],
            representative_dtype=entry["first_dtype"],
            num_views=entry["num_views"],
            view_shapes=tuple(sorted(entry["shapes"], key=lambda s: -len(s))),
        )
    return storages


def find_tensors_for_data_ptrs(
    target_ptrs: set,
    include_cpu: bool = False,
) -> list[torch.Tensor]:
    """Return all live ``torch.Tensor`` objects whose storage data_ptr is in
    ``target_ptrs``.

    Useful for inspecting WHICH Python tensor objects are keeping a leaked
    storage alive — combine with ``gc.get_referrers()`` on the result to
    walk back to the Python container holding the reference.

    Args:
        target_ptrs: Set of integer ``storage.data_ptr()`` values to find.
        include_cpu: If ``True``, CPU tensors are also returned.

    Returns:
        A list of live tensors. The list itself is a fresh allocation;
        the tensors retain whatever existing references they have.
    """
    import warnings

    matches: list[torch.Tensor] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor):
                    continue
                if not include_cpu and not obj.is_cuda:
                    continue
                try:
                    # Same narrow exception set as in
                    # ``collect_storage_inventory``: storageless tensors
                    # raise ``RuntimeError`` and exotic subclasses can
                    # raise ``AttributeError``.
                    ptr = obj.untyped_storage().data_ptr()
                except (RuntimeError, AttributeError):
                    continue
                if ptr in target_ptrs:
                    matches.append(obj)
            except Exception:
                continue
    return matches


def describe_referrers(
    obj: object,
    max_depth: int = 2,
    max_referrers: int = 10,
) -> list[str]:
    """Return human-readable descriptions of objects referring to ``obj``.

    Walks ``gc.get_referrers(obj)`` to find Python containers holding a
    reference to ``obj``. The result is a list of short descriptive strings
    (truncated to ``max_referrers`` per level) that include the type, repr
    (truncated), and any helpful identifying info.

    Args:
        obj: The Python object to find referrers of.
        max_depth: How many levels of indirection to walk. ``1`` lists
            direct referrers; ``2`` also lists referrers-of-referrers; etc.
        max_referrers: Maximum referrers to describe per level.

    Returns:
        A flat list of description strings, one per referrer at any level
        within ``max_depth``.
    """
    descriptions: list[str] = []
    seen_ids = {id(obj)}
    current_level = [obj]

    for depth in range(1, max_depth + 1):
        next_level = []
        for item in current_level:
            referrers = gc.get_referrers(item)
            for ref in referrers[:max_referrers]:
                # Avoid descriptions of the gc machinery itself, frames, etc.
                if ref is descriptions or ref is current_level or ref is next_level:
                    continue
                if id(ref) in seen_ids:
                    continue
                seen_ids.add(id(ref))

                ref_type = type(ref).__name__
                # Build a short description with helpful context
                desc_parts = [f"depth={depth} {ref_type}"]
                if ref_type in ("list", "tuple", "set", "frozenset"):
                    desc_parts.append(f"len={len(ref)}")
                elif ref_type == "dict":
                    desc_parts.append(f"len={len(ref)}")
                    # Try to identify the key our object is stored under
                    for k, v in list(ref.items())[:50]:
                        if v is item:
                            desc_parts.append(f"key={k!r}")
                            break
                elif ref_type == "frame":
                    desc_parts.append(
                        f"file={ref.f_code.co_filename}:{ref.f_lineno} func={ref.f_code.co_name}"
                    )
                elif hasattr(ref, "__class__"):
                    cls = ref.__class__
                    if hasattr(cls, "__module__"):
                        desc_parts.append(f"module={cls.__module__}")

                descriptions.append(" ".join(desc_parts))
                next_level.append(ref)
        current_level = next_level
        if not current_level:
            break
    return descriptions[: max_referrers * max_depth]


@dataclass(frozen=True)
class StorageDiffBucket:
    """A group of new storages with the same size/device/dtype."""

    nbytes: int
    device: str
    representative_shape: tuple[int, ...]
    representative_dtype: str
    count: int
    total_bytes: int
    # Example view_shapes from one storage in this bucket — useful for
    # debugging "32 MB storage with only a scalar tensor visible" patterns.
    example_view_shapes: tuple[tuple[int, ...], ...] = ()


def diff_storage_inventories(
    before: dict[int, StorageInfo],
    after: dict[int, StorageInfo],
) -> tuple[list[StorageDiffBucket], int]:
    """Return ``(buckets, total_bytes)`` describing NEW storages in ``after``.

    Storages are grouped by ``(nbytes, device, representative_dtype)`` so
    that e.g. "28 new 32 MiB bf16 storages on cuda:0" appears as a single row.
    Buckets are sorted by total_bytes descending.

    Note: storages that **disappear** are intentionally not reported here.
    The goal is to find tensors that are leaking, not ones that were freed.
    """
    new_ids = set(after.keys()) - set(before.keys())

    bucket_accum: dict[tuple[int, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "total_bytes": 0,
            "representative_shape": (),
            "example_view_shapes": (),
        }
    )

    for ptr in new_ids:
        info = after[ptr]
        key = (info.nbytes, info.device, info.representative_dtype)
        bucket_accum[key]["count"] += 1
        bucket_accum[key]["total_bytes"] += info.nbytes
        # Capture the first representative shape + view_shapes we see for
        # this bucket so the user can spot view-vs-storage mismatches.
        if not bucket_accum[key]["representative_shape"]:
            bucket_accum[key]["representative_shape"] = info.representative_shape
            bucket_accum[key]["example_view_shapes"] = info.view_shapes

    buckets = [
        StorageDiffBucket(
            nbytes=nbytes,
            device=device,
            representative_shape=acc["representative_shape"],
            representative_dtype=dtype,
            count=acc["count"],
            total_bytes=acc["total_bytes"],
            example_view_shapes=acc["example_view_shapes"],
        )
        for (nbytes, device, dtype), acc in bucket_accum.items()
    ]
    buckets.sort(key=lambda b: -b.total_bytes)
    total_bytes = sum(b.total_bytes for b in buckets)
    return buckets, total_bytes


def log_storage_diff(
    before: dict[int, StorageInfo],
    after: dict[int, StorageInfo],
    label: str = "storage_diff",
    max_buckets: int = 30,
    log: logging.Logger | None = None,
) -> tuple[list[StorageDiffBucket], int]:
    """Pretty-print the storage diff and return the buckets/total."""
    lg = log or logger
    buckets, total_bytes = diff_storage_inventories(before, after)
    lg.info(
        "STORAGE_DIFF [%s] NEW storages: count=%d total=%.3fGiB",
        label,
        sum(b.count for b in buckets),
        total_bytes / 1024**3,
    )
    rows = buckets if max_buckets <= 0 else buckets[:max_buckets]
    for bucket in rows:
        lg.info(
            "  NEW storage %.1fMiB on %s  ×%d  total=%.1fMiB  (rep: %s %s)",
            bucket.nbytes / 1024**2,
            bucket.device,
            bucket.count,
            bucket.total_bytes / 1024**2,
            bucket.representative_shape,
            bucket.representative_dtype,
        )
        # Show all distinct tensor shapes that share one of these storages.
        # If a 32 MiB storage is referenced only by tensors with shape ``()``
        # (scalars), this row will show ``view_shapes=((),)`` — a smoking gun
        # for "scalar view of a large buffer keeping it alive".
        if bucket.example_view_shapes:
            lg.info(
                "    view_shapes (in one example storage of this bucket): %s",
                list(bucket.example_view_shapes),
            )
    if 0 < max_buckets < len(buckets):
        lg.info("  ... (%d more buckets)", len(buckets) - max_buckets)
    return buckets, total_bytes


@dataclass(frozen=True)
class TensorDiffBucket:
    """A group of new tensors with the same shape/dtype/device."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    count: int
    total_bytes: int


def diff_tensor_inventories(
    before: dict[int, TensorInfo],
    after: dict[int, TensorInfo],
) -> tuple[list[TensorDiffBucket], int]:
    """Return ``(buckets, total_bytes)`` describing NEW tensors in ``after``.

    Tensors are grouped by ``(shape, dtype, device)`` so that e.g. "28 new
    ``(1, 1024, 2048) bf16`` tensors on cuda:0" appears as a single row
    rather than 28 individual entries. Buckets are sorted by total bytes
    descending so the largest leaks come first.

    Args:
        before: Inventory snapshot taken earlier (e.g. at eval batch 5).
        after: Inventory snapshot taken later (e.g. at eval batch 15).

    Returns:
        ``(buckets, total_bytes)`` where ``buckets`` is a list of
        :class:`TensorDiffBucket` sorted by total_bytes desc, and
        ``total_bytes`` is the total bytes of all new tensors.
    """
    new_ids = set(after.keys()) - set(before.keys())

    bucket_accum: dict[tuple[tuple[int, ...], str, str], dict[str, int]] = defaultdict(
        lambda: {"count": 0, "total_bytes": 0}
    )

    for tid in new_ids:
        info = after[tid]
        key = (info.shape, info.dtype, info.device)
        bucket_accum[key]["count"] += 1
        bucket_accum[key]["total_bytes"] += info.nbytes

    buckets = [
        TensorDiffBucket(
            shape=shape,
            dtype=dtype,
            device=device,
            count=acc["count"],
            total_bytes=acc["total_bytes"],
        )
        for (shape, dtype, device), acc in bucket_accum.items()
    ]
    buckets.sort(key=lambda b: -b.total_bytes)
    total_bytes = sum(b.total_bytes for b in buckets)
    return buckets, total_bytes


def log_tensor_diff(
    before: dict[int, TensorInfo],
    after: dict[int, TensorInfo],
    label: str = "diff",
    max_buckets: int = 30,
    log: logging.Logger | None = None,
) -> tuple[list[TensorDiffBucket], int]:
    """Pretty-print the tensor diff and return the buckets/total.

    Args:
        before: Inventory snapshot taken earlier.
        after: Inventory snapshot taken later.
        label: String label used in log lines to identify this diff.
        max_buckets: Maximum number of bucket rows to log (the largest are
            logged first). Use ``0`` for no limit.
        log: Logger to write to. Defaults to this module's logger.

    Returns:
        Same as :func:`diff_tensor_inventories`.
    """
    lg = log or logger
    buckets, total_bytes = diff_tensor_inventories(before, after)
    lg.info(
        "TENSOR_DIFF [%s] NEW tensors: count=%d total=%.3fGiB",
        label,
        sum(b.count for b in buckets),
        total_bytes / 1024**3,
    )
    rows = buckets if max_buckets <= 0 else buckets[:max_buckets]
    for bucket in rows:
        lg.info(
            "  NEW %s %s on %s  ×%d  total=%.1fMiB",
            bucket.shape,
            bucket.dtype,
            bucket.device,
            bucket.count,
            bucket.total_bytes / 1024**2,
        )
    if 0 < max_buckets < len(buckets):
        lg.info("  ... (%d more buckets)", len(buckets) - max_buckets)
    return buckets, total_bytes


@contextmanager
def record_cuda_memory_history(
    snapshot_path: str,
    max_entries: int = 100_000,
) -> Iterator[None]:
    """Context manager that records CUDA allocation history and dumps a pickle.

    Uses the (private but long-stable) PyTorch memory history API:
    ``torch.cuda.memory._record_memory_history`` and ``_dump_snapshot``.
    The resulting pickle can be opened at https://pytorch.org/memory_viz
    to view a timeline of every allocation with full backtraces.

    Args:
        snapshot_path: Where to write the pickle. The parent directory is
            created if missing.
        max_entries: Maximum number of allocation events to record. The
            default is generous; reduce on memory-constrained systems.

    Yields:
        None. Allocations inside the ``with`` block are recorded.

    Example:

    .. code-block:: python

        with record_cuda_memory_history("/tmp/leak.pickle"):
            for batch in eval_loader:
                distiller.prediction_step(...)
    """
    if not torch.cuda.is_available():
        logger.warning("record_cuda_memory_history: CUDA is not available, no-op")
        yield
        return

    os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
    try:
        torch.cuda.memory._record_memory_history(enabled="all", max_entries=max_entries)
    except Exception as e:
        logger.warning("record_cuda_memory_history: could not start history: %s", e)
        yield
        return
    try:
        yield
    finally:
        try:
            torch.cuda.memory._dump_snapshot(snapshot_path)
            logger.info("CUDA memory history dumped to %s", snapshot_path)
        except Exception as e:
            logger.warning("record_cuda_memory_history: could not dump snapshot: %s", e)
        with suppress(Exception):
            torch.cuda.memory._record_memory_history(enabled=None)


def format_memory_stats(device: torch.device | None = None) -> str:
    """Return a one-line summary of current CUDA memory state.

    Format: ``alloc=X.XXXGiB reserved=X.XXXGiB peak=X.XXXGiB``

    Args:
        device: CUDA device to query. Defaults to the current device.
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserv = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return f"alloc={alloc:.3f}GiB reserved={reserv:.3f}GiB peak={peak:.3f}GiB"


# ── TrainerCallback integration ─────────────────────────────────────────────


class MemoryLeakDetectionCallback(TrainerCallback):
    """Trainer callback that runs tensor-inventory diffs during eval.

    Logs CUDA memory stats at every eval batch. Optionally:

    * Takes a tensor-inventory snapshot at ``diff_between_batches[0]``
      and diffs it against a second snapshot at ``diff_between_batches[1]``,
      logging every shape+dtype bucket that appears new.
    * Records a CUDA memory history pickle covering a window around
      ``snapshot_at_batch``, viewable at https://pytorch.org/memory_viz.

    Both features are opt-in. With their defaults of ``None``, the callback
    only logs per-batch memory stats — a lightweight sanity check that is
    cheap enough to enable by default in tests and CI.

    Args:
        diff_between_batches: Pair ``(first, second)`` of eval batch indices
            between which to diff live tensors. The inventories are taken
            at the START of each named batch (before that batch's forward
            runs) so the diff reflects state that persisted between batches.
            Set to ``None`` to disable diffing.
        snapshot_at_batch: Eval batch index at which to dump the CUDA
            memory history pickle. The recording starts 5 batches earlier
            so the batches leading up to the snapshot are covered. Set to
            ``None`` to disable snapshot dumping.
        snapshot_path: Where to write the memory history pickle. Required
            when ``snapshot_at_batch`` is set.
        log_every_n_batches: Log memory stats every N eval batches.
            Set to ``0`` to log every batch (default).
        log_first_n_batches: Always log memory stats for the first N
            batches regardless of ``log_every_n_batches`` (default 10).
            Useful for catching warmup-phase allocations.
        max_diff_buckets: Maximum diff buckets to log per diff. Use 0 for
            unlimited.
        logger_: Logger to write to. Defaults to this module's logger.
    """

    def __init__(
        self,
        diff_between_batches: tuple[int, int] | None = None,
        snapshot_at_batch: int | None = None,
        snapshot_path: str | None = None,
        log_every_n_batches: int = 0,
        log_first_n_batches: int = 10,
        max_diff_buckets: int = 30,
        logger_: logging.Logger | None = None,
    ):
        if snapshot_at_batch is not None and snapshot_path is None:
            raise ValueError("snapshot_path is required when snapshot_at_batch is set")
        if diff_between_batches is not None:
            b1, b2 = diff_between_batches
            if b1 >= b2:
                raise ValueError(
                    f"diff_between_batches must be (first, second) with "
                    f"first < second, got ({b1}, {b2})"
                )
        self._diff_between = diff_between_batches
        self._snapshot_at = snapshot_at_batch
        self._snapshot_path = snapshot_path
        self._log_every = max(0, int(log_every_n_batches))
        self._log_first = max(0, int(log_first_n_batches))
        self._max_diff_buckets = int(max_diff_buckets)
        self._logger = logger_ or logger

        # Runtime state
        self._eval_batch = 0
        self._inventory_before: dict[int, TensorInfo] | None = None
        self._storage_before: dict[int, StorageInfo] | None = None
        self._history_active = False

    # -- internal helpers ----------------------------------------------------

    def _should_log_stats(self) -> bool:
        if self._eval_batch <= self._log_first:
            return True
        if self._log_every == 0:
            return True
        return self._eval_batch % self._log_every == 0

    def _log_stats(self) -> None:
        self._logger.info(
            "MEMORY [eval_batch_%d] %s",
            self._eval_batch,
            format_memory_stats(),
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _maybe_start_history(self) -> None:
        if self._snapshot_at is None or self._history_active:
            return
        # Start recording a few batches before the snapshot so the batches
        # leading up to it are captured.
        start_at = max(self._snapshot_at - 5, 1)
        if self._eval_batch < start_at:
            return
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.memory._record_memory_history(enabled="all", max_entries=100_000)
            self._history_active = True
            self._logger.info(
                "MEMORY_SNAPSHOT: recording history from eval batch %d",
                self._eval_batch,
            )
        except Exception as e:
            self._logger.warning("MEMORY_SNAPSHOT: could not start history: %s", e)

    def _maybe_dump_snapshot(self) -> None:
        if self._snapshot_at is None or not self._history_active:
            return
        if self._eval_batch != self._snapshot_at:
            return
        assert self._snapshot_path is not None
        try:
            torch.cuda.memory._dump_snapshot(self._snapshot_path)
            self._logger.info("MEMORY_SNAPSHOT: dumped to %s", self._snapshot_path)
        except Exception as e:
            self._logger.warning("MEMORY_SNAPSHOT: could not dump: %s", e)
        with suppress(Exception):
            torch.cuda.memory._record_memory_history(enabled=None)
        self._history_active = False

    def _maybe_diff_tensors(self) -> None:
        if self._diff_between is None:
            return
        b1, b2 = self._diff_between
        if self._eval_batch == b1:
            self._logger.info("DIFF: collecting inventories at eval batch %d", b1)
            self._inventory_before = collect_tensor_inventory()
            self._storage_before = collect_storage_inventory()
            self._logger.info(
                "DIFF: %d CUDA tensors / %d unique storages at eval batch %d",
                len(self._inventory_before),
                len(self._storage_before),
                b1,
            )
        elif self._eval_batch == b2 and self._inventory_before is not None:
            self._logger.info("DIFF: collecting inventories at eval batch %d", b2)
            after_tensors = collect_tensor_inventory()
            after_storages = collect_storage_inventory()
            self._logger.info(
                "DIFF: %d CUDA tensors / %d unique storages at eval batch %d",
                len(after_tensors),
                len(after_storages),
                b2,
            )
            # Tensor-level diff (counts visible torch.Tensor instances)
            log_tensor_diff(
                self._inventory_before,
                after_tensors,
                label=f"eval_batches_{b1}_to_{b2}",
                max_buckets=self._max_diff_buckets,
                log=self._logger,
            )
            # Storage-level diff (deduplicated by data_ptr; this is the
            # memory-faithful view that catches view-vs-storage mismatches).
            if self._storage_before is not None:
                log_storage_diff(
                    self._storage_before,
                    after_storages,
                    label=f"eval_batches_{b1}_to_{b2}",
                    max_buckets=self._max_diff_buckets,
                    log=self._logger,
                )
                # For each leaked storage of significant size, walk
                # gc.get_referrers() to identify the Python container
                # that's holding the leaked tensor.
                self._describe_leaked_referrers(after_storages)

    def _describe_leaked_referrers(self, after_storages: dict[int, StorageInfo]) -> None:
        """For the largest new storages in the diff, identify the Python
        objects that hold them via gc.get_referrers().

        Skips storages smaller than 1 MiB to avoid log spam from incidental
        small allocations. Only walks the first few largest leaked storages
        to keep output manageable.
        """
        if self._storage_before is None:
            return
        new_ptrs = set(after_storages.keys()) - set(self._storage_before.keys())
        # Filter to "significant" leaks: ≥ 1 MiB
        significant = sorted(
            [(p, after_storages[p]) for p in new_ptrs if after_storages[p].nbytes >= 1024**2],
            key=lambda kv: -kv[1].nbytes,
        )
        if not significant:
            return
        # Pick the first few unique sizes to investigate
        seen_sizes = set()
        sample_ptrs = []
        for ptr, info in significant:
            if info.nbytes in seen_sizes:
                continue
            seen_sizes.add(info.nbytes)
            sample_ptrs.append(ptr)
            if len(sample_ptrs) >= 3:
                break

        target = set(sample_ptrs)
        # Find the Python tensor objects with these data_ptrs
        tensors = find_tensors_for_data_ptrs(target)
        self._logger.info(
            "REFERRERS: walking %d sample leaked tensors",
            len(tensors),
        )
        for t in tensors[:6]:  # cap to avoid log spam
            try:
                ptr = t.untyped_storage().data_ptr()
                size_mib = t.untyped_storage().nbytes() / 1024**2
                self._logger.info(
                    "  Tensor shape=%s dtype=%s storage=%.1fMiB ptr=0x%x",
                    tuple(t.shape),
                    t.dtype,
                    size_mib,
                    ptr,
                )
                refs = describe_referrers(t, max_depth=2, max_referrers=8)
                for r in refs:
                    self._logger.info("    REF: %s", r)
            except Exception as e:
                self._logger.warning("REFERRERS: error describing tensor: %s", e)

    # -- TrainerCallback hooks ------------------------------------------------

    def on_prediction_step(self, args, state, control, **kwargs) -> None:
        """Per-eval-batch hook: log stats, snapshot, and diff tensors."""
        self._eval_batch += 1
        self._maybe_start_history()
        if self._should_log_stats():
            self._log_stats()
        self._maybe_dump_snapshot()
        self._maybe_diff_tensors()

    def on_evaluate(self, args, state, control, **kwargs) -> None:
        """End-of-evaluate hook: reset per-eval-loop counters and inventories."""
        # Reset per-eval-loop counter so that if evaluate() is called
        # multiple times (e.g. at different training steps), each call
        # starts fresh.
        self._eval_batch = 0
        self._inventory_before = None
        self._storage_before = None


__all__ = [
    "MemoryLeakDetectionCallback",
    "StorageDiffBucket",
    "StorageInfo",
    "TensorDiffBucket",
    "TensorInfo",
    "collect_storage_inventory",
    "collect_tensor_inventory",
    "describe_referrers",
    "diff_storage_inventories",
    "diff_tensor_inventories",
    "find_tensors_for_data_ptrs",
    "format_memory_stats",
    "log_storage_diff",
    "log_tensor_diff",
    "record_cuda_memory_history",
]
