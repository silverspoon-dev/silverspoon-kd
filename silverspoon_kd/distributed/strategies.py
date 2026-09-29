"""Modular teacher placement strategies.

Each strategy is a small focused function that places a teacher model
using PyTorch-native distributed primitives.
"""

import contextlib
import functools
import logging
from typing import Any, cast

import torch
from accelerate.utils import send_to_device
from torch import nn

logger = logging.getLogger(__name__)

# Fallback threshold for size-based FSDP wrapping when no wrap_cls is found.
_SIZE_BASED_WRAP_MIN_PARAMS = 100_000_000


@contextlib.contextmanager
def _cuda_device_ctx(device_index: int):
    """Temporarily switch ``torch.cuda.current_device()`` and restore on exit."""
    prev = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    try:
        yield
    finally:
        torch.cuda.set_device(prev)


# ---------------------------------------------------------------------------
# String strategies (all-ranks, no GPU splitting)
# ---------------------------------------------------------------------------


def place_teacher_replicated(teacher: nn.Module, device: torch.device) -> nn.Module:
    """Move the full teacher copy to the given device (default strategy)."""
    teacher.to(device)
    return teacher


def shard_teacher_fsdp_all_ranks(
    teacher: nn.Module,
    device_id: int,
    wrap_cls: str | list[str] | None = None,
) -> nn.Module:
    """FSDP ``full_shard`` teacher across ALL ranks (default process group).

    Uses the model's ``_no_split_modules`` for wrapping granularity, or
    falls back to a size-based policy.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    policy = _build_wrap_policy(teacher, wrap_cls)
    teacher = FSDP(
        teacher,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        auto_wrap_policy=policy,
    )
    return teacher


# ---------------------------------------------------------------------------
# Split-GPU strategies (dedicated teacher GPUs)
# ---------------------------------------------------------------------------


def place_teacher_pp(
    teacher: nn.Module,
    remapped_teacher_devices: list[int],
    device_type: str = "cuda",
) -> nn.Module:
    """Pipeline-style placement of one teacher copy across dedicated devices.

    The teacher's layer stack — its largest ``ModuleList``/``Sequential``,
    e.g. ``model.layers`` — is split into contiguous, parameter-balanced
    chunks, one per device.  Modules registered before the stack
    (embeddings) go on the first device and modules registered after it
    (final norm, output head) on the last.  A model without such a stack
    is split over its top-level children instead.  Pre-forward hooks move
    each placed submodule's inputs to its device, so a single forward call
    runs across the devices without changes to the model code.
    """
    if len(remapped_teacher_devices) == 1:
        teacher.to(f"{device_type}:{remapped_teacher_devices[0]}")
        return teacher

    device_map = _build_balanced_device_map(teacher, remapped_teacher_devices, device_type)
    return _apply_device_map(teacher, device_map)


def parallelize_teacher_tp(
    teacher: nn.Module,
    remapped_teacher_devices: list[int],
    device_type: str = "cuda",
) -> nn.Module:
    """Tensor-parallel placement using ``torch.distributed.tensor.parallel``.

    Requires the model to expose a ``_tp_plan`` attribute (dict mapping
    module path → column/row parallel style).

    Each process shards the teacher on its own remapped teacher GPU.
    ``len(remapped_teacher_devices)`` must equal ``world_size``.
    """
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.parallel import parallelize_module

    tp_plan = getattr(teacher, "_tp_plan", None)
    if tp_plan is None:
        raise ValueError(
            "Teacher model does not expose a _tp_plan attribute. "
            "Tensor-parallel placement requires an explicit parallelization plan."
        )

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    teacher_gpu = remapped_teacher_devices[rank]

    with _cuda_device_ctx(teacher_gpu):
        teacher.to(f"{device_type}:{teacher_gpu}")

        # DeviceMesh expects process ranks, not device indices.
        # We use ranks [0..world_size-1] and rely on torch.cuda.set_device()
        # above to route each rank's operations to its teacher GPU.
        mesh = DeviceMesh(device_type, list(range(world_size)))
        parallelize_module(teacher, mesh, tp_plan)
    return teacher


def shard_teacher_fsdp_split(
    teacher: nn.Module,
    remapped_teacher_devices: list[int],
    device_type: str = "cuda",
    wrap_cls: str | list[str] | None = None,
) -> nn.Module:
    """FSDP ``full_shard`` teacher on dedicated (split) GPUs.

    Creates a separate process group for teacher FSDP and shards across
    the remapped teacher GPU indices.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    teacher_gpu = remapped_teacher_devices[rank]

    # All ranks participate — create a group from all ranks
    teacher_group = torch.distributed.new_group(ranks=list(range(world_size)))

    policy = _build_wrap_policy(teacher, wrap_cls)
    teacher = FSDP(
        teacher,
        process_group=teacher_group,  # type: ignore[arg-type]
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.device(f"{device_type}:{teacher_gpu}"),
        auto_wrap_policy=policy,
    )
    return teacher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_wrap_policy(
    model: nn.Module,
    wrap_cls: str | list[str] | None = None,
):
    """Build an FSDP auto-wrap policy.

    Resolution order:
    1. Explicit ``wrap_cls`` class name(s)
    2. ``model._no_split_modules`` (HuggingFace convention)
    3. Size-based fallback (100M params)
    """
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    target_classes = _resolve_wrap_classes(model, wrap_cls)
    if target_classes:
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=target_classes,
        )
    return functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=_SIZE_BASED_WRAP_MIN_PARAMS,
    )


def _resolve_wrap_classes(
    model: nn.Module,
    wrap_cls: str | list[str] | None = None,
) -> set:
    """Resolve class names to actual classes found in the model."""
    names = []
    if wrap_cls is not None:
        names = [wrap_cls] if isinstance(wrap_cls, str) else list(wrap_cls)
    elif hasattr(model, "_no_split_modules") and model._no_split_modules:
        names = list(model._no_split_modules)  # type: ignore[arg-type]

    if not names:
        return set()

    # Build name→class mapping from all modules in the model
    cls_map: dict[str, type] = {}
    for m in model.modules():
        cls_map[type(m).__name__] = type(m)

    resolved = set()
    for name in names:
        if name in cls_map:
            resolved.add(cls_map[name])
        else:
            logger.warning("wrap_cls %r not found in model modules, skipping.", name)
    return resolved


def _find_layer_stack(model: nn.Module) -> tuple[str, nn.Module] | None:
    """Return ``(path, container)`` of the largest layer stack in ``model``.

    A layer stack is a ``ModuleList`` or ``Sequential`` with at least two
    children; among several, the one holding the most parameters wins.
    """
    best: tuple[str, nn.Module] | None = None
    best_params = -1
    for path, module in model.named_modules():
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) >= 2:
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > best_params:
                best, best_params = (path, module), num_params
    return best


def _assign_contiguous(
    model: nn.Module,
    names: list[str],
    devices: list[int],
    device_type: str,
) -> dict[str, torch.device]:
    """Split ``names`` (in order) into contiguous chunks balanced by parameter count."""
    counts = [sum(p.numel() for p in model.get_submodule(n).parameters()) for n in names]
    total = sum(counts)
    n_dev = len(devices)
    device_map: dict[str, torch.device] = {}
    if total == 0:
        for i, name in enumerate(names):
            device_map[name] = torch.device(f"{device_type}:{devices[i * n_dev // len(names)]}")
        return device_map
    running = 0
    for name, count in zip(names, counts, strict=True):
        # A module goes to the device whose share of the parameter budget
        # contains the module's midpoint; the midpoint is monotonic, so the
        # chunks are contiguous.
        idx = min(n_dev - 1, int(n_dev * (running + count / 2) / total))
        device_map[name] = torch.device(f"{device_type}:{devices[idx]}")
        running += count
    return device_map


def _build_balanced_device_map(
    model: nn.Module,
    devices: list[int],
    device_type: str,
) -> dict[str, torch.device]:
    """Build the submodule → device map used by :func:`place_teacher_pp`.

    See :func:`place_teacher_pp` for the placement rules.
    """
    stack = _find_layer_stack(model)
    top_level = [name for name, _ in model.named_children()]
    if stack is None or len(list(stack[1].children())) < len(devices):
        if not top_level:
            return {"": torch.device(f"{device_type}:{devices[0]}")}
        return _assign_contiguous(model, top_level, devices, device_type)

    stack_path, container = stack
    first = torch.device(f"{device_type}:{devices[0]}")
    last = torch.device(f"{device_type}:{devices[-1]}")
    device_map: dict[str, torch.device] = {}

    # Walk from the root down to the stack.  At every level, siblings
    # registered before the branch towards the stack go on the first device,
    # siblings registered after it on the last.
    parent = model
    prefix = ""
    for part in stack_path.split(".") if stack_path else []:
        before = True
        for name, _ in parent.named_children():
            if name == part:
                before = False
                continue
            device_map[f"{prefix}{name}"] = first if before else last
        parent = getattr(parent, part)
        prefix = f"{prefix}{part}."

    layer_names = [f"{prefix}{name}" for name, _ in container.named_children()]
    device_map.update(_assign_contiguous(model, layer_names, devices, device_type))
    return device_map


def _apply_device_map(
    model: nn.Module,
    device_map: dict[str, torch.device],
) -> nn.Module:
    """Move each submodule to its assigned device per the device_map.

    Adds pre-forward hooks to move input tensors (including tensors nested
    in tuples, lists and dicts) to the target device automatically,
    enabling cross-device data flow during forward passes.

    For container modules (ModuleList, Sequential, ModuleDict) that are
    iterated rather than called, hooks are registered on each child
    instead of the container itself.

    Safe to call multiple times: previous device-map hooks are removed
    before new ones are registered.
    """

    def _register_hook(module: nn.Module, device: torch.device) -> None:
        # Remove any previously registered device-map hooks
        for key, hook_fn in list(module._forward_pre_hooks.items()):
            if getattr(hook_fn, "_skd_device_hook", False):
                del module._forward_pre_hooks[key]
        module.register_forward_pre_hook(_make_device_hook(device), with_kwargs=True)

    for name, device in device_map.items():
        submodule = model if name == "" else model.get_submodule(name)
        submodule.to(device)
        if isinstance(submodule, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
            for child in submodule.children():
                _register_hook(child, device)
        else:
            _register_hook(submodule, device)
    return model


def _make_device_hook(device: torch.device):
    """Create a pre-forward hook that moves all tensor inputs to ``device``."""

    def hook(
        module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        moved_args = cast(tuple[Any, ...], send_to_device(args, device))
        moved_kwargs = cast(dict[str, Any], send_to_device(kwargs, device))
        return moved_args, moved_kwargs

    hook._skd_device_hook = True  # type: ignore[attr-defined]
    return hook


def install_tp_device_hooks(teacher: nn.Module) -> None:
    """Install hooks to switch ``torch.cuda.current_device()`` around teacher forward.

    TP (via ``DeviceMesh``) checks ``torch.cuda.current_device()`` during
    DTensor operations. When the teacher is on a different GPU than the
    student, we must switch to the teacher GPU before forward and restore
    the student GPU after.

    Also moves input tensors to the teacher device via a pre-forward hook
    with kwargs support, so that inputs from the student device are
    transparently transferred.
    """
    try:
        teacher_device = next(teacher.parameters()).device
    except StopIteration:
        return

    if teacher_device.type != "cuda":
        return

    teacher_gpu = teacher_device.index

    def _pre_hook(module, args, kwargs):
        module._prev_cuda_device = torch.cuda.current_device()
        torch.cuda.set_device(teacher_gpu)
        # Move input tensors to teacher device
        new_args = tuple(
            a.to(f"cuda:{teacher_gpu}") if isinstance(a, torch.Tensor) else a for a in args
        )
        new_kwargs = {
            k: v.to(f"cuda:{teacher_gpu}") if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return new_args, new_kwargs

    def _post_hook(module, args, output):
        prev = getattr(module, "_prev_cuda_device", None)
        if prev is not None:
            torch.cuda.set_device(prev)

    teacher.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    teacher.register_forward_hook(_post_hook)
