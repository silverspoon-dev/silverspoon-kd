"""
Utility functions for alignment creation and checkpoint management.
"""

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers import PreTrainedModel

from ..utils import _ModelModuleNameMapper, get_modules_by_names
from .alignment import Alignment, block_module_name
from .output_selector import OutputSelector

# Optional dependency; TYPE_CHECKING import keeps pyright narrowing
# correct when the runtime import below fails.
if TYPE_CHECKING:
    from safetensors.torch import load_file

try:
    from safetensors.torch import load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def create_alignments(
    teacher_model: PreTrainedModel | nn.Module,
    student_model: PreTrainedModel | nn.Module,
    modules: str | list[str] | dict[str, str],
    loss_function: str | Callable | None = None,
    loss_function_kwargs: dict[str, Any] | None = None,
    output_selector_index: int | None = None,
    auto_projector: bool = True,
    projector_init: str | Callable | None = None,
    max_grad_norm: float | int | None = None,
) -> list[Alignment]:
    """Create [Alignment][silverspoon_kd.Alignment] instances between teacher and student modules.

    This is the primary API for creating alignments. Module names are matched
    using regex patterns against both models' named modules.

    Args:
        teacher_model: The teacher model
        student_model: The student model (singular)
        modules: Module matching specification:
            - str: Single regex matched against both teacher and student module names
            - List[str]: Multiple regexes, each matched against both models
            - Dict[str, str]: Teacher regex -> student replacement pattern
              (supports backreferences like \\1, \\2)
        loss_function: Optional loss function (string name or callable). Defaults to MSE.
        loss_function_kwargs: Optional keyword arguments forwarded to the loss
            factory when ``loss_function`` is a string name.  Pass model
            attributes (weights, sub-modules) directly::

                loss_function="mahal_mse",
                loss_function_kwargs={
                    "weight_matrix": teacher.lm_head.weight,
                },
        output_selector_index: Optional index for OutputSelector. When provided, sets
            both teacher_output_selector and student_output_selector to extract the element at this
            index from tuple outputs.
        auto_projector: If True (default), automatically infer and create projectors
            based on shape mismatches detected during the first forward pass.
        projector_init: Initialization for auto-created projectors.
            ``None`` (default, kaiming), ``"normal"`` (std=0.02), ``"xavier"``,
            or a callable ``fn(module)``.
        max_grad_norm: Maximum gradient norm for per-student clipping.

    Returns:
        List of [Alignment][silverspoon_kd.Alignment] instances

    Raises:
        ValueError: If no modules match the patterns in either model

    Example:
        ```python
        # Match same module names in both models
        alignments = create_alignments(teacher, student, r"model\\.layers\\.\\d+")
        # Match specific layers
        alignments = create_alignments(teacher, student, ["model.layers.0", "model.layers.1"])
        # Map teacher to student with different names
        alignments = create_alignments(teacher, student, {
            r"teacher\\.layers\\.(\\d+)": r"student\\.blocks\\.\\1"
        })
        # Mahalanobis loss with teacher weight matrix
        alignments = create_alignments(
            teacher, student, r"model\\.layers\\.\\d+",
            loss_function="mahal_mse",
            loss_function_kwargs={"weight_matrix": teacher.lm_head.weight},
        )
        ```
    """
    teacher_model_name = getattr(teacher_model, "name_or_path", teacher_model.__class__.__name__)
    student_model_name = getattr(student_model, "name_or_path", student_model.__class__.__name__)

    teacher_mapper = _ModelModuleNameMapper(teacher_model)
    student_mapper = _ModelModuleNameMapper(student_model)

    # Normalize modules to a list of
    # (teacher_module, student_module, teacher_name, student_name) tuples.
    pairs: list[tuple[nn.Module, nn.Module, str, str]] = []

    if isinstance(modules, str):
        modules = [modules]

    if isinstance(modules, list):
        # List of regex patterns: match against both models
        teacher_modules = get_modules_by_names(teacher_model, modules)
        for teacher_module in teacher_modules:
            teacher_name = teacher_mapper.get_name(teacher_module)
            student_modules_matched = get_modules_by_names(student_model, [teacher_name])
            if not student_modules_matched:
                raise ValueError(
                    f"Could not find student module matching pattern '{teacher_name}' "
                    f"in student model ({student_model_name})"
                )
            student_module = student_modules_matched[0]
            student_name = student_mapper.get_name(student_module)
            pairs.append((teacher_module, student_module, teacher_name, student_name))
    else:
        # Dict: teacher regex -> student replacement pattern
        for teacher_pattern, student_replacement in modules.items():
            teacher_modules = get_modules_by_names(teacher_model, [teacher_pattern])
            teacher_regex = re.compile(teacher_pattern)

            for teacher_module in teacher_modules:
                teacher_name = teacher_mapper.get_name(teacher_module)

                # Check for backreferences
                if "\\" in student_replacement and any(
                    f"\\{i}" in student_replacement for i in range(1, 10)
                ):
                    match = teacher_regex.search(teacher_name)
                    if match:
                        student_pattern = teacher_regex.sub(student_replacement, teacher_name)
                    else:
                        raise ValueError(
                            f"Teacher module name '{teacher_name}' did not match "
                            f"teacher pattern '{teacher_pattern}'"
                        )
                else:
                    student_pattern = student_replacement

                student_modules_matched = get_modules_by_names(student_model, [student_pattern])
                if not student_modules_matched:
                    raise ValueError(
                        f"Could not find student module matching "
                        f"pattern '{student_pattern}' in student "
                        f"model ({student_model_name})"
                    )
                student_module = student_modules_matched[0]
                student_name = student_mapper.get_name(student_module)
                pairs.append((teacher_module, student_module, teacher_name, student_name))

    if not pairs:
        raise ValueError(
            f"No modules matched the given patterns in teacher model ({teacher_model_name})"
        )

    # Build Alignment objects
    alignments: list[Alignment] = []
    for teacher_module, student_module, teacher_name, student_name in pairs:
        align_kwargs = {
            "teacher_block": teacher_module,
            "student_block": student_module,
            "teacher_model_name": teacher_model_name,
            "student_model_name": student_model_name,
            "teacher_module_name": teacher_name,
            "student_module_name": student_name,
            "auto_projector": auto_projector,
            "projector_init": projector_init,
            "loss_function": loss_function,
            "loss_function_kwargs": loss_function_kwargs,
            "max_grad_norm": max_grad_norm,
        }

        if output_selector_index is not None:
            align_kwargs["teacher_output_selector"] = OutputSelector(index=output_selector_index)
            align_kwargs["student_output_selector"] = OutputSelector(index=output_selector_index)

        alignments.append(Alignment(**align_kwargs))

    return alignments


_MODEL_FILE_SAFETENSORS = "model.safetensors"
_MODEL_FILE_TORCH = "pytorch_model.bin"
_PROJECTOR_STATE_FILE = "projector_state.pt"
_BLOCK_PREFIX = "block_"


def _load_model_state_dict(
    checkpoint_dir: Path, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """Load the model weights the Trainer wrote to ``checkpoint_dir``.

    Prefers ``model.safetensors`` over ``pytorch_model.bin``.
    """
    safetensors_path = checkpoint_dir / _MODEL_FILE_SAFETENSORS
    torch_path = checkpoint_dir / _MODEL_FILE_TORCH
    if safetensors_path.exists():
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors is required to load this checkpoint. "
                "Install it with: pip install safetensors"
            )
        return load_file(safetensors_path, device=str(device))
    if torch_path.exists():
        return torch.load(torch_path, map_location=device, weights_only=True)
    if (checkpoint_dir / f"{_MODEL_FILE_SAFETENSORS}.index.json").exists() or (
        checkpoint_dir / f"{_MODEL_FILE_TORCH}.index.json"
    ).exists():
        raise ValueError(
            f"{checkpoint_dir} holds a sharded checkpoint; load it with the model "
            f"class's ``from_pretrained`` instead."
        )
    raise FileNotFoundError(
        f"No checkpoint file found in {checkpoint_dir}. "
        f"Expected '{_MODEL_FILE_SAFETENSORS}' or '{_MODEL_FILE_TORCH}'"
    )


def _split_block_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Group a BlockwiseDistiller state dict by student block.

    Keys look like ``block_<name>.<param>``; the result maps each
    ``block_<name>`` to its ``{param: tensor}`` state.  Empty when the state
    dict holds no block entries, i.e. for an end-to-end checkpoint.
    """
    blocks: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not key.startswith(_BLOCK_PREFIX):
            continue
        block_name, _, param_name = key.partition(".")
        blocks.setdefault(block_name, {})[param_name] = value
    return blocks


def _resolve_block_modules(
    block_names: Iterable[str],
    student_model: nn.Module,
    student_model_name: str,
) -> dict[str, str]:
    """Map each ``block_<name>`` entry to the student module it was trained for.

    A block is registered under
    :func:`~silverspoon_kd.alignments.alignment.block_module_name` of the
    alignment name ``"{student_model_name}.{module_name}"``, so the module
    is found by computing that name for every module of ``student_model``.
    """
    lookup: dict[str, list[str]] = {}
    for module_name, _ in student_model.named_modules():
        if module_name:
            key = block_module_name(f"{student_model_name}.{module_name}")
            lookup.setdefault(key, []).append(module_name)

    resolved: dict[str, str] = {}
    unknown: list[str] = []
    for block_name in block_names:
        candidates = lookup.get(block_name, [])
        if len(candidates) == 1:
            resolved[block_name] = candidates[0]
        elif candidates:
            raise ValueError(
                f"Checkpoint entry '{block_name}' matches several student modules "
                f"{candidates}; module names that differ only in '.' vs '__' cannot "
                f"be told apart."
            )
        else:
            unknown.append(block_name)
    if unknown:
        raise ValueError(
            f"No weights found for student model '{student_model_name}' in checkpoint: "
            f"entries {sorted(unknown)} match no module of the given student_model. "
            f"Check that student_model_name is the name the alignments were created "
            f"with and that student_model has the architecture that was trained."
        )
    return resolved


def _load_projector_states(
    checkpoint_dir: Path, device: torch.device | str = "cpu"
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Load ``projector_state.pt`` as ``{alignment name: {role: state dict}}``.

    Returns an empty dict when the checkpoint has no projectors.
    """
    path = checkpoint_dir / _PROJECTOR_STATE_FILE
    if not path.exists():
        return {}
    return torch.load(path, map_location=device, weights_only=True)


def load_student_weights_from_checkpoint(
    student_model: PreTrainedModel | nn.Module,
    checkpoint_dir: str | Path,
    student_model_name: str,
    strict: bool = True,
) -> dict[str, list[str]]:
    """
    Load trained student weights from a distiller checkpoint into a student model.

    Handles both checkpoint layouts the distillers write:

    * :class:`~silverspoon_kd.distillers.BlockwiseDistiller` saves only the
      trained student blocks (``block_<name>.*`` entries).  Each block is
      loaded into the matching module of ``student_model``; modules that were
      not part of an alignment are left untouched.
    * :class:`~silverspoon_kd.distillers.HolisticDistiller` and
      :class:`~silverspoon_kd.distillers.ResponseBasedDistiller` save the full
      student state dict, which is loaded as a whole.

    Projectors are not loaded.  Use
    :func:`load_student_with_projectors_from_checkpoint` when blockwise-trained
    blocks must run inside the teacher together with their projectors.

    Args:
        student_model: The student model to load weights into.
        checkpoint_dir: Directory containing ``model.safetensors`` (or
            ``pytorch_model.bin``), e.g. ``./runs/checkpoint-1000``.
        student_model_name: The ``student_model_name`` the alignments were
            created with (e.g. ``'Qwen/Qwen3-0.6B'``).  Used to match blockwise
            entries to modules; ignored for end-to-end checkpoints.
        strict: If True, raise ``ValueError`` when any block (or the full
            state dict) has missing or unexpected keys.  If False, return
            them instead.

    Returns:
        Dictionary with ``'missing_keys'`` and ``'unexpected_keys'`` lists,
        qualified with the module path for blockwise checkpoints.

    Raises:
        FileNotFoundError: If the checkpoint directory or model file is missing.
        ValueError: If a blockwise entry matches no module of ``student_model``
            under ``student_model_name``, or in strict mode when keys are
            missing or unexpected.

    Example:
        ```python
        student_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        result = load_student_weights_from_checkpoint(
            student_model,
            "./runs/checkpoint-1000",
            "Qwen/Qwen3-0.6B",
        )
        student_model.save_pretrained("./trained_student")
        ```
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    state_dict = _load_model_state_dict(checkpoint_dir)
    blocks = _split_block_state(state_dict)

    missing: list[str] = []
    unexpected: list[str] = []
    if not blocks:
        # End-to-end checkpoint: the full student state dict.
        result = student_model.load_state_dict(state_dict, strict=False)
        missing.extend(result.missing_keys)
        unexpected.extend(result.unexpected_keys)
    else:
        modules = _resolve_block_modules(blocks, student_model, student_model_name)
        unexpected.extend(key for key in state_dict if not key.startswith(_BLOCK_PREFIX))
        for block_name, block_state in blocks.items():
            module_name = modules[block_name]
            result = student_model.get_submodule(module_name).load_state_dict(
                block_state, strict=False
            )
            missing.extend(f"{module_name}.{k}" for k in result.missing_keys)
            unexpected.extend(f"{module_name}.{k}" for k in result.unexpected_keys)

    if strict and (missing or unexpected):
        error_msg = []
        if missing:
            error_msg.append(f"Missing keys: {missing}")
        if unexpected:
            error_msg.append(f"Unexpected keys: {unexpected}")
        raise ValueError("Failed to load checkpoint in strict mode.\n" + "\n".join(error_msg))

    return {"missing_keys": missing, "unexpected_keys": unexpected}


class _ProjectorWrapper(nn.Module):
    """Stitches input/output projectors around a wrapped student module.

    Forwards through ``modules_list`` in order, dispatching to each
    module's ``forward`` while propagating the ``(args, kwargs)`` tuple
    convention used by input-mode projectors (whose ``forward`` returns
    ``(args, kwargs)`` instead of a tensor).
    """

    def __init__(self, modules_list):
        from .projectors import (  # local to keep top-level imports flat
            GenericConv2dProjector,
            GenericLinearProjector,
        )

        super().__init__()
        self.modules_list = nn.ModuleList(modules_list)
        self._projector_types = (GenericLinearProjector, GenericConv2dProjector)
        self.has_input_projector = any(
            isinstance(m, self._projector_types) and m.mode == "input" for m in modules_list
        )

    def forward(self, *args, **kwargs):
        """Run ``args``/``kwargs`` through every wrapped module in order."""
        result_args = args
        result_kwargs = kwargs

        for module in self.modules_list:
            is_input_projector = (
                isinstance(module, self._projector_types) and module.mode == "input"
            )
            if is_input_projector:
                # Input projector returns (args, kwargs) tuple
                result_args, result_kwargs = module(*result_args, **result_kwargs)
            else:
                # Regular module or output projector
                result = module(*result_args, **result_kwargs)
                result_args = (result,) if not isinstance(result, tuple) else result
                result_kwargs = {}

        return result_args[0] if len(result_args) == 1 else result_args


def _build_projector_from_state(
    projector_state: dict[str, torch.Tensor],
    mode: str,
    device: torch.device,
) -> nn.Module | None:
    """Build the right projector kind for ``projector_state``, if any.

    Inspects the ``"weight"`` shape to choose between Linear (2D) and
    Conv2d (4D); returns ``None`` if no usable weight is present or the
    state dict is empty.  The projector is moved to ``device`` and its
    weights are loaded.
    """
    if not projector_state:
        return None

    # Move state dict tensors to the correct device first.
    projector_state = {k: v.to(device) for k, v in projector_state.items()}

    weight = projector_state.get("weight")
    if weight is None:
        return None

    from .projectors import GenericConv2dProjector, GenericLinearProjector

    weight_shape = weight.shape
    projector: nn.Module | None = None
    if len(weight_shape) == 2:
        # Linear projector: [out_features, in_features]
        out_features, in_features = weight_shape
        projector = GenericLinearProjector(
            in_features=in_features,
            out_features=out_features,
            mode=mode,
            apply_to_arg=0 if mode == "input" else None,
        )
    elif len(weight_shape) == 4:
        # Conv2d projector: [out_channels, in_channels, kernel_h, kernel_w]
        out_channels, in_channels = weight_shape[0], weight_shape[1]
        projector = GenericConv2dProjector(
            in_channels=in_channels,
            out_channels=out_channels,
            mode=mode,
            apply_to_arg=0 if mode == "input" else None,
        )

    if projector is not None:
        projector.to(device)
        projector.load_state_dict(projector_state)
    return projector


def _install_replacement_module(
    teacher_model: nn.Module,
    teacher_mapper: _ModelModuleNameMapper,
    teacher_module_name: str,
    replacement: nn.Module,
) -> None:
    """Replace the named teacher submodule with ``replacement`` in place."""
    parent_name, _, child_name = teacher_module_name.rpartition(".")
    parent = teacher_mapper.get_module(parent_name) if parent_name else teacher_model
    setattr(parent, child_name, replacement)


def _build_student_replacement(
    block_state: dict[str, torch.Tensor],
    projector_states: dict[str, dict[str, torch.Tensor]],
    student_mapper: _ModelModuleNameMapper,
    module_name: str,
    device: torch.device,
) -> nn.Module:
    """Materialise the student block + projectors that replace one teacher module."""
    import copy

    try:
        source_student_module = student_mapper.get_module(module_name)
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"Module '{module_name}' not found in provided student_model. "
            f"Ensure student_model has the same module structure as teacher_model."
        ) from exc

    student_module = copy.deepcopy(source_student_module)
    try:
        student_module.load_state_dict(block_state)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint weights for '{module_name}' do not fit the module of the given "
            f"student_model: {exc}"
        ) from exc
    student_module = student_module.to(device)

    input_projector = _build_projector_from_state(
        projector_states.get("input_projector", {}), mode="input", device=device
    )
    output_projector = _build_projector_from_state(
        projector_states.get("output_projector", {}), mode="output", device=device
    )

    layers: list[nn.Module] = []
    if input_projector is not None:
        layers.append(input_projector)
    layers.append(student_module)
    if output_projector is not None:
        layers.append(output_projector)

    if len(layers) == 1:
        return student_module
    return _ProjectorWrapper(layers)


def load_student_with_projectors_from_checkpoint(
    teacher_model: PreTrainedModel | nn.Module,
    checkpoint_dir: str | Path,
    student_model_name: str,
    student_model: PreTrainedModel | nn.Module,
    device: torch.device | None = None,
) -> PreTrainedModel | nn.Module:
    """
    Install blockwise-distilled student blocks, with their projectors, into the teacher.

    :class:`~silverspoon_kd.distillers.BlockwiseDistiller` trains student blocks
    in isolation, with input/output projectors bridging the dimension gap to
    the teacher.  This function reads such a checkpoint, finds the teacher
    modules that were distilled from its ``block_<name>`` entries, and replaces
    each of them in place with ``[input_proj, student_block, output_proj]``
    (projectors are omitted when absent), so the returned model can be used
    directly for inference.

    Args:
        teacher_model: The teacher model; modified in place and returned.
        checkpoint_dir: Directory containing ``model.safetensors`` (or
            ``pytorch_model.bin``) and, when projectors were trained,
            ``projector_state.pt``.
        student_model_name: The ``student_model_name`` the alignments were
            created with (e.g. ``'google-bert/bert-base-uncased'``).
        student_model: A student model with the trained architecture (e.g.
            smaller hidden dims); its modules are the templates for the
            blocks.  Module names must match the teacher's.
        device: Device for the loaded modules (defaults to the teacher's).

    Returns:
        The teacher model with the student blocks and projectors installed.

    Raises:
        FileNotFoundError: If the checkpoint directory or model file is missing.
        ValueError: If the checkpoint is not a blockwise checkpoint, an entry
            matches no module of ``student_model`` under ``student_model_name``,
            or a distilled module does not exist in ``teacher_model``.

    Example:
        ```python
        from transformers import AutoModelForMaskedLM, AutoConfig
        # Load teacher
        teacher = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        # Create student with smaller dimensions
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_attention_heads = 6
        student = AutoModelForMaskedLM.from_config(config)
        # Load checkpoint with projectors
        result_model = load_student_with_projectors_from_checkpoint(
            teacher,
            "./runs/checkpoint-1000",
            "google-bert/bert-base-uncased",
            student
        )
        output = result_model(inputs)
        ```
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if device is None:
        device = next(teacher_model.parameters()).device

    state_dict = _load_model_state_dict(checkpoint_dir, device)
    blocks = _split_block_state(state_dict)
    if not blocks:
        raise ValueError(
            f"{checkpoint_dir} is not a BlockwiseDistiller checkpoint (no '{_BLOCK_PREFIX}' "
            f"entries). Students trained end-to-end are loaded with "
            f"load_student_weights_from_checkpoint."
        )
    modules = _resolve_block_modules(blocks, student_model, student_model_name)
    prefix = f"{student_model_name}."
    projectors = {
        name[len(prefix) :]: entry
        for name, entry in _load_projector_states(checkpoint_dir, device).items()
        if name.startswith(prefix)
    }

    teacher_mapper = _ModelModuleNameMapper(teacher_model)
    student_mapper = _ModelModuleNameMapper(student_model)

    # Resolve every teacher module first so that a missing module fails the
    # whole call before we mutate the teacher in place — preserves the
    # all-or-nothing semantics that callers may rely on for error recovery.
    for module_name in modules.values():
        try:
            teacher_mapper.get_module(module_name)
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                f"Module '{module_name}' found in checkpoint but does not exist in teacher model"
            ) from exc

    for block_name, module_name in sorted(modules.items(), key=lambda item: item[1]):
        replacement = _build_student_replacement(
            blocks[block_name],
            projectors.get(module_name, {}),
            student_mapper,
            module_name,
            device,
        )
        _install_replacement_module(teacher_model, teacher_mapper, module_name, replacement)

    return teacher_model
