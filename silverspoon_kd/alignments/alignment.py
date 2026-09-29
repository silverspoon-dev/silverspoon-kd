"""
Alignment class for knowledge distillation.

Represents a single teacher-module <-> student-module pair.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn, optim
from transformers import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.training_args import OptimizerNames

from ..losses.registry import get_loss_function
from ..training_arguments import TrainingArguments
from .output_selector import OutputSelector
from .projectors import GenericConv2dProjector, GenericLinearProjector

logger = logging.getLogger(__name__)


def _identity(x: torch.Tensor) -> torch.Tensor:
    """Identity function that returns input unchanged."""
    return x


def block_module_name(alignment_name: str) -> str:
    """Return the submodule name under which a student block is registered.

    :class:`~silverspoon_kd.distillers.BlockwiseDistiller` holds the student
    blocks in a container module and saves that container as the model
    checkpoint, so every block's weights appear under this name.  ``/`` and
    ``.`` are not valid in attribute names and are replaced by ``__``.
    """
    return "block_" + alignment_name.replace("/", "__").replace(".", "__")


_default_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss()

# Stateless singleton — used as the default for ``student_output_selector``.
_default_student_output_selector = OutputSelector()


class Alignment:
    """A single teacher-module <-> student-module pair for knowledge distillation.

    This class encapsulates the complete alignment between one teacher block and
    one student block, including output selectors, projectors, loss function, optimizer,
    scheduler, and device/dtype matching configuration.
    """

    def __init__(
        self,
        teacher_block: nn.Module,
        student_block: nn.Module,
        teacher_model_name: str = "",
        student_model_name: str = "",
        teacher_module_name: str = "",
        student_module_name: str = "",
        # Output selectors
        teacher_output_selector: Callable[[torch.Tensor], torch.Tensor] = _identity,
        student_output_selector: Callable[
            [torch.Tensor], torch.Tensor
        ] = _default_student_output_selector,
        # Projectors
        input_projector: nn.Module | None = None,
        output_projector: nn.Module | None = None,
        auto_projector: bool = False,
        projector_init: str | Callable[[nn.Module], None] | None = None,
        # Training config
        loss_function: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        loss_function_kwargs: dict[str, Any] | None = None,
        optimizer: optim.Optimizer | None = None,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
        max_grad_norm: float | int | None = None,
        loss_weight: float = 1.0,
        # Device/dtype
        auto_device_match: bool = False,
        auto_dtype_match: bool = False,
    ):
        """
        Initialize an Alignment instance.

        Args:
            teacher_block: The teacher module to capture outputs from
            student_block: The student module to train
            teacher_model_name: Label for the teacher model (auto-set by
                [create_alignments][silverspoon_kd.create_alignments])
            student_model_name: Label for the student model (auto-set by
                [create_alignments][silverspoon_kd.create_alignments])
            teacher_module_name: Name of the teacher module (auto-set by
                [create_alignments][silverspoon_kd.create_alignments])
            student_module_name: Name of the student module (auto-set by
                [create_alignments][silverspoon_kd.create_alignments])
            teacher_output_selector: Function to select/extract teacher output before comparison
            student_output_selector: Function to select/extract student output before comparison
            input_projector: Optional module to project inputs before student forward pass
            output_projector: Optional module to project student output before loss computation
            auto_projector: If True, automatically infer and create projectors based on
                           shape mismatches detected during the first forward pass
            projector_init: Initialization for auto-created projectors.
                - ``None`` (default): PyTorch default (kaiming_uniform).
                - ``"normal"``: ``normal_(0, 0.02)`` (BERT/TextBrewer-style).
                - ``"xavier"``: Xavier uniform.
                - A callable ``fn(module)`` applied via ``module.apply(fn)``.
            loss_function: Loss function for alignment loss (defaults to MSE).
                Can be a callable or a string name from the loss registry.
            loss_function_kwargs: Optional keyword arguments forwarded to the
                loss factory when ``loss_function`` is a string name.  Ignored
                when ``loss_function`` is already a callable.  Example:
                ``{"temperature": 3.0}`` or
                ``{"weight_matrix": tensor}``.
            optimizer: Optimizer for training the student block. If None, auto-created.
            scheduler: Learning rate scheduler. If None, auto-created.
            max_grad_norm: Maximum gradient norm for clipping
            loss_weight: Scalar weight for this alignment's loss contribution
                when accumulating total loss. Only the ratios between weights
                matter, not their absolute values (e.g. [2, 3] gives the same
                gradient proportions as [0.4, 0.6]). Default: 1.0 (no scaling).
                Supported by HolisticDistiller; BlockwiseDistiller will warn
                if non-default weights are set.
            auto_device_match: Whether to auto-move inputs/teacher outputs to student device
            auto_dtype_match: Whether to auto-cast inputs/teacher outputs to student dtype
        """
        self.teacher_block = teacher_block
        self.student_block = student_block
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.teacher_module_name = teacher_module_name
        self.student_module_name = student_module_name
        self.teacher_output_selector = teacher_output_selector
        self.student_output_selector = student_output_selector
        self.input_projector = input_projector
        self.output_projector = output_projector
        self.auto_projector = auto_projector
        self._projector_init = projector_init
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm: float | int | None = max_grad_norm
        self.loss_weight: float = loss_weight
        self.auto_device_match = auto_device_match
        self.auto_dtype_match = auto_dtype_match

        self.id: int | None = None

        # Auto-projector state tracking
        self._auto_projector_initialized = False
        self._awaiting_input_shape = auto_projector and input_projector is None
        self._awaiting_output_shape = auto_projector and output_projector is None

        # Set default loss function if not provided
        if loss_function is None:
            self.loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = _default_loss
        elif isinstance(loss_function, str):
            self.loss_function = get_loss_function(loss_function, **(loss_function_kwargs or {}))
        else:
            self.loss_function = loss_function

    def _apply_projector_init(self, projector: nn.Module) -> nn.Module:
        """Apply custom initialization to an auto-created projector."""
        init = self._projector_init
        if init is None:
            return projector
        if callable(init) and not isinstance(init, str):
            projector.apply(init)
            return projector
        if init == "normal":
            for m in projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif init == "xavier":
            for m in projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            raise ValueError(
                f"Unknown projector_init: {init!r}. Use None, 'normal', 'xavier', or a callable."
            )
        return projector

    def get_name(self) -> str:
        """Get the display name for this alignment.

        Returns the student's model_name.module_name, which is used as the
        metric key for per-layer loss reporting.
        """
        return f"{self.student_model_name}.{self.student_module_name}"

    # -------------------------------------------------------------------------
    # Auto-projector inference
    # -------------------------------------------------------------------------

    def _get_compute_dtype(self) -> torch.dtype:
        """Get effective compute dtype of student block.

        With FSDP mixed precision, stored param dtype (fp32) differs from the
        compute dtype (e.g. bf16) used during forward.  Check the FSDP mixed
        precision config first, then fall back to the parameter dtype.
        """
        block = self.student_block
        mp = getattr(block, "mixed_precision", None)
        if mp is not None and getattr(mp, "param_dtype", None) is not None:
            return mp.param_dtype
        try:
            return next(block.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _find_all_tensors(
        self, args: tuple, kwargs: dict
    ) -> list[tuple[torch.Tensor, str, int | None, str | None]]:
        """Find the tensors passed directly in args and kwargs and record their locations."""
        found_tensors = []
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                found_tensors.append((arg, "arg", idx, None))
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                found_tensors.append((value, "kwarg", None, key))
        return found_tensors

    def _try_infer_input_projector(
        self,
        input_args: tuple,
        input_kwargs: dict,
    ) -> nn.Module | None:
        """Try to infer and create an input projector by attempting a forward pass."""
        compute_dtype = self._get_compute_dtype()
        probe_args = tuple(
            t.to(compute_dtype) if isinstance(t, torch.Tensor) else t for t in input_args
        )
        probe_kwargs = {
            k: v.to(compute_dtype) if isinstance(v, torch.Tensor) else v
            for k, v in input_kwargs.items()
        }
        try:
            with torch.no_grad():
                _ = self.student_block(*probe_args, **probe_kwargs)
            return None
        except (RuntimeError, ValueError, TypeError) as e:
            error_msg = str(e).lower()
            if not any(
                keyword in error_msg for keyword in ["size", "dimension", "shape", "mismatch"]
            ):
                raise

            expected_dim = self._infer_student_input_dim()
            if expected_dim is None:
                logger.warning(
                    "[%s] Cannot infer input projector: Unable to determine expected "
                    "input dimension from student block architecture. "
                    "Consider manually specifying input_projector.",
                    self.get_name(),
                )
                return None

            found_tensors = self._find_all_tensors(input_args, input_kwargs)
            if not found_tensors:
                logger.warning(
                    "[%s] Cannot infer input projector: No tensors found in inputs. "
                    "Student block expects dim=%s but cannot determine which input to project.",
                    self.get_name(),
                    expected_dim,
                )
                return None

            device = next(self.student_block.parameters()).device
            dtype = compute_dtype

            for tensor, location_type, arg_idx, kwarg_name in found_tensors:
                projectors_to_try = self._create_input_projectors_for_tensor(
                    tensor,
                    expected_dim,
                    location_type,
                    arg_idx,
                    kwarg_name,
                    device,
                    dtype,
                )
                for projector, projector_desc in projectors_to_try:
                    try:
                        with torch.no_grad():
                            test_args, test_kwargs = projector(*input_args, **input_kwargs)
                            _ = self.student_block(*test_args, **test_kwargs)
                        logger.info(
                            "[%s] Auto-created input projector (%s): for %s",
                            self.get_name(),
                            projector_desc,
                            ("arg[" + str(arg_idx) + "]" if location_type == "arg" else kwarg_name),
                        )
                        return projector
                    except Exception:
                        continue

            logger.warning(
                "[%s] Cannot infer input projector: Tried projecting all %s "
                "input tensor(s) to dim=%s, but none resolved the shape mismatch. "
                "You may need to manually specify input_projector "
                "or check your student block architecture.",
                self.get_name(),
                len(found_tensors),
                expected_dim,
            )
            return None

    def _create_input_projectors_for_tensor(
        self,
        tensor: torch.Tensor,
        expected_dim: int,
        location_type: str,
        arg_idx: int | None,
        kwarg_name: str | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[tuple[nn.Module, str]]:
        """Create candidate input projectors for a given tensor."""
        projectors = []

        if len(tensor.shape) == 4:
            _, channels, _, _ = tensor.shape
            if channels != expected_dim:
                if location_type == "arg":
                    projector = GenericConv2dProjector(
                        in_channels=channels,
                        out_channels=expected_dim,
                        mode="input",
                        apply_to_arg=arg_idx,
                    ).to(device=device, dtype=dtype)
                else:
                    projector = GenericConv2dProjector(
                        in_channels=channels,
                        out_channels=expected_dim,
                        mode="input",
                        apply_to_kwarg=kwarg_name,
                    ).to(device=device, dtype=dtype)
                projectors.append((projector, f"Conv2D {channels}->{expected_dim} channels"))

        if tensor.shape[-1] != expected_dim:
            if location_type == "arg":
                projector = GenericLinearProjector(
                    in_features=tensor.shape[-1],
                    out_features=expected_dim,
                    mode="input",
                    apply_to_arg=arg_idx,
                ).to(device=device, dtype=dtype)
            else:
                projector = GenericLinearProjector(
                    in_features=tensor.shape[-1],
                    out_features=expected_dim,
                    mode="input",
                    apply_to_kwarg=kwarg_name,
                ).to(device=device, dtype=dtype)
            self._apply_projector_init(projector)
            projectors.append((projector, f"Linear {tensor.shape[-1]}->{expected_dim} features"))

        return projectors

    def _try_infer_output_projector(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> nn.Module | None:
        """Try to infer and create an output projector by comparing shapes."""
        if student_output.shape == teacher_output.shape:
            return None

        try:
            device = next(self.student_block.parameters()).device
        except StopIteration:
            device = student_output.device
        # Use student_output dtype — reflects actual compute dtype
        # (handles FSDP mixed precision where stored param dtype differs)
        dtype = student_output.dtype

        if len(student_output.shape) == 4 and len(teacher_output.shape) == 4:
            _, s_channels, s_h, s_w = student_output.shape
            _, t_channels, t_h, t_w = teacher_output.shape
            if s_channels != t_channels and s_h == t_h and s_w == t_w:
                projector = GenericConv2dProjector(
                    in_channels=s_channels, out_channels=t_channels, mode="output"
                ).to(device=device, dtype=dtype)
                self._apply_projector_init(projector)
                logger.info(
                    "[%s] Auto-created output projector (Conv2D): %s -> %s channels",
                    self.get_name(),
                    s_channels,
                    t_channels,
                )
                return projector

        if (
            student_output.shape[-1] != teacher_output.shape[-1]
            and student_output.shape[:-1] == teacher_output.shape[:-1]
        ):
            projector = GenericLinearProjector(
                in_features=student_output.shape[-1],
                out_features=teacher_output.shape[-1],
                mode="output",
            ).to(device=device, dtype=dtype)
            self._apply_projector_init(projector)
            logger.info(
                "[%s] Auto-created output projector (Linear): %s -> %s features%s",
                self.get_name(),
                student_output.shape[-1],
                teacher_output.shape[-1],
                f" (init={self._projector_init})" if self._projector_init else "",
            )
            return projector

        logger.warning(
            "[%s] Cannot infer output projector: Shape mismatch "
            "student %s vs teacher %s. "
            "No simple projection pattern detected. Consider manually specifying output_projector.",
            self.get_name(),
            student_output.shape,
            teacher_output.shape,
        )
        return None

    def _infer_student_input_dim(self) -> int | None:
        """Attempt to infer the expected input dimension of the student block."""
        for module in self.student_block.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
            if isinstance(module, nn.LayerNorm) and module.normalized_shape:
                return module.normalized_shape[-1]
            if isinstance(module, nn.Conv1d):
                return module.in_channels
            if isinstance(module, nn.Conv2d):
                return module.in_channels
            if isinstance(module, nn.Embedding):
                continue
        return None

    def initialize_auto_projectors(
        self,
        input_args: tuple,
        input_kwargs: dict,
        teacher_output: torch.Tensor,
    ) -> None:
        """Initialize auto projectors on the first forward pass if auto_projector is enabled."""
        if not self.auto_projector or self._auto_projector_initialized:
            return

        if self._awaiting_input_shape and self.input_projector is None:
            inferred_projector = self._try_infer_input_projector(input_args, input_kwargs)
            if inferred_projector is not None:
                self.input_projector = inferred_projector
            self._awaiting_input_shape = False

        if self._awaiting_output_shape and self.output_projector is None:
            try:
                with torch.no_grad():
                    compute_dtype = self._get_compute_dtype()
                    test_args = tuple(
                        t.to(compute_dtype) if isinstance(t, torch.Tensor) else t
                        for t in input_args
                    )
                    test_kwargs = {
                        k: v.to(compute_dtype) if isinstance(v, torch.Tensor) else v
                        for k, v in input_kwargs.items()
                    }
                    if self.input_projector is not None:
                        from copy import deepcopy

                        test_args = deepcopy(test_args)
                        test_kwargs = deepcopy(test_kwargs)
                        test_args, test_kwargs = self.input_projector(*test_args, **test_kwargs)
                    student_output = self.student_block(*test_args, **test_kwargs)
                    student_output = self.student_output_selector(student_output)
                    inferred_projector = self._try_infer_output_projector(
                        student_output, teacher_output
                    )
                    if inferred_projector is not None:
                        self.output_projector = inferred_projector
            except Exception as e:
                logger.warning(
                    "[%s] Cannot infer output projector: %s. "
                    "Consider manually specifying output_projector.",
                    self.get_name(),
                    e,
                )
            self._awaiting_output_shape = False

        self._auto_projector_initialized = True

        if self.optimizer is not None and (
            self.input_projector is not None or self.output_projector is not None
        ):
            self._add_projector_params_to_optimizer()

    def _try_init_output_projector(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> None:
        """Initialize output projector from captured outputs.

        Used by holistic / response-based distillers.  Unlike
        ``initialize_auto_projectors`` (which needs to run the student block),
        this method works when both outputs are already available from the
        capture engines.
        """
        if self.output_projector is not None or not self._awaiting_output_shape:
            self._auto_projector_initialized = True
            return

        inferred = self._try_infer_output_projector(student_output, teacher_output)
        if inferred is not None:
            self.output_projector = inferred
        self._awaiting_output_shape = False
        self._auto_projector_initialized = True

        if self.optimizer is not None and self.output_projector is not None:
            self._add_projector_params_to_optimizer()

    def _add_projector_params_to_optimizer(self) -> None:
        """Add projector parameters to the existing optimizer."""
        if self.optimizer is None:
            return

        new_named_params = []
        if self.input_projector is not None:
            new_named_params.extend(self.input_projector.named_parameters())
        if self.output_projector is not None:
            new_named_params.extend(self.output_projector.named_parameters())

        if not new_named_params:
            return

        existing_params = set()
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                existing_params.add(id(param))

        new_named_params = [(n, p) for n, p in new_named_params if id(p) not in existing_params]

        if not new_named_params:
            return

        decay_group = self.optimizer.param_groups[0]
        no_decay_group = self.optimizer.param_groups[-1]
        for name, param in new_named_params:
            if "bias" in name:
                no_decay_group["params"].append(param)
            else:
                decay_group["params"].append(param)
            self.optimizer.state[param] = {}

    # -------------------------------------------------------------------------
    # Optimizer / scheduler creation
    # -------------------------------------------------------------------------

    def _get_param_groups(self, training_args: TrainingArguments) -> list:
        """Build optimizer param groups for this alignment (decay vs no-decay)."""
        decay_parameters = get_parameter_names(self.student_block, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        all_params = list(self.student_block.named_parameters())
        if self.input_projector is not None:
            all_params.extend(self.input_projector.named_parameters())
        if self.output_projector is not None:
            all_params.extend(self.output_projector.named_parameters())

        return [
            {
                "params": [p for n, p in all_params if n in decay_parameters],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in all_params if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

    def _create_optimizer(self, training_args: TrainingArguments) -> optim.Optimizer:
        """Create an optimizer from TrainingArguments following HuggingFace Trainer conventions."""
        optimizer_grouped_parameters = self._get_param_groups(training_args)
        optimizer_cls_and_kwargs = self._get_optimizer_cls_and_kwargs(training_args)
        optimizer_cls, optimizer_kwargs = optimizer_cls_and_kwargs

        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def _get_optimizer_cls_and_kwargs(
        self, training_args: TrainingArguments
    ) -> tuple[type[optim.Optimizer], dict[str, Any]]:
        """Get the optimizer class and kwargs based on TrainingArguments."""
        optimizer_kwargs = {"lr": training_args.learning_rate}

        adam_kwargs = {
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
        }

        if training_args.optim == OptimizerNames.ADAMW_TORCH:
            optimizer_cls = optim.AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif training_args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
            optimizer_cls = optim.AdamW
            optimizer_kwargs.update(adam_kwargs)
            optimizer_kwargs["fused"] = True
        elif training_args.optim == OptimizerNames.SGD:
            optimizer_cls = optim.SGD
            optimizer_kwargs["momentum"] = 0.0
        elif training_args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = optim.Adagrad
        elif training_args.optim == OptimizerNames.ADAFACTOR:
            from transformers.optimization import Adafactor

            optimizer_cls = Adafactor
            optimizer_kwargs["scale_parameter"] = False
            optimizer_kwargs["relative_step"] = False
        elif training_args.optim in ("adamw_bnb_8bit", "adamw_8bit"):
            try:
                import bitsandbytes as bnb

                optimizer_cls = bnb.optim.AdamW8bit  # pyright: ignore[reportPrivateImportUsage]
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to standard AdamW")
                optimizer_cls = optim.AdamW
                optimizer_kwargs.update(adam_kwargs)
        else:
            optimizer_cls = optim.AdamW
            optimizer_kwargs.update(adam_kwargs)

        return optimizer_cls, optimizer_kwargs

    def _create_scheduler(
        self,
        training_args: TrainingArguments,
        num_training_steps: int,
    ) -> optim.lr_scheduler.LRScheduler:
        """Create a learning rate scheduler from TrainingArguments."""
        assert self.optimizer is not None, "optimizer must be created before scheduler"
        return get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )

    def _prepare_for_training(
        self,
        alignment_id: int,
        training_args: TrainingArguments,
        num_training_steps: int | None = None,
        defer_optimizer: bool = False,
    ) -> None:
        """Prepare this alignment for training.

        Args:
            alignment_id: Unique index for this alignment
            training_args: Training configuration arguments
            num_training_steps: Total number of training steps
            defer_optimizer: If True, skip optimizer/scheduler creation.
                Used when the distiller defers optimizer creation to
                ``create_optimizer()`` (e.g. to support FSDP wrapping
                the student before optimizer param refs are taken).
        """
        self.id = alignment_id

        if not defer_optimizer:
            self._ensure_optimizer_and_scheduler(training_args, num_training_steps)

        if self.max_grad_norm is None:
            self.max_grad_norm = training_args.max_grad_norm

    def _ensure_optimizer_and_scheduler(
        self,
        training_args: TrainingArguments,
        num_training_steps: int | None = None,
    ) -> None:
        """Create optimizer and scheduler if not already set."""
        if self.optimizer is None:
            self.optimizer = self._create_optimizer(training_args)

        if self.scheduler is None:
            if num_training_steps is not None:
                self.scheduler = self._create_scheduler(training_args, num_training_steps)
            else:
                raise ValueError(
                    "Either provide a scheduler in __init__ or pass num_training_steps to "
                    "_prepare_for_training to auto-create the scheduler"
                )
