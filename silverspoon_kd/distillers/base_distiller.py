"""
Base distiller class for knowledge distillation trainers.

This module provides the BaseDistiller abstract base class that contains shared
functionality for all distillation approaches (response-based, holistic, blockwise).
"""

import contextlib
import inspect
import logging
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from accelerate.utils import send_to_device
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from transformers import PreTrainedModel, Trainer
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from ..alignments.alignment import Alignment
from ..distributed.teacher_placement import TeacherPlacement
from ..engines.module_capture_engine import ModuleCaptureEngine, _TruncatedForwardException
from ..engines.profiler import create_profiler
from ..training_arguments import TrainingArguments
from ..utils import _estimate_num_training_steps, freeze_parameters

# Optional dependency (see alignments/utils.py for the pattern).
if TYPE_CHECKING:
    import weightwatcher as ww

try:
    import weightwatcher as ww

    WEIGHTWATCHER_AVAILABLE = True
except ImportError:
    WEIGHTWATCHER_AVAILABLE = False

logger = logging.getLogger(__name__)


_NestedT = TypeVar("_NestedT")

# Mapping of distiller-specific TrainingArguments params to their defaults.
# Used by _warn_irrelevant_args() to detect non-default values that a
# particular distiller will silently ignore.
_DISTILLER_ARG_DEFAULTS: dict[str, object] = {
    "alpha": 0.0,
    "auto_device_match": False,
    "auto_dtype_match": False,
    "backward_per_block": False,
    "deepcopy_captured_args_and_kwargs": False,
    "magnitude_aware_weighting": False,
}


def _warn_irrelevant_args(
    args: Any,
    used_params: set[str],
    distiller_name: str,
) -> None:
    """Warn when distiller-specific args are set but irrelevant for this distiller.

    Args:
        args: The TrainingArguments instance.
        used_params: Set of param names this distiller actually reads.
        distiller_name: Human-readable name for the warning message.
    """
    irrelevant = []
    for name, default in _DISTILLER_ARG_DEFAULTS.items():
        if name not in used_params and getattr(args, name, default) != default:
            irrelevant.append(f"{name}={getattr(args, name)!r}")
    if irrelevant:
        logger.warning(
            "%s does not use %s — %s will be ignored.",
            distiller_name,
            ", ".join(irrelevant),
            "it" if len(irrelevant) == 1 else "they",
        )


def send_to_dtype(tensor_or_nested: _NestedT, dtype: torch.dtype) -> _NestedT:
    """Recursively cast all floating-point tensors in a nested structure to the specified dtype."""
    obj: Any = tensor_or_nested
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return cast(_NestedT, obj.to(dtype=dtype))
        return cast(_NestedT, obj)
    if isinstance(obj, (tuple, list)):
        return cast(_NestedT, type(obj)(send_to_dtype(item, dtype) for item in obj))
    if isinstance(obj, dict):
        return cast(
            _NestedT,
            {key: send_to_dtype(val, dtype) for key, val in obj.items()},
        )
    return cast(_NestedT, obj)


class SilentProgressCallback(ProgressCallback):
    """
    Progress callback that displays progress bars but doesn't print per-step metrics.

    This is useful for distillation where many per-module metrics would
    clutter the console output. Metrics are still logged to TensorBoard and other
    configured loggers.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """No-op override that suppresses per-step metric printing."""


class _AlignmentLossAccumulator:
    """Running total for alignment losses, with magnitude-aware support.

    Tracks the weighted sum (gradient path) and, when magnitude-aware
    weighting is enabled, also the raw weighted sum (logged value).
    ``finalize`` applies the straight-through trick so the returned
    tensor's *value* is the raw weighted sum (interpretable, matches
    ``magnitude_aware_weighting=False``) while its *gradient* is that
    of the normalized total.  See ``_combine_losses`` for the rationale.

    Used by both the post-hoc ``_compute_alignment_losses`` loop and the
    incremental ``HolisticDistiller._on_student_capture`` callback so the
    two paths produce bit-identical losses and gradients.
    """

    def __init__(self, magnitude_aware: bool):
        self.magnitude_aware = magnitude_aware
        self.total: torch.Tensor | None = None
        self.raw: torch.Tensor | None = None

    def add(
        self,
        alignment: "Alignment",
        raw_loss: torch.Tensor,
        weighted_loss: torch.Tensor,
    ) -> None:
        self.total = weighted_loss if self.total is None else self.total + weighted_loss
        if self.magnitude_aware:
            # ``.detach().clone()`` is required because PyTorch's fused loss
            # functions (e.g. F.mse_loss) return scalar tensors that view
            # into an input-sized scratch buffer; without cloning, the
            # running accumulator would keep that buffer alive across batches.
            raw_weighted = alignment.loss_weight * raw_loss.detach().clone()
            self.raw = raw_weighted if self.raw is None else self.raw + raw_weighted

    def finalize(self) -> torch.Tensor | None:
        if self.magnitude_aware and self.total is not None and self.raw is not None:
            return self.raw.detach() + (self.total - self.total.detach())
        return self.total


class BaseDistiller(Trainer):
    """
    Abstract base class for knowledge distillation trainers.

    This class extends HuggingFace's Trainer to provide common functionality
    for all distillation approaches:
    - Metrics tracking (including per-layer metrics)
    - FLOP counting
    - Capture engine lifecycle management
    - Teacher input preparation

    Subclasses must implement:
    - compute_distillation_loss(model, inputs, is_training): Define the
      distillation loss computation. Called by the template methods
      training_step() and compute_loss() provided by this base class.

    Subclasses may optionally override:
    - _backward_loss(loss): Custom backward pass (default: accelerator.backward)
    - _after_training_step(loss): Hook called after each training step
    - _save_distiller_state(output_dir) / _load_distiller_state(checkpoint_dir):
      Custom checkpoint save/load logic (call super() when overriding)
    - _get_e2e_student_models(): Return full student models for end-to-end eval
    - _compute_e2e_eval_loss(eval_dataset, metric_key_prefix): Custom e2e eval
    """

    args: TrainingArguments  # pyright: ignore[reportIncompatibleVariableOverride]
    teacher_model: PreTrainedModel | nn.Module
    _USE_COMPOSITE_OPTIMIZER: bool = False
    # True when prediction_step returns logits for compute_metrics (the
    # Trainer's main eval loop handles metric computation).  False when
    # prediction_step cannot provide logits — e.g. BlockwiseDistiller —
    # and the e2e eval pass computes metrics instead.
    _MAIN_LOOP_COMPUTES_METRICS: bool = False

    @property
    def student_model(self):
        """Alias for self.model — the student model being trained."""
        return self.model

    @student_model.setter
    def student_model(self, value):
        self.model = value

    def __init__(
        self,
        teacher_model: PreTrainedModel | nn.Module,
        alignments: list[Alignment],
        args: TrainingArguments | None = None,
        train_dataset=None,
        eval_dataset=None,
        prepare_teacher_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        prepare_student_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the BaseDistiller.

        Args:
            teacher_model: The teacher model to distill from
            alignments: List of Alignment instances (flat, one teacher-student pair each)
            args: TrainingArguments with distillation-specific parameters
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            prepare_teacher_inputs: Optional callable to transform inputs
                                   before passing to teacher. Takes inputs
                                   dict and returns modified dict. If None,
                                   passes inputs as-is.
            **kwargs: Additional keyword arguments passed to Trainer
        """
        # Store teacher model and alignments before calling super().__init__
        self.teacher_model = teacher_model
        self.alignments: list[Alignment] = alignments if alignments is not None else []

        # Freeze teacher: eval mode + disable gradients (once, not per step)
        self.teacher_model.eval()
        freeze_parameters(self.teacher_model, [".*"])
        self.prepare_teacher_inputs = prepare_teacher_inputs
        self._prepare_student_inputs_fn = prepare_student_inputs

        # Validate that alignments are provided for distillers that require them
        if not self.alignments and self._USE_COMPOSITE_OPTIMIZER:
            raise ValueError(
                "No alignments provided. This usually means the ``modules`` pattern "
                "given to create_alignments did not match any modules in the teacher "
                "model. Check that it is a valid regex with properly escaped backslashes."
            )

        # Create default TrainingArguments if none provided
        if args is None:
            args = TrainingArguments()

        # Save the user's original torch_compile setting. Subclasses like
        # BlockwiseDistiller may set args.torch_compile=False before calling
        # super().__init__ to prevent the Trainer from compiling the dummy model;
        # they pre-set self._torch_compile_teacher so we respect the original value.
        if not hasattr(self, "_torch_compile_teacher"):
            self._torch_compile_teacher = getattr(args, "torch_compile", False)

        # Call Trainer's __init__ - subclasses should set up model before calling super().__init__
        Trainer.__init__(
            self,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        # Compile the teacher model if torch_compile was originally enabled.
        # The Trainer only compiles self.model (the student); the teacher also
        # runs a full forward pass every step and benefits equally.
        if self._torch_compile_teacher:
            self.teacher_model = torch.compile(self.teacher_model)  # type: ignore[assignment]

        # Calculate num_training_steps for scheduler creation
        self.num_training_steps = (
            args.max_steps
            if args.max_steps > 0
            else (_estimate_num_training_steps(self.train_dataset, self.args))
        )

        # Prepare all alignments for training.
        # When FSDP is enabled and we use CompositeOptimizer, defer optimizer
        # creation until create_optimizer() — FSDP wrapping replaces model
        # parameters, making early param refs stale.
        defer_opt = self._USE_COMPOSITE_OPTIMIZER and bool(self.alignments) and self.is_fsdp_enabled
        for alignment_id, alignment in enumerate(self.alignments):
            alignment._prepare_for_training(
                alignment_id,
                self.args,
                self.num_training_steps,
                defer_optimizer=defer_opt,
            )

        # Propagate max_grad_norm to alignments that don't have explicit values.
        for alignment in self.alignments:
            if alignment.max_grad_norm is None:
                alignment.max_grad_norm = self.args.max_grad_norm

        # Build CompositeOptimizer/Scheduler from per-alignment optimizers
        if self._USE_COMPOSITE_OPTIMIZER and self.alignments:
            if not defer_opt:
                self._build_composite_optimizer()
            # Per-student clipping handled in training_step; disable Trainer's clipping
            self.args.max_grad_norm = 0

        # Student-only gradient clipping (exclude projector params).
        # When clip_projectors=False, disable the Trainer's built-in clipping
        # and clip manually in training_step using only model.parameters().
        self._student_only_clip_norm: float | None = None
        if not getattr(self.args, "clip_projectors", True) and self.args.max_grad_norm > 0:
            self._student_only_clip_norm = self.args.max_grad_norm
            self.args.max_grad_norm = 0  # disable Trainer's clipping

        # Metrics tracking
        self.current_step_metrics: dict[str, float | torch.Tensor] = {}
        self.step_losses: list[torch.Tensor] = []
        self.eval_losses: list[torch.Tensor] = []
        # Running-sum accumulators for training metrics — mirrors how the HF
        # Trainer tracks ``tr_loss`` as a device tensor to avoid per-step
        # ``.item()`` GPU→CPU syncs.  Scalar conversion deferred to log time.
        self._training_metric_accumulator: dict[str, Any] = {}
        self._is_accumulating: bool = False
        # Own step counter for averaging: the Trainer updates
        # ``_globalstep_last_logged`` *before* calling ``log()``, so by the
        # time our override runs the delta is always 0.  We track separately.
        self._metric_accumulator_steps: int = 0

        # FLOP counting
        self.flop_counter: int = 0
        self.flops_per_step: int = 0

        # Replace noisy callbacks with silent versions
        self.remove_callback(PrinterCallback)
        self.remove_callback(ProgressCallback)
        self.add_callback(SilentProgressCallback)

        # Capture engines (subclasses append their engines in __init__)
        self._capture_engines: list = []

        # Forward overlap (set in _setup_forward_overlap)
        self._overlap_teacher_forward = False
        self._teacher_stream = None
        self._teacher_placed = False
        # Projectors already broadcast from rank 0 (see _sync_projectors_from_rank0).
        self._synced_projector_ids: set[int] = set()

        # Initialize profiler if enabled.
        # ``_profiler_stack`` owns the profiler's context-manager lifetime
        # across the split ``_start_profiler`` / ``_stop_profiler`` calls.
        self.profiler = None
        self._profiler_stack: contextlib.ExitStack | None = None
        self._init_profiler()

    # -------------------------------------------------------------------------
    # Composite optimizer / scheduler
    # -------------------------------------------------------------------------

    def _build_composite_optimizer(self) -> None:
        """Build CompositeOptimizer/Scheduler from per-alignment optimizers."""
        from ..optim import CompositeOptimizer, CompositeScheduler

        child_optimizers = {}
        child_schedulers = {}
        for alignment in self.alignments:
            name = alignment.get_name()
            child_optimizers[name] = alignment.optimizer
            child_schedulers[name] = alignment.scheduler

        self.optimizer = CompositeOptimizer(child_optimizers)
        self.lr_scheduler = CompositeScheduler(child_schedulers)

    def create_optimizer(self, model: torch.nn.Module | None = None):
        """Override Trainer's create_optimizer to use CompositeOptimizer.

        Called by the Trainer after FSDP/DeepSpeed wrapping (if any), so
        parameter references are up-to-date at this point.

        Under DeepSpeed, CompositeOptimizer is incompatible (DeepSpeed needs
        a single flat optimizer for ZeRO partitioning). Instead, we build a
        single standard optimizer with per-alignment param_groups tagged by
        ``_alignment_name``, preserving per-alignment LR tracking and grad
        clipping.

        Args:
            model: The model to create the optimizer for. Passed by newer
                versions of the HF Trainer but unused here (we build
                optimizers from alignment parameters directly).
        """
        if self._USE_COMPOSITE_OPTIMIZER:
            if self.optimizer is None:
                if self.is_deepspeed_enabled:
                    self._build_deepspeed_compatible_optimizer()
                else:
                    # Create per-alignment optimizers/schedulers first
                    for alignment in self.alignments:
                        alignment._ensure_optimizer_and_scheduler(
                            self.args, self.num_training_steps
                        )
                    self._build_composite_optimizer()
            return self.optimizer
        return super().create_optimizer(model)

    def create_scheduler(self, num_training_steps, optimizer=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Override Trainer's create_scheduler to use CompositeScheduler."""
        if self._USE_COMPOSITE_OPTIMIZER:
            if self.lr_scheduler is None:
                if self.is_deepspeed_enabled:
                    # Under DeepSpeed, let Trainer/DS build the scheduler
                    return super().create_scheduler(num_training_steps, optimizer or self.optimizer)
                self._build_composite_optimizer()
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def _build_deepspeed_compatible_optimizer(self) -> None:
        """Build a single flat optimizer with per-alignment param_groups.

        DeepSpeed requires a single ``torch.optim.Optimizer`` so it can
        flatten and partition params for ZeRO.  We collect param_groups from
        every alignment and tag each group with ``_alignment_name`` so that
        ``_clip_student_gradients`` and ``_track_student_learning_rates``
        can still operate per-alignment.
        """
        all_groups = []
        for alignment in self.alignments:
            groups = alignment._get_param_groups(self.args)
            for g in groups:
                g["_alignment_name"] = alignment.get_name()
            all_groups.extend(groups)

        optimizer_cls, optimizer_kwargs = self.alignments[0]._get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(all_groups, **optimizer_kwargs)

    def _get_grad_norm(self, model=None, grad_norm=None):
        """Compute gradient norm, avoiding FSDP.clip_grad_norm_ for per-block wrapping.

        The HF Trainer (v5.3+) calls ``_get_grad_norm`` after each training
        step via ``accelerator.clip_grad_norm_(model.parameters(), inf)``.
        This triggers ``FSDP._lazy_init`` which fails when student blocks are
        individually FSDP-wrapped (BKD), because the root/non-root hierarchy
        doesn't match what FSDP expects.

        When using a composite optimizer (BKD/HKD), we compute the norm
        directly from the per-alignment parameters instead.
        """
        if grad_norm is not None:
            return grad_norm
        if self._USE_COMPOSITE_OPTIMIZER:
            # Collect all trainable parameters from alignment optimizers
            all_params = []
            for alignment in self.alignments:
                if alignment.optimizer is not None:
                    for g in alignment.optimizer.param_groups:
                        all_params.extend(p for p in g["params"] if p.grad is not None)
            if all_params:
                return torch.nn.utils.clip_grad_norm_(all_params, float("inf"))
            return torch.tensor(0.0)
        return super()._get_grad_norm(model, grad_norm)

    def _excluded_from_clipping(self) -> set[int]:
        """IDs of parameters left out of gradient clipping.

        With ``clip_projectors=False`` the projector parameters are neither
        counted in an alignment's gradient norm nor scaled by the clip.
        """
        if getattr(self.args, "clip_projectors", True):
            return set()
        excluded: set[int] = set()
        for alignment in self.alignments:
            for proj in (alignment.input_projector, alignment.output_projector):
                if proj is not None:
                    excluded.update(id(p) for p in proj.parameters())
        return excluded

    def _clip_student_gradients(self) -> None:
        """Clip gradients per-alignment and log norms."""
        if self._USE_COMPOSITE_OPTIMIZER and self.is_deepspeed_enabled:
            # Under DeepSpeed, param_groups are in self.optimizer tagged
            # with _alignment_name — clip per-alignment from there.
            self._clip_deepspeed_gradients()
            return
        excluded = self._excluded_from_clipping()
        for alignment in self.alignments:
            if alignment.max_grad_norm is not None and alignment.optimizer is not None:
                params = [
                    p
                    for g in alignment.optimizer.param_groups
                    for p in g["params"]
                    if id(p) not in excluded
                ]
                if not params:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(params, alignment.max_grad_norm)
                self._store_grad_norm_metric(alignment.get_name(), norm)

    def _clip_deepspeed_gradients(self) -> None:
        """Clip gradients using tagged param_groups under DeepSpeed."""
        from collections import defaultdict

        excluded = self._excluded_from_clipping()
        # Collect params per alignment name
        alignment_params = defaultdict(list)
        for g in self.optimizer.param_groups:
            name = g.get("_alignment_name")
            if name is not None:
                alignment_params[name].extend(p for p in g["params"] if id(p) not in excluded)

        # Look up max_grad_norm from alignments
        alignment_norms = {a.get_name(): a.max_grad_norm for a in self.alignments}
        for name, params in alignment_params.items():
            max_norm = alignment_norms.get(name)
            if max_norm is not None and params:
                norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
                self._store_grad_norm_metric(name, norm)

    def _track_student_learning_rates(self) -> None:
        """Log per-alignment learning rates."""
        if self._USE_COMPOSITE_OPTIMIZER and self.is_deepspeed_enabled:
            # Under DeepSpeed, read LR from tagged param_groups
            seen = set()
            for g in self.optimizer.param_groups:
                name = g.get("_alignment_name")
                if name is not None and name not in seen:
                    seen.add(name)
                    self._store_lr_metric(name, g["lr"])
            return
        for alignment in self.alignments:
            if alignment.optimizer is None:
                continue
            lr = alignment.optimizer.param_groups[0]["lr"]
            self._store_lr_metric(alignment.get_name(), lr)

    # -------------------------------------------------------------------------
    # Profiler methods
    # -------------------------------------------------------------------------

    def _init_profiler(self) -> None:
        """Initialize the torch.profiler if enabled in training arguments."""
        if not getattr(self.args, "enable_profiling", False):
            return

        assert self.args.output_dir is not None
        output_dir: str = getattr(self.args, "profiling_output_dir", None) or str(
            Path(self.args.output_dir) / "profiling"
        )

        self.profiler = create_profiler(
            output_dir,
            wait=getattr(self.args, "profiling_wait", 20),
            warmup=getattr(self.args, "profiling_warmup", 3),
            active=getattr(self.args, "profiling_active", 3),
            repeat=getattr(self.args, "profiling_repeat", 1),
            with_stack=getattr(self.args, "profiling_with_stack", False),
        )
        logger.info("Profiler initialized. Traces will be saved to: %s", output_dir)

    def _start_profiler(self) -> None:
        """Start the torch.profiler context.

        The profiler is entered via an :class:`contextlib.ExitStack`
        owned on ``self`` rather than by calling ``__enter__`` directly,
        so that ``_stop_profiler`` can close it with normal
        context-manager teardown (exception-safe, matches the
        ``with`` statement's semantics).
        """
        if self.profiler is None:
            return
        self._profiler_stack = contextlib.ExitStack()
        self._profiler_stack.enter_context(self.profiler)

    def _stop_profiler(self) -> None:
        """Stop the torch.profiler context."""
        if self._profiler_stack is not None:
            self._profiler_stack.close()
            self._profiler_stack = None

    def _profiler_step(self) -> None:
        """Signal to the profiler that a training step has completed."""
        if self.profiler is not None:
            self.profiler.step()

    # -------------------------------------------------------------------------
    # Metrics tracking methods
    # -------------------------------------------------------------------------

    def _reset_step_metrics(self, is_training: bool = True) -> None:
        """
        Reset metrics collectors for a new step.

        Args:
            is_training: If True, reset step_losses; if False, reset eval_losses
        """
        self.current_step_metrics = {}
        self._is_accumulating = is_training
        if is_training:
            self.step_losses = []
            self._metric_accumulator_steps += 1
        else:
            self.eval_losses = []

    def _track_metric(self, name: str, value) -> None:
        """
        Store a metric for the current step and accumulate it for averaging.

        Per-step storage (current_step_metrics) is used by eval and by subclass
        code that needs single-batch values.  The accumulator is only populated
        during training steps so that eval-path calls don't pollute it.

        Accepts both plain floats and GPU tensors.  When a tensor is passed it
        is detached AND cloned but ``.item()`` is **not** called — this avoids
        a costly GPU→CPU synchronisation every training step.  Instead, the
        accumulator maintains a running sum per metric (mirroring how the HF
        Trainer tracks ``tr_loss`` as a device tensor) and only calls
        ``.item()`` at ``logging_steps`` cadence in ``_get_averaged_metrics()``.

        Why ``.detach().clone()`` and not just ``.detach()``: PyTorch's fused
        loss functions (``F.mse_loss``, ``F.smooth_l1_loss``, etc.) return
        scalar tensors that are 0-dim VIEWS into a scratch buffer the size of
        the input. Calling ``.detach()`` shares storage with this scratch
        buffer, so storing detached scalar losses persistently would keep
        the (potentially many-MB) scratch buffer alive for as long as the
        metric is referenced. ``.clone()`` allocates a fresh tiny storage
        for the scalar, freeing the scratch buffer after the next collection.

        Args:
            name: Metric key (e.g. "loss/student_0")
            value: Scalar metric value (float or 0-dim tensor)
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().clone()
        self.current_step_metrics[name] = value
        if self._is_accumulating:
            prev = self._training_metric_accumulator.get(name)
            if prev is None:
                self._training_metric_accumulator[name] = value
            else:
                self._training_metric_accumulator[name] = prev + value

    def _store_loss_metric(self, student_name: str, loss: torch.Tensor) -> None:
        """
        Store a loss metric for a student.

        Args:
            student_name: Name of the student (from student.get_name())
            loss: The loss tensor
        """
        self._track_metric(f"loss/{student_name}", loss)

    def _store_lr_metric(self, student_name: str, lr: float) -> None:
        """
        Store a learning rate metric for a student.

        Args:
            student_name: Name of the student
            lr: Current learning rate
        """
        self._track_metric(f"learning_rate/{student_name}", lr)

    def _store_grad_norm_metric(self, student_name: str, norm: float | torch.Tensor) -> None:
        """
        Store a gradient norm metric for a student.

        Args:
            student_name: Name of the student
            norm: Gradient norm value
        """
        self._track_metric(f"grad_norm/{student_name}", norm)

    @staticmethod
    def _is_model_fsdp_wrapped(model: "nn.Module") -> bool:
        """Check if a model is FSDP-wrapped."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            return isinstance(model, FSDP) or any(isinstance(m, FSDP) for m in model.modules())
        except ImportError:
            return False

    def _resolve_auto_truncate(
        self,
        auto_truncate: bool,
        is_fsdp: bool,
        model_name: str,
        needs_backward: bool = True,
    ) -> bool:
        """Check FSDP compatibility and return effective auto_truncate value.

        FSDP's backward hooks expect every module that participated in
        forward to also participate in backward (``FORWARD`` →
        ``BACKWARD_PRE`` → ``BACKWARD_POST`` state transitions). When a
        model only runs forward (e.g. the teacher under ``torch.no_grad``),
        there is no backward pass and truncation is safe even with FSDP.

        Args:
            auto_truncate: Requested auto_truncate value.
            is_fsdp: Whether the model is (or will be) FSDP-wrapped.
            model_name: Human-readable name for the warning message
                (e.g. "teacher", "student").
            needs_backward: Whether backward will run through this model.
                False for teacher models (forward-only under no_grad).

        Returns:
            Effective auto_truncate value (False if incompatible).
        """
        if not auto_truncate:
            return False
        if is_fsdp and needs_backward:
            logger.warning(
                "auto_truncate=True was requested but is being disabled for "
                "the FSDP-wrapped %s. FSDP's backward hooks require all "
                "forward-participating modules to complete the backward state "
                "transition. The full forward will run instead (no early exit).",
                model_name,
            )
            return False
        return True

    @staticmethod
    def _reset_fsdp_states(model: "nn.Module") -> None:
        """Reset FSDP handle/module training states to IDLE on a model.

        After auto-truncation raises ``_TruncatedForwardException``, outer
        FSDP wrappers may be stuck in ``FORWARD`` state (they unsharped for
        their forward but the exception prevented resharding). This resets
        all FSDP modules to ``IDLE`` so subsequent operations work correctly.
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp._common_utils import HandleTrainingState, TrainingState
        except ImportError:
            return

        for module in model.modules():
            if isinstance(module, FSDP):
                if hasattr(module, "training_state"):
                    module.training_state = TrainingState.IDLE
                handle = getattr(module, "_handle", None)
                if handle is not None:
                    handle._training_state = HandleTrainingState.IDLE

    @staticmethod
    def _extract_loss(outputs) -> "torch.Tensor | None":
        """Extract loss from model outputs, if present.

        Handles HuggingFace-style outputs (object with ``.loss``),
        dict outputs, and returns ``None`` if no loss is found.
        """
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        return None

    def _warn_once(self, key: str, message: str, *args) -> None:
        """Log a warning at most once per key per distiller instance."""
        if not hasattr(self, "_warned_keys"):
            self._warned_keys = set()
        if key not in self._warned_keys:
            logger.warning(message, *args)
            self._warned_keys.add(key)

    def _combine_losses(
        self,
        losses: dict[str, torch.Tensor],
        weights: dict[str, float],
        track_metrics: bool = True,
    ) -> torch.Tensor | None:
        """Combine named losses with weights and optional magnitude normalization.

        When ``magnitude_aware_weighting`` is enabled, each loss is divided
        by its detached magnitude before weighting, ensuring gradient
        contributions are proportional to the specified weights.

        **Returned value vs. backprop gradient.** With magnitude-aware on,
        the naive normalized total ``sum(w_i * L_i / |L_i|)`` is ~constant
        (≈``sum(w_i)``), which is useless for monitoring and breaks early
        stopping.  We use the standard "straight-through" trick to decouple
        the two concerns:

            L_display = L_raw.detach() + (L_norm - L_norm.detach())

        - Forward value is ``L_raw = sum(w_i * L_i)`` — the quantity a user
          would expect to see in logs, and the same number you'd get with
          ``magnitude_aware_weighting=False``.
        - Backward gradient is ``∇_θ L_norm`` — each component pre-divided
          by its own magnitude so contributions balance as intended.

        This lets the tracked ``loss``, wandb ``loss`` curve, eval loss,
        and EarlyStoppingCallback all see the same interpretable raw loss
        while the optimizer still takes magnitude-balanced gradient steps.

        Args:
            losses: Map of component name to scalar loss tensor
                (e.g. ``{"soft": soft_loss, "hard": hard_loss}``).
            weights: Map of component name to weight. Only the ratios
                matter, not absolute values.
            track_metrics: Whether to log per-component and total metrics.

        Returns:
            Combined loss scalar whose value is ``L_raw`` and whose
            gradient is ``∇_θ L_norm`` when magnitude-aware is on, or just
            ``L_raw`` otherwise.
        """
        magnitude_aware = getattr(self.args, "magnitude_aware_weighting", False)
        total = None
        total_raw = None

        for name, loss in losses.items():
            w = weights[name]
            if track_metrics:
                self._track_metric(f"loss/{name}", loss)

            # Guard against non-finite per-component losses.  If the student
            # forward produces a NaN or Inf (bf16 overflow, gradient blowup
            # from a prior step, etc.), propagating it through the
            # magnitude-aware divisor poisons the entire combined total and
            # silently zeros every subsequent backward.  Treat the bad
            # component as contributing 0 to this step so the rest of the
            # alignments can still update — one bad batch does not kill the
            # whole run.  The warning fires at most once per component name.
            if not torch.isfinite(loss.detach()).all():
                self._warn_once(
                    f"nonfinite_component_{name}",
                    "Non-finite loss component %r (value=%s); zeroing its "
                    "contribution for this step.  Recurring non-finite "
                    "losses usually mean the student has diverged — check "
                    "learning rate / gradient clipping / dtype.",
                    name,
                    loss.detach().item(),
                )
                continue

            if magnitude_aware:
                # Divide by |loss| (not loss) so the normalized contribution
                # is always +w regardless of the raw sign. Using plain
                # ``clamp(min=1e-8)`` lets negative values pass through
                # unchanged, so a modest -1e3 would get divided by the
                # pinned 1e-8 denominator and amplify to -1e11 — turning a
                # transient numerical blip into a training-destroying loss.
                weighted = w * loss / loss.detach().abs().clamp(min=1e-8)
            else:
                weighted = w * loss

            total = weighted if total is None else total + weighted

            if magnitude_aware:
                raw = w * loss.detach()
                total_raw = raw if total_raw is None else total_raw + raw

        # Straight-through: value comes from total_raw (interpretable),
        # gradient comes from the normalized ``total`` (magnitude-balanced).
        # No separate ``loss/total`` metric is tracked — the returned
        # tensor's value IS the raw weighted sum, and HF Trainer will log
        # it as the top-level ``loss`` / ``eval_loss`` field automatically.
        if magnitude_aware and total is not None and total_raw is not None:
            return total_raw.detach() + (total - total.detach())
        return total

    def _apply_loss_weighting(self, loss: torch.Tensor, alignment) -> torch.Tensor:
        """Apply per-alignment weight and optional magnitude normalization.

        When ``magnitude_aware_weighting`` is enabled in the training args,
        each loss is divided by its detached magnitude before weighting.
        This ensures gradient contributions are proportional to weights
        regardless of absolute loss magnitudes.

        The denominator is ``|loss|`` (not ``loss``): if we just used
        ``clamp(min=1e-8)`` on a signed value, a negative loss (from
        weight corruption, bf16 rounding, exploding gradients, etc.)
        would pass through with its sign intact, be divided by the
        pinned-positive 1e-8, and amplify by ~1e8 — destroying training.

        A non-finite (NaN / Inf) loss from the student forward would
        poison the accumulated total in ``_compute_alignment_losses``
        — its contribution is replaced with a detached zero so one
        bad batch cannot kill the whole run.  See ``_combine_losses``
        for the corresponding guard on the ResKD path.

        Args:
            loss: Raw scalar loss for one alignment.
            alignment: The Alignment instance (carries ``loss_weight``).

        Returns:
            Weighted (and optionally magnitude-normalized) loss, or a
            detached zero if the input was non-finite.
        """
        if not torch.isfinite(loss.detach()).all():
            self._warn_once(
                f"nonfinite_alignment_{alignment.get_name()}",
                "Non-finite loss on alignment %r (value=%s); zeroing its "
                "contribution for this step.  Recurring non-finite losses "
                "usually mean the student has diverged — check learning "
                "rate / gradient clipping / dtype.",
                alignment.get_name(),
                loss.detach().item(),
            )
            # Detached zero — zero gradient, zero value, safe to sum.
            return torch.zeros((), device=loss.device, dtype=loss.dtype)
        weight = alignment.loss_weight
        if getattr(self.args, "magnitude_aware_weighting", False):
            return weight * loss / loss.detach().abs().clamp(min=1e-8)
        return weight * loss if weight != 1.0 else loss

    @staticmethod
    def _match_device_dtype(
        source: torch.Tensor,
        target: torch.Tensor,
        auto_device: bool = False,
        auto_dtype: bool = False,
    ) -> torch.Tensor:
        """Match source tensor's device and dtype to a target tensor.

        Args:
            source: Tensor to cast (typically teacher output).
            target: Reference tensor to match against (typically student output).
            auto_device: If True, move source to target's device.
            auto_dtype: If True, cast source to target's dtype.

        Returns:
            Source tensor, potentially moved/cast.
        """
        if auto_device:
            source = source.to(device=target.device)
        if auto_dtype:
            source = source.to(dtype=target.dtype)
        return source

    def _compute_alignment_loss(
        self,
        alignment: "Alignment",
        student_output,
        teacher_output,
        is_training: bool,
    ) -> torch.Tensor:
        """Apply output projection, compute loss, and record metrics for one alignment.

        Shared pipeline used by both holistic (via ``_compute_alignment_losses``)
        and blockwise distillation. Expects ``student_output`` to have already
        been through ``alignment.student_output_selector``.

        Steps:
        1. Apply ``output_projector`` if present
        2. Compute loss via ``alignment.loss_function``
        3. Record loss metric and append to step/eval loss list

        Args:
            alignment: The Alignment instance.
            student_output: Student output (after output selection).
            teacher_output: Teacher output (after selection and device/dtype matching).
            is_training: Whether this is a training step.

        Returns:
            Scalar loss tensor (unweighted).
        """
        if alignment.output_projector is not None:
            student_output = alignment.output_projector(student_output)
        loss = alignment.loss_function(student_output, teacher_output)
        self._store_loss_metric(alignment.get_name(), loss)
        # ``.detach().clone()`` (not just ``.detach()``) is required because
        # PyTorch's fused loss kernels (``F.mse_loss``, ``F.smooth_l1_loss``,
        # etc.) return scalar tensors that view into an input-sized scratch
        # buffer.  See ``_track_metric`` docstring for details.
        (self.step_losses if is_training else self.eval_losses).append(loss.detach().clone())
        return loss

    def _make_loss_accumulator(self) -> "_AlignmentLossAccumulator":
        """Build a fresh accumulator wired to this distiller's args."""
        return _AlignmentLossAccumulator(
            magnitude_aware=getattr(self.args, "magnitude_aware_weighting", False),
        )

    def _setup_capture_events(self, engine: "ModuleCaptureEngine") -> dict[int, Any] | None:
        """Install per-alignment CUDA events on a capture engine for stream
        overlap.  Returns the dict (callers may keep it for ``wait_event``)
        or None when overlap is disabled.  Caller is responsible for
        clearing ``engine.capture_events`` after the forward.
        """
        if not getattr(self, "_overlap_teacher_forward", False):
            return None
        events = {i: torch.cuda.Event() for i in range(len(self.alignments))}
        engine.capture_events = events
        return events

    def _process_one_alignment(
        self,
        alignment: "Alignment",
        teacher_output,
        student_output,
        is_training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the shared per-alignment pipeline.

        Selectors → device/dtype match → lazy auto-projector init →
        loss → weight.  Used by both the post-hoc holistic path
        (``_compute_alignment_losses``) and the incremental hook
        callback (``HolisticDistiller._on_student_capture``).  Keeping
        a single implementation guarantees the two paths produce
        bit-identical gradients.

        Returns:
            ``(raw_loss, weighted_loss)``.  Caller decides how to
            accumulate (plain sum vs magnitude-aware straight-through).
        """
        teacher_output = alignment.teacher_output_selector(teacher_output)
        student_output = alignment.student_output_selector(student_output)

        if alignment.auto_device_match:
            teacher_output = send_to_device(teacher_output, student_output.device)
        if alignment.auto_dtype_match:
            teacher_output = send_to_dtype(teacher_output, student_output.dtype)

        if alignment.auto_projector and not alignment._auto_projector_initialized:
            alignment._try_init_output_projector(student_output, teacher_output)
            self._sync_projectors_from_rank0(alignment)

        raw_loss = self._compute_alignment_loss(
            alignment, student_output, teacher_output, is_training
        )
        weighted_loss = self._apply_loss_weighting(raw_loss, alignment)
        return raw_loss, weighted_loss

    def _compute_alignment_losses(
        self,
        teacher_capture,
        student_capture,
        alignment_id_to_module_id: dict[int, int],
        is_training: bool = True,
    ) -> torch.Tensor | None:
        """Compute alignment losses between all teacher-student pairs from captured data.

        Args:
            teacher_capture: ModuleCaptureEngine for teacher.
            student_capture: ModuleCaptureEngine for student.
            alignment_id_to_module_id: Map from alignment index to capture module index.
            is_training: Whether in training mode (affects which loss list to use).

        Returns:
            Total weighted loss across all alignments.
        """
        accumulator = self._make_loss_accumulator()

        for alignment_id, alignment in enumerate(self.alignments):
            module_id = alignment_id_to_module_id[alignment_id]

            # Pop captured outputs — frees memory as soon as each loss is computed
            teacher_output = teacher_capture.captured_outputs.pop(module_id)
            student_output = student_capture.captured_outputs.pop(module_id)

            raw_loss, weighted_loss = self._process_one_alignment(
                alignment, teacher_output, student_output, is_training
            )
            accumulator.add(alignment, raw_loss, weighted_loss)

        return accumulator.finalize()

    # -------------------------------------------------------------------------
    # FLOP counting methods
    # -------------------------------------------------------------------------

    def _get_flop_context(self, count_now: bool) -> contextlib.nullcontext | FlopCounterMode:
        """
        Get the appropriate context manager for FLOP counting.

        Args:
            count_now: Whether to actually count FLOPs in this step

        Returns:
            FlopCounterMode if counting, nullcontext otherwise
        """
        if count_now:
            return FlopCounterMode(display=False)
        return contextlib.nullcontext()

    def _record_flops(self, flop_context: contextlib.nullcontext | FlopCounterMode) -> None:
        """
        Record FLOPs from a FlopCounterMode context.

        Args:
            flop_context: The context manager used during forward pass
        """
        if isinstance(flop_context, FlopCounterMode):
            counter: FlopCounterMode = flop_context
            self.flops_per_step = counter.get_total_flops()

    def _should_count_flops(self) -> bool:
        """
        Determine if FLOPs should be counted on this step.

        Returns:
            True if this is the first step and count_flops is enabled
        """
        return self.state.global_step == 0 and getattr(self.args, "count_flops", False)

    def _update_flop_counter(self) -> None:
        """Update the cumulative FLOP counter if counting is enabled."""
        if getattr(self.args, "count_flops", False):
            self.flop_counter += self.flops_per_step

    # -------------------------------------------------------------------------
    # Teacher input preparation
    # -------------------------------------------------------------------------

    # Keys to always strip from teacher inputs — the teacher's loss is never
    # used by any distiller, and passing labels to a pipeline-parallel teacher
    # can cause cross-device errors when the model computes loss internally.
    _TEACHER_STRIP_KEYS = frozenset({"labels"})

    def _prepare_teacher_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare inputs for the teacher model.

        If a prepare_teacher_inputs callable was provided, use it to transform
        the inputs. Otherwise, automatically filters inputs to only include keys
        that the teacher model's forward() method accepts (excluding ``labels``,
        which are never needed by the teacher), and sets use_cache=False if
        supported.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Transformed inputs dictionary for the teacher model
        """
        if self.prepare_teacher_inputs is not None:
            return self.prepare_teacher_inputs(inputs)

        # Auto-filter based on teacher's forward signature (cached)
        accepted_params = self._get_teacher_accepted_params()
        if accepted_params is None:
            # Teacher accepts **kwargs — return as-is if no keys to strip,
            # otherwise create a filtered copy
            if self._TEACHER_STRIP_KEYS.isdisjoint(inputs):
                return inputs
            filtered = {k: v for k, v in inputs.items() if k not in self._TEACHER_STRIP_KEYS}
        else:
            filtered = {
                k: v
                for k, v in inputs.items()
                if k in accepted_params and k not in self._TEACHER_STRIP_KEYS
            }

        # Disable KV caching during distillation if teacher supports it
        if accepted_params is not None and "use_cache" in accepted_params:
            filtered["use_cache"] = False

        return filtered

    def _prepare_student_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Prepare inputs for the student model.

        If a prepare_student_inputs callable was provided, use it to transform
        the inputs. Otherwise, return inputs unchanged.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Transformed inputs dictionary for the student model
        """
        if self._prepare_student_inputs_fn is not None:
            return self._prepare_student_inputs_fn(inputs)
        return inputs

    def _get_teacher_accepted_params(self) -> set | None:
        """
        Get the set of parameter names accepted by the teacher model's forward().

        Returns None if the teacher accepts **kwargs (meaning any key is valid).
        Result is cached after first call.
        """
        if not hasattr(self, "_cached_teacher_params"):
            fwd = getattr(self.teacher_model, "forward", self.teacher_model)
            sig = inspect.signature(fwd)
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if has_var_keyword:
                self._cached_teacher_params = None
            else:
                self._cached_teacher_params = set(sig.parameters.keys())
        return self._cached_teacher_params

    # -------------------------------------------------------------------------
    # Student training state persistence
    # -------------------------------------------------------------------------

    _STUDENT_TRAINING_STATE_FILE = "student_training_state.pt"
    _PROJECTOR_STATE_FILE = "projector_state.pt"

    def _save_student_training_state(self, output_dir: str) -> None:
        """
        Save per-alignment optimizer and scheduler states to a checkpoint file.

        Args:
            output_dir: Directory to save the training state file
        """
        output_path = Path(output_dir)
        optimizer_states = {}
        scheduler_states = {}

        for alignment in self.alignments:
            name = alignment.get_name()
            if alignment.optimizer is not None:
                optimizer_states[name] = alignment.optimizer.state_dict()
            if alignment.scheduler is not None:
                scheduler_states[name] = alignment.scheduler.state_dict()

        torch.save(
            {
                "optimizers": optimizer_states,
                "schedulers": scheduler_states,
            },
            output_path / self._STUDENT_TRAINING_STATE_FILE,
        )

    def _load_student_training_state(self, checkpoint_dir: str) -> None:
        """
        Load per-alignment optimizer and scheduler states from a checkpoint.

        Args:
            checkpoint_dir: Directory containing the training state file
        """
        training_state_path = Path(checkpoint_dir) / self._STUDENT_TRAINING_STATE_FILE
        if not training_state_path.exists():
            return

        training_state = torch.load(training_state_path)

        optimizer_states = training_state.get("optimizers", {})
        scheduler_states = training_state.get("schedulers", {})

        for alignment in self.alignments:
            name = alignment.get_name()

            if name in optimizer_states and alignment.optimizer is not None:
                try:
                    alignment.optimizer.load_state_dict(optimizer_states[name])
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        "Could not restore optimizer state for %s: %s. "
                        "Continuing with fresh optimizer.",
                        name,
                        e,
                    )

            if name in scheduler_states and alignment.scheduler is not None:
                try:
                    alignment.scheduler.load_state_dict(scheduler_states[name])
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        "Could not restore scheduler state for %s: %s. "
                        "Continuing with fresh scheduler.",
                        name,
                        e,
                    )

    # -------------------------------------------------------------------------
    # Checkpoint save/load
    # -------------------------------------------------------------------------

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint including per-alignment training state."""
        super()._save(output_dir, state_dict)
        resolved_dir = output_dir or self.args.output_dir
        assert resolved_dir is not None
        if self._USE_COMPOSITE_OPTIMIZER and self.alignments:
            self._save_student_training_state(resolved_dir)
        self._save_distiller_state(resolved_dir)

    def _projector_state_keys(self) -> list[str]:
        """Return the ``projector_state.pt`` key of every alignment, in order.

        The key is the alignment name.  Alignments built without names all
        share the same name, so a repeated name gets the alignment's index
        appended (``name#idx``) to keep the entries apart.
        """
        names = [alignment.get_name() for alignment in self.alignments]
        counts = Counter(names)
        return [f"{name}#{idx}" if counts[name] > 1 else name for idx, name in enumerate(names)]

    def _get_projectors(self) -> dict[str, dict[str, nn.Module]]:
        """Return all projectors, keyed by alignment and then by role.

        Keys come from :meth:`_projector_state_keys`; roles are
        ``"input_projector"`` and ``"output_projector"``.  Projectors are
        always saved/loaded via ``projector_state.pt``, separate from the
        model checkpoint, in this same nested layout.
        """
        projectors: dict[str, dict[str, nn.Module]] = {}
        for key, alignment in zip(self._projector_state_keys(), self.alignments, strict=True):
            entry = {
                role: proj
                for role, proj in (
                    ("input_projector", alignment.input_projector),
                    ("output_projector", alignment.output_projector),
                )
                if proj is not None
            }
            if entry:
                projectors[key] = entry
        return projectors

    def _save_distiller_state(self, output_dir: str) -> None:
        """Save projector weights to projector_state.pt.

        Subclasses should call super()._save_distiller_state() when overriding.
        """
        projectors = self._get_projectors()
        if projectors:
            torch.save(
                {
                    name: {role: proj.state_dict() for role, proj in entry.items()}
                    for name, entry in projectors.items()
                },
                Path(output_dir) / self._PROJECTOR_STATE_FILE,
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load model checkpoint, then restore projector weights."""
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self._load_projector_state(resume_from_checkpoint)

    def _load_projector_state(self, checkpoint_dir: str) -> None:
        """Load projector weights from ``projector_state.pt``.

        For auto-projectors that were saved in the checkpoint but have not
        yet lazy-initialised on the current (resuming) distiller, this
        method materialises them from the saved state: a
        ``GenericLinearProjector`` of matching shape is attached to the
        corresponding alignment and its weights are loaded.  Without this
        step, HKD's single optimiser (created before the first forward
        pass) would not include the saved projector params, and a
        subsequent ``optimizer.load_state_dict`` would fail with a
        ``ValueError: loaded state dict contains a parameter group that
        doesn't match the size of optimizer's group``.
        """
        projector_path = Path(checkpoint_dir) / self._PROJECTOR_STATE_FILE
        if not projector_path.exists():
            return
        saved = torch.load(projector_path, weights_only=True, map_location="cpu")

        for key, alignment in zip(self._projector_state_keys(), self.alignments, strict=True):
            entry = saved.get(key)
            if not entry:
                continue
            for role, state_dict in entry.items():
                proj = getattr(alignment, role, None)
                if proj is not None:
                    proj.load_state_dict(state_dict)
                else:
                    self._materialise_saved_projector(alignment, role, state_dict)

    def _materialise_saved_projector(
        self, alignment: Alignment, role: str, state_dict: dict[str, torch.Tensor]
    ) -> None:
        """Create projector ``role`` on ``alignment`` from a saved state dict.

        Infers (in_features, out_features, bias) from the saved weight shape
        and attaches a fresh ``GenericLinearProjector`` to the alignment.  If
        the alignment already has an optimiser (BKD case), the projector's
        params are also added to it; for HKD the optimiser is created later
        in ``HolisticDistiller.create_optimizer``, which picks up the
        now-materialised projector via
        ``Alignment._add_projector_params_to_optimizer``.
        """
        from ..alignments.projectors import GenericLinearProjector

        if role not in ("input_projector", "output_projector"):
            return
        weight = state_dict.get("weight")
        if weight is None or weight.dim() != 2:
            return
        out_features, in_features = weight.shape
        has_bias = "bias" in state_dict
        mode = "input" if role == "input_projector" else "output"
        # Infer device/dtype from the student block if it has parameters,
        # otherwise fall back to the student model (handles parameter-free
        # blocks like pooling layers used in RelKD alignments).
        block_params = list(alignment.student_block.parameters())
        if block_params:
            device = block_params[0].device
            dtype = block_params[0].dtype
        else:
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
        proj = GenericLinearProjector(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            mode=mode,
        ).to(device=device, dtype=dtype)
        proj.load_state_dict(state_dict)
        setattr(alignment, role, proj)
        # Mark auto-projector as fully initialised so the lazy-init
        # hook doesn't try to create a duplicate on the first forward.
        alignment._awaiting_input_shape = False
        alignment._awaiting_output_shape = False
        alignment._auto_projector_initialized = True
        # If the alignment already has its own optimiser (BKD), add
        # the new projector's params now.  HKD wires this up later
        # when create_optimizer runs.
        if alignment.optimizer is not None:
            alignment._add_projector_params_to_optimizer()

    def _save_optimizer_and_scheduler(self, output_dir):
        """Save optimizer/scheduler, bypassing FSDP's optimizer save.

        FSDP's ``save_fsdp_optimizer`` calls ``FSDP.optim_state_dict()``
        which expects a standard optimizer state dict and properly
        initialized FSDP state-dict settings on the model.  Both
        ``CompositeOptimizer`` (custom format) and manually-spawned
        processes (missing FSDP state-dict context) can cause failures.

        We bypass FSDP's optimizer save entirely and write the optimizer
        state dict directly.
        """
        if self.is_fsdp_enabled:
            # Unwrap AcceleratedOptimizer if present
            raw_optim: Any = getattr(self.optimizer, "optimizer", self.optimizer)
            if self.args.should_save:
                torch.save(
                    raw_optim.state_dict(),
                    os.path.join(output_dir, "optimizer.pt"),
                )
            return

        super()._save_optimizer_and_scheduler(output_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Load optimizer/scheduler from checkpoint."""
        if checkpoint:
            state_path = Path(checkpoint) / self._STUDENT_TRAINING_STATE_FILE
            if state_path.exists():
                self._load_student_training_state(str(checkpoint))
            else:
                super()._load_optimizer_and_scheduler(checkpoint)
            self._load_distiller_state(str(checkpoint))

    def _load_distiller_state(self, checkpoint_dir: str) -> None:
        """Hook for subclasses to load distiller-specific training state.

        Projector weights are loaded once via ``_load_from_checkpoint``.
        Subclasses should call super()._load_distiller_state() when overriding.
        """

    # -------------------------------------------------------------------------
    # Capture engine lifecycle
    # -------------------------------------------------------------------------

    def _is_capture_active(self) -> bool:
        """Check if capture hooks are currently registered.

        Returns True if no capture engines exist (nothing to register).
        Otherwise returns True if any engine is registered.
        """
        if not self._capture_engines:
            return True  # No engines = no capture needed
        return any(e.is_registered for e in self._capture_engines)

    def _register_capture(self) -> None:
        """Register forward hooks and input capture wrappers for all capture engines."""
        for engine in self._capture_engines:
            engine.register()

    def _deregister_capture(self) -> None:
        """Remove all forward hooks and restore original forward methods."""
        for engine in self._capture_engines:
            engine.deregister()

    def _clear_captured_data(self) -> None:
        """Clear captured data from all capture engines."""
        for engine in self._capture_engines:
            engine.clear_captured()

    # -------------------------------------------------------------------------
    # Teacher placement
    # -------------------------------------------------------------------------

    def _setup_teacher_placement(self) -> None:
        """Place the teacher model according to ``args.teacher_placement``.

        Dispatches to the appropriate strategy in
        :mod:`silverspoon_kd.distributed.strategies`.  Runs once, at the
        start of :meth:`train` or of :meth:`evaluate`, whichever comes first.
        """
        if self._teacher_placed:
            return
        self._teacher_placed = True

        # Skip if teacher is already wrapped in FSDP (user pre-sharded it)
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(self.teacher_model, FSDP):
                return
        except ImportError:
            pass

        # Skip if teacher is already on CUDA and no explicit placement was
        # requested.  Default "replicated" would move the teacher to the
        # student device, overwriting deliberate pre-placement (e.g. PP).
        placement = getattr(self.args, "teacher_placement", "replicated")
        if placement == "replicated":
            try:
                first_device = next(self.teacher_model.parameters()).device
                if first_device.type == "cuda":
                    return
            except StopIteration:
                pass

        from ..distributed import strategies

        if isinstance(placement, str):
            if placement == "replicated":
                assert isinstance(self.teacher_model, nn.Module)
                strategies.place_teacher_replicated(self.teacher_model, self.args.device)
            elif placement == "sharded":
                if not torch.distributed.is_initialized():
                    raise RuntimeError(
                        "teacher_placement='sharded' requires torch.distributed "
                        "to be initialized (use torchrun or similar launcher)."
                    )
                device_id = (
                    torch.distributed.get_rank() % torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else 0
                )
                self.teacher_model = strategies.shard_teacher_fsdp_all_ranks(
                    self.teacher_model,
                    device_id=device_id,
                    wrap_cls=None,
                )
                self.teacher_model.eval()
            return

        if not isinstance(placement, TeacherPlacement):
            return

        # Split-GPU strategies — require distributed context
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                f"TeacherPlacement(strategy={placement.strategy!r}) requires "
                "torch.distributed to be initialized."
            )

        from ..distributed.split_gpu import compute_student_gpus, get_remapped_teacher_devices

        student_gpus = compute_student_gpus(placement.teacher_only_devices)
        remapped = get_remapped_teacher_devices(placement.teacher_only_devices, student_gpus)
        device_type = placement.device_type or "cuda"

        if placement.strategy == "pp":
            self.teacher_model = strategies.place_teacher_pp(
                self.teacher_model, remapped, device_type
            )
        elif placement.strategy == "tp":
            self.teacher_model = strategies.parallelize_teacher_tp(
                self.teacher_model, remapped, device_type
            )
            # TP requires torch.cuda.current_device() to match the
            # DeviceMesh during forward.  Install hooks to switch device
            # context around teacher forward and back to student after.
            strategies.install_tp_device_hooks(self.teacher_model)
        elif placement.strategy == "sharded":
            self.teacher_model = strategies.shard_teacher_fsdp_split(
                self.teacher_model,
                remapped,
                device_type,
                wrap_cls=placement.wrap_cls,
            )
            self.teacher_model.eval()

    # -------------------------------------------------------------------------
    # Forward overlap
    # -------------------------------------------------------------------------

    def _setup_forward_overlap(self) -> None:
        """Enable teacher/student forward overlap when on different CUDA devices.

        When the teacher is on different GPU(s) from the student, running both
        forward passes in parallel via CUDA streams saves wall-clock time.
        Auto-detected unless ``args.overlap_teacher_forward`` explicitly overrides.
        """
        explicit = getattr(self.args, "overlap_teacher_forward", None)
        if explicit is False:
            self._overlap_teacher_forward = False
            self._teacher_stream = None
            return

        if explicit is True or self._should_overlap_forward():
            devices = {p.device for p in self.teacher_model.parameters()}
            if not devices:
                return
            teacher_device = next(iter(devices))
            if teacher_device.type != "cuda":
                return
            self._overlap_teacher_forward = True
            self._teacher_stream = torch.cuda.Stream(device=teacher_device)
            logger.info(
                "Forward overlap enabled: teacher stream on %s, student on %s",
                teacher_device,
                self.args.device,
            )

    def _should_overlap_forward(self) -> bool:
        """Check if teacher and student are on different CUDA devices."""
        if not torch.cuda.is_available():
            return False
        devices = {p.device for p in self.teacher_model.parameters()}
        if len(devices) != 1:
            return False  # PP with multiple devices — skip
        teacher_device = next(iter(devices))
        student_device = self.args.device
        return (
            teacher_device.type == "cuda"
            and student_device.type == "cuda"
            and teacher_device != student_device
        )

    # -------------------------------------------------------------------------
    # Forward pass helpers
    # -------------------------------------------------------------------------

    def _run_teacher_forward(
        self,
        teacher_inputs: dict[str, Any],
        catch_truncation: bool = False,
    ) -> Any:
        """Run teacher forward pass with optional stream overlap and truncation handling.

        Handles:
        - CUDA stream context for forward overlap (when enabled)
        - ``torch.no_grad()`` context (teacher never needs gradients)
        - Optional ``_TruncatedForwardException`` catching (for auto-truncated models)

        Args:
            teacher_inputs: Pre-prepared inputs for the teacher model.
            catch_truncation: If True, catch ``_TruncatedForwardException``
                and return None instead of propagating.

        Returns:
            Teacher model output, or None if truncation was caught.
        """
        stream_ctx = (
            torch.cuda.stream(self._teacher_stream)
            if self._overlap_teacher_forward
            else contextlib.nullcontext()
        )
        with stream_ctx, torch.no_grad():
            if catch_truncation:
                try:
                    return self.teacher_model(**teacher_inputs)
                except _TruncatedForwardException:
                    return None
            else:
                return self.teacher_model(**teacher_inputs)

    def _sync_teacher_stream(self) -> None:
        """Wait for the teacher CUDA stream to complete.

        No-op when forward overlap is not enabled.
        """
        if self._overlap_teacher_forward and self._teacher_stream is not None:
            torch.cuda.current_stream().wait_stream(self._teacher_stream)

    @staticmethod
    def _grad_context(is_training: bool):
        """Return the appropriate gradient context for a forward pass.

        During training, returns a no-op context (gradients flow normally).
        During evaluation, returns ``torch.no_grad()``.

        Args:
            is_training: Whether this is a training step.

        Returns:
            Context manager for gradient computation.
        """
        return contextlib.nullcontext() if is_training else torch.no_grad()

    # -------------------------------------------------------------------------
    # Training lifecycle
    # -------------------------------------------------------------------------

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """
        Train the student model using knowledge distillation.

        This method:
        1. Sets up teacher placement (if configured)
        2. Initializes FLOP counter
        3. Starts profiler (if enabled)
        4. Registers forward hooks and input capture wrappers
        5. Performs training
        6. Deregisters hooks after training
        7. Stops profiler and generates report (if enabled)

        Args:
            *args: Positional arguments passed to parent Trainer.train()
            **kwargs: Keyword arguments passed to parent Trainer.train()

        Returns:
            Training results from parent Trainer.train()
        """
        # Set up teacher placement
        self._setup_teacher_placement()
        self._setup_forward_overlap()

        # Ensure model supports gradient_checkpointing_enable if requested
        if getattr(self.args, "gradient_checkpointing", False) and not hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            logger.warning(
                "gradient_checkpointing=True but model (%s) does not "
                "support gradient_checkpointing_enable(). Adding no-op. "
                "Use a PreTrainedModel for full gradient checkpointing.",
                type(self.model).__name__,
            )
            # Accept both Trainer conventions (transformers >= 5.16 also passes
            # ``every_n_layers``).
            self.model.gradient_checkpointing_enable = (
                lambda gradient_checkpointing_kwargs=None, **kwargs: None
            )
            self.model.gradient_checkpointing_disable = lambda: None

        # Initialize FLOP counter
        self.flop_counter = 0

        # Start profiler if enabled
        self._start_profiler()

        # Register forward hooks and input capture wrappers
        self._register_capture()

        try:
            result = super().train(*args, **kwargs)
        finally:
            # Clean up: remove hooks and restore original forward methods
            self._deregister_capture()

            # Stop profiler and generate report
            self._stop_profiler()

        return result

    def training_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Template training step: forward → loss → backward → clip → profiler → clear.

        Subclasses implement compute_distillation_loss() for their specific logic.
        """
        model.train()
        self._zero_projector_gradients()
        self._sync_projectors_from_rank0()

        self._reset_step_metrics(is_training=True)
        inputs = self._prepare_inputs(inputs)

        flop_context = self._get_flop_context(self._should_count_flops())
        with flop_context:
            # Always pass self.model (unwrapped) — Trainer may wrap `model`
            # in DataParallel, which would replicate capture engine hooks.
            loss = self.compute_distillation_loss(self.model, inputs, is_training=True)
        self._record_flops(flop_context)
        self._update_flop_counter()
        if self.flops_per_step:
            self._track_metric("flops/step", float(self.flops_per_step))

        self._backward_loss(loss)

        # Projectors live outside the student model, so DDP/FSDP never reduce
        # their gradients; average them here once per accumulation window.
        if self.accelerator.sync_gradients:
            self._all_reduce_projector_gradients()

        # Student-only gradient clipping (when clip_projectors=False)
        if self._student_only_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._student_only_clip_norm)

        if self._USE_COMPOSITE_OPTIMIZER:
            self._clip_student_gradients()
            self._track_student_learning_rates()

        # ``sync_gradients`` is True on the micro-batch that completes an
        # accumulation window; the Trainer steps the optimizer right after.
        if self.accelerator.sync_gradients:
            self._clear_projector_grads_next_step = True

        self._after_training_step(loss)
        self._profiler_step()
        self._clear_captured_data()
        # Ensure detached loss is on the device the Trainer expects (args.device)
        # for cross-device setups where the student may be on a different GPU.
        detached = loss.detach()
        if detached.device != self.args.device:
            detached = detached.to(self.args.device)
        return detached

    # -------------------------------------------------------------------------
    # Evaluation methods
    # -------------------------------------------------------------------------

    def compute_loss(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        """
        Template compute_loss: forward → loss → store metrics → clear.

        Subclasses implement compute_distillation_loss() for their specific logic.
        """
        self._reset_step_metrics(is_training=self.model.training)
        inputs = self._prepare_inputs(inputs)
        # Always pass self.model (unwrapped) — Trainer may wrap `model`
        # in DataParallel, which would replicate capture engine hooks.
        loss: torch.Tensor = self.compute_distillation_loss(
            self.model, inputs, is_training=self.model.training
        )
        if not self.model.training:
            self._store_eval_metrics()
        self._clear_captured_data()
        # Ensure loss is on the device the Trainer expects (args.device)
        # for cross-device setups where the student may be on a different GPU.
        if loss.device != self.args.device:
            loss = loss.to(self.args.device)
        return (loss, None) if return_outputs else loss

    def compute_distillation_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        is_training: bool,
    ) -> torch.Tensor:
        """Compute the distillation loss. Subclasses must override this."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_distillation_loss()"
        )

    def _anchor_fsdp_output(self, loss: torch.Tensor, model_output: Any) -> torch.Tensor:
        """Include a zero-weighted term from the model output in the loss.

        When the student is FSDP-wrapped, FSDP registers pre-backward hooks
        on the model's forward output tensors during ``_post_forward`` and
        then sets its training state to ``IDLE``.  The pre-backward hooks
        restore the state to ``FORWARD_BACKWARD`` when backward reaches them.

        Distillers compute the loss from *captured intermediate outputs*
        (via the capture engine) rather than from the model's final output.
        If the final output is discarded, the pre-backward hooks never fire
        and the FSDP state stays ``IDLE``, causing an assertion error during
        the ``FlatParameter`` post-backward hook.

        Adding ``0 * output.sum()`` keeps the output in the autograd graph
        so the pre-backward hooks fire correctly.  The ``* 0`` ensures zero
        gradient contribution.
        """
        if model_output is None:
            if self.is_fsdp_enabled:
                logger.warning(
                    "Student forward produced no output (likely truncated), but "
                    "the student is FSDP-wrapped. FSDP pre-backward hooks will "
                    "not fire, which may leave FSDP in an inconsistent state. "
                    "This should not happen — auto_truncate is normally disabled "
                    "for FSDP students. If you see FSDP assertion errors after "
                    "this warning, ensure auto_truncate=False for FSDP students."
                )
            return loss
        out = model_output
        if hasattr(out, "logits"):
            out = out.logits
        if isinstance(out, torch.Tensor) and out.requires_grad:
            return loss + 0 * out.sum()
        return loss

    # -------------------------------------------------------------------------
    # Cross-rank projector synchronisation
    # -------------------------------------------------------------------------

    @staticmethod
    def _world_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def _sync_projectors_from_rank0(self, alignment: Alignment | None = None) -> None:
        """Broadcast projector parameters from rank 0 the first time each is seen.

        Projectors are owned by the alignments, not by the student model, so
        DDP and FSDP never synchronise them.  Every rank creates its own copy,
        auto-projectors even with rank-local random initialisation; this makes
        all copies identical before they are used.  Called at the start of
        every training step (covers projectors given explicitly) and right
        after an auto-projector is materialised, on all ranks in the same
        order.
        """
        if self._world_size() == 1:
            return
        alignments = [alignment] if alignment is not None else self.alignments
        for align in alignments:
            for proj in (align.input_projector, align.output_projector):
                if proj is None or id(proj) in self._synced_projector_ids:
                    continue
                for tensor in list(proj.parameters()) + list(proj.buffers()):
                    torch.distributed.broadcast(tensor.data, src=0)
                self._synced_projector_ids.add(id(proj))

    def _all_reduce_projector_gradients(self) -> None:
        """Average projector gradients across ranks, as DDP does for the student's."""
        world_size = self._world_size()
        if world_size == 1:
            return
        for align in self.alignments:
            for proj in (align.input_projector, align.output_projector):
                if proj is None:
                    continue
                for param in proj.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                        param.grad.div_(world_size)

    # Set at the end of a training step that completes a gradient-accumulation
    # window (the Trainer runs the optimizer step right after it), so the next
    # training step starts by clearing the projector gradients.
    _clear_projector_grads_next_step: bool = False

    def _zero_projector_gradients(self) -> None:
        """Clear projector gradients at the start of an accumulation window.

        The HF Trainer clears gradients via ``model.zero_grad()`` after each
        optimizer step, but that only covers ``model.parameters()``.
        Projectors are owned by :class:`~silverspoon_kd.alignments.Alignment`
        objects and are not registered as submodules of the student model, so
        their gradients would otherwise carry over into the next optimizer
        step.

        Called at the start of every :meth:`training_step`.  Gradients are
        cleared only on the first micro-batch after an optimizer step, so
        with ``gradient_accumulation_steps > 1`` the projector gradients
        accumulate across micro-batches exactly like the student's do.
        """
        if not self._clear_projector_grads_next_step:
            return
        self._clear_projector_grads_next_step = False
        for alignment in self.alignments:
            for proj in (alignment.input_projector, alignment.output_projector):
                if proj is not None:
                    proj.zero_grad(set_to_none=True)

    def _backward_loss(self, loss: torch.Tensor) -> None:
        """Perform backward pass. Override for custom backward (e.g., streaming blockwise)."""
        self.accelerator.backward(loss)

    def _after_training_step(self, loss: torch.Tensor) -> None:
        """Hook called after each training step. Override for post-step logic."""

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Override prediction_step to force compute_loss call even without labels.

        The default prediction_step skips compute_loss when labels aren't present,
        but distillation doesn't use labels - it aligns hidden states.

        When ``compute_metrics`` is set and labels are present in *inputs*,
        a separate student-only forward pass produces logits so that HF
        Trainer's evaluation loop can call ``compute_metrics``.  This
        enables accuracy tracking and accuracy-based early stopping during
        distillation.

        Args:
            model: The student model
            inputs: Dictionary of input tensors
            prediction_loss_only: Whether only loss is needed
            ignore_keys: Keys to ignore in outputs (unused)

        Returns:
            Tuple of (loss, logits, labels).  *logits* and *labels* are
            non-None only when ``compute_metrics`` is set and labels are
            available in the inputs.
        """
        # Tell compute_distillation_loss whether to stash the student output
        # so we can extract logits for compute_metrics.  The flag is checked
        # by subclass overrides (HolisticDistiller, ResponseBasedDistiller)
        # to avoid unconditionally holding a (batch, seq_len, vocab_size)
        # tensor — which would OOM on large-vocab LLMs.
        self._needs_student_logits = (
            self.compute_metrics is not None and not prediction_loss_only and "labels" in inputs
        )

        # Wrap in torch.no_grad() — Trainer's evaluation_loop delegates
        # gradient context management to prediction_step.
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)

        # Handle tuple return from compute_loss (in case return_outputs is True)
        if isinstance(loss, tuple):
            loss = loss[0]

        # Advance the profiler schedule so torch.profiler records per-eval-batch
        # activity when profiling is enabled.
        self._profiler_step()

        logits = None
        labels = None
        if self._needs_student_logits:
            # Reuse the student output already produced by compute_loss →
            # compute_distillation_loss, avoiding a redundant forward pass.
            student_output = getattr(self, "_last_student_output", None)
            if student_output is not None and hasattr(student_output, "logits"):
                logits = student_output.logits.detach()
                labels = inputs["labels"]

        # Free the stashed output and reset the flag immediately to avoid
        # holding a full (batch, seq_len, vocab_size) tensor until the next
        # step.
        self._last_student_output = None
        self._needs_student_logits = False

        return (loss.detach(), logits, labels)

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Evaluate the student model with per-layer metrics.

        This method:
        1. Initializes per-layer metric collection
        2. Runs the standard evaluation loop (which calls compute_loss)
        3. Aggregates and logs per-layer metrics
        4. Returns combined metrics

        Args:
            eval_dataset: Dataset to evaluate on (defaults to self.eval_dataset)
            ignore_keys: Output keys to ignore (unused in distillation)
            metric_key_prefix: Prefix for metric names (e.g., "eval", "test")

        Returns:
            Dictionary of evaluation metrics including per-layer losses
        """
        # The teacher is placed lazily so that evaluate() works before train().
        self._setup_teacher_placement()

        # Auto-register capture hooks if they aren't active (e.g. called after train())
        needs_capture = not self._is_capture_active()
        if needs_capture:
            logger.warning(
                "Capture hooks were not registered before evaluate(). "
                "Auto-registering them. For explicit control, call "
                "_register_capture() before evaluate()."
            )
            self._register_capture()

        try:
            # Store prefix for use in compute_loss
            self.metric_key_prefix = metric_key_prefix
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

            # Initialize per-layer metric aggregation dictionary for all alignments
            self.per_layer_eval_metrics: dict[str, list] = {}
            for alignment in self.alignments:
                key = f"{metric_key_prefix}_loss/{alignment.get_name()}"
                self.per_layer_eval_metrics[key] = []

            # Run standard evaluation loop (calls our custom compute_loss)
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            new_metrics = {}

            # Average per-layer metrics across all evaluation steps.
            # Values may be GPU tensors (scalar conversion deferred to here,
            # mirroring how the HF Trainer's EvalLoopContainer defers
            # numpy conversion to the end of the evaluation loop).
            for key, values in self.per_layer_eval_metrics.items():
                if values:
                    new_metrics[key] = torch.stack(values).mean().item()

            # Hook for subclasses to add additional metrics (e.g., WeightWatcher)
            additional_metrics = self._compute_additional_eval_metrics(metric_key_prefix)
            new_metrics.update(additional_metrics)

            # Log and merge per-layer metrics with overall metrics
            self.log(new_metrics)
            metrics.update(new_metrics)

            # End-to-end evaluation loss (runs full student model if configured)
            e2e_metrics = self._compute_e2e_eval_loss(eval_dataset, metric_key_prefix)
            if e2e_metrics:
                self.log(e2e_metrics)
                metrics.update(e2e_metrics)

            # Clean up temporary attributes
            del self.per_layer_eval_metrics
            del self.metric_key_prefix

            return metrics
        finally:
            if needs_capture:
                self._deregister_capture()

    def _get_e2e_student_models(self) -> dict[str, nn.Module] | None:
        """
        Return full student models for end-to-end evaluation, keyed by model name.

        Returns None by default (e2e eval not available). Subclasses that have
        access to complete student models should override this.

        Returns:
            Dictionary mapping model names to student models, or None
        """
        return None

    def _compute_e2e_eval_loss(
        self,
        eval_dataset: Any | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Compute end-to-end evaluation metrics by running the full student model(s).

        Runs when *either* condition is met:
        - ``e2e_eval_loss`` is configured (computes e2e loss), or
        - ``compute_metrics`` is set on a distiller whose
          :meth:`prediction_step` cannot produce logits
          (``_MAIN_LOOP_COMPUTES_METRICS = False``, e.g.
          :class:`BlockwiseDistiller`).

        Handles the full lifecycle: save/clear terminal modules, deregister
        capture, run optional warmup, evaluate, re-register, restore terminals.

        Args:
            eval_dataset: Dataset to evaluate on (defaults to self.eval_dataset)
            metric_key_prefix: Prefix for metric names

        Returns:
            Dictionary of e2e metrics, empty if disabled or unavailable
        """
        needs_e2e_loss = getattr(self.args, "e2e_eval_loss", None) is not None
        needs_e2e_metrics = (
            self.compute_metrics is not None and not self._MAIN_LOOP_COMPUTES_METRICS
        )
        if not needs_e2e_loss and not needs_e2e_metrics:
            return {}

        if not self._get_e2e_student_models():
            return {}

        saved_auto_truncate = [(e, e.auto_truncate) for e in self._capture_engines]
        for e, _ in saved_auto_truncate:
            e.auto_truncate = False

        self._deregister_capture()
        try:
            self._warmup_before_e2e_eval(eval_dataset)
            return self._run_e2e_eval(eval_dataset, metric_key_prefix)
        finally:
            self._register_capture()
            for e, saved in saved_auto_truncate:
                e.auto_truncate = saved

    def _warmup_before_e2e_eval(self, eval_dataset=None):
        """Hook for subclass-specific warmup before e2e eval (e.g., BatchNorm warmup)."""

    def _run_e2e_eval(
        self,
        eval_dataset: Any | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Run the actual e2e evaluation loop over student models.

        When ``compute_metrics`` is set and :meth:`prediction_step` cannot
        produce logits (``_MAIN_LOOP_COMPUTES_METRICS = False``, e.g. in
        :class:`BlockwiseDistiller`), this method also collects
        predictions from the *first* student model and calls
        ``compute_metrics`` at the end.  This piggybacks on the same
        forward pass used for e2e loss, avoiding a redundant pass.
        """
        student_models = self._get_e2e_student_models()
        if student_models is None:
            return {}
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert eval_dataset is not None, "eval_dataset required for e2e eval"

        # Build a DataLoader without the Trainer's RemoveColumnsCollator so that
        # all dataset columns (including 'labels') are preserved regardless of
        # which model self.model points to (e.g., a DummyModel in Blockwise).
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )

        collect_predictions = (
            self.compute_metrics is not None and not self._MAIN_LOOP_COMPUTES_METRICS
        )
        compute_e2e_loss = getattr(self.args, "e2e_eval_loss", None) is not None

        metrics = {}
        single_model = len(student_models) == 1
        # Collect predictions from the first student model only.
        first_model = True
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for model_name, model in student_models.items():
            model.eval()

            # Get accepted forward params for input filtering
            sig = inspect.signature(model.forward)
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            accepted_params = None if has_var_keyword else set(sig.parameters.keys())

            total_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                batch = self._prepare_inputs(batch)

                # Filter batch keys to match model's forward signature
                if accepted_params is not None:
                    filtered = {k: v for k, v in batch.items() if k in accepted_params}
                else:
                    filtered = batch

                with torch.no_grad():
                    outputs = model(**filtered)

                if compute_e2e_loss:
                    loss = self._extract_loss(outputs)

                    if loss is not None:
                        total_loss += loss.item()
                        num_batches += 1
                    else:
                        self._warn_once(
                            f"e2e_no_loss_{model_name}",
                            "e2e_eval_loss='%s': model '%s' forward() did not "
                            "return a loss. Ensure the dataset includes 'labels' "
                            "and the model computes loss when labels are provided.",
                            self.args.e2e_eval_loss,
                            model_name,
                        )

                if collect_predictions and first_model:
                    logits = getattr(outputs, "logits", None)
                    batch_labels = batch.get("labels")
                    if logits is not None and batch_labels is not None:
                        all_preds.append(logits.detach().cpu())
                        all_labels.append(batch_labels.cpu())

            if compute_e2e_loss and num_batches > 0:
                avg_loss = total_loss / num_batches
                if single_model:
                    metrics[f"{metric_key_prefix}_loss/e2e"] = avg_loss
                else:
                    metrics[f"{metric_key_prefix}_loss/e2e/{model_name}"] = avg_loss

            first_model = False

        # Call compute_metrics with collected predictions.
        if all_preds and self.compute_metrics is not None:
            from transformers import EvalPrediction

            preds = torch.cat(all_preds).numpy()
            label_ids = torch.cat(all_labels).numpy()
            cm = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            for k, v in cm.items():
                metrics[f"{metric_key_prefix}_{k}"] = v

        return metrics

    def _compute_additional_eval_metrics(self, metric_key_prefix: str) -> dict[str, float]:
        """
        Compute additional evaluation metrics including WeightWatcher analysis.

        Args:
            metric_key_prefix: Prefix for metric names

        Returns:
            Dictionary of additional metrics
        """
        metrics = {}

        if self.args.use_weightwatcher and WEIGHTWATCHER_AVAILABLE:
            ww_metrics = self._run_weightwatcher_analysis(metric_key_prefix)
            metrics.update(ww_metrics)

        return metrics

    def _run_weightwatcher_analysis(self, metric_key_prefix: str) -> dict[str, float]:
        """
        Run WeightWatcher analysis on the student model.

        When alignments are present (HKD/BKD), analyses each student block
        individually. Otherwise analyses the full student model.

        Args:
            metric_key_prefix: Prefix for metric names

        Returns:
            Dictionary of WeightWatcher metrics
        """
        ww_metrics = {}

        if self.alignments:
            # Feature-based: analyze each aligned student block
            for alignment in self.alignments:
                name = alignment.get_name()
                try:
                    watcher = ww.WeightWatcher(model=alignment.student_block)
                    details = watcher.analyze()
                    summary = watcher.get_summary(details)
                    for metric_name, metric_value in summary.items():
                        ww_metrics[f"{metric_key_prefix}_ww_{metric_name}/{name}"] = float(
                            metric_value
                        )
                except Exception as e:
                    logger.warning("WeightWatcher analysis failed for %s: %s", name, e)
        else:
            # Response-based: analyze the full student model
            try:
                import weightwatcher

                watcher = weightwatcher.WeightWatcher(model=self.student_model)
                details = watcher.analyze()
                summary = watcher.get_summary(details)
                for metric_name, metric_value in summary.items():
                    ww_metrics[f"{metric_key_prefix}_ww_{metric_name}/student"] = float(
                        metric_value
                    )
            except Exception as e:
                logger.warning("WeightWatcher analysis failed for student model: %s", e)

        return ww_metrics

    def _store_eval_metrics(self) -> None:
        """
        Store current step metrics into per-layer evaluation metrics.

        This should be called during compute_loss to accumulate metrics
        for later averaging in evaluate().
        """
        if hasattr(self, "per_layer_eval_metrics"):
            for key, value in self.current_step_metrics.items():
                full_key = f"{self.metric_key_prefix}_{key}"
                if full_key in self.per_layer_eval_metrics:
                    self.per_layer_eval_metrics[full_key].append(value)

    def _get_averaged_metrics(self) -> dict[str, float]:
        """
        Return averaged metrics from the training accumulator and clear it.

        Called by log() at the Trainer's logging_steps cadence so that custom
        metrics are averaged over the same window as ``train/loss``.  This is
        the **only** place where ``.item()`` is called on accumulated tensors,
        mirroring how the HF Trainer defers scalar conversion of ``tr_loss``
        to log time.

        Returns:
            Dictionary mapping metric names to their averaged values.
        """
        num_steps = max(self._metric_accumulator_steps, 1)
        averaged = {}
        for k, v in self._training_metric_accumulator.items():
            scalar = v.item() if isinstance(v, torch.Tensor) else float(v)
            averaged[k] = scalar / num_steps
        self._training_metric_accumulator.clear()
        self._metric_accumulator_steps = 0
        return averaged

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Override log method to inject averaged training metrics and filter
        the global learning_rate metric.

        The HuggingFace Trainer averages ``train/loss`` over ``logging_steps``
        batches.  This override applies the same averaging to all custom
        distiller metrics (per-layer losses, learning rates, gradient norms)
        that have been accumulated via ``_track_metric``.

        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for computing throughput metrics
        """
        if self._training_metric_accumulator:
            averaged = self._get_averaged_metrics()
            # Auto-filter global learning_rate when per-student LR metrics exist
            if any(k.startswith("learning_rate/") for k in averaged):
                logs.pop("learning_rate", None)
            logs.update(averaged)
            if self.flop_counter:
                logs["flops/total"] = float(self.flop_counter)
        super().log(logs, start_time=start_time)
