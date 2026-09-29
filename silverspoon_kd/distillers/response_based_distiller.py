"""
Response-based knowledge distiller for classic output-level distillation.

This module provides the
[ResponseBasedDistiller][silverspoon_kd.ResponseBasedDistiller] class for
Hinton-style knowledge distillation where the student learns from the
teacher's soft output probabilities.

Reference:
    Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
    https://arxiv.org/abs/1503.02531
"""

import logging
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn
from transformers import PreTrainedModel, Trainer

from ..losses.kl import kl_divergence_loss
from ..losses.liger import (
    LIGER_KERNEL_AVAILABLE,
    FusedLinearKLDivLoss,
    LigerFusedLinearJSDLoss,
)
from ..losses.registry import get_loss_function
from ..training_arguments import TrainingArguments
from .base_distiller import WEIGHTWATCHER_AVAILABLE, BaseDistiller, _warn_irrelevant_args

logger = logging.getLogger(__name__)


class ResponseBasedDistiller(BaseDistiller):
    """
    Response-based knowledge distillation (ResKD).

    A distiller for response-based (output-level) knowledge distillation.
    The student learns to match the teacher's final output predictions (soft
    targets/logits), in contrast to feature-based approaches
    ([HolisticDistiller][silverspoon_kd.HolisticDistiller],
    [BlockwiseDistiller][silverspoon_kd.BlockwiseDistiller]) which match
    intermediate representations.

    The soft loss function is user-provided (any callable matching the
    ``(student_logits, teacher_logits) → scalar`` signature used by all
    registered loss functions). The hard loss comes from the model's own
    ``outputs.loss`` (the standard HuggingFace convention).

    Example:
        ```python
        from silverspoon_kd.losses import kl_divergence_loss

        args = TrainingArguments(alpha=0.5)
        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            soft_loss_fn=kl_divergence_loss(temperature=4.0, chunk_size=1024),
            train_dataset=dataset,
            args=args,
        )
        distiller.train()
        ```
    """

    args: TrainingArguments  # pyright: ignore[reportIncompatibleVariableOverride]
    _MAIN_LOOP_COMPUTES_METRICS: bool = True

    # Populated only while ``evaluate`` runs: per-batch soft/hard loss values
    # for the component breakdown, and the metric prefix used for logging.
    _eval_component_losses: dict[str, list[float]] | None = None
    _eval_metric_prefix: str | None = None

    def __init__(
        self,
        student_model: PreTrainedModel | nn.Module,
        teacher_model: PreTrainedModel | nn.Module,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        args: TrainingArguments | None = None,
        soft_loss_fn: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        soft_loss_fn_kwargs: dict[str, Any] | None = None,
        prepare_teacher_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        prepare_student_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        use_liger_kernel: bool = False,
        output_head_layer: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the ResponseBasedDistiller.

        Args:
            student_model: The student model to train. Should output logits.
            teacher_model: The teacher model to distill from. Should output logits.
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            args: TrainingArguments with training parameters.
                Contains alpha (soft/hard weighting), magnitude_aware_weighting,
                and device/dtype matching flags.
            soft_loss_fn: Loss function for soft targets. Can be:
                - A callable ``(student_logits, teacher_logits) → scalar``
                - A string name from the loss registry (e.g. "kl_div", "jsd")
                - None (defaults to ``kl_divergence_loss()``)
            soft_loss_fn_kwargs: Optional keyword arguments forwarded to the
                loss factory when ``soft_loss_fn`` is a string name. Ignored
                when ``soft_loss_fn`` is already a callable.
            prepare_teacher_inputs: Optional callable to transform inputs before
                                   passing to teacher.
            prepare_student_inputs: Optional callable to transform inputs before
                                   passing to student.
            use_liger_kernel: Whether to use Liger fused linear kernel for
                memory-efficient loss computation. When True, the soft loss is
                computed by a fused kernel that avoids materializing the full
                logit tensor. Requires the ``liger-kernel`` package and
                ``output_head_layer`` to be set. Default: False
            output_head_layer: Attribute path to the output head linear layer
                (e.g. "lm_head" for language models, "classifier" for
                classification models). Required when use_liger_kernel=True.
            **kwargs: Additional keyword arguments passed to Trainer
        """
        if args is None:
            args = TrainingArguments()

        _warn_irrelevant_args(
            args,
            used_params={
                "alpha",
                "auto_device_match",
                "auto_dtype_match",
                "magnitude_aware_weighting",
            },
            distiller_name="ResponseBasedDistiller",
        )

        # Validate and set up Liger kernel
        if use_liger_kernel:
            if not LIGER_KERNEL_AVAILABLE:
                raise ImportError(
                    "use_liger_kernel=True requires the liger-kernel package. "
                    "Install it with: pip install liger-kernel>=0.4.0"
                )
            if output_head_layer is None:
                raise ValueError(
                    "output_head_layer is required when use_liger_kernel=True. "
                    "Provide the attribute path to the output head linear "
                    "layer (e.g. output_head_layer='lm_head' for language "
                    "models, or 'classifier' for classification models)."
                )

        super().__init__(
            teacher_model=teacher_model,
            alignments=[],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            model=student_model,
            prepare_teacher_inputs=prepare_teacher_inputs,
            prepare_student_inputs=prepare_student_inputs,
            **kwargs,
        )

        self.output_head_layer = output_head_layer
        self.use_liger_kernel = use_liger_kernel

        # Resolve soft loss function
        if soft_loss_fn is None:
            self.soft_loss_fn = kl_divergence_loss()
        elif isinstance(soft_loss_fn, str):
            self.soft_loss_fn = get_loss_function(soft_loss_fn, **(soft_loss_fn_kwargs or {}))
        else:
            self.soft_loss_fn = soft_loss_fn

        # Set up Liger fused loss if requested
        self._liger_loss: Any = None
        if use_liger_kernel:
            # Liger fused kernels bypass the standard soft_loss_fn path.
            # They are configured here and used in _compute_fused_loss.
            if (
                FusedLinearKLDivLoss is not None and isinstance(soft_loss_fn, FusedLinearKLDivLoss)
            ) or (
                LIGER_KERNEL_AVAILABLE
                and LigerFusedLinearJSDLoss is not None
                and isinstance(soft_loss_fn, LigerFusedLinearJSDLoss)
            ):
                self._liger_loss = soft_loss_fn
            elif FusedLinearKLDivLoss is not None:
                self._liger_loss = FusedLinearKLDivLoss()

        # Ensure "labels" is preserved by the Trainer's RemoveColumnsCollator
        # when hard loss is needed (alpha > 0).
        if args.alpha > 0:
            self._set_signature_columns_if_needed()
            if self._signature_columns is not None and "labels" not in self._signature_columns:
                self._signature_columns.append("labels")

        # Note: e2e_eval_loss is supported. Although ResponseBasedDistiller
        # runs the full student forward, its eval_loss is the distillation
        # loss (soft/hard combination). When e2e_eval_loss="forward", the
        # base class's _compute_e2e_eval_loss separately logs the student's
        # own forward loss as eval_loss/e2e.

    def _get_e2e_student_models(self):
        """Return the student model for e2e eval (used by _compute_e2e_eval_loss)."""
        name = getattr(self.model, "name_or_path", self.model.__class__.__name__)
        return {name: self.model}

    def compute_distillation_loss(self, model, inputs, is_training):
        """Run teacher/student forward passes and compute distillation loss."""
        teacher_inputs = self._prepare_teacher_inputs(inputs)
        student_inputs = self._prepare_student_inputs(inputs)

        # Liger fused path: hidden states + output head weights → fused kernel
        if self._liger_loss is not None:
            return self._compute_fused_loss(
                model, inputs, teacher_inputs, student_inputs, is_training
            )

        student_logits, teacher_logits, hard_loss = self._forward_pass(
            model, teacher_inputs, student_inputs
        )
        total_loss, soft_loss = self._compute_total_loss(
            student_logits,
            teacher_logits,
            hard_loss,
            track_metrics=is_training,
        )
        # Accumulate component losses for eval reporting; the accumulator only
        # exists while ``evaluate`` is running, so training steps skip this.
        if self._eval_component_losses is not None:
            self._eval_component_losses["soft"].append(soft_loss.item())
            if hard_loss is not None:
                self._eval_component_losses["hard"].append(hard_loss.item())
        return total_loss

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        """
        Extract logits from model outputs.

        Handles raw tensors, HuggingFace model outputs, and DataParallel
        gathered outputs (where .logits may be a ``map`` object instead of a
        tensor because ``DataParallel.gather`` maps over iterable outputs).

        Args:
            outputs: Model outputs (tensor or object with .logits attribute)

        Returns:
            Logits tensor
        """
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            if isinstance(logits, torch.Tensor):
                return logits
            # DataParallel.gather may produce a map/list; materialise it.
            return self._extract_logits(logits)
        # Bare iterable (e.g. map or list from DataParallel gather)
        if isinstance(outputs, (map, list, tuple)):
            items = list(outputs)
            if items:
                return self._extract_logits(items[0])
            raise ValueError(
                "Cannot extract logits: received an empty iterable "
                f"(type {type(outputs).__name__}). This can happen when "
                "DataParallel.gather produces an empty map."
            )
        raise ValueError(
            f"Cannot extract logits from outputs of type {type(outputs)}. "
            "Expected a tensor or an object with a 'logits' attribute."
        )

    # -------------------------------------------------------------------------
    # Liger fused kernel support
    # -------------------------------------------------------------------------

    def _get_output_head(self, model: nn.Module) -> nn.Linear:
        """
        Resolve the output head linear layer using the user-provided attribute path.

        Unwraps DataParallel if present.

        Args:
            model: The model to extract the output head from

        Returns:
            The output head linear layer

        Raises:
            ValueError: If output_head_layer is None or the attribute doesn't exist
        """
        if self.output_head_layer is None:
            raise ValueError(
                "output_head_layer is required for Liger fused kernel support. "
                "Provide the attribute path to the output head linear layer "
                "(e.g. output_head_layer='lm_head' for language models)."
            )

        # Unwrap DataParallel
        unwrapped = getattr(model, "module", model)

        # Resolve dotted path
        obj = unwrapped
        for attr in self.output_head_layer.split("."):
            obj = getattr(obj, attr)
        return cast(nn.Linear, obj)

    def _extract_hidden_states(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
    ) -> torch.Tensor:
        """
        Get last hidden states by hooking the output head's input.

        Registers a forward hook on the output head layer to capture its input
        tensor. Runs model(**inputs) and intercepts the hidden states
        right before the output projection.

        Args:
            model: The model to run
            inputs: Input tensors dict

        Returns:
            Hidden states tensor (B, S, H)
        """
        output_head = self._get_output_head(model)
        captured = {}

        def hook_fn(module, args, kwargs):
            captured["hidden"] = args[0]

        handle = output_head.register_forward_pre_hook(hook_fn, with_kwargs=True)
        try:
            model(**inputs)
        finally:
            handle.remove()

        return captured["hidden"]

    def _extract_hidden_states_and_output(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
    ) -> tuple[torch.Tensor, Any]:
        """Get last hidden states AND model output by hooking the output head's input.

        Like ``_extract_hidden_states`` but also returns the full model output,
        which is needed when combining the fused soft loss with a hard loss.

        Args:
            model: The model to run
            inputs: Input tensors dict

        Returns:
            Tuple of (hidden_states, model_output)
        """
        output_head = self._get_output_head(model)
        captured = {}

        def hook_fn(module, args, kwargs):
            captured["hidden"] = args[0]

        handle = output_head.register_forward_pre_hook(hook_fn, with_kwargs=True)
        try:
            output = model(**inputs)
        finally:
            handle.remove()

        return captured["hidden"], output

    def _compute_fused_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        teacher_inputs: dict[str, torch.Tensor | Any],
        student_inputs: dict[str, torch.Tensor | Any],
        is_training: bool,
    ) -> torch.Tensor:
        """
        Compute loss using Liger fused linear kernel.

        Extracts hidden states from both models and passes them with the
        output head weights to the fused loss function.  When ``alpha > 0``,
        also computes a hard (cross-entropy) loss from the student model's
        output and combines it with the fused soft loss.

        Args:
            model: The student model
            inputs: Original input tensors dict (for label access)
            teacher_inputs: Pre-prepared inputs for the teacher model
            student_inputs: Pre-prepared inputs for the student model
            is_training: Whether this is a training step

        Returns:
            Combined loss scalar
        """
        # Teacher hidden states (no gradients)
        with torch.no_grad():
            teacher_hidden = self._extract_hidden_states(self.teacher_model, teacher_inputs)

        # Student forward: capture hidden states, and model output when we
        # need hard loss (the forward already runs through the output head, so
        # capturing the output adds no extra computation).
        labels = inputs.get("labels")
        need_hard_loss = self.args.alpha > 0 and labels is not None

        if need_hard_loss:
            student_hidden, student_output = self._extract_hidden_states_and_output(
                model, student_inputs
            )
        else:
            student_hidden = self._extract_hidden_states(model, student_inputs)
            student_output = None

        # Auto device/dtype matching on hidden states
        teacher_hidden = self._match_device_dtype(
            teacher_hidden,
            student_hidden,
            self.args.auto_device_match,
            self.args.auto_dtype_match,
        )

        # Get output head weights
        student_head = self._get_output_head(model)
        teacher_head = self._get_output_head(self.teacher_model)

        soft_loss = self._liger_loss(
            student_hidden,
            student_head.weight,
            teacher_hidden,
            teacher_head.weight,
            labels,
            student_head.bias,
            teacher_head.bias,
        )

        # Combine with hard loss when alpha > 0
        if need_hard_loss:
            hard_loss = self._extract_loss(student_output)

            if hard_loss is not None:
                total = self._combine_losses(
                    {"soft": soft_loss, "hard": hard_loss},
                    {"soft": 1 - self.args.alpha, "hard": self.args.alpha},
                    track_metrics=is_training,
                )
                assert total is not None
                return total
            if is_training:
                self._track_metric("loss/soft", soft_loss)
            self._warn_once(
                "no_hard_loss",
                "alpha=%.2f and labels present, but the model's forward() "
                "did not return a loss. The effective loss is 100%% soft. "
                "Ensure the model computes loss when labels are provided.",
                self.args.alpha,
            )
        else:
            if is_training:
                self._track_metric("loss/soft", soft_loss)

        return soft_loss

    # -------------------------------------------------------------------------
    # Standard (non-Liger) forward pass and loss
    # -------------------------------------------------------------------------

    def _forward_pass(
        self,
        model: nn.Module,
        teacher_inputs: dict[str, torch.Tensor | Any],
        student_inputs: dict[str, torch.Tensor | Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Run forward passes through teacher and student, returning logits and hard loss.

        The hard loss is obtained from the model's own ``outputs.loss`` when
        available (architecture-agnostic — the model handles label shifting,
        masking, etc. internally).

        Args:
            model: The student model
            teacher_inputs: Pre-prepared inputs for the teacher model
            student_inputs: Pre-prepared inputs for the student model

        Returns:
            Tuple of (student_logits, teacher_logits, hard_loss)
        """
        # Teacher forward — optionally on a separate CUDA stream for overlap
        teacher_outputs = self._run_teacher_forward(teacher_inputs)
        teacher_logits = self._extract_logits(teacher_outputs)

        # Student forward (runs in parallel when overlapping)
        student_outputs = model(**student_inputs)
        if getattr(self, "_needs_student_logits", False):
            self._last_student_output = student_outputs
        student_logits = self._extract_logits(student_outputs)

        # Sync teacher stream before using teacher_logits
        self._sync_teacher_stream()

        # Hard loss: use the model's own loss (architecture-agnostic)
        hard_loss = None
        if self.args.alpha > 0:
            hard_loss = self._extract_loss(student_outputs)

        # Auto device/dtype matching
        teacher_logits = self._match_device_dtype(
            teacher_logits,
            student_logits,
            self.args.auto_device_match,
            self.args.auto_dtype_match,
        )

        return student_logits, teacher_logits, hard_loss

    def _compute_total_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_loss: torch.Tensor | None,
        track_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the combined soft + hard loss.

        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model
            hard_loss: Pre-computed hard loss from the model, or None
            track_metrics: Whether to store metrics for logging

        Returns:
            Tuple of (total_loss, soft_loss)
        """
        soft_loss = self.soft_loss_fn(student_logits, teacher_logits)

        if self.args.alpha > 0 and hard_loss is not None:
            total = self._combine_losses(
                {"soft": soft_loss, "hard": hard_loss},
                {"soft": 1 - self.args.alpha, "hard": self.args.alpha},
                track_metrics=track_metrics,
            )
            assert total is not None
            return total, soft_loss

        if track_metrics:
            self._track_metric("loss/soft", soft_loss)

        if self.args.alpha > 0 and hard_loss is None:
            self._warn_once(
                "no_hard_loss",
                "alpha=%.2f but no hard loss available — either labels were "
                "not found in the batch, or the model's forward() did not "
                "return a loss. The effective loss is 100%% soft. Ensure "
                "your dataset includes a 'labels' column and the model "
                "computes loss when labels are provided.",
                self.args.alpha,
            )

        return soft_loss, soft_loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Override log to inject eval-specific component losses.

        During evaluation, averaged component losses (eval_loss/soft, eval_loss/hard)
        are injected from the accumulator populated by compute_loss().

        Training metrics are handled by BaseDistiller.log() via the accumulator.

        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for computing throughput metrics
        """
        prefix = self._eval_metric_prefix

        if prefix and f"{prefix}_loss" in logs:
            # Eval logging — inject averaged component losses
            acc = self._eval_component_losses or {}
            if acc.get("soft"):
                logs[f"{prefix}_loss/soft"] = sum(acc["soft"]) / len(acc["soft"])
            if acc.get("hard"):
                logs[f"{prefix}_loss/hard"] = sum(acc["hard"]) / len(acc["hard"])

        super().log(logs, start_time=start_time)

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Evaluate the student model, reporting component loss breakdown
        (loss/soft, loss/hard) alongside the aggregate eval_loss.

        Optionally runs WeightWatcher analysis.

        Args:
            eval_dataset: Dataset to evaluate on (defaults to self.eval_dataset)
            ignore_keys: Output keys to ignore
            metric_key_prefix: Prefix for metric names

        Returns:
            Dictionary of evaluation metrics
        """
        # The teacher is placed lazily so that evaluate() works before train().
        self._setup_teacher_placement()

        # Set up eval component loss accumulator (populated by compute_loss)
        self._eval_component_losses = {"soft": [], "hard": []}
        self._eval_metric_prefix = metric_key_prefix
        # Clear stale training metrics so they don't leak into eval log entries
        self.current_step_metrics = {}

        # Call Trainer.evaluate directly (skip BaseDistiller.evaluate which does
        # per-layer alignment metrics we don't need).
        # log() injects component metrics into the log_history entry.
        metrics = Trainer.evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)

        # Also add component metrics to the returned dict
        acc = self._eval_component_losses or {}
        if acc.get("soft"):
            metrics[f"{metric_key_prefix}_loss/soft"] = sum(acc["soft"]) / len(acc["soft"])
        if acc.get("hard"):
            metrics[f"{metric_key_prefix}_loss/hard"] = sum(acc["hard"]) / len(acc["hard"])

        # End-to-end evaluation loss (runs full student model if configured)
        e2e_metrics = self._compute_e2e_eval_loss(eval_dataset, metric_key_prefix)
        if e2e_metrics:
            self.log(e2e_metrics)
            metrics.update(e2e_metrics)

        # Clean up eval state so training steps stop accumulating.
        self._eval_metric_prefix = None
        self._eval_component_losses = None

        # WeightWatcher (shared implementation in BaseDistiller)
        if getattr(self.args, "use_weightwatcher", False) and WEIGHTWATCHER_AVAILABLE:
            ww_metrics = self._run_weightwatcher_analysis(metric_key_prefix)
            if ww_metrics:
                self.log(ww_metrics)
                metrics.update(ww_metrics)

        return metrics
