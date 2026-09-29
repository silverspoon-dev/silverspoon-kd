"""
Holistic distiller for end-to-end knowledge distillation.

This module provides the [HolisticDistiller][silverspoon_kd.HolisticDistiller]
class which extends [BaseDistiller][silverspoon_kd.distillers.BaseDistiller]
to perform end-to-end knowledge distillation by running complete forward passes
through both teacher and student models before computing alignment losses.
"""

import contextlib
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel

from ..alignments.alignment import Alignment
from ..engines.module_capture_engine import (
    ModuleCaptureEngine,
    _TruncatedForwardException,
)
from ..training_arguments import TrainingArguments
from .base_distiller import BaseDistiller, _AlignmentLossAccumulator, _warn_irrelevant_args

logger = logging.getLogger(__name__)


class HolisticDistiller(BaseDistiller):
    """
    Holistic Knowledge Distillation (HKD), a type of Feature-Based Knowledge
    Distillation (FBKD).

    A trainer that performs end-to-end knowledge distillation. The entire student
    network is optimized jointly, allowing error gradients to propagate through
    the whole network.

    Unlike [BlockwiseDistiller][silverspoon_kd.BlockwiseDistiller] which trains
    each block independently with immediate backpropagation,
    HolisticDistiller runs complete forward passes
    through both teacher and student models, accumulates alignment losses across
    all blocks, and performs a single backpropagation step.

    This approach:
    1. Runs teacher model forward pass with capture engine to record activations
    2. Runs student model forward pass with capture engine to record activations
    3. Computes alignment losses between corresponding teacher-student layer pairs
    4. Sums all losses and performs a single backward pass
    5. Updates all student parameters together via a single optimizer
       covering the full student model plus any alignment projectors.
    """

    # End-to-end training requires a single optimizer over the full student;
    # per-alignment scoping (as in BKD) leaves non-aligned params un-updated.
    _USE_COMPOSITE_OPTIMIZER: bool = False
    _MAIN_LOOP_COMPUTES_METRICS: bool = True

    def __init__(
        self,
        student_model: PreTrainedModel | nn.Module,
        teacher_model: PreTrainedModel | nn.Module,
        alignments: list[Alignment],
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        args: TrainingArguments | None = None,
        auto_truncate: bool = False,
        prepare_teacher_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        prepare_student_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        overlap_alignment_loss: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the HolisticDistiller.

        Args:
            student_model: The student model to train
            teacher_model: The teacher model to distill from
            alignments: List of Alignment instances
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            args: TrainingArguments with distillation-specific parameters
            auto_truncate: If True, automatically stop each forward pass
                as soon as all aligned modules have been captured. For
                example, if a 24-layer teacher is aligned at layers 0-5,
                the remaining 18 layers are skipped — saving compute and
                memory. Defaults to False because exception-based truncation
                is incompatible with FSDP and torch.compile. Safe to enable
                for single-GPU or DDP setups without torch.compile.
            prepare_teacher_inputs: Optional callable to transform inputs
                                   before passing to teacher
            prepare_student_inputs: Optional callable to transform inputs
                                   before passing to student
            overlap_alignment_loss: Opt-in optimisation, default ``False``.
                When True and CUDA is available, each alignment's loss
                kernel runs on a dedicated CUDA stream so it overlaps with
                the next student layer's forward on the default stream.
                Gradients are equal to the ``False`` path within
                floating-point tolerance (the parity is verified in tests).

                Off by default because the speed-up is small in practice
                and there are two real friction points to be aware of:

                * **AccumulateGrad stream mismatch.** PyTorch creates each
                  parameter's ``AccumulateGrad`` autograd node on whichever
                  stream first produced its gradient.  With overlap on,
                  that's the loss stream; on the next iteration backward
                  hits it from the default stream and PyTorch warns about
                  a stream mismatch.  This adds an implicit synchronisation
                  (eroding the perf gain) and **breaks CUDA-graph capture**
                  — set this to ``False`` if you use ``torch.compile`` with
                  CUDA graphs or anything else that requires a clean graph.
                * **Floating-point reordering.** Stream-level reordering of
                  loss kernels can produce non-bitwise-identical losses vs
                  the default-stream path.  Tests use ``torch.allclose``
                  with a small tolerance for this reason.

                Auto-disabled on CPU.
            **kwargs: Keyword arguments passed to Trainer
        """
        if args is None:
            args = TrainingArguments()

        _warn_irrelevant_args(
            args,
            used_params={"alpha", "deepcopy_captured_args_and_kwargs", "magnitude_aware_weighting"},
            distiller_name="HolisticDistiller",
        )

        if not alignments:
            raise ValueError(
                "HolisticDistiller requires at least one alignment. This usually means the "
                "``modules`` pattern given to create_alignments did not match any modules."
            )

        super().__init__(
            teacher_model=teacher_model,
            alignments=alignments,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            model=student_model,
            prepare_teacher_inputs=prepare_teacher_inputs,
            prepare_student_inputs=prepare_student_inputs,
            **kwargs,
        )

        # Ensure "labels" survives RemoveColumnsCollator when hard loss
        # is needed.  Without this, a student whose forward() doesn't
        # list ``labels`` explicitly would have labels stripped before
        # compute_distillation_loss ever sees them — silently disabling
        # the hard-loss path despite alpha > 0.
        if args.alpha > 0:
            self._set_signature_columns_if_needed()
            if self._signature_columns is not None and "labels" not in self._signature_columns:
                self._signature_columns.append("labels")

        if self.model is None:
            raise ValueError("Student model cannot be None")

        # Build list of teacher modules and corresponding student modules
        teacher_modules: list[nn.Module] = []
        student_modules: list[nn.Module] = []
        self.alignment_id_to_module_id: dict[int, int] = {}

        for alignment_id, alignment in enumerate(self.alignments):
            module_id = len(teacher_modules)
            self.alignment_id_to_module_id[alignment_id] = module_id
            teacher_modules.append(alignment.teacher_block)
            student_modules.append(alignment.student_block)

        # FSDP expects every wrapped module to participate in every
        # forward-backward cycle. Exception-based truncation skips modules,
        # leaving their FSDP state uninitialized for backward. Disable
        # auto_truncate on any FSDP-wrapped model.
        teacher_auto = self._resolve_auto_truncate(
            auto_truncate,
            self._is_model_fsdp_wrapped(teacher_model),
            "teacher",
            needs_backward=False,  # teacher runs under no_grad
        )
        student_auto = self._resolve_auto_truncate(
            auto_truncate,
            self.is_fsdp_enabled,
            "student",
            needs_backward=True,  # student backward triggers FSDP state machine
        )

        # Create capture engines and register with base.
        # HKD only uses captured outputs (not inputs), so capture_inputs=False
        # avoids storing N unnecessary input snapshots per forward pass.
        self.teacher_capture = ModuleCaptureEngine(
            model=teacher_model,
            modules_to_capture=teacher_modules,
            deepcopy_captured_args_and_kwargs=args.deepcopy_captured_args_and_kwargs,
            capture_inputs=False,
            auto_truncate=teacher_auto,
        )

        self.student_capture = ModuleCaptureEngine(
            model=self.student_model,
            modules_to_capture=student_modules,
            deepcopy_captured_args_and_kwargs=args.deepcopy_captured_args_and_kwargs,
            detach_outputs=False,
            capture_inputs=False,
            auto_truncate=student_auto,
            output_callback=self._on_student_capture,
        )

        self._capture_engines = [self.teacher_capture, self.student_capture]

        # Reverse map for the incremental loss callback (alignment_id == module_id
        # in the current registration order, but keep the indirection explicit
        # so changes to the registration loop above don't silently break us).
        self._module_id_to_alignment_id: dict[int, int] = {
            module_id: alignment_id
            for alignment_id, module_id in self.alignment_id_to_module_id.items()
        }

        # Incremental alignment loss is the only path: each alignment's
        # loss is computed inside the student hook as the layer fires,
        # not after the full forward.  This caps live aligned activations
        # at O(1) instead of O(N alignments).  See ``compute_distillation_loss``.
        # The accumulator is rebuilt every step.
        self._loss_accumulator: _AlignmentLossAccumulator | None = None
        self._incremental_processed: set[int] = set()
        self._incremental_is_training: bool = False

        # Optional CUDA stream for overlapping alignment-loss compute with
        # the next student layer's forward.  ``None`` means "no overlap"
        # — the only source of truth; no separate enabled-flag.
        self._loss_stream: torch.cuda.Stream | None = None
        if overlap_alignment_loss and torch.cuda.is_available():
            try:
                stream_device = torch.device(self.args.device)
            except (TypeError, ValueError):
                stream_device = None
            if stream_device is not None and stream_device.type == "cuda":
                self._loss_stream = torch.cuda.Stream(device=stream_device)

        # Warn about input projectors — HKD captures outputs from full forward
        # passes, so input projectors have no effect (only BKD uses them).
        for alignment in self.alignments:
            if alignment.input_projector is not None:
                logger.warning(
                    "Alignment '%s' has an input_projector, but HolisticDistiller "
                    "does not apply input projectors. Input projectors only take "
                    "effect with BlockwiseDistiller. The input_projector will be "
                    "ignored. Use output_projector instead, which works with all "
                    "distiller types.",
                    alignment.get_name(),
                )
                break  # one warning is enough

    def _get_e2e_student_models(self):
        """Return the full student model for e2e eval."""
        name = self.alignments[0].student_model_name
        return {name: self.student_model}

    def create_optimizer(self, model=None):
        """Single optimizer over the full student + alignment projectors.

        Delegates to HF Trainer's default, then wires each alignment's
        ``.optimizer`` at the main optimizer so that lazy auto-projectors
        created on the first forward pass are registered via
        ``Alignment._add_projector_params_to_optimizer``.
        """
        if self.optimizer is not None:
            return self.optimizer

        super().create_optimizer(model)

        for alignment in self.alignments:
            alignment.optimizer = self.optimizer
            # Register any already-materialised projectors now; lazy ones
            # get added later via _try_init_output_projector on first forward.
            alignment._add_projector_params_to_optimizer()

        return self.optimizer

    def compute_distillation_loss(self, model, inputs, is_training):
        """Run teacher and student forward passes, compute alignment losses.

        Alignment losses are computed incrementally inside the student
        forward via :meth:`_on_student_capture` — each layer's loss is
        accumulated as that layer's hook fires, then the captured tensors
        are dropped.  Peak alignment-activation memory is O(1) instead of
        O(N alignments).  Gradients are mathematically identical to the
        previous all-at-once path (same weighted sum, same autograd graph).

        When ``alpha > 0`` and labels are present, the student model's own
        loss (cross-entropy against labels) is mixed into the total:
        ``total = (1-alpha) * alignment_loss + alpha * hard_loss``.
        """
        teacher_inputs = self._prepare_teacher_inputs(inputs)
        student_inputs = self._prepare_student_inputs(inputs)

        # If alpha > 0, ensure labels reach the student so it computes
        # its own CE loss (student_output.loss).
        labels = inputs.get("labels")
        need_hard_loss = self.args.alpha > 0 and labels is not None
        if need_hard_loss and "labels" not in student_inputs:
            student_inputs["labels"] = labels

        # Reset per-step incremental state.  ``_incremental_processed``
        # guards against gradient-checkpointing recompute firing the same
        # hook twice — losses must only be accumulated on the first firing.
        self._loss_accumulator = self._make_loss_accumulator()
        self._incremental_processed = set()
        self._incremental_is_training = is_training

        # When teacher forward runs on its own CUDA stream, install
        # per-alignment events so the student hook can wait per-block
        # instead of for the whole teacher forward — preserves pipelining.
        self._setup_capture_events(self.teacher_capture)

        # Teacher forward — optionally on a separate CUDA stream for overlap
        self._run_teacher_forward(teacher_inputs, catch_truncation=True)

        # Student forward — alignment losses are computed in
        # ``_on_student_capture`` as each hook fires.
        student_output = None
        with self._grad_context(is_training):
            with contextlib.suppress(_TruncatedForwardException):
                student_output = model(**student_inputs)
        if getattr(self, "_needs_student_logits", False):
            self._last_student_output = student_output

        # If we dispatched alignment losses to a separate CUDA stream,
        # the running total was being accumulated there — make the
        # default stream wait so subsequent ops (alpha mixing, backward)
        # see a consistent loss tensor.
        if self._loss_stream is not None:
            torch.cuda.current_stream().wait_stream(self._loss_stream)

        # Final sync for any teacher stream work the per-block events
        # didn't cover (e.g., teacher forward beyond the last alignment).
        self._sync_teacher_stream()
        self.teacher_capture.capture_events = None

        assert self._loss_accumulator is not None
        alignment_loss = self._loss_accumulator.finalize()
        if alignment_loss is None:
            logger.warning(
                "All alignments produced no loss (possibly all truncated). "
                "Returning zero loss for this step."
            )
            alignment_loss = torch.zeros((), device=self.args.device, requires_grad=True).sum()

        # Mix in hard loss when alpha > 0
        if need_hard_loss and student_output is not None:
            hard_loss = self._extract_loss(student_output)
            if hard_loss is not None:
                loss = self._combine_losses(
                    {"alignment": alignment_loss, "hard": hard_loss},
                    {"alignment": 1 - self.args.alpha, "hard": self.args.alpha},
                    track_metrics=is_training,
                )
                assert loss is not None, (
                    "Both alignment and hard loss are non-finite; training "
                    "has diverged. Check learning rate / gradient clipping / dtype."
                )
                return self._anchor_fsdp_output(loss, student_output)
            self._warn_once(
                "no_hard_loss",
                "alpha=%.2f and labels present, but the student model's "
                "forward() did not return a loss. The effective loss is "
                "100%% alignment. Ensure the student computes loss when "
                "labels are provided.",
                self.args.alpha,
            )

        return self._anchor_fsdp_output(alignment_loss, student_output)

    def _on_student_capture(self, module_id: int, _input: Any, student_output: Any) -> None:
        """Hook callback: compute one alignment's loss as the student layer fires.

        Waits on the teacher's per-block event (when streams overlap), pops
        both captures, runs the shared per-alignment pipeline, and adds the
        result to ``_loss_accumulator``.  When ``_loss_stream`` is set, the
        loss kernel is dispatched there so the next student layer can launch
        immediately on the default stream.

        Idempotent w.r.t. gradient-checkpointing recompute via
        ``_incremental_processed``.
        """
        if self._loss_accumulator is None:
            # Hook fired outside ``compute_distillation_loss`` (e.g. another
            # distiller sharing the same module triggered our hook).  Nothing
            # to accumulate into.
            return
        if module_id in self._incremental_processed:
            return
        self._incremental_processed.add(module_id)

        alignment_id = self._module_id_to_alignment_id.get(module_id)
        if alignment_id is None:
            return
        alignment = self.alignments[alignment_id]

        events = self.teacher_capture.capture_events
        if events is not None:
            event = events.get(module_id)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)

        try:
            teacher_output = self.teacher_capture.captured_outputs.pop(module_id)
        except KeyError:
            # Teacher truncated before this layer (auto_truncate raised
            # after capturing fewer modules than the student).
            return
        # Drop the dict's reference to the student output — autograd still
        # holds it through the loss tensor we're about to build.
        self.student_capture.captured_outputs.pop(module_id, None)

        if self._loss_stream is not None:
            event = torch.cuda.Event()
            event.record()
            with torch.cuda.stream(self._loss_stream):
                self._loss_stream.wait_event(event)
                raw, weighted = self._process_one_alignment(
                    alignment, teacher_output, student_output, self._incremental_is_training
                )
                self._loss_accumulator.add(alignment, raw, weighted)
        else:
            raw, weighted = self._process_one_alignment(
                alignment, teacher_output, student_output, self._incremental_is_training
            )
            self._loss_accumulator.add(alignment, raw, weighted)
