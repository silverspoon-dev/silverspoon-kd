"""
Blockwise distiller for layer-by-layer knowledge distillation.

This module provides the [BlockwiseDistiller][silverspoon_kd.BlockwiseDistiller]
class which extends [BaseDistiller][silverspoon_kd.distillers.BaseDistiller]
to perform blockwise knowledge distillation by aligning individual blocks between
teacher and student models.
"""

import contextlib
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from accelerate.utils import send_to_device
from torch import nn
from transformers import PreTrainedModel

from ..alignments.alignment import Alignment, block_module_name
from ..engines.module_capture_engine import ModuleCaptureEngine
from ..training_arguments import TrainingArguments
from .base_distiller import BaseDistiller, _warn_irrelevant_args, send_to_dtype

logger = logging.getLogger(__name__)


def _activation_offload_context(offload: bool, block: nn.Module):
    """Context that keeps a checkpointed block's saved activations in host memory.

    A no-op when ``offload`` is False or the block lives on the CPU (its
    activations are already on the host).  Pinned memory is used for CUDA,
    where it makes the device-to-host copies asynchronous.
    """
    if not offload:
        return contextlib.nullcontext()
    try:
        device = next(block.parameters()).device
    except StopIteration:
        return contextlib.nullcontext()
    if device.type == "cpu":
        return contextlib.nullcontext()
    return torch.autograd.graph.save_on_cpu(
        pin_memory=device.type == "cuda", device_type=device.type
    )


class StudentBlocksContainer(nn.Module):
    """Holds student blocks as named submodules for Trainer save/load/zero_grad.

    Registers student blocks as submodules so that model.train()/eval()/
    zero_grad()/state_dict() work correctly through the standard Trainer
    infrastructure.  Projectors are saved/loaded separately via
    ``projector_state.pt`` in
    [BaseDistiller][silverspoon_kd.distillers.BaseDistiller].

    Note:
        ``forward`` is installed as an *instance* attribute in
        :meth:`__init__` (with a teacher-derived signature), not as a
        class-level method.  HF Trainer inspects
        ``model.forward.__signature__`` to decide which batch columns to
        pass through, so the per-instance pattern is load-bearing.
    """

    def __init__(
        self,
        teacher_model: PreTrainedModel | nn.Module,
        alignments: list[Alignment],
    ):
        super().__init__()

        for alignment in alignments:
            self.register_module(block_module_name(alignment.get_name()), alignment.student_block)

        # Alignment names in block order, used by ``gradient_checkpointing_enable``
        # to honour ``every_n_layers`` by name rather than by module identity
        # (blocks may later be FSDP-wrapped in place).
        self._alignment_names: list[str] = [a.get_name() for a in alignments]
        self._gradient_checkpointing = False
        self._gradient_checkpointing_kwargs: dict = {}
        self._gradient_checkpointing_blocks: frozenset[str] = frozenset()
        self._gradient_checkpointing_offload = False

        # Copy teacher forward signature so Trainer column filtering works.
        if hasattr(teacher_model, "forward"):
            teacher_sig = inspect.signature(teacher_model.forward)

            def fwd(*args, **kwargs):
                return None

            # HF Trainer inspects ``forward.__signature__`` to filter
            # batch columns; setattr bypasses the static type check
            # on FunctionType.
            setattr(fwd, "__signature__", teacher_sig)
            self.forward = fwd
        else:
            self.forward = lambda **kwargs: None

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict | None = None,
        every_n_layers: int = 1,
        offload: bool = False,
    ) -> None:
        """Enable gradient checkpointing for the student blocks.

        Mirrors ``PreTrainedModel.gradient_checkpointing_enable`` so the HF
        Trainer can call it with either convention: transformers < 5.16 passes
        everything inside ``gradient_checkpointing_kwargs`` (so
        ``every_n_layers`` and ``offload`` are accepted there too), newer
        versions pass them as their own keyword arguments.

        Args:
            gradient_checkpointing_kwargs: Extra keyword arguments for
                :func:`torch.utils.checkpoint.checkpoint` (e.g. ``use_reentrant``).
            every_n_layers: Checkpoint only every ``every_n_layers``-th block, in
                alignment order. ``1`` (the default) checkpoints every block.
            offload: Keep the activations saved for recomputation in pinned
                host memory instead of on the accelerator
                (``torch.autograd.graph.save_on_cpu``).
        """
        kwargs = dict(gradient_checkpointing_kwargs or {})
        every_n_layers = int(kwargs.pop("every_n_layers", every_n_layers))
        offload = bool(kwargs.pop("offload", offload))
        if every_n_layers < 1:
            raise ValueError(f"every_n_layers must be >= 1, got {every_n_layers}")
        self._gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = kwargs
        self._gradient_checkpointing_offload = offload
        self._gradient_checkpointing_blocks = frozenset(
            name for i, name in enumerate(self._alignment_names) if i % every_n_layers == 0
        )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing for the student blocks."""
        self._gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {}
        self._gradient_checkpointing_blocks = frozenset()
        self._gradient_checkpointing_offload = False

    @property
    def is_gradient_checkpointing(self):
        """Return True if gradient checkpointing is currently enabled."""
        return getattr(self, "_gradient_checkpointing", False)


class BlockwiseDistiller(BaseDistiller):
    """Block-Wise Knowledge Distillation (BKD), a type of Feature-Based
    Knowledge Distillation (FBKD).

    A trainer that performs block-wise (layer-by-layer) knowledge distillation.
    The student is divided into blocks trained in isolation. Each student block
    receives input directly from the corresponding teacher block, treating each
    block as an independent regression problem.

    Student blocks and their projectors are held inside a
    ``StudentBlocksContainer`` so that the standard Trainer save/load/zero_grad
    infrastructure works.
    """

    args: TrainingArguments  # pyright: ignore[reportIncompatibleVariableOverride]
    _USE_COMPOSITE_OPTIMIZER: bool = True

    def __init__(
        self,
        teacher_model: PreTrainedModel | nn.Module,
        alignments: list[Alignment],
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        args: TrainingArguments | None = None,
        auto_truncate: bool = False,
        prepare_teacher_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        student_models: dict[str, nn.Module] | None = None,
        **kwargs: Any,
    ):
        """Initialize the BlockwiseDistiller.

        Args:
            teacher_model: The teacher model to distill from
            alignments: List of Alignment instances
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            args: TrainingArguments with distillation-specific parameters
            auto_truncate: If True, automatically stop the teacher forward
                pass as soon as all aligned modules have been captured. For
                example, if a 24-layer teacher is aligned at layers 0-5,
                the remaining 18 layers are skipped — saving compute and
                memory. Defaults to False because exception-based truncation
                is incompatible with FSDP and torch.compile. Safe to enable
                for single-GPU or DDP setups without torch.compile.
            prepare_teacher_inputs: Optional callable to transform teacher inputs
            student_models: Optional dict mapping names to complete student models
                for checkpoint saving
            **kwargs: Additional keyword arguments passed to Trainer
        """
        container = StudentBlocksContainer(teacher_model, alignments)

        if args is None:
            args = TrainingArguments()

        _warn_irrelevant_args(
            args,
            used_params={"backward_per_block", "deepcopy_captured_args_and_kwargs"},
            distiller_name="BlockwiseDistiller",
        )

        # Disable torch_compile so Trainer doesn't compile the container.
        # BaseDistiller.__init__ can still compile the teacher.
        self._torch_compile_teacher = args.torch_compile
        args.torch_compile = False

        super().__init__(
            teacher_model=teacher_model,
            alignments=alignments,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            model=container,
            prepare_teacher_inputs=prepare_teacher_inputs,
            **kwargs,
        )

        self.student_models = student_models

        # Warn about unsupported per-alignment loss weights
        non_default = [a for a in alignments if getattr(a, "loss_weight", 1.0) != 1.0]
        if non_default:
            logger.warning(
                "BlockwiseDistiller does not support per-alignment loss_weight. "
                "Alignments %s have non-default loss_weight which will be "
                "ignored.",
                [a.get_name() for a in non_default],
            )

        # FSDP expects every wrapped module to participate in every
        # forward-backward cycle. Disable auto_truncate if the teacher
        # is FSDP-wrapped (e.g., teacher_placement="sharded").
        effective_auto_truncate = self._resolve_auto_truncate(
            auto_truncate,
            self._is_model_fsdp_wrapped(teacher_model),
            "teacher",
            needs_backward=False,  # teacher runs under no_grad
        )

        # Create and register the capture engine (teacher only)
        teacher_modules = [a.teacher_block for a in self.alignments]
        self.capture_engine = ModuleCaptureEngine(
            model=teacher_model,
            modules_to_capture=teacher_modules,
            deepcopy_captured_args_and_kwargs=args.deepcopy_captured_args_and_kwargs,
            auto_truncate=effective_auto_truncate,
        )
        self._capture_engines = [self.capture_engine]

        # Configure FSDP to individually wrap each student block so that
        # calling block(*args) triggers proper FSDP unshard → forward → reshard.
        if self.is_fsdp_enabled:
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            fsdp_plugin = self.accelerator.state.fsdp_plugin
            assert fsdp_plugin is not None
            block_cls_names = list({type(a.student_block).__name__ for a in self.alignments})
            fsdp_plugin.transformer_cls_names_to_wrap = block_cls_names
            fsdp_plugin.auto_wrap_policy = transformer_auto_wrap_policy

    # -------------------------------------------------------------------------
    # Core distillation logic
    # -------------------------------------------------------------------------

    def compute_distillation_loss(self, model, inputs, is_training):
        """Run teacher forward, iterate captured blocks, compute student losses."""
        teacher_inputs = self._prepare_teacher_inputs(inputs)

        # Teacher forward — optionally on a separate CUDA stream.
        # With overlap, per-block CUDA events let us start each student block
        # as soon as its corresponding teacher block finishes (pipelining).
        events = self._setup_capture_events(self.capture_engine)

        self._run_teacher_forward(teacher_inputs, catch_truncation=True)

        fsdp_ctx = self._fsdp_param_context(model)

        bw_per_block = is_training and self.args.backward_per_block
        total_loss = None

        with self._grad_context(is_training), fsdp_ctx:
            for idx, alignment in enumerate(self.alignments):
                # Wait for this teacher block to finish on the teacher stream
                if events is not None:
                    torch.cuda.current_stream().wait_event(events[idx])

                input_args, input_kwargs = self.capture_engine.pop_captured_inputs(idx)
                teacher_output = alignment.teacher_output_selector(
                    self.capture_engine.pop_captured_output(idx)
                ).detach()

                loss = self._compute_student_loss(
                    alignment,
                    input_args,
                    input_kwargs,
                    teacher_output,
                    is_training,
                )

                if bw_per_block:
                    # Backward per-block: frees activations immediately,
                    # reducing peak memory from O(N blocks) to O(1 block).
                    self.accelerator.backward(loss)
                    loss = loss.detach()

                # Move loss to a common device when blocks are on different GPUs
                if total_loss is not None and loss.device != total_loss.device:
                    loss = loss.to(total_loss.device)
                total_loss = loss if total_loss is None else total_loss + loss

        if events is not None:
            self.capture_engine.capture_events = None

        if total_loss is None:
            logger.warning(
                "All alignments produced no loss (possibly all truncated). "
                "Returning zero loss for this step."
            )
            return torch.zeros((), device=self.args.device, requires_grad=True).sum()
        return total_loss

    def _backward_loss(self, loss: torch.Tensor) -> None:
        """No-op when backward_per_block is enabled (backward already happened per-block)."""
        if not self.args.backward_per_block:
            super()._backward_loss(loss)

    @staticmethod
    def _fsdp_param_context(model):
        """No-op: per-block FSDP wrapping handles unshard/reshard automatically."""
        return contextlib.nullcontext()

    def _compute_student_loss(
        self,
        alignment,
        input_args: tuple,
        input_kwargs: dict[str, Any],
        teacher_output,
        is_training,
    ):
        """Run one student block and compute its alignment loss."""
        student_layer = alignment.student_block

        # Auto device matching
        if alignment.auto_device_match:
            student_device = next(student_layer.parameters()).device
            input_args = cast(tuple, send_to_device(input_args, student_device))
            input_kwargs = cast(dict[str, Any], send_to_device(input_kwargs, student_device))
            teacher_output = send_to_device(teacher_output, student_device)

        # Auto dtype matching
        if alignment.auto_dtype_match:
            student_dtype = alignment._get_compute_dtype()
            input_args = send_to_dtype(input_args, student_dtype)
            input_kwargs = send_to_dtype(input_kwargs, student_dtype)
            teacher_output = send_to_dtype(teacher_output, student_dtype)

        # Initialize auto-projectors on first forward pass
        if alignment.auto_projector and not alignment._auto_projector_initialized:
            alignment.initialize_auto_projectors(input_args, input_kwargs, teacher_output)
            self._sync_projectors_from_rank0(alignment)

        # Remove cache-related kwargs that may be incompatible across
        # architectures (e.g. teacher KV cache sized for 8 heads vs student 4).
        # BKD trains blocks in isolation — no incremental decoding needed.
        for key in ("past_key_values", "use_cache", "cache_position"):
            input_kwargs.pop(key, None)

        # Forward pass through student layer
        if alignment.input_projector is not None:
            input_args, input_kwargs = alignment.input_projector(*input_args, **input_kwargs)
        gc_blocks = getattr(self.model, "_gradient_checkpointing_blocks", None)
        use_gc = (
            is_training
            and getattr(self.model, "_gradient_checkpointing", False)
            and (gc_blocks is None or alignment.get_name() in gc_blocks)
        )
        if use_gc:
            from torch.utils.checkpoint import checkpoint as grad_checkpoint

            gc_kwargs = getattr(self.model, "_gradient_checkpointing_kwargs", {})
            with _activation_offload_context(
                getattr(self.model, "_gradient_checkpointing_offload", False), student_layer
            ):
                student_output = grad_checkpoint(
                    student_layer,
                    *input_args,
                    use_reentrant=gc_kwargs.get("use_reentrant", False),
                    **input_kwargs,
                )
        else:
            student_output = student_layer(*input_args, **input_kwargs)
        student_output = alignment.student_output_selector(student_output)
        return self._compute_alignment_loss(alignment, student_output, teacher_output, is_training)

    # -------------------------------------------------------------------------
    # FSDP block reference sync
    # -------------------------------------------------------------------------

    def create_optimizer(self, model=None):
        """Sync FSDP-wrapped block references before building per-alignment optimizers."""
        if self.is_fsdp_enabled and self.optimizer is None:
            self._sync_fsdp_block_refs()
            self._fix_fsdp_root_hierarchy()
        return super().create_optimizer(model)

    def _sync_fsdp_block_refs(self):
        """Update alignment.student_block to point to FSDP-wrapped blocks.

        After accelerator.prepare() wraps the StudentBlocksContainer in FSDP,
        each student block becomes an individual FSDP unit.  The alignment
        references still point to the unwrapped originals, so we update them
        to the FSDP-wrapped versions that handle unshard/reshard properly.

        Also initializes the root FSDP hierarchy so that child blocks are
        marked as non-root before any per-block forward calls.  Without this,
        each block's first forward triggers _lazy_init which marks it as root,
        and later state_dict() fails with a root-hierarchy assertion.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(self.model, FSDP):
            # Trigger root's _lazy_init before any per-block forwards.
            # This sets child _is_root=False, preventing assertion errors
            # during state_dict().
            from torch.distributed.fsdp._runtime_utils import _lazy_init

            _lazy_init(self.model, self.model)

        unwrapped = self.accelerator.unwrap_model(self.model)
        for alignment in self.alignments:
            name = block_module_name(alignment.get_name())
            alignment.student_block = getattr(unwrapped, name)

    # -------------------------------------------------------------------------
    # E2E eval support
    # -------------------------------------------------------------------------

    def _get_e2e_student_models(self):
        """Return full student models for e2e eval, if provided at init."""
        return self.student_models

    # -------------------------------------------------------------------------
    # FSDP state management
    # -------------------------------------------------------------------------

    def save_model(self, output_dir=None, _internal_call=False):
        """Reset FSDP states before save.

        Per-block forward calls (bypassing root FSDP forward) leave inner
        FSDP units in ``BACKWARD_POST`` state and with stale ``_is_root``
        flags.  Reset both so ``state_dict()`` can run without assertion
        errors.
        """
        if self.is_fsdp_enabled:
            self._fix_fsdp_root_hierarchy()
            self._reset_fsdp_handle_states()
        super().save_model(output_dir, _internal_call)

    def _fix_fsdp_root_hierarchy(self):
        """Fix root/non-root flags on per-block FSDP modules.

        BKD wraps each student block with FSDP individually (for independent
        unshard/reshard during per-block forward).  This sets ``_is_root=True``
        on every block.  When the Trainer or Accelerate later calls an FSDP
        operation on the container (e.g. ``state_dict``, ``clip_grad_norm_``),
        ``_lazy_init`` expects only the outermost module to be root.

        This method clears the root flag on child FSDP modules so that
        ``_lazy_init`` can re-establish a clean hierarchy.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        root_found = False
        for module in self.model.modules():
            if isinstance(module, FSDP):
                if not root_found:
                    root_found = True  # first FSDP module is the root
                else:
                    # Child FSDP modules must not claim to be root
                    if getattr(module, "_is_root", None) is True:
                        module._is_root = None

    def _reset_fsdp_handle_states(self):
        """Reset all FSDP handle/module training states to IDLE."""
        self._reset_fsdp_states(self.model)

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint using standard Trainer format + optional student models."""
        super()._save(output_dir, state_dict)

        resolved_dir = output_dir or self.args.output_dir
        assert resolved_dir is not None
        output_path = Path(resolved_dir)

        if self.student_models is not None:
            for model_name, student_model in self.student_models.items():
                student_dir = output_path / f"student_model_{model_name.replace('/', '_')}"
                student_dir.mkdir(parents=True, exist_ok=True)

                if isinstance(student_model, PreTrainedModel):
                    student_model.save_pretrained(student_dir)
                else:
                    torch.save(
                        student_model.state_dict(),
                        student_dir / "pytorch_model.bin",
                    )
