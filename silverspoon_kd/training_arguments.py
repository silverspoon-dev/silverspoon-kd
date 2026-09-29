"""Training arguments for SilverSpoon distillation.

All distiller-specific parameters live on ``TrainingArguments`` with
safe defaults.  A single class works for every distiller type.
"""

import logging
from typing import Any

from transformers import TrainingArguments as HfTrainingArguments

logger = logging.getLogger(__name__)


class TrainingArguments(HfTrainingArguments):
    """Training arguments for all SilverSpoon distillers.

    Extends HuggingFace ``TrainingArguments`` with distillation-specific
    parameters.  Every parameter has a safe default, so the same class
    works for any distiller type
    ([BlockwiseDistiller][silverspoon_kd.BlockwiseDistiller],
    [HolisticDistiller][silverspoon_kd.HolisticDistiller],
    [ResponseBasedDistiller][silverspoon_kd.ResponseBasedDistiller]).
    Parameters unused by a particular distiller are simply ignored.

    Example::

        # Works for any distiller — no need to pick a subclass.
        args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=3,
            alpha=0.5,               # ResponseBased: soft/hard weighting
            backward_per_block=True,  # Blockwise: per-block backward
        )
    """

    def __init__(
        self,
        *args,
        count_flops: bool = False,
        use_weightwatcher: bool = False,
        report_to: str | list[str] | None = None,
        # Teacher placement
        teacher_placement: Any = None,
        # Forward overlap
        overlap_teacher_forward: bool | None = None,
        # Profiling (torch.profiler)
        enable_profiling: bool = False,
        profiling_output_dir: str | None = None,
        profiling_wait: int = 20,
        profiling_warmup: int = 3,
        profiling_active: int = 3,
        profiling_repeat: int = 1,
        profiling_with_stack: bool = False,
        # End-to-end evaluation
        e2e_eval_loss: str | None = None,
        # ── ResponseBased-specific ──
        alpha: float = 0.0,
        auto_device_match: bool = False,
        auto_dtype_match: bool = False,
        # ── Feature-based shared (Holistic + Blockwise) ──
        deepcopy_captured_args_and_kwargs: bool = False,
        # ── Blockwise-specific ──
        backward_per_block: bool = False,
        # ── Holistic + ResponseBased ──
        magnitude_aware_weighting: bool = False,
        # ── Feature-based shared (Holistic + Blockwise) ──
        clip_projectors: bool = True,
        **kwargs,
    ):
        """Initialize TrainingArguments with distillation parameters.

        Args:
            count_flops: Measure the FLOPs of the first training step and log
                them as ``flops/step`` and, cumulatively, ``flops/total``.
            use_weightwatcher: Whether to use WeightWatcher for model analysis.
            report_to: Where to report training metrics. Defaults to ``[]``
                (no reporting).
            teacher_placement: How the teacher model is distributed.
                - ``None`` or ``"replicated"`` (default): full copy on every rank.
                - ``"sharded"``: FSDP full_shard across all ranks.
                - ``TeacherPlacement(teacher_only_devices=[...], strategy=...)``:
                  dedicated teacher GPUs with PP/TP/sharded strategy.
                - ``dict``: auto-converted to ``TeacherPlacement(**dict)``.
            overlap_teacher_forward: Whether to overlap teacher and student
                forward passes on separate CUDA streams. Auto-detected when
                ``None`` (default).
            enable_profiling: Enable CPU/GPU/memory profiling via torch.profiler.
            profiling_output_dir: Directory for trace files. Defaults to
                ``{output_dir}/profiling``.
            profiling_wait: Steps to skip before profiling begins.
            profiling_warmup: Warmup steps (not traced).
            profiling_active: Steps to actively trace.
            profiling_repeat: Repeat count for wait/warmup/active cycle.
            profiling_with_stack: Record Python call stacks in traces.
            e2e_eval_loss: End-to-end evaluation loss mode.
                - ``None``: disabled (default).
                - ``"forward"``: use the student model's own forward loss.
            alpha: *ResponseBased / Holistic.* Weight of the student's own
                task loss (``labels`` must be present) relative to the
                distillation loss. ``alpha=0``: distillation loss only.
                ``alpha=1``: task loss only.
                ``total = (1-alpha) * distillation + alpha * task``.
                Default: ``0.0``.
            auto_device_match: *ResponseBased only.* Automatically move teacher
                logits to the student device before loss computation.
            auto_dtype_match: *ResponseBased only.* Automatically cast teacher
                logits to student dtype before loss computation.
            deepcopy_captured_args_and_kwargs: *Feature-based only.* Deepcopy
                captured inputs in the capture engine. Needed when model hooks
                modify tensors in-place. Default: ``False``.
            backward_per_block: *Blockwise only.* Run backward after each
                block instead of summing all losses first. Reduces peak
                memory from O(N blocks) to O(1 block). Default: ``False``.
            magnitude_aware_weighting: *Holistic / ResponseBased.* Normalize
                each loss component by its detached magnitude before applying
                weights, ensuring gradient contributions are proportional to
                the specified ratios regardless of raw magnitudes. Default: ``False``.
            clip_projectors: *Feature-based (Holistic + Blockwise).* Whether to
                include projector parameters in gradient clipping. When ``True``
                (default), projector gradients contribute to the global norm
                used for clipping. Set to ``False`` to clip only the student
                model parameters, excluding projector params from the norm
                calculation. This can increase effective student gradient
                magnitudes when projector gradients are large. Default: ``True``.
            **kwargs: Additional keyword arguments for HF TrainingArguments.
                When ``fsdp`` is set, ``fsdp_config["version"]`` defaults to
                ``1``: the distillers rely on FSDP1 and do not support the
                DTensor-based FSDP2 that newer Trainer versions select by
                default.
        """
        # Default to no reporting
        if report_to is None:
            report_to = []

        # Warn about mixed-precision flags that have no effect.
        _bypassed_flags = [
            f for f in ("fp16", "bf16", "fp16_full_eval", "bf16_full_eval") if kwargs.get(f, False)
        ]
        if _bypassed_flags:
            logger.warning(
                "%s set, but silverspoon-kd distillers override training_step "
                "and prediction_step, bypassing the Trainer's autocast context. "
                "Models will run in their native parameter dtype. To control "
                "precision, load models with the desired torch_dtype in "
                "from_pretrained() or cast them with .to(dtype).",
                ", ".join(f"{f}=True" for f in _bypassed_flags),
            )

        # The distillers are built on FSDP1 (FlatParameter wrapping, per-block
        # wrapping in BlockwiseDistiller, plain projector tensors sharing the
        # student's optimizer).  Recent transformers versions default the
        # Trainer to FSDP2, whose DTensor parameters are not supported here,
        # so select FSDP1 unless the caller chose a version explicitly.
        if kwargs.get("fsdp"):
            fsdp_config = kwargs.get("fsdp_config")
            if fsdp_config is None:
                kwargs["fsdp_config"] = {"version": 1}
            elif isinstance(fsdp_config, dict):
                kwargs["fsdp_config"] = {"version": 1, **fsdp_config}

        super().__init__(*args, report_to=report_to, **kwargs)

        # Teacher placement
        from .distributed.teacher_placement import normalize_teacher_placement

        self.teacher_placement = normalize_teacher_placement(teacher_placement)

        # General
        self.count_flops = count_flops
        self.use_weightwatcher = use_weightwatcher

        # Profiling
        self.enable_profiling = enable_profiling
        self.profiling_output_dir = profiling_output_dir
        self.profiling_wait = profiling_wait
        self.profiling_warmup = profiling_warmup
        self.profiling_active = profiling_active
        self.profiling_repeat = profiling_repeat
        self.profiling_with_stack = profiling_with_stack

        # Forward overlap
        self.overlap_teacher_forward = overlap_teacher_forward

        # End-to-end evaluation
        self.e2e_eval_loss = e2e_eval_loss

        # ResponseBased
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"alpha must be between 0 and 1, got {alpha}. "
                f"alpha=0 means pure soft (distillation) loss, "
                f"alpha=1 means pure hard (cross-entropy) loss."
            )
        self.alpha = alpha
        self.auto_device_match = auto_device_match
        self.auto_dtype_match = auto_dtype_match

        # Feature-based
        self.deepcopy_captured_args_and_kwargs = deepcopy_captured_args_and_kwargs

        # Blockwise
        self.backward_per_block = backward_per_block

        # Holistic / ResponseBased
        self.magnitude_aware_weighting = magnitude_aware_weighting
        self.clip_projectors = clip_projectors
