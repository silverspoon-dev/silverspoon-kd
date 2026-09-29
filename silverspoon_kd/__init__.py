"""SilverSpoon-KD: Knowledge distillation library for transformer models."""

from .alignments import (
    Alignment,
    OutputSelector,
    create_alignments,
    load_student_weights_from_checkpoint,
    load_student_with_projectors_from_checkpoint,
)
from .alignments.projectors import (
    GenericConv2dProjector,
    GenericLinearProjector,
    fuse_projectors_into_module,
)
from .distillers import (
    BaseDistiller,
    BlockwiseDistiller,
    Distiller,
    HolisticDistiller,
    ResponseBasedDistiller,
    SilentProgressCallback,
)
from .distributed import TeacherPlacement, setup_split_gpu
from .engines import ModuleCaptureEngine
from .losses import ContrastiveDistillationLoss, get_loss_function
from .losses.kl import jsd_loss, kl_divergence_loss
from .training_arguments import TrainingArguments
from .utils import (
    freeze_parameters,
    partial_summarize_layer_names,
    prune_model,
    reconfig_model,
    summarize_layer_names,
)

__version__ = "0.1.0"

# ``__all__`` is the **primary public API** — what ``from silverspoon_kd
# import *`` exports and what IDE auto-completion / docs prioritise.
# Names not listed here are still importable for advanced use.
#
# Categories (sorted alphabetically to satisfy RUF022):
#   Distillers:         BlockwiseDistiller, HolisticDistiller, ResponseBasedDistiller
#   Training args:      TrainingArguments (single class for all distillers)
#   Alignments:         Alignment, create_alignments
#   Losses:             get_loss_function, jsd_loss, kl_divergence_loss
#   Model preparation:  freeze_parameters, prune_model, reconfig_model
#   Checkpoint loading: load_student_weights_from_checkpoint,
#                       load_student_with_projectors_from_checkpoint
#   Multi-GPU split:    TeacherPlacement, setup_split_gpu
__all__ = [
    "Alignment",
    "BlockwiseDistiller",
    "Distiller",
    "HolisticDistiller",
    "ResponseBasedDistiller",
    "TeacherPlacement",
    "TrainingArguments",
    "create_alignments",
    "freeze_parameters",
    "get_loss_function",
    "jsd_loss",
    "kl_divergence_loss",
    "load_student_weights_from_checkpoint",
    "load_student_with_projectors_from_checkpoint",
    "prune_model",
    "reconfig_model",
    "setup_split_gpu",
]
