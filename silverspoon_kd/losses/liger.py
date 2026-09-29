"""Liger kernel fused loss functions for response-based knowledge distillation.

These losses fuse the linear projection and loss computation into a single
CUDA kernel call using the ``liger-kernel`` package, reducing peak memory.

Note:
    These losses have a different signature from the standard registry losses
    and are NOT added to ``LOSS_REGISTRY``.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

# Optional dependency; TYPE_CHECKING import so the base class is
# visible to pyright even when the runtime import below fails.
if TYPE_CHECKING:
    from liger_kernel.chunked_loss.fused_linear_distillation import (
        LigerFusedLinearDistillationBase,
    )

try:
    from liger_kernel.chunked_loss.fused_linear_distillation import (
        LigerFusedLinearDistillationBase,
    )

    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False


__all__ = ["FusedLinearKLDivLoss", "LigerFusedLinearJSDLoss"]


if LIGER_KERNEL_AVAILABLE:
    from liger_kernel.chunked_loss import (
        LigerFusedLinearJSDLoss,  # pyright: ignore[reportAssignmentType]
    )

    class _FusedLinearKLDivFunction(LigerFusedLinearDistillationBase):
        """Custom autograd function that fuses a linear projection with KL.

        Inherits ``LigerFusedLinearDistillationBase``'s chunked forward /
        backward machinery and supplies the KL divergence as the
        distillation loss function.
        """

        @staticmethod
        def distillation_loss_fn(  # pyright: ignore[reportIncompatibleMethodOverride]
            student_logits, teacher_logits, target=None, ignore_index=-100, **kwargs
        ):
            """KL(teacher || student) reduction used by the fused kernel."""
            del kwargs  # Unused — required for the base class signature.
            # Note: the base class already divides logits by temperature before
            # calling this function, so we operate on pre-scaled logits.
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="none")
            kl = kl.sum(dim=-1)  # sum over vocab

            # Mask out ignore_index positions
            if target is not None:
                mask = target != ignore_index
                kl = kl.masked_fill(~mask, 0.0)

            return kl.sum()

        @classmethod
        def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
            cls,
            ctx,
            student_input: torch.Tensor,
            student_weight: torch.Tensor,
            teacher_input: torch.Tensor,
            teacher_weight: torch.Tensor,
            target,
            student_bias=None,
            teacher_bias=None,
            temperature: float = 4.0,
            compiled: bool = True,
            chunk_size: int = 1024,
        ):
            """Dispatch to the chunked fused-linear-distillation kernel."""
            return super().forward(
                cls=cls,
                ctx=ctx,
                student_input=student_input,
                student_weight=student_weight,
                teacher_input=teacher_input,
                teacher_weight=teacher_weight,
                target=target,
                student_bias=student_bias,
                teacher_bias=teacher_bias,
                chunk_size=chunk_size,
                ignore_index=-100,
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
                beta=0.5,
                compute_ce_loss=False,
                temperature=temperature,
                compiled=compiled,
            )

        @staticmethod
        def backward(ctx, grad_output, *args):  # pyright: ignore[reportIncompatibleMethodOverride]
            """Pad the base-class grads with ``None`` for extra forward args."""
            grads = LigerFusedLinearDistillationBase.backward(ctx, grad_output, *args)[:6]
            return (
                *grads,
                None,  # teacher_bias
                None,  # temperature
                None,  # compiled
                None,  # chunk_size
            )

    class FusedLinearKLDivLoss(nn.Module):  # pyright: ignore[reportRedeclaration]
        """Fused linear KL divergence loss using Liger kernel infrastructure."""

        def __init__(self, temperature: float = 4.0, chunk_size: int = 1024):
            super().__init__()
            self.temperature = temperature
            self.chunk_size = chunk_size

        def forward(
            self,
            student_input,
            output_head_weight,
            teacher_input,
            teacher_output_head_weight,
            target=None,
            output_head_bias=None,
            teacher_output_head_bias=None,
        ):
            """Project hidden states through both heads and compute fused KL."""
            # Liger 0.7.0 requires target to be a tensor (it chunks it alongside
            # inputs).  When no labels are available, pass a dummy ignore-index
            # tensor so the CE path is a no-op.
            if target is None:
                seq_len = (
                    student_input.shape[0]
                    if student_input.dim() == 2
                    else student_input.shape[0] * student_input.shape[1]
                )
                target = torch.full(
                    (seq_len,),
                    -100,
                    dtype=torch.long,
                    device=student_input.device,
                )
            return _FusedLinearKLDivFunction.apply(
                student_input,
                output_head_weight,
                teacher_input,
                teacher_output_head_weight,
                target,
                output_head_bias,
                teacher_output_head_bias,
                self.temperature,
                True,  # compiled
                self.chunk_size,
            )

else:
    # Placeholder classes that raise on instantiation — preserves a
    # consistent type across the if/else branches and fails loudly when
    # liger-kernel isn't installed.
    _LIGER_INSTALL_HINT = (
        "liger-kernel is not installed. Install it with: pip install silverspoon-kd[liger]"
    )

    class FusedLinearKLDivLoss(nn.Module):  # pyright: ignore[reportRedeclaration]
        """Placeholder — the real class requires liger-kernel."""

        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ImportError(_LIGER_INSTALL_HINT)

    class LigerFusedLinearJSDLoss(nn.Module):  # pyright: ignore[reportRedeclaration]
        """Placeholder — the real class requires liger-kernel."""

        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ImportError(_LIGER_INSTALL_HINT)
