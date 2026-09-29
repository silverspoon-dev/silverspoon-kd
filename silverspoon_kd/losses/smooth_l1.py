"""Smooth L1 (Huber) loss for knowledge distillation."""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def smooth_l1_loss(
    *, beta: float = 1.0, reduction: str = "mean"
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Smooth L1 (Huber) loss.

    Args:
        beta: Transition point between L2 (``|x| < beta``) and L1 behaviour.
            Default: ``1.0``.
        reduction: Reduction applied to the per-element loss (``'mean'``,
            ``'sum'``, or ``'none'``).  Default: ``'mean'``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(student, teacher, beta=beta, reduction=reduction)

    return loss_fn
