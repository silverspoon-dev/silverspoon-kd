"""MSE-based loss functions for knowledge distillation."""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def mse_loss(*, reduction: str = "mean") -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Mean Squared Error loss.

    Args:
        reduction: Reduction applied to the per-element loss (``'mean'``,
            ``'sum'``, or ``'none'``).  Default: ``'mean'``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student, teacher, reduction=reduction)

    return loss_fn


def normalized_mse_loss(
    *, eps: float = 1e-5
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Normalized MSE loss (z-score both tensors, then MSE).

    Matches relational structure (which tokens differ from the mean) without
    being distracted by absolute activation magnitudes.  Addresses the main
    flaw of plain MSE for quantized or scaled representations.

    Args:
        eps: Numerical-stability term added to the standard deviation
            denominator.  Default: ``1e-5``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s_norm = (student - student.mean(dim=-1, keepdim=True)) / (
            student.std(dim=-1, keepdim=True) + eps
        )
        t_norm = (teacher - teacher.mean(dim=-1, keepdim=True)) / (
            teacher.std(dim=-1, keepdim=True) + eps
        )
        return F.mse_loss(s_norm, t_norm)

    return loss_fn
