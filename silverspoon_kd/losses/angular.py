"""Angular + magnitude decomposed loss for knowledge distillation."""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def angular_magnitude_loss(
    *, alpha: float = 1.0, beta: float = 1.0, dim: int = -1
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Decomposed angular + magnitude loss.

    Separately matches direction (cosine) and scale (norm difference),
    giving explicit control over both components that plain MSE conflates.

    Args:
        alpha: Weight for the angular (cosine) term.  Default: ``1.0``.
        beta: Weight for the magnitude term.  Default: ``1.0``.
        dim: Dimension along which cosine similarity is computed.  Default: ``-1``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        angular = 1 - F.cosine_similarity(student, teacher, dim=dim).mean()
        s_norm = student.norm(dim=dim)
        t_norm = teacher.norm(dim=dim)
        magnitude = F.mse_loss(s_norm, t_norm)
        return alpha * angular + beta * magnitude

    return loss_fn
