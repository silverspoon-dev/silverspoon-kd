"""Cosine similarity loss for knowledge distillation."""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def cosine_loss(*, dim: int = -1) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Cosine similarity loss (``1 - cosine_similarity``).

    Args:
        dim: Dimension along which cosine similarity is computed.  Default: ``-1``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(student, teacher, dim=dim).mean()

    return loss_fn
