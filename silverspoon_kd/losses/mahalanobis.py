"""Mahalanobis-metric loss functions for knowledge distillation.

Measures distance between student and teacher hidden states using a
precomputed metric matrix ``M = W^T W``, where ``W`` is typically the
model's output projection (e.g. LM head, classifier head).  This weights
hidden-state directions by their downstream importance.

Architecture-agnostic: the loss receives ``M`` (or the weight matrix it
is derived from) as a configuration parameter — it does not assume any
particular model structure.
"""

from collections.abc import Callable

import torch
from torch import nn


def _resolve_metric(
    weight_matrix: torch.Tensor | None,
    metric_matrix: torch.Tensor | None,
) -> torch.Tensor:
    """Resolve the ``(D, D)`` metric matrix from one of the input tensors.

    Accepts either:
      - ``weight_matrix``: ``(V, D)`` tensor → computes ``M = W^T W`` once.
      - ``metric_matrix``: ``(D, D)`` tensor → used directly.

    Returns:
        Detached metric matrix ``M`` of shape ``(D, D)``.

    Raises:
        ValueError: If neither is provided.
    """
    if weight_matrix is not None:
        W = weight_matrix.detach().float()
        return (W.T @ W).to(weight_matrix.dtype)
    if metric_matrix is not None:
        return metric_matrix.detach()
    raise ValueError(
        "Mahalanobis losses require either 'weight_matrix' (V, D) or 'metric_matrix' (D, D)."
    )


def mahal_mse_loss(
    *,
    weight_matrix: torch.Tensor | None = None,
    metric_matrix: torch.Tensor | None = None,
    pre_norm: nn.Module | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """MSE loss under Mahalanobis metric ``M = W^T W``.

    Computes ``(s - t)^T M (s - t)`` averaged over all positions, which is
    equivalent to MSE in logit space but at ``O(D²)`` cost instead of
    ``O(V·D)``.

    Architecture-agnostic: operates on raw hidden states.  The metric
    matrix is supplied as an argument, not derived from model internals.

    Args:
        weight_matrix: Output-projection weights ``(V, D)``.  ``M = W^T W``
            is computed once at construction.  Mutually exclusive with
            ``metric_matrix``.
        metric_matrix: Precomputed PSD metric matrix ``(D, D)``.
            Mutually exclusive with ``weight_matrix``.
        pre_norm: Optional normalization module applied to both student
            and teacher before the metric (e.g. the model's final
            RMSNorm / LayerNorm).

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """
    M = _resolve_metric(weight_matrix, metric_matrix)

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s, t = student, teacher
        if pre_norm is not None:
            s = pre_norm(s)
            with torch.no_grad():
                t = pre_norm(t)
        M_dev = M.to(device=s.device, dtype=s.dtype)
        diff = s - t
        # (diff @ M) * diff → element-wise dot then sum over D gives
        # the quadratic form per position; mean over all positions.
        return (diff @ M_dev * diff).sum(-1).mean()

    return loss_fn


def mahal_cosine_loss(
    *,
    weight_matrix: torch.Tensor | None = None,
    metric_matrix: torch.Tensor | None = None,
    pre_norm: nn.Module | None = None,
    eps: float = 1e-8,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Cosine loss under Mahalanobis metric ``M = W^T W``.

    Computes ``1 - cos_M(s, t)`` where
    ``cos_M(s, t) = s^T M t / sqrt(s^T M s · t^T M t)``,
    averaged over all positions.  Equivalent to cosine similarity in
    logit space at ``O(D²)`` instead of ``O(V·D)``.

    Architecture-agnostic: operates on raw hidden states.

    Args:
        weight_matrix: Output-projection weights ``(V, D)``.  ``M = W^T W``
            is computed once at construction.  Mutually exclusive with
            ``metric_matrix``.
        metric_matrix: Precomputed PSD metric matrix ``(D, D)``.
            Mutually exclusive with ``weight_matrix``.
        pre_norm: Optional normalization module applied to both student
            and teacher before the metric.
        eps: Numerical-stability term for the denominator.  Default: ``1e-8``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """
    M = _resolve_metric(weight_matrix, metric_matrix)

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s, t = student, teacher
        if pre_norm is not None:
            s = pre_norm(s)
            with torch.no_grad():
                t = pre_norm(t)
        M_dev = M.to(device=s.device, dtype=s.dtype)
        sM = s @ M_dev  # (*, D)
        tM = t @ M_dev
        sMt = (sM * t).sum(-1)  # (*, )
        sMs = (sM * s).sum(-1)
        tMt = (tM * t).sum(-1)
        denom = torch.clamp(sMs.sqrt() * tMt.sqrt(), min=eps)
        cos_M = sMt / denom
        return (1.0 - cos_M).mean()

    return loss_fn
