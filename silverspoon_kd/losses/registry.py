"""
Loss function registry for knowledge distillation.

Provides a centralized registry of loss functions and a factory function
to create loss functions by name from configuration.
"""

from collections.abc import Callable
from typing import Any

import torch

from .angular import angular_magnitude_loss
from .contrastive import contrastive_loss
from .cosine import cosine_loss
from .kl import jsd_loss, kl_divergence_loss, logit_lens_kl_loss
from .mahalanobis import mahal_cosine_loss, mahal_mse_loss
from .mse import mse_loss, normalized_mse_loss
from .relkd import relkd_angle_loss, relkd_da_loss, relkd_distance_loss
from .smooth_l1 import smooth_l1_loss

# Registry mapping loss type names to factory functions.
# Full names are the canonical keys; abbreviations are aliases.
LOSS_REGISTRY: dict[str, Callable[..., Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = {
    # ── Primary names ──
    "mse": mse_loss,
    "normalized_mse": normalized_mse_loss,
    "cosine": cosine_loss,
    "smooth_l1": smooth_l1_loss,
    "kl_divergence": kl_divergence_loss,
    "jsd": jsd_loss,
    "logit_lens_kl": logit_lens_kl_loss,
    "contrastive": contrastive_loss,
    "angular_magnitude": angular_magnitude_loss,
    "mahalanobis_mse": mahal_mse_loss,
    "mahalanobis_cosine": mahal_cosine_loss,
    "relkd_distance": relkd_distance_loss,
    "relkd_angle": relkd_angle_loss,
    "relkd_distance_angle": relkd_da_loss,
    # ── Aliases ──
    "kl_div": kl_divergence_loss,
    "mahal_mse": mahal_mse_loss,
    "mahal_cosine": mahal_cosine_loss,
    "relkd_da": relkd_da_loss,
}


def get_loss_function(
    loss_type: str, **kwargs: Any
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Get a loss function by type name.

    Args:
        loss_type: Name of the loss function.  Canonical names:
            ``mse``, ``normalized_mse``, ``cosine``, ``smooth_l1``,
            ``kl_divergence``, ``jsd``, ``logit_lens_kl``, ``contrastive``,
            ``angular_magnitude``, ``mahalanobis_mse``, ``mahalanobis_cosine``,
            ``relkd_distance``, ``relkd_angle``, ``relkd_distance_angle``.
            Aliases ``kl_div``, ``mahal_mse``, ``mahal_cosine``, ``relkd_da``
            are also accepted.
        **kwargs: Additional arguments passed to the loss function factory

    Returns:
        A callable loss function that takes (student_output, teacher_output) tensors

    Raises:
        ValueError: If loss_type is not recognized

    Example:
        ```python
        loss_fn = get_loss_function("mse")
        loss_fn = get_loss_function("kl_divergence", temperature=3.0)
        loss_fn = get_loss_function("mahalanobis_mse", weight_matrix=W)
        ```
    """
    if loss_type not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise ValueError(f"Unknown loss type: '{loss_type}'. Available: {available}")

    return LOSS_REGISTRY[loss_type](**kwargs)
