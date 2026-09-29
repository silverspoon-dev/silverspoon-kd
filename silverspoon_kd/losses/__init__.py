"""
Loss functions for knowledge distillation.

This module provides specialized loss functions designed for distillation tasks,
where a student model learns to match a teacher model's representations.
"""

from .angular import angular_magnitude_loss
from .contrastive import ContrastiveDistillationLoss, contrastive_loss
from .cosine import cosine_loss
from .kl import jsd_loss, kl_divergence_loss, logit_lens_kl_loss
from .liger import LIGER_KERNEL_AVAILABLE, FusedLinearKLDivLoss, LigerFusedLinearJSDLoss
from .mahalanobis import mahal_cosine_loss, mahal_mse_loss
from .mse import mse_loss, normalized_mse_loss
from .registry import LOSS_REGISTRY, get_loss_function
from .relkd import relkd_angle_loss, relkd_da_loss, relkd_distance_loss
from .smooth_l1 import smooth_l1_loss

__all__ = [
    "LIGER_KERNEL_AVAILABLE",
    "LOSS_REGISTRY",
    "ContrastiveDistillationLoss",
    "FusedLinearKLDivLoss",
    "LigerFusedLinearJSDLoss",
    "angular_magnitude_loss",
    "contrastive_loss",
    "cosine_loss",
    "get_loss_function",
    "jsd_loss",
    "kl_divergence_loss",
    "logit_lens_kl_loss",
    "mahal_cosine_loss",
    "mahal_mse_loss",
    "mse_loss",
    "normalized_mse_loss",
    "relkd_angle_loss",
    "relkd_da_loss",
    "relkd_distance_loss",
    "smooth_l1_loss",
]
