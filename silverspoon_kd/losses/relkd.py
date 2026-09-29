"""Relation-based KD (RelKD) losses.

Implements distance-wise and angle-wise distillation losses from:
    Park et al., "Relational Knowledge Distillation", CVPR 2019.

These losses transfer mutual relations of data examples rather than
individual outputs. They operate on pairwise (distance) or ternary
(angle) relations across the batch, making them naturally invariant
to the feature dimension of teacher vs student.
"""

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix [B, B] from [B, D] features."""
    return torch.cdist(x, x, p=2)


def _mean_normalize_distances(dist: torch.Tensor) -> torch.Tensor:
    """Normalize distance matrix by its mean (mu-normalization from the paper).

    When the distance matrix has no positive entries (e.g. batch_size=1,
    where the only entry is the self-distance 0), mu would be NaN.  Fall
    back to 1.0 so the result is the (all-zero) matrix itself.
    """
    positive = dist[dist > 0]
    if positive.numel() == 0:
        return dist
    mu = positive.mean()
    return dist / (mu + 1e-8)


def relkd_distance_loss() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """RelKD distance-wise distillation loss.

    Penalizes differences in pairwise Euclidean distance structure between
    teacher and student representations.

    .. warning::
        Relational losses require ``batch_size >= 2`` because they operate
        on *pairwise* relations between examples.  With ``batch_size=1``
        the pairwise distance matrix is trivially zero and the loss is
        always 0 regardless of student quality — effectively a no-op step.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """
    _warned_batch1 = False

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        nonlocal _warned_batch1
        if student.size(0) < 2 and not _warned_batch1:
            _warned_batch1 = True
            logger.warning(
                "relkd_distance_loss received batch_size=%d. Relational losses "
                "require batch_size >= 2 to compute meaningful pairwise relations; "
                "this step will return zero loss.",
                student.size(0),
            )
        # Flatten to [B, D] if higher-dimensional
        if student.dim() > 2:
            student = student.view(student.size(0), -1)
        if teacher.dim() > 2:
            teacher = teacher.view(teacher.size(0), -1)

        with torch.no_grad():
            t_dist = _mean_normalize_distances(_pairwise_distances(teacher))
        s_dist = _mean_normalize_distances(_pairwise_distances(student))

        return F.smooth_l1_loss(s_dist, t_dist)

    return loss_fn


def relkd_angle_loss() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """RelKD angle-wise distillation loss.

    Penalizes differences in the angle formed by triplets of examples
    between teacher and student representations.

    .. warning::
        Relational losses require ``batch_size >= 2`` because they operate
        on *pairwise* relations between examples.  With ``batch_size=1``
        the angle tensor is trivially zero and the loss is always 0.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """
    _warned_batch1 = False

    def _compute_angles(x: torch.Tensor) -> torch.Tensor:
        """Compute [B, B, B] angle tensor from [B, D] features.

        For efficiency, computes the [B, B] cosine similarity matrix of
        difference vectors e_ij = x_i - x_j, which encodes the angle at
        each pivot point.  We flatten the upper triangle to avoid redundancy.

        Following the paper, we compute angles via the cosine of the angle
        at each "relay" point k: angle(i, k, j) = cos(x_i - x_k, x_j - x_k).
        We use the [B, B] Gram matrix of normalized difference vectors.
        """
        # e_ij = x_i - x_j  → [B, B, D]
        diff = x.unsqueeze(0) - x.unsqueeze(1)  # [B, B, D]
        # Normalize each difference vector
        diff = F.normalize(diff, p=2, dim=2)
        # Cosine similarity between all pairs of difference vectors from same pivot
        # For each pivot k: cos(e_ik, e_jk) = dot(norm(x_i-x_k), norm(x_j-x_k))
        # This gives [B, B, B] but we compute it as batched matmul
        # diff[k] is [B, D] = normalized (x_i - x_k) for all i
        # angles[k] = diff[k] @ diff[k].T → [B, B]
        angles = torch.bmm(diff, diff.transpose(1, 2))  # [B, B, B]
        return angles

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        nonlocal _warned_batch1
        if student.size(0) < 2 and not _warned_batch1:
            _warned_batch1 = True
            logger.warning(
                "relkd_angle_loss received batch_size=%d. Relational losses "
                "require batch_size >= 2 to compute meaningful pairwise relations; "
                "this step will return zero loss.",
                student.size(0),
            )
        if student.dim() > 2:
            student = student.view(student.size(0), -1)
        if teacher.dim() > 2:
            teacher = teacher.view(teacher.size(0), -1)

        with torch.no_grad():
            t_angles = _compute_angles(teacher)
        s_angles = _compute_angles(student)

        return F.smooth_l1_loss(s_angles, t_angles)

    return loss_fn


def relkd_da_loss(
    *, dist_weight: float = 1.0, angle_weight: float = 2.0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Combined RelKD distance + angle loss (RelKD-DA).

    Default weights follow the paper: ``dist_weight=1, angle_weight=2``.

    Args:
        dist_weight: Weight for the distance term.  Default: ``1.0``.
        angle_weight: Weight for the angle term.  Default: ``2.0``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """
    _dist_fn = relkd_distance_loss()
    _angle_fn = relkd_angle_loss()

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return dist_weight * _dist_fn(student, teacher) + angle_weight * _angle_fn(student, teacher)

    return loss_fn
