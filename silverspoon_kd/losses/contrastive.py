"""
Contrastive loss functions for knowledge distillation.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveDistillationLoss(nn.Module):
    """
    InfoNCE-based contrastive loss for knowledge distillation.

    Trains the student to produce representations that are similar to the
    corresponding teacher representations (positive pairs) while being
    dissimilar to other samples in the batch (negative pairs).

    References:
        - CRD: Contrastive Representation Distillation (ICLR 2020)
        - CoDIR: Contrastive Distillation on Intermediate Representations (EMNLP 2020)

    Note:
        For sequence inputs (3D tensors), this implementation flattens all tokens
        into independent samples. If you need sequence-level representations,
        apply pooling (e.g., mean or CLS token) before passing to this loss.

    Args:
        temperature: Temperature for softmax scaling. Lower values make the
            distribution sharper. Default: 0.07.
        student_dim: Dimensionality of student features. If None, inferred on
            first forward pass.
        teacher_dim: Dimensionality of teacher features. If None, inferred on
            first forward pass.

    Example:
        ```python
        # Simplest usage (dimensions inferred automatically):
        loss_fn = ContrastiveDistillationLoss()

        # With explicit dimensions (creates projector upfront if they differ):
        loss_fn = ContrastiveDistillationLoss(student_dim=256, teacher_dim=512)
        ```
    """

    def __init__(
        self,
        temperature: float = 0.07,
        student_dim: int | None = None,
        teacher_dim: int | None = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.projector: nn.Linear | None = None

        # If dimensions provided upfront and differ, create projector now
        if student_dim is not None and teacher_dim is not None:
            if student_dim != teacher_dim:
                self.projector = nn.Linear(student_dim, teacher_dim)
            self._initialized = True
        else:
            self._initialized = False

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive distillation loss.

        Args:
            student_features: Student representations of shape [B, D_s] or [B, L, D_s].
            teacher_features: Teacher representations of shape [B, D_t] or [B, L, D_t].

        Returns:
            Scalar loss tensor.
        """
        # Flatten sequence dimension if present: [B, L, D] -> [B*L, D]
        if student_features.dim() > 2:
            student_features = student_features.view(-1, student_features.size(-1))
            teacher_features = teacher_features.view(-1, teacher_features.size(-1))

        # Lazily initialize projector on first forward if dimensions differ
        if not self._initialized:
            student_dim = student_features.size(-1)
            teacher_dim = teacher_features.size(-1)
            if student_dim != teacher_dim:
                self.projector = nn.Linear(student_dim, teacher_dim, device=student_features.device)
            self._initialized = True

        # Project student to teacher space if needed
        if self.projector is not None:
            student_features = self.projector(student_features)

        # L2 normalize (critical for contrastive loss)
        student_norm = F.normalize(student_features, dim=1)
        teacher_norm = F.normalize(teacher_features, dim=1)

        # Compute similarity matrix: [N, N] where N = batch size (or B*L)
        # Entry (i, j) is the cosine similarity between student_i and teacher_j
        logits = torch.matmul(student_norm, teacher_norm.T) / self.temperature

        # Target: student_i should match teacher_i (diagonal entries)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Cross-entropy encourages high similarity on diagonal, low off-diagonal
        loss = F.cross_entropy(logits, labels)

        return loss


def contrastive_loss(
    *,
    temperature: float = 0.07,
    student_dim: int | None = None,
    teacher_dim: int | None = None,
) -> ContrastiveDistillationLoss:
    """Contrastive distillation loss (InfoNCE-based).

    Args:
        temperature: Softmax temperature.  Default: ``0.07``.
        student_dim: Student feature dimension.  If ``None``, inferred on
            first forward pass.
        teacher_dim: Teacher feature dimension.  If ``None``, inferred on
            first forward pass.

    Returns:
        A ``ContrastiveDistillationLoss`` instance.
    """
    return ContrastiveDistillationLoss(
        temperature=temperature, student_dim=student_dim, teacher_dim=teacher_dim
    )
