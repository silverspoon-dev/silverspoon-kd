"""KL divergence and JSD loss functions for knowledge distillation."""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def kl_divergence_loss(
    *, temperature: float = 1.0, chunk_size: int = 0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """KL Divergence loss for logit distillation.

    Supports optional chunked computation for large vocabularies to reduce
    peak memory usage.

    Args:
        temperature: Temperature for softmax scaling.  Default: ``1.0``.
        chunk_size: Number of tokens per chunk for memory-efficient
            computation. Set to ``0`` to disable chunking.  Default: ``0``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        # Flatten all dims except last: (..., V) -> (N, V)
        original_shape = student.shape
        student_flat = student.view(-1, original_shape[-1])
        teacher_flat = teacher.view(-1, original_shape[-1])
        num_tokens = student_flat.shape[0]

        # Upcast to float32 before softmax — lower-precision dtypes round the
        # small logit differences after temperature scaling to zero, killing gradients.
        if 0 < chunk_size < num_tokens:
            total_loss = torch.zeros((), device=student_flat.device, dtype=torch.float32)
            for i in range(0, num_tokens, chunk_size):
                end_i = min(i + chunk_size, num_tokens)
                s_log = F.log_softmax(student_flat[i:end_i].float() / temperature, dim=-1)
                t_prob = F.softmax(teacher_flat[i:end_i].float() / temperature, dim=-1)
                total_loss = total_loss + F.kl_div(s_log, t_prob, reduction="sum")
            return (total_loss / num_tokens) * (temperature**2)
        s_log = F.log_softmax(student_flat.float() / temperature, dim=-1)
        t_prob = F.softmax(teacher_flat.float() / temperature, dim=-1)
        return F.kl_div(s_log, t_prob, reduction="batchmean") * (temperature**2)

    return loss_fn


def jsd_loss(
    *, temperature: float = 1.0, beta: float = 0.5, chunk_size: int = 0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Jensen-Shannon Divergence loss for logit distillation.

    JSD(P||Q) = beta * KL(P||M) + (1 - beta) * KL(Q||M)
    where M = beta * P + (1 - beta) * Q.

    Supports optional chunked computation for large vocabularies.

    Args:
        temperature: Temperature for softmax scaling.  Default: ``1.0``.
        beta: Interpolation weight strictly in ``(0, 1)``.  ``0.5`` is
            symmetric JSD; values near ``0``/``1`` bias toward one
            direction of the KL.  Default: ``0.5``.
        chunk_size: Number of tokens per chunk.  Set to ``0`` to disable.
            Default: ``0``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """
    if not (0 < beta < 1):
        raise ValueError(
            f"jsd_loss requires 0 < beta < 1, got beta={beta}. "
            f"beta=0 and beta=1 collapse JSD to a one-sided KL divergence "
            f"with a log(0) term — use kl_divergence_loss instead."
        )
    log_beta = math.log(beta)
    log_1_minus_beta = math.log(1 - beta)

    def _jsd_chunk(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute sum-reduced JSD for a single chunk."""
        s_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
        t_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)

        # Mixture M = beta * teacher + (1-beta) * student (in log-space)
        log_m = torch.logaddexp(t_log + log_beta, s_log + log_1_minus_beta)

        t_prob = t_log.exp()
        s_prob = s_log.exp()
        kl_t_m = F.kl_div(log_m, t_prob, reduction="sum")
        kl_s_m = F.kl_div(log_m, s_prob, reduction="sum")
        return beta * kl_t_m + (1 - beta) * kl_s_m

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        original_shape = student.shape
        student_flat = student.view(-1, original_shape[-1])
        teacher_flat = teacher.view(-1, original_shape[-1])
        num_tokens = student_flat.shape[0]

        if 0 < chunk_size < num_tokens:
            total_loss = torch.zeros((), device=student_flat.device, dtype=torch.float32)
            for i in range(0, num_tokens, chunk_size):
                end_i = min(i + chunk_size, num_tokens)
                total_loss = total_loss + _jsd_chunk(student_flat[i:end_i], teacher_flat[i:end_i])
            return (total_loss / num_tokens) * (temperature**2)
        return (_jsd_chunk(student_flat, teacher_flat) / num_tokens) * (temperature**2)

    return loss_fn


def logit_lens_kl_loss(
    *,
    output_head: nn.Module | None = None,
    temperature: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Logit-lens KL loss (project through output head, then compare distributions).

    Inspired by the FDD trajectory loss (Gong et al., ACL 2025).  Projects
    both student and teacher hidden states through a shared output head and
    compares the resulting token distributions via KL divergence.  This
    measures functional equivalence — whether the layer outputs *mean the
    same thing for prediction* — rather than raw numerical similarity.

    Args:
        output_head: **Required.** The output head to project through
            (e.g. an LM head for language models).
        temperature: Softmax temperature.  Default: ``1.0``.

    Returns:
        A callable ``(student, teacher) -> scalar`` loss function.
    """
    head = output_head
    if head is None:
        raise ValueError(
            "logit_lens_kl requires 'output_head' kwarg (the module to "
            "project hidden states through before comparing distributions)"
        )

    def loss_fn(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_logits = head(teacher)
        student_logits = head(student)
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)

    return loss_fn
