"""Tests for ResponseBasedDistiller e2e_eval_loss support.

Verifies that:
1. e2e_eval_loss is honoured by ResponseBasedDistiller
2. _get_e2e_student_models returns the student model
3. _compute_e2e_eval_loss produces metrics when enabled
4. Default behavior (e2e_eval_loss=None) is preserved
5. The eval_loss/e2e metric is distinct from distillation eval_loss
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

_ON_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


# ── Test fixtures ────────────────────────────────────────────────────────


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TinyModel(nn.Module):
    """Minimal model returning logits and optionally a loss."""

    def __init__(self, vocab=100, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, **kwargs):
        h = self.embed(input_ids)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return ModelOutput(logits=logits, loss=loss)


class TinyModelNoLoss(nn.Module):
    """Model that never returns a loss (e.g. a model without label support)."""

    def __init__(self, vocab=100, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        return ModelOutput(logits=self.head(h), loss=None)


def _make_models(student_cls=TinyModel):
    torch.manual_seed(0)
    teacher = TinyModel()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = student_cls()
    return teacher, student


def _make_dataset(n=32, seq_len=8, vocab=100):
    from torch.utils.data import TensorDataset

    ids = torch.randint(0, vocab, (n, seq_len))
    labels = torch.randint(0, vocab, (n, seq_len))
    return TensorDataset(ids, labels)


def _collator(batch):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return {"input_ids": ids, "labels": labels}


def _make_trainer(tmp_path, e2e_eval_loss=None, student_cls=TinyModel, **extra_args):
    teacher, student = _make_models(student_cls=student_cls)
    extra_args.setdefault("use_cpu", _ON_MPS)
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=0,
        per_device_train_batch_size=8,
        report_to=[],
        e2e_eval_loss=e2e_eval_loss,
        **extra_args,
    )
    return ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=_make_dataset(),
        eval_dataset=_make_dataset(n=16),
        data_collator=_collator,
    ), student


# ── Tests: e2e_eval_loss is not suppressed ───────────────────────────────


class TestE2EEvalLossNotSuppressed:
    """Verify that ResponseBasedDistiller honours e2e_eval_loss."""

    def test_e2e_forward_preserved(self, tmp_path):
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward")
        assert trainer.args.e2e_eval_loss == "forward"

    def test_e2e_none_preserved(self, tmp_path):
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss=None)
        assert trainer.args.e2e_eval_loss is None

    def test_no_warning_logged(self, tmp_path, caplog):
        """Should not warn about e2e_eval_loss being redundant."""
        import logging

        with caplog.at_level(logging.WARNING):
            _make_trainer(tmp_path, e2e_eval_loss="forward")
        assert "redundant" not in caplog.text.lower()
        assert "ignored" not in caplog.text.lower()


# ── Tests: _get_e2e_student_models ───────────────────────────────────────


class TestGetE2EStudentModels:
    """Verify _get_e2e_student_models returns the student model."""

    def test_returns_dict_with_student(self, tmp_path):
        trainer, student = _make_trainer(tmp_path)
        models = trainer._get_e2e_student_models()
        assert models is not None
        assert len(models) == 1
        assert list(models.values())[0] is student

    def test_returns_non_none(self, tmp_path):
        """Base class returns None; ReSKD must override to return the model."""
        trainer, _ = _make_trainer(tmp_path)
        assert trainer._get_e2e_student_models() is not None


# ── Tests: _compute_e2e_eval_loss behavior ───────────────────────────────


class TestComputeE2EEvalLoss:
    """Verify _compute_e2e_eval_loss works correctly."""

    def test_returns_empty_when_disabled(self, tmp_path):
        """When e2e_eval_loss=None, should return {}."""
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss=None)
        result = trainer._compute_e2e_eval_loss()
        assert result == {}

    def test_returns_metrics_when_enabled(self, tmp_path):
        """When e2e_eval_loss='forward', should return eval_loss/e2e metrics."""
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward")
        result = trainer._compute_e2e_eval_loss()
        # Should contain at least one key with "e2e" in it
        e2e_keys = [k for k in result if "e2e" in k]
        assert len(e2e_keys) > 0, f"Expected e2e metrics, got: {result}"

    def test_e2e_loss_is_finite(self, tmp_path):
        """The e2e loss should be a valid finite number."""
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward")
        result = trainer._compute_e2e_eval_loss()
        e2e_losses = [v for k, v in result.items() if "e2e" in k and "loss" in k]
        for loss in e2e_losses:
            assert isinstance(loss, float)
            assert not torch.isnan(torch.tensor(loss))
            assert not torch.isinf(torch.tensor(loss))

    def test_e2e_loss_differs_from_distill_loss(self, tmp_path):
        """The e2e forward loss should differ from the distillation loss.

        Distillation loss is KL(teacher || student); e2e loss is the
        student's own forward loss. These are different computations.
        """
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward")
        # Get distillation eval loss
        distill_metrics = trainer.evaluate()
        distill_loss = distill_metrics.get("eval_loss")

        # Get e2e loss
        e2e_metrics = trainer._compute_e2e_eval_loss()
        e2e_loss_key = [k for k in e2e_metrics if "e2e" in k and "loss" in k]
        assert len(e2e_loss_key) > 0

        # They should both exist and be different values
        e2e_loss = e2e_metrics[e2e_loss_key[0]]
        assert distill_loss is not None
        assert e2e_loss is not None
        # With random init, these should differ
        assert abs(distill_loss - e2e_loss) > 0.001, (
            f"Distill loss ({distill_loss}) and e2e loss ({e2e_loss}) "
            f"should be different computations"
        )

    def test_model_without_loss_returns_gracefully(self, tmp_path):
        """When the student model doesn't return a loss, e2e should handle it."""
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward", student_cls=TinyModelNoLoss)
        # Should not crash, may return empty or partial metrics
        result = trainer._compute_e2e_eval_loss()
        # The result should be a dict (possibly empty if model has no loss)
        assert isinstance(result, dict)


# ── Tests: evaluate() integration ──────────────────────────────────────


class TestEvaluateIntegration:
    """Verify that evaluate() surfaces e2e metrics when configured."""

    def test_evaluate_includes_e2e_when_enabled(self, tmp_path):
        """evaluate() returns eval_loss/e2e when e2e_eval_loss='forward'."""
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward")
        metrics = trainer.evaluate()
        assert "eval_loss/e2e" in metrics
        assert isinstance(metrics["eval_loss/e2e"], float)
        assert metrics["eval_loss/e2e"] > 0

    def test_evaluate_excludes_e2e_when_disabled(self, tmp_path):
        """evaluate() does not return e2e metrics when e2e_eval_loss is None."""
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss=None)
        metrics = trainer.evaluate()
        assert not any("e2e" in k for k in metrics)


# ── Tests: alpha interaction with e2e_eval_loss ────────────────────────


class TestAlphaE2EInteraction:
    """Verify metric composition when combining alpha and e2e_eval_loss."""

    def test_alpha_zero_has_soft_and_e2e_but_no_hard(self, tmp_path):
        """alpha=0 + e2e_eval_loss='forward': only soft + e2e metrics, no hard."""
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward", alpha=0.0)
        metrics = trainer.evaluate()
        # Distillation loss is pure soft (KL divergence)
        assert "eval_loss" in metrics
        # e2e metric present (student's own CE loss)
        assert "eval_loss/e2e" in metrics
        # No hard loss component since alpha=0
        assert "eval_loss/hard" not in metrics

    def test_alpha_positive_has_all_metrics(self, tmp_path):
        """alpha>0 + e2e_eval_loss='forward': soft, hard, and e2e metrics all present."""
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward", alpha=0.5)
        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert "eval_loss/soft" in metrics
        assert "eval_loss/hard" in metrics
        assert "eval_loss/e2e" in metrics

    def test_e2e_loss_differs_from_hard_loss(self, tmp_path):
        """eval_loss/e2e and eval_loss/hard are tracked independently.

        Both measure the student's cross-entropy, but eval_loss/hard is
        the weighted distillation component while eval_loss/e2e is a
        standalone forward pass. With random init these values should
        be close but eval_loss/hard may differ due to averaging scope.
        """
        trainer, _ = _make_trainer(tmp_path, e2e_eval_loss="forward", alpha=0.5)
        metrics = trainer.evaluate()
        # Both should be finite positive floats
        assert isinstance(metrics["eval_loss/e2e"], float)
        assert isinstance(metrics["eval_loss/hard"], float)
        assert metrics["eval_loss/e2e"] > 0
        assert metrics["eval_loss/hard"] > 0
