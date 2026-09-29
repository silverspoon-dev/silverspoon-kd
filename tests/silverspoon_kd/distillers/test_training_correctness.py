"""
Algorithmic correctness tests for all distiller types.

Verify fundamental training properties that, if violated, indicate bugs:
- Loss decreases over training
- Teacher model parameters stay frozen
- Student parameters actually get updated
- Non-aligned parameters are not modified by per-student optimizers
- Identical teacher/student produces near-zero loss
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
from silverspoon_kd.distillers.holistic_distiller import HolisticDistiller
from silverspoon_kd.distillers.response_based_distiller import ResponseBasedDistiller
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)
from tests.silverspoon_kd.conftest import SimpleModel

# ── Helpers ──────────────────────────────────────────────────────────────


class _DetDataset(Dataset):
    """Pre-generated deterministic dataset."""

    def __init__(self, num_samples=60, seq_len=16, seed=42):
        gen = torch.Generator().manual_seed(seed)
        self.samples = [
            {
                "input_ids": torch.randint(0, 128, (seq_len,), generator=gen),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": torch.randint(0, 128, (seq_len,), generator=gen),
            }
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


_DATASET = _DetDataset()
_NUM_LAYERS = 2


def _args(tmp_path, device, max_steps=20, args_cls=TrainingArguments, **extra):
    args = args_cls(
        output_dir=str(tmp_path),
        num_train_epochs=100,
        per_device_train_batch_size=4,
        logging_steps=1,
        save_strategy="no",
        max_steps=max_steps,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=(device.type == "cpu"),
        **extra,
    )
    # Prevent DataParallel wrapping on multi-GPU machines
    args._n_gpu = 1
    return args


def _alignments(teacher, student):
    """Create alignments (same hidden dim, no projectors needed)."""
    a = []
    for i in range(_NUM_LAYERS):
        alignment = Alignment(
            teacher_block=teacher.get_layer(i),
            student_block=student.get_layer(i),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
        )
        a.append(alignment)
    return a


def _snapshot(module):
    """Clone all parameter tensors."""
    return {n: p.data.clone() for n, p in module.named_parameters()}


def _any_changed(before, after):
    """True if any parameter differs."""
    return any(not torch.equal(before[n], after[n]) for n in before)


def _logged_losses(distiller):
    """Extract per-step training losses from Trainer's log history."""
    return [e["loss"] for e in distiller.state.log_history if "loss" in e]


def _student_block_snapshot(distiller):
    """Snapshot all student block parameters across all alignments."""
    params = {}
    for alignment in distiller.alignments:
        for n, p in alignment.student_block.named_parameters():
            params[f"{alignment.get_name()}/{n}"] = p.data.clone()
    return params


# ── Factories ────────────────────────────────────────────────────────────


def _make_blockwise(device, tmp_path, **kw):
    teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    d = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=_alignments(teacher, student),
        args=_args(tmp_path, device, args_cls=TrainingArguments, **kw),
        train_dataset=_DATASET,
    )
    return d, teacher, student


def _make_holistic(device, tmp_path, **kw):
    teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    d = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=_alignments(teacher, student),
        args=_args(tmp_path, device, args_cls=TrainingArguments, **kw),
        train_dataset=_DATASET,
    )
    return d, teacher, student


def _make_response_based(device, tmp_path, **kw):
    teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    kw.setdefault("learning_rate", 1e-3)
    d = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_args(tmp_path, device, args_cls=TrainingArguments, **kw),
        train_dataset=_DATASET,
    )
    return d, teacher, student


# ── Loss Decreases ───────────────────────────────────────────────────────


class TestLossDecreases:
    """Training loss should decrease for every distiller type."""

    STEPS = 20

    @staticmethod
    def _check(losses):
        assert len(losses) >= 4, f"Expected >= 4 logged losses, got {len(losses)}"
        q = max(1, len(losses) // 4)
        first = sum(losses[:q]) / q
        last = sum(losses[-q:]) / q
        assert last < first, (
            f"Loss did not decrease: first-quarter mean={first:.4f}, last-quarter mean={last:.4f}"
        )

    def test_blockwise(self, device, tmp_path):
        d, _, _ = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        d.train()
        self._check(_logged_losses(d))

    def test_holistic(self, device, tmp_path):
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        d.train()
        self._check(_logged_losses(d))

    def test_response_based(self, device, tmp_path):
        d, _, _ = _make_response_based(device, tmp_path, max_steps=self.STEPS)
        d.train()
        self._check(_logged_losses(d))


# ── Teacher Frozen ───────────────────────────────────────────────────────


class TestTeacherFrozen:
    """Teacher parameters must not change during training."""

    STEPS = 10

    def test_blockwise(self, device, tmp_path):
        d, teacher, _ = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        before = _snapshot(teacher)
        d.train()
        assert not _any_changed(before, _snapshot(teacher)), "Teacher params changed"

    def test_holistic(self, device, tmp_path):
        d, teacher, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        before = _snapshot(teacher)
        d.train()
        assert not _any_changed(before, _snapshot(teacher)), "Teacher params changed"

    def test_response_based(self, device, tmp_path):
        d, teacher, _ = _make_response_based(device, tmp_path, max_steps=self.STEPS)
        before = _snapshot(teacher)
        d.train()
        assert not _any_changed(before, _snapshot(teacher)), "Teacher params changed"


# ── Student Updates ──────────────────────────────────────────────────────


class TestStudentUpdates:
    """Student parameters must change during training."""

    STEPS = 10

    def test_blockwise(self, device, tmp_path):
        d, _, _ = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        before = _student_block_snapshot(d)
        d.train()
        assert _any_changed(before, _student_block_snapshot(d)), "Student blocks unchanged"

    def test_holistic(self, device, tmp_path):
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        before = _student_block_snapshot(d)
        d.train()
        assert _any_changed(before, _student_block_snapshot(d)), "Student blocks unchanged"

    def test_response_based(self, device, tmp_path):
        d, _, student = _make_response_based(device, tmp_path, max_steps=self.STEPS)
        before = _snapshot(student)
        d.train()
        assert _any_changed(before, _snapshot(student)), "Student model unchanged"


# ── No Unintended Parameter Updates ──────────────────────────────────────


class TestNoUnintendedUpdates:
    """
    Gradient-flow correctness for the whole-student optimizer.

    HolisticDistiller runs the student end-to-end and uses a single
    optimizer over the full student model (plus projectors).  That means:
    - Parameters upstream of any alignment (e.g. embeddings feeding the
      first aligned encoder layer) receive gradients and MUST update.
    - Parameters downstream of the LAST alignment that do NOT feed into
      any loss (e.g. ``lm_head`` when only per-layer encoder alignments
      exist) receive no gradients and stay frozen — not because the
      optimizer skips them, but because backward never populates their
      ``.grad``.

    An optimizer scoped to ``alignment.student_block`` alone would leave
    parameters outside those blocks un-updated even when they receive
    gradients (embedding parameters that are never stepped).  These tests
    pin the correct behaviour.
    """

    STEPS = 10

    def test_holistic_upstream_params_updated(self, device, tmp_path):
        """Embeddings (upstream of aligned layers) must update.

        With alignments on all encoder layers and embeddings feeding the
        first layer, gradients flow back through the embeddings during
        backward.  The single whole-student optimizer must step them.
        """
        d, _, student = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        emb_before = _snapshot(student.embedding)

        d.train()

        assert _any_changed(emb_before, _snapshot(student.embedding)), (
            "student.embedding was NOT updated despite receiving gradients. "
            "Parameters outside any alignment.student_block must still be "
            "stepped by the whole-student optimizer."
        )

    def test_holistic_downstream_noflow_params_frozen(self, device, tmp_path):
        """lm_head (downstream of alignments, no gradient path) stays frozen.

        With alignments only on encoder layers and no response alignment on
        lm_head, lm_head's output does not feed any loss — so it receives
        no gradients and correctly stays unchanged.  This is distinct from a
        parameter that no optimizer covers: here it's frozen because
        ``.grad`` is None, which is the *correct* outcome.
        """
        d, _, student = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        head_before = _snapshot(student.lm_head)

        d.train()

        assert not _any_changed(head_before, _snapshot(student.lm_head)), (
            "student.lm_head was modified despite having no gradient path "
            "to any loss — the main optimizer may be stepping uninitialised "
            "gradients, or a new response alignment was implicitly added."
        )

    def test_blockwise_non_aligned_params_frozen(self, device, tmp_path):
        """Embedding and lm_head must stay frozen in BlockwiseDistiller.

        BlockwiseDistiller uses StudentBlocksContainer as the model, so
        the Trainer should not be updating embedding or lm_head.
        """
        d, _, student = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        emb_before = _snapshot(student.embedding)
        head_before = _snapshot(student.lm_head)

        d.train()

        assert not _any_changed(emb_before, _snapshot(student.embedding)), (
            "student.embedding was modified - BlockwiseDistiller should not "
            "update non-aligned parameters"
        )
        assert not _any_changed(head_before, _snapshot(student.lm_head)), (
            "student.lm_head was modified - BlockwiseDistiller should not "
            "update non-aligned parameters"
        )

    def test_response_based_all_params_update(self, device, tmp_path):
        """In ResponseBasedDistiller all student params should update (standard Trainer)."""
        d, _, student = _make_response_based(device, tmp_path, max_steps=self.STEPS)
        emb_before = _snapshot(student.embedding)
        head_before = _snapshot(student.lm_head)

        d.train()

        assert _any_changed(emb_before, _snapshot(student.embedding)), (
            "student.embedding should change in ResponseBasedDistiller"
        )
        assert _any_changed(head_before, _snapshot(student.lm_head)), (
            "student.lm_head should change in ResponseBasedDistiller"
        )


# ── Identical Models -> Zero Loss ────────────────────────────────────────


class TestIdenticalModelsZeroLoss:
    """
    When student has the same weights as teacher, the alignment loss should
    be near zero.  Verifies the loss computation pipeline is correct.
    """

    def test_blockwise(self, device, tmp_path):
        d, teacher, student = _make_blockwise(device, tmp_path, max_steps=1)
        # Copy teacher block weights into student blocks
        for i in range(_NUM_LAYERS):
            student.get_layer(i).load_state_dict(teacher.get_layer(i).state_dict())
        d.train()
        losses = _logged_losses(d)
        assert losses[0] < 1e-6, f"Expected ~0 loss with identical blocks, got {losses[0]}"

    def test_holistic(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        # Copy ALL weights so full forward passes produce identical outputs
        student.load_state_dict(teacher.state_dict())
        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=_alignments(teacher, student),
            args=_args(tmp_path, device, max_steps=1, args_cls=TrainingArguments),
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        assert losses[0] < 1e-6, f"Expected ~0 loss with identical models, got {losses[0]}"

    def test_response_based(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student.load_state_dict(teacher.state_dict())
        d = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=_args(tmp_path, device, max_steps=1, args_cls=TrainingArguments),
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        assert losses[0] < 1e-6, f"Expected ~0 loss with identical models, got {losses[0]}"


# ── Helpers for projector tests ──────────────────────────────────────────


def _alignments_with_projectors(teacher, student):
    """Create alignments between different-dim models (projectors required).

    For BlockwiseDistiller, blocks after the first receive teacher outputs
    (teacher.hidden_dim) as input, so they need input projectors to map
    from teacher_dim → student_dim.  All blocks need output projectors to
    map student_dim → teacher_dim for the loss computation.
    """
    from silverspoon_kd.alignments.projectors import GenericLinearProjector

    a = []
    for i in range(_NUM_LAYERS):
        dev = next(student.get_layer(i).parameters()).device
        output_projector = nn.Linear(student.hidden_dim, teacher.hidden_dim).to(dev)

        # Blocks after the first need input projectors because in BlockwiseDistiller
        # they receive teacher outputs (teacher.hidden_dim) but expect student input dim
        input_projector = None
        if i > 0 and student.hidden_dim != teacher.hidden_dim:
            input_projector = GenericLinearProjector(
                in_features=teacher.hidden_dim,
                out_features=student.hidden_dim,
                mode="input",
                apply_to_arg=0,
            ).to(dev)

        alignment = Alignment(
            teacher_block=teacher.get_layer(i),
            student_block=student.get_layer(i),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            input_projector=input_projector,
            output_projector=output_projector,
        )
        a.append(alignment)
    return a


def _projector_snapshot(distiller):
    """Snapshot all projector parameters across all alignments."""
    params = {}
    for alignment in distiller.alignments:
        if alignment.output_projector is not None:
            for n, p in alignment.output_projector.named_parameters():
                params[f"{alignment.get_name()}/output_proj.{n}"] = p.data.clone()
        if alignment.input_projector is not None:
            for n, p in alignment.input_projector.named_parameters():
                params[f"{alignment.get_name()}/input_proj.{n}"] = p.data.clone()
    return params


# ── Gradient Clipping Includes Projectors ────────────────────────────────


class TestGradientClippingIncludesProjectors:
    """
    Verify gradient clipping covers all optimized parameters, not just
    student blocks.  If clipping only saw the student blocks, projector
    gradients would go unclipped.
    """

    def test_blockwise(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        aligns = _alignments_with_projectors(teacher, student)
        max_norm = 1.0
        for alignment in aligns:
            alignment.max_grad_norm = max_norm
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=_args(tmp_path, device, max_steps=1, args_cls=TrainingArguments),
            train_dataset=_DATASET,
        )

        # Record gradient norms right after clipping (before zero_grad clears
        # them). StudentBlocksContainer registers student blocks as submodules,
        # so model.zero_grad() now propagates to student parameters.
        recorded_norms = []
        _original_clip = d._clip_student_gradients

        def _recording_clip():
            _original_clip()
            for alignment in d.alignments:
                all_params = [p for g in alignment.optimizer.param_groups for p in g["params"]]
                grads = [p.grad for p in all_params if p.grad is not None]
                if grads:
                    total_norm = torch.stack([g.data.norm(2) for g in grads]).norm(2)
                    recorded_norms.append(total_norm.item())

        d._clip_student_gradients = _recording_clip
        d.train()

        assert recorded_norms, "Expected gradient norms to be recorded during training"
        for norm in recorded_norms:
            assert norm <= max_norm + 1e-5, (
                f"Gradient norm {norm:.4f} exceeds max_grad_norm {max_norm} "
                "- projector gradients may not be included in clipping"
            )

    def test_holistic(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        aligns = _alignments_with_projectors(teacher, student)
        max_norm = 1.0
        for alignment in aligns:
            alignment.max_grad_norm = max_norm
        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=aligns,
            args=_args(tmp_path, device, max_steps=1, args_cls=TrainingArguments),
            train_dataset=_DATASET,
        )
        d.train()

        # HolisticDistiller calls model.zero_grad() at the end, which clears
        # gradients on the student MODEL but not on standalone projectors.
        # So projector gradients survive and we can check they are finite.
        for alignment in d.alignments:
            if alignment.output_projector is None:
                continue
            proj_grads = [
                p.grad for p in alignment.output_projector.parameters() if p.grad is not None
            ]
            if proj_grads:
                for g in proj_grads:
                    assert g.isfinite().all(), "Non-finite projector gradient"


# ── Gradient Accumulation ────────────────────────────────────────────────


class TestGradientAccumulation:
    """
    All distillers should support gradient_accumulation_steps > 1 via the
    CompositeOptimizer (feature-based) or standard Trainer (response-based).
    """

    def test_response_based_allows_accumulation(self, device, tmp_path):
        """ResponseBasedDistiller delegates to Trainer and SHOULD support accumulation."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        # Must NOT raise
        d = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=_args(
                tmp_path,
                device,
                gradient_accumulation_steps=2,
                args_cls=TrainingArguments,
            ),
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        assert len(losses) > 0, "Training with gradient accumulation should produce losses"

    def test_holistic_allows_accumulation(self, device, tmp_path):
        """HolisticDistiller supports accumulation via CompositeOptimizer."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=_alignments(teacher, student),
            args=_args(
                tmp_path,
                device,
                gradient_accumulation_steps=2,
                args_cls=TrainingArguments,
            ),
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        assert len(losses) > 0, "Training with gradient accumulation should produce losses"


# ── Teacher Eval Mode During Evaluation ──────────────────────────────────


class TestTeacherEvalDuringEvaluation:
    """
    Teacher model must be in eval mode when compute_loss runs during
    evaluation.  BaseDistiller.__init__ freezes the teacher (eval mode +
    requires_grad=False) once, so no per-step correction is needed.
    """

    def _check_teacher_eval_during_forward(self, distiller, teacher, eval_dataset):
        """Evaluate and verify teacher is in eval mode during forward pass."""
        modes_during_forward = []

        def record_mode(module, input, output):
            modes_during_forward.append(module.training)

        # Teacher should already be in eval mode from __init__ freeze
        assert not teacher.training, "Teacher should be in eval mode after distiller init"

        hook = teacher.register_forward_hook(record_mode)
        try:
            distiller.evaluate(eval_dataset=eval_dataset)
        finally:
            hook.remove()

        assert modes_during_forward, "Teacher forward hook never fired during evaluation"
        assert all(not m for m in modes_during_forward), (
            "Teacher was in training mode during evaluation"
        )

    def test_blockwise(self, device, tmp_path):
        d, teacher, _ = _make_blockwise(device, tmp_path, max_steps=1)
        d.train()
        self._check_teacher_eval_during_forward(d, teacher, _DATASET)

    def test_holistic(self, device, tmp_path):
        d, teacher, _ = _make_holistic(device, tmp_path, max_steps=1)
        d.train()
        # HolisticDistiller.compute_loss needs capture hooks to be active.
        # After train(), hooks are deregistered in BaseDistiller's finally block.
        # Re-register them so evaluate() can capture teacher/student outputs.
        d._register_capture()
        try:
            self._check_teacher_eval_during_forward(d, teacher, _DATASET)
        finally:
            d._deregister_capture()

    def test_response_based(self, device, tmp_path):
        d, teacher, _ = _make_response_based(device, tmp_path, max_steps=1)
        d.train()
        self._check_teacher_eval_during_forward(d, teacher, _DATASET)


# ── Projector Parameters Update ──────────────────────────────────────────


class TestProjectorParametersUpdate:
    """Projector parameters must change during training when present."""

    STEPS = 10

    def test_blockwise(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        aligns = _alignments_with_projectors(teacher, student)
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=_args(
                tmp_path,
                device,
                max_steps=self.STEPS,
                args_cls=TrainingArguments,
            ),
            train_dataset=_DATASET,
        )
        before = _projector_snapshot(d)
        assert before, "Test setup error: no projectors found"
        d.train()
        assert _any_changed(before, _projector_snapshot(d)), (
            "Projector parameters did not change during training"
        )

    def test_holistic(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        aligns = _alignments_with_projectors(teacher, student)
        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=aligns,
            args=_args(
                tmp_path,
                device,
                max_steps=self.STEPS,
                args_cls=TrainingArguments,
            ),
            train_dataset=_DATASET,
        )
        before = _projector_snapshot(d)
        assert before, "Test setup error: no projectors found"
        d.train()
        assert _any_changed(before, _projector_snapshot(d)), (
            "Projector parameters did not change during training"
        )


# ── Projector Gradient Accumulation (Regression) ─────────────────────────
#
# The HF Trainer calls ``model.zero_grad()`` after each optimizer step, but
# projectors are not submodules of the model, so their gradients were
# silently accumulating across steps.  These tests reproduce the real
# Trainer code path (``model.zero_grad()``, NOT ``optimizer.zero_grad()``)
# to ensure ``BaseDistiller._zero_projector_gradients()`` prevents this.


class TestProjectorGradientAccumulation:
    """Projector gradients must not accumulate across training steps."""

    @staticmethod
    def _run_two_steps(distiller, model, device):
        """Run 2 training steps with model.zero_grad() and return per-step grad norms."""
        distiller._register_capture()
        distiller.create_optimizer()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        distiller.training_step(model, batch)
        distiller.optimizer.step()
        model.zero_grad()  # What the HF Trainer does — NOT optimizer.zero_grad()

        projs = [a.output_projector for a in distiller.alignments if a.output_projector is not None]
        assert projs, "No projectors created — test setup error"
        g1 = (
            sum(
                p.grad.norm().item() ** 2
                for proj in projs
                for p in proj.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        distiller.training_step(model, batch)
        g2 = (
            sum(
                p.grad.norm().item() ** 2
                for proj in projs
                for p in proj.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        distiller._deregister_capture()
        return g1, g2

    def test_holistic(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=_alignments_with_projectors(teacher, student),
            args=_args(tmp_path, device, max_steps=2, args_cls=TrainingArguments),
            train_dataset=_DATASET,
        )
        g1, g2 = self._run_two_steps(d, student, device)
        ratio = g2 / max(g1, 1e-8)
        assert ratio < 1.5, (
            f"HKD projector gradient accumulated: step1={g1:.4f}, step2={g2:.4f}, "
            f"ratio={ratio:.2f}. _zero_projector_gradients() is not working."
        )

    def test_blockwise(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=_alignments_with_projectors(teacher, student),
            args=_args(tmp_path, device, max_steps=2, args_cls=TrainingArguments),
            train_dataset=_DATASET,
        )
        g1, g2 = self._run_two_steps(d, d.model, device)
        ratio = g2 / max(g1, 1e-8)
        assert ratio < 1.5, (
            f"BKD projector gradient accumulated: step1={g1:.4f}, step2={g2:.4f}, "
            f"ratio={ratio:.2f}. _zero_projector_gradients() is not working."
        )


# ── Scheduler Stepping Correct (Regression) ──────────────────────────────


class TestSchedulerSteppingCorrect:
    """
    Regression test: each step advances the scheduler exactly once.

    BKD uses a ``CompositeScheduler`` with per-alignment children, so every
    alignment's scheduler must be advanced.  HKD uses HF Trainer's default
    single scheduler covering the full student + projectors.
    """

    STEPS = 5

    def test_blockwise(self, device, tmp_path):
        d, _, _ = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        d.train()
        for alignment in d.alignments:
            assert alignment.scheduler.last_epoch == self.STEPS

    def test_holistic(self, device, tmp_path):
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        d.train()
        # HKD uses HF Trainer's default single scheduler.  Verify it advanced
        # the expected number of steps.
        assert d.lr_scheduler is not None, "HKD did not create a scheduler"
        assert d.lr_scheduler.last_epoch == self.STEPS, (
            f"Expected scheduler.last_epoch == {self.STEPS}, got {d.lr_scheduler.last_epoch}"
        )


# ── Captured Data Lifecycle (Regression) ─────────────────────────────────


class TestCapturedDataLifecycle:
    """
    Regression test: captured data must not accumulate across training steps.
    The capture engine should overwrite (not append) on each step.
    """

    STEPS = 10

    def test_blockwise_no_accumulation(self, device, tmp_path):
        d, _, _ = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        d.train()
        # Outputs are overwritten each step; dict should have at most
        # _NUM_LAYERS entries (one per captured module).
        assert len(d.capture_engine.captured_outputs) <= _NUM_LAYERS

    def test_holistic_cleared_after_training(self, device, tmp_path):
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        d.train()
        # HolisticDistiller explicitly clears captured data at end of each step.
        assert len(d.teacher_capture.captured_outputs) == 0
        assert len(d.student_capture.captured_outputs) == 0


# ── Every Block Trained (Blockwise) ──────────────────────────────────


class TestEveryBlockTrained:
    """
    Verify ALL student blocks (not just "any") are updated during training.

    A bug could cause some blocks to be silently skipped (wrong alignment
    index, capture engine off-by-one, optimizer param group missing).
    """

    STEPS = 10

    def test_blockwise_all_blocks_updated(self, device, tmp_path):
        """Every student block must have at least one parameter changed."""
        d, _, _ = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        # Per-block snapshots before training
        block_before = {}
        for alignment in d.alignments:
            name = alignment.get_name()
            block_before[name] = {
                k: v.clone() for k, v in alignment.student_block.state_dict().items()
            }

        d.train()

        for alignment in d.alignments:
            name = alignment.get_name()
            changed = any(
                not torch.equal(v, block_before[name][k])
                for k, v in alignment.student_block.state_dict().items()
            )
            assert changed, f"Block {name} was never updated during training"

    def test_holistic_all_blocks_updated(self, device, tmp_path):
        """Every student block must be updated in holistic training."""
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        block_before = {}
        for alignment in d.alignments:
            name = alignment.get_name()
            block_before[name] = {
                k: v.clone() for k, v in alignment.student_block.state_dict().items()
            }

        d.train()

        for alignment in d.alignments:
            name = alignment.get_name()
            changed = any(
                not torch.equal(v, block_before[name][k])
                for k, v in alignment.student_block.state_dict().items()
            )
            assert changed, f"Block {name} was never updated during holistic training"


# ── Gradient Accumulation Correctness ─────────────────────────────────


class TestGradientAccumulationCorrectness:
    """
    Verify gradient accumulation actually works (loss decreases, not just
    "no crash") for blockwise, holistic, and response-based distillers with
    gradient_accumulation_steps > 1.
    """

    STEPS = 20

    @staticmethod
    def _check_loss_decrease(losses):
        assert len(losses) >= 4, f"Expected >= 4 logged losses, got {len(losses)}"
        q = max(1, len(losses) // 4)
        first = sum(losses[:q]) / q
        last = sum(losses[-q:]) / q
        assert last < first, (
            f"Loss did not decrease with gradient accumulation: "
            f"first-quarter={first:.4f}, last-quarter={last:.4f}"
        )

    def test_blockwise_grad_accum_loss_decreases(self, device, tmp_path):
        """Blockwise with gradient_accumulation_steps=2 produces decreasing loss."""
        d, _, _ = _make_blockwise(
            device,
            tmp_path,
            max_steps=self.STEPS,
            gradient_accumulation_steps=2,
        )
        d.train()
        self._check_loss_decrease(_logged_losses(d))

    def test_holistic_grad_accum_loss_decreases(self, device, tmp_path):
        """Holistic with gradient_accumulation_steps=2 produces decreasing loss."""
        d, _, _ = _make_holistic(
            device,
            tmp_path,
            max_steps=self.STEPS,
            gradient_accumulation_steps=2,
        )
        d.train()
        self._check_loss_decrease(_logged_losses(d))

    def test_response_based_grad_accum_loss_decreases(self, device, tmp_path):
        """ResponseBased with gradient_accumulation_steps=2 produces decreasing loss."""
        d, _, _ = _make_response_based(
            device,
            tmp_path,
            max_steps=self.STEPS,
            gradient_accumulation_steps=2,
        )
        d.train()
        self._check_loss_decrease(_logged_losses(d))


# ── Teacher Gradient Leakage ─────────────────────────────────────────


class TestTeacherGradientLeakage:
    """
    Verify no gradients are computed for teacher parameters.

    Goes beyond TestTeacherFrozen (which checks parameter values) by
    inspecting .grad attributes directly after a training step.
    """

    def test_blockwise_teacher_no_grads(self, device, tmp_path):
        """Teacher parameters should have no gradients after blockwise training."""
        d, teacher, _ = _make_blockwise(device, tmp_path, max_steps=3)
        d.train()
        for name, param in teacher.named_parameters():
            assert param.grad is None or param.grad.abs().max() == 0, (
                f"Teacher param {name} has non-zero gradient — gradient leakage"
            )

    def test_holistic_teacher_no_grads(self, device, tmp_path):
        """Teacher parameters should have no gradients after holistic training."""
        d, teacher, _ = _make_holistic(device, tmp_path, max_steps=3)
        d.train()
        for name, param in teacher.named_parameters():
            assert param.grad is None or param.grad.abs().max() == 0, (
                f"Teacher param {name} has non-zero gradient — gradient leakage"
            )

    def test_response_based_teacher_no_grads(self, device, tmp_path):
        """Teacher parameters should have no gradients after response-based training."""
        d, teacher, _ = _make_response_based(device, tmp_path, max_steps=3)
        d.train()
        for name, param in teacher.named_parameters():
            assert param.grad is None or param.grad.abs().max() == 0, (
                f"Teacher param {name} has non-zero gradient — gradient leakage"
            )


# ── Loss Values Finite and Non-Zero ──────────────────────────────────


class TestLossValuesFiniteNonZero:
    """
    Verify every logged loss value is finite and positive throughout training.

    Catches NaN/Inf loss bugs that TestLossDecreases might miss (a sequence
    like [10, NaN, NaN, 5] could pass a first-quarter/last-quarter check).
    """

    STEPS = 10

    @staticmethod
    def _check_all_finite_positive(losses):
        for i, loss in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss)), f"Loss at step {i} is not finite: {loss}"
            assert loss > 0, f"Loss at step {i} is not positive: {loss}"

    def test_blockwise(self, device, tmp_path):
        d, _, _ = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        d.train()
        self._check_all_finite_positive(_logged_losses(d))

    def test_holistic(self, device, tmp_path):
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        d.train()
        self._check_all_finite_positive(_logged_losses(d))

    def test_response_based(self, device, tmp_path):
        d, _, _ = _make_response_based(device, tmp_path, max_steps=self.STEPS)
        d.train()
        self._check_all_finite_positive(_logged_losses(d))


# ── Eval During Training ─────────────────────────────────────────────


class TestEvalDuringTraining:
    """Verify evaluation runs during training and produces valid metrics."""

    STEPS = 10
    EVAL_STEPS = 5

    def _check_eval_metrics(self, distiller):
        eval_entries = [e for e in distiller.state.log_history if "eval_loss" in e]
        assert len(eval_entries) > 0, "No eval metrics logged during training"
        for entry in eval_entries:
            assert entry["eval_loss"] > 0, f"eval_loss not positive: {entry['eval_loss']}"
            assert torch.isfinite(torch.tensor(entry["eval_loss"])), (
                f"eval_loss not finite: {entry['eval_loss']}"
            )

    def test_blockwise(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=_alignments(teacher, student),
            args=_args(
                tmp_path,
                device,
                max_steps=self.STEPS,
                args_cls=TrainingArguments,
                eval_strategy="steps",
                eval_steps=self.EVAL_STEPS,
            ),
            train_dataset=_DATASET,
            eval_dataset=_DATASET,
        )
        d.train()
        self._check_eval_metrics(d)

    def test_holistic(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=_alignments(teacher, student),
            args=_args(
                tmp_path,
                device,
                max_steps=self.STEPS,
                args_cls=TrainingArguments,
                eval_strategy="steps",
                eval_steps=self.EVAL_STEPS,
            ),
            train_dataset=_DATASET,
            eval_dataset=_DATASET,
        )
        d.train()
        self._check_eval_metrics(d)

    def test_response_based(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        d = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=_args(
                tmp_path,
                device,
                max_steps=self.STEPS,
                args_cls=TrainingArguments,
                learning_rate=1e-3,
                eval_strategy="steps",
                eval_steps=self.EVAL_STEPS,
            ),
            train_dataset=_DATASET,
            eval_dataset=_DATASET,
        )
        d.train()
        self._check_eval_metrics(d)


# ── compute_metrics integration ─────────────────────────────────────
#
# ``prediction_step`` returns ``(loss, logits, labels)`` only when ALL of:
#   1. ``compute_metrics`` is set
#   2. ``prediction_loss_only`` is False
#   3. ``"labels"`` is present in the input batch
#
# Otherwise logits/labels must be None to avoid OOM on large-vocab LLMs.
# HKD and ReSKD *can* produce logits (student output has ``.logits``);
# BKD *never* can (``StudentBlocksContainer`` has no ``.logits``).
#
# The tests below verify every (distiller × condition) combination.


def _make_hkd(device, tmp_path, compute_metrics=None, eval_dataset=None, **extra_args):
    teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    d = HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=_alignments(teacher, student),
        args=_args(tmp_path, device, args_cls=TrainingArguments, **extra_args),
        train_dataset=_DATASET,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    return d, student


def _make_reskd(device, tmp_path, compute_metrics=None, eval_dataset=None, **extra_args):
    teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    d = ResponseBasedDistiller(
        student_model=student,
        teacher_model=teacher,
        args=_args(tmp_path, device, args_cls=TrainingArguments, learning_rate=1e-3, **extra_args),
        train_dataset=_DATASET,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    return d, student


def _make_bkd(device, tmp_path, compute_metrics=None, eval_dataset=None, **extra_args):
    teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
    d = BlockwiseDistiller(
        teacher_model=teacher,
        alignments=_alignments(teacher, student),
        args=_args(tmp_path, device, args_cls=TrainingArguments, **extra_args),
        train_dataset=_DATASET,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    return d, student


def _compute_metrics(eval_pred):
    """Dummy compute_metrics that returns accuracy."""
    logits, labels = eval_pred
    assert logits is not None, "logits must not be None"
    assert labels is not None, "labels must not be None"
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


def _batch(device, include_labels=True):
    b = {
        "input_ids": torch.randint(0, 128, (2, 16), device=device),
        "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
    }
    if include_labels:
        b["labels"] = torch.randint(0, 128, (2, 16), device=device)
    return b


def _prepare_for_prediction(d):
    """Call _register_capture() if the distiller uses the capture engine
    (HKD, BKD).  ReSKD does its own forward pass and doesn't need it."""
    if isinstance(d, HolisticDistiller):
        d._register_capture()


def _assert_never_stashed(d, fn, *args, **kwargs):
    """Call *fn* and assert _last_student_output was never set to a non-None
    value during the call.  prediction_step always clears it at the end, so
    checking after the call would not catch temporary stashing that wastes
    memory."""
    _orig = d.compute_distillation_loss
    stashed = []

    def _spy(model, inputs, is_training):
        result = _orig(model, inputs, is_training)
        # Check immediately after compute_distillation_loss — this is
        # where subclasses conditionally stash the student output.
        val = getattr(d, "_last_student_output", None)
        if val is not None:
            stashed.append(True)
        return result

    d.compute_distillation_loss = _spy
    try:
        ret = fn(*args, **kwargs)
    finally:
        d.compute_distillation_loss = _orig
    assert not stashed, (
        "_last_student_output was populated during a suppressed "
        "prediction_step — memory was temporarily wasted"
    )
    return ret


import pytest  # noqa: E402 — placed near usage for readability

# ── Happy path: logits ARE produced ──────────────────────────────────


class TestComputeMetricsProducesLogits:
    """HKD and ReSKD produce logits+labels when all conditions are met.
    BKD never does (StudentBlocksContainer has no .logits).
    """

    STEPS = 10
    EVAL_STEPS = 5

    def _check_accuracy_logged(self, distiller):
        eval_entries = [e for e in distiller.state.log_history if "eval_loss" in e]
        assert len(eval_entries) > 0, "No eval metrics logged"
        for entry in eval_entries:
            assert "eval_accuracy" in entry, (
                f"compute_metrics was not called — eval_accuracy missing. "
                f"Keys: {list(entry.keys())}"
            )
            assert 0.0 <= entry["eval_accuracy"] <= 1.0

    @pytest.mark.parametrize("make", [_make_hkd, _make_reskd], ids=["hkd", "reskd"])
    def test_full_loop_accuracy_logged(self, device, tmp_path, make):
        """Full training loop: accuracy appears in eval logs."""
        d, _ = make(
            device,
            tmp_path,
            compute_metrics=_compute_metrics,
            eval_dataset=_DATASET,
            max_steps=self.STEPS,
            eval_strategy="steps",
            eval_steps=self.EVAL_STEPS,
        )
        d.train()
        self._check_accuracy_logged(d)

    @pytest.mark.parametrize("make", [_make_hkd, _make_reskd], ids=["hkd", "reskd"])
    def test_prediction_step_returns_logits(self, device, tmp_path, make):
        """Unit test: prediction_step returns (loss, logits, labels)."""
        d, student = make(device, tmp_path, compute_metrics=_compute_metrics, max_steps=1)
        _prepare_for_prediction(d)
        loss, logits, labels = d.prediction_step(
            student, _batch(device), prediction_loss_only=False
        )
        assert isinstance(loss, torch.Tensor)
        assert logits is not None, "Expected logits when all conditions met"
        assert labels is not None
        assert d._last_student_output is None, "output not freed"
        assert d._needs_student_logits is False, "flag not reset"

    def test_blockwise_graceful_degradation(self, device, tmp_path):
        """BKD: eval runs fine but accuracy is not logged (no .logits)."""
        d, _ = _make_bkd(
            device,
            tmp_path,
            compute_metrics=_compute_metrics,
            eval_dataset=_DATASET,
            max_steps=self.STEPS,
            eval_strategy="steps",
            eval_steps=self.EVAL_STEPS,
        )
        d.train()
        eval_entries = [e for e in d.state.log_history if "eval_loss" in e]
        assert len(eval_entries) > 0, "No eval metrics logged"
        assert "eval_accuracy" not in eval_entries[0]


# ── Suppression: logits must be None ─────────────────────────────────


class TestLogitsSuppressed:
    """Verify logits are None when any of the three conditions is unmet.

    Each condition is tested in isolation for every distiller type.
    """

    STEPS = 10
    EVAL_STEPS = 5

    # -- Condition 1: compute_metrics is None --

    @pytest.mark.parametrize("make", [_make_hkd, _make_reskd], ids=["hkd", "reskd"])
    def test_no_compute_metrics_unit(self, device, tmp_path, make):
        """prediction_step returns None logits without compute_metrics.
        Also verifies _last_student_output was never populated."""
        d, student = make(device, tmp_path, compute_metrics=None, max_steps=1)
        _prepare_for_prediction(d)
        loss, logits, labels = _assert_never_stashed(
            d, d.prediction_step, student, _batch(device), prediction_loss_only=False
        )
        assert isinstance(loss, torch.Tensor)
        assert logits is None
        assert labels is None

    @pytest.mark.parametrize(
        "make", [_make_hkd, _make_reskd, _make_bkd], ids=["hkd", "reskd", "bkd"]
    )
    def test_no_compute_metrics_full_loop(self, device, tmp_path, make):
        """Full loop without compute_metrics: no accuracy logged."""
        d, _ = make(
            device,
            tmp_path,
            compute_metrics=None,
            eval_dataset=_DATASET,
            max_steps=self.STEPS,
            eval_strategy="steps",
            eval_steps=self.EVAL_STEPS,
        )
        d.train()
        eval_entries = [e for e in d.state.log_history if "eval_loss" in e]
        assert len(eval_entries) > 0
        assert "eval_accuracy" not in eval_entries[0]

    # -- Condition 2: prediction_loss_only is True --

    @pytest.mark.parametrize("make", [_make_hkd, _make_reskd], ids=["hkd", "reskd"])
    def test_prediction_loss_only_unit(self, device, tmp_path, make):
        """prediction_loss_only=True suppresses logits even with
        compute_metrics set.  Also verifies _last_student_output was
        never populated (not just cleared at the end)."""
        d, student = make(device, tmp_path, compute_metrics=_compute_metrics, max_steps=1)
        _prepare_for_prediction(d)
        loss, logits, labels = _assert_never_stashed(
            d, d.prediction_step, student, _batch(device), prediction_loss_only=True
        )
        assert isinstance(loss, torch.Tensor)
        assert logits is None, (
            "logits must be None when prediction_loss_only=True — "
            "returning them would OOM on large-vocab models"
        )
        assert labels is None

    @pytest.mark.parametrize(
        "make", [_make_hkd, _make_reskd, _make_bkd], ids=["hkd", "reskd", "bkd"]
    )
    def test_prediction_loss_only_full_loop(self, device, tmp_path, make):
        """Full loop with prediction_loss_only=True: no accuracy logged."""
        d, _ = make(
            device,
            tmp_path,
            compute_metrics=_compute_metrics,
            eval_dataset=_DATASET,
            max_steps=self.STEPS,
            eval_strategy="steps",
            eval_steps=self.EVAL_STEPS,
            prediction_loss_only=True,
        )
        d.train()
        eval_entries = [e for e in d.state.log_history if "eval_loss" in e]
        assert len(eval_entries) > 0
        assert "eval_accuracy" not in eval_entries[0]

    # -- Condition 3: "labels" not in inputs --

    @pytest.mark.parametrize("make", [_make_hkd, _make_reskd], ids=["hkd", "reskd"])
    def test_no_labels_unit(self, device, tmp_path, make):
        """No labels in batch → no logits, even with compute_metrics set.
        Also verifies _last_student_output was never populated."""
        d, student = make(device, tmp_path, compute_metrics=_compute_metrics, max_steps=1)
        _prepare_for_prediction(d)
        loss, logits, labels = _assert_never_stashed(
            d,
            d.prediction_step,
            student,
            _batch(device, include_labels=False),
            prediction_loss_only=False,
        )
        assert isinstance(loss, torch.Tensor)
        assert logits is None
        assert labels is None


# ── Memory safety ────────────────────────────────────────────────────


class TestComputeMetricsMemorySafety:
    """Verify _last_student_output is freed and _needs_student_logits is
    never True during training — critical for avoiding OOM on large-vocab
    models."""

    STEPS = 10
    EVAL_STEPS = 5

    @pytest.mark.parametrize("make", [_make_hkd, _make_reskd], ids=["hkd", "reskd"])
    def test_output_freed_after_prediction_step(self, device, tmp_path, make):
        """_last_student_output is None after prediction_step."""
        d, student = make(device, tmp_path, compute_metrics=_compute_metrics, max_steps=1)
        _prepare_for_prediction(d)
        d.prediction_step(student, _batch(device), prediction_loss_only=False)
        assert d._last_student_output is None, (
            "_last_student_output not freed — risks OOM on large-vocab models"
        )
        assert d._needs_student_logits is False

    @pytest.mark.parametrize("make", [_make_hkd, _make_reskd], ids=["hkd", "reskd"])
    def test_training_never_stashes(self, device, tmp_path, make):
        """_needs_student_logits is never True during training steps, even
        when compute_metrics is set."""
        d, _ = make(
            device,
            tmp_path,
            compute_metrics=_compute_metrics,
            eval_dataset=_DATASET,
            max_steps=self.STEPS,
            eval_strategy="steps",
            eval_steps=self.EVAL_STEPS,
        )
        _orig = d.compute_distillation_loss
        stash_flags_during_training = []

        def _spy(model, inputs, is_training):
            if is_training:
                stash_flags_during_training.append(getattr(d, "_needs_student_logits", False))
            return _orig(model, inputs, is_training)

        d.compute_distillation_loss = _spy
        d.train()
        assert len(stash_flags_during_training) > 0, "No training steps"
        assert not any(stash_flags_during_training), (
            "_needs_student_logits was True during training — "
            "this would stash logits and waste memory"
        )


# ── Learning Rate Tracking ───────────────────────────────────────────


class TestLearningRateTracking:
    """Verify learning rates are logged to Trainer's log history.

    BKD uses ``CompositeOptimizer`` and logs per-alignment LRs with keys
    ``learning_rate/{alignment.get_name()}`` via
    ``BaseDistiller._track_student_learning_rates``.  HKD uses HF Trainer's
    default single optimizer, so HF Trainer logs a single ``learning_rate``
    field via its standard callback.
    """

    STEPS = 5

    def test_holistic_lr_logged(self, device, tmp_path):
        """HKD uses HF Trainer's default optimizer → single ``learning_rate`` key."""
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS)
        d.train()

        lr_entries = [e for e in d.state.log_history if "learning_rate" in e]
        assert len(lr_entries) >= self.STEPS, (
            f"Expected >= {self.STEPS} log entries with 'learning_rate', got {len(lr_entries)}"
        )
        for entry in lr_entries:
            lr = entry["learning_rate"]
            assert isinstance(lr, (int, float)), f"learning_rate is not numeric: {type(lr)}"
            assert lr >= 0, f"learning_rate is negative: {lr}"

    def test_blockwise_lr_logged(self, device, tmp_path):
        d, _, _ = _make_blockwise(device, tmp_path, max_steps=self.STEPS)
        d.train()

        lr_entries = [
            e for e in d.state.log_history if any(k.startswith("learning_rate/") for k in e)
        ]
        assert len(lr_entries) >= self.STEPS, (
            f"Expected >= {self.STEPS} log entries with LR metrics, got {len(lr_entries)}"
        )
        for entry in lr_entries:
            lr_keys = [k for k in entry if k.startswith("learning_rate/")]
            for key in lr_keys:
                assert isinstance(entry[key], float), (
                    f"LR value for {key} is not a float: {type(entry[key])}"
                )
                assert entry[key] > 0, f"LR value for {key} is not positive: {entry[key]}"


# ── Different Loss Per Alignment ─────────────────────────────────────


class TestDifferentLossPerAlignment:
    """Verify that different alignments can use different loss functions."""

    STEPS = 30

    def test_holistic_mixed_losses(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        aligns = []
        loss_names = ["mse", "smooth_l1"]
        for i in range(_NUM_LAYERS):
            alignment = Alignment(
                teacher_block=teacher.get_layer(i),
                student_block=student.get_layer(i),
                teacher_model_name="teacher",
                student_model_name="student",
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                loss_function=loss_names[i % len(loss_names)],
            )
            aligns.append(alignment)
        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=aligns,
            args=_args(
                tmp_path,
                device,
                max_steps=self.STEPS,
                args_cls=TrainingArguments,
            ),
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        assert len(losses) >= 4
        for loss in losses:
            assert torch.isfinite(torch.tensor(loss))
            assert loss > 0
        q = max(1, len(losses) // 4)
        first = sum(losses[:q]) / q
        last = sum(losses[-q:]) / q
        assert last < first

    def test_blockwise_mixed_losses(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        aligns = []
        loss_names = ["mse", "smooth_l1"]
        for i in range(_NUM_LAYERS):
            alignment = Alignment(
                teacher_block=teacher.get_layer(i),
                student_block=student.get_layer(i),
                teacher_model_name="teacher",
                student_model_name="student",
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                loss_function=loss_names[i % len(loss_names)],
            )
            aligns.append(alignment)
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=_args(
                tmp_path,
                device,
                max_steps=self.STEPS,
                args_cls=TrainingArguments,
            ),
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        assert len(losses) >= 4
        for loss in losses:
            assert torch.isfinite(torch.tensor(loss))
            assert loss > 0
        q = max(1, len(losses) // 4)
        first = sum(losses[:q]) / q
        last = sum(losses[-q:]) / q
        assert last < first


# ── HKD Hard Loss (alpha > 0) End-to-End ─────────────────────────────


class TestHolisticHardLossE2E:
    """End-to-end training correctness with HKD hard loss (alpha > 0).

    These complement the unit tests in test_holistic_distiller.py by
    running full training loops and verifying the same invariants as the
    pure-alignment tests (loss decreases, teacher frozen, student updates).
    """

    STEPS = 20

    def test_loss_decreases_with_alpha(self, device, tmp_path):
        """Loss should decrease when alpha=0.5 (mixed alignment + hard)."""
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS, alpha=0.5)
        d.train()
        losses = _logged_losses(d)
        assert len(losses) >= 4
        q = max(1, len(losses) // 4)
        first = sum(losses[:q]) / q
        last = sum(losses[-q:]) / q
        assert last < first, (
            f"Loss did not decrease with alpha=0.5: "
            f"first-quarter={first:.4f}, last-quarter={last:.4f}"
        )

    def test_teacher_frozen_with_alpha(self, device, tmp_path):
        """Teacher parameters must not change during alpha > 0 training."""
        d, teacher, _ = _make_holistic(device, tmp_path, max_steps=10, alpha=0.5)
        before = _snapshot(teacher)
        d.train()
        assert not _any_changed(before, _snapshot(teacher)), (
            "Teacher params changed during alpha=0.5 training"
        )

    def test_student_updates_with_alpha(self, device, tmp_path):
        """Student blocks must update during alpha > 0 training."""
        d, _, _ = _make_holistic(device, tmp_path, max_steps=10, alpha=0.5)
        before = _student_block_snapshot(d)
        d.train()
        assert _any_changed(before, _student_block_snapshot(d)), (
            "Student blocks unchanged during alpha=0.5 training"
        )

    def test_hard_loss_metrics_logged(self, device, tmp_path):
        """loss/hard and loss/alignment must appear in log_history with alpha > 0."""
        d, _, _ = _make_holistic(device, tmp_path, max_steps=10, alpha=0.5)
        d.train()

        all_keys = set()
        hard_values = []
        alignment_values = []
        for entry in d.state.log_history:
            all_keys.update(entry.keys())
            if "loss/hard" in entry:
                hard_values.append(entry["loss/hard"])
            if "loss/alignment" in entry:
                alignment_values.append(entry["loss/alignment"])

        assert "loss/hard" in all_keys, (
            f"loss/hard not logged with alpha=0.5. Keys: {sorted(all_keys)}"
        )
        assert "loss/alignment" in all_keys, (
            f"loss/alignment not logged with alpha=0.5. Keys: {sorted(all_keys)}"
        )
        assert all(v > 0 for v in hard_values), f"Non-positive hard loss: {hard_values}"
        assert all(v > 0 for v in alignment_values), (
            f"Non-positive alignment loss: {alignment_values}"
        )

    def test_alpha_one_pure_hard_loss_converges(self, device, tmp_path):
        """alpha=1.0 (pure hard loss) should still produce valid training."""
        d, _, _ = _make_holistic(device, tmp_path, max_steps=self.STEPS, alpha=1.0)
        d.train()
        losses = _logged_losses(d)
        assert len(losses) >= 4
        for v in losses:
            assert torch.isfinite(torch.tensor(v)), f"Non-finite loss: {v}"

    def test_alpha_with_magnitude_aware(self, device, tmp_path):
        """alpha > 0 + magnitude_aware_weighting should train without issues."""
        d, _, _ = _make_holistic(
            device, tmp_path, max_steps=self.STEPS, alpha=0.5, magnitude_aware_weighting=True
        )
        d.train()
        losses = _logged_losses(d)
        assert len(losses) >= 4
        for v in losses:
            assert torch.isfinite(torch.tensor(v)) and v > 0, f"Bad loss: {v}"
