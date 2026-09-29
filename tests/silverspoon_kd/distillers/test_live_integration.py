"""
Live integration tests exercising user-facing features through full train() calls.

These tests cover features that work in unit tests but have never been exercised
through the complete Trainer loop, capture engines, and checkpoint system.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
from silverspoon_kd.distillers.holistic_distiller import HolisticDistiller
from silverspoon_kd.distillers.response_based_distiller import ResponseBasedDistiller
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)
from tests.silverspoon_kd.conftest import SimpleModel

# ── Constants ─────────────────────────────────────────────────────────────

FEATURE_BASED_IDS = ["blockwise", "holistic"]
ALL_IDS = ["blockwise", "holistic", "response_based"]
_NUM_LAYERS = 2


# ── Helpers ───────────────────────────────────────────────────────────────


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


def _args(tmp_path, device, max_steps=20, args_cls=TrainingArguments, **extra):
    extra.setdefault("save_strategy", "no")
    args = args_cls(
        output_dir=str(tmp_path),
        num_train_epochs=100,
        per_device_train_batch_size=4,
        logging_steps=1,
        max_steps=max_steps,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=(device.type == "cpu"),
        **extra,
    )
    # Prevent DataParallel wrapping on multi-GPU machines
    args._n_gpu = 1
    return args


def _logged_losses(distiller):
    """Extract per-step training losses from Trainer's log history."""
    return [e["loss"] for e in distiller.state.log_history if "loss" in e]


def _alignments(teacher, student, **sa_kwargs):
    """Create alignments with optional extra kwargs passed to Alignment."""
    a = []
    for i in range(_NUM_LAYERS):
        alignment = Alignment(
            teacher_block=teacher.get_layer(i),
            student_block=student.get_layer(i),
            teacher_model_name="teacher",
            student_model_name="student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            **sa_kwargs,
        )
        a.append(alignment)
    return a


def _alignments_with_projectors(teacher, student, **sa_kwargs):
    """Create alignments with projectors for dimension mismatch."""
    from silverspoon_kd.alignments.projectors import GenericLinearProjector

    a = []
    for i in range(_NUM_LAYERS):
        dev = next(student.get_layer(i).parameters()).device
        output_projector = nn.Linear(student.hidden_dim, teacher.hidden_dim).to(dev)

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
            **sa_kwargs,
        )
        a.append(alignment)
    return a


def _make_distiller(
    distiller_id,
    device,
    tmp_path,
    teacher_hdim=128,
    student_hdim=128,
    sa_kwargs=None,
    **args_kwargs,
):
    """Factory to create a distiller by ID with same-dim models."""
    teacher = SimpleModel(64, teacher_hdim, _NUM_LAYERS).to(device)
    student = SimpleModel(64, student_hdim, _NUM_LAYERS).to(device)
    kwargs = sa_kwargs or {}

    if distiller_id == "response_based":
        args_kwargs.setdefault("learning_rate", 1e-3)
        d = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=_args(tmp_path, device, args_cls=TrainingArguments, **args_kwargs),
            train_dataset=_DATASET,
        )
    else:
        aligns = _alignments(teacher, student, **kwargs)
        if distiller_id == "blockwise":
            args = _args(tmp_path, device, args_cls=TrainingArguments, **args_kwargs)
            d = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=aligns,
                args=args,
                train_dataset=_DATASET,
            )
        elif distiller_id == "holistic":
            args = _args(tmp_path, device, args_cls=TrainingArguments, **args_kwargs)
            d = HolisticDistiller(
                student_model=student,
                teacher_model=teacher,
                alignments=aligns,
                args=args,
                train_dataset=_DATASET,
            )
        else:
            raise ValueError(f"Unknown distiller_id: {distiller_id}")
    return d, teacher, student


def _check_finite_positive(losses):
    """Assert all losses are finite and positive."""
    assert len(losses) > 0, "No losses recorded"
    for i, loss in enumerate(losses):
        assert math.isfinite(loss), f"Loss at step {i} is not finite: {loss}"
        assert loss > 0, f"Loss at step {i} is not positive: {loss}"


# ── 1. TestCustomLossFunctions ────────────────────────────────────────────


class TestCustomLossFunctions:
    """Test that non-MSE losses from the registry work through train()."""

    STEPS = 10

    @pytest.mark.parametrize("loss_name", ["cosine", "smooth_l1", "kl_div"])
    @pytest.mark.parametrize("distiller_id", FEATURE_BASED_IDS)
    def test_registry_loss(self, loss_name, distiller_id, device, tmp_path):
        d, _, _ = _make_distiller(
            distiller_id,
            device,
            tmp_path,
            max_steps=self.STEPS,
            sa_kwargs={"loss_function": loss_name},
        )
        d.train()
        _check_finite_positive(_logged_losses(d))

    def test_callable_loss(self, device, tmp_path):
        """A custom callable loss works through the full training loop."""
        d, _, _ = _make_distiller(
            "holistic",
            device,
            tmp_path,
            max_steps=self.STEPS,
            sa_kwargs={"loss_function": lambda s, t: F.l1_loss(s, t)},
        )
        d.train()
        _check_finite_positive(_logged_losses(d))


# ── 2. TestDataCollator ──────────────────────────────────────────────────


class TestDataCollator:
    """Test that a custom data_collator is used during training."""

    def test_custom_collator_called(self, device, tmp_path):
        call_count = {"n": 0}

        def tracking_collator(features):
            call_count["n"] += 1
            batch = {}
            for key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])
            return batch

        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        aligns = _alignments(teacher, student)
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=_args(tmp_path, device, max_steps=5, args_cls=TrainingArguments),
            train_dataset=_DATASET,
            data_collator=tracking_collator,
        )
        d.train()
        assert call_count["n"] > 0, "Custom data_collator was never called"


# ── 3. TestTorchCompile ──────────────────────────────────────────────────


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="torch.compile with default backend requires CUDA",
)
class TestTorchCompile:
    """Test that torch_compile=True works through a full training loop."""

    STEPS = 5

    def test_blockwise(self, device, tmp_path):
        """Blockwise: teacher compiled, container not compiled."""
        d, _, _ = _make_distiller(
            "blockwise",
            device,
            tmp_path,
            max_steps=self.STEPS,
            torch_compile=True,
        )
        d.train()
        _check_finite_positive(_logged_losses(d))

    def test_holistic(self, device, tmp_path):
        """Holistic: both teacher and student compiled."""
        d, _, _ = _make_distiller(
            "holistic",
            device,
            tmp_path,
            max_steps=self.STEPS,
            torch_compile=True,
        )
        d.train()
        _check_finite_positive(_logged_losses(d))


# ── 4. TestAutoDtypeMatch ────────────────────────────────────────────────


class TestAutoDtypeMatch:
    """Test auto_dtype_match with float64 student blocks through full training."""

    STEPS = 10

    @pytest.mark.parametrize("distiller_id", ["blockwise", "holistic"])
    def test_auto_dtype_match(self, distiller_id, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)

        if distiller_id == "blockwise":
            # Blockwise trains blocks independently — only blocks need float64
            for i in range(_NUM_LAYERS):
                student.get_layer(i).to(dtype=torch.float64)
        else:
            # Holistic runs full student forward pass — entire model must match dtype
            student = student.to(dtype=torch.float64)

        aligns = _alignments(teacher, student, auto_dtype_match=True)

        if distiller_id == "blockwise":
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
        else:
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
        _check_finite_positive(_logged_losses(d))

        # Verify student blocks are still float64
        for i in range(_NUM_LAYERS):
            for p in student.get_layer(i).parameters():
                assert p.dtype == torch.float64, (
                    f"Student block {i} param dtype changed from float64 to {p.dtype}"
                )


# ── 5. TestAutoProjector ─────────────────────────────────────────────────


class TestAutoProjector:
    """Test auto_projector=True with lazy projector creation through full training.

    auto_projector creates output projectors on-the-fly when teacher/student
    dimensions differ. Verifies projectors are created, losses are valid,
    and checkpoints contain projector_state.pt across all feature-based
    distiller types.
    """

    STEPS = 10

    # ── Dimension mismatch → projectors created ─────────────────────────

    def test_blockwise_auto_projector(self, device, tmp_path):
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        aligns = _alignments(teacher, student, auto_projector=True)
        args = _args(
            tmp_path,
            device,
            max_steps=self.STEPS,
            args_cls=TrainingArguments,
            save_strategy="steps",
            save_steps=5,
        )

        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=args,
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        _check_finite_positive(losses)

        # 1. Projectors were created by auto-inference
        for alignment in d.alignments:
            assert alignment.output_projector is not None, (
                f"Auto output projector not created for {alignment.get_name()}"
            )

        # 2. Checkpoint contains projector_state.pt
        checkpoint_dirs = sorted(Path(str(tmp_path)).glob("checkpoint-*"))
        assert len(checkpoint_dirs) > 0, "No checkpoint saved"
        assert (checkpoint_dirs[0] / "projector_state.pt").exists(), (
            "projector_state.pt missing from checkpoint"
        )

    def test_holistic_auto_projector(self, device, tmp_path):
        """HKD auto_projector creates output projectors for dimension mismatch."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        aligns = _alignments(teacher, student, auto_projector=True)
        args = _args(
            tmp_path,
            device,
            max_steps=self.STEPS,
            args_cls=TrainingArguments,
            save_strategy="steps",
            save_steps=5,
        )

        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=aligns,
            args=args,
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        _check_finite_positive(losses)

        for alignment in d.alignments:
            assert alignment.output_projector is not None, (
                f"Auto output projector not created for {alignment.get_name()}"
            )

        checkpoint_dirs = sorted(Path(str(tmp_path)).glob("checkpoint-*"))
        assert len(checkpoint_dirs) > 0, "No checkpoint saved"
        assert (checkpoint_dirs[0] / "projector_state.pt").exists(), (
            "projector_state.pt missing from checkpoint"
        )

    # ── Losses decrease (projectors are training) ────────────────────────

    def test_blockwise_auto_projector_losses_decrease(self, device, tmp_path):
        """Auto-created projectors train and contribute to loss decrease."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        aligns = _alignments(teacher, student, auto_projector=True)

        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=_args(tmp_path, device, max_steps=20, args_cls=TrainingArguments),
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        _check_finite_positive(losses)

        # Loss should decrease over training (projectors are learning)
        q = max(1, len(losses) // 4)
        first = sum(losses[:q]) / q
        last = sum(losses[-q:]) / q
        assert last < first, (
            f"Loss did not decrease with auto projectors: "
            f"first-quarter={first:.4f}, last-quarter={last:.4f}"
        )

    def test_holistic_auto_projector_losses_decrease(self, device, tmp_path):
        """HKD auto-created projectors train and contribute to loss decrease."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 64, _NUM_LAYERS).to(device)
        aligns = _alignments(teacher, student, auto_projector=True)

        d = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=aligns,
            args=_args(tmp_path, device, max_steps=20, args_cls=TrainingArguments),
            train_dataset=_DATASET,
        )
        d.train()
        losses = _logged_losses(d)
        _check_finite_positive(losses)

        q = max(1, len(losses) // 4)
        first = sum(losses[:q]) / q
        last = sum(losses[-q:]) / q
        assert last < first, (
            f"Loss did not decrease with auto projectors (HKD): "
            f"first-quarter={first:.4f}, last-quarter={last:.4f}"
        )

    # ── Same dimensions → no projectors created ─────────────────────────

    @pytest.mark.parametrize("distiller_id", FEATURE_BASED_IDS)
    def test_same_dim_no_projector(self, distiller_id, device, tmp_path):
        """auto_projector=True with matching dims should NOT create projectors."""
        d, _, _ = _make_distiller(
            distiller_id,
            device,
            tmp_path,
            teacher_hdim=128,
            student_hdim=128,
            max_steps=self.STEPS,
            sa_kwargs={"auto_projector": True},
        )
        d.train()
        _check_finite_positive(_logged_losses(d))

        for alignment in d.alignments:
            assert alignment.output_projector is None, (
                f"Projector should not be created for same-dim alignment {alignment.get_name()}"
            )


# ── 6. TestTrainingArgumentsMixedPrecision ────────────────────────────────


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="AMP requires CUDA",
)
class TestTrainingArgumentsMixedPrecision:
    """Test Trainer's native AMP via fp16=True."""

    STEPS = 5

    def test_blockwise_fp16(self, device, tmp_path):
        d, _, _ = _make_distiller(
            "blockwise",
            device,
            tmp_path,
            max_steps=self.STEPS,
            fp16=True,
        )
        d.train()
        _check_finite_positive(_logged_losses(d))

    def test_response_based_fp16(self, device, tmp_path):
        d, _, _ = _make_distiller(
            "response_based",
            device,
            tmp_path,
            max_steps=self.STEPS,
            fp16=True,
        )
        d.train()
        _check_finite_positive(_logged_losses(d))


# ── 10. TestCreateAlignmentsByModuleIntegration ───────────────────────────


class TestCreateAlignmentsByModuleIntegration:
    """Test create_alignments → distiller → train()."""

    STEPS = 10

    def test_basic(self, device, tmp_path):
        from silverspoon_kd.alignments.utils import create_alignments

        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(device)
        teacher.name_or_path = "test_teacher"
        student.name_or_path = "test_student"

        aligns = create_alignments(
            teacher_model=teacher,
            student_model=student,
            modules=r"layers\.\d+",
        )

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
        _check_finite_positive(_logged_losses(d))

        # Verify auto_projector was enabled by create_alignments
        for alignment in aligns:
            assert alignment.auto_projector is True
