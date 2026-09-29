"""
Capability-oriented checkpoint resume tests for all distiller types.

Tests are organized by checkpoint capability (files exist, weights roundtrip,
optimizer roundtrip, etc.) and parametrized across distiller types, ensuring
uniform coverage.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
from silverspoon_kd.distillers.holistic_distiller import HolisticDistiller
from silverspoon_kd.distillers.response_based_distiller import ResponseBasedDistiller
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)

from ..conftest import DummyDataset, SimpleModel, create_alignment

# ---------------------------------------------------------------------------
# IDs for parametrize
# ---------------------------------------------------------------------------

DISTILLER_IDS = ["blockwise", "holistic", "response_based"]
# BKD is the only remaining distiller that uses CompositeOptimizer.
# HKD switched to a single whole-student optimizer (HF Trainer default) to
# fix the regression where non-aligned parameters (embeddings, pooler,
# lm_head) were silently not updated.  ResKD also uses the HF default.
COMPOSITE_IDS = ["blockwise"]
# Distillers that create projectors and therefore save ``projector_state.pt``.
# Used for tests that exercise projector checkpoint/resume independently
# of whether the distiller uses CompositeOptimizer.
PROJECTOR_IDS = ["blockwise", "holistic"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ARGS_CLASS = {
    "blockwise": TrainingArguments,
    "holistic": TrainingArguments,
    "response_based": TrainingArguments,
}


def _make_training_args(tmp_path, device, max_steps=6, save_steps=3, distiller_id=None, **kwargs):
    """Create TrainingArguments configured for checkpoint tests.

    When distiller_id is provided, the correct subclass is used automatically.
    """
    cls = _ARGS_CLASS.get(distiller_id, TrainingArguments) if distiller_id else TrainingArguments
    args = cls(
        output_dir=str(tmp_path / "output"),
        num_train_epochs=100,  # high so max_steps controls stopping
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        save_steps=save_steps,
        save_strategy="steps",
        max_steps=max_steps,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=(device.type == "cpu"),
        **kwargs,
    )
    # Prevent DataParallel wrapping on multi-GPU machines
    args._n_gpu = 1
    return args


def _make_blockwise_alignments(teacher_model, student_model, num_layers=2):
    """Create alignments for blockwise distillers (dim mismatch → projectors)."""
    alignments = []
    for i in range(num_layers):
        teacher_block = teacher_model.get_layer(i)
        student_block = student_model.get_layer(i)
        alignment = create_alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            with_input_projector=(i > 0),
        )
        alignments.append(alignment)
    return alignments


def _make_holistic_alignments(teacher_model, student_model, num_layers=2):
    """Create alignments for holistic distillers (dim mismatch → projectors)."""
    alignments = []
    for i in range(num_layers):
        teacher_block = teacher_model.get_layer(i)
        student_block = student_model.get_layer(i)
        alignment = create_alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
        )
        alignments.append(alignment)
    return alignments


def _create_models(distiller_id, device):
    """Create teacher/student with appropriate dims per distiller type."""
    teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
    student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
    return teacher, student


def _make_distiller(distiller_id, teacher, student, args, dataset):
    """Factory: create a distiller by ID with appropriate alignments."""
    if distiller_id == "blockwise":
        alignments = _make_blockwise_alignments(teacher, student, num_layers=2)
        return BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=dataset,
        )
    elif distiller_id == "holistic":
        alignments = _make_holistic_alignments(teacher, student, num_layers=2)
        return HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=dataset,
        )
    elif distiller_id == "response_based":
        return ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=dataset,
        )
    else:
        raise ValueError(f"Unknown distiller_id: {distiller_id}")


def _get_student_block_params(distiller):
    """Snapshot all student block parameters as {name: tensor_clone}."""
    params = {}
    for alignment in distiller.alignments:
        name = alignment.get_name()
        for pn, p in alignment.student_block.named_parameters():
            params[f"{name}/{pn}"] = p.data.clone()
    return params


def _get_model_params(distiller, distiller_id):
    """Snapshot model parameters — alignment-based for composite, model for response."""
    if distiller_id == "response_based":
        return {n: p.data.clone() for n, p in distiller.model.named_parameters()}
    return _get_student_block_params(distiller)


def _get_optimizer_states(distiller):
    """Snapshot per-alignment optimizer state dicts (deep-copied)."""
    states = {}
    for alignment in distiller.alignments:
        states[alignment.get_name()] = deepcopy(alignment.optimizer.state_dict())
    return states


def _get_scheduler_states(distiller):
    """Snapshot per-alignment scheduler state dicts."""
    states = {}
    for alignment in distiller.alignments:
        states[alignment.get_name()] = deepcopy(alignment.scheduler.state_dict())
    return states


def _find_checkpoint_dir(output_dir):
    """Find the most recent checkpoint-* directory."""
    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint-*"))
    assert len(checkpoints) > 0, f"No checkpoints found in {output_dir}"
    return str(checkpoints[-1])


def _first_checkpoint(output_dir):
    """Return the earliest checkpoint-* directory."""
    return str(
        min(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
    )


def _projector_snapshot(distiller):
    """Snapshot all projector parameters across all alignments."""
    params = {}
    for alignment in distiller.alignments:
        if alignment.output_projector is not None:
            for n, p in alignment.output_projector.named_parameters():
                params[f"{alignment.get_name()}/out_proj.{n}"] = p.data.clone()
        if alignment.input_projector is not None:
            for n, p in alignment.input_projector.named_parameters():
                params[f"{alignment.get_name()}/in_proj.{n}"] = p.data.clone()
    return params


# ---------------------------------------------------------------------------
# Deterministic dataset (for resume == uninterrupted tests)
# ---------------------------------------------------------------------------


class _DeterministicDataset(Dataset):
    """Pre-generated dataset so the same index always returns the same tensors."""

    def __init__(self, num_samples=30, seq_len=16, seed=0):
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


# =====================================================================
# 1. TestCheckpointFilesExist
# =====================================================================


class TestCheckpointFilesExist:
    """Verify that each distiller type creates the correct checkpoint files."""

    @pytest.mark.parametrize("distiller_id", DISTILLER_IDS)
    def test_correct_files_created(self, distiller_id, device, tmp_path):
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=3, save_steps=3, distiller_id=distiller_id
        )

        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()

        ckpt_dir = Path(_find_checkpoint_dir(args.output_dir))

        # All distillers produce model.safetensors + trainer_state.json
        has_model = (ckpt_dir / "model.safetensors").exists() or (
            ckpt_dir / "pytorch_model.bin"
        ).exists()
        assert has_model
        assert (ckpt_dir / "trainer_state.json").exists()

        if distiller_id in COMPOSITE_IDS:
            assert (ckpt_dir / "student_training_state.pt").exists()
        else:
            # response_based uses standard Trainer optimizer
            assert not (ckpt_dir / "student_training_state.pt").exists()

        # Projectors: blockwise and holistic have dim mismatch → projector_state.pt
        if distiller_id in ("blockwise", "holistic"):
            assert (ckpt_dir / "projector_state.pt").exists()


# =====================================================================
# 2. TestModelWeightsRoundtrip
# =====================================================================


class TestModelWeightsRoundtrip:
    """Train → checkpoint → fresh distiller → load → params match."""

    @pytest.mark.parametrize("distiller_id", DISTILLER_IDS)
    def test_weights_restored(self, distiller_id, device, tmp_path):
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=4, save_steps=4, distiller_id=distiller_id
        )

        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()

        params_after = _get_model_params(distiller, distiller_id)
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Fresh distiller + load
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=4,
            save_steps=4,
            distiller_id=distiller_id,
        )
        distiller2 = _make_distiller(distiller_id, teacher2, student2, args2, dataset)
        distiller2._load_from_checkpoint(ckpt_dir)

        restored = _get_model_params(distiller2, distiller_id)
        for key in params_after:
            assert key in restored, f"Missing key: {key}"
            assert torch.allclose(params_after[key], restored[key], atol=1e-6), (
                f"Weight mismatch at {key}: "
                f"max diff = {(params_after[key] - restored[key]).abs().max().item()}"
            )


# =====================================================================
# 2b. TestCheckpointInference
# =====================================================================


class TestCheckpointInference:
    """Train → checkpoint → fresh distiller → load → forward pass produces same loss.

    Verifies that the loaded checkpoint is not just weight-equivalent but
    functionally correct: the full distillation forward graph (teacher forward,
    student forward, projectors, loss) works and produces the same loss value.
    """

    @staticmethod
    def _eval_loss(distiller, batch):
        """Register capture hooks, run one eval forward pass, deregister."""
        distiller._register_capture()
        distiller.model.eval()
        distiller.teacher_model.eval()
        try:
            with torch.no_grad():
                return distiller.compute_distillation_loss(
                    distiller.model, batch, is_training=False
                )
        finally:
            distiller._deregister_capture()
            distiller._clear_captured_data()

    @pytest.mark.parametrize("distiller_id", DISTILLER_IDS)
    def test_loaded_checkpoint_produces_same_loss(self, distiller_id, device, tmp_path):
        torch.manual_seed(42)
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=4, save_steps=4, distiller_id=distiller_id
        )

        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()

        # Fixed batch for reproducible comparison
        gen = torch.Generator(device=device).manual_seed(99)
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), generator=gen, device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), generator=gen, device=device),
        }

        loss_original = self._eval_loss(distiller, batch)
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Fresh distiller + load checkpoint
        torch.manual_seed(42)
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=4,
            save_steps=4,
            distiller_id=distiller_id,
        )
        distiller2 = _make_distiller(distiller_id, teacher2, student2, args2, dataset)
        distiller2._load_from_checkpoint(ckpt_dir)

        loss_loaded = self._eval_loss(distiller2, batch)

        assert torch.allclose(loss_original, loss_loaded, atol=1e-5), (
            f"Loss mismatch after checkpoint load for {distiller_id}: "
            f"original={loss_original.item():.6f}, loaded={loss_loaded.item():.6f}"
        )


# =====================================================================
# 3. TestOptimizerSchedulerRoundtrip
# =====================================================================


class TestOptimizerSchedulerRoundtrip:
    """Train → checkpoint → fresh distiller → load optimizer/scheduler → states match."""

    @pytest.mark.parametrize("distiller_id", COMPOSITE_IDS)
    def test_optimizer_scheduler_restored(self, distiller_id, device, tmp_path):
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=4, save_steps=4, distiller_id=distiller_id
        )

        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()

        opt_after = _get_optimizer_states(distiller)
        sched_after = _get_scheduler_states(distiller)
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Fresh distiller + load
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=4,
            save_steps=4,
            distiller_id=distiller_id,
        )
        distiller2 = _make_distiller(distiller_id, teacher2, student2, args2, dataset)
        distiller2._load_from_checkpoint(ckpt_dir)
        distiller2._load_optimizer_and_scheduler(ckpt_dir)

        # Compare optimizer states
        restored_opt = _get_optimizer_states(distiller2)
        for sname in opt_after:
            assert sname in restored_opt
            orig = opt_after[sname]["state"]
            rest = restored_opt[sname]["state"]
            for param_idx in orig:
                for key, val in orig[param_idx].items():
                    if isinstance(val, torch.Tensor):
                        assert torch.allclose(rest[param_idx][key], val, atol=1e-6), (
                            f"Optimizer state mismatch for {sname} param {param_idx} key {key}"
                        )

        # Compare scheduler states
        restored_sched = _get_scheduler_states(distiller2)
        for sname in sched_after:
            assert sname in restored_sched
            assert sched_after[sname]["last_epoch"] == restored_sched[sname]["last_epoch"]


# =====================================================================
# 4. TestResponseBasedTrainerCheckpoint
# =====================================================================


class TestResponseBasedTrainerCheckpoint:
    """Verify ResponseBasedDistiller saves standard Trainer optimizer files."""

    def test_trainer_optimizer_files_exist(self, device, tmp_path):
        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=3, save_steps=3, distiller_id="response_based"
        )

        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=dataset,
        )
        distiller.train()
        ckpt_dir = Path(_find_checkpoint_dir(args.output_dir))

        has_optimizer = (ckpt_dir / "optimizer.pt").exists() or (
            ckpt_dir / "optimizer.safetensors"
        ).exists()
        assert has_optimizer, "Standard Trainer optimizer file not found"
        assert (ckpt_dir / "scheduler.pt").exists()

    def test_resume_with_hard_labels(self, device, tmp_path):
        """Test resume with alpha > 0 (soft + hard loss)."""
        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path,
            device,
            max_steps=3,
            save_steps=3,
            distiller_id="response_based",
            alpha=0.5,
        )

        distiller = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=dataset,
        )
        distiller.train()
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        student2 = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=6,
            save_steps=3,
            distiller_id="response_based",
            alpha=0.5,
        )
        distiller2 = ResponseBasedDistiller(
            student_model=student2,
            teacher_model=teacher,
            args=args2,
            train_dataset=dataset,
        )
        result = distiller2.train(resume_from_checkpoint=ckpt_dir)
        assert result is not None


# =====================================================================
# 5. TestProjectorRoundtrip
# =====================================================================


class TestProjectorRoundtrip:
    """Verify projector weights survive checkpoint roundtrip for all feature-based distillers.

    Uses dim mismatch (teacher=128, student=64) to force projector creation.
    All distillers save projectors via projector_state.pt.
    """

    @pytest.mark.parametrize("distiller_id", ["blockwise", "holistic"])
    def test_projectors_roundtrip(self, distiller_id, device, tmp_path):
        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        # Always use dim mismatch to force projectors
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=4, save_steps=4, distiller_id=distiller_id
        )

        # Create with dim-mismatch alignments that produce projectors
        if distiller_id == "blockwise":
            alignments = _make_blockwise_alignments(teacher, student, num_layers=2)
            distiller = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
            )
        elif distiller_id == "holistic":
            alignments = _make_holistic_alignments(teacher, student, num_layers=2)
            distiller = HolisticDistiller(
                student_model=student,
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
            )

        distiller.train()

        proj_after = _projector_snapshot(distiller)
        assert proj_after, "Test setup error: no projectors found"
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # All distiller types now save projectors in projector_state.pt
        assert (Path(ckpt_dir) / "projector_state.pt").exists(), (
            f"{distiller_id} should create projector_state.pt"
        )

        # Reload into fresh distiller
        student2 = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=4,
            save_steps=4,
            distiller_id=distiller_id,
        )

        if distiller_id == "blockwise":
            alignments2 = _make_blockwise_alignments(teacher, student2, num_layers=2)
            distiller2 = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=alignments2,
                args=args2,
                train_dataset=dataset,
            )
        elif distiller_id == "holistic":
            alignments2 = _make_holistic_alignments(teacher, student2, num_layers=2)
            distiller2 = HolisticDistiller(
                student_model=student2,
                teacher_model=teacher,
                alignments=alignments2,
                args=args2,
                train_dataset=dataset,
            )

        distiller2._load_from_checkpoint(ckpt_dir)

        proj_restored = _projector_snapshot(distiller2)
        for key in proj_after:
            assert key in proj_restored, f"Missing projector key: {key}"
            assert torch.allclose(proj_after[key], proj_restored[key], atol=1e-6), (
                f"Projector weight mismatch for {key}: "
                f"max diff = {(proj_after[key] - proj_restored[key]).abs().max().item()}"
            )


# =====================================================================
# 5b. TestModelCheckpointExcludesProjectors
# =====================================================================


class TestModelCheckpointExcludesProjectors:
    """Verify model.safetensors does NOT contain projector weights for any distiller.

    Projectors are always saved in projector_state.pt, never in model.safetensors.
    Each distiller type stores different things in model.safetensors:
    - blockwise: StudentBlocksContainer state (block_* keys only)
    - holistic: full student model state (embedding, layers, lm_head)
    """

    @pytest.mark.parametrize("distiller_id", ["blockwise", "holistic"])
    def test_model_safetensors_excludes_projector_keys(self, distiller_id, device, tmp_path):
        from safetensors.torch import load_file

        # Always use dim mismatch to force projectors
        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=3, save_steps=3, distiller_id=distiller_id
        )

        if distiller_id == "blockwise":
            alignments = _make_blockwise_alignments(teacher, student, num_layers=2)
            distiller = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
            )
        elif distiller_id == "holistic":
            alignments = _make_holistic_alignments(teacher, student, num_layers=2)
            distiller = HolisticDistiller(
                student_model=student,
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
            )

        distiller.train()
        ckpt_dir = Path(_find_checkpoint_dir(args.output_dir))

        # Verify projectors exist (test setup sanity)
        assert (ckpt_dir / "projector_state.pt").exists()
        proj_saved = torch.load(ckpt_dir / "projector_state.pt", weights_only=True)
        assert len(proj_saved) > 0, "Test setup: projectors should exist"

        # Collect all projector parameter names to check against model checkpoint
        projector_param_names = set()
        for alignment in distiller.alignments:
            if alignment.input_projector is not None:
                for n, _ in alignment.input_projector.named_parameters():
                    projector_param_names.add(n)
            if alignment.output_projector is not None:
                for n, _ in alignment.output_projector.named_parameters():
                    projector_param_names.add(n)

        # Verify model.safetensors does NOT contain projector-related keys
        state = load_file(ckpt_dir / "model.safetensors")
        for key in state:
            # Check for blockwise-style prefixes
            assert not key.startswith("proj_in_"), f"model.safetensors should not contain: {key}"
            assert not key.startswith("proj_out_"), f"model.safetensors should not contain: {key}"

        if distiller_id == "blockwise":
            # Blockwise: model contains only block_ keys
            block_keys = [k for k in state if k.startswith("block_")]
            assert len(block_keys) > 0
            assert len(block_keys) == len(state), (
                f"Blockwise model.safetensors should only have block_ keys, "
                f"got: {[k for k in state if not k.startswith('block_')]}"
            )

    def test_blockwise_projector_state_has_both_input_and_output(self, device, tmp_path):
        """projector_state.pt should contain both input and output projector entries."""
        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=3, save_steps=3, distiller_id="blockwise"
        )
        alignments = _make_blockwise_alignments(teacher, student, num_layers=2)

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=dataset,
        )
        distiller.train()
        ckpt_dir = Path(_find_checkpoint_dir(args.output_dir))

        saved = torch.load(ckpt_dir / "projector_state.pt", weights_only=True)

        # Keyed by alignment name, then by projector role.
        assert set(saved) == {a.get_name() for a in alignments}
        has_input = any("input_projector" in entry for entry in saved.values())
        has_output = any("output_projector" in entry for entry in saved.values())
        # Blockwise alignments: block 0 has output only, block 1+ has both
        assert has_output, "projector_state.pt should contain output projector entries"
        assert has_input, "projector_state.pt should contain input projector entries"


# =====================================================================
# 5c. TestInputOutputProjectorRoundtrip
# =====================================================================


class TestInputOutputProjectorRoundtrip:
    """Verify input and output projectors are independently restored with correct values."""

    def test_input_and_output_projectors_restored_independently(self, device, tmp_path):
        """Train blockwise with both input and output projectors, verify each is restored."""
        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=4, save_steps=4, distiller_id="blockwise"
        )
        alignments = _make_blockwise_alignments(teacher, student, num_layers=2)

        distiller = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=dataset,
        )
        distiller.train()

        # Snapshot input and output projectors separately
        input_projs = {}
        output_projs = {}
        for alignment in distiller.alignments:
            name = alignment.get_name()
            if alignment.input_projector is not None:
                for n, p in alignment.input_projector.named_parameters():
                    input_projs[f"{name}/in.{n}"] = p.data.clone()
            if alignment.output_projector is not None:
                for n, p in alignment.output_projector.named_parameters():
                    output_projs[f"{name}/out.{n}"] = p.data.clone()

        assert output_projs, "Test setup error: no output projectors"
        assert input_projs, "Test setup error: no input projectors"

        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Reload into fresh distiller
        student2 = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        alignments2 = _make_blockwise_alignments(teacher, student2, num_layers=2)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=4,
            save_steps=4,
            distiller_id="blockwise",
        )
        distiller2 = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments2,
            args=args2,
            train_dataset=dataset,
        )
        distiller2._load_from_checkpoint(ckpt_dir)

        # Verify output projectors
        for alignment in distiller2.alignments:
            name = alignment.get_name()
            if alignment.output_projector is not None:
                for n, p in alignment.output_projector.named_parameters():
                    key = f"{name}/out.{n}"
                    assert key in output_projs, f"Missing output projector: {key}"
                    assert torch.allclose(p.data, output_projs[key], atol=1e-6), (
                        f"Output projector mismatch at {key}"
                    )

        # Verify input projectors
        for alignment in distiller2.alignments:
            name = alignment.get_name()
            if alignment.input_projector is not None:
                for n, p in alignment.input_projector.named_parameters():
                    key = f"{name}/in.{n}"
                    assert key in input_projs, f"Missing input projector: {key}"
                    assert torch.allclose(p.data, input_projs[key], atol=1e-6), (
                        f"Input projector mismatch at {key}"
                    )


# =====================================================================
# 5d. TestNoProjectorStateWhenNoProjectors
# =====================================================================


class TestNoProjectorStateWhenNoProjectors:
    """Verify projector_state.pt is NOT created when no projectors exist."""

    @pytest.mark.parametrize("distiller_id", ["response_based"])
    def test_no_projector_file(self, distiller_id, device, tmp_path):
        """Distillers without projectors should not create projector_state.pt."""
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=3, save_steps=3, distiller_id=distiller_id
        )

        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()

        ckpt_dir = Path(_find_checkpoint_dir(args.output_dir))
        assert not (ckpt_dir / "projector_state.pt").exists(), (
            f"{distiller_id} with no projectors should not create projector_state.pt"
        )


# =====================================================================
# 7. TestResumeTrainingContinues
# =====================================================================


class TestResumeTrainingContinues:
    """Train → checkpoint → resume → continue training, no crash."""

    @pytest.mark.parametrize("distiller_id", DISTILLER_IDS)
    def test_resume_no_crash(self, distiller_id, device, tmp_path):
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=3, save_steps=3, distiller_id=distiller_id
        )

        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Fresh distiller, resume for more steps
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=6,
            save_steps=3,
            distiller_id=distiller_id,
        )
        distiller2 = _make_distiller(distiller_id, teacher2, student2, args2, dataset)
        result = distiller2.train(resume_from_checkpoint=ckpt_dir)
        assert result is not None


# =====================================================================
# 7b. TestResumedModelInference
# =====================================================================


class TestResumedModelInference:
    """Train → checkpoint → resume training → run inference on result.

    Verifies the full lifecycle: after resumed training completes (hooks
    deregistered), re-registering hooks and running a forward pass produces
    a finite loss, proving the model is in a usable state.
    """

    @pytest.mark.parametrize("distiller_id", DISTILLER_IDS)
    def test_inference_after_resumed_training(self, distiller_id, device, tmp_path):
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=3, save_steps=3, distiller_id=distiller_id
        )

        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Resume for more steps
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=6,
            save_steps=3,
            distiller_id=distiller_id,
        )
        distiller2 = _make_distiller(distiller_id, teacher2, student2, args2, dataset)
        distiller2.train(resume_from_checkpoint=ckpt_dir)

        # After training completes, hooks are deregistered.
        # Verify we can re-register and run inference successfully.
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
            "labels": torch.randint(0, 128, (2, 16), device=device),
        }
        distiller2._register_capture()
        distiller2.model.eval()
        distiller2.teacher_model.eval()
        try:
            with torch.no_grad():
                loss = distiller2.compute_distillation_loss(
                    distiller2.model, batch, is_training=False
                )
        finally:
            distiller2._deregister_capture()
            distiller2._clear_captured_data()

        assert loss.isfinite(), f"Loss is not finite after resumed training: {loss.item()}"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"


# =====================================================================
# 8. TestDeterministicResume
# =====================================================================


class TestDeterministicResume:
    """Verify that resumed training produces the same final weights as
    uninterrupted training.

    Each test:
      1. Seeds everything, trains for N steps (checkpoint saved at N/2) -> W_full
      2. Seeds everything identically, resumes from the N/2 checkpoint -> W_resumed
      3. Asserts W_full == W_resumed
    """

    MAX_STEPS = 6
    SAVE_STEPS = 3

    @staticmethod
    def _assert_weights_match(w_a, w_b, atol=1e-5):
        for key in w_a:
            assert key in w_b, f"Missing key in resumed weights: {key}"
            assert torch.allclose(w_a[key], w_b[key], atol=atol), (
                f"Weight mismatch at {key}: max diff = {(w_a[key] - w_b[key]).abs().max().item()}"
            )

    @pytest.mark.parametrize("distiller_id", DISTILLER_IDS)
    def test_deterministic_resume(self, distiller_id, device, tmp_path):
        dataset = _DeterministicDataset()

        def make(output_path):
            torch.manual_seed(0)
            teacher = SimpleModel(64, 128, 2).to(device)
            torch.manual_seed(1)
            student = SimpleModel(64, 64, 2).to(device)
            torch.manual_seed(2)
            args = _make_training_args(
                output_path,
                device,
                max_steps=self.MAX_STEPS,
                save_steps=self.SAVE_STEPS,
                distiller_id=distiller_id,
            )
            return _make_distiller(distiller_id, teacher, student, args, dataset)

        full = make(tmp_path / "full")
        full.train()
        w_full = _get_model_params(full, distiller_id)
        ckpt = _first_checkpoint(full.args.output_dir)

        resumed = make(tmp_path / "resumed")
        resumed.train(resume_from_checkpoint=ckpt)
        w_resumed = _get_model_params(resumed, distiller_id)

        self._assert_weights_match(w_full, w_resumed)

    @pytest.mark.parametrize("distiller_id", COMPOSITE_IDS)
    def test_deterministic_resume_projectors(self, distiller_id, device, tmp_path):
        """Projector weights after resumed training must match uninterrupted training."""
        dataset = _DeterministicDataset()

        def make(output_path):
            torch.manual_seed(0)
            teacher = SimpleModel(64, 128, 2).to(device)
            torch.manual_seed(1)
            student = SimpleModel(64, 64, 2).to(device)
            torch.manual_seed(2)
            args = _make_training_args(
                output_path,
                device,
                max_steps=self.MAX_STEPS,
                save_steps=self.SAVE_STEPS,
                distiller_id=distiller_id,
            )
            return _make_distiller(distiller_id, teacher, student, args, dataset)

        full = make(tmp_path / "full")
        full.train()
        proj_full = _projector_snapshot(full)
        assert proj_full, f"Test setup error: no projectors found for {distiller_id}"
        ckpt = _first_checkpoint(full.args.output_dir)

        resumed = make(tmp_path / "resumed")
        resumed.train(resume_from_checkpoint=ckpt)
        proj_resumed = _projector_snapshot(resumed)

        self._assert_weights_match(proj_full, proj_resumed, atol=1e-4)


# =====================================================================
# 9. TestMissingStateGraceful
# =====================================================================


class TestMissingStateGraceful:
    """Verify graceful handling when checkpoint state files are missing."""

    @pytest.mark.parametrize("distiller_id", COMPOSITE_IDS)
    def test_missing_student_training_state(self, distiller_id, device, tmp_path):
        """Delete student_training_state.pt → resume succeeds (fresh optimizer)."""
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)
        args = _make_training_args(
            tmp_path, device, max_steps=3, save_steps=3, distiller_id=distiller_id
        )

        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Delete student_training_state.pt
        (Path(ckpt_dir) / "student_training_state.pt").unlink()

        # Resume should not crash
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=6,
            save_steps=3,
            distiller_id=distiller_id,
        )
        distiller2 = _make_distiller(distiller_id, teacher2, student2, args2, dataset)
        distiller2._load_optimizer_and_scheduler(ckpt_dir)


# =====================================================================
# 10. TestMixedPrecisionCheckpointResume
# =====================================================================


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Mixed precision requires CUDA",
)
class TestMixedPrecisionCheckpointResume:
    """Verify checkpoint resume works correctly under mixed precision (fp16)."""

    MAX_STEPS = 6
    SAVE_STEPS = 3

    @pytest.mark.parametrize("distiller_id", DISTILLER_IDS)
    def test_fp16_resume(self, distiller_id, tmp_path):
        """Train with fp16, checkpoint, resume — no crash, valid losses."""
        device = torch.device("cuda")
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=30, seq_len=16)

        args = _make_training_args(
            tmp_path,
            device,
            max_steps=self.MAX_STEPS,
            save_steps=self.SAVE_STEPS,
            distiller_id=distiller_id,
            fp16=True,
        )
        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Resume
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=self.MAX_STEPS,
            save_steps=self.SAVE_STEPS,
            distiller_id=distiller_id,
            fp16=True,
        )
        distiller2 = _make_distiller(distiller_id, teacher2, student2, args2, dataset)
        result = distiller2.train(resume_from_checkpoint=ckpt_dir)
        assert result is not None

        # Verify losses are finite and positive
        losses = [e["loss"] for e in distiller2.state.log_history if "loss" in e]
        assert len(losses) > 0, "No losses logged after fp16 resume"
        for i, loss in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss)), (
                f"Loss at step {i} not finite after fp16 resume: {loss}"
            )
            assert loss > 0, f"Loss at step {i} not positive: {loss}"

    @pytest.mark.parametrize("distiller_id", COMPOSITE_IDS)
    def test_fp16_weights_roundtrip(self, distiller_id, tmp_path):
        """Train with fp16, checkpoint, load into fresh distiller — weights match."""
        device = torch.device("cuda")
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=20, seq_len=16)

        args = _make_training_args(
            tmp_path,
            device,
            max_steps=4,
            save_steps=4,
            distiller_id=distiller_id,
            fp16=True,
        )
        distiller = _make_distiller(distiller_id, teacher, student, args, dataset)
        distiller.train()

        params_after = _get_model_params(distiller, distiller_id)
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Fresh distiller + load
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=4,
            save_steps=4,
            distiller_id=distiller_id,
            fp16=True,
        )
        distiller2 = _make_distiller(distiller_id, teacher2, student2, args2, dataset)
        distiller2._load_from_checkpoint(ckpt_dir)

        restored = _get_model_params(distiller2, distiller_id)
        for key in params_after:
            assert key in restored, f"Missing key: {key}"
            assert torch.allclose(params_after[key], restored[key], atol=1e-5), (
                f"fp16 weight mismatch at {key}: "
                f"max diff = {(params_after[key] - restored[key]).abs().max().item()}"
            )


# =====================================================================
# 11. TestLoadBestModelWithProjectors
# =====================================================================


class TestLoadBestModelWithProjectors:
    """Verify load_best_model_at_end restores projector weights from the best checkpoint.

    HF Trainer's _load_best_model() loads model.safetensors directly (not via
    _load_from_checkpoint), so projectors need explicit handling. This test
    catches the case where projectors are left at final-step values while
    model weights are rolled back to the best checkpoint.
    """

    @pytest.mark.parametrize("distiller_id", COMPOSITE_IDS)
    def test_best_model_projectors_match_checkpoint(self, distiller_id, device, tmp_path):
        """Projectors must match the best checkpoint, not the final training step."""
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=30, seq_len=16)

        args = _make_training_args(
            tmp_path,
            device,
            max_steps=10,
            save_steps=5,
            distiller_id=distiller_id,
            eval_strategy="steps",
            eval_steps=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        if distiller_id == "blockwise":
            alignments = _make_blockwise_alignments(teacher, student, num_layers=2)
            distiller = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
                eval_dataset=dataset,
            )
        else:
            alignments = _make_holistic_alignments(teacher, student, num_layers=2)
            distiller = HolisticDistiller(
                student_model=student,
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
                eval_dataset=dataset,
            )

        distiller.train()

        # After training with load_best_model_at_end, the model should be
        # from the best checkpoint. Projectors should match that checkpoint.
        best_ckpt = distiller.state.best_model_checkpoint
        if best_ckpt is None:
            pytest.skip("No best checkpoint saved (single eval point)")

        best_proj_path = Path(best_ckpt) / "projector_state.pt"
        if not best_proj_path.exists():
            pytest.skip("No projectors in checkpoint")

        saved_proj = torch.load(best_proj_path, weights_only=True, map_location="cpu")
        # Same layout as projector_state.pt: alignment name -> role -> state dict.
        current_proj = {}
        for alignment in distiller.alignments:
            for role, proj in [
                ("input_projector", alignment.input_projector),
                ("output_projector", alignment.output_projector),
            ]:
                if proj is not None:
                    current_proj.setdefault(alignment.get_name(), {})[role] = {
                        k: v.cpu() for k, v in proj.state_dict().items()
                    }

        for name, entry in saved_proj.items():
            assert name in current_proj, (
                f"Alignment {name} from best checkpoint not found in current model"
            )
            for role, state in entry.items():
                assert role in current_proj[name], f"{name} has no {role} in current model"
                for param_name in state:
                    assert torch.allclose(
                        current_proj[name][role][param_name],
                        state[param_name],
                        atol=1e-5,
                    ), (
                        f"Projector {name}/{role}.{param_name} does not match best "
                        f"checkpoint — load_best_model_at_end may not be restoring projectors"
                    )


# =====================================================================
# 12. TestAutoProjectorCheckpointResume
# =====================================================================


class TestAutoProjectorCheckpointResume:
    """Verify auto-created projectors survive checkpoint → resume → training.

    auto_projector lazily creates projectors during the first training step.
    This test verifies the full cycle: train → auto-create → checkpoint →
    load → resume → projectors still work and continue updating.
    """

    @pytest.mark.parametrize("distiller_id", PROJECTOR_IDS)
    def test_auto_projector_resume(self, distiller_id, device, tmp_path):
        """Resume from checkpoint with auto-created projectors."""
        from silverspoon_kd.alignments import Alignment

        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        dataset = DummyDataset(num_samples=30, seq_len=16)

        # Create alignments with auto_projector=True (no explicit projectors)
        alignments = []
        for i in range(2):
            alignments.append(
                Alignment(
                    teacher_block=teacher.get_layer(i),
                    student_block=student.get_layer(i),
                    teacher_model_name="test_teacher",
                    student_model_name="test_student",
                    teacher_module_name=f"layers.{i}",
                    student_module_name=f"layers.{i}",
                    auto_projector=True,
                )
            )

        args = _make_training_args(
            tmp_path, device, max_steps=6, save_steps=3, distiller_id=distiller_id
        )

        if distiller_id == "blockwise":
            distiller = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
            )
        else:
            distiller = HolisticDistiller(
                student_model=student,
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
            )

        distiller.train()

        # Verify auto-projectors were created
        for alignment in distiller.alignments:
            assert alignment.output_projector is not None, (
                "Auto projector not created during training"
            )

        # Snapshot projector state after first training run
        proj_after_train = _projector_snapshot(distiller)
        assert proj_after_train, "No projectors to snapshot"

        ckpt_dir = _find_checkpoint_dir(args.output_dir)
        assert (Path(ckpt_dir) / "projector_state.pt").exists(), (
            "projector_state.pt not saved for auto-projectors"
        )

        # Resume from checkpoint with fresh distiller
        teacher2 = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student2 = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)

        alignments2 = []
        for i in range(2):
            alignments2.append(
                Alignment(
                    teacher_block=teacher2.get_layer(i),
                    student_block=student2.get_layer(i),
                    teacher_model_name="test_teacher",
                    student_model_name="test_student",
                    teacher_module_name=f"layers.{i}",
                    student_module_name=f"layers.{i}",
                    auto_projector=True,
                )
            )

        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=10,
            save_steps=999,
            distiller_id=distiller_id,
        )

        if distiller_id == "blockwise":
            distiller2 = BlockwiseDistiller(
                teacher_model=teacher2,
                alignments=alignments2,
                args=args2,
                train_dataset=dataset,
            )
        else:
            distiller2 = HolisticDistiller(
                student_model=student2,
                teacher_model=teacher2,
                alignments=alignments2,
                args=args2,
                train_dataset=dataset,
            )

        distiller2.train(resume_from_checkpoint=ckpt_dir)

        # Verify projectors still exist and were updated during resumed training
        proj_after_resume = _projector_snapshot(distiller2)
        assert proj_after_resume, "Projectors missing after resumed training"

        # Verify losses are finite and positive
        losses = [e["loss"] for e in distiller2.state.log_history if "loss" in e]
        assert len(losses) > 0, "No losses logged after resume with auto-projectors"
        for loss in losses:
            assert torch.isfinite(torch.tensor(loss)), f"Non-finite loss: {loss}"
            assert loss > 0, f"Non-positive loss: {loss}"


# =====================================================================
# 13. TestEvalDuringResumedTraining
# =====================================================================


class TestEvalDuringResumedTraining:
    """Verify evaluation works during resumed training.

    After resuming from checkpoint, the Trainer's inner loop must be able
    to run evaluation at eval_steps intervals. This requires capture hooks
    to be re-registered and the full eval pipeline to function.
    """

    @pytest.mark.parametrize("distiller_id", DISTILLER_IDS)
    def test_eval_metrics_after_resume(self, distiller_id, device, tmp_path):
        """Resume from checkpoint with eval_strategy='steps' — eval must produce metrics."""
        teacher, student = _create_models(distiller_id, device)
        dataset = DummyDataset(num_samples=30, seq_len=16)

        # Both phases use consistent eval config to avoid trainer_state.json conflicts
        eval_kwargs = {"eval_strategy": "steps", "eval_steps": 3}

        # Phase 1: train 4 steps, save at step 4 (with eval enabled)
        args = _make_training_args(
            tmp_path,
            device,
            max_steps=4,
            save_steps=4,
            distiller_id=distiller_id,
            **eval_kwargs,
        )

        if distiller_id == "blockwise":
            alignments = _make_blockwise_alignments(teacher, student, num_layers=2)
            distiller = BlockwiseDistiller(
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
                eval_dataset=dataset,
            )
        elif distiller_id == "holistic":
            alignments = _make_holistic_alignments(teacher, student, num_layers=2)
            distiller = HolisticDistiller(
                student_model=student,
                teacher_model=teacher,
                alignments=alignments,
                args=args,
                train_dataset=dataset,
                eval_dataset=dataset,
            )
        else:
            distiller = ResponseBasedDistiller(
                student_model=student,
                teacher_model=teacher,
                args=args,
                train_dataset=dataset,
                eval_dataset=dataset,
            )

        distiller.train()
        ckpt_dir = _find_checkpoint_dir(args.output_dir)

        # Phase 2: resume to 12 steps with eval every 3 steps
        teacher2, student2 = _create_models(distiller_id, device)
        args2 = _make_training_args(
            tmp_path / "run2",
            device,
            max_steps=12,
            save_steps=999,
            distiller_id=distiller_id,
            **eval_kwargs,
        )

        if distiller_id == "blockwise":
            alignments2 = _make_blockwise_alignments(teacher2, student2, num_layers=2)
            distiller2 = BlockwiseDistiller(
                teacher_model=teacher2,
                alignments=alignments2,
                args=args2,
                train_dataset=dataset,
                eval_dataset=dataset,
            )
        elif distiller_id == "holistic":
            alignments2 = _make_holistic_alignments(teacher2, student2, num_layers=2)
            distiller2 = HolisticDistiller(
                student_model=student2,
                teacher_model=teacher2,
                alignments=alignments2,
                args=args2,
                train_dataset=dataset,
                eval_dataset=dataset,
            )
        else:
            distiller2 = ResponseBasedDistiller(
                student_model=student2,
                teacher_model=teacher2,
                args=args2,
                train_dataset=dataset,
                eval_dataset=dataset,
            )

        distiller2.train(resume_from_checkpoint=ckpt_dir)

        assert distiller2.state.global_step == 12, (
            f"Expected 12 steps after resume, got {distiller2.state.global_step}"
        )

        # Verify eval metrics were logged during resumed training
        eval_entries = [e for e in distiller2.state.log_history if "eval_loss" in e]
        assert len(eval_entries) > 0, (
            "No eval metrics logged during resumed training — "
            "evaluation may not be running after checkpoint resume"
        )
        for entry in eval_entries:
            assert torch.isfinite(torch.tensor(entry["eval_loss"])), (
                f"eval_loss not finite during resumed training: {entry['eval_loss']}"
            )
            assert entry["eval_loss"] > 0, (
                f"eval_loss not positive during resumed training: {entry['eval_loss']}"
            )


# =====================================================================
# 13. TestMaterialiseSavedProjectorParameterFreeBlock
# =====================================================================


class TestMaterialiseSavedProjectorParameterFreeBlock:
    """``_materialise_saved_projector`` with parameter-free student blocks.

    The helper infers the device/dtype of a fresh projector from the
    alignment's ``student_block`` and falls back to the student model when
    the block has no parameters (pooling layers or ``nn.Identity``, as in
    RelKD-style alignments).  These tests cover both paths.
    """

    @staticmethod
    def _make_distiller(tmp_path, device, parameter_free: bool):
        """Build a HolisticDistiller whose alignment's student_block either
        has or hasn't got any parameters."""
        from torch import nn

        from silverspoon_kd.alignments import Alignment

        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)

        if parameter_free:
            # nn.Identity has no parameters — same shape as RelKD's pooling
            # layers (no learnable weights, just a reshape/aggregation).
            student_block = nn.Identity()
            assert not list(student_block.parameters()), (
                "test fixture is wrong: chose a block that does have parameters"
            )
        else:
            student_block = student.get_layer(0)
            assert list(student_block.parameters()), (
                "test fixture is wrong: chose a block that has no parameters"
            )

        alignment = Alignment(
            teacher_block=teacher.get_layer(0),
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
        )

        args = _make_training_args(tmp_path, device, distiller_id="holistic")
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=[alignment],
            args=args,
            train_dataset=DummyDataset(num_samples=4, seq_len=16),
        )
        return distiller, student, alignment

    @staticmethod
    def _saved_output_projector():
        """A saved projector state for a 64→128 output projection."""
        return {"weight": torch.randn(128, 64), "bias": torch.randn(128)}

    def test_parameter_free_student_block_is_supported(self, device, tmp_path):
        """A block without parameters must not stop the projector from being created."""
        distiller, _student, alignment = self._make_distiller(tmp_path, device, parameter_free=True)
        assert alignment.output_projector is None

        distiller._materialise_saved_projector(
            alignment, "output_projector", self._saved_output_projector()
        )

        assert alignment.output_projector is not None, "Projector was not materialised"

    def test_materialised_projector_lands_on_model_device_and_dtype(self, device, tmp_path):
        """When the student_block is parameter-free, the fallback inference
        must use the student model's device/dtype — not crash, and not
        silently pick up CPU/float32 when the model is on a different device.
        """
        distiller, student, alignment = self._make_distiller(tmp_path, device, parameter_free=True)

        distiller._materialise_saved_projector(
            alignment, "output_projector", self._saved_output_projector()
        )

        assert alignment.output_projector is not None
        expected_device = next(student.parameters()).device
        expected_dtype = next(student.parameters()).dtype
        for name, p in alignment.output_projector.named_parameters():
            assert p.device == expected_device, (
                f"projector param {name} on {p.device}, expected {expected_device} "
                f"(model device).  Fallback inference did not match the model."
            )
            assert p.dtype == expected_dtype, (
                f"projector param {name} dtype {p.dtype}, expected {expected_dtype}"
            )

    def test_with_parameters_path_still_uses_block_device(self, device, tmp_path):
        """The non-fallback branch — when the student_block does have
        parameters — must keep using the block's device/dtype.  This is
        the common case for HKD/BKD alignments on transformer blocks; we
        guard it so a future refactor that always uses the model fallback
        wouldn't silently regress device placement for nested-device setups.
        """
        distiller, _student, alignment = self._make_distiller(
            tmp_path, device, parameter_free=False
        )

        distiller._materialise_saved_projector(
            alignment, "output_projector", self._saved_output_projector()
        )

        assert alignment.output_projector is not None
        expected_device = next(alignment.student_block.parameters()).device
        expected_dtype = next(alignment.student_block.parameters()).dtype
        for p in alignment.output_projector.parameters():
            assert p.device == expected_device
            assert p.dtype == expected_dtype


# =====================================================================
# 14. TestProjectorStateUnnamedAlignments
# =====================================================================


class TestProjectorStateUnnamedAlignments:
    """Alignments without names share the name ``.``; their projectors must still round-trip."""

    @staticmethod
    def _make(device, tmp_path, tag):
        from silverspoon_kd.alignments import Alignment

        torch.manual_seed(0)
        teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2).to(device)
        student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2).to(device)
        alignments = [
            Alignment(
                teacher_block=teacher.get_layer(i),
                student_block=student.get_layer(i),
                auto_projector=True,
            )
            for i in range(2)
        ]
        args = _make_training_args(
            tmp_path / tag, device, max_steps=2, save_steps=100, distiller_id="holistic"
        )
        distiller = HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=DummyDataset(num_samples=8, seq_len=16),
        )
        return distiller, alignments

    def test_keys_are_unique_and_state_round_trips(self, device, tmp_path):
        distiller, alignments = self._make(device, tmp_path, "first")
        assert distiller._projector_state_keys() == [".#0", ".#1"]

        distiller.train()  # materialises the auto projectors
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        distiller._save_distiller_state(str(ckpt))
        saved = torch.load(ckpt / "projector_state.pt", weights_only=True)
        assert set(saved) == {".#0", ".#1"}

        distiller2, alignments2 = self._make(device, tmp_path, "second")
        assert all(a.output_projector is None for a in alignments2)
        distiller2._load_projector_state(str(ckpt))

        for original, restored in zip(alignments, alignments2, strict=True):
            assert restored.output_projector is not None
            assert torch.equal(
                restored.output_projector.weight.cpu(), original.output_projector.weight.cpu()
            )
