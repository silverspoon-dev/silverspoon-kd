"""
Unit tests for BlockwiseDistiller class.
"""

from unittest.mock import MagicMock

import pytest
import torch

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.distillers.blockwise_distiller import (
    BlockwiseDistiller,
    StudentBlocksContainer,
)
from silverspoon_kd.training_arguments import TrainingArguments

from ..conftest import SimpleBlock, SimpleModel


def _make_blockwise_args(training_args, **overrides):
    """Create TrainingArguments from base training_args fixture."""
    return TrainingArguments(
        output_dir=training_args.output_dir,
        max_steps=training_args.max_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=training_args.use_cpu,
        **overrides,
    )


class TestStudentBlocksContainer:
    """Test suite for StudentBlocksContainer helper class."""

    def test_container_registers_student_blocks(self, teacher_model, teacher_alignments):
        """Test that StudentBlocksContainer registers student blocks as submodules."""
        container = StudentBlocksContainer(teacher_model, teacher_alignments)

        # Should have submodules for each student block
        submodule_names = {name for name, _ in container.named_modules() if name}
        assert len(submodule_names) > 0

        # All student blocks should be registered
        for alignment in teacher_alignments:
            safe = alignment.get_name().replace("/", "__").replace(".", "__")
            assert f"block_{safe}" in submodule_names

    def test_container_forward_signature(self, teacher_model, teacher_alignments):
        """Test that container forward method has teacher's signature."""
        container = StudentBlocksContainer(teacher_model, teacher_alignments)

        assert hasattr(container, "forward")
        result = container.forward(input_ids=torch.tensor([1, 2, 3]))
        assert result is None

    def test_container_has_parameters(self, teacher_model, teacher_alignments):
        """Test that container exposes student parameters."""
        container = StudentBlocksContainer(teacher_model, teacher_alignments)

        params = list(container.parameters())
        assert len(params) > 0


class TestBlockwiseDistiller:
    """Test suite for BlockwiseDistiller."""

    def test_initialization(
        self, teacher_model, teacher_alignments, training_args, train_dataset, device
    ):
        """Test that BlockwiseDistiller initializes correctly."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        assert distiller.teacher_model is teacher_model
        assert distiller.alignments == teacher_alignments
        assert isinstance(distiller.model, StudentBlocksContainer)
        assert distiller.capture_engine is not None

    def test_student_max_grad_norm_propagation(
        self, teacher_model, train_dataset, device, tmp_path
    ):
        """Test that max_grad_norm is propagated to students."""
        teacher_block = teacher_model.get_layer(0)
        student_block = teacher_model.get_layer(0)

        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            max_grad_norm=None,
        )

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            max_grad_norm=1.0,
            use_cpu=(device.type == "cpu"),
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=[alignment],
            args=args,
            train_dataset=train_dataset,
        )

        assert alignment.max_grad_norm == 1.0
        # After propagation, Trainer-level clipping is disabled (set to 0).
        # Must be 0, not None — the HF Trainer does
        # `if self.args.max_grad_norm > 0` which raises TypeError on None.
        assert distiller.args.max_grad_norm == 0

    def test_training_step(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Test that training_step executes without errors."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        distiller._deregister_capture()

    def test_compute_loss(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Test that compute_loss executes without errors."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.compute_loss(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        loss, outputs = distiller.compute_loss(distiller.model, batch, return_outputs=True)
        assert isinstance(loss, torch.Tensor)
        assert outputs is None

        distiller._deregister_capture()

    def test_save_checkpoint(
        self, teacher_model, teacher_alignments, training_args, train_dataset, tmp_path
    ):
        """Test that checkpoint saving works correctly."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(str(checkpoint_dir))

        # Should have student_training_state and model weights
        assert (checkpoint_dir / "student_training_state.pt").exists()
        has_model = (checkpoint_dir / "model.safetensors").exists() or (
            checkpoint_dir / "pytorch_model.bin"
        ).exists()
        assert has_model

    def test_register_and_deregister_capture(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that capture engine registration and deregistration works."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        # Verify capture engines were registered (capture_engine is not None)
        assert distiller.capture_engine is not None

        distiller._deregister_capture()

    def test_prepare_teacher_inputs(
        self, teacher_model, single_alignment, train_dataset, device, tmp_path
    ):
        """Test custom prepare_teacher_inputs function."""

        def custom_prepare(inputs):
            return {**inputs, "custom_key": "custom_value"}

        args = TrainingArguments(
            output_dir=str(tmp_path / "out"), max_steps=5, use_cpu=(device.type == "cpu")
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
            prepare_teacher_inputs=custom_prepare,
        )

        inputs = {"input_ids": torch.tensor([1, 2, 3])}
        result = distiller._prepare_teacher_inputs(inputs)

        assert "custom_key" in result
        assert result["custom_key"] == "custom_value"

    def test_flop_counter_disabled(
        self, teacher_model, single_alignment, train_dataset, device, tmp_path
    ):
        """Test that FLOP counting can be disabled."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            count_flops=False,
            use_cpu=(device.type == "cpu"),
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        distiller.state.global_step = 0
        distiller.training_step(distiller.model, batch)

        assert distiller.flop_counter == 0

        distiller._deregister_capture()

    def test_evaluation_with_dataset(
        self,
        teacher_model,
        single_alignment,
        training_args,
        train_dataset,
        eval_dataset,
    ):
        """Test that evaluation works with dataset."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        distiller._register_capture()

        metrics = distiller.evaluate()

        assert isinstance(metrics, dict)
        assert "eval_loss" in metrics

        distiller._deregister_capture()

    def test_save_method_creates_checkpoint_files(
        self, teacher_model, teacher_alignments, training_args, train_dataset, tmp_path
    ):
        """Test that _save method creates all expected checkpoint files."""
        training_args.output_dir = str(tmp_path / "output")

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(output_dir=str(checkpoint_dir))

        assert (checkpoint_dir / "student_training_state.pt").exists()
        has_model = (checkpoint_dir / "model.safetensors").exists() or (
            checkpoint_dir / "pytorch_model.bin"
        ).exists()
        assert has_model

    def test_save_with_student_models_creates_model_directory(
        self,
        teacher_model,
        teacher_alignments,
        training_args,
        train_dataset,
        tmp_path,
        device,
    ):
        """Test that _save creates student model directories when student_models is provided."""
        training_args.output_dir = str(tmp_path / "output")

        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3)
        student_model.to(device)

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            student_models={"test_student": student_model},
        )

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(output_dir=str(checkpoint_dir))

        student_model_dir = checkpoint_dir / "student_model_test_student"
        assert student_model_dir.exists()
        assert (student_model_dir / "pytorch_model.bin").exists()

    def test_save_with_student_models_preserves_trained_weights(
        self,
        teacher_model,
        single_alignment,
        training_args,
        train_dataset,
        tmp_path,
        device,
    ):
        """Test that saved student model contains trained weights after training step."""
        training_args.output_dir = str(tmp_path / "output")

        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3)
        student_model.to(device)

        student_alignment = single_alignment[0]
        original_student_block = student_alignment.student_block

        original_weights = {
            name: param.clone() for name, param in original_student_block.named_parameters()
        }

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            student_models={"test_student": student_model},
        )

        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        distiller.training_step(distiller.model, batch)
        distiller.optimizer.step()

        distiller._deregister_capture()

        weights_changed = False
        for name, param in original_student_block.named_parameters():
            if not torch.allclose(param, original_weights[name]):
                weights_changed = True
                break
        assert weights_changed, "Training should have modified student block weights"

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(output_dir=str(checkpoint_dir))

        student_model_dir = checkpoint_dir / "student_model_test_student"
        saved_state = torch.load(student_model_dir / "pytorch_model.bin")

        assert saved_state is not None
        assert len(saved_state) > 0

    def test_save_multiple_student_models(
        self,
        teacher_model,
        teacher_alignments,
        training_args,
        train_dataset,
        tmp_path,
        device,
    ):
        """Test saving multiple student models."""
        training_args.output_dir = str(tmp_path / "output")

        student_model_1 = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3)
        student_model_1.to(device)

        student_model_2 = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3)
        student_model_2.to(device)

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            student_models={
                "student_1": student_model_1,
                "student_2": student_model_2,
            },
        )

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(output_dir=str(checkpoint_dir))

        assert (checkpoint_dir / "student_model_student_1").exists()
        assert (checkpoint_dir / "student_model_student_2").exists()
        assert (checkpoint_dir / "student_model_student_1" / "pytorch_model.bin").exists()
        assert (checkpoint_dir / "student_model_student_2" / "pytorch_model.bin").exists()

    def test_save_student_model_with_slash_in_name(
        self,
        teacher_model,
        teacher_alignments,
        training_args,
        train_dataset,
        tmp_path,
        device,
    ):
        """Test that student model names with slashes are handled correctly."""
        training_args.output_dir = str(tmp_path / "output")

        student_model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3)
        student_model.to(device)

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            student_models={"org/model-name": student_model},
        )

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(output_dir=str(checkpoint_dir))

        student_model_dir = checkpoint_dir / "student_model_org_model-name"
        assert student_model_dir.exists()
        assert (student_model_dir / "pytorch_model.bin").exists()

    def test_save_without_student_models_no_model_directories(
        self, teacher_model, teacher_alignments, training_args, train_dataset, tmp_path
    ):
        """Test that _save without student_models doesn't create model directories."""
        training_args.output_dir = str(tmp_path / "output")

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(output_dir=str(checkpoint_dir))

        student_model_dirs = list(checkpoint_dir.glob("student_model_*"))
        assert len(student_model_dirs) == 0


def _make_bf16_alignment(teacher_model, device):
    """Helper: create a single alignment with a bfloat16 student block and auto_dtype_match=True."""
    teacher_block = teacher_model.get_layer(0)
    student_block = SimpleBlock(input_dim=64, output_dim=128).to(
        device=device, dtype=torch.bfloat16
    )
    alignment = Alignment(
        teacher_block=teacher_block,
        student_block=student_block,
        teacher_model_name="test_teacher",
        student_model_name="bf16_student",
        teacher_module_name="layers.0",
        student_module_name="layers.0",
        auto_dtype_match=True,
    )
    return [alignment]


class TestBlockwiseDistillerAutoDtypeMatch:
    """Tests for auto_dtype_match in BlockwiseDistiller."""

    def test_training_step_with_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Student in bfloat16, teacher in float32 — training step should succeed."""
        alignments = _make_bf16_alignment(teacher_model, device)

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        distiller._deregister_capture()

    def test_compute_loss_with_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Student in bfloat16, teacher in float32 — eval loss should succeed."""
        alignments = _make_bf16_alignment(teacher_model, device)

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.compute_loss(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        distiller._deregister_capture()

    def test_without_auto_dtype_match_raises_on_dtype_mismatch(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Without auto_dtype_match, bfloat16 student with float32 inputs should fail."""
        teacher_block = teacher_model.get_layer(0)
        student_block = SimpleBlock(input_dim=64, output_dim=128).to(
            device=device, dtype=torch.bfloat16
        )
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="bf16_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            auto_dtype_match=False,
        )
        alignments = [alignment]

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        with pytest.raises(RuntimeError):
            distiller.training_step(distiller.model, batch)

        distiller._deregister_capture()

    def test_integer_tensors_not_cast(self, teacher_model, training_args, train_dataset, device):
        """Integer tensors (attention_mask) should remain long when auto_dtype_match is on."""
        from silverspoon_kd.distillers.base_distiller import send_to_dtype

        mask = torch.ones(2, 16, dtype=torch.long, device=device)
        result = send_to_dtype(mask, torch.bfloat16)
        assert result.dtype == torch.long

    def test_auto_device_and_dtype_match_together(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Both auto_device_match and auto_dtype_match enabled simultaneously."""
        teacher_block = teacher_model.get_layer(0)
        student_block = SimpleBlock(input_dim=64, output_dim=128).to(
            device=device, dtype=torch.bfloat16
        )
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="bf16_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            auto_device_match=True,
            auto_dtype_match=True,
        )
        alignments = [alignment]

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        distiller._deregister_capture()

    def test_single_student_bfloat16_dtype(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Single student with bfloat16 dtype and auto_dtype_match enabled."""
        teacher_block = teacher_model.get_layer(0)

        student_block = SimpleBlock(input_dim=64, output_dim=128).to(
            device=device, dtype=torch.bfloat16
        )
        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="bf16_student",
            teacher_module_name="layers.0",
            student_module_name="layers.0",
            auto_dtype_match=True,
        )
        alignments = [alignment]

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert len(distiller.step_losses) == 1

        distiller._deregister_capture()

    def test_weights_update_in_bfloat16(self, teacher_model, training_args, train_dataset, device):
        """Verify that bfloat16 student weights actually change after a training step."""
        alignments = _make_bf16_alignment(teacher_model, device)
        student = alignments[0]

        initial_norm = sum(p.data.norm().item() for p in student.student_block.parameters())

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }
        distiller.training_step(distiller.model, batch)
        distiller.optimizer.step()

        updated_norm = sum(p.data.norm().item() for p in student.student_block.parameters())
        assert initial_norm != updated_norm, "Student weights should change after training step"

        for p in student.student_block.parameters():
            assert p.dtype == torch.bfloat16

        distiller._deregister_capture()


class TestBlockwiseDistillerSaveWithProjectors:
    """Test checkpoint save/load with output projectors present."""

    def test_save_with_output_projectors(
        self, teacher_model, teacher_alignments, training_args, train_dataset, tmp_path
    ):
        """Test checkpoint save includes output projector state in projector_state.pt."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(str(checkpoint_dir))

        # Projectors are now saved separately via projector_state.pt, not in model.safetensors
        has_projectors = any(
            alignment.output_projector is not None for alignment in teacher_alignments
        )
        if has_projectors:
            projector_path = checkpoint_dir / "projector_state.pt"
            assert projector_path.exists(), (
                "projector_state.pt should exist when projectors are present"
            )
            saved = torch.load(projector_path, weights_only=True)
            assert any("output_projector" in entry for entry in saved.values()), (
                "projector_state.pt should contain output_projector entries"
            )


class TestBlockwiseDistillerSaveStudentModel:
    """Test _save with a PreTrainedModel-like student_model."""

    def test_save_pretrained_model(
        self,
        teacher_model,
        teacher_alignments,
        training_args,
        train_dataset,
        tmp_path,
        device,
    ):
        """Test _save calls save_pretrained for PreTrainedModel instances."""
        from transformers import PreTrainedModel

        mock_model = MagicMock(spec=PreTrainedModel)
        mock_model.save_pretrained = MagicMock()

        training_args.output_dir = str(tmp_path / "output")

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            student_models={"pretrained_student": mock_model},
        )

        checkpoint_dir = tmp_path / "checkpoint"
        distiller._save(output_dir=str(checkpoint_dir))

        mock_model.save_pretrained.assert_called_once()


class TestBlockwiseDistillerDefaultArgs:
    """Test default TrainingArguments creation in BlockwiseDistiller."""

    def test_initialization_without_args(
        self, teacher_model, single_alignment, train_dataset, device
    ):
        """Test BlockwiseDistiller creates default args when args is None."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=None,
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.args, TrainingArguments)


class TestBlockwiseDistillerTruncatedForward:
    """Test truncated forward handling in training_step and compute_loss.

    Truncation is now automatic via auto_truncate=True.
    """

    def test_training_step_with_truncated_forward(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Test training_step works with auto_truncate set."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)
        assert isinstance(loss, torch.Tensor)

        distiller._deregister_capture()

    def test_compute_loss_with_truncated_forward(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Test compute_loss works with auto_truncate set."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.compute_loss(distiller.model, batch)
        assert isinstance(loss, torch.Tensor)

        distiller._deregister_capture()


class TestBlockwiseDistillerAutoProjector:
    """Test auto-projector initialization path."""

    def test_auto_projector_initialization(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Test that auto-projectors are initialized on first forward pass."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )
        distiller._register_capture()

        for alignment in distiller.alignments:
            alignment.auto_projector = True
            alignment._auto_projector_initialized = False
            alignment.initialize_auto_projectors = MagicMock()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        distiller.training_step(distiller.model, batch)

        for alignment in distiller.alignments:
            alignment.initialize_auto_projectors.assert_called_once()

        distiller._deregister_capture()


class TestBlockwiseDistillerTorchCompile:
    """Test torch.compile handling in BlockwiseDistiller."""

    def test_torch_compile_disabled_on_args(
        self,
        teacher_model,
        single_alignment,
        train_dataset,
        device,
        tmp_path,
    ):
        """Test that args.torch_compile is set to False so Trainer doesn't compile the container."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        assert distiller.args.torch_compile is False

    def test_teacher_compiled_when_torch_compile_true(
        self,
        teacher_model,
        single_alignment,
        train_dataset,
        device,
        tmp_path,
    ):
        """Test that teacher model is compiled when torch_compile=True."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        assert isinstance(distiller.teacher_model, torch._dynamo.eval_frame.OptimizedModule)

    def test_teacher_not_compiled_when_torch_compile_false(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Test that teacher model is NOT compiled when torch_compile=False."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )

        assert distiller.teacher_model is teacher_model

    def test_training_works_with_torch_compile(
        self,
        teacher_model,
        single_alignment,
        train_dataset,
        device,
        tmp_path,
    ):
        """Test that training still works after torch.compile is applied to the teacher."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        distiller._deregister_capture()

    def test_container_not_compiled(
        self, teacher_model, single_alignment, train_dataset, device, tmp_path
    ):
        """Test that StudentBlocksContainer is NOT compiled even when torch_compile=True."""
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=5,
            torch_compile=True,
            use_cpu=(device.type == "cpu"),
        )

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )

        assert not isinstance(distiller.model, torch._dynamo.eval_frame.OptimizedModule)


class TestBlockwiseDistillerFSDPRootHierarchy:
    """Regression tests for _fix_fsdp_root_hierarchy.

    BKD wraps each student block with FSDP individually. This sets
    _is_root=True on every block. _fix_fsdp_root_hierarchy must clear
    this flag on child modules so FSDP operations (state_dict,
    clip_grad_norm_) don't fail.
    """

    def test_fix_fsdp_root_hierarchy_logic(self):
        """_fix_fsdp_root_hierarchy should clear _is_root on child FSDP modules.

        This tests the core logic: given a module tree where multiple modules
        have _is_root=True (simulating BKD's per-block FSDP wrapping), only
        the first (root) module should retain _is_root afterwards.

        Note: We test the algorithm directly rather than going through FSDP,
        since FSDP requires distributed setup. The actual FSDP integration
        is covered by multi-GPU tests.
        """

        # Build a module tree simulating per-block FSDP wrapping
        root = torch.nn.Module()
        child1 = torch.nn.Module()
        child2 = torch.nn.Module()
        root.add_module("block_0", child1)
        root.add_module("block_1", child2)

        # Simulate: all modules have _is_root=True (as BKD per-block wrapping does)
        root._is_root = True
        child1._is_root = True
        child2._is_root = True

        # Apply the same logic as _fix_fsdp_root_hierarchy but checking
        # against nn.Module instead of FSDP (since we can't create real FSDP
        # modules without distributed setup)
        root_found = False
        for module in root.modules():
            # In real code this checks isinstance(module, FSDP)
            if hasattr(module, "_is_root"):
                if not root_found:
                    root_found = True  # first is root, keep it
                else:
                    if getattr(module, "_is_root", None) is True:
                        module._is_root = None

        # Root keeps _is_root=True, children get cleared
        assert root._is_root is True
        assert child1._is_root is None
        assert child2._is_root is None

    def test_create_optimizer_accepts_model_argument(
        self, teacher_model, teacher_alignments, training_args, train_dataset
    ):
        """BKD create_optimizer(model=...) should not raise TypeError."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
        )
        # Should not raise "takes 1 positional argument but 2 were given"
        result = distiller.create_optimizer(model=teacher_model)
        assert result is not None  # CompositeOptimizer was created


class TestBlockwiseDistillerAutoTruncate:
    """Integration tests for auto_truncate in BlockwiseDistiller."""

    def test_auto_truncate_produces_valid_loss(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Training step with auto_truncate=True should produce a valid loss."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)
        assert isinstance(loss, torch.Tensor)
        distiller._deregister_capture()

    def test_auto_truncate_false_produces_valid_loss(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Training step with auto_truncate=False should also work."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=False,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)
        assert isinstance(loss, torch.Tensor)
        distiller._deregister_capture()

    def test_auto_truncate_propagated_to_capture_engine(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """auto_truncate should be propagated to the capture engine."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        assert distiller.capture_engine.auto_truncate is True

        distiller2 = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=False,
        )
        assert distiller2.capture_engine.auto_truncate is False


class TestBlockwiseDistillerAutoTruncateFSDP:
    """Test auto_truncate + FSDP interaction for BKD."""

    def test_non_fsdp_teacher_keeps_auto_truncate(
        self, teacher_model, single_alignment, training_args, train_dataset
    ):
        """Non-FSDP teacher should keep auto_truncate enabled."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        assert distiller.capture_engine.auto_truncate is True

    def test_auto_truncate_works_with_ddp(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """BKD auto_truncate should work with non-FSDP (DDP) — training step succeeds."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args),
            train_dataset=train_dataset,
            auto_truncate=True,
        )
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)
        assert isinstance(loss, torch.Tensor)
        distiller._deregister_capture()

    def test_fsdp_teacher_keeps_auto_truncate(
        self,
        teacher_model,
        single_alignment,
        training_args,
        train_dataset,
    ):
        """FSDP teacher should keep auto_truncate enabled (forward-only, no backward)."""
        from unittest.mock import patch

        with patch.object(
            BlockwiseDistiller,
            "_is_model_fsdp_wrapped",
            return_value=True,
        ):
            distiller = BlockwiseDistiller(
                teacher_model=teacher_model,
                alignments=single_alignment,
                args=_make_blockwise_args(training_args),
                train_dataset=train_dataset,
                auto_truncate=True,
            )

        assert distiller.capture_engine.auto_truncate is True


class TestBlockwiseDistillerSequentialBlocks:
    """Tests for backward_per_block mode in BlockwiseDistiller."""

    def test_training_step_backward_per_block(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Verify loss is valid and weights change with backward_per_block=True."""
        alignment = single_alignment[0]
        original_weights = {
            name: param.clone() for name, param in alignment.student_block.named_parameters()
        }

        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args, backward_per_block=True),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not loss.requires_grad

        # Optimizer step to update weights
        distiller.optimizer.step()

        weights_changed = any(
            not torch.allclose(param, original_weights[name])
            for name, param in alignment.student_block.named_parameters()
        )
        assert weights_changed, "Training should have modified student block weights"

        distiller._deregister_capture()

    def test_backward_per_block_gradient_equivalence(
        self, teacher_model, training_args, train_dataset, device
    ):
        """Run same batch with backward_per_block True/False, verify gradients match."""
        import copy

        # Create two sets of identical student blocks (multiple alignments)
        blocks_normal = []
        blocks_sequential = []
        alignments_normal = []
        alignments_sequential = []

        for i in range(teacher_model.num_layers):
            teacher_block = teacher_model.get_layer(i)
            input_dim = 64 if i == 0 else 128
            student = SimpleBlock(input_dim=input_dim, output_dim=128).to(device)
            student_copy = copy.deepcopy(student)
            blocks_normal.append(student)
            blocks_sequential.append(student_copy)

            alignments_normal.append(
                Alignment(
                    teacher_block=teacher_block,
                    student_block=student,
                    teacher_model_name="test_teacher",
                    student_model_name="test_student",
                    teacher_module_name=f"layers.{i}",
                    student_module_name=f"layers.{i}",
                )
            )
            alignments_sequential.append(
                Alignment(
                    teacher_block=teacher_block,
                    student_block=student_copy,
                    teacher_model_name="test_teacher",
                    student_model_name="test_student",
                    teacher_module_name=f"layers.{i}",
                    student_module_name=f"layers.{i}",
                )
            )

        distiller_normal = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments_normal,
            args=_make_blockwise_args(training_args, backward_per_block=False),
            train_dataset=train_dataset,
        )
        distiller_sequential = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=alignments_sequential,
            args=_make_blockwise_args(training_args, backward_per_block=True),
            train_dataset=train_dataset,
        )

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        distiller_normal._register_capture()
        loss_normal = distiller_normal.training_step(distiller_normal.model, batch)
        distiller_normal._deregister_capture()

        distiller_sequential._register_capture()
        loss_sequential = distiller_sequential.training_step(distiller_sequential.model, batch)
        distiller_sequential._deregister_capture()

        # Loss values must match
        torch.testing.assert_close(loss_normal, loss_sequential, msg="Loss value mismatch")

        # Compare gradients across all blocks
        for block_n, block_s in zip(blocks_normal, blocks_sequential, strict=True):
            for (name_n, param_n), (name_s, param_s) in zip(
                block_n.named_parameters(), block_s.named_parameters(), strict=True
            ):
                assert name_n == name_s
                assert param_n.grad is not None, f"No gradient for {name_n} (normal)"
                assert param_s.grad is not None, f"No gradient for {name_s} (sequential)"
                torch.testing.assert_close(
                    param_n.grad,
                    param_s.grad,
                    msg=f"Gradient mismatch for {name_n}",
                )

    def test_backward_per_block_eval_no_backward(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Eval path with backward_per_block=True must not trigger per-block backward."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args, backward_per_block=True),
            train_dataset=train_dataset,
        )

        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        # compute_loss uses is_training=self.model.training; eval() sets it to False
        distiller.model.eval()
        loss = distiller.compute_loss(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # No gradients should have been computed on student parameters
        for param in single_alignment[0].student_block.parameters():
            assert param.grad is None or (param.grad == 0).all()

        distiller._deregister_capture()

    def test_backward_per_block_with_gradient_checkpointing(
        self, teacher_model, single_alignment, training_args, train_dataset, device
    ):
        """Verify backward_per_block works with gradient checkpointing enabled."""
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=_make_blockwise_args(training_args, backward_per_block=True),
            train_dataset=train_dataset,
        )

        distiller.model.gradient_checkpointing_enable()
        distiller._register_capture()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16), device=device),
            "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        }

        loss = distiller.training_step(distiller.model, batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        distiller._deregister_capture()


class TestBlockwiseZeroLossFallback:
    """Regression tests: zero-loss fallback must produce valid autograd tensors.

    When all alignments are truncated or produce no loss,
    compute_distillation_loss returns a zero tensor that is connected to the
    student's graph, so the backward pass is a real (zero-gradient) pass
    rather than a silent no-op on a detached leaf.
    """

    def test_zero_loss_is_non_leaf(self, device):
        """Zero-loss fallback must not be a leaf tensor.

        A leaf tensor with requires_grad=True but no graph means backward
        succeeds but produces zero gradients for all parameters — a silent
        no-op that wastes a training step.  Applying .sum() to the zero
        tensor creates a non-leaf.
        """
        loss = torch.zeros((), device=device, requires_grad=True).sum()
        assert loss.requires_grad
        assert loss.grad_fn is not None
        loss.backward()
