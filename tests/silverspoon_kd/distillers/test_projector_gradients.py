"""Projector gradients accumulate across micro-batches like the student's."""

import torch

from silverspoon_kd import HolisticDistiller, TrainingArguments
from tests.silverspoon_kd.conftest import create_batch


class TestProjectorGradientAccumulation:
    """With gradient_accumulation_steps > 1 the projector gradient is the sum over the window."""

    def test_projector_gradient_sums_over_accumulation_window(
        self, teacher_model, student_model, single_alignment, train_dataset, device, tmp_path
    ):
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            report_to=[],
            logging_steps=999,
            save_strategy="no",
            use_cpu=(device.type == "cpu"),
        )
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        projector = single_alignment[0].output_projector
        assert projector is not None
        gradient_state = distiller.accelerator.gradient_state

        torch.manual_seed(0)
        batch_1 = create_batch(device=device)
        batch_2 = create_batch(device=device)

        # One accumulation window: the Trainer flags the last micro-batch with
        # sync_gradients=True and runs the optimizer step after it.
        gradient_state._set_sync_gradients(False)
        distiller.training_step(distiller.model, batch_1)
        grad_first = projector.weight.grad.clone()
        gradient_state._set_sync_gradients(True)
        distiller.training_step(distiller.model, batch_2)
        grad_window = projector.weight.grad.clone()

        # Reference: the second micro-batch alone, in a fresh window.
        distiller.model.zero_grad(set_to_none=True)
        projector.weight.grad = None
        distiller.training_step(distiller.model, batch_2)
        grad_second = projector.weight.grad.clone()

        assert not torch.allclose(grad_first, torch.zeros_like(grad_first))
        assert torch.allclose(grad_window, grad_first + grad_second, atol=1e-6)

    def test_gradients_are_cleared_after_an_optimizer_step(
        self, teacher_model, student_model, single_alignment, train_dataset, device, tmp_path
    ):
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=1,
            per_device_train_batch_size=2,
            report_to=[],
            logging_steps=999,
            save_strategy="no",
            use_cpu=(device.type == "cpu"),
        )
        distiller = HolisticDistiller(
            student_model=student_model,
            teacher_model=teacher_model,
            alignments=single_alignment,
            args=args,
            train_dataset=train_dataset,
        )
        distiller._register_capture()
        projector = single_alignment[0].output_projector
        assert projector is not None
        distiller.accelerator.gradient_state._set_sync_gradients(True)

        torch.manual_seed(0)
        batch = create_batch(device=device)
        distiller.training_step(distiller.model, batch)
        grad_single = projector.weight.grad.clone()
        # The step above completed a window, so the next step starts fresh.
        distiller.training_step(distiller.model, batch)

        assert torch.allclose(projector.weight.grad, grad_single, atol=1e-6)
