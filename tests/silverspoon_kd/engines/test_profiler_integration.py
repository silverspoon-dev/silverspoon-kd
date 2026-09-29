"""
Integration tests for profiling across all distiller types.

Verifies that enable_profiling=True produces trace files and does not
break training.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.distillers import (
    BlockwiseDistiller,
    HolisticDistiller,
    ResponseBasedDistiller,
)
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)

DISTILLER_TYPES = ["blockwise", "holistic", "response_based"]


# ── Lightweight helpers (self-contained, no conftest dependency) ──────


class _Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.linear(x))


class _Model(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_dim)
        self.layers = nn.ModuleList([_Block(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, 128)
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)

        class _Out:
            def __init__(self, logits):
                self.logits = logits
                self.loss = None

            def __iter__(self):
                yield self.logits

        return _Out(logits)

    def get_layer(self, idx):
        return self.layers[idx]


class _Dataset(Dataset):
    def __init__(self, n=20, seq_len=16):
        self.n = n
        self.seq_len = seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 128, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.randint(0, 128, (self.seq_len,)),
        }


def _make_alignments(teacher, student):
    alignments = []
    for i in range(teacher.num_layers):
        t_block = teacher.get_layer(i)
        s_block = student.get_layer(i)
        alignment = Alignment(
            teacher_block=t_block,
            student_block=s_block,
            teacher_model_name="prof_teacher",
            student_model_name="prof_student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
        )
        alignments.append(alignment)
    return alignments


def _make_args(tmp_path, device, args_cls=TrainingArguments, **overrides):
    defaults = {
        "output_dir": str(tmp_path / "output"),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "logging_steps": 999999,
        "save_steps": 999999,
        "eval_steps": 999999,
        "max_steps": 3,
        "dataloader_num_workers": 0,
        "report_to": [],
        "use_cpu": (device.type == "cpu"),
        "disable_tqdm": True,
        "enable_profiling": True,
        "profiling_wait": 0,
        "profiling_warmup": 1,
        "profiling_active": 2,
        "profiling_repeat": 1,
    }
    defaults.update(overrides)
    return args_cls(**defaults)


_PROF_ARGS_CLS = {
    "blockwise": TrainingArguments,
    "holistic": TrainingArguments,
    "response_based": TrainingArguments,
}


def _create_distiller(distiller_type, device, tmp_path, eval_dataset=None, **args_overrides):
    teacher = _Model(hidden_dim=64, num_layers=2).to(device).eval()
    student = _Model(hidden_dim=64, num_layers=2).to(device)
    dataset = _Dataset()
    args = _make_args(tmp_path, device, args_cls=_PROF_ARGS_CLS[distiller_type], **args_overrides)

    kwargs = {
        "args": args,
        "train_dataset": dataset,
    }
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset

    if distiller_type == "blockwise":
        return BlockwiseDistiller(
            teacher_model=teacher,
            alignments=_make_alignments(teacher, student),
            **kwargs,
        )
    elif distiller_type == "holistic":
        return HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=_make_alignments(teacher, student),
            **kwargs,
        )
    elif distiller_type == "response_based":
        return ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            **kwargs,
        )
    raise ValueError(f"Unknown distiller type: {distiller_type}")


# ── Tests ─────────────────────────────────────────────────────────────


class TestProfilerIntegration:
    """End-to-end profiler integration tests for all distiller types."""

    @pytest.mark.parametrize("distiller_type", DISTILLER_TYPES)
    def test_training_produces_trace_files(self, distiller_type, device, tmp_path):
        """Training with enable_profiling=True should produce trace JSON files."""
        distiller = _create_distiller(distiller_type, device, tmp_path)
        distiller.train()

        profiling_dir = tmp_path / "output" / "profiling"
        assert profiling_dir.is_dir(), "Profiling output directory was not created"

        trace_files = list(profiling_dir.glob("trace_step_*.json"))
        assert len(trace_files) > 0, "No trace files were produced"

    @pytest.mark.parametrize("distiller_type", DISTILLER_TYPES)
    def test_training_results_unaffected_by_profiling(self, distiller_type, device, tmp_path):
        """Training with profiling enabled should complete and return results."""
        distiller = _create_distiller(distiller_type, device, tmp_path)
        result = distiller.train()
        assert result is not None, "Training returned None"

    @pytest.mark.parametrize("distiller_type", DISTILLER_TYPES)
    def test_evaluate_produces_trace_files(self, distiller_type, device, tmp_path):
        """evaluate() with enable_profiling=True should produce trace JSON files.

        Regression test: prediction_step must call _profiler_step() so the
        profiler's schedule advances past "wait" during the eval loop.
        Otherwise eval-only memory leaks would be invisible to the built-in
        profiler.
        """
        eval_dataset = _Dataset(n=8)
        distiller = _create_distiller(
            distiller_type,
            device,
            tmp_path,
            eval_dataset=eval_dataset,
            # Keep training short so we can isolate eval stepping. Eval runs
            # 4 prediction_step calls (8 samples / batch_size=2), which
            # comfortably covers wait=0 + warmup=1 + active=2 = 3 steps.
            max_steps=1,
        )
        # Run evaluate() directly (no training). This should still advance
        # the profiler schedule via prediction_step and produce trace files.
        distiller._register_capture()
        try:
            distiller._start_profiler()
            try:
                distiller.evaluate()
            finally:
                distiller._stop_profiler()
        finally:
            distiller._deregister_capture()

        profiling_dir = tmp_path / "output" / "profiling"
        assert profiling_dir.is_dir(), "Profiling output directory was not created"

        trace_files = list(profiling_dir.glob("trace_step_*.json"))
        assert len(trace_files) > 0, (
            f"No trace files produced during evaluate() for {distiller_type}. "
            "prediction_step must call _profiler_step() to advance the "
            "schedule during the eval loop."
        )

    @pytest.mark.parametrize("distiller_type", DISTILLER_TYPES)
    def test_prediction_step_advances_profiler(self, distiller_type, device, tmp_path):
        """prediction_step must call _profiler_step() so the profiler
        records per-batch activity during evaluation.

        This is a focused unit test using a Mock profiler to verify the
        call is actually made, independent of whether trace files land
        on disk.
        """
        from unittest.mock import MagicMock

        distiller = _create_distiller(distiller_type, device, tmp_path)
        # Register capture hooks (normally done by train()/evaluate()).
        distiller._register_capture()
        try:
            # Replace the real profiler with a mock that tracks step() calls.
            mock_profiler = MagicMock()
            distiller.profiler = mock_profiler

            inputs = {
                "input_ids": torch.randint(0, 128, (2, 16), device=device),
                "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
                "labels": torch.randint(0, 128, (2, 16), device=device),
            }
            distiller.model.eval()
            with torch.no_grad():
                distiller.prediction_step(distiller.model, inputs, prediction_loss_only=True)

            mock_profiler.step.assert_called_once()
        finally:
            distiller._deregister_capture()
