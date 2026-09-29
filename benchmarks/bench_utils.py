"""Shared utilities for benchmark tests.

Provides model/dataset factories, memory tracking, and alignment helpers
for performance benchmarking of all silverspoon-kd components.
"""

import tracemalloc

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.alignments.projectors import GenericLinearProjector
from silverspoon_kd.distillers import (
    BlockwiseDistiller,
    HolisticDistiller,
    ResponseBasedDistiller,
)
from silverspoon_kd.training_arguments import TrainingArguments

# ═══════════════════════════════════════════════════════════════════════
#  Memory Tracking
# ═══════════════════════════════════════════════════════════════════════


class MemoryTracker:
    """Context manager for tracking peak CPU and GPU memory usage.

    Usage::

        with MemoryTracker() as mem:
            model.train()
        print(f"CPU: {mem.peak_cpu_mb:.1f} MB, GPU: {mem.peak_gpu_mb:.1f} MB")
    """

    def __init__(self):
        self.peak_gpu_bytes = 0
        self.peak_cpu_bytes = 0
        self._tracemalloc_was_tracing = False

    @property
    def peak_gpu_mb(self):
        return self.peak_gpu_bytes / (1024 * 1024)

    @property
    def peak_cpu_mb(self):
        return self.peak_cpu_bytes / (1024 * 1024)

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._tracemalloc_was_tracing = tracemalloc.is_tracing()
        if not self._tracemalloc_was_tracing:
            tracemalloc.start()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_gpu_bytes = torch.cuda.max_memory_allocated()
        _, self.peak_cpu_bytes = tracemalloc.get_traced_memory()
        if not self._tracemalloc_was_tracing:
            tracemalloc.stop()
        return False


# ═══════════════════════════════════════════════════════════════════════
#  Test Models (self-contained for benchmark isolation)
# ═══════════════════════════════════════════════════════════════════════


class SimpleBlock(nn.Module):
    """A simple transformer-like block for benchmarking."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, output_dim)
        self.ffn = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = self.attention(x) if x.shape[-1] != self.output_dim else self.attention(x) + x
        x = self.ffn(x) + x
        x = self.norm(x)
        return x


class SimpleModel(nn.Module):
    """Configurable test model for benchmarking."""

    def __init__(self, input_dim=64, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(1000, input_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                SimpleBlock(
                    input_dim=input_dim if i == 0 else hidden_dim,
                    output_dim=hidden_dim,
                )
            )
        self.lm_head = nn.Linear(hidden_dim, 1000)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)

        class Output:
            def __init__(self, logits):
                self.logits = logits
                self.loss = None

            def __iter__(self):
                yield self.logits

        return Output(logits)

    def get_layer(self, idx):
        return self.layers[idx]


class DummyDataset(Dataset):
    """Random dataset for benchmarking."""

    def __init__(self, num_samples=100, seq_len=32):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.randint(0, 1000, (self.seq_len,)),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Model & Alignment Factories
# ═══════════════════════════════════════════════════════════════════════


def make_models(device, teacher_dim=128, student_dim=64, num_layers=3, dtype=None):
    """Create teacher and student models on the given device.

    Args:
        dtype: Optional parameter dtype (e.g. ``torch.bfloat16``). The
            distillers run models in their native parameter dtype, so this
            is how reduced-precision training is selected.
    """
    teacher = SimpleModel(input_dim=64, hidden_dim=teacher_dim, num_layers=num_layers)
    teacher.to(device)
    teacher.eval()
    student = SimpleModel(input_dim=64, hidden_dim=student_dim, num_layers=num_layers)
    student.to(device)
    if dtype is not None:
        teacher.to(dtype)
        student.to(dtype)
    return teacher, student


def make_alignments(teacher, student):
    """Create teacher-student alignments with projectors for dimension matching.

    Automatically creates:
    - Output projectors when student_dim != teacher_dim
    - Input projectors for blocks 1+ when student_dim != teacher_dim
      (needed because blocks 1+ receive teacher-dimension inputs from the previous
      teacher block, but student blocks expect student-dimension inputs)

    Projectors are placed on the student block's device and dtype.
    """
    alignments = []
    teacher_dim = teacher.hidden_dim

    for i in range(teacher.num_layers):
        teacher_block = teacher.get_layer(i)
        student_block = student.get_layer(i)
        student_param = next(student_block.parameters())
        device, dtype = student_param.device, student_param.dtype

        # Determine student output dim
        student_dim = None
        for mod in student_block.modules():
            if isinstance(mod, nn.Linear):
                student_dim = mod.out_features
                break

        # Output projector (student -> teacher dim)
        output_projector = None
        if student_dim is not None and student_dim != teacher_dim:
            output_projector = nn.Linear(student_dim, teacher_dim).to(device=device, dtype=dtype)

        # Input projector (teacher -> student dim, for blocks 1+)
        input_projector = None
        if i > 0 and student_dim is not None and student_dim != teacher_dim:
            input_projector = GenericLinearProjector(
                in_features=teacher_dim,
                out_features=student_dim,
                mode="input",
                apply_to_arg=0,
            ).to(device=device, dtype=dtype)

        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="bench_teacher",
            student_model_name="bench_student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            input_projector=input_projector,
            output_projector=output_projector,
        )
        alignments.append(alignment)

    return alignments


def make_training_args(output_dir, device, max_steps=5, batch_size=2, **kwargs):
    """Create training arguments optimized for benchmarking (minimal I/O and logging).

    Extra keyword arguments are forwarded to ``TrainingArguments``.
    """
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=999999,
        save_steps=999999,
        eval_steps=999999,
        max_steps=max_steps,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=(device.type == "cpu"),
        disable_tqdm=True,
        **kwargs,
    )
    args._n_gpu = 1
    return args


# ═══════════════════════════════════════════════════════════════════════
#  Distiller Factory
# ═══════════════════════════════════════════════════════════════════════

DISTILLER_TYPES = ["blockwise", "holistic", "response_based"]


def create_distiller(
    distiller_type,
    device,
    output_dir,
    max_steps=3,
    batch_size=2,
    teacher_dim=128,
    student_dim=64,
    num_layers=3,
    seq_len=16,
    num_samples=20,
    **distiller_kwargs,
):
    """Factory to create any distiller type with consistent configuration."""
    teacher, student = make_models(device, teacher_dim, student_dim, num_layers)
    args = make_training_args(output_dir, device, max_steps, batch_size)
    dataset = DummyDataset(num_samples, seq_len)

    if distiller_type == "blockwise":
        alignments = make_alignments(teacher, student)
        return BlockwiseDistiller(
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=dataset,
            **distiller_kwargs,
        )
    elif distiller_type == "holistic":
        alignments = make_alignments(teacher, student)
        return HolisticDistiller(
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=args,
            train_dataset=dataset,
            **distiller_kwargs,
        )
    elif distiller_type == "response_based":
        return ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=dataset,
            **distiller_kwargs,
        )
    else:
        raise ValueError(f"Unknown distiller type: {distiller_type}")
