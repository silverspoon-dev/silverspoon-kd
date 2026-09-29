"""
Pytest fixtures and test utilities for distiller tests.
"""

import importlib.util
import os
import tempfile
import types
from collections import namedtuple
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)


def load_module_copy(module: types.ModuleType) -> types.ModuleType:
    """Execute ``module``'s source as a fresh module object.

    Lets a test observe import-time behaviour (for example an optional
    dependency being absent) without ``importlib.reload``: the real module,
    and every class it defines, stays untouched for the other tests, which
    matters because ``except SomeError`` only matches the original class.
    """
    assert module.__file__ is not None
    spec = importlib.util.spec_from_file_location(f"{module.__name__}_probe", module.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return probe


@pytest.fixture(autouse=True)
def _reset_accelerate_state():
    """Reset accelerate singleton state after every test.

    Creating TrainingArguments or a Distiller can cause accelerate to set
    environment variables (e.g. ACCELERATE_MIXED_PRECISION, ACCELERATE_DYNAMO_BACKEND)
    and cache state in AcceleratorState._shared_state (Borg pattern singleton).
    These leak into mp.spawn child processes of subsequent tests, causing
    spurious dtype mismatches or unwanted torch.compile behaviour.
    """
    yield
    from accelerate.state import AcceleratorState, PartialState

    AcceleratorState._reset_state()
    PartialState._reset_state()
    for key in list(os.environ):
        if key.startswith("ACCELERATE_"):
            del os.environ[key]
    torch._dynamo.reset()


# Pytree-compatible output type — FSDP uses tree_flatten to find output
# tensors for registering pre-backward hooks. Custom classes are opaque
# leaves; namedtuples are registered pytree nodes.
SimpleOutput = namedtuple("SimpleOutput", ["logits", "loss"])


class SimpleBlock(nn.Module):
    """A simple transformer-like block that can be captured by hooks."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.attention = nn.Linear(input_dim, output_dim)
        self.ffn = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project from input_dim to output_dim if needed
        x = self.attention(x) if x.shape[-1] != self.output_dim else self.attention(x) + x
        x = self.ffn(x) + x
        x = self.norm(x)
        return x


class SimpleModel(nn.Module):
    """Simple model for testing with configurable layers."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create embedding layer
        self.embedding = nn.Embedding(128, input_dim)

        # Create transformer-like blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = SimpleBlock(
                input_dim=input_dim if i == 0 else hidden_dim,
                output_dim=hidden_dim,
            )
            self.layers.append(layer)

        # Output head
        self.lm_head = nn.Linear(hidden_dim, 128)  # vocab_size=128

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # Embedding
        x = self.embedding(input_ids)

        # Process through layers - call forward() so hooks are triggered
        for layer in self.layers:
            x = layer(x)

        # Output logits
        logits = self.lm_head(x)

        # Compute loss when labels are provided (mirrors real HF models).
        # Deliberately does NOT move labels to logits.device — if a caller
        # passes labels on the wrong device, we want to crash loudly rather
        # than mask a bug.
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return SimpleOutput(logits=logits, loss=loss)

    def get_layer(self, idx: int) -> nn.Module:
        """Get a specific layer by index."""
        return self.layers[idx]


class TPSimpleBlock(nn.Module):
    """A block with TP-compatible architecture (separate column/row linear layers)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up) + x
        return self.norm(x)


class TPSimpleModel(nn.Module):
    """Simple model with TP plan and _no_split_modules for testing distributed strategies."""

    _no_split_modules = ["TPSimpleBlock"]

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(128, input_dim)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )

        self.layers = nn.ModuleList([TPSimpleBlock(hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, 128)

    @property
    def _tp_plan(self):
        """Tensor parallel plan for torch.distributed.tensor.parallel."""
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        plan = {}
        for i in range(self.num_layers):
            plan[f"layers.{i}.gate_proj"] = ColwiseParallel()
            plan[f"layers.{i}.up_proj"] = ColwiseParallel()
            plan[f"layers.{i}.down_proj"] = RowwiseParallel()
        return plan

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        x = self.embedding(input_ids)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return SimpleOutput(logits=logits, loss=None)

    def get_layer(self, idx: int) -> nn.Module:
        return self.layers[idx]


class DummyDataset(Dataset):
    """Simple deterministic dataset for testing.

    Samples are pre-generated with a seeded generator so the same index
    always returns the same tensors, regardless of access order.
    """

    def __init__(self, num_samples: int = 100, seq_len: int = 32, seed: int = 42):
        self.num_samples = num_samples
        self.seq_len = seq_len
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
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]


@pytest.fixture
def teacher_model(device):
    """Create a teacher model for testing."""
    model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def student_model(device):
    """Create a student model for testing (smaller than teacher)."""
    model = SimpleModel(input_dim=64, hidden_dim=64, num_layers=3)
    model.to(device)
    return model


@pytest.fixture
def small_student_model(device):
    """Create a very small student model for testing."""
    model = SimpleModel(input_dim=64, hidden_dim=32, num_layers=3)
    model.to(device)
    return model


@pytest.fixture
def same_dim_student_model(device):
    """Create a student model with same dimensions as teacher."""
    model = SimpleModel(input_dim=64, hidden_dim=128, num_layers=3)
    model.to(device)
    return model


@pytest.fixture
def train_dataset():
    """Create a small training dataset."""
    return DummyDataset(num_samples=20, seq_len=16)


@pytest.fixture
def eval_dataset():
    """Create a small evaluation dataset."""
    return DummyDataset(num_samples=10, seq_len=16)


@pytest.fixture
def training_args(tmp_path, device):
    """Create basic training arguments."""
    args = TrainingArguments(
        output_dir=str(tmp_path / "output"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        max_steps=5,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=(device.type == "cpu"),
    )
    return args


def create_alignment(
    teacher_block: nn.Module,
    student_block: nn.Module,
    teacher_model_name: str = "test_teacher",
    student_model_name: str = "test_student",
    teacher_module_name: str = "",
    student_module_name: str = "",
    teacher_hidden_dim: int = 128,
    with_input_projector: bool = False,
):
    """Helper to create an Alignment with projectors for dimension matching.

    Args:
        teacher_block: The teacher module
        student_block: The student module
        teacher_model_name: Name identifier for the teacher model
        student_model_name: Name identifier for the student model
        teacher_module_name: Name of the teacher module
        student_module_name: Name of the student module
        teacher_hidden_dim: Expected teacher hidden dimension for projector sizing
        with_input_projector: Whether to also create an input projector
    """
    from silverspoon_kd.alignments.projectors import GenericLinearProjector

    # Get student hidden dim from first parameter shape
    student_dim = None
    for module in student_block.modules():
        if isinstance(module, nn.Linear):
            student_dim = module.out_features
            break

    device = next(student_block.parameters()).device

    # Create output projector to match teacher hidden dim
    output_projector = None
    if student_dim is not None and student_dim != teacher_hidden_dim:
        output_projector = nn.Linear(student_dim, teacher_hidden_dim)
        output_projector = output_projector.to(device)

    # Create input projector to match student input dim (needed for replacement engine)
    input_projector = None
    if with_input_projector and student_dim is not None and student_dim != teacher_hidden_dim:
        # Use GenericLinearProjector in 'input' mode to project first arg
        input_projector = GenericLinearProjector(
            in_features=teacher_hidden_dim,
            out_features=student_dim,
            mode="input",
            apply_to_arg=0,  # Project the first positional argument (hidden states)
        )
        input_projector = input_projector.to(device)

    return Alignment(
        teacher_block=teacher_block,
        student_block=student_block,
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name,
        teacher_module_name=teacher_module_name,
        student_module_name=student_module_name,
        input_projector=input_projector,
        output_projector=output_projector,
    )


@pytest.fixture
def teacher_alignments(teacher_model, student_model, device):
    """Create teacher-student alignments for testing.

    Uses both input and output projectors to handle dimension mismatches between
    teacher (hidden_dim=128) and student (hidden_dim=64).

    Note: Block 0 doesn't need an input projector since its input comes from the
    embedding layer (64-dim), not from a previous teacher block (128-dim).
    """
    alignments = []

    # Create alignment for each teacher layer
    for i in range(teacher_model.num_layers):
        teacher_block = teacher_model.get_layer(i)
        student_block = student_model.get_layer(i)

        alignment = create_alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            with_input_projector=(i > 0),  # Only for blocks after the first
        )

        alignments.append(alignment)

    return alignments


@pytest.fixture
def single_alignment(teacher_model, student_model, device):
    """Create a single teacher-student alignment for testing."""
    teacher_block = teacher_model.get_layer(0)
    student_block = student_model.get_layer(0)

    alignment = create_alignment(
        teacher_block=teacher_block,
        student_block=student_block,
        teacher_module_name="layers.0",
        student_module_name="layers.0",
    )

    return [alignment]


@pytest.fixture
def tmp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_batch(batch_size: int = 2, seq_len: int = 16, device: str = "cpu"):
    """Helper function to create a batch of data."""
    return {
        "input_ids": torch.randint(0, 128, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "labels": torch.randint(0, 128, (batch_size, seq_len), device=device),
    }


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_norm(model: nn.Module) -> float:
    """Get the L2 norm of all parameters in a model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm += p.data.norm(2).item() ** 2
    return total_norm**0.5
