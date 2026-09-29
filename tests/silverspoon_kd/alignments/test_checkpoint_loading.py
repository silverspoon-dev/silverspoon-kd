"""
Tests for loading student weights and projectors from distiller checkpoints.

``BlockwiseDistiller`` checkpoints are written by the Trainer from the
student-blocks container: ``model.safetensors`` (or ``pytorch_model.bin``)
with ``block_<name>.*`` entries, plus ``projector_state.pt``.  End-to-end
distillers write the full student state dict.  The helpers below produce
both layouts exactly as the distillers do, and one test round-trips a real
``BlockwiseDistiller``.
"""

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from silverspoon_kd import BlockwiseDistiller, TrainingArguments
from silverspoon_kd.alignments import (
    load_student_weights_from_checkpoint,
    load_student_with_projectors_from_checkpoint,
)
from silverspoon_kd.alignments.alignment import block_module_name
from silverspoon_kd.alignments.projectors import GenericLinearProjector
from silverspoon_kd.alignments.utils import (
    _load_model_state_dict,
    _resolve_block_modules,
    _split_block_state,
)
from tests.silverspoon_kd.conftest import SimpleModel as ConftestSimpleModel
from tests.silverspoon_kd.conftest import load_module_copy


class SimpleModel(nn.Module):
    """Simple model for testing alignments."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.name_or_path = "test_model"
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(128, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class NestedModel(nn.Module):
    """Model with nested module structure for testing."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.name_or_path = "nested_model"

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.self_attn = nn.Linear(dim, dim)
                self.ffn = nn.Linear(dim, dim)

            def forward(self, x):
                return self.ffn(self.self_attn(x))

        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


# ── Checkpoint writers (mirror what the distillers produce) ─────────────────


def _write_blockwise_checkpoint(
    checkpoint_dir: Path,
    student_model_name: str,
    blocks: dict[str, nn.Module],
    projectors: dict[str, dict[str, nn.Module]] | None = None,
    fmt: str = "safetensors",
) -> None:
    """Write ``blocks`` (module name -> module) the way BlockwiseDistiller saves them."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    for module_name, module in blocks.items():
        prefix = block_module_name(f"{student_model_name}.{module_name}")
        for key, value in module.state_dict().items():
            state[f"{prefix}.{key}"] = value.detach().clone().contiguous()
    if fmt == "safetensors":
        save_file(state, checkpoint_dir / "model.safetensors")
    else:
        torch.save(state, checkpoint_dir / "pytorch_model.bin")
    if projectors:
        torch.save(
            {
                f"{student_model_name}.{module_name}": {
                    role: proj.state_dict() for role, proj in entry.items()
                }
                for module_name, entry in projectors.items()
            },
            checkpoint_dir / "projector_state.pt",
        )


def _write_e2e_checkpoint(checkpoint_dir: Path, model: nn.Module, fmt: str = "safetensors"):
    """Write a full student state dict the way the end-to-end distillers save it."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    if fmt == "safetensors":
        save_file(state, checkpoint_dir / "model.safetensors")
    else:
        torch.save(state, checkpoint_dir / "pytorch_model.bin")


def _random_linear_projector(in_features, out_features, mode):
    proj = GenericLinearProjector(
        in_features, out_features, mode=mode, apply_to_arg=0 if mode == "input" else None
    )
    with torch.no_grad():
        proj.weight.normal_()
        proj.bias.normal_()
    return proj


def _is_wrapper(module: nn.Module) -> bool:
    """True for the loader's projector wrapper.

    Compared by name because ``TestAlignmentUtilsImportFallback`` reloads the
    utils module, which re-creates the class object.
    """
    return type(module).__name__ == "_ProjectorWrapper"


def _assert_state_equal(module: nn.Module, expected: nn.Module) -> None:
    for key, value in expected.state_dict().items():
        assert torch.equal(module.state_dict()[key], value), key


@pytest.fixture
def student_model():
    return SimpleModel(hidden_dim=64, num_layers=3)


# ── load_student_weights_from_checkpoint ─────────────────────────────────────


class TestLoadStudentWeightsBlockwise:
    """Blockwise checkpoints: each ``block_<name>`` entry loads into its module."""

    @pytest.mark.parametrize("fmt", ["safetensors", "pytorch_bin"])
    def test_loads_trained_blocks_and_leaves_the_rest(self, student_model, tmp_path, fmt):
        trained = SimpleModel(hidden_dim=64, num_layers=3)
        _write_blockwise_checkpoint(
            tmp_path / "ckpt",
            "test_student",
            {"layers.0": trained.layers[0], "layers.1": trained.layers[1]},
            fmt=fmt,
        )
        untouched_layer = copy.deepcopy(student_model.layers[2])

        result = load_student_weights_from_checkpoint(
            student_model, tmp_path / "ckpt", "test_student"
        )

        assert result == {"missing_keys": [], "unexpected_keys": []}
        _assert_state_equal(student_model.layers[0], trained.layers[0])
        _assert_state_equal(student_model.layers[1], trained.layers[1])
        _assert_state_equal(student_model.layers[2], untouched_layer)

    def test_model_name_with_slash_and_dot(self, student_model, tmp_path):
        """Names like Hugging Face ids survive the attribute-safe block naming."""
        trained = SimpleModel(hidden_dim=64, num_layers=3)
        name = "Qwen/Qwen3-0.6B"
        _write_blockwise_checkpoint(tmp_path / "ckpt", name, {"layers.1": trained.layers[1]})

        load_student_weights_from_checkpoint(student_model, tmp_path / "ckpt", name)

        _assert_state_equal(student_model.layers[1], trained.layers[1])

    def test_nested_module_names(self, tmp_path):
        trained = NestedModel(hidden_dim=32, num_layers=2)
        student = NestedModel(hidden_dim=32, num_layers=2)
        _write_blockwise_checkpoint(
            tmp_path / "ckpt", "nested", {"model.layers.1": trained.model.layers[1]}
        )

        load_student_weights_from_checkpoint(student, tmp_path / "ckpt", "nested")

        _assert_state_equal(student.model.layers[1], trained.model.layers[1])

    def test_wrong_student_name_raises(self, student_model, tmp_path):
        _write_blockwise_checkpoint(
            tmp_path / "ckpt", "other_student", {"layers.0": student_model.layers[0]}
        )
        with pytest.raises(ValueError, match="No weights found for student model 'wrong_name'"):
            load_student_weights_from_checkpoint(student_model, tmp_path / "ckpt", "wrong_name")

    def test_strict_mode_raises_for_missing_keys(self, student_model, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        prefix = block_module_name("test_student.layers.0")
        torch.save({f"{prefix}.weight": torch.randn(64, 64)}, ckpt / "pytorch_model.bin")

        with pytest.raises(ValueError, match=r"Missing keys: \['layers.0.bias'\]"):
            load_student_weights_from_checkpoint(student_model, ckpt, "test_student", strict=True)

    def test_non_strict_mode_returns_missing_keys(self, student_model, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        prefix = block_module_name("test_student.layers.0")
        torch.save({f"{prefix}.weight": torch.randn(64, 64)}, ckpt / "pytorch_model.bin")

        result = load_student_weights_from_checkpoint(
            student_model, ckpt, "test_student", strict=False
        )

        assert result["missing_keys"] == ["layers.0.bias"]
        assert result["unexpected_keys"] == []

    def test_strict_mode_raises_for_unexpected_keys(self, student_model, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        prefix = block_module_name("test_student.layers.0")
        state = {f"{prefix}.{k}": v for k, v in student_model.layers[0].state_dict().items()}
        state[f"{prefix}.extra"] = torch.zeros(1)
        torch.save(state, ckpt / "pytorch_model.bin")

        with pytest.raises(ValueError, match="Unexpected keys"):
            load_student_weights_from_checkpoint(student_model, ckpt, "test_student", strict=True)


class TestLoadStudentWeightsEndToEnd:
    """Checkpoints of end-to-end distillers hold the full student state dict."""

    @pytest.mark.parametrize("fmt", ["safetensors", "pytorch_bin"])
    def test_loads_full_state_dict(self, student_model, tmp_path, fmt):
        trained = SimpleModel(hidden_dim=64, num_layers=3)
        _write_e2e_checkpoint(tmp_path / "ckpt", trained, fmt=fmt)

        result = load_student_weights_from_checkpoint(
            student_model, tmp_path / "ckpt", "test_student"
        )

        assert result == {"missing_keys": [], "unexpected_keys": []}
        _assert_state_equal(student_model, trained)

    def test_strict_mode_raises_for_missing_keys(self, student_model, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        save_file({"layers.0.weight": torch.randn(64, 64)}, ckpt / "model.safetensors")

        with pytest.raises(ValueError, match="Missing keys"):
            load_student_weights_from_checkpoint(student_model, ckpt, "test_student", strict=True)

    def test_non_strict_mode_returns_missing_keys(self, student_model, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        save_file({"layers.0.weight": torch.randn(64, 64)}, ckpt / "model.safetensors")

        result = load_student_weights_from_checkpoint(
            student_model, ckpt, "test_student", strict=False
        )

        assert "layers.0.bias" in result["missing_keys"]
        assert "embedding.weight" in result["missing_keys"]


class TestLoadStudentWeightsErrors:
    def test_raises_for_nonexistent_checkpoint(self, student_model, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
            load_student_weights_from_checkpoint(student_model, tmp_path / "missing", "test")

    def test_raises_for_missing_checkpoint_file(self, student_model, tmp_path):
        (tmp_path / "ckpt").mkdir()
        with pytest.raises(FileNotFoundError, match="No checkpoint file"):
            load_student_weights_from_checkpoint(student_model, tmp_path / "ckpt", "test")

    def test_sharded_checkpoint_points_to_from_pretrained(self, student_model, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "model.safetensors.index.json").write_text("{}")
        with pytest.raises(ValueError, match="sharded"):
            load_student_weights_from_checkpoint(student_model, ckpt, "test")

    def test_safetensors_not_available_raises(self, student_model, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "model.safetensors").touch()

        import silverspoon_kd.alignments.utils as utils_mod

        with patch.object(utils_mod, "SAFETENSORS_AVAILABLE", False):
            with pytest.raises(ImportError, match="safetensors is required"):
                load_student_weights_from_checkpoint(student_model, ckpt, "test")


class TestAlignmentUtilsImportFallback:
    """Tests for import fallback paths."""

    def test_safetensors_import_error(self):
        """SAFETENSORS_AVAILABLE is False when safetensors cannot be imported."""
        import silverspoon_kd.alignments.utils as mod

        with patch.dict(sys.modules, {"safetensors": None, "safetensors.torch": None}):
            probe = load_module_copy(mod)
        assert probe.SAFETENSORS_AVAILABLE is False


# ── load_student_with_projectors_from_checkpoint ─────────────────────────────


class TestLoadStudentWithProjectors:
    @pytest.fixture
    def checkpoint_setup(self, tmp_path):
        """Teacher (128-d), student (64-d), and a checkpoint of layers 0 and 1 with projectors."""
        torch.manual_seed(0)
        teacher = SimpleModel(hidden_dim=128, num_layers=3)
        trained = SimpleModel(hidden_dim=64, num_layers=3)
        projectors = {
            f"layers.{i}": {
                "input_projector": _random_linear_projector(128, 64, "input"),
                "output_projector": _random_linear_projector(64, 128, "output"),
            }
            for i in range(2)
        }
        _write_blockwise_checkpoint(
            tmp_path / "ckpt",
            "test_student",
            {"layers.0": trained.layers[0], "layers.1": trained.layers[1]},
            projectors,
        )
        return {
            "teacher": teacher,
            "student": SimpleModel(hidden_dim=64, num_layers=3),
            "trained": trained,
            "projectors": projectors,
            "checkpoint_dir": tmp_path / "ckpt",
            "student_model_name": "test_student",
        }

    def test_replaces_distilled_layers_only(self, checkpoint_setup):
        teacher = checkpoint_setup["teacher"]
        original_layer2 = teacher.layers[2]

        result = load_student_with_projectors_from_checkpoint(
            teacher_model=teacher,
            checkpoint_dir=checkpoint_setup["checkpoint_dir"],
            student_model_name=checkpoint_setup["student_model_name"],
            student_model=checkpoint_setup["student"],
        )

        assert result is teacher
        assert _is_wrapper(teacher.layers[0])
        assert _is_wrapper(teacher.layers[1])
        assert teacher.layers[2] is original_layer2

    def test_wrapper_matches_manual_projector_chain(self, checkpoint_setup):
        teacher = checkpoint_setup["teacher"]
        load_student_with_projectors_from_checkpoint(
            teacher,
            checkpoint_setup["checkpoint_dir"],
            checkpoint_setup["student_model_name"],
            checkpoint_setup["student"],
        )

        projectors = checkpoint_setup["projectors"]["layers.1"]
        block = checkpoint_setup["trained"].layers[1]
        x = torch.randn(2, 10, 128)
        with torch.no_grad():
            (projected,), _ = projectors["input_projector"](x)
            expected = projectors["output_projector"](block(projected))
            actual = teacher.layers[1](x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_forward_pass_through_loaded_model(self, checkpoint_setup):
        teacher = checkpoint_setup["teacher"]
        load_student_with_projectors_from_checkpoint(
            teacher,
            checkpoint_setup["checkpoint_dir"],
            checkpoint_setup["student_model_name"],
            checkpoint_setup["student"],
        )
        with torch.no_grad():
            output = teacher(torch.randint(0, 128, (2, 10)))
        assert output.shape == (2, 10, 128)

    def test_pytorch_bin_format(self, tmp_path):
        teacher = SimpleModel(hidden_dim=64, num_layers=2)
        trained = SimpleModel(hidden_dim=64, num_layers=2)
        _write_blockwise_checkpoint(
            tmp_path / "ckpt", "s", {"layers.0": trained.layers[0]}, fmt="pytorch_bin"
        )

        load_student_with_projectors_from_checkpoint(
            teacher, tmp_path / "ckpt", "s", SimpleModel(hidden_dim=64, num_layers=2)
        )

        # Without projectors the block itself is installed, not a wrapper.
        assert not _is_wrapper(teacher.layers[0])
        _assert_state_equal(teacher.layers[0], trained.layers[0])

    def test_loaded_blocks_change_the_output(self, tmp_path):
        teacher = SimpleModel(hidden_dim=64, num_layers=2)
        trained = SimpleModel(hidden_dim=64, num_layers=2)
        x = torch.randint(0, 128, (2, 10))
        with torch.no_grad():
            before = teacher(x).clone()
        _write_blockwise_checkpoint(
            tmp_path / "ckpt", "s", {"layers.0": trained.layers[0], "layers.1": trained.layers[1]}
        )

        load_student_with_projectors_from_checkpoint(
            teacher, tmp_path / "ckpt", "s", SimpleModel(hidden_dim=64, num_layers=2)
        )

        with torch.no_grad():
            after = teacher(x)
        assert not torch.allclose(before, after)

    def test_conv2d_projectors(self, tmp_path):
        from silverspoon_kd.alignments.projectors import GenericConv2dProjector

        class CNNModel(nn.Module):
            def __init__(self, channels, num_layers=2):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(num_layers)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        teacher = CNNModel(64)
        trained = CNNModel(32)
        projectors = {
            f"layers.{i}": {
                "input_projector": GenericConv2dProjector(64, 32, mode="input", apply_to_arg=0),
                "output_projector": GenericConv2dProjector(32, 64, mode="output"),
            }
            for i in range(2)
        }
        _write_blockwise_checkpoint(
            tmp_path / "ckpt",
            "cnn",
            {"layers.0": trained.layers[0], "layers.1": trained.layers[1]},
            projectors,
        )

        load_student_with_projectors_from_checkpoint(
            teacher, tmp_path / "ckpt", "cnn", CNNModel(32)
        )

        with torch.no_grad():
            output = teacher(torch.randn(2, 64, 8, 8))
        assert output.shape == (2, 64, 8, 8)

    def test_explicit_device(self, checkpoint_setup):
        teacher = checkpoint_setup["teacher"]
        load_student_with_projectors_from_checkpoint(
            teacher,
            checkpoint_setup["checkpoint_dir"],
            checkpoint_setup["student_model_name"],
            checkpoint_setup["student"],
            device=torch.device("cpu"),
        )
        assert all(p.device.type == "cpu" for p in teacher.layers[0].parameters())

    def test_end_to_end_checkpoint_raises(self, tmp_path):
        teacher = SimpleModel(hidden_dim=64, num_layers=2)
        _write_e2e_checkpoint(tmp_path / "ckpt", SimpleModel(hidden_dim=64, num_layers=2))

        with pytest.raises(ValueError, match="not a BlockwiseDistiller checkpoint"):
            load_student_with_projectors_from_checkpoint(
                teacher, tmp_path / "ckpt", "s", SimpleModel(hidden_dim=64, num_layers=2)
            )

    def test_wrong_student_name_raises(self, checkpoint_setup):
        with pytest.raises(ValueError, match="No weights found"):
            load_student_with_projectors_from_checkpoint(
                checkpoint_setup["teacher"],
                checkpoint_setup["checkpoint_dir"],
                "wrong_name",
                checkpoint_setup["student"],
            )

    def test_nonexistent_checkpoint_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
            load_student_with_projectors_from_checkpoint(
                SimpleModel(), tmp_path / "missing", "s", SimpleModel(hidden_dim=64)
            )

    def test_no_checkpoint_file_raises(self, tmp_path):
        (tmp_path / "ckpt").mkdir()
        with pytest.raises(FileNotFoundError, match="No checkpoint file"):
            load_student_with_projectors_from_checkpoint(
                SimpleModel(), tmp_path / "ckpt", "s", SimpleModel(hidden_dim=64)
            )

    def test_safetensors_not_available_raises(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "model.safetensors").touch()

        import silverspoon_kd.alignments.utils as utils_mod

        with patch.object(utils_mod, "SAFETENSORS_AVAILABLE", False):
            with pytest.raises(ImportError, match="safetensors is required"):
                load_student_with_projectors_from_checkpoint(
                    SimpleModel(), ckpt, "s", SimpleModel(hidden_dim=64)
                )


class TestLoadStudentWithProjectorsAtomicValidation:
    """Every distilled module is checked against the teacher before any replacement.

    Otherwise a checkpoint whose entries include a module the teacher lacks
    would leave the teacher half-replaced with no clean recovery path.
    """

    def _setup(self, tmp_path):
        # The student has a third layer the teacher does not, and the checkpoint
        # covers all three.
        teacher = SimpleModel(hidden_dim=128, num_layers=2)
        student = SimpleModel(hidden_dim=64, num_layers=3)
        _write_blockwise_checkpoint(
            tmp_path / "ckpt",
            "s",
            {f"layers.{i}": student.layers[i] for i in range(3)},
        )
        return teacher, student

    def test_partial_checkpoint_does_not_mutate_teacher(self, tmp_path):
        teacher, student = self._setup(tmp_path)
        original_layers = list(teacher.layers)
        original_modules = dict(teacher.named_modules())

        with pytest.raises(ValueError, match="does not exist in teacher model"):
            load_student_with_projectors_from_checkpoint(teacher, tmp_path / "ckpt", "s", student)

        for idx, layer in enumerate(teacher.layers):
            assert layer is original_layers[idx]
        for name, module in teacher.named_modules():
            assert original_modules.get(name) is module

    def test_error_message_names_the_missing_module(self, tmp_path):
        teacher, student = self._setup(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            load_student_with_projectors_from_checkpoint(teacher, tmp_path / "ckpt", "s", student)
        assert "layers.2" in str(exc_info.value)


# ── Round trip through a real BlockwiseDistiller ─────────────────────────────


class TestBlockwiseDistillerRoundTrip:
    """Train, save, and load back with the public loaders."""

    @pytest.fixture
    def trained_checkpoint(
        self, teacher_model, student_model, teacher_alignments, train_dataset, device, tmp_path
    ):
        # ``student_model`` here is the conftest fixture (SimpleModel from conftest,
        # 64-d), overriding this module's fixture of the same name.
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=2,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            use_cpu=(device.type == "cpu"),
        )
        distiller = BlockwiseDistiller(
            teacher_model=teacher_model,
            alignments=teacher_alignments,
            args=args,
            train_dataset=train_dataset,
        )
        distiller.train()
        checkpoint_dir = tmp_path / "ckpt"
        distiller._save(str(checkpoint_dir))
        return {
            "checkpoint_dir": checkpoint_dir,
            "alignments": teacher_alignments,
            "teacher": teacher_model,
            "device": device,
        }

    @pytest.fixture
    def student_model(self, device):
        # Conftest layout (SimpleModel with ``embedding``/``layers``/``lm_head``,
        # 64-d hidden) so the alignments from ``teacher_alignments`` apply.
        return ConftestSimpleModel(input_dim=64, hidden_dim=64, num_layers=3).to(device)

    def test_weights_loader_restores_every_trained_block(self, trained_checkpoint, device):
        fresh = ConftestSimpleModel(input_dim=64, hidden_dim=64, num_layers=3).to(device)

        result = load_student_weights_from_checkpoint(
            fresh, trained_checkpoint["checkpoint_dir"], "test_student"
        )

        assert result == {"missing_keys": [], "unexpected_keys": []}
        for i, alignment in enumerate(trained_checkpoint["alignments"]):
            _assert_state_equal(fresh.layers[i], alignment.student_block)

    def test_projector_loader_installs_trained_blocks_and_projectors(
        self, trained_checkpoint, device
    ):
        teacher = trained_checkpoint["teacher"]
        template = ConftestSimpleModel(input_dim=64, hidden_dim=64, num_layers=3).to(device)

        load_student_with_projectors_from_checkpoint(
            teacher, trained_checkpoint["checkpoint_dir"], "test_student", template
        )

        for i, alignment in enumerate(trained_checkpoint["alignments"]):
            wrapper = teacher.layers[i]
            assert _is_wrapper(wrapper)
            modules = list(wrapper.modules_list)
            block = modules[1] if alignment.input_projector is not None else modules[0]
            _assert_state_equal(block, alignment.student_block)
            _assert_state_equal(modules[-1], alignment.output_projector)
            if alignment.input_projector is not None:
                _assert_state_equal(modules[0], alignment.input_projector)

        with torch.no_grad():
            output = teacher(
                input_ids=torch.randint(0, 128, (2, 8), device=device),
                attention_mask=torch.ones(2, 8, dtype=torch.long, device=device),
            )
        assert output.logits.shape == (2, 8, 128)


# ── Private helpers ──────────────────────────────────────────────────────────


class TestSplitBlockState:
    def test_groups_by_block(self):
        state = {
            "block_a.weight": torch.zeros(1),
            "block_a.bias": torch.zeros(1),
            "block_b.sub.weight": torch.zeros(1),
        }
        blocks = _split_block_state(state)
        assert set(blocks) == {"block_a", "block_b"}
        assert set(blocks["block_a"]) == {"weight", "bias"}
        assert set(blocks["block_b"]) == {"sub.weight"}

    def test_end_to_end_state_gives_no_blocks(self):
        assert _split_block_state({"layers.0.weight": torch.zeros(1)}) == {}


class TestResolveBlockModules:
    def test_maps_blocks_to_module_paths(self):
        student = NestedModel(hidden_dim=8, num_layers=2)
        names = [block_module_name("s.model.layers.1"), block_module_name("s.model.layers.0.ffn")]
        resolved = _resolve_block_modules(names, student, "s")
        assert resolved == {names[0]: "model.layers.1", names[1]: "model.layers.0.ffn"}

    def test_unknown_block_raises(self):
        student = NestedModel(hidden_dim=8, num_layers=2)
        with pytest.raises(ValueError, match="No weights found for student model 's'"):
            _resolve_block_modules([block_module_name("s.model.layers.5")], student, "s")


class TestLoadModelStateDict:
    def test_prefers_safetensors_over_pytorch_bin(self, tmp_path):
        save_file({"w": torch.ones(1)}, tmp_path / "model.safetensors")
        torch.save({"w": torch.zeros(1)}, tmp_path / "pytorch_model.bin")
        assert _load_model_state_dict(tmp_path)["w"].item() == 1.0

    def test_loads_pytorch_bin_when_only_bin_present(self, tmp_path):
        torch.save({"w": torch.zeros(1)}, tmp_path / "pytorch_model.bin")
        assert _load_model_state_dict(tmp_path)["w"].item() == 0.0

    def test_raises_when_neither_present(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No checkpoint file"):
            _load_model_state_dict(tmp_path)


class TestBuildProjectorFromState:
    """Direct tests for ``_build_projector_from_state``."""

    def _helper(self):
        from silverspoon_kd.alignments.utils import _build_projector_from_state

        return _build_projector_from_state

    def test_empty_state_returns_none(self):
        helper = self._helper()
        assert helper({}, mode="output", device=torch.device("cpu")) is None

    def test_missing_weight_returns_none(self):
        helper = self._helper()
        state = {"bias": torch.randn(4)}
        assert helper(state, mode="output", device=torch.device("cpu")) is None

    def test_2d_weight_builds_linear_projector(self):
        from silverspoon_kd.alignments.projectors import GenericLinearProjector

        helper = self._helper()
        state = {
            "weight": torch.randn(8, 16),  # [out, in]
            "bias": torch.randn(8),
        }
        proj = helper(state, mode="output", device=torch.device("cpu"))
        assert proj is not None
        assert isinstance(proj, GenericLinearProjector)
        assert proj.in_features == 16
        assert proj.out_features == 8
        assert proj.mode == "output"
        # The projector's weights match the state dict
        assert torch.allclose(proj.weight, state["weight"])
        assert torch.allclose(proj.bias, state["bias"])

    def test_4d_weight_builds_conv2d_projector(self):
        from silverspoon_kd.alignments.projectors import GenericConv2dProjector

        helper = self._helper()
        state = {
            "weight": torch.randn(8, 4, 1, 1),  # [out_ch, in_ch, kh, kw]
        }
        proj = helper(state, mode="output", device=torch.device("cpu"))
        assert proj is not None
        assert isinstance(proj, GenericConv2dProjector)
        assert proj.in_channels == 4
        assert proj.out_channels == 8
        assert proj.mode == "output"

    def test_input_mode_sets_apply_to_arg(self):
        helper = self._helper()
        state = {"weight": torch.randn(8, 16), "bias": torch.randn(8)}
        proj = helper(state, mode="input", device=torch.device("cpu"))
        assert proj is not None
        assert proj.mode == "input"
        # Input mode should default to apply_to_arg=0
        assert proj.apply_to_arg == 0
        assert proj.apply_to_kwarg is None

    def test_output_mode_does_not_set_apply_to_kwarg(self):
        helper = self._helper()
        state = {"weight": torch.randn(8, 16), "bias": torch.randn(8)}
        proj = helper(state, mode="output", device=torch.device("cpu"))
        assert proj is not None
        assert proj.mode == "output"
        assert proj.apply_to_kwarg is None

    def test_unsupported_weight_shape_returns_none(self):
        helper = self._helper()
        # 3D weight is not supported (neither Linear nor Conv2d).
        state = {"weight": torch.randn(4, 4, 4)}
        result = helper(state, mode="output", device=torch.device("cpu"))
        assert result is None


class TestInstallReplacementModule:
    """Direct tests for ``_install_replacement_module``."""

    def _helpers(self):
        from silverspoon_kd.alignments.utils import _install_replacement_module
        from silverspoon_kd.utils import _ModelModuleNameMapper

        return _install_replacement_module, _ModelModuleNameMapper

    def test_installs_top_level_module(self):
        install, Mapper = self._helpers()

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(8, 8)

        teacher = M()
        replacement = nn.Linear(4, 4)
        mapper = Mapper(teacher)
        install(teacher, mapper, "encoder", replacement)
        assert teacher.encoder is replacement

    def test_installs_nested_module(self):
        install, Mapper = self._helpers()

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 8)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Inner()

        teacher = M()
        original_block = teacher.block
        replacement = nn.Linear(4, 4)
        mapper = Mapper(teacher)
        install(teacher, mapper, "block.proj", replacement)
        assert teacher.block.proj is replacement
        # Parent block should NOT be replaced
        assert teacher.block is original_block


class TestProjectorWrapper:
    """Direct tests for ``_ProjectorWrapper`` (was inline before refactor)."""

    def test_no_projectors_chains_modules_directly(self):
        from silverspoon_kd.alignments.utils import _ProjectorWrapper

        # ProjectorWrapper is normally used with at least one projector,
        # but it should still work with a plain Linear module list.
        layers = [nn.Linear(8, 4)]
        wrapper = _ProjectorWrapper(layers)
        x = torch.randn(2, 8)
        y = wrapper(x)
        assert y.shape == (2, 4)
        assert wrapper.has_input_projector is False

    def test_input_projector_propagates_args_kwargs(self):
        from silverspoon_kd.alignments.projectors import GenericLinearProjector
        from silverspoon_kd.alignments.utils import _ProjectorWrapper

        # Input projector projects 16 -> 8, then student takes 8 -> 4.
        input_proj = GenericLinearProjector(16, 8, mode="input", apply_to_arg=0)
        student = nn.Linear(8, 4)

        wrapper = _ProjectorWrapper([input_proj, student])
        assert wrapper.has_input_projector is True

        x = torch.randn(2, 16)
        y = wrapper(x)
        assert y.shape == (2, 4)

    def test_output_projector_wraps_result(self):
        from silverspoon_kd.alignments.projectors import GenericLinearProjector
        from silverspoon_kd.alignments.utils import _ProjectorWrapper

        # student takes 16 -> 8, output projector lifts 8 -> 16.
        student = nn.Linear(16, 8)
        output_proj = GenericLinearProjector(8, 16, mode="output")

        wrapper = _ProjectorWrapper([student, output_proj])
        assert wrapper.has_input_projector is False

        x = torch.randn(2, 16)
        y = wrapper(x)
        assert y.shape == (2, 16)

    def test_full_input_student_output_chain(self):
        from silverspoon_kd.alignments.projectors import GenericLinearProjector
        from silverspoon_kd.alignments.utils import _ProjectorWrapper

        # 16 -> 8 (input proj) -> 8 (student) -> 16 (output proj)
        input_proj = GenericLinearProjector(16, 8, mode="input", apply_to_arg=0)
        student = nn.Linear(8, 8)
        output_proj = GenericLinearProjector(8, 16, mode="output")

        wrapper = _ProjectorWrapper([input_proj, student, output_proj])
        assert wrapper.has_input_projector is True

        x = torch.randn(2, 16)
        y = wrapper(x)
        assert y.shape == (2, 16)

    def test_modules_list_is_a_modulelist(self):
        from silverspoon_kd.alignments.utils import _ProjectorWrapper

        layers = [nn.Linear(8, 4)]
        wrapper = _ProjectorWrapper(layers)
        # Must be a ModuleList so that .parameters(), .to(), .train() etc. work.
        assert isinstance(wrapper.modules_list, nn.ModuleList)
        # And the parameters of inner modules show up in wrapper.parameters().
        wrapper_params = list(wrapper.parameters())
        layer_params = list(layers[0].parameters())
        assert len(wrapper_params) == len(layer_params)
