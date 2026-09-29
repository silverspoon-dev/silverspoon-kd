"""
Tests for general utilities: output selection, parameter freezing, module name mapping, etc.
"""

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class SimpleConfig(PretrainedConfig):
    """Simple config for testing."""

    model_type = "simple"

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 4,
        num_key_value_heads: int | None = None,
        intermediate_size: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.intermediate_size = intermediate_size


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""

    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # FFN
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple forward (not a real transformer, just for testing shapes)
        residual = x
        x = self.input_layernorm(x)
        x = self.o_proj(self.v_proj(x))  # Simplified attention
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.down_proj(torch.relu(self.up_proj(x)))
        x = residual + x
        return x


class SimpleModel(PreTrainedModel):
    """Simple PreTrainedModel for testing reconfig_model."""

    config_class = SimpleConfig

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.config = config

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                SimpleTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        # Output
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights
        self.post_init()

    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class TestOutputSelector:
    """Tests for OutputSelector utility class."""

    def test_selector_extracts_tuple_element(self):
        from silverspoon_kd.alignments import OutputSelector

        selector = OutputSelector(index=0)
        output = (torch.randn(2, 64), torch.randn(2, 32))
        result = selector(output)
        assert torch.equal(result, output[0])

    def test_selector_extracts_second_element(self):
        from silverspoon_kd.alignments import OutputSelector

        selector = OutputSelector(index=1)
        output = (torch.randn(2, 64), torch.randn(2, 32))
        result = selector(output)
        assert torch.equal(result, output[1])

    def test_selector_returns_non_tuple_directly(self):
        from silverspoon_kd.alignments import OutputSelector

        selector = OutputSelector(index=0)
        output = torch.randn(2, 64)
        result = selector(output)
        assert torch.equal(result, output)

    def test_selector_default_index(self):
        from silverspoon_kd.alignments import OutputSelector

        selector = OutputSelector()
        assert selector.index == 0


class TestModelModuleNameMapper:
    """Tests for _ModelModuleNameMapper."""

    def test_get_name(self):
        from silverspoon_kd.utils import _ModelModuleNameMapper

        model = SimpleModel(SimpleConfig())
        mapper = _ModelModuleNameMapper(model)
        name = mapper.get_name(model.layers[0])
        assert name == "layers.0"

    def test_get_module(self):
        from silverspoon_kd.utils import _ModelModuleNameMapper

        model = SimpleModel(SimpleConfig())
        mapper = _ModelModuleNameMapper(model)
        module = mapper.get_module("layers.0")
        assert module is model.layers[0]

    def test_get_name_unknown_module_raises(self):
        from silverspoon_kd.utils import _ModelModuleNameMapper

        model = SimpleModel(SimpleConfig())
        mapper = _ModelModuleNameMapper(model)
        foreign_module = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Module not found"):
            mapper.get_name(foreign_module)


class TestFreezeParameters:
    """Tests for freeze_parameters utility."""

    def test_freeze_matching_parameters(self):
        from silverspoon_kd.utils import freeze_parameters

        model = SimpleModel(SimpleConfig())
        freeze_parameters(model, [r"layers\.\d+\.q_proj"])

        for layer in model.layers:
            assert not layer.q_proj.weight.requires_grad
            # Other params should still be trainable
            assert layer.k_proj.weight.requires_grad

    def test_freeze_all_parameters(self):
        from silverspoon_kd.utils import freeze_parameters

        model = SimpleModel(SimpleConfig())
        freeze_parameters(model, [r".*"])

        for param in model.parameters():
            assert not param.requires_grad

    def test_thaw_not_matched(self):
        from silverspoon_kd.utils import freeze_parameters

        model = SimpleModel(SimpleConfig())
        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Now freeze q_proj and thaw everything else
        freeze_parameters(model, [r"q_proj"], thaw_not_matched=True)

        for layer in model.layers:
            assert not layer.q_proj.weight.requires_grad
            assert layer.k_proj.weight.requires_grad


class TestGetModulesByNames:
    """Tests for get_modules_by_names utility."""

    def test_get_modules_by_regex(self):
        from silverspoon_kd.utils import get_modules_by_names

        model = SimpleModel(SimpleConfig())
        modules = get_modules_by_names(model, [r"layers\.\d+$"])
        assert len(modules) == model.config.num_hidden_layers

    def test_get_specific_module(self):
        from silverspoon_kd.utils import get_modules_by_names

        model = SimpleModel(SimpleConfig())
        modules = get_modules_by_names(model, [r"layers\.0$"])
        assert len(modules) == 1
        assert modules[0] is model.layers[0]

    def test_no_match_returns_empty(self):
        from silverspoon_kd.utils import get_modules_by_names

        model = SimpleModel(SimpleConfig())
        modules = get_modules_by_names(model, [r"nonexistent"])
        assert len(modules) == 0


class TestEstimateNumTrainingSteps:
    """Tests for _estimate_num_training_steps."""

    @staticmethod
    def _args(**overrides):
        from types import SimpleNamespace

        values = {
            "per_device_train_batch_size": 10,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 2,
            "world_size": 1,
            "dataloader_drop_last": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_estimate(self):
        from silverspoon_kd.utils import _estimate_num_training_steps

        dataset = list(range(100))  # 100 samples
        result = _estimate_num_training_steps(dataset, self._args())
        assert result == 20  # (100 / (10 * 1)) * 2

    def test_partial_final_batch_counts_as_a_step(self):
        """A partial last batch is still one optimizer step (dataloader_drop_last=False)."""
        from silverspoon_kd.utils import _estimate_num_training_steps

        dataset = list(range(105))
        assert _estimate_num_training_steps(dataset, self._args(num_train_epochs=1)) == 11
        assert (
            _estimate_num_training_steps(
                dataset, self._args(num_train_epochs=1, dataloader_drop_last=True)
            )
            == 10
        )

    def test_world_size_divides_the_dataset(self):
        """Under data parallelism each process sees 1/world_size of the batches."""
        from silverspoon_kd.utils import _estimate_num_training_steps

        dataset = list(range(100))
        # 100 samples / (10 per device * 4 processes) = 2.5 -> 3 batches per epoch, 2 epochs.
        assert _estimate_num_training_steps(dataset, self._args(world_size=4)) == 6

    def test_gradient_accumulation_reduces_steps(self):
        from silverspoon_kd.utils import _estimate_num_training_steps

        dataset = list(range(100))
        args = self._args(gradient_accumulation_steps=4, num_train_epochs=1)
        assert _estimate_num_training_steps(dataset, args) == 3  # ceil(10 / 4)

    def test_dataset_without_length_requires_max_steps(self):
        from silverspoon_kd.utils import _estimate_num_training_steps

        def samples():
            yield from range(10)

        with pytest.raises(ValueError, match="max_steps"):
            _estimate_num_training_steps(samples(), self._args())


class TestSummarizeLayerNames:
    """Tests for summarize_layer_names and partial_summarize_layer_names."""

    def test_partial_summarize(self):
        from silverspoon_kd.utils import partial_summarize_layer_names

        keys = [
            "layers.0.attn.q_proj",
            "layers.1.attn.q_proj",
            "layers.2.attn.q_proj",
        ]
        result = partial_summarize_layer_names(keys)
        assert len(result) == 1
        assert "[0-2]" in result[0]

    def test_summarize_layer_names(self):
        from silverspoon_kd.utils import summarize_layer_names

        keys = [
            "model.layers.0.attn.q_proj",
            "model.layers.1.attn.q_proj",
            "model.layers.2.attn.q_proj",
        ]
        result = summarize_layer_names(keys)
        assert len(result) == 1
        assert "[0-2]" in result[0]

    def test_unmatched_keys_preserved(self):
        from silverspoon_kd.utils import partial_summarize_layer_names

        keys = ["model.norm.weight", "model.norm.bias"]
        result = partial_summarize_layer_names(keys)
        assert "model.norm.weight" in result
        assert "model.norm.bias" in result

    def test_non_consecutive_ranges(self):
        """Test summarization with non-consecutive layer numbers."""
        from silverspoon_kd.utils import partial_summarize_layer_names

        keys = [
            "layers.0.attn.q_proj",
            "layers.1.attn.q_proj",
            "layers.5.attn.q_proj",
            "layers.8.attn.q_proj",
            "layers.9.attn.q_proj",
            "layers.10.attn.q_proj",
        ]
        result = partial_summarize_layer_names(keys)
        assert len(result) == 1
        # Should contain non-consecutive ranges like [0-1,5,8-10]
        assert "0-1" in result[0]
        assert "5" in result[0]
        assert "8-10" in result[0]

    def test_recursive_summarization_nested(self):
        """Test recursive summarization with multiply-nested numbers."""
        from silverspoon_kd.utils import summarize_layer_names

        keys = [
            "model.layers.0.attn.head.0.q_proj",
            "model.layers.0.attn.head.1.q_proj",
            "model.layers.1.attn.head.0.q_proj",
            "model.layers.1.attn.head.1.q_proj",
        ]
        result = summarize_layer_names(keys)
        # Should recursively summarize both layer and head numbers
        assert len(result) == 1
        assert "[0-1]" in result[0]
