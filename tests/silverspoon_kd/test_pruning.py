"""
Tests for prune_to_config, attention head pruning, and related utilities.
"""

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from silverspoon_kd import reconfig_model


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


class _SelfAttn(nn.Module):
    """Self-attention module with separate Q/K/V/O projections for testing."""

    HEAD_DIM = 16

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, num_heads * self.HEAD_DIM)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.HEAD_DIM)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.HEAD_DIM)
        self.o_proj = nn.Linear(num_heads * self.HEAD_DIM, hidden_size)

    def forward(self, x):
        q = self.q_proj(x)
        # k/v computed but not used in output (simplified for testing)
        self.k_proj(x)
        self.v_proj(x)
        return self.o_proj(q)


class AttnBlock(nn.Module):
    """Transformer block with self_attn submodule for testing attention pruning."""

    def __init__(self, config):
        super().__init__()
        self.self_attn = _SelfAttn(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
        )
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.down_proj(torch.relu(self.up_proj(x)))
        x = residual + x
        return x


class AttnSimpleModel(PreTrainedModel):
    """Model with self_attn submodules for testing attention head pruning."""

    config_class = SimpleConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([AttnBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class TestMutableDefaultArgSafety:
    """Tests to ensure mutable default arguments don't cause cross-call contamination.

    These tests guard against the Python mutable default argument bug (W0102),
    where using {} or [] as default parameter values causes shared state across calls.
    """

    @pytest.fixture
    def base_model(self):
        """Create a base model for testing."""
        config = SimpleConfig()
        return SimpleModel(config)

    def test_reconfig_model_diff_none_does_not_share_state(self, base_model):
        """Test that calling reconfig_model with diff=None twice doesn't share the dict."""
        # Call once with diff=None (default)
        model1 = reconfig_model(base_model, name_or_path="model1")
        # Call again with diff=None (default)
        model2 = reconfig_model(base_model, name_or_path="model2")

        # Both should work independently
        assert model1.config.name_or_path == "model1"
        assert model2.config.name_or_path == "model2"

    def test_reconfig_model_diff_none_not_mutated_across_calls(self, base_model):
        """Test that the default diff dict is not mutated between calls."""
        # First call with explicit diff
        reconfig_model(
            base_model,
            name_or_path="model1",
            diff={"num_hidden_layers": 2},
        )

        # Second call with default diff=None - should not be affected by first call
        model2 = reconfig_model(base_model, name_or_path="model2")
        assert model2.config.num_hidden_layers == base_model.config.num_hidden_layers

    def test_reconfig_model_multiple_none_calls_independent(self, base_model):
        """Test that multiple calls with diff=None create independent configs."""
        models = []
        for i in range(5):
            m = reconfig_model(base_model, name_or_path=f"model_{i}")
            models.append(m)

        # All should have independent configs
        for i, m in enumerate(models):
            assert m.config.name_or_path == f"model_{i}"


# ─────────────────────────────────────────────────────────────────────────────
# prune_model tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPruneToConfigGuards:
    """Tests for prune_model input validation (no torch-pruning needed)."""

    def test_raises_import_error_when_tp_missing(self):
        from unittest.mock import patch

        with patch("silverspoon_kd.utils.TORCH_PRUNING_AVAILABLE", False):
            from silverspoon_kd.utils import prune_model

            model = SimpleModel(SimpleConfig())
            with pytest.raises(ImportError, match="torch-pruning"):
                prune_model(model, "test", example_inputs=torch.randint(0, 128, (1, 8)))

    def test_raises_value_error_when_no_example_inputs(self):
        from silverspoon_kd.utils import prune_model

        model = SimpleModel(SimpleConfig())
        with pytest.raises(ValueError, match="example_inputs"):
            prune_model(model, "test", example_inputs=None)


tp = pytest.importorskip("torch_pruning")


class TestPruneToConfig:
    """Tests for prune_model function (requires torch-pruning)."""

    @pytest.fixture
    def source_model(self):
        config = SimpleConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256,
        )
        return SimpleModel(config)

    @pytest.fixture
    def example_inputs(self):
        return torch.randint(0, 128, (1, 8))

    # ── Core shape/dimension tests ───────────────────────────────────────

    def test_prune_intermediate_size(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)
            assert layer.down_proj.weight.shape == (64, 128)

    def test_prune_intermediate_size_uneven(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 200},
            example_inputs=example_inputs,
        )
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (200, 64)
            assert layer.down_proj.weight.shape == (64, 200)

    def test_prune_fewer_layers(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"num_hidden_layers": 2},
            example_inputs=example_inputs,
        )
        assert len(result.layers) == 2

    def test_prune_width_and_depth_combined(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128, "num_hidden_layers": 2},
            example_inputs=example_inputs,
        )
        assert len(result.layers) == 2
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)
            assert layer.down_proj.weight.shape == (64, 128)

    def test_no_diff_returns_identical_copy(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={},
            example_inputs=example_inputs,
        )
        for (n1, p1), (n2, p2) in zip(
            source_model.named_parameters(), result.named_parameters(), strict=False
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Weights differ for {n1}"

    def test_no_diff_none(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff=None,
            example_inputs=example_inputs,
        )
        for (n1, p1), (_n2, p2) in zip(
            source_model.named_parameters(), result.named_parameters(), strict=False
        ):
            assert torch.equal(p1, p2), f"Weights differ for {n1}"

    def test_config_matches_target(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test_model",
            diff={"intermediate_size": 128, "num_hidden_layers": 2},
            example_inputs=example_inputs,
        )
        assert result.config.intermediate_size == 128
        assert result.config.num_hidden_layers == 2
        assert result.config.hidden_size == 64
        assert result.config.vocab_size == 1000
        assert result.config.num_attention_heads == 4

    # ── Weight integrity tests ───────────────────────────────────────────

    def test_pruned_weights_come_from_source(self, example_inputs):
        from silverspoon_kd.utils import prune_model

        config = SimpleConfig()
        model = SimpleModel(config)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.5)

        result = prune_model(
            model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        for name, param in result.named_parameters():
            assert torch.all(param == 0.5), (
                f"{name} has non-0.5 values: {param.min():.4f}-{param.max():.4f}"
            )

    def test_unchanged_weights_perfectly_preserved(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        # embed_tokens, lm_head, norm, attention projections, layer norms
        assert torch.equal(source_model.embed_tokens.weight, result.embed_tokens.weight)
        assert torch.equal(source_model.lm_head.weight, result.lm_head.weight)
        assert torch.equal(source_model.norm.weight, result.norm.weight)
        for i in range(len(result.layers)):
            src = source_model.layers[i]
            res = result.layers[i]
            assert torch.equal(src.v_proj.weight, res.v_proj.weight)
            assert torch.equal(src.o_proj.weight, res.o_proj.weight)
            assert torch.equal(src.input_layernorm.weight, res.input_layernorm.weight)
            assert torch.equal(
                src.post_attention_layernorm.weight,
                res.post_attention_layernorm.weight,
            )

    def test_pruned_weights_are_subset_of_source(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        # Each row of pruned up_proj.weight should exist in source up_proj.weight
        src_rows = {tuple(r.tolist()) for r in source_model.layers[0].up_proj.weight}
        for row in result.layers[0].up_proj.weight:
            assert tuple(row.tolist()) in src_rows

    def test_source_model_completely_unchanged(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        original_state = {n: p.clone() for n, p in source_model.named_parameters()}
        prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        for name, param in source_model.named_parameters():
            assert torch.equal(param, original_state[name]), f"Source {name} was modified"

    def test_independent_from_source(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={},
            example_inputs=example_inputs,
        )
        with torch.no_grad():
            result.embed_tokens.weight.fill_(999.0)
        assert not torch.equal(source_model.embed_tokens.weight, result.embed_tokens.weight)

    # ── Config and metadata tests ────────────────────────────────────────

    def test_name_or_path_set(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test_model",
            diff={},
            example_inputs=example_inputs,
        )
        assert result.config.name_or_path == "test_model"

    def test_preserves_model_class(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        assert type(result) is type(source_model)

    def test_original_config_unchanged(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        original_intermediate = source_model.config.intermediate_size
        original_name = source_model.config.name_or_path
        prune_model(
            source_model,
            "new_name",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        assert source_model.config.intermediate_size == original_intermediate
        assert source_model.config.name_or_path == original_name

    # ── Importance and customization tests ───────────────────────────────

    def test_importance_affects_selection(self, example_inputs):
        from silverspoon_kd.utils import prune_model

        config = SimpleConfig(intermediate_size=256, hidden_size=64)
        model = SimpleModel(config)
        # Make specific rows very large so they're "most important"
        with torch.no_grad():
            model.layers[0].up_proj.weight[:128].fill_(0.01)
            model.layers[0].up_proj.weight[128:].fill_(100.0)

        result = prune_model(
            model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        # The 128 large-weight rows should be kept
        assert torch.all(result.layers[0].up_proj.weight > 1.0)

    def test_custom_ignored_layers(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
            ignored_layers=[source_model.lm_head, source_model.embed_tokens],
        )
        # Shapes should still be correct
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)

    def test_custom_output_transform(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
            output_transform=lambda x: x,
        )
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_same_architecture_no_pruning(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={},
            example_inputs=example_inputs,
        )
        for (n1, p1), (_n2, p2) in zip(
            source_model.named_parameters(), result.named_parameters(), strict=False
        ):
            assert torch.equal(p1, p2), f"Weights differ for {n1}"

    def test_unknown_config_key_warning(self, source_model, example_inputs, caplog):
        import logging

        from silverspoon_kd.utils import prune_model

        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            prune_model(
                source_model,
                "test",
                diff={"unknown_key": 42},
                example_inputs=example_inputs,
            )
        assert "Config has no attribute 'unknown_key'" in caplog.text

    def test_prune_multiple_dimensions(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128, "vocab_size": 500},
            example_inputs=example_inputs,
        )
        assert result.config.intermediate_size == 128
        assert result.config.vocab_size == 500
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)

    def test_prune_to_single_layer(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"num_hidden_layers": 1},
            example_inputs=example_inputs,
        )
        assert len(result.layers) == 1

    def test_pruned_model_forward_pass(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128, "num_hidden_layers": 2},
            example_inputs=example_inputs,
        )
        output = result(example_inputs)
        assert output is not None

    def test_pruned_model_produces_correct_output_shape(self, source_model, example_inputs):
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        output = result(example_inputs)
        assert output.shape == (1, 8, 1000)

    # ── Logging tests ────────────────────────────────────────────────────

    def test_logs_pruning_summary(self, source_model, example_inputs, caplog):
        import logging

        from silverspoon_kd.utils import prune_model

        with caplog.at_level(logging.INFO, logger="silverspoon_kd.utils"):
            prune_model(
                source_model,
                "test",
                diff={"intermediate_size": 128},
                example_inputs=example_inputs,
            )
        assert "groups to match target config" in caplog.text

    def test_logs_group_details(self, source_model, example_inputs, caplog):
        import logging

        from silverspoon_kd.utils import prune_model

        with caplog.at_level(logging.DEBUG, logger="silverspoon_kd.utils"):
            prune_model(
                source_model,
                "test",
                diff={"intermediate_size": 128},
                example_inputs=example_inputs,
            )
        assert "channels" in caplog.text

    def test_logs_no_pruning_needed(self, source_model, example_inputs, caplog):
        import logging

        from silverspoon_kd.utils import prune_model

        with caplog.at_level(logging.INFO, logger="silverspoon_kd.utils"):
            prune_model(
                source_model,
                "test",
                diff={},
                example_inputs=example_inputs,
            )
        assert "No width pruning needed" in caplog.text

    def test_no_print_output(self, source_model, example_inputs, capsys):
        from silverspoon_kd.utils import prune_model

        prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        captured = capsys.readouterr()
        assert captured.out == ""


class TestPruneToConfigHeadAndGroupAware:
    """Tests for head-aware and group-aware pruning paths in prune_model."""

    @pytest.fixture
    def source_model(self):
        return SimpleModel(
            SimpleConfig(
                vocab_size=1000,
                hidden_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=256,
            )
        )

    @pytest.fixture
    def example_inputs(self):
        return torch.randint(0, 128, (1, 8))

    def test_head_aware_divisible_pruning(self, source_model, example_inputs):
        """Test head-aware path when n_to_prune is divisible by head_dim."""
        from unittest.mock import patch

        from silverspoon_kd.utils import prune_model

        def fake_detect(model, config):
            heads = {}
            for name, mod in model.named_modules():
                if name.endswith("up_proj"):
                    heads[mod] = 4
            return heads, {}

        with patch("silverspoon_kd.utils._auto_detect_head_structure", fake_detect):
            result = prune_model(
                source_model,
                "test",
                diff={"intermediate_size": 128},
                example_inputs=example_inputs,
            )

        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)
            assert layer.down_proj.weight.shape == (64, 128)

    def test_head_aware_non_divisible_fallback(self, source_model, example_inputs, caplog):
        """Test head-aware path falls back to per-channel when not divisible."""
        import logging
        from unittest.mock import patch

        from silverspoon_kd.utils import prune_model

        def fake_detect(model, config):
            heads = {}
            for name, mod in model.named_modules():
                if name.endswith("up_proj"):
                    heads[mod] = 3
            return heads, {}

        with patch("silverspoon_kd.utils._auto_detect_head_structure", fake_detect):
            with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
                result = prune_model(
                    source_model,
                    "test",
                    diff={"intermediate_size": 128},
                    example_inputs=example_inputs,
                )

        assert "not divisible by head_dim" in caplog.text
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)

    def test_group_aware_pruning(self, source_model, example_inputs):
        """Test out_channel_groups-aware pruning path."""
        from unittest.mock import patch

        from silverspoon_kd.utils import prune_model

        def fake_detect(model, config):
            groups = {}
            for name, mod in model.named_modules():
                if name.endswith("up_proj"):
                    groups[mod] = 2
            return {}, groups

        with patch("silverspoon_kd.utils._auto_detect_head_structure", fake_detect):
            result = prune_model(
                source_model,
                "test",
                diff={"intermediate_size": 128},
                example_inputs=example_inputs,
            )

        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)
            assert layer.down_proj.weight.shape == (64, 128)

    def test_head_aware_retains_important_heads(self, example_inputs):
        """Test that head-aware pruning keeps the most important heads."""
        from unittest.mock import patch

        from silverspoon_kd.utils import prune_model

        model = SimpleModel(SimpleConfig(intermediate_size=256))
        with torch.no_grad():
            model.layers[0].up_proj.weight[:128].fill_(0.01)
            model.layers[0].up_proj.weight[128:].fill_(100.0)

        def fake_detect(m, config):
            heads = {}
            for name, mod in m.named_modules():
                if name.endswith("up_proj"):
                    heads[mod] = 4
            return heads, {}

        with patch("silverspoon_kd.utils._auto_detect_head_structure", fake_detect):
            result = prune_model(
                model,
                "test",
                diff={"intermediate_size": 128},
                example_inputs=example_inputs,
            )

        assert torch.all(result.layers[0].up_proj.weight > 1.0)

    def test_group_aware_retains_important_channels(self, example_inputs):
        """Test that group-aware pruning keeps important channels per group."""
        from unittest.mock import patch

        from silverspoon_kd.utils import prune_model

        model = SimpleModel(SimpleConfig(intermediate_size=256))
        with torch.no_grad():
            w = model.layers[0].up_proj.weight
            w[:64].fill_(0.01)
            w[64:128].fill_(100.0)
            w[128:192].fill_(0.01)
            w[192:].fill_(100.0)

        def fake_detect(m, config):
            groups = {}
            for name, mod in m.named_modules():
                if name.endswith("up_proj"):
                    groups[mod] = 2
            return {}, groups

        with patch("silverspoon_kd.utils._auto_detect_head_structure", fake_detect):
            result = prune_model(
                model,
                "test",
                diff={"intermediate_size": 128},
                example_inputs=example_inputs,
            )

        assert torch.all(result.layers[0].up_proj.weight > 1.0)

    def test_increasing_layers_triggers_module_mismatch(self, source_model, example_inputs):
        """Test that increasing num_hidden_layers handles missing pruning modules."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"num_hidden_layers": 6},
            example_inputs=example_inputs,
        )
        # Pruned model keeps its original 4 layers (can't add layers via pruning)
        assert len(result.layers) == 4
        assert result.config.num_hidden_layers == 6


class TestPruneAttentionHeads:
    """Tests for attention head pruning (requires self_attn submodules)."""

    @pytest.fixture
    def source_model(self):
        config = SimpleConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=256,
        )
        return AttnSimpleModel(config)

    @pytest.fixture
    def example_inputs(self):
        return torch.randint(0, 128, (1, 8))

    def test_prune_attention_heads_q_o(self, source_model, example_inputs):
        """Test Q<->O coupled head pruning."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"num_attention_heads": 2},
            example_inputs=example_inputs,
        )
        head_dim = _SelfAttn.HEAD_DIM
        for layer in result.layers:
            assert layer.self_attn.q_proj.weight.shape == (2 * head_dim, 64)
            assert layer.self_attn.o_proj.weight.shape == (64, 2 * head_dim)

    def test_prune_attention_heads_k_v(self, source_model, example_inputs):
        """Test K<->V coupled head pruning."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"num_key_value_heads": 2},
            example_inputs=example_inputs,
        )
        head_dim = _SelfAttn.HEAD_DIM
        for layer in result.layers:
            assert layer.self_attn.k_proj.weight.shape == (2 * head_dim, 64)
            assert layer.self_attn.v_proj.weight.shape == (2 * head_dim, 64)

    def test_prune_all_heads_and_width(self, source_model, example_inputs):
        """Test combined attention head + MLP width pruning."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
            },
            example_inputs=example_inputs,
        )
        head_dim = _SelfAttn.HEAD_DIM
        for layer in result.layers:
            assert layer.self_attn.q_proj.weight.shape == (2 * head_dim, 64)
            assert layer.self_attn.k_proj.weight.shape == (2 * head_dim, 64)
            assert layer.up_proj.weight.shape == (128, 64)
            assert layer.down_proj.weight.shape == (64, 128)

    def test_prune_heads_retains_important(self, example_inputs):
        """Pruning keeps the heads with highest Q+O magnitude."""
        from silverspoon_kd.utils import prune_model

        config = SimpleConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model = AttnSimpleModel(config)
        head_dim = _SelfAttn.HEAD_DIM

        with torch.no_grad():
            # Make heads 2,3 very important (Q output channels)
            model.layers[0].self_attn.q_proj.weight[: 2 * head_dim].fill_(0.01)
            model.layers[0].self_attn.q_proj.weight[2 * head_dim :].fill_(100.0)
            # Make O input channels match
            model.layers[0].self_attn.o_proj.weight[:, : 2 * head_dim].fill_(0.01)
            model.layers[0].self_attn.o_proj.weight[:, 2 * head_dim :].fill_(100.0)

        result = prune_model(
            model,
            "test",
            diff={"num_attention_heads": 2},
            example_inputs=example_inputs,
        )
        # Heads 2,3 (channels 32-63) should be kept
        assert torch.all(result.layers[0].self_attn.q_proj.weight > 1.0)

    def test_attn_pruning_forward_pass(self, source_model, example_inputs):
        """Pruned model with fewer heads still produces valid output."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
            },
            example_inputs=example_inputs,
        )
        output = result(example_inputs)
        assert output.shape == (1, 8, 1000)

    def test_attn_pruning_source_unchanged(self, source_model, example_inputs):
        """Attention pruning does not modify source model."""
        from silverspoon_kd.utils import prune_model

        orig_q_shape = source_model.layers[0].self_attn.q_proj.weight.shape
        prune_model(
            source_model,
            "test",
            diff={"num_attention_heads": 2},
            example_inputs=example_inputs,
        )
        assert source_model.layers[0].self_attn.q_proj.weight.shape == orig_q_shape

    def test_attn_pruning_logs_summary(self, source_model, example_inputs, caplog):
        """Attention pruning logs the pruning summary."""
        import logging

        from silverspoon_kd.utils import prune_model

        with caplog.at_level(logging.INFO, logger="silverspoon_kd.utils"):
            prune_model(
                source_model,
                "test",
                diff={"num_attention_heads": 2},
                example_inputs=example_inputs,
            )
        assert "attention" in caplog.text

    def test_attn_pruning_round_to(self, example_inputs):
        """round_to adjusts attention channel targets."""
        from silverspoon_kd.utils import prune_model

        # 3 heads * 16 head_dim = 48 channels. round_to=32 -> 32 channels
        config = SimpleConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model = AttnSimpleModel(config)
        result = prune_model(
            model,
            "test",
            diff={"num_attention_heads": 3},
            example_inputs=example_inputs,
            round_to=32,
        )
        assert result.layers[0].self_attn.q_proj.weight.shape[0] == 32

    def test_gqa_prune_kv_fewer_than_q(self, example_inputs):
        """GQA model: prune KV heads while keeping Q heads."""
        from silverspoon_kd.utils import prune_model

        config = SimpleConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model = AttnSimpleModel(config)
        result = prune_model(
            model,
            "test",
            diff={"num_key_value_heads": 2},
            example_inputs=example_inputs,
        )
        head_dim = _SelfAttn.HEAD_DIM
        # Q unchanged, KV pruned
        assert result.layers[0].self_attn.q_proj.weight.shape == (4 * head_dim, 64)
        assert result.layers[0].self_attn.k_proj.weight.shape == (2 * head_dim, 64)
        assert result.layers[0].self_attn.v_proj.weight.shape == (2 * head_dim, 64)


class TestPruneModelRoundTo:
    """Tests for round_to parameter in prune_model."""

    @pytest.fixture
    def source_model(self):
        return SimpleModel(
            SimpleConfig(
                vocab_size=1000,
                hidden_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=256,
            )
        )

    @pytest.fixture
    def example_inputs(self):
        return torch.randint(0, 128, (1, 8))

    def test_round_to_rounds_down_intermediate(self, source_model, example_inputs):
        """round_to rounds intermediate_size down to nearest multiple."""
        from silverspoon_kd.utils import prune_model

        # Target 200, round_to=8 -> 200 (already aligned)
        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 200},
            example_inputs=example_inputs,
            round_to=8,
        )
        for layer in result.layers:
            assert layer.up_proj.weight.shape[0] == 200

    def test_round_to_adjusts_unaligned_target(self, source_model, example_inputs):
        """round_to adjusts unaligned target down to nearest multiple."""
        from silverspoon_kd.utils import prune_model

        # Target 201, round_to=8 -> 200
        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 201},
            example_inputs=example_inputs,
            round_to=8,
        )
        for layer in result.layers:
            assert layer.up_proj.weight.shape[0] == 200

    def test_round_to_none_no_rounding(self, source_model, example_inputs):
        """round_to=None preserves exact target."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 201},
            example_inputs=example_inputs,
            round_to=None,
        )
        for layer in result.layers:
            assert layer.up_proj.weight.shape[0] == 201

    def test_round_to_1_no_rounding(self, source_model, example_inputs):
        """round_to=1 has no effect."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 201},
            example_inputs=example_inputs,
            round_to=1,
        )
        for layer in result.layers:
            assert layer.up_proj.weight.shape[0] == 201

    def test_round_to_preserves_forward_pass(self, source_model, example_inputs):
        """Model with rounded dimensions still produces valid output."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 201},
            example_inputs=example_inputs,
            round_to=8,
        )
        output = result(example_inputs)
        assert output.shape == (1, 8, 1000)


class TestRoundChannels:
    """Tests for _round_channels helper."""

    def test_none_no_rounding(self):
        from silverspoon_kd.utils import _round_channels

        assert _round_channels(201, None) == 201

    def test_round_to_1_no_rounding(self):
        from silverspoon_kd.utils import _round_channels

        assert _round_channels(201, 1) == 201

    def test_round_down(self):
        from silverspoon_kd.utils import _round_channels

        assert _round_channels(201, 8) == 200

    def test_already_aligned(self):
        from silverspoon_kd.utils import _round_channels

        assert _round_channels(200, 8) == 200

    def test_minimum_is_round_to(self):
        from silverspoon_kd.utils import _round_channels

        assert _round_channels(3, 8) == 8

    def test_exact_multiple(self):
        from silverspoon_kd.utils import _round_channels

        assert _round_channels(64, 4) == 64

    def test_round_to_4(self):
        from silverspoon_kd.utils import _round_channels

        assert _round_channels(127, 4) == 124


class TestGroupImportance:
    """Tests for group magnitude importance (mean reduction across coupled layers)."""

    @pytest.fixture
    def example_inputs(self):
        return torch.randint(0, 128, (1, 8))

    def test_group_importance_considers_coupled_layers(self, example_inputs):
        """Group importance should consider both up_proj and down_proj weights."""
        from silverspoon_kd.utils import prune_model

        model = SimpleModel(SimpleConfig(intermediate_size=256))
        with torch.no_grad():
            # up_proj: first 128 channels have high weight, last 128 low
            model.layers[0].up_proj.weight[:128].fill_(100.0)
            model.layers[0].up_proj.weight[128:].fill_(0.01)
            # down_proj: reverse pattern (first 128 input channels low, last 128 high)
            model.layers[0].down_proj.weight[:, :128].fill_(0.01)
            model.layers[0].down_proj.weight[:, 128:].fill_(100.0)

        result = prune_model(
            model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        # With group mean importance, both sets of channels have similar importance
        # (high up + low down vs low up + high down). The result should still
        # produce a valid model with correct shapes.
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (128, 64)
            assert layer.down_proj.weight.shape == (64, 128)

    def test_group_importance_retains_jointly_important(self, example_inputs):
        """Channels important in BOTH coupled layers should be preferred."""
        from silverspoon_kd.utils import prune_model

        model = SimpleModel(SimpleConfig(intermediate_size=256))
        with torch.no_grad():
            # Channels 128-255 are important in BOTH up_proj and down_proj
            model.layers[0].up_proj.weight[:128].fill_(0.01)
            model.layers[0].up_proj.weight[128:].fill_(100.0)
            model.layers[0].down_proj.weight[:, :128].fill_(0.01)
            model.layers[0].down_proj.weight[:, 128:].fill_(100.0)

        result = prune_model(
            model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        # Channels 128-255 have high importance in both layers, so they should be kept
        assert torch.all(result.layers[0].up_proj.weight > 1.0)


class TestPruneModelFreezeCopiedWeights:
    """Tests for freeze_copied_weights parameter in prune_model."""

    @pytest.fixture
    def source_model(self):
        return SimpleModel(
            SimpleConfig(
                vocab_size=1000,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=256,
            )
        )

    @pytest.fixture
    def example_inputs(self):
        return torch.randint(0, 128, (1, 8))

    def test_freeze_unchanged_params(self, source_model, example_inputs):
        """freeze_copied_weights=True freezes params with unchanged shapes."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
            freeze_copied_weights=True,
        )
        # Unchanged params (embed_tokens, lm_head, norm, attention projections)
        assert not result.embed_tokens.weight.requires_grad
        assert not result.lm_head.weight.requires_grad
        assert not result.norm.weight.requires_grad
        assert not result.layers[0].v_proj.weight.requires_grad

    def test_pruned_params_remain_trainable(self, source_model, example_inputs):
        """freeze_copied_weights=True leaves pruned params trainable."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
            freeze_copied_weights=True,
        )
        # Pruned params (up_proj, down_proj changed shape)
        assert result.layers[0].up_proj.weight.requires_grad
        assert result.layers[0].down_proj.weight.requires_grad

    def test_freeze_false_all_trainable(self, source_model, example_inputs):
        """freeze_copied_weights=False keeps all params trainable."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
            freeze_copied_weights=False,
        )
        for name, param in result.named_parameters():
            assert param.requires_grad, f"{name} should be trainable"

    def test_freeze_default_is_false(self, source_model, example_inputs):
        """Default freeze_copied_weights is False."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={"intermediate_size": 128},
            example_inputs=example_inputs,
        )
        for name, param in result.named_parameters():
            assert param.requires_grad, f"{name} should be trainable by default"

    def test_freeze_logs_summary(self, source_model, example_inputs, caplog):
        """freeze_copied_weights logs frozen/trainable parameter counts."""
        import logging

        from silverspoon_kd.utils import prune_model

        with caplog.at_level(logging.INFO, logger="silverspoon_kd.utils"):
            prune_model(
                source_model,
                "test",
                diff={"intermediate_size": 128},
                example_inputs=example_inputs,
                freeze_copied_weights=True,
            )
        assert "Froze" in caplog.text
        assert "trainable" in caplog.text

    def test_freeze_no_pruning_freezes_all(self, source_model, example_inputs):
        """When no pruning is done, all params are frozen."""
        from silverspoon_kd.utils import prune_model

        result = prune_model(
            source_model,
            "test",
            diff={},
            example_inputs=example_inputs,
            freeze_copied_weights=True,
        )
        for name, param in result.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_freeze_with_attn_pruning(self, example_inputs):
        """freeze_copied_weights works with attention head pruning."""
        from silverspoon_kd.utils import prune_model

        config = SimpleConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model = AttnSimpleModel(config)
        result = prune_model(
            model,
            "test",
            diff={"num_attention_heads": 2, "intermediate_size": 128},
            example_inputs=example_inputs,
            freeze_copied_weights=True,
        )
        # Attention projections were pruned -> trainable
        assert result.layers[0].self_attn.q_proj.weight.requires_grad
        assert result.layers[0].self_attn.o_proj.weight.requires_grad
        # Embeddings unchanged -> frozen
        assert not result.embed_tokens.weight.requires_grad


class TestPruneModelLargeModel:
    """Test that prune_model handles a deep model."""

    def test_prunes_every_layer_of_deep_model(self):
        """prune_model resizes every layer of a 24-layer model."""
        from silverspoon_kd.utils import prune_model

        # Use a larger model (24 layers, 512 hidden) so graph traversal spans many layers
        config = SimpleConfig(
            vocab_size=1000,
            hidden_size=512,
            num_hidden_layers=24,
            intermediate_size=2048,
        )
        model = SimpleModel(config)
        example_inputs = torch.randint(0, 128, (1, 8))

        result = prune_model(
            model,
            "test",
            diff={"intermediate_size": 1024},
            example_inputs=example_inputs,
        )

        assert len(result.layers) == 24
        for layer in result.layers:
            assert layer.up_proj.weight.shape == (1024, 512)


class TestAutoDetection:
    """Tests for auto-detection helper functions (requires torch-pruning)."""

    @pytest.fixture
    def source_model(self):
        return SimpleModel(SimpleConfig())

    def test_auto_detect_ignored_layers_finds_lm_head(self, source_model):
        from silverspoon_kd.utils import _auto_detect_ignored_layers

        ignored = _auto_detect_ignored_layers(source_model)
        assert source_model.lm_head in ignored

    def test_auto_detect_ignored_layers_finds_embed_tokens(self, source_model):
        from silverspoon_kd.utils import _auto_detect_ignored_layers

        ignored = _auto_detect_ignored_layers(source_model)
        assert source_model.embed_tokens in ignored

    def test_auto_detect_ignored_layers_empty_for_plain_model(self):
        from silverspoon_kd.utils import _auto_detect_ignored_layers

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        ignored = _auto_detect_ignored_layers(model)
        assert len(ignored) == 0

    def test_auto_detect_head_structure_warns_when_not_found(self, source_model, caplog):
        import logging

        from silverspoon_kd.utils import _auto_detect_head_structure

        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            heads, _groups = _auto_detect_head_structure(source_model, source_model.config)
        assert len(heads) == 0
        assert "Could not auto-detect" in caplog.text

    def test_remove_excess_layers_truncates(self, source_model):
        from copy import deepcopy

        from silverspoon_kd.utils import _remove_excess_layers

        model = deepcopy(source_model)
        assert _remove_excess_layers(model, 2) is True
        assert len(model.layers) == 2

    def test_remove_excess_layers_updates_config(self, source_model):
        from copy import deepcopy

        from silverspoon_kd.utils import _remove_excess_layers

        model = deepcopy(source_model)
        _remove_excess_layers(model, 2)
        assert model.config.num_hidden_layers == 2

    def test_remove_excess_layers_preserves_first_n(self, source_model):
        from copy import deepcopy

        from silverspoon_kd.utils import _remove_excess_layers

        model = deepcopy(source_model)
        orig_w0 = source_model.layers[0].v_proj.weight.clone()
        orig_w1 = source_model.layers[1].v_proj.weight.clone()
        _remove_excess_layers(model, 2)
        assert torch.equal(model.layers[0].v_proj.weight, orig_w0)
        assert torch.equal(model.layers[1].v_proj.weight, orig_w1)

    def test_remove_excess_layers_warns_if_no_modulelist(self, caplog):
        import logging

        from silverspoon_kd.utils import _remove_excess_layers

        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        model.config = SimpleConfig(num_hidden_layers=3)
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            result = _remove_excess_layers(model, 1)
        assert result is False
        assert "Could not find ModuleList" in caplog.text

    def test_validate_pruned_dimensions_exact_match(self, source_model, caplog):
        import logging
        from copy import deepcopy

        from silverspoon_kd.utils import _validate_pruned_dimensions

        target = deepcopy(source_model)
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            _validate_pruned_dimensions(source_model, target)
        # No warnings should be logged
        assert "Shape mismatches after pruning" not in caplog.text

    def test_validate_pruned_dimensions_mismatch_warns(self, caplog):
        import logging

        from silverspoon_kd.utils import _validate_pruned_dimensions

        pruned = SimpleModel(SimpleConfig(intermediate_size=128))
        target = SimpleModel(SimpleConfig(intermediate_size=256))
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            _validate_pruned_dimensions(pruned, target)
        assert "Shape mismatches after pruning" in caplog.text
        assert "up_proj" in caplog.text

    def test_remove_excess_layers_no_config_attr(self, caplog):
        import logging

        from silverspoon_kd.utils import _remove_excess_layers

        model = nn.Sequential(nn.Linear(10, 20))
        model.config = type("Config", (), {})()  # Config without num_hidden_layers
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            result = _remove_excess_layers(model, 2)
        assert result is False
        assert "no num_hidden_layers" in caplog.text

    def test_auto_detect_head_structure_finds_self_attn(self):
        """Test head detection with a model that has self_attn submodules."""
        from silverspoon_kd.utils import _auto_detect_head_structure

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

        model = FakeLayer()
        config = SimpleConfig(num_attention_heads=4)
        heads, _groups = _auto_detect_head_structure(model, config)
        assert len(heads) == 3  # q, k, v
        assert model.self_attn.q_proj in heads
        assert heads[model.self_attn.q_proj] == 4

    def test_auto_detect_head_structure_finds_qkv_proj(self):
        """Test head detection with a fused qkv_proj."""
        from silverspoon_kd.utils import _auto_detect_head_structure

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv_proj = nn.Linear(64, 192)

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

        model = FakeLayer()
        config = SimpleConfig(num_attention_heads=4)
        heads, _groups = _auto_detect_head_structure(model, config)
        assert model.self_attn.qkv_proj in heads
        assert heads[model.self_attn.qkv_proj] == 4

    def test_auto_detect_head_structure_finds_gate_up_proj(self):
        """Test detection of fused gate_up_proj."""
        from silverspoon_kd.utils import _auto_detect_head_structure

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.gate_up_proj = nn.Linear(64, 512)

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

        model = FakeLayer()
        config = SimpleConfig(num_attention_heads=4)
        _heads, groups = _auto_detect_head_structure(model, config)
        assert model.self_attn.gate_up_proj in groups
        assert groups[model.self_attn.gate_up_proj] == 2

    def test_validate_pruned_dimensions_missing_param(self, caplog):
        """Test validation when pruned model is missing a parameter."""
        import logging

        from silverspoon_kd.utils import _validate_pruned_dimensions

        # Create two models where one has extra params
        pruned = nn.Linear(10, 5)
        target = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            _validate_pruned_dimensions(pruned, target)
        assert "missing in pruned model" in caplog.text


# ----------------------------------------------------------------------
# Direct unit tests for the private helpers extracted from
# ``_prune_attention_heads`` and ``prune_model``.  Before these were
# extracted, the only coverage came from the public functions' end-to-end
# tests, which made it hard to pin down regressions to a specific helper.
# ----------------------------------------------------------------------


class TestKeptHeadIndices:
    """Tests for ``_kept_head_indices`` (used by Q↔O / K↔V coupling)."""

    def _helper(self):
        from silverspoon_kd.utils import _kept_head_indices

        return _kept_head_indices

    def test_divisible_returns_whole_head_blocks(self):
        helper = self._helper()
        # 4 heads × 4 head_dim = 16 channels.  Keep 2 heads (= 8 indices).
        # Score head 1 and head 3 highest so they get picked.
        scores = torch.tensor(
            [
                0.1,
                0.1,
                0.1,
                0.1,  # head 0 mean = 0.1
                0.9,
                0.9,
                0.9,
                0.9,  # head 1 mean = 0.9 (KEEP)
                0.2,
                0.2,
                0.2,
                0.2,  # head 2 mean = 0.2
                0.8,
                0.8,
                0.8,
                0.8,  # head 3 mean = 0.8 (KEEP)
            ]
        )
        keep = helper(scores, n_heads=4, target_channels=8)
        # Expected: indices for heads 1 and 3, sorted: [4,5,6,7,12,13,14,15]
        assert keep.tolist() == [4, 5, 6, 7, 12, 13, 14, 15]

    def test_non_divisible_falls_back_to_per_channel(self):
        helper = self._helper()
        # 4 heads × 4 head_dim = 16 channels.  Target 14 (not a head boundary).
        scores = torch.arange(16, dtype=torch.float32)
        # Keep top-14 channels = drop the 2 lowest (indices 0, 1)
        keep = helper(scores, n_heads=4, target_channels=14)
        # All indices except [0, 1], sorted
        assert keep.tolist() == list(range(2, 16))

    def test_returns_sorted_indices(self):
        helper = self._helper()
        # Force head 0 (lowest score) to be dropped
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
        keep = helper(scores, n_heads=3, target_channels=8)
        # Should keep heads 1 and 2 = indices [4..11]
        sorted_keep = sorted(keep.tolist())
        assert keep.tolist() == sorted_keep
        assert keep.tolist() == [4, 5, 6, 7, 8, 9, 10, 11]

    def test_keeps_top_scoring_heads(self):
        helper = self._helper()
        # 6 heads × 2 = 12 channels; keep 3 heads
        scores = torch.tensor(
            [
                5.0,
                5.0,  # head 0
                1.0,
                1.0,  # head 1
                4.0,
                4.0,  # head 2
                2.0,
                2.0,  # head 3
                3.0,
                3.0,  # head 4
                6.0,
                6.0,  # head 5
            ]
        )
        # Top 3 head means: head 5 (6), head 0 (5), head 2 (4)
        keep = helper(scores, n_heads=6, target_channels=6)
        # Expected indices for heads 0, 2, 5: [0,1,4,5,10,11]
        assert keep.tolist() == [0, 1, 4, 5, 10, 11]


class TestPrunedIndicesHeadAware:
    """Tests for ``_pruned_indices_head_aware`` (used by width-pruning loop)."""

    def _helper(self):
        from silverspoon_kd.utils import _pruned_indices_head_aware

        return _pruned_indices_head_aware

    def test_divisible_drops_whole_head_blocks(self):
        helper = self._helper()
        # 4 heads × 4 head_dim = 16 channels.  Drop 2 heads (= 8 indices).
        # Score head 0 and head 2 lowest so they get dropped.
        scores = torch.tensor(
            [
                0.1,
                0.1,
                0.1,
                0.1,  # head 0 (DROP)
                0.9,
                0.9,
                0.9,
                0.9,  # head 1
                0.2,
                0.2,
                0.2,
                0.2,  # head 2 (DROP)
                0.8,
                0.8,
                0.8,
                0.8,  # head 3
            ]
        )
        prune = helper(scores, n_heads=4, n_to_prune=8, module_label="layers.0.q_proj")
        # Drop indices for heads 0 and 2: [0..3, 8..11]
        assert prune.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]

    def test_non_divisible_warns_and_falls_back(self, caplog):
        import logging

        helper = self._helper()
        scores = torch.arange(16, dtype=torch.float32)
        # n_to_prune=2 / head_dim=4 is not divisible
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            prune = helper(scores, n_heads=4, n_to_prune=2, module_label="layers.5.up_proj")

        assert "not divisible by head_dim" in caplog.text
        # Should fall back to per-channel: pick the 2 lowest scores
        assert prune.tolist() == [0, 1]

    def test_warning_includes_module_label(self, caplog):
        import logging

        helper = self._helper()
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            helper(
                torch.arange(16, dtype=torch.float32),
                n_heads=4,
                n_to_prune=2,
                module_label="some.module.name",
            )
        assert "some.module.name" in caplog.text

    def test_returns_sorted_indices(self):
        helper = self._helper()
        prune = helper(
            torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0, 6.0]),
            n_heads=2,
            n_to_prune=3,
            module_label="x",
        )
        assert prune.tolist() == sorted(prune.tolist())


class TestPrunedIndicesGroupAware:
    """Tests for ``_pruned_indices_group_aware`` (used for fused gate_up_proj)."""

    def _helper(self):
        from silverspoon_kd.utils import _pruned_indices_group_aware

        return _pruned_indices_group_aware

    def test_preserves_group_structure(self):
        helper = self._helper()
        # 2 groups × 4 channels = 8 channels.  Target 4 channels (2 per group).
        scores = torch.tensor(
            [
                # group 0:  worst at 0 and 1
                0.1,
                0.2,
                0.9,
                0.8,
                # group 1:  worst at 4 and 5
                0.1,
                0.2,
                0.9,
                0.8,
            ]
        )
        prune = helper(scores, n_groups=2, current_channels=8, target_channels=4)
        # Expected: drop indices [0, 1, 4, 5]
        assert prune.tolist() == [0, 1, 4, 5]

    def test_three_groups(self):
        helper = self._helper()
        # 3 groups × 4 channels = 12 channels.  Target 6 channels (2 per group).
        scores = torch.tensor(
            [
                0.0,
                0.1,
                0.9,
                0.8,  # group 0
                0.0,
                0.1,
                0.9,
                0.8,  # group 1
                0.0,
                0.1,
                0.9,
                0.8,  # group 2
            ]
        )
        prune = helper(scores, n_groups=3, current_channels=12, target_channels=6)
        assert prune.tolist() == [0, 1, 4, 5, 8, 9]

    def test_returns_sorted_indices(self):
        helper = self._helper()
        # Make worst index of each group out of order to verify sorting
        scores = torch.tensor([2.0, 0.0, 5.0, 1.0])
        prune = helper(scores, n_groups=2, current_channels=4, target_channels=2)
        assert prune.tolist() == sorted(prune.tolist())


class TestSliceProjectionOutChannels:
    """Tests for ``_slice_projection_out_channels``."""

    def _helper(self):
        from silverspoon_kd.utils import _slice_projection_out_channels

        return _slice_projection_out_channels

    def test_slices_weight_along_out_dim(self):
        helper = self._helper()
        proj = nn.Linear(8, 16, bias=True)
        original_weight = proj.weight.data.clone()
        keep = torch.tensor([0, 2, 5, 7, 9, 11, 13, 15])
        helper(proj, keep)
        # New weight should match original at the kept rows
        assert proj.weight.shape == (8, 8)
        assert torch.allclose(proj.weight.data, original_weight[keep])

    def test_slices_bias_when_present(self):
        helper = self._helper()
        proj = nn.Linear(8, 16, bias=True)
        original_bias = proj.bias.data.clone()
        keep = torch.tensor([1, 3, 5, 7])
        helper(proj, keep)
        assert proj.bias.shape == (4,)
        assert torch.allclose(proj.bias.data, original_bias[keep])

    def test_skips_bias_when_none(self):
        helper = self._helper()
        proj = nn.Linear(8, 16, bias=False)
        assert proj.bias is None
        keep = torch.tensor([0, 1, 2])
        helper(proj, keep)
        # Should not raise; bias remains None
        assert proj.bias is None
        assert proj.weight.shape == (3, 8)

    def test_updates_out_features(self):
        helper = self._helper()
        proj = nn.Linear(8, 16, bias=True)
        keep = torch.tensor([0, 1, 2, 3, 4])
        helper(proj, keep)
        assert proj.out_features == 5


class TestPruneQOCoupling:
    """Tests for ``_prune_qo_coupling``."""

    def _helper(self):
        from silverspoon_kd.utils import _prune_qo_coupling

        return _prune_qo_coupling

    def test_returns_false_when_target_ge_source(self):
        helper = self._helper()
        q = nn.Linear(64, 32)
        o = nn.Linear(32, 64)
        target_q = nn.Linear(64, 32)  # same out_features → no pruning
        original_q_weight = q.weight.data.clone()
        original_o_weight = o.weight.data.clone()

        result = helper(
            "self_attn",
            q,
            o,
            target_q,
            num_heads={q: 4},
            round_to=None,
        )
        assert result is False
        # Weights must be unchanged
        assert torch.allclose(q.weight.data, original_q_weight)
        assert torch.allclose(o.weight.data, original_o_weight)
        assert q.out_features == 32
        assert o.in_features == 32

    def test_prunes_q_out_and_o_in_to_target(self):
        helper = self._helper()
        # Source: 4 heads * 16 head_dim = 64 channels
        # Target: 2 heads * 16 head_dim = 32 channels
        q = nn.Linear(64, 64)
        o = nn.Linear(64, 64)
        target_q = nn.Linear(64, 32)

        result = helper(
            "self_attn",
            q,
            o,
            target_q,
            num_heads={q: 4},
            round_to=None,
        )
        assert result is True
        assert q.out_features == 32
        assert o.in_features == 32
        assert q.weight.shape == (32, 64)
        assert o.weight.shape == (64, 32)

    def test_logs_with_module_name(self, caplog):
        import logging

        helper = self._helper()
        q = nn.Linear(64, 64)
        o = nn.Linear(64, 64)
        target_q = nn.Linear(64, 32)
        with caplog.at_level(logging.DEBUG, logger="silverspoon_kd.utils"):
            helper(
                "model.layers.0.self_attn",
                q,
                o,
                target_q,
                num_heads={q: 4},
                round_to=None,
            )
        assert "model.layers.0.self_attn" in caplog.text
        assert "Q↔O" in caplog.text
        # Should log original -> target
        assert "64" in caplog.text and "32" in caplog.text

    def test_no_op_when_target_equal_source(self):
        """target_q.out_features == q.out_features should also be a no-op."""
        helper = self._helper()
        q = nn.Linear(64, 32)
        o = nn.Linear(32, 64)
        target_q = nn.Linear(64, 32)  # equal, not less
        result = helper(
            "x",
            q,
            o,
            target_q,
            num_heads={q: 4},
            round_to=None,
        )
        assert result is False

    def test_uses_default_n_heads_when_not_in_dict(self):
        """If num_heads dict doesn't contain q_proj, fall back to per-channel."""
        helper = self._helper()
        q = nn.Linear(8, 8)
        o = nn.Linear(8, 8)
        target_q = nn.Linear(8, 4)
        # Empty num_heads dict — falls back to q.out_features as n_heads_val
        result = helper("x", q, o, target_q, num_heads={}, round_to=None)
        assert result is True
        assert q.out_features == 4
        assert o.in_features == 4


class TestPruneKVCoupling:
    """Tests for ``_prune_kv_coupling``."""

    def _helper(self):
        from silverspoon_kd.utils import _prune_kv_coupling

        return _prune_kv_coupling

    def test_returns_false_when_target_ge_source(self):
        helper = self._helper()
        k = nn.Linear(64, 32)
        v = nn.Linear(64, 32)
        target_k = nn.Linear(64, 32)
        original_k_weight = k.weight.data.clone()
        original_v_weight = v.weight.data.clone()
        result = helper(
            "self_attn",
            k,
            v,
            target_k,
            num_heads={k: 2},
            round_to=None,
        )
        assert result is False
        assert torch.allclose(k.weight.data, original_k_weight)
        assert torch.allclose(v.weight.data, original_v_weight)

    def test_prunes_both_k_and_v_out_channels(self):
        helper = self._helper()
        # Source: 4 KV heads * 16 head_dim = 64 channels
        # Target: 2 KV heads * 16 head_dim = 32 channels
        k = nn.Linear(64, 64)
        v = nn.Linear(64, 64)
        target_k = nn.Linear(64, 32)
        result = helper(
            "self_attn",
            k,
            v,
            target_k,
            num_heads={k: 4},
            round_to=None,
        )
        assert result is True
        # Both k and v should be pruned along the OUT dim
        assert k.out_features == 32
        assert v.out_features == 32
        assert k.weight.shape == (32, 64)
        assert v.weight.shape == (32, 64)

    def test_logs_with_module_name(self, caplog):
        import logging

        helper = self._helper()
        k = nn.Linear(64, 64)
        v = nn.Linear(64, 64)
        target_k = nn.Linear(64, 32)
        with caplog.at_level(logging.DEBUG, logger="silverspoon_kd.utils"):
            helper(
                "layers.7.self_attn",
                k,
                v,
                target_k,
                num_heads={k: 4},
                round_to=None,
            )
        assert "layers.7.self_attn" in caplog.text
        assert "K↔V" in caplog.text


class TestCollectAttentionLinearModules:
    """Tests for ``_collect_attention_linear_modules``."""

    def _helper(self):
        from silverspoon_kd.utils import _collect_attention_linear_modules

        return _collect_attention_linear_modules

    def test_collects_only_linears_in_attention_blocks(self):
        helper = self._helper()

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.norm = nn.LayerNorm(8)  # not a Linear, should be skipped

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attn()
                self.mlp = nn.Linear(8, 8)  # outside attention, should be skipped

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(), Block()])

        model = Model()
        attn_linears = helper(model)

        # Should collect q_proj and k_proj from BOTH layers (4 total)
        assert len(attn_linears) == 4
        for layer in model.layers:
            assert layer.self_attn.q_proj in attn_linears
            assert layer.self_attn.k_proj in attn_linears
            # Non-attention MLP should NOT be collected
            assert layer.mlp not in attn_linears

    def test_returns_empty_for_model_without_attention(self):
        helper = self._helper()
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        assert helper(model) == set()


class TestBuildTargetOutChannelMap:
    """Tests for ``_build_target_out_channel_map``."""

    def _helper(self):
        from silverspoon_kd.utils import _build_target_out_channel_map

        return _build_target_out_channel_map

    def test_includes_only_shrinking_modules(self):
        helper = self._helper()

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.shrinking = nn.Linear(8, 16)  # shrinks 16 -> 8
                self.growing = nn.Linear(8, 4)  # would grow 4 -> 8 — skip
                self.same = nn.Linear(8, 8)  # equal — skip

        pruning = M()
        target = M()
        target.shrinking = nn.Linear(8, 8)
        target.growing = nn.Linear(8, 8)
        target.same = nn.Linear(8, 8)

        pruning_name_to_module = dict(pruning.named_modules())
        result = helper(
            target_model=target,
            pruning_name_to_module=pruning_name_to_module,
            ignored_set=set(),
            attn_modules=set(),
            round_to=None,
        )
        assert pruning.shrinking in result
        assert result[pruning.shrinking] == 8
        assert pruning.growing not in result
        assert pruning.same not in result

    def test_skips_ignored_modules(self):
        helper = self._helper()

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 16)
                self.b = nn.Linear(8, 16)

        pruning = M()
        target = M()
        target.a = nn.Linear(8, 8)
        target.b = nn.Linear(8, 8)

        result = helper(
            target_model=target,
            pruning_name_to_module=dict(pruning.named_modules()),
            ignored_set={pruning.a},
            attn_modules=set(),
            round_to=None,
        )
        assert pruning.a not in result
        assert pruning.b in result

    def test_skips_attn_modules(self):
        helper = self._helper()

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 16)
                self.mlp = nn.Linear(8, 16)

        pruning = M()
        target = M()
        target.q_proj = nn.Linear(8, 8)
        target.mlp = nn.Linear(8, 8)

        result = helper(
            target_model=target,
            pruning_name_to_module=dict(pruning.named_modules()),
            ignored_set=set(),
            attn_modules={pruning.q_proj},
            round_to=None,
        )
        assert pruning.q_proj not in result
        assert pruning.mlp in result

    def test_round_to_aligns_target(self):
        helper = self._helper()

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(8, 100)

        pruning = M()
        target = M()
        target.layer = nn.Linear(8, 50)  # 50 not divisible by 8

        result = helper(
            target_model=target,
            pruning_name_to_module=dict(pruning.named_modules()),
            ignored_set=set(),
            attn_modules=set(),
            round_to=8,
        )
        # 50 rounded down to nearest multiple of 8 = 48
        assert result[pruning.layer] == 48


class TestFreezeUnchangedAfterPruning:
    """Tests for ``_freeze_unchanged_after_pruning``."""

    def _helper(self):
        from silverspoon_kd.utils import _freeze_unchanged_after_pruning

        return _freeze_unchanged_after_pruning

    def test_freezes_only_unchanged_parameters(self):
        helper = self._helper()

        class M(nn.Module):
            def __init__(self, dim_a, dim_b):
                super().__init__()
                self.frozen_layer = nn.Linear(8, 8)  # unchanged
                self.pruned_layer = nn.Linear(dim_a, dim_b)  # changed shape

        source = M(8, 16)
        pruned = M(8, 8)  # pruned_layer shrunk

        # Make frozen_layer match source exactly
        pruned.frozen_layer.load_state_dict(source.frozen_layer.state_dict())

        helper(pruned, source)

        # Frozen layer should be frozen (matches source shape)
        assert not pruned.frozen_layer.weight.requires_grad
        assert not pruned.frozen_layer.bias.requires_grad
        # Pruned layer should remain trainable
        assert pruned.pruned_layer.weight.requires_grad
        assert pruned.pruned_layer.bias.requires_grad

    def test_no_op_when_nothing_unchanged(self):
        """When all parameters differ from the source, no freezing happens."""
        helper = self._helper()

        source = nn.Linear(8, 16)
        pruned = nn.Linear(8, 8)  # entirely different

        helper(pruned, source)
        # All params still trainable
        assert pruned.weight.requires_grad
        assert pruned.bias.requires_grad


class TestCoupledImportanceScores:
    """Tests for ``_coupled_importance_scores``.

    Uses a tiny model + DependencyGraph to drive the helper through its
    coupled-layers averaging path.  This is the only helper that needs the
    real torch-pruning DG, so we keep the model trivially small.
    """

    @pytest.fixture
    def tp_module(self):
        try:
            import torch_pruning as tp
        except ImportError:
            pytest.skip("torch-pruning not installed")
        return tp

    def test_falls_back_to_module_scores_when_no_couplings(self, tp_module):
        from silverspoon_kd.utils import _coupled_importance_scores

        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        dep_graph = tp_module.DependencyGraph().build_dependency(
            model,
            example_inputs=torch.randn(1, 8),
            ignored_layers=[model[2]],
        )

        target_module = model[0]
        pruner = dep_graph.get_pruner_of_module(target_module)
        scores = _coupled_importance_scores(target_module, dep_graph, pruner, ignored_set=set())
        # Scores must have one entry per output channel of the target
        assert scores.shape == (16,)
        # And must be non-negative (L2 norms)
        assert (scores >= 0).all()
