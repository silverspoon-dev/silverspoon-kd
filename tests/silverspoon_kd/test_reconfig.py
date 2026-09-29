"""
Tests for reconfig_model and freeze_copied_weights functions.
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


class TestReconfigModel:
    """Test suite for reconfig_model function."""

    @pytest.fixture
    def base_model(self):
        """Create a base model for testing."""
        config = SimpleConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256,
        )
        model = SimpleModel(config)
        # Initialize with specific values so we can verify weight copying
        with torch.no_grad():
            for _name, param in model.named_parameters():
                param.fill_(0.5)  # Fill with 0.5 for easy verification
        return model

    @pytest.fixture
    def base_model_random(self):
        """Create a base model with random weights for testing."""
        config = SimpleConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256,
        )
        return SimpleModel(config)

    def test_basic_reconfig_no_diff(self, base_model):
        """Test reconfig with empty diff creates model with same config."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test_model",
            diff={},
        )

        assert new_model.config.vocab_size == base_model.config.vocab_size
        assert new_model.config.hidden_size == base_model.config.hidden_size
        assert new_model.config.num_hidden_layers == base_model.config.num_hidden_layers
        assert new_model.config.num_attention_heads == base_model.config.num_attention_heads
        assert new_model.config.intermediate_size == base_model.config.intermediate_size
        assert new_model.config.name_or_path == "test_model"

    def test_reconfig_changes_config(self, base_model):
        """Test that diff changes are applied to config."""
        new_model = reconfig_model(
            base_model,
            name_or_path="smaller_model",
            diff={
                "num_hidden_layers": 2,
                "intermediate_size": 128,
            },
        )

        assert new_model.config.num_hidden_layers == 2
        assert new_model.config.intermediate_size == 128
        # Unchanged values
        assert new_model.config.hidden_size == base_model.config.hidden_size
        assert new_model.config.vocab_size == base_model.config.vocab_size

    def test_reconfig_fewer_layers(self, base_model):
        """Test reducing number of layers."""
        original_layers = base_model.config.num_hidden_layers
        new_model = reconfig_model(
            base_model,
            name_or_path="fewer_layers",
            diff={"num_hidden_layers": 2},
        )

        assert len(new_model.layers) == 2
        assert original_layers == 4  # Sanity check

    def test_reconfig_preserves_model_class(self, base_model):
        """Test that reconfig uses the same model class."""
        new_model = reconfig_model(
            base_model,
            name_or_path="same_class",
            diff={},
        )

        assert type(new_model) is type(base_model)
        assert isinstance(new_model, SimpleModel)

    def test_reconfig_unknown_config_key_warning(self, base_model, caplog):
        """Test that unknown config keys produce a warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.utils"):
            reconfig_model(
                base_model,
                name_or_path="test",
                diff={"unknown_key": 42},
            )

        assert "Config has no attribute 'unknown_key'" in caplog.text

    def test_copy_matching_weights_false_by_default(self, base_model):
        """Test that weights are NOT copied by default (random initialization)."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
            copy_matching_weights=False,
        )

        # Weights should be different (randomly initialized)
        base_embed = base_model.embed_tokens.weight.data
        new_embed = new_model.embed_tokens.weight.data
        assert not torch.allclose(base_embed, new_embed)

    def test_copy_matching_weights_same_architecture(self, base_model):
        """Test that all weights are copied when architecture is identical."""
        new_model = reconfig_model(
            base_model,
            name_or_path="copied_model",
            diff={},
            copy_matching_weights=True,
        )

        # All weights should match since architecture is the same
        for (name1, param1), (name2, param2) in zip(
            base_model.named_parameters(), new_model.named_parameters(), strict=False
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2), f"Weights differ for {name1}"

    def test_copy_matching_weights_fewer_layers(self, base_model):
        """Test weight copying when reducing number of layers."""
        new_model = reconfig_model(
            base_model,
            name_or_path="fewer_layers",
            diff={"num_hidden_layers": 2},
            copy_matching_weights=True,
        )

        # Embedding should be copied (same shape)
        assert torch.allclose(
            base_model.embed_tokens.weight.data,
            new_model.embed_tokens.weight.data,
        )

        # lm_head should be copied (same shape)
        assert torch.allclose(
            base_model.lm_head.weight.data,
            new_model.lm_head.weight.data,
        )

        # Final layer norm should be copied
        assert torch.allclose(
            base_model.norm.weight.data,
            new_model.norm.weight.data,
        )

        # First 2 layers should have weights copied
        for i in range(2):
            base_layer = base_model.layers[i]
            new_layer = new_model.layers[i]
            for (name1, param1), (_name2, param2) in zip(
                base_layer.named_parameters(), new_layer.named_parameters(), strict=False
            ):
                assert torch.allclose(param1, param2), f"Layer {i} {name1} differs"

    def test_copy_matching_weights_different_hidden_size(self, base_model):
        """Test that weights with shape mismatch are NOT copied (random init)."""
        new_model = reconfig_model(
            base_model,
            name_or_path="different_hidden",
            diff={"hidden_size": 128},  # Change from 64 to 128
            copy_matching_weights=True,
        )

        # Hidden size changed, so most weights should NOT match
        # (they have different shapes and can't be copied)
        base_embed = base_model.embed_tokens.weight.data  # shape: (1000, 64)
        new_embed = new_model.embed_tokens.weight.data  # shape: (1000, 128)

        assert base_embed.shape != new_embed.shape
        assert base_embed.shape[1] == 64
        assert new_embed.shape[1] == 128

    def test_copy_matching_weights_different_intermediate_size(self, base_model):
        """Test partial weight copying when only intermediate_size changes."""
        new_model = reconfig_model(
            base_model,
            name_or_path="different_intermediate",
            diff={"intermediate_size": 128},  # Change from 256 to 128
            copy_matching_weights=True,
        )

        # Embedding should be copied (hidden_size unchanged)
        assert torch.allclose(
            base_model.embed_tokens.weight.data,
            new_model.embed_tokens.weight.data,
        )

        # lm_head should be copied (hidden_size unchanged)
        assert torch.allclose(
            base_model.lm_head.weight.data,
            new_model.lm_head.weight.data,
        )

        # Layer norms should be copied (hidden_size unchanged)
        assert torch.allclose(
            base_model.norm.weight.data,
            new_model.norm.weight.data,
        )

        # Attention projections should be copied (hidden_size unchanged)
        for i in range(len(new_model.layers)):
            base_layer = base_model.layers[i]
            new_layer = new_model.layers[i]

            # Q, K, V, O projections depend on hidden_size only
            assert torch.allclose(base_layer.q_proj.weight.data, new_layer.q_proj.weight.data)
            assert torch.allclose(base_layer.k_proj.weight.data, new_layer.k_proj.weight.data)
            assert torch.allclose(base_layer.v_proj.weight.data, new_layer.v_proj.weight.data)
            assert torch.allclose(base_layer.o_proj.weight.data, new_layer.o_proj.weight.data)

            # up_proj and down_proj have different shapes, should NOT match
            assert base_layer.up_proj.weight.shape != new_layer.up_proj.weight.shape
            assert base_layer.down_proj.weight.shape != new_layer.down_proj.weight.shape

    def test_copy_matching_weights_verbose_output(self, base_model, caplog):
        """Test that verbose mode outputs weight transfer information."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="silverspoon_kd.utils"):
            reconfig_model(
                base_model,
                name_or_path="test",
                diff={"intermediate_size": 128},
                copy_matching_weights=True,
            )

        assert "Weight transfer:" in caplog.text
        assert "copied" in caplog.text
        assert "shape mismatches" in caplog.text

    def test_copy_matching_weights_verbose_shows_copied_params(self, base_model, caplog):
        """Test that verbose mode shows which parameters were copied."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="silverspoon_kd.utils"):
            reconfig_model(
                base_model,
                name_or_path="test",
                diff={},  # Same architecture, all weights copied
                copy_matching_weights=True,
            )

        assert "Copied parameters:" in caplog.text

    def test_copy_matching_weights_verbose_shows_mismatches(self, base_model, caplog):
        """Test that verbose mode shows shape mismatches."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="silverspoon_kd.utils"):
            reconfig_model(
                base_model,
                name_or_path="test",
                diff={"hidden_size": 128},  # Different hidden size
                copy_matching_weights=True,
            )

        assert "Shape mismatches" in caplog.text

    def test_copy_matching_weights_does_not_modify_source(self, base_model_random):
        """Test that source model is not modified during weight copying."""
        # Store original weights
        original_weights = {
            name: param.clone() for name, param in base_model_random.named_parameters()
        }

        reconfig_model(
            base_model_random,
            name_or_path="test",
            diff={},
            copy_matching_weights=True,
        )

        # Verify source model weights unchanged
        for name, param in base_model_random.named_parameters():
            assert torch.allclose(param, original_weights[name]), (
                f"Source weight {name} was modified"
            )

    def test_copy_matching_weights_new_model_independent(self, base_model):
        """Test that new model weights are independent (not sharing memory)."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
            copy_matching_weights=True,
        )

        # Modify new model weights
        with torch.no_grad():
            new_model.embed_tokens.weight.fill_(999.0)

        # Original should be unchanged
        assert not torch.allclose(
            base_model.embed_tokens.weight.data,
            new_model.embed_tokens.weight.data,
        )
        assert torch.allclose(
            base_model.embed_tokens.weight.data,
            torch.full_like(base_model.embed_tokens.weight.data, 0.5),
        )

    def test_reconfig_name_or_path_set(self, base_model):
        """Test that name_or_path is correctly set in config."""
        new_model = reconfig_model(
            base_model,
            name_or_path="my/custom/path",
            diff={},
        )

        assert new_model.config.name_or_path == "my/custom/path"

    def test_reconfig_original_config_unchanged(self, base_model):
        """Test that original model's config is not modified."""
        original_name = base_model.config.name_or_path
        original_layers = base_model.config.num_hidden_layers

        reconfig_model(
            base_model,
            name_or_path="new_name",
            diff={"num_hidden_layers": 1},
        )

        # Original config should be unchanged
        assert base_model.config.name_or_path == original_name
        assert base_model.config.num_hidden_layers == original_layers

    def test_copy_matching_weights_with_vocab_size_change(self, base_model):
        """Test weight copying when vocab_size changes (embedding/lm_head affected)."""
        new_model = reconfig_model(
            base_model,
            name_or_path="different_vocab",
            diff={"vocab_size": 500},  # Reduce from 1000 to 500
            copy_matching_weights=True,
        )

        # Embedding has different shape, should not be copied
        assert base_model.embed_tokens.weight.shape != new_model.embed_tokens.weight.shape

        # lm_head has different shape, should not be copied
        assert base_model.lm_head.weight.shape != new_model.lm_head.weight.shape

        # But layer weights should be copied (hidden_size unchanged)
        for i in range(len(new_model.layers)):
            assert torch.allclose(
                base_model.layers[i].q_proj.weight.data,
                new_model.layers[i].q_proj.weight.data,
            )

    def test_copy_matching_weights_num_attention_heads_change(self, base_model):
        """Test weight copying when num_attention_heads changes.

        Note: In this simple model, attention projections don't depend on num_heads
        (they just use hidden_size). In real transformers, the shapes might differ
        if head_dim changes.
        """
        new_model = reconfig_model(
            base_model,
            name_or_path="different_heads",
            diff={"num_attention_heads": 2},  # Change from 4 to 2
            copy_matching_weights=True,
        )

        # In our simple model, hidden_size is unchanged, so projections have same shape
        # and should be copied
        for i in range(len(new_model.layers)):
            assert torch.allclose(
                base_model.layers[i].q_proj.weight.data,
                new_model.layers[i].q_proj.weight.data,
            )

    def test_reconfig_creates_new_instance(self, base_model):
        """Test that reconfig always creates a new model instance."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
        )

        assert new_model is not base_model
        assert new_model.layers is not base_model.layers

    def test_copy_matching_weights_handles_biases(self, base_model):
        """Test that biases are also copied when shapes match."""
        # Add biases to the model for testing
        with torch.no_grad():
            base_model.embed_tokens.weight.fill_(0.5)
            for layer in base_model.layers:
                if layer.q_proj.bias is not None:
                    layer.q_proj.bias.fill_(0.1)

        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
            copy_matching_weights=True,
        )

        # Check that bias is copied if it exists
        for i in range(len(new_model.layers)):
            if base_model.layers[i].q_proj.bias is not None:
                assert torch.allclose(
                    base_model.layers[i].q_proj.bias.data,
                    new_model.layers[i].q_proj.bias.data,
                )


class TestReconfigModelEdgeCases:
    """Test edge cases for reconfig_model."""

    @pytest.fixture
    def base_model(self):
        """Create a base model for testing."""
        config = SimpleConfig()
        return SimpleModel(config)

    def test_empty_diff_dict(self, base_model):
        """Test with explicitly empty diff dict."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
        )
        assert new_model is not None
        assert new_model.config.hidden_size == base_model.config.hidden_size

    def test_no_print_output(self, base_model, capsys):
        """Test that reconfig_model produces no print output (uses logging instead)."""
        reconfig_model(
            base_model,
            name_or_path="test",
            diff={"num_hidden_layers": 2},
            copy_matching_weights=True,
        )

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_multiple_config_changes(self, base_model):
        """Test applying multiple config changes at once."""
        new_model = reconfig_model(
            base_model,
            name_or_path="multiple_changes",
            diff={
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 2,
            },
        )

        assert new_model.config.num_hidden_layers == 2
        assert new_model.config.intermediate_size == 128
        assert new_model.config.num_attention_heads == 2
        # Unchanged
        assert new_model.config.hidden_size == base_model.config.hidden_size
        assert new_model.config.vocab_size == base_model.config.vocab_size

    def test_copy_weights_with_multiple_config_changes(self, base_model):
        """Test weight copying with multiple config changes."""
        with torch.no_grad():
            base_model.embed_tokens.weight.fill_(0.5)
            base_model.norm.weight.fill_(0.5)

        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={
                "num_hidden_layers": 2,
                "intermediate_size": 128,
            },
            copy_matching_weights=True,
        )

        # Embedding should be copied (hidden_size, vocab_size unchanged)
        assert torch.allclose(
            base_model.embed_tokens.weight.data,
            new_model.embed_tokens.weight.data,
        )

        # Final norm should be copied (hidden_size unchanged)
        assert torch.allclose(
            base_model.norm.weight.data,
            new_model.norm.weight.data,
        )


class TestFreezeCopiedWeights:
    """Test suite for freeze_copied_weights parameter in reconfig_model."""

    @pytest.fixture
    def base_model(self):
        """Create a base model for testing."""
        config = SimpleConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256,
        )
        model = SimpleModel(config)
        with torch.no_grad():
            for _name, param in model.named_parameters():
                param.fill_(0.5)
        return model

    def test_freeze_copied_weights_freezes_all_when_same_architecture(self, base_model):
        """Test that all copied weights are frozen when architecture is identical."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
            copy_matching_weights=True,
            freeze_copied_weights=True,
        )

        # All parameters should be frozen since all were copied
        for name, param in new_model.named_parameters():
            assert not param.requires_grad, f"Parameter {name} should be frozen"

    def test_freeze_copied_weights_false_leaves_trainable(self, base_model):
        """Test that freeze_copied_weights=False leaves parameters trainable."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
            copy_matching_weights=True,
            freeze_copied_weights=False,
        )

        # All parameters should be trainable
        for name, param in new_model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be trainable"

    def test_freeze_copied_weights_partial_freeze(self, base_model):
        """Test that only copied weights are frozen when shapes differ."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={"intermediate_size": 128},  # Changes up_proj and down_proj shapes
            copy_matching_weights=True,
            freeze_copied_weights=True,
        )

        # Embedding, lm_head, layer norms, and attention projections should be frozen
        # (they have matching shapes and were copied)
        assert not new_model.embed_tokens.weight.requires_grad
        assert not new_model.lm_head.weight.requires_grad
        assert not new_model.norm.weight.requires_grad

        for layer in new_model.layers:
            # Attention projections should be frozen (same shape)
            assert not layer.q_proj.weight.requires_grad
            assert not layer.k_proj.weight.requires_grad
            assert not layer.v_proj.weight.requires_grad
            assert not layer.o_proj.weight.requires_grad
            assert not layer.input_layernorm.weight.requires_grad
            assert not layer.post_attention_layernorm.weight.requires_grad

            # FFN projections should be trainable (different shape, not copied)
            assert layer.up_proj.weight.requires_grad
            assert layer.down_proj.weight.requires_grad

    def test_freeze_copied_weights_fewer_layers(self, base_model):
        """Test freeze behavior when reducing number of layers."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={"num_hidden_layers": 2},
            copy_matching_weights=True,
            freeze_copied_weights=True,
        )

        # All parameters should be frozen (all shapes match for existing layers)
        for name, param in new_model.named_parameters():
            assert not param.requires_grad, f"Parameter {name} should be frozen"

    def test_freeze_copied_weights_ignored_when_copy_false(self, base_model):
        """Test that freeze_copied_weights is ignored when copy_matching_weights=False."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
            copy_matching_weights=False,
            freeze_copied_weights=True,  # Should be ignored
        )

        # All parameters should be trainable (copy_matching_weights=False)
        for name, param in new_model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be trainable"

    def test_freeze_copied_weights_verbose_output(self, base_model, caplog):
        """Test that logging outputs freeze information."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="silverspoon_kd.utils"):
            reconfig_model(
                base_model,
                name_or_path="test",
                diff={},
                copy_matching_weights=True,
                freeze_copied_weights=True,
            )

        assert "Froze" in caplog.text
        assert "copied parameters" in caplog.text

    def test_freeze_copied_weights_vocab_size_change(self, base_model):
        """Test freeze behavior when vocab_size changes (embedding/lm_head not copied)."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={"vocab_size": 500},
            copy_matching_weights=True,
            freeze_copied_weights=True,
        )

        # Embedding and lm_head should be trainable (different shapes, not copied)
        assert new_model.embed_tokens.weight.requires_grad
        assert new_model.lm_head.weight.requires_grad

        # Layer weights should be frozen (shapes match)
        for layer in new_model.layers:
            assert not layer.q_proj.weight.requires_grad
            assert not layer.input_layernorm.weight.requires_grad

    def test_freeze_copied_weights_hidden_size_change(self, base_model):
        """Test that most weights are trainable when hidden_size changes."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={"hidden_size": 128},
            copy_matching_weights=True,
            freeze_copied_weights=True,
        )

        # Most weights should be trainable since hidden_size affects most layers
        # Only weights that don't depend on hidden_size would be frozen
        trainable_count = sum(1 for p in new_model.parameters() if p.requires_grad)
        total_count = sum(1 for p in new_model.parameters())

        # Most parameters should be trainable due to shape mismatch
        assert trainable_count > total_count * 0.5

    def test_freeze_copied_weights_does_not_affect_source(self, base_model):
        """Test that freezing doesn't affect the source model."""
        # Ensure source model params are trainable
        for param in base_model.parameters():
            param.requires_grad = True

        reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
            copy_matching_weights=True,
            freeze_copied_weights=True,
        )

        # Source model parameters should still be trainable
        for name, param in base_model.named_parameters():
            assert param.requires_grad, f"Source parameter {name} should still be trainable"

    def test_freeze_copied_weights_with_biases(self, base_model):
        """Test that biases are also frozen when copied."""
        new_model = reconfig_model(
            base_model,
            name_or_path="test",
            diff={},
            copy_matching_weights=True,
            freeze_copied_weights=True,
        )

        # Check biases are frozen if they exist
        for layer in new_model.layers:
            if layer.q_proj.bias is not None:
                assert not layer.q_proj.bias.requires_grad
