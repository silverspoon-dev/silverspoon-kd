"""
Unit tests for projector classes: GenericLinearProjector, GenericConv2dProjector.
"""

import pytest
import torch

from silverspoon_kd.alignments.projectors import (
    GenericConv2dProjector,
    GenericLinearProjector,
)

# ============================================================================
# GenericLinearProjector Tests
# ============================================================================


class TestGenericLinearProjectorInitialization:
    """Tests for GenericLinearProjector initialization."""

    def test_basic_initialization_output_mode(self):
        """Test basic initialization in output mode."""
        projector = GenericLinearProjector(in_features=768, out_features=512)

        assert projector.in_features == 768
        assert projector.out_features == 512
        assert projector.mode == "output"
        assert projector.apply_to_arg == 0
        assert projector.apply_to_kwarg is None

    def test_initialization_input_mode(self):
        """Test initialization in input mode."""
        projector = GenericLinearProjector(in_features=768, out_features=512, mode="input")

        assert projector.mode == "input"

    def test_initialization_with_apply_to_arg(self):
        """Test initialization with apply_to_arg."""
        projector = GenericLinearProjector(
            in_features=768, out_features=512, mode="input", apply_to_arg=1
        )

        assert projector.apply_to_arg == 1

    def test_initialization_with_apply_to_kwarg(self):
        """Test initialization with apply_to_kwarg."""
        projector = GenericLinearProjector(
            in_features=768,
            out_features=512,
            mode="input",
            apply_to_kwarg="hidden_states",
        )

        assert projector.apply_to_kwarg == "hidden_states"

    def test_initialization_with_bias(self):
        """Test initialization with bias."""
        projector = GenericLinearProjector(in_features=768, out_features=512, bias=True)

        assert projector.bias is not None

    def test_initialization_without_bias(self):
        """Test initialization without bias."""
        projector = GenericLinearProjector(in_features=768, out_features=512, bias=False)

        assert projector.bias is None

    def test_raises_when_both_apply_to_specified(self):
        """Test that specifying both apply_to_arg and apply_to_kwarg raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            GenericLinearProjector(
                in_features=768,
                out_features=512,
                mode="input",
                apply_to_arg=0,
                apply_to_kwarg="hidden_states",
            )

    def test_raises_for_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be"):
            GenericLinearProjector(in_features=768, out_features=512, mode="invalid")


class TestGenericLinearProjectorOutputMode:
    """Tests for GenericLinearProjector in output mode."""

    def test_projects_tensor_correctly(self):
        """Test that output mode projects tensor correctly."""
        projector = GenericLinearProjector(in_features=768, out_features=512)

        x = torch.randn(32, 128, 768)
        output = projector(x)

        assert output.shape == (32, 128, 512)

    def test_preserves_batch_dimensions(self):
        """Test that batch dimensions are preserved."""
        projector = GenericLinearProjector(in_features=768, out_features=512)

        # Test various batch shapes
        shapes = [(2, 768), (4, 10, 768), (2, 4, 8, 768)]
        for shape in shapes:
            x = torch.randn(*shape)
            output = projector(x)
            expected_shape = (*shape[:-1], 512)
            assert output.shape == expected_shape

    def test_raises_without_positional_args(self):
        """Test that output mode raises error without positional args."""
        projector = GenericLinearProjector(in_features=768, out_features=512)

        with pytest.raises(ValueError, match="requires at least one positional argument"):
            projector()

    def test_gradient_flow(self):
        """Test that gradients flow through projector."""
        projector = GenericLinearProjector(in_features=768, out_features=512)

        x = torch.randn(2, 768, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert projector.weight.grad is not None


class TestGenericLinearProjectorInputMode:
    """Tests for GenericLinearProjector in input mode."""

    def test_projects_positional_arg(self):
        """Test that input mode projects the specified positional arg."""
        projector = GenericLinearProjector(
            in_features=768, out_features=512, mode="input", apply_to_arg=0
        )

        x = torch.randn(32, 128, 768)
        args, kwargs = projector(x)

        assert args[0].shape == (32, 128, 512)
        assert kwargs == {}

    def test_projects_specific_positional_arg(self):
        """Test projecting a specific positional arg index."""
        projector = GenericLinearProjector(
            in_features=768, out_features=512, mode="input", apply_to_arg=1
        )

        x1 = torch.randn(32, 128, 768)
        x2 = torch.randn(32, 128, 768)
        args, _kwargs = projector(x1, x2)

        assert args[0].shape == (32, 128, 768)  # Unchanged
        assert args[1].shape == (32, 128, 512)  # Projected

    def test_projects_keyword_arg(self):
        """Test that input mode projects the specified keyword arg."""
        projector = GenericLinearProjector(
            in_features=768,
            out_features=512,
            mode="input",
            apply_to_kwarg="hidden_states",
        )

        hidden_states = torch.randn(32, 128, 768)
        args, kwargs = projector(hidden_states=hidden_states)

        assert args == ()
        assert kwargs["hidden_states"].shape == (32, 128, 512)

    def test_returns_tuple(self):
        """Test that input mode returns (args, kwargs) tuple."""
        projector = GenericLinearProjector(in_features=768, out_features=512, mode="input")

        x = torch.randn(32, 128, 768)
        result = projector(x)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_other_args_unchanged(self):
        """Test that other args are unchanged."""
        projector = GenericLinearProjector(
            in_features=768, out_features=512, mode="input", apply_to_arg=0
        )

        x = torch.randn(2, 768)
        other = torch.randn(2, 128)
        args, kwargs = projector(x, other, some_kwarg="value")

        assert args[1] is other
        assert kwargs["some_kwarg"] == "value"


# ============================================================================
# GenericConv2dProjector Tests
# ============================================================================


class TestGenericConv2dProjectorInitialization:
    """Tests for GenericConv2dProjector initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=512)

        assert projector.in_channels == 128
        assert projector.out_channels == 512
        assert projector.kernel_size == (1, 1)
        assert projector.stride == (1, 1)
        assert projector.padding == (0, 0)
        assert projector.mode == "output"

    def test_initialization_without_bias(self):
        """Test that bias defaults to False."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=512)

        assert projector.bias is None

    def test_initialization_with_bias(self):
        """Test initialization with bias."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=512, bias=True)

        assert projector.bias is not None

    def test_initialization_input_mode(self):
        """Test initialization in input mode."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=512, mode="input")

        assert projector.mode == "input"

    def test_raises_when_both_apply_to_specified(self):
        """Test that specifying both apply_to_arg and apply_to_kwarg raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            GenericConv2dProjector(
                in_channels=128,
                out_channels=512,
                mode="input",
                apply_to_arg=0,
                apply_to_kwarg="features",
            )

    def test_raises_for_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be"):
            GenericConv2dProjector(in_channels=128, out_channels=512, mode="invalid")


class TestGenericConv2dProjectorOutputMode:
    """Tests for GenericConv2dProjector in output mode."""

    def test_projects_tensor_correctly(self):
        """Test that output mode projects tensor correctly."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=512)

        x = torch.randn(8, 128, 14, 14)
        output = projector(x)

        assert output.shape == (8, 512, 14, 14)

    def test_preserves_spatial_dimensions(self):
        """Test that spatial dimensions are preserved."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=256)

        # Test various spatial sizes
        spatial_sizes = [(7, 7), (14, 14), (28, 28), (56, 56)]
        for h, w in spatial_sizes:
            x = torch.randn(2, 128, h, w)
            output = projector(x)
            assert output.shape == (2, 256, h, w)

    def test_raises_without_positional_args(self):
        """Test that output mode raises error without positional args."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=512)

        with pytest.raises(ValueError, match="requires at least one positional argument"):
            projector()

    def test_gradient_flow(self):
        """Test that gradients flow through projector."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=256)

        x = torch.randn(2, 128, 14, 14, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert projector.weight.grad is not None


class TestGenericConv2dProjectorInputMode:
    """Tests for GenericConv2dProjector in input mode."""

    def test_projects_positional_arg(self):
        """Test that input mode projects the specified positional arg."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=512, mode="input")

        x = torch.randn(8, 128, 14, 14)
        args, _kwargs = projector(x)

        assert args[0].shape == (8, 512, 14, 14)

    def test_projects_keyword_arg(self):
        """Test that input mode projects the specified keyword arg."""
        projector = GenericConv2dProjector(
            in_channels=128, out_channels=512, mode="input", apply_to_kwarg="features"
        )

        features = torch.randn(8, 128, 14, 14)
        _args, kwargs = projector(features=features)

        assert kwargs["features"].shape == (8, 512, 14, 14)

    def test_returns_tuple(self):
        """Test that input mode returns (args, kwargs) tuple."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=512, mode="input")

        x = torch.randn(8, 128, 14, 14)
        result = projector(x)

        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================================
# Dimension Mismatch Tests
# ============================================================================


class TestProjectorDimensionMismatch:
    """Tests for handling dimension mismatches."""

    def test_linear_projector_wrong_input_dim(self):
        """Test linear projector with wrong input dimension."""
        projector = GenericLinearProjector(in_features=768, out_features=512)

        # Wrong input dimension should raise error
        x = torch.randn(2, 256)  # Expected 768
        with pytest.raises(RuntimeError):
            projector(x)

    def test_conv2d_projector_wrong_input_channels(self):
        """Test conv2d projector with wrong input channels."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=256)

        # Wrong input channels should raise error
        x = torch.randn(2, 64, 14, 14)  # Expected 128 channels
        with pytest.raises(RuntimeError):
            projector(x)


# ============================================================================
# Device Placement Tests
# ============================================================================


class TestProjectorDevicePlacement:
    """Tests for device placement."""

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_linear_projector_cuda(self):
        """Test linear projector on CUDA."""
        projector = GenericLinearProjector(in_features=768, out_features=512).cuda()

        x = torch.randn(2, 768).cuda()
        output = projector(x)

        assert output.device.type == "cuda"

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_conv2d_projector_cuda(self):
        """Test conv2d projector on CUDA."""
        projector = GenericConv2dProjector(in_channels=128, out_channels=256).cuda()

        x = torch.randn(2, 128, 14, 14).cuda()
        output = projector(x)

        assert output.device.type == "cuda"


# ============================================================================
# Memory Tests
# ============================================================================


class TestProjectorMemory:
    """Tests for memory handling."""

    def test_linear_projector_large_batch(self):
        """Test linear projector with large batch."""
        projector = GenericLinearProjector(in_features=768, out_features=512)

        # Large batch size
        x = torch.randn(128, 512, 768)
        output = projector(x)

        assert output.shape == (128, 512, 512)

    def test_conv2d_projector_large_spatial(self):
        """Test conv2d projector with large spatial dimensions."""
        projector = GenericConv2dProjector(in_channels=64, out_channels=128)

        # Large spatial dimensions
        x = torch.randn(4, 64, 224, 224)
        output = projector(x)

        assert output.shape == (4, 128, 224, 224)

    def test_projector_parameter_count(self):
        """Test that projector has expected parameter count."""
        # Linear: in_features * out_features + out_features (bias)
        linear_proj = GenericLinearProjector(in_features=768, out_features=512, bias=True)
        linear_params = sum(p.numel() for p in linear_proj.parameters())
        assert linear_params == 768 * 512 + 512

        # Conv2d: in_channels * out_channels * 1 * 1 (no bias by default)
        conv_proj = GenericConv2dProjector(in_channels=128, out_channels=256, bias=False)
        conv_params = sum(p.numel() for p in conv_proj.parameters())
        assert conv_params == 128 * 256
