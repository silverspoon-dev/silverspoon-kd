"""
Unit tests for projector fusion utilities.
"""

import re

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.alignments.projectors import (
    GenericConv2dProjector,
    GenericLinearProjector,
    fuse_projectors_into_module,
)


class TestFuseLinearProjectors:
    """Tests for fusing linear projectors into linear modules."""

    def test_no_projectors_returns_original(self):
        """Test that no projectors returns original module."""
        module = nn.Linear(64, 128)
        fused = fuse_projectors_into_module(module, None, None)
        assert fused is module

    def test_fuse_input_projector_only(self):
        """Test fusing only input projector."""
        module = nn.Linear(64, 128)
        input_proj = GenericLinearProjector(32, 64, mode="input")
        fused = fuse_projectors_into_module(module, input_proj, None)
        assert fused.in_features == 32
        assert fused.out_features == 128

    def test_fuse_output_projector_only(self):
        """Test fusing only output projector."""
        module = nn.Linear(64, 128)
        output_proj = GenericLinearProjector(32, 64, mode="output")
        fused = fuse_projectors_into_module(module, None, output_proj)
        assert fused.in_features == 32
        assert fused.out_features == 128

    def test_fuse_both_projectors(self):
        """Test fusing both input and output projectors."""
        module = nn.Linear(64, 128)
        input_proj = GenericLinearProjector(32, 64, mode="input")
        output_proj = GenericLinearProjector(16, 32, mode="output")
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        assert fused.in_features == 16
        assert fused.out_features == 128

    def test_fused_module_produces_same_output(self):
        """Test that fused module produces same output as chained modules."""
        torch.manual_seed(42)
        module = nn.Linear(64, 128)
        input_proj = GenericLinearProjector(32, 64, mode="input", apply_to_arg=0)
        output_proj = GenericLinearProjector(16, 32, mode="output")

        x = torch.randn(2, 16)
        # output_proj in "output" mode takes a tensor and returns a tensor
        x_projected_out = output_proj(x)
        # input_proj in "input" mode takes args/kwargs and returns (args, kwargs)
        projected_args, _ = input_proj(x_projected_out)
        original_output = module(projected_args[0])

        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        fused_output = fused(x)
        assert torch.allclose(original_output, fused_output, rtol=1e-4, atol=1e-4)

    def test_inplace_fusion(self):
        """Test in-place fusion modifies original module."""
        module = nn.Linear(64, 128)
        original_id = id(module)
        input_proj = GenericLinearProjector(32, 64, mode="input")
        fused = fuse_projectors_into_module(module, input_proj, None, inplace=True)
        assert id(fused) == original_id
        assert fused.in_features == 32

    def test_raises_for_wrong_projector_type(self):
        """Test that wrong projector type raises error."""
        module = nn.Linear(64, 128)
        conv_proj = GenericConv2dProjector(32, 64, mode="input")
        with pytest.raises(ValueError, match="GenericLinearProjector"):
            fuse_projectors_into_module(module, conv_proj, None)


class TestFuseConv2dProjectors:
    """Tests for fusing Conv2D projectors into Conv2D modules."""

    def test_fuse_input_projector_only(self):
        """Test fusing only input projector."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        input_proj = GenericConv2dProjector(32, 64, mode="input")
        fused = fuse_projectors_into_module(module, input_proj, None)
        assert fused.in_channels == 32
        assert fused.out_channels == 128

    def test_fuse_output_projector_only(self):
        """Test fusing only output projector."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        output_proj = GenericConv2dProjector(32, 64, mode="output")
        fused = fuse_projectors_into_module(module, None, output_proj)
        assert fused.in_channels == 32
        assert fused.out_channels == 128

    def test_preserves_kernel_size(self):
        """Test that kernel size is preserved."""
        module = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        input_proj = GenericConv2dProjector(32, 64, mode="input")
        fused = fuse_projectors_into_module(module, input_proj, None)
        assert fused.kernel_size == (5, 5)

    def test_preserves_stride_and_padding(self):
        """Test that stride and padding are preserved."""
        module = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        input_proj = GenericConv2dProjector(32, 64, mode="input")
        fused = fuse_projectors_into_module(module, input_proj, None)
        assert fused.stride == (2, 2)
        assert fused.padding == (1, 1)

    def test_raises_for_wrong_projector_type(self):
        """Test that wrong projector type raises error."""
        module = nn.Conv2d(64, 128, kernel_size=3)
        linear_proj = GenericLinearProjector(32, 64, mode="input")
        with pytest.raises(ValueError, match="GenericConv2dProjector"):
            fuse_projectors_into_module(module, linear_proj, None)


class TestFusionUnsupportedModules:
    """Tests for unsupported module types."""

    def test_raises_for_unsupported_module_type(self):
        """Test that unsupported module types raise error."""
        module = nn.ReLU()
        with pytest.raises(ValueError, match=re.escape("Only nn.Linear and nn.Conv2d")):
            fuse_projectors_into_module(module, None, None)

    def test_raises_for_embedding(self):
        """Test that Embedding raises error."""
        module = nn.Embedding(128, 256)
        with pytest.raises(ValueError, match=re.escape("Only nn.Linear and nn.Conv2d")):
            fuse_projectors_into_module(module, None, None)


class TestFusionEdgeCases:
    """Edge case tests for fusion."""

    def test_large_dimension_mismatch(self):
        """Test fusion with large dimension mismatch."""
        module = nn.Linear(1024, 2048)
        input_proj = GenericLinearProjector(64, 1024, mode="input")
        output_proj = GenericLinearProjector(32, 64, mode="output")
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        assert fused.in_features == 32
        assert fused.out_features == 2048

    def test_fused_has_fewer_parameters(self):
        """Test that fused module has fewer parameters than separate modules."""
        module = nn.Linear(64, 128)
        input_proj = GenericLinearProjector(32, 64, mode="input")
        output_proj = GenericLinearProjector(16, 32, mode="output")

        original_params = (
            sum(p.numel() for p in module.parameters())
            + sum(p.numel() for p in input_proj.parameters())
            + sum(p.numel() for p in output_proj.parameters())
        )

        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        fused_params = sum(p.numel() for p in fused.parameters())
        assert fused_params < original_params

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_module_cuda(self):
        """Test that fused module works on CUDA."""
        module = nn.Linear(64, 128).cuda()
        input_proj = GenericLinearProjector(32, 64, mode="input").cuda()
        fused = fuse_projectors_into_module(module, input_proj, None)
        assert next(fused.parameters()).device.type == "cuda"

        x = torch.randn(2, 32).cuda()
        output = fused(x)
        assert output.device.type == "cuda"


class TestFuseLinearProjectorsBiasEdgeCases:
    """Tests for bias edge cases in linear projector fusion."""

    def test_input_proj_bias_when_module_has_no_bias(self):
        """Test fusion when input projector has bias but module doesn't."""
        module = nn.Linear(64, 128, bias=False)
        input_proj = GenericLinearProjector(32, 64, mode="input", bias=True)
        fused = fuse_projectors_into_module(module, input_proj, None)
        assert fused.bias is not None
        assert fused.in_features == 32

    def test_output_proj_bias_when_module_has_no_bias(self):
        """Test fusion when output projector has bias but module and input proj don't."""
        module = nn.Linear(64, 128, bias=False)
        output_proj = GenericLinearProjector(32, 64, mode="output", bias=True)
        fused = fuse_projectors_into_module(module, None, output_proj)
        assert fused.bias is not None
        assert fused.in_features == 32

    def test_both_proj_bias_when_module_has_no_bias(self):
        """Test fusion when both projectors have bias but module doesn't."""
        module = nn.Linear(64, 128, bias=False)
        input_proj = GenericLinearProjector(32, 64, mode="input", bias=True)
        output_proj = GenericLinearProjector(16, 32, mode="output", bias=True)
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        assert fused.bias is not None
        assert fused.in_features == 16

    def test_wrong_output_projector_type_for_linear(self):
        """Test that wrong output projector type raises error for linear module."""
        module = nn.Linear(64, 128)
        conv_proj = GenericConv2dProjector(32, 64, mode="output")
        with pytest.raises(ValueError, match="GenericLinearProjector"):
            fuse_projectors_into_module(module, None, conv_proj)

    def test_inplace_without_bias(self):
        """Test inplace fusion when result has no bias."""
        module = nn.Linear(64, 128, bias=False)
        input_proj = GenericLinearProjector(32, 64, mode="input", bias=False)
        fused = fuse_projectors_into_module(module, input_proj, None, inplace=True)
        assert fused is module
        assert fused.bias is None
        assert fused.in_features == 32


class TestFuseConv2dProjectorsBiasEdgeCases:
    """Tests for bias edge cases in Conv2d projector fusion."""

    def test_input_proj_bias_when_module_has_no_bias(self):
        """Test fusion when conv input projector has bias but module doesn't."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        input_proj = GenericConv2dProjector(32, 64, mode="input", bias=True)
        fused = fuse_projectors_into_module(module, input_proj, None)
        assert fused.bias is not None
        assert fused.in_channels == 32

    def test_output_proj_bias_when_module_has_no_bias(self):
        """Test fusion when conv output projector has bias but module doesn't."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        output_proj = GenericConv2dProjector(32, 64, mode="output", bias=True)
        fused = fuse_projectors_into_module(module, None, output_proj)
        assert fused.bias is not None
        assert fused.in_channels == 32

    def test_both_proj_bias_when_module_has_no_bias(self):
        """Test fusion when both conv projectors have bias but module doesn't."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        input_proj = GenericConv2dProjector(32, 64, mode="input", bias=True)
        output_proj = GenericConv2dProjector(16, 32, mode="output", bias=True)
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        assert fused.bias is not None
        assert fused.in_channels == 16

    def test_wrong_output_projector_type_for_conv2d(self):
        """Test that wrong output projector type raises error for conv2d module."""
        module = nn.Conv2d(64, 128, kernel_size=3)
        linear_proj = GenericLinearProjector(32, 64, mode="output")
        with pytest.raises(ValueError, match="GenericConv2dProjector"):
            fuse_projectors_into_module(module, None, linear_proj)

    def test_non_1x1_input_projector_raises(self):
        """Test that non-1x1 input projector raises RuntimeError."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Create a 3x3 conv projector by manually changing kernel_size
        input_proj = GenericConv2dProjector(32, 64, mode="input")
        # Override to make it non-1x1
        input_proj.kernel_size = (3, 3)
        with pytest.raises(RuntimeError, match="1x1 convolution"):
            fuse_projectors_into_module(module, input_proj, None)

    def test_non_1x1_output_projector_raises(self):
        """Test that non-1x1 output projector raises RuntimeError."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        output_proj = GenericConv2dProjector(32, 64, mode="output")
        output_proj.kernel_size = (3, 3)
        with pytest.raises(RuntimeError, match="1x1 convolution"):
            fuse_projectors_into_module(module, None, output_proj)

    def test_inplace_conv2d_fusion(self):
        """Test inplace fusion for Conv2d."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        original_id = id(module)
        input_proj = GenericConv2dProjector(32, 64, mode="input")
        fused = fuse_projectors_into_module(module, input_proj, None, inplace=True)
        assert id(fused) == original_id
        assert fused.in_channels == 32

    def test_inplace_conv2d_without_bias(self):
        """Test inplace conv2d fusion when result has no bias."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        input_proj = GenericConv2dProjector(32, 64, mode="input", bias=False)
        fused = fuse_projectors_into_module(module, input_proj, None, inplace=True)
        assert fused is module
        assert fused.bias is None
        assert fused.in_channels == 32

    def test_fuse_both_conv2d_projectors(self):
        """Test fusing both input and output Conv2d projectors."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        input_proj = GenericConv2dProjector(32, 64, mode="input")
        output_proj = GenericConv2dProjector(16, 32, mode="output")
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        assert fused.in_channels == 16
        assert fused.out_channels == 128

    def test_fused_conv2d_produces_same_output_no_bias(self):
        """Test that fused Conv2d (no bias) produces same output as chained modules."""
        torch.manual_seed(42)
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        input_proj = GenericConv2dProjector(32, 64, mode="input", apply_to_arg=0, bias=False)
        output_proj = GenericConv2dProjector(16, 32, mode="output", bias=False)

        x = torch.randn(2, 16, 8, 8)
        x_projected_out = output_proj(x)
        projected_args, _ = input_proj(x_projected_out)
        original_output = module(projected_args[0])

        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        fused_output = fused(x)
        assert torch.allclose(original_output, fused_output, rtol=1e-5, atol=1e-5)

    def test_fused_conv2d_produces_same_output_1x1_with_bias(self):
        """Test that fused 1x1 Conv2d with bias produces same output as chained modules."""
        torch.manual_seed(42)
        module = nn.Conv2d(64, 128, kernel_size=1, bias=True)
        input_proj = GenericConv2dProjector(32, 64, mode="input", apply_to_arg=0, bias=True)
        output_proj = GenericConv2dProjector(16, 32, mode="output", bias=True)

        x = torch.randn(2, 16, 8, 8)
        x_projected_out = output_proj(x)
        projected_args, _ = input_proj(x_projected_out)
        original_output = module(projected_args[0])

        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        fused_output = fused(x)
        assert torch.allclose(original_output, fused_output, rtol=1e-5, atol=1e-5)

    @staticmethod
    def _chain(module, input_proj, output_proj, x):
        """Reference output of x -> output_proj -> input_proj -> module."""
        if output_proj is not None:
            x = output_proj(x)
        if input_proj is not None:
            (x,), _ = input_proj(x)
        return module(x)

    @pytest.mark.parametrize("which", ["input", "output", "both"])
    def test_fused_3x3_with_biases_matches_on_interior(self, which):
        """Bias fusion through a spatial kernel is exact away from the zero-padded border."""
        torch.manual_seed(0)
        module = nn.Conv2d(6, 5, kernel_size=3, padding=1, bias=True)
        input_proj = (
            GenericConv2dProjector(4, 6, mode="input", apply_to_arg=0, bias=True)
            if which in ("input", "both")
            else None
        )
        output_proj = (
            GenericConv2dProjector(3, 4, mode="output", bias=True) if which == "both" else None
        )
        if which == "output":
            output_proj = GenericConv2dProjector(3, 6, mode="output", bias=True)
        for proj in (input_proj, output_proj):
            if proj is not None:
                with torch.no_grad():
                    proj.weight.normal_()
                    proj.bias.normal_()

        x = torch.randn(2, 4 if which == "input" else 3, 8, 8)
        expected = self._chain(module, input_proj, output_proj, x)
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        actual = fused(x)

        interior = (slice(None), slice(None), slice(1, -1), slice(1, -1))
        assert torch.allclose(actual[interior], expected[interior], rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
    def test_fused_3x3_with_biases_matches_everywhere_without_zero_padding(self, padding_mode):
        """With padding=0 (or a non-zero padding mode) the fusion is exact on every pixel."""
        torch.manual_seed(0)
        padding = 0 if padding_mode == "zeros" else 1
        module = nn.Conv2d(
            6, 5, kernel_size=3, padding=padding, padding_mode=padding_mode, bias=True
        )
        input_proj = GenericConv2dProjector(4, 6, mode="input", apply_to_arg=0, bias=True)
        output_proj = GenericConv2dProjector(3, 4, mode="output", bias=True)
        for proj in (input_proj, output_proj):
            with torch.no_grad():
                proj.weight.normal_()
                proj.bias.normal_()

        x = torch.randn(2, 3, 8, 8)
        expected = self._chain(module, input_proj, output_proj, x)
        actual = fuse_projectors_into_module(module, input_proj, output_proj)(x)
        assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_fused_conv2d_input_only_produces_same_output(self):
        """Test that fused Conv2d with input projector only produces same output."""
        torch.manual_seed(42)
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        input_proj = GenericConv2dProjector(32, 64, mode="input", apply_to_arg=0, bias=False)

        x = torch.randn(2, 32, 8, 8)
        projected_args, _ = input_proj(x)
        original_output = module(projected_args[0])

        fused = fuse_projectors_into_module(module, input_proj, None)
        fused_output = fused(x)
        assert torch.allclose(original_output, fused_output, rtol=1e-5, atol=1e-5)

    def test_input_proj_bias_when_module_has_bias(self):
        """Test conv2d fusion when both module and input projector have bias."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        input_proj = GenericConv2dProjector(32, 64, mode="input", bias=True)
        fused = fuse_projectors_into_module(module, input_proj, None)
        assert fused.bias is not None
        assert fused.in_channels == 32

    def test_output_proj_bias_when_module_has_bias(self):
        """Test conv2d fusion when both module and output projector have bias."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        output_proj = GenericConv2dProjector(32, 64, mode="output", bias=True)
        fused = fuse_projectors_into_module(module, None, output_proj)
        assert fused.bias is not None
        assert fused.in_channels == 32

    def test_both_proj_bias_when_module_has_bias(self):
        """Test conv2d fusion when module and both projectors have bias."""
        module = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        input_proj = GenericConv2dProjector(32, 64, mode="input", bias=True)
        output_proj = GenericConv2dProjector(16, 32, mode="output", bias=True)
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        assert fused.bias is not None
        assert fused.in_channels == 16
