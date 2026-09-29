"""Speed benchmarks for projectors and projector fusion.

Measures forward pass cost for linear/conv2d projectors and compares
fused vs unfused execution paths.
"""

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.alignments.projectors import (
    GenericConv2dProjector,
    GenericLinearProjector,
    fuse_projectors_into_module,
)

# ═══════════════════════════════════════════════════════════════════════
#  Linear Projector Benchmarks
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="projectors-linear")
class TestLinearProjector:
    @pytest.mark.parametrize(
        "dims",
        [
            pytest.param((64, 128), id="64to128"),
            pytest.param((128, 256), id="128to256"),
            pytest.param((256, 512), id="256to512"),
        ],
    )
    def test_output_mode(self, benchmark, device, dims):
        """Output projector: projects student output to teacher dimension."""
        in_dim, out_dim = dims
        projector = GenericLinearProjector(in_dim, out_dim, mode="output").to(device)
        x = torch.randn(4, 32, in_dim, device=device)
        benchmark(projector, x)

    @pytest.mark.parametrize(
        "dims",
        [
            pytest.param((128, 64), id="128to64"),
            pytest.param((256, 128), id="256to128"),
        ],
    )
    def test_input_mode(self, benchmark, device, dims):
        """Input projector: projects teacher input to student dimension."""
        in_dim, out_dim = dims
        projector = GenericLinearProjector(in_dim, out_dim, mode="input", apply_to_arg=0).to(device)
        x = torch.randn(4, 32, in_dim, device=device)

        def run():
            projector(x)

        benchmark(run)


# ═══════════════════════════════════════════════════════════════════════
#  Conv2d Projector Benchmarks
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="projectors-conv2d")
class TestConv2dProjector:
    @pytest.mark.parametrize(
        "channels",
        [
            pytest.param((32, 64), id="32to64"),
            pytest.param((64, 128), id="64to128"),
        ],
    )
    def test_output_mode(self, benchmark, device, channels):
        """Output projector for convolutional models (1x1 conv)."""
        in_ch, out_ch = channels
        projector = GenericConv2dProjector(in_ch, out_ch, mode="output").to(device)
        x = torch.randn(4, in_ch, 8, 8, device=device)
        benchmark(projector, x)


# ═══════════════════════════════════════════════════════════════════════
#  Projector Fusion Benchmarks
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="projectors-fusion")
class TestProjectorFusion:
    def test_fusion_operation(self, benchmark, device):
        """Measure the one-time cost of fusing projectors into a module."""
        # Chain: prev(16) → output_proj(16→32) → input_proj(32→64) → module(64→128)
        base_layer = nn.Linear(64, 128).to(device)
        input_proj = GenericLinearProjector(32, 64, mode="input", apply_to_arg=0).to(device)
        output_proj = GenericLinearProjector(16, 32, mode="output").to(device)

        benchmark(fuse_projectors_into_module, base_layer, input_proj, output_proj)

    def test_fused_forward(self, benchmark, device):
        """Forward pass through a fused layer (single matmul)."""
        base_layer = nn.Linear(64, 128).to(device)
        input_proj = GenericLinearProjector(32, 64, mode="input", apply_to_arg=0).to(device)
        output_proj = GenericLinearProjector(16, 32, mode="output").to(device)
        fused = fuse_projectors_into_module(base_layer, input_proj, output_proj)
        x = torch.randn(4, 32, 16, device=device)
        benchmark(fused, x)

    def test_unfused_forward(self, benchmark, device):
        """Forward pass through separate projectors + base layer (3 matmuls)."""
        # Chain: prev(16) → output_proj(16→32) → input_proj(32→64) → module(64→128)
        base_layer = nn.Linear(64, 128).to(device)
        input_proj = GenericLinearProjector(32, 64, mode="input", apply_to_arg=0).to(device)
        output_proj = GenericLinearProjector(16, 32, mode="output").to(device)
        x = torch.randn(4, 32, 16, device=device)

        def run():
            projected = output_proj(x)
            args, kwargs = input_proj(projected)
            return base_layer(*args, **kwargs)

        benchmark(run)
