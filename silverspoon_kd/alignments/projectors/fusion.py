"""
Projector fusion utilities for combining projectors into module weights.

This module provides functionality to fuse adjacent projectors into a module's weights,
eliminating runtime overhead while maintaining the same mathematical transformation.
"""

import logging

import torch
from torch import nn

from .generic_conv2d_projector import GenericConv2dProjector
from .generic_linear_projector import GenericLinearProjector

logger = logging.getLogger(__name__)


def fuse_projectors_into_module(
    module: nn.Module,
    input_projector: nn.Module | None = None,
    output_projector: nn.Module | None = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Fuse input and output projectors into a module's weights to eliminate runtime overhead.

    The fusion follows the flow: prev_layer → output_projector → input_projector → [module]

    This function absorbs both projectors into the module weights, creating a fused module
    that maintains student dimensions while performing the same transformation.

    Mathematical formulation:
    - For Linear layers: W_fused = W_module @ W_input_proj @ W_output_proj
    - For Conv2d layers: Similar channel-wise fusion using einsum operations

    Args:
        module: The module to fuse projectors into (nn.Linear or nn.Conv2d)
        input_projector: Optional input projector to fuse (applied after output_projector)
        output_projector: Optional output projector from preceding layer (applied first)
        inplace: Whether to modify the module in-place (default: False)

    Returns:
        A new fused module with projectors absorbed into weights

    Raises:
        ValueError: If module or projector types are unsupported
        RuntimeError: If projector configurations are incompatible with fusion

    Supported combinations:
        - nn.Linear with GenericLinearProjector(s)
        - nn.Conv2d with GenericConv2dProjector(s) (1x1 conv only)

    Limitations:
        - Projectors must be GenericLinearProjector or GenericConv2dProjector
        - Conv2d projectors must be 1x1 convolutions with stride=1, padding=0
        - No non-linear activations between projectors
        - For a Conv2d module with zero padding, projector biases are fused
          exactly on interior pixels only: at the border, kernel taps that
          read padded zeros carried no projector bias before fusion but do
          after.  Fusion is exact everywhere for ``padding=0`` and for the
          non-zero padding modes (``replicate``, ``reflect``, ``circular``).

    Example:
        ```python
        # Fuse linear projectors
        module = nn.Linear(256, 512)
        input_proj = GenericLinearProjector(128, 256, mode='input')
        output_proj = GenericLinearProjector(64, 128, mode='output')
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        # fused is now a Linear(64, 512) layer equivalent to output_proj -> input_proj -> module

        # Fuse CNN projectors
        module = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        input_proj = GenericConv2dProjector(16, 32, mode='input')
        output_proj = GenericConv2dProjector(8, 16, mode='output')
        fused = fuse_projectors_into_module(module, input_proj, output_proj)
        # fused is Conv2d(8, 64, kernel_size=3, padding=1) with absorbed projectors
        ```
    """
    # Validate module type
    if not isinstance(module, (nn.Linear, nn.Conv2d)):
        raise ValueError(
            f"Only nn.Linear and nn.Conv2d modules are supported for fusion, "
            f"got {type(module).__name__}"
        )

    # Handle case with no projectors
    if input_projector is None and output_projector is None:
        logger.info("No projectors to fuse, returning original module")
        return module

    # Route to appropriate fusion function
    if isinstance(module, nn.Linear):
        return _fuse_linear_projectors(module, input_projector, output_projector, inplace=inplace)
    # nn.Conv2d
    return _fuse_conv2d_projectors(module, input_projector, output_projector, inplace=inplace)


def _fuse_linear_projectors(
    module: nn.Linear,
    input_projector: nn.Module | None,
    output_projector: nn.Module | None,
    inplace: bool = False,
) -> nn.Linear:
    """
    Fuse linear projectors into a linear module.

    Flow: prev_layer → output_projector → input_projector → module

    Matrix multiplication order:
        y = module(input_proj(output_proj(x)))
        y = W_mod @ (W_in @ (W_out @ x + b_out) + b_in) + b_mod

    For weight fusion:
        W_fused = W_mod @ W_in @ W_out

    For bias fusion (if all biases present):
        b_fused = W_mod @ W_in @ b_out + W_mod @ b_in + b_mod
    """
    # Validate projector types
    if input_projector is not None and not isinstance(input_projector, GenericLinearProjector):
        raise ValueError(
            f"input_projector must be GenericLinearProjector for Linear module fusion, "
            f"got {type(input_projector).__name__}"
        )

    if output_projector is not None and not isinstance(output_projector, GenericLinearProjector):
        raise ValueError(
            f"output_projector must be GenericLinearProjector for Linear module fusion, "
            f"got {type(output_projector).__name__}"
        )

    # Get module weights and bias
    W_mod = module.weight.data  # [out_features, in_features]
    b_mod = module.bias.data if module.bias is not None else None

    # Start with module weights
    W_fused = W_mod
    b_fused = b_mod

    # Apply input projector fusion
    if input_projector is not None:
        W_in = input_projector.weight.data  # [module_in, input_proj_in]
        b_in = input_projector.bias.data if input_projector.bias is not None else None

        # Fuse weights: W_fused = W_mod @ W_in
        W_fused = W_fused @ W_in

        # Fuse bias: b_fused = W_mod @ b_in + b_fused
        if b_in is not None:
            b_fused = W_mod @ b_in if b_fused is None else W_mod @ b_in + b_fused

    # Apply output projector fusion
    if output_projector is not None:
        W_out = output_projector.weight.data  # [input_proj_in, output_proj_in]
        b_out = output_projector.bias.data if output_projector.bias is not None else None

        # Fuse weights: W_fused = W_fused @ W_out
        W_fused = W_fused @ W_out

        # Fuse bias: b_fused = W_fused_partial @ b_out + b_fused
        # Note: W_fused_partial is W_mod @ W_in (before applying W_out)
        if b_out is not None:
            W_fused_partial = (
                W_mod @ input_projector.weight.data if input_projector is not None else W_mod
            )
            if b_fused is None:
                b_fused = W_fused_partial @ b_out
            else:
                b_fused = W_fused_partial @ b_out + b_fused

    # Determine output and input dimensions
    out_features = W_fused.shape[0]  # module's output dim
    in_features = W_fused.shape[
        1
    ]  # output_projector's input dim (or input_projector's if no output_proj)

    has_bias = b_fused is not None

    if inplace:
        module.in_features = in_features
        module.out_features = out_features
        module.weight = nn.Parameter(W_fused)
        if has_bias:
            module.bias = nn.Parameter(b_fused)
        else:
            module.register_parameter("bias", None)
        fused_module = module
    else:
        # Create fused module
        fused_module = nn.Linear(in_features, out_features, bias=has_bias)

        # Copy fused weights
        fused_module.weight.data = W_fused
        if has_bias:
            fused_module.bias.data = b_fused

    # Log fusion details
    projector_info = []
    if output_projector is not None:
        projector_info.append(
            f"output_proj({output_projector.in_features}→{output_projector.out_features})"
        )
    if input_projector is not None:
        projector_info.append(
            f"input_proj({input_projector.in_features}→{input_projector.out_features})"
        )

    logger.info(
        "Fused Linear projectors: %s → module(%s→%s) = fused(%s→%s)",
        " → ".join(projector_info),
        module.in_features,
        module.out_features,
        in_features,
        out_features,
    )

    return fused_module


def _fuse_conv2d_projectors(
    module: nn.Conv2d,
    input_projector: nn.Module | None,
    output_projector: nn.Module | None,
    inplace: bool = False,
) -> nn.Conv2d:
    """
    Fuse Conv2d projectors (1x1 convolutions) into a Conv2d module.

    Flow: prev_layer → output_projector → input_projector → module

    For 1x1 convolutions, fusion is similar to linear layers but operates on channels:
        W_conv shape: [out_channels, in_channels, kernel_h, kernel_w]
        W_proj shape: [out_channels, in_channels, 1, 1]

    Fusion for 1x1 projectors:
        W_fused = einsum('oihw,ji,kj->okhw', W_mod, W_in, W_out)

    Simplified for channel dimension:
        W_fused[o,i,:,:] = sum_j sum_k W_mod[o,j,:,:] * W_in[j,k] * W_out[k,i]
    """
    # Validate projector types and configurations
    if input_projector is not None:
        if not isinstance(input_projector, GenericConv2dProjector):
            raise ValueError(
                f"input_projector must be GenericConv2dProjector for Conv2d module fusion, "
                f"got {type(input_projector).__name__}"
            )
        # Verify it's a 1x1 convolution
        if (
            input_projector.kernel_size != (1, 1)
            or input_projector.stride != (1, 1)
            or input_projector.padding != (0, 0)
        ):
            raise RuntimeError(
                f"input_projector must be 1x1 convolution with stride=1, padding=0 for fusion. "
                f"Got kernel_size={input_projector.kernel_size}, stride={input_projector.stride}, "
                f"padding={input_projector.padding}"
            )

    if output_projector is not None:
        if not isinstance(output_projector, GenericConv2dProjector):
            raise ValueError(
                f"output_projector must be GenericConv2dProjector for Conv2d module fusion, "
                f"got {type(output_projector).__name__}"
            )
        # Verify it's a 1x1 convolution
        if (
            output_projector.kernel_size != (1, 1)
            or output_projector.stride != (1, 1)
            or output_projector.padding != (0, 0)
        ):
            raise RuntimeError(
                f"output_projector must be 1x1 convolution with stride=1, padding=0 for fusion. "
                f"Got kernel_size={output_projector.kernel_size}, "
                f"stride={output_projector.stride}, "
                f"padding={output_projector.padding}"
            )

    # Get module weights and bias
    W_mod = module.weight.data  # [out_channels, in_channels, kernel_h, kernel_w]
    b_mod = module.bias.data if module.bias is not None else None

    # Start with module weights
    W_fused = W_mod
    b_fused = b_mod

    # Apply input projector fusion
    if input_projector is not None:
        W_in = input_projector.weight.data.squeeze(-1).squeeze(
            -1
        )  # [module_in, input_proj_in] from [module_in, input_proj_in, 1, 1]
        b_in = input_projector.bias.data if input_projector.bias is not None else None

        # Fuse weights: W_fused[o,i,h,w] = sum_j W_mod[o,j,h,w] * W_in[j,i]
        # This is equivalent to: W_fused = einsum('ojhw,ji->oihw', W_mod, W_in)
        W_fused = torch.einsum("ojhw,ji->oihw", W_fused, W_in)

        # Fuse bias.  The 1x1 projector adds b_in[j] to every pixel of input
        # channel j, so each kernel tap contributes W_mod[o,j,h,w] * b_in[j]
        # to output channel o:
        #     b_fused[o] = sum_{j,h,w} W_mod[o,j,h,w] * b_in[j] + b_fused[o]
        # Exact wherever every tap reads a real pixel (see the module
        # docstring for the zero-padding border caveat).
        if b_in is not None:
            W_mod_channel = W_mod.sum(dim=(-2, -1))  # [out_channels, in_channels]
            b_in_contribution = W_mod_channel @ b_in  # [out_channels]

            b_fused = b_in_contribution if b_fused is None else b_in_contribution + b_fused

    # Apply output projector fusion
    if output_projector is not None:
        W_out = output_projector.weight.data.squeeze(-1).squeeze(
            -1
        )  # [input_proj_in, output_proj_in]
        b_out = output_projector.bias.data if output_projector.bias is not None else None

        # Fuse weights: W_fused[o,i,h,w] = sum_j W_fused[o,j,h,w] * W_out[j,i]
        W_fused = torch.einsum("ojhw,ji->oihw", W_fused, W_out)

        # Fuse bias: b_out passes through the input projector's weights (if
        # any) and then through the kernel sum, exactly as b_in above.
        if b_out is not None:
            if input_projector is not None:
                W_partial = torch.einsum(
                    "ojhw,ji->oihw",
                    W_mod,
                    input_projector.weight.data.squeeze(-1).squeeze(-1),
                )
            else:
                W_partial = W_mod
            W_partial_channel = W_partial.sum(dim=(-2, -1))
            b_out_contribution = W_partial_channel @ b_out

            b_fused = b_out_contribution if b_fused is None else b_out_contribution + b_fused

    # Determine output and input channels
    out_channels = W_fused.shape[0]
    in_channels = W_fused.shape[1]
    has_bias = b_fused is not None

    if inplace:
        module.in_channels = in_channels
        module.out_channels = out_channels
        module.groups = 1
        module.weight = nn.Parameter(W_fused)
        if has_bias:
            module.bias = nn.Parameter(b_fused)
        else:
            module.register_parameter("bias", None)
        fused_module = module
    else:
        # Create fused module with same configuration as original
        fused_module = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=module.kernel_size,  # type: ignore[arg-type]
            stride=module.stride,  # type: ignore[arg-type]
            padding=module.padding,  # type: ignore[arg-type]
            dilation=module.dilation,  # type: ignore[arg-type]
            groups=module.groups,
            bias=has_bias,
            padding_mode=module.padding_mode,
        )

        # Copy fused weights
        fused_module.weight.data = W_fused
        if has_bias and fused_module.bias is not None:
            fused_module.bias.data = b_fused

    # Log fusion details
    projector_info = []
    if output_projector is not None:
        projector_info.append(
            f"output_proj({output_projector.in_channels}→{output_projector.out_channels})"
        )
    if input_projector is not None:
        projector_info.append(
            f"input_proj({input_projector.in_channels}→{input_projector.out_channels})"
        )

    logger.info(
        "Fused Conv2d projectors: %s → module(%s→%s) = fused(%s→%s)",
        " → ".join(projector_info),
        module.in_channels,
        module.out_channels,
        in_channels,
        out_channels,
    )

    return fused_module
