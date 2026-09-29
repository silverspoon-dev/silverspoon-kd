"""1x1-convolution projector wrapper with input- and output-mode hooks."""

from typing import Any

import torch
from torch import nn

from ._mode_mixin import _ProjectorModeMixin


class GenericConv2dProjector(_ProjectorModeMixin, nn.Conv2d):
    """
    A 1x1 convolution projector for CNNs that supports both input and output modes.

    Projects the channel dimension of 4D tensors using 1x1 convolution.
    Preserves spatial dimensions while mapping from student channels to teacher channels.

    Args:
        in_channels: Number of input channels (student)
        out_channels: Number of output channels (teacher)
        bias: Whether to include bias term (default: False, common for distillation)
        mode: 'input' or 'output' (default: 'output')
        apply_to_arg: For input mode, which positional arg to project (default: 0)
        apply_to_kwarg: For input mode, which keyword arg to project (default: None)

    Shape:
        - Input mode: Takes *args, **kwargs, returns (args, kwargs)
        - Output mode: Takes tensor, returns projected tensor
        - Tensor shape: `(batch, in_channels, H, W)` -> `(batch, out_channels, H, W)`

    Example:
        ```python
        # Output mode (default)
        projector = GenericConv2dProjector(128, 512)
        features = torch.randn(8, 128, 14, 14)
        projected = projector(features)
        projected.shape  # torch.Size([8, 512, 14, 14])

        # Input mode
        projector = GenericConv2dProjector(128, 512, mode='input')
        args, kwargs = projector(torch.randn(8, 128, 14, 14))
        args[0].shape  # torch.Size([8, 512, 14, 14])
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        mode: str = "output",
        apply_to_arg: int | None = None,
        apply_to_kwarg: str | None = None,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self._init_projector_mode(mode, apply_to_arg, apply_to_kwarg)

    def forward(
        self,
        input: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Project the input tensor, or the targeted arg/kwarg in input mode.

        Output mode returns a :class:`Tensor`; input mode returns the
        full ``(args, kwargs)`` tuple for a ``forward_pre_hook``.
        """
        all_args: tuple = args if input is None else (input, *args)
        return self._dispatch_projection(
            lambda tensor: nn.Conv2d.forward(self, tensor), all_args, kwargs
        )
