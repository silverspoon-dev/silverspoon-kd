"""Linear projector wrapper that supports input- and output-mode hooks."""

from typing import Any

import torch
from torch import nn

from ._mode_mixin import _ProjectorModeMixin


class GenericLinearProjector(_ProjectorModeMixin, nn.Linear):
    """
    A linear projector for transformers and dense layers that supports both input and output modes.

    Projects the last dimension of tensors (e.g., hidden states in transformers).
    Supports both input projection (modifying args/kwargs) and output
    projection (direct tensor projection).

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias term (default: True)
        mode: 'input' or 'output' (default: 'output')
        apply_to_arg: For input mode, which positional arg to project (default: 0)
        apply_to_kwarg: For input mode, which keyword arg to project (default: None)

    Shape:
        - Input mode: Takes *args, **kwargs, returns (args, kwargs)
        - Output mode: Takes tensor, returns projected tensor
        - Tensor shape: `(*, in_features)` -> `(*, out_features)` where `*` is any dimensions

    Example:
        ```python
        # Output mode (default)
        projector = GenericLinearProjector(768, 512)
        hidden_states = torch.randn(32, 128, 768)
        projected = projector(hidden_states)
        projected.shape  # torch.Size([32, 128, 512])

        # Input mode
        projector = GenericLinearProjector(
            768, 512, mode='input', apply_to_kwarg='hidden_states'
        )
        args, kwargs = projector(hidden_states=torch.randn(32, 128, 768))
        kwargs['hidden_states'].shape  # torch.Size([32, 128, 512])
        ```
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "output",
        apply_to_arg: int | None = None,
        apply_to_kwarg: str | None = None,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
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
            lambda tensor: nn.Linear.forward(self, tensor), all_args, kwargs
        )
