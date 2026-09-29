"""Shared input/output-mode plumbing for projector modules.

Both :class:`GenericLinearProjector` and :class:`GenericConv2dProjector`
support two modes:

* ``"output"`` — a plain projection of a single input tensor to a
  single output tensor (the conventional behaviour of ``nn.Linear`` /
  ``nn.Conv2d``).
* ``"input"`` — a projection that lives *inside* a ``forward_pre_hook``:
  it receives the positional/keyword arguments of a downstream module,
  projects one of them, and returns ``(args, kwargs)`` for the hook
  machinery to pass along.

The mode-handling logic is identical for both projector types; only
the atomic "project one tensor" step differs (``nn.Linear.forward``
vs. ``nn.Conv2d.forward``).  :class:`_ProjectorModeMixin` factors out
the shared logic so neither projector has to carry its own copy.
"""

from collections.abc import Callable
from typing import Any


class _ProjectorModeMixin:
    """Mix-in providing the mode / apply-to-arg dispatch for projectors.

    Concrete subclasses must:

    1. Call :meth:`_init_projector_mode` in their ``__init__`` with the
       user-supplied ``mode``, ``apply_to_arg`` and ``apply_to_kwarg``.
    2. In their ``forward`` implementation, call
       :meth:`_dispatch_projection` with a ``project_fn`` that performs
       the atomic single-tensor projection (typically a bound call to
       the underlying ``nn.Linear`` / ``nn.Conv2d`` forward).

    This keeps the mode/validation/dispatch logic in exactly one place.
    """

    # Populated by ``_init_projector_mode``; declared at class level so
    # static checkers don't flag attribute access in the mixin methods.
    mode: str
    apply_to_arg: int
    apply_to_kwarg: str | None

    def _init_projector_mode(
        self,
        mode: str,
        apply_to_arg: int | None,
        apply_to_kwarg: str | None,
    ) -> None:
        """Validate and store the mode configuration.

        Raises:
            ValueError: If ``mode`` is not ``"input"`` or ``"output"``,
                or if both ``apply_to_arg`` and ``apply_to_kwarg`` are
                provided.
        """
        if apply_to_arg is not None and apply_to_kwarg is not None:
            raise ValueError("Cannot specify both apply_to_arg and apply_to_kwarg")
        if mode not in ("input", "output"):
            raise ValueError(f"mode must be 'input' or 'output', got {mode}")
        self.mode = mode
        self.apply_to_arg = apply_to_arg if apply_to_arg is not None else 0
        self.apply_to_kwarg = apply_to_kwarg

    def _dispatch_projection(
        self,
        project_fn: Callable[[Any], Any],
        args: tuple[Any, ...],
        kwargs: dict,
    ) -> Any:
        """Apply ``project_fn`` according to the configured mode.

        * In ``"input"`` mode, projects the selected positional or
          keyword argument in place and returns ``(args, kwargs)`` so
          the forward-pre-hook can pass it along.
        * In ``"output"`` mode, projects ``args[0]`` directly and
          returns the resulting tensor.
        """
        if self.mode == "input":
            # Input mode: project within args/kwargs and return tuple
            if self.apply_to_kwarg is not None:
                kwargs[self.apply_to_kwarg] = project_fn(kwargs[self.apply_to_kwarg])
            else:
                arg_list = list(args)
                arg_list[self.apply_to_arg] = project_fn(arg_list[self.apply_to_arg])
                args = tuple(arg_list)
            return args, kwargs
        # Output mode: project single tensor directly
        if len(args) == 0:
            raise ValueError("Output mode requires at least one positional argument")
        return project_fn(args[0])
