"""
Module capture engine for capturing inputs and outputs of neural network modules.

This module provides the ModuleCaptureEngine class which manages the registration
of hooks and forward wrappers to capture inputs and outputs of specified modules.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from torch import nn
from transformers.trainer_pt_utils import nested_detach


def _maybe_to_local(tensor_or_nested):
    """Convert DTensors to local tensors, pass through everything else."""
    try:
        from torch.distributed._tensor import DTensor  # pyright: ignore[reportPrivateImportUsage]
    except ImportError:
        return tensor_or_nested

    if isinstance(tensor_or_nested, DTensor):
        return tensor_or_nested.to_local()
    if isinstance(tensor_or_nested, (tuple, list)):
        converted = [_maybe_to_local(item) for item in tensor_or_nested]
        return type(tensor_or_nested)(converted)
    if isinstance(tensor_or_nested, dict):
        return {k: _maybe_to_local(v) for k, v in tensor_or_nested.items()}
    return tensor_or_nested


class _TruncatedForwardException(Exception):
    """Signal to stop a truncated forward pass early.

    Raised by the capture hook after all needed modules have been captured
    (auto_truncate) to abort the forward pass and skip remaining layers.
    Caught by the distiller's try/except around the model forward call.
    """


class ModuleCaptureEngine:
    """
    Engine for capturing inputs and outputs of neural network modules during forward passes.

    This class provides a general-purpose mechanism to:
    1. Capture inputs to specified modules
    2. Capture outputs from specified modules
    3. Optionally truncate the forward pass after a terminal module

    The captured data can be used for various purposes such as knowledge distillation,
    analysis, or debugging.
    """

    def __init__(
        self,
        model: nn.Module,
        modules_to_capture: list[nn.Module],
        deepcopy_captured_args_and_kwargs: bool = False,
        output_callback: Callable[[int, Any, Any], None] | None = None,
        detach_outputs: bool = True,
        capture_inputs: bool = True,
        auto_truncate: bool = False,
    ):
        """
        Initialize the ModuleCaptureEngine.

        Args:
            model: The model containing the modules to capture
            modules_to_capture: List of modules whose inputs and outputs should be captured
            deepcopy_captured_args_and_kwargs: Whether to deepcopy
                captured args/kwargs (slower but safer)
            output_callback: Optional callback function called after each
                module's forward pass.
                Signature: callback(module_id, input, output) -> None
            detach_outputs: Whether to detach captured outputs from
                computation graph (default True). Set to False when
                outputs need to retain gradients for backpropagation.
            capture_inputs: Whether to capture module inputs (args/kwargs).
                Set to False when only outputs are needed (e.g. HKD) to
                save memory.
            auto_truncate: If True, automatically stop the forward pass
                as soon as all ``modules_to_capture`` have been captured.
                This is a performance optimization — layers after the last
                captured module are skipped entirely, saving both compute
                and memory. Defaults to False because exception-based
                truncation is incompatible with FSDP and torch.compile.
        """
        self.model = model
        self.modules_to_capture = modules_to_capture
        self.deepcopy_captured_args_and_kwargs = deepcopy_captured_args_and_kwargs
        self.output_callback = output_callback
        self.detach_outputs = detach_outputs
        self.capture_inputs = capture_inputs
        self.auto_truncate = auto_truncate
        self._num_modules_to_capture = len(modules_to_capture)

        # Storage for captured data
        self.captured_args: dict[int, Any] = {}
        self.captured_kwargs: dict[int, Any] = {}
        self.captured_outputs: dict[int, Any] = {}

        # Hook management
        self.hook_handles: dict[int, Any] = {}
        self.original_forwards: dict[int, Callable] = {}

        # Optional per-block CUDA events for pipelined overlap.
        # When set, each hook records an event after capture so that a
        # consumer on another stream can wait for per-block readiness.
        self.capture_events: dict[int, Any] | None = None

        # Track registration state
        self.is_registered = False

    def _generate_hook(self, module_id: int, module: nn.Module) -> Any:
        """
        Generate and register a forward hook for a module.

        Args:
            module_id: The module identifier
            module: The module to attach the hook to

        Returns:
            The hook handle
        """

        def hook(_model: nn.Module, _input: Any, output: Any) -> None:
            """Forward hook that captures output and optionally calls callback."""
            # Convert DTensors to local tensors (TP/FSDP may produce these)
            output = _maybe_to_local(output)

            # Store the output (optionally detached)
            if self.detach_outputs:
                self.captured_outputs[module_id] = nested_detach(output)
            else:
                self.captured_outputs[module_id] = output

            # Call the output callback if provided
            if self.output_callback is not None:
                self.output_callback(module_id, _input, output)

            # Record CUDA event for pipelined overlap (before terminal check
            # so the event is recorded even for the last captured block).
            # ``capture_events`` is populated by the distiller with one
            # ``torch.cuda.Event()`` per alignment index — so when it's set,
            # every value is a real Event and ``.get()`` returns either an
            # Event or ``None`` (for indices outside the registered range).
            events = self.capture_events
            event = events.get(module_id) if events is not None else None
            if event is not None:
                event.record()

            # Stop the forward pass early once all modules have been
            # captured — layers beyond the last alignment are skipped.
            if self.auto_truncate and len(self.captured_outputs) >= self._num_modules_to_capture:
                raise _TruncatedForwardException()

        return module.register_forward_hook(hook)

    def _create_forward_wrapper(self, module_id: int, original_forward: Callable) -> Callable:
        """
        Create a wrapper that captures inputs before calling original forward.

        Args:
            module_id: The module identifier
            original_forward: The original forward method

        Returns:
            The wrapped forward method
        """

        def forward_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Only once per step should args and kwargs be captured
            assert module_id not in self.captured_args, (
                f"Args already captured for module {module_id}"
            )
            assert module_id not in self.captured_kwargs, (
                f"Kwargs already captured for module {module_id}"
            )

            # Create a snapshot of args/kwargs before forward pass
            if self.deepcopy_captured_args_and_kwargs:
                self.captured_args[module_id] = deepcopy(args)
                self.captured_kwargs[module_id] = deepcopy(kwargs)
            else:
                self.captured_args[module_id] = nested_detach(args)
                self.captured_kwargs[module_id] = nested_detach(kwargs)

            return original_forward(*args, **kwargs)

        return forward_wrapper

    def register(self) -> None:
        """
        Register forward hooks and input capture wrappers for all modules.

        This sets up the infrastructure to capture module inputs/outputs during forward passes.
        """
        if self.is_registered:
            raise RuntimeError("ModuleCaptureEngine is already registered")

        # Register hooks for all modules.
        # If an exception occurs mid-loop, clean up any already-registered
        # hooks so we don't leave the model in a partially-hooked state.
        try:
            for module_id, module in enumerate(self.modules_to_capture):
                # Register forward hook (captures outputs)
                self.hook_handles[module_id] = self._generate_hook(module_id, module)

                # Wrap forward to capture inputs (skip when only outputs needed)
                if self.capture_inputs:
                    self.original_forwards[module_id] = module.forward
                    module.forward = self._create_forward_wrapper(
                        module_id, self.original_forwards[module_id]
                    )
        except Exception:
            self.is_registered = True  # allow deregister() to clean up
            self.deregister()
            raise

        self.is_registered = True

    def deregister(self) -> None:
        """Remove all forward hooks and restore original forward methods."""
        if not self.is_registered:
            return

        # Remove the hooks
        for _module_id, handle in self.hook_handles.items():
            handle.remove()
        self.hook_handles = {}

        # Restore the original forward methods (only if inputs were captured)
        for module_id in list(self.original_forwards):
            self.modules_to_capture[module_id].forward = self.original_forwards[module_id]
        self.original_forwards = {}

        self.is_registered = False

    def clear_captured(self) -> None:
        """Clear all captured data."""
        self.captured_args.clear()
        self.captured_kwargs.clear()
        self.captured_outputs.clear()

    def get_captured(self, module_id: int) -> tuple[Any, Any, Any]:
        """
        Get the captured data for a specific module.

        Args:
            module_id: The module identifier

        Returns:
            Tuple of (args, kwargs, output)

        Raises:
            KeyError: If no data was captured for the given module_id
        """
        return (
            self.captured_args[module_id],
            self.captured_kwargs[module_id],
            self.captured_outputs[module_id],
        )

    def pop_captured_inputs(self, module_id: int) -> tuple[Any, Any]:
        """
        Pop (retrieve and remove) the captured inputs for a specific module.

        Args:
            module_id: The module identifier

        Returns:
            Tuple of (args, kwargs)

        Raises:
            KeyError: If no inputs were captured for the given module_id
        """
        args = self.captured_args.pop(module_id)
        kwargs = self.captured_kwargs.pop(module_id)
        return (args, kwargs)

    def pop_captured_output(self, module_id: int) -> Any:
        """
        Pop (retrieve and remove) the captured output for a specific module.

        Args:
            module_id: The module identifier

        Returns:
            The captured output

        Raises:
            KeyError: If no output was captured for the given module_id
        """
        return self.captured_outputs.pop(module_id)

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deregister hooks."""
        self.deregister()
        # Suppress _TruncatedForwardException (auto-truncation)
        return exc_type is _TruncatedForwardException
