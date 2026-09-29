"""
Unit tests for ModuleCaptureEngine.
"""

import contextlib
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.engines.module_capture_engine import (
    ModuleCaptureEngine,
    _maybe_to_local,
    _TruncatedForwardException,
)


class SimpleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class SimpleModel(nn.Module):
    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([SimpleBlock(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestModuleCaptureEngineBasic:
    """Tests for basic capture functionality."""

    def test_register_and_deregister(self):
        model = SimpleModel(64)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()
        assert engine.is_registered is True
        engine.deregister()
        assert engine.is_registered is False

    def test_register_twice_raises(self):
        model = SimpleModel(64)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()
        with pytest.raises(RuntimeError, match="already registered"):
            engine.register()
        engine.deregister()

    def test_deregister_when_not_registered_is_noop(self):
        model = SimpleModel(64)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.deregister()  # Should not raise

    def test_captures_outputs(self):
        model = SimpleModel(64, num_layers=2)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        assert 0 in engine.captured_outputs
        assert 1 in engine.captured_outputs
        engine.deregister()

    def test_captures_args_and_kwargs(self):
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        assert 0 in engine.captured_args
        assert 0 in engine.captured_kwargs
        engine.deregister()

    def test_get_captured(self):
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        args, _kwargs, output = engine.get_captured(0)
        assert args is not None
        assert output is not None
        engine.deregister()

    def test_pop_captured_inputs(self):
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        args, _kwargs = engine.pop_captured_inputs(0)
        assert args is not None
        assert 0 not in engine.captured_args
        assert 0 not in engine.captured_kwargs
        engine.deregister()

    def test_clear_captured(self):
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        engine.clear_captured()
        assert len(engine.captured_args) == 0
        assert len(engine.captured_kwargs) == 0
        assert len(engine.captured_outputs) == 0
        engine.deregister()


class TestModuleCaptureEngineCallbacks:
    """Tests for output callbacks."""

    def test_output_callback_called(self):
        model = SimpleModel(64, num_layers=2)
        callback_calls = []

        def callback(module_id, input, output):
            callback_calls.append(module_id)

        engine = ModuleCaptureEngine(
            model, list(model.layers), output_callback=callback, auto_truncate=False
        )
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        assert callback_calls == [0, 1]
        engine.deregister()


class TestModuleCaptureEngineDetach:
    """Tests for detach_outputs option."""

    def test_detach_outputs_true(self):
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(
            model, list(model.layers), detach_outputs=True, auto_truncate=False
        )
        engine.register()

        x = torch.randn(2, 64, requires_grad=True)
        model(x)

        output = engine.captured_outputs[0]
        assert not output.requires_grad
        engine.deregister()

    def test_detach_outputs_false(self):
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(
            model, list(model.layers), detach_outputs=False, auto_truncate=False
        )
        engine.register()

        x = torch.randn(2, 64, requires_grad=True)
        model(x)

        output = engine.captured_outputs[0]
        assert output.requires_grad
        engine.deregister()


class TestModuleCaptureEngineDeepCopy:
    """Tests for deepcopy_captured_args_and_kwargs option."""

    def test_deepcopy_captured(self):
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(
            model,
            list(model.layers),
            deepcopy_captured_args_and_kwargs=True,
            auto_truncate=False,
        )
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        assert 0 in engine.captured_args
        engine.deregister()


class TestModuleCaptureEngineContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_registers_and_deregisters(self):
        model = SimpleModel(64)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)

        with engine:
            assert engine.is_registered is True

        assert engine.is_registered is False

    def test_context_manager_with_auto_truncate(self):
        """Test context manager suppresses auto-truncation exception."""
        model = SimpleModel(64, num_layers=3)
        engine = ModuleCaptureEngine(model, [model.layers[0]], auto_truncate=True)

        # Should not raise — context manager suppresses the exception
        with engine:
            model(torch.randn(2, 64))

        assert not engine.is_registered

    def test_context_manager_does_not_suppress_other_exceptions(self):
        model = SimpleModel(64)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)

        with pytest.raises(ValueError), engine:
            raise ValueError("test error")

        assert engine.is_registered is False


class TestCaptureInputsFalse:
    """Regression tests for capture_inputs=False optimization."""

    def test_outputs_captured_without_input_capture(self):
        """capture_inputs=False must still capture outputs."""
        model = SimpleModel(64, num_layers=2)
        engine = ModuleCaptureEngine(
            model, list(model.layers), capture_inputs=False, auto_truncate=False
        )
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        assert 0 in engine.captured_outputs
        assert 1 in engine.captured_outputs
        engine.deregister()

    def test_inputs_not_captured_when_disabled(self):
        """capture_inputs=False must not store args/kwargs."""
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(
            model, list(model.layers), capture_inputs=False, auto_truncate=False
        )
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        assert len(engine.captured_args) == 0
        assert len(engine.captured_kwargs) == 0
        engine.deregister()

    def test_forward_not_wrapped_when_inputs_disabled(self):
        """capture_inputs=False must not store original forwards (no wrapping)."""
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(
            model, list(model.layers), capture_inputs=False, auto_truncate=False
        )
        engine.register()

        # No original forwards stored means forward was not wrapped
        assert len(engine.original_forwards) == 0
        engine.deregister()

    def test_deregister_safe_when_inputs_disabled(self):
        """Deregistering with capture_inputs=False must not error."""
        model = SimpleModel(64, num_layers=2)
        engine = ModuleCaptureEngine(
            model, list(model.layers), capture_inputs=False, auto_truncate=False
        )
        engine.register()
        engine.deregister()  # Must not raise
        assert engine.is_registered is False


class TestPopCapturedData:
    """Regression tests for pop_captured_inputs/pop_captured_output freeing memory."""

    def test_pop_output_removes_from_storage(self):
        """pop_captured_output must remove the entry from captured_outputs."""
        model = SimpleModel(64, num_layers=2)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        output_0 = engine.pop_captured_output(0)
        assert output_0 is not None
        assert 0 not in engine.captured_outputs
        # Module 1 should still be present
        assert 1 in engine.captured_outputs
        engine.deregister()

    def test_pop_inputs_removes_from_storage(self):
        """pop_captured_inputs must remove both args and kwargs."""
        model = SimpleModel(64, num_layers=2)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        args, _kwargs = engine.pop_captured_inputs(0)
        assert args is not None
        assert 0 not in engine.captured_args
        assert 0 not in engine.captured_kwargs
        # Module 1 should still be present
        assert 1 in engine.captured_args
        engine.deregister()

    def test_double_pop_raises(self):
        """Popping the same module twice must raise KeyError."""
        model = SimpleModel(64, num_layers=1)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        engine.pop_captured_output(0)
        with pytest.raises(KeyError):
            engine.pop_captured_output(0)
        engine.deregister()


class TestMaybeToLocal:
    """Tests for the _maybe_to_local helper."""

    def test_regular_tensor_passes_through(self):
        """Regular tensors should be returned as-is (identity)."""
        t = torch.randn(2, 4)
        result = _maybe_to_local(t)
        assert result is t

    def test_nested_tuple_preserves_type(self):
        """Tuple in, tuple out, elements unchanged."""
        a = torch.randn(2, 4)
        b = torch.randn(3, 5)
        result = _maybe_to_local((a, b))
        assert isinstance(result, tuple)
        assert result[0] is a
        assert result[1] is b

    def test_nested_list_preserves_type(self):
        """List in, list out, elements unchanged."""
        a = torch.randn(2, 4)
        b = torch.randn(3, 5)
        result = _maybe_to_local([a, b])
        assert isinstance(result, list)
        assert result[0] is a
        assert result[1] is b

    def test_nested_dict_recursed(self):
        """Dict values should be recursed."""
        a = torch.randn(2, 4)
        result = _maybe_to_local({"x": a, "y": 42})
        assert isinstance(result, dict)
        assert result["x"] is a
        assert result["y"] == 42

    def test_non_tensor_passes_through(self):
        """int, string, None pass through unchanged."""
        assert _maybe_to_local(42) == 42
        assert _maybe_to_local("hello") == "hello"
        assert _maybe_to_local(None) is None

    def test_dtensor_calls_to_local(self):
        """Mock DTensor class to verify to_local() is called."""
        local_tensor = torch.randn(2, 4)

        # Create a mock DTensor class
        mock_dtensor_cls = type("DTensor", (), {})
        mock_dtensor = mock_dtensor_cls()
        mock_dtensor.to_local = MagicMock(return_value=local_tensor)

        # ``_maybe_to_local`` imports DTensor at call time, so patching the
        # module in sys.modules for the duration of the call is enough.
        mock_module = MagicMock()
        mock_module.DTensor = mock_dtensor_cls
        with patch.dict(sys.modules, {"torch.distributed._tensor": mock_module}):
            result = _maybe_to_local(mock_dtensor)

        mock_dtensor.to_local.assert_called_once()
        assert result is local_tensor


class TestCaptureEvents:
    """Tests for capture_events CUDA event recording."""

    def test_events_recorded_on_forward(self):
        """Events should be recorded once per module during forward."""
        model = SimpleModel(64, num_layers=2)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.capture_events = {0: MagicMock(), 1: MagicMock()}
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        engine.capture_events[0].record.assert_called_once()
        engine.capture_events[1].record.assert_called_once()
        engine.deregister()

    def test_no_error_when_events_none(self):
        """Default None capture_events should not cause errors."""
        model = SimpleModel(64, num_layers=2)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        assert engine.capture_events is None
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)  # Should not raise

        assert 0 in engine.captured_outputs
        engine.deregister()

    def test_events_only_for_matching_ids(self):
        """Events should only be recorded for module IDs present in the dict."""
        model = SimpleModel(64, num_layers=3)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=False)
        engine.capture_events = {0: MagicMock(), 2: MagicMock()}
        engine.register()

        x = torch.randn(2, 64)
        with torch.no_grad():
            model(x)

        engine.capture_events[0].record.assert_called_once()
        engine.capture_events[2].record.assert_called_once()
        # Module 1 has no event entry — no error, no record call
        engine.deregister()


class TestAutoTruncate:
    """Tests for automatic forward truncation.

    These tests verify the *effect* of auto_truncate (only the expected
    modules are captured) rather than testing exception propagation, because
    PyTorch's ``Module._call_impl`` may swallow exceptions from hooks when
    dynamo state leaks between pytest-xdist workers.

    The context manager pattern (``with engine: model(x)``) mirrors how the
    distillers actually use auto_truncate in production.
    """

    def test_auto_truncate_stops_after_last_captured_module(self):
        """Only aligned modules should be captured, later layers skipped.

        Independently verifies truncation by attaching pre-hooks to layers
        after the last captured one and asserting they never fire — this
        is robust to the pytest-xdist exception-propagation issue because
        it observes the *effect* on later layers rather than the raised
        exception itself.
        """
        model = SimpleModel(64, num_layers=5)
        # Only capture layers 0 and 1 (out of 5)
        captured_modules = [model.layers[0], model.layers[1]]
        engine = ModuleCaptureEngine(model, captured_modules, auto_truncate=True)

        # Independently track whether layers 2-4 ever ran
        later_layers_called: list[int] = []
        watch_handles = [
            model.layers[i].register_forward_pre_hook(
                lambda _m, _a, idx=i: later_layers_called.append(idx)
            )
            for i in (2, 3, 4)
        ]

        try:
            with engine, contextlib.suppress(_TruncatedForwardException):
                model(torch.randn(2, 64))
        finally:
            for h in watch_handles:
                h.remove()

        # Layers 0 and 1 should be captured
        assert 0 in engine.captured_outputs
        assert 1 in engine.captured_outputs
        assert len(engine.captured_outputs) == 2
        # Layers 2-4 must never have been entered — proves the forward
        # pass was actually truncated, not just that hookless layers were
        # silently skipped from the captured-outputs dict.
        assert later_layers_called == []

    def test_auto_truncate_disabled_runs_full_forward(self):
        """With auto_truncate=False, forward should run to completion."""
        model = SimpleModel(64, num_layers=5)
        captured_modules = [model.layers[0], model.layers[1]]
        engine = ModuleCaptureEngine(model, captured_modules, auto_truncate=False)

        with engine:
            result = model(torch.randn(2, 64))

        # Layers 0 and 1 captured, forward ran to completion
        assert 0 in engine.captured_outputs
        assert 1 in engine.captured_outputs
        assert result is not None

    def test_auto_truncate_all_layers_captured(self):
        """When all layers are aligned, auto_truncate fires after the last one."""
        model = SimpleModel(64, num_layers=3)
        engine = ModuleCaptureEngine(model, list(model.layers), auto_truncate=True)

        with engine, contextlib.suppress(_TruncatedForwardException):
            model(torch.randn(2, 64))

        # All 3 modules captured
        assert len(engine.captured_outputs) == 3

    def test_auto_truncate_single_module(self):
        """Auto-truncation works with a single captured module."""
        model = SimpleModel(64, num_layers=5)
        engine = ModuleCaptureEngine(model, [model.layers[0]], auto_truncate=True)

        with engine, contextlib.suppress(_TruncatedForwardException):
            model(torch.randn(2, 64))

        assert 0 in engine.captured_outputs
        assert len(engine.captured_outputs) == 1

    def test_auto_truncate_default_is_false(self):
        """auto_truncate should default to False (FSDP/torch.compile safe)."""
        model = SimpleModel(64)
        engine = ModuleCaptureEngine(model, list(model.layers))
        assert engine.auto_truncate is False

    def test_auto_truncate_resets_between_steps(self):
        """Auto-truncation should work correctly across multiple forward passes."""
        model = SimpleModel(64, num_layers=3)
        engine = ModuleCaptureEngine(model, [model.layers[0]], auto_truncate=True)

        x = torch.randn(2, 64)
        # First forward
        with engine:
            with contextlib.suppress(_TruncatedForwardException):
                model(x)
            assert 0 in engine.captured_outputs
            assert len(engine.captured_outputs) == 1

            # Clear and run again
            engine.clear_captured()
            with contextlib.suppress(_TruncatedForwardException):
                model(x)
            assert 0 in engine.captured_outputs
            assert len(engine.captured_outputs) == 1

    def test_context_manager_suppresses_auto_truncate_exception(self):
        """Context manager should suppress the auto-truncation exception."""
        model = SimpleModel(64, num_layers=5)
        engine = ModuleCaptureEngine(model, [model.layers[0]], auto_truncate=True)

        with engine:
            model(torch.randn(2, 64))  # should not raise

        assert 0 in engine.captured_outputs
        assert not engine.is_registered


class TestPartialRegistrationCleanup:
    """Regression test: failed registration must not leave partial hooks."""

    def test_exception_during_register_cleans_up(self):
        """If _generate_hook fails mid-loop, already-registered hooks are removed."""
        model = SimpleModel(64, num_layers=3)
        modules = list(model.layers)
        engine = ModuleCaptureEngine(model, modules, auto_truncate=False)

        # Monkey-patch _generate_hook to fail on the second module
        original_generate = engine._generate_hook
        call_count = 0

        def failing_generate(module_id, module):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated hook failure")
            return original_generate(module_id, module)

        engine._generate_hook = failing_generate

        with pytest.raises(RuntimeError, match="simulated hook failure"):
            engine.register()

        # Engine must not be registered
        assert engine.is_registered is False
        # No leftover hooks
        assert len(engine.hook_handles) == 0
        # No leftover forward wrappers
        assert len(engine.original_forwards) == 0
