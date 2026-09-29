"""Speed benchmarks for ModuleCaptureEngine.

Measures hook registration/deregistration overhead and forward pass cost
with varying numbers of captured modules and capture modes.
"""

import contextlib

import pytest
import torch
from bench_utils import SimpleModel

from silverspoon_kd.engines.module_capture_engine import (
    ModuleCaptureEngine,
    _TruncatedForwardException,
)


def _make_batch(device, batch_size=4, seq_len=32):
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
    }


# ═══════════════════════════════════════════════════════════════════════
#  ModuleCaptureEngine Benchmarks
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="capture-engine")
class TestCaptureEngineRegistration:
    """Measure hook registration and deregistration overhead."""

    @pytest.mark.parametrize("num_layers", [2, 4, 6])
    def test_register_deregister_cycle(self, benchmark, device, num_layers):
        model = SimpleModel(hidden_dim=128, num_layers=num_layers).to(device)
        modules = list(model.layers)
        engine = ModuleCaptureEngine(model=model, modules_to_capture=modules)

        def cycle():
            engine.register()
            engine.deregister()

        benchmark(cycle)


@pytest.mark.benchmark(group="capture-engine")
class TestCaptureEngineForward:
    """Measure forward pass overhead with capture hooks active."""

    @pytest.mark.parametrize("num_layers", [2, 4, 6])
    def test_forward_detach_mode(self, benchmark, device, num_layers):
        """Standard capture mode: detach outputs (default, cheapest)."""
        model = SimpleModel(hidden_dim=128, num_layers=num_layers).to(device)
        modules = list(model.layers)
        engine = ModuleCaptureEngine(
            model=model,
            modules_to_capture=modules,
            detach_outputs=True,
        )
        engine.register()
        batch = _make_batch(device)

        def forward_and_clear():
            with torch.no_grad():
                model(**batch)
            engine.clear_captured()

        benchmark(forward_and_clear)
        engine.deregister()

    @pytest.mark.parametrize("num_layers", [2, 4, 6])
    def test_forward_deepcopy_mode(self, benchmark, device, num_layers):
        """Deepcopy capture mode: deepcopy inputs (expensive, but safe)."""
        model = SimpleModel(hidden_dim=128, num_layers=num_layers).to(device)
        modules = list(model.layers)
        engine = ModuleCaptureEngine(
            model=model,
            modules_to_capture=modules,
            deepcopy_captured_args_and_kwargs=True,
        )
        engine.register()
        batch = _make_batch(device)

        def forward_and_clear():
            with torch.no_grad():
                model(**batch)
            engine.clear_captured()

        benchmark(forward_and_clear)
        engine.deregister()

    def test_forward_with_output_callback(self, benchmark, device):
        """Capture with an output callback (used by HolisticDistiller's student capture)."""
        model = SimpleModel(hidden_dim=128, num_layers=3).to(device)
        modules = list(model.layers)
        callback_count = [0]

        def callback(module_id, inp, out):
            callback_count[0] += 1

        engine = ModuleCaptureEngine(
            model=model,
            modules_to_capture=modules,
            output_callback=callback,
        )
        engine.register()
        batch = _make_batch(device)

        def forward_and_clear():
            callback_count[0] = 0
            with torch.no_grad():
                model(**batch)
            engine.clear_captured()

        benchmark(forward_and_clear)
        engine.deregister()

    def test_forward_with_auto_truncate(self, benchmark, device):
        """Capture with early truncation (auto_truncate)."""
        model = SimpleModel(hidden_dim=128, num_layers=6).to(device)
        modules = list(model.layers)
        # Only capture layers 0-2 (skip layers 3-5 via auto_truncate)
        engine = ModuleCaptureEngine(
            model=model,
            modules_to_capture=modules[:3],
            auto_truncate=True,
        )
        engine.register()
        batch = _make_batch(device)

        def forward_and_clear():
            # The engine aborts the forward pass by raising
            # ``_TruncatedForwardException`` once every captured module has
            # run; the distillers catch it around the model call, so do the
            # same here.
            with torch.no_grad(), contextlib.suppress(_TruncatedForwardException):
                model(**batch)
            engine.clear_captured()

        benchmark(forward_and_clear)
        engine.deregister()
